import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from binaryconnect_cifar10 import BinaryConnectCifar10

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

# channels_last: 今のモデルは forward で .view() を使っているのでエラーになる(実測済み)。
# 帯域律速で効果も薄い見込みなので False にしておく。
USE_CHANNELS_LAST = False

# 1個体の評価を何枚ずつに分割して流すか(chunk)。
# RTX 4080での実測: 500 が最速(一括の1.45倍)。400や800は遅いので、近い値に変えないこと。
# 20000/10000/5000 はどれも500で割り切れる。eval_size を変えても500のままでよい。
# None なら eval_size 枚を一括で流す(従来と同じ)。
EVAL_BATCH = 500

# 試し実行のときだけ変える(本番のCSV/PNGを上書きしないため)
GENERATIONS = 5          # 試すときは 5 など
OUTPUT_TAG = ''            # 試すときは '_fasttest' など
OUT_BASE = './output/ga_cifar_noaug_3conv_nobias_128_BPseed44_best_val_300ep_2000batch_0.0001_5000'


# ---------------------------------------------------------------
# データ: 最初に一度だけGPUへ載せる(毎世代 dataset[i] を回さない)
# ---------------------------------------------------------------
def prepare_x(x):
    """fp16化(autocastが内部でやる変換を先に済ませる)+必要ならchannels_last"""
    x = x.half()
    if USE_CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)
    else:
        # permuteで作ったテンソルはchannels_last配置になっているので、NCHWに揃え直す
        x = x.contiguous()
    return x


def load_cifar_tensors(dataset, device):
    """
    ToTensor + Normalize と同じ演算(uint8/255 -> (x-mean)/std)をGPU上で一括実行する。
    """
    x = torch.from_numpy(dataset.data).to(device).permute(0, 3, 1, 2).float().div(255.0)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    y = torch.tensor(dataset.targets, device=device)
    return prepare_x(x), y


def take(x, idx):
    """画像番号idxの画像を取り出す(メモリ形式を揃える)"""
    out = x[idx]
    if USE_CHANNELS_LAST:
        out = out.contiguous(memory_format=torch.channels_last)
    else:
        out = out.contiguous()
    return out


# ---------------------------------------------------------------
# 評価
# ---------------------------------------------------------------
def set_weights(model, w):
    """一次元のGA遺伝子をmodel.layersへ層ごとに書き込む"""
    offset = 0
    for layer in model.layers:
        size = layer.weight.numel()
        layer.weight.data.copy_(w[offset:offset + size].view_as(layer.weight))
        offset += size


@torch.no_grad()
def evaluate_gpu(model, w, x, y):
    """
    正答率(0~1)とLossをGPUテンソルのまま返す。
    .item() を呼ばないのでGPUとCPUの同期が起きない。
    """
    set_weights(model, w)
    n = x.shape[0]

    # 一括評価(既定)
    if EVAL_BATCH is None or EVAL_BATCH >= n:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(x)
        outputs = outputs.float()
        loss = F.cross_entropy(outputs, y)
        acc = (outputs.argmax(dim=1) == y).float().mean()
        return acc, loss

    # 分割評価: 合計をGPU上に溜めるだけで .item() は呼ばない(同期が増えない)
    loss_sum = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    for s in range(0, n, EVAL_BATCH):
        xb, yb = x[s:s + EVAL_BATCH], y[s:s + EVAL_BATCH]
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(xb)
        outputs = outputs.float()
        loss_sum += F.cross_entropy(outputs, yb, reduction='sum')
        correct += (outputs.argmax(dim=1) == yb).sum()
    return correct / n, loss_sum / n


def evaluate_many(model, ws, x, y):
    """複数個体を評価し、(accのlist, lossのlist)を返す。同期は最後の1回だけ。"""
    accs = torch.empty(len(ws), device=x.device)
    losses = torch.empty(len(ws), device=x.device)
    for i, w in enumerate(ws):
        accs[i], losses[i] = evaluate_gpu(model, w, x, y)
    return accs.tolist(), losses.tolist()


def evaluate(model, w, x, y, device):
    """
    1個体用(従来と同じ使い方)。正答率は0~1で返す。
    """
    model.eval()
    acc, loss = evaluate_gpu(model, w.to(device), x, y)
    return acc.item(), loss.item()


# ---------------------------------------------------------------
# GA
# ---------------------------------------------------------------
def make_initial_population(origin_w, population, random, origin, mutation_rate):
    """
    初期個体を生成する関数
    """
    device = origin_w.device

    mutated = population - random - origin

    # 元の個体の読み込みと元個体をめっちゃコピー
    n_weight = origin_w.numel()
    population_w = origin_w.repeat(population, 1)

    # 元個体をもとに突然変異させた個体をmutated個(元個体も一体残す)
    for i in range(origin, mutated + origin):
        # 突然変異個体の中で何番目か
        j = i - origin

        if j < int(mutated * 0.7):
            individual_rate = 0.0001
        elif j < int(mutated * 0.9):
            individual_rate = 0.001
        else:
            individual_rate = 0.01

        mutation_mask = (torch.rand(n_weight, device=device) < individual_rate)
        population_w[i, mutation_mask] *= -1.0

    # random体は、事前BPとは無関係な完全ランダム二値重みとする。
    population_w[mutated + origin:] = torch.where(torch.rand(random, n_weight, device=device) < 0.5, -1.0, 1.0,)

    return population_w


def tournament_select(scores, losses, k):
    """
    トーナメント選択: k個体を無作為に選び、正答率が最大(同率ならLoss最小)の個体番号を返す。
    """
    candidates = np.random.choice(len(scores), k, replace=False)
    best_idx = candidates[0]

    for i in candidates:
        if scores[i] > scores[best_idx] or (scores[i] == scores[best_idx] and losses[i] < losses[best_idx]):
            best_idx = i

    return best_idx


def genetic_algorithm(model, x_all, y_all, train_indices, final_indices, device, eval_size, x_test, y_test):
    """
    x_all, y_all: 学習データ全体(GPU上, fp16)。train_indices/final_indicesはその中の画像番号。
    """
    population = 500
    random = 50
    origin = 1
    generations = GENERATIONS
    elite_size = 50
    mutation_rate = 0.0001
    random_seed = 42

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # 画像抽出専用の乱数生成器(CPU側。従来と同じ画像が選ばれる)
    data_generator = torch.Generator().manual_seed(random_seed)
    train_idx_t = torch.tensor(train_indices, device=device)

    model.eval()

    # 全対象層の重みを1次元にまとめる
    origin_w = torch.cat([layer.weight.detach().reshape(-1) for layer in model.layers])
    # 二値化
    origin_w = torch.where(origin_w >= 0, 1.0, -1.0)
    # 探索する重み総数を取得
    n_weight = origin_w.numel()

    # 初期個体を作成
    population_w = make_initial_population(origin_w, population, random, origin, mutation_rate)

    # 最良個体を元のパラメータで初期化
    best_w = origin_w.clone()
    history = []
    best_candidates = []

    # GA終了後の個体選択に使う固定5,000枚
    final_idx_t = torch.tensor(final_indices, device=device)
    x_final = take(x_all, final_idx_t)
    y_final = y_all[final_idx_t]

    # 全個体を評価
    for gen in range(generations + 1):
        t_start = time.time()

        # 前の世代から持ち越した候補を、今回の画像で評価し直す
        order = torch.randperm(len(train_indices), generator=data_generator)[:eval_size].to(device)
        sel = train_idx_t[order]

        # 今回の世代で使う画像とラベル(GPU上のテンソルから抜くだけ)
        x_eval = take(x_all, sel)
        y_eval = y_all[sel]

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_data = time.time() - t_start

        scores, losses = evaluate_many(model, population_w, x_eval, y_eval)
        t_eval = time.time() - t_start - t_data

        # 正答率の降順、同率ならLossの昇順に並べる
        ranked_indices = sorted(
            range(population),
            key=lambda i: (scores[i], -losses[i]), reverse=True
        )

        # この世代で最も良い個体
        best_idx = ranked_indices[0]
        best_w = population_w[best_idx].clone()
        best_acc = scores[best_idx]
        best_loss = losses[best_idx]
        # 各世代の最大スコアを保存
        best_candidates.append(best_w.clone())

        bp_acc, bp_loss = evaluate(model, origin_w, x_eval, y_eval, device)

        test_acc, test_loss = evaluate(model, best_w, x_test, y_test, device)

        # 変更された重みの数
        changed = (best_w != origin_w).sum().item()

        print(
            f"  元BP: {bp_acc * 100:.2f}% | "
            f"世代最良: {best_acc * 100:.2f}% | "
            f"正答率差: {(best_acc - bp_acc) * 100:+.2f}ポイント"
        )

        history.append({
            'generation': gen,
            'accuracy': best_acc,
            'loss': best_loss,
            'changed_weights': changed,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        })

        # 完了した世代までを毎回保存する
        save_csv(history)

        print(
            f"世代 [{gen}/{generations}] | "
            f"Train Acc: {best_acc * 100:.2f}% | "
            f"Train Loss: {best_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f}",
            f"符号変化: {changed}個",
            flush=True
        )

        # 速度確認用(10世代ごと)
        if gen % 10 == 0:
            print(f"  [time] 画像準備 {t_data:.2f}s | 個体評価 {t_eval:.2f}s | 世代全体 {time.time() - t_start:.2f}s", flush=True)

        if gen == generations:
            break

        # 現在の集団を保存してから次世代を作る
        parent_population = population_w
        next_population = torch.empty_like(population_w)

        # 上位elite_size個体をそのまま次世代へ保存
        elite_indices = ranked_indices[:elite_size]

        for next_i, elite_idx in enumerate(elite_indices):
            next_population[next_i] = parent_population[elite_idx]

        # エリート以外の個体を交叉と突然変異で生成
        for i in range(elite_size, population):
            parent1_idx = tournament_select(scores, losses, 3)
            parent2_idx = tournament_select(scores, losses, 3)

            # 同じ個体同士の交叉を避ける
            while parent2_idx == parent1_idx:
                parent2_idx = tournament_select(scores, losses, 3)

            # 2個体による一様交叉
            cross_mask = torch.rand(n_weight, device=device) < 0.5
            child = torch.where(cross_mask, parent_population[parent1_idx], parent_population[parent2_idx])

            # 二値重みの符号を突然変異させる
            # 子個体(population - elite_size体)を，突然変異率の異なる3群に分ける
            n_children = population - elite_size
            child_index = i - elite_size

            if child_index < int(n_children * 0.7):
                child_mutation_rate = 0.0001
            elif child_index < int(n_children * 0.9):
                child_mutation_rate = 0.001
            else:
                child_mutation_rate = 0.01

            # この子個体に適用する突然変異
            mutation_mask = (torch.rand(n_weight, device=device) < child_mutation_rate)
            child[mutation_mask] *= -1.0
            next_population[i] = child

        # 新しい集団に更新
        population_w = next_population

    # 元BPを最初の候補にする
    final_best_w = origin_w.clone()
    final_best_acc, final_best_loss = evaluate(model, final_best_w, x_final, y_final, device)

    # 最終世代の個体と，今までの世代の最高スコアの個体を固定5,000枚で評価
    candidates = list(population_w) + best_candidates[:-1]
    cand_accs, cand_losses = evaluate_many(model, candidates, x_final, y_final)

    for candidate_w, acc, loss in zip(candidates, cand_accs, cand_losses):
        # 正答率が高い個体を選ぶ
        # 同率の場合はLossが小さい個体を選ぶ
        if (acc > final_best_acc or (acc == final_best_acc and loss < final_best_loss)):
            final_best_acc = acc
            final_best_loss = loss
            final_best_w = candidate_w.clone()

    changed = (final_best_w != origin_w).sum().item()

    print("\n固定5000枚による最終選択")
    print(
        f"Accuracy: {final_best_acc * 100:.2f}% | "
        f"Loss: {final_best_loss:.4f} | "
        f"符号変化: {changed}個"
    )

    # 選択した個体をモデルに反映
    evaluate(model, final_best_w, x_final, y_final, device)

    return final_best_w, history


def save_csv(history, after_acc=None, after_loss=None):
    """
    GA前後の公式テストAccuracyとLossをCSVに保存する。
    """

    csv_path = OUT_BASE + OUTPUT_TAG + '.csv'

    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow([
            'stage',
            'generation',
            'sample_accuracy_percent',
            'sample_loss',
            'changed_weights',
            'test_accuracy_percent',
            'test_loss'
        ])

        for h in history:
            writer.writerow([
                'generation',
                h['generation'],
                h['accuracy'] * 100,
                h['loss'],
                h['changed_weights'],
                h['test_accuracy'] * 100,
                h['test_loss'],
            ])

        if after_acc is not None and after_loss is not None:
            writer.writerow([
                'after_ga', '', '', '', '',
                after_acc * 100, after_loss
            ])


def main():

    eval_size = 20000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # cudnn.benchmark は実測で速度差がなく、実行ごとに結果が揺れる原因になるので使わない
    torch.backends.cudnn.benchmark = False

    # BPの学習済みモデルを読み込む
    load_path = './output/binaryconnect_cifar_noaug_3conv_nobias_128_seed44_300ep_best_val.pt'
    checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
    model = BinaryConnectCifar10().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    train_dataset = datasets.CIFAR10('./data', train=True, download=True)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True)

    # 画像は全てGPUに載せておく(fp16, 学習50,000枚で約0.3GB)
    x_train_all, y_train_all = load_cifar_tensors(train_dataset, device)

    # GA進化用45,000枚の画像番号
    ga_train_indices = list(range(45000))
    # 最終選択用5,000枚の画像番号
    ga_final_indices = list(range(45000, 50000))

    # 公式テスト画像は候補選択に使わない
    x_test, y_test = load_cifar_tensors(test_dataset, device)

    # GA前を評価
    origin_w = torch.cat([layer.weight.detach().cpu().reshape(-1) for layer in model.layers])
    origin_w = torch.where(origin_w >= 0, 1.0, -1.0)
    before_acc, before_loss = evaluate(model, origin_w, x_test, y_test, device)
    print(f"GA前 Test Accuracy: {before_acc * 100:.2f}%")

    # GA実行とGA後の評価
    best_w, history = genetic_algorithm(model, x_train_all, y_train_all, ga_train_indices, ga_final_indices, device, eval_size, x_test, y_test)
    after_acc, after_loss = evaluate(model, best_w, x_test, y_test, device)

    print(f"Test Accuracy: {before_acc * 100:.2f}% → {after_acc * 100:.2f}%")
    print(f"変化: {(after_acc - before_acc) * 100:+.2f}ポイント")

    generations = [h['generation'] for h in history]

    plt.figure(figsize=(12, 5))

    # 正答率
    plt.subplot(1, 2, 1)

    plt.plot(
        generations,
        [h['accuracy'] * 100 for h in history],
        label='Sampled Train Accuracy',
        color='tab:green'
    )
    plt.plot(
        generations,
        [h['test_accuracy'] * 100 for h in history],
        label='Test Accuracy',
        color='tab:blue'
    )
    plt.axhline(
        y=before_acc * 100,
        label='BP Test Accuracy',
        color='tab:gray',
        linestyle='--'
    )

    plt.xlabel('Generation')
    plt.ylabel('Accuracy (%)')
    plt.title('BP + GA Accuracy')
    plt.grid(True)
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)

    plt.plot(
        generations,
        [h['loss'] for h in history],
        label='Sampled Train Loss',
        color='tab:red'
    )
    plt.plot(
        generations,
        [h['test_loss'] for h in history],
        label='Test Loss',
        color='tab:orange'
    )
    plt.axhline(
        y=before_loss,
        label='BP Test Loss',
        color='tab:gray',
        linestyle='--'
    )

    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.title('BP + GA Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    os.makedirs('./output', exist_ok=True)

    plt.savefig(
        OUT_BASE + OUTPUT_TAG + '.png',
        dpi=300
    )
    plt.close()

    save_csv(history, after_acc, after_loss)

if __name__ == '__main__':
    main()