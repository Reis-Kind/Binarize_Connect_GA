import os
import csv
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from binaryconnect_cifar10 import BinaryConnectCifar10

# =====================================================================
# 計測用の設定(これ以外のGAのロジックは元のコードと同じ)
# =====================================================================
PROFILE_GENERATIONS = 5      # 計測で回す世代数(3以上)。本番は500
FULL_GENERATIONS = 500       # 本番の世代数。所要時間の推定にだけ使う
OUTPUT_TAG = '_profile'      # 出力ファイル名の末尾に付ける(本番のCSV/PNGを上書きしないため)

CUDA = torch.cuda.is_available()
PROF = defaultdict(list)     # 世代ごとの記録(世代0が終わったらクリアする)
ONCE = defaultdict(list)     # 1回しか実行しない処理の記録

# 計測項目の名前
K_A1 = 'A1 画像変換(CPU)'
K_A2 = 'A2 画像転送(CPU→GPU)'
K_A3 = 'A3 ラベル作成'
K_C = 'C 個体評価(全個体)'
K_D = 'D 元BPの評価'
K_E = 'E Testの評価'
K_F = 'F CSV保存'
K_G = 'G 次世代生成'
K_G1 = 'G1 親選択(CPU)'
K_GEN = 'gen全体'
K_S1 = 'x_test作成'
K_S2 = 'x_final作成'
K_S3 = '初期個体生成'
K_H = '最終選択'


@contextmanager
def section(name, store=None, sync=True):
    """with文の中にかかった時間を記録する。GPUは非同期なので前後で同期してから測る。"""
    store = PROF if store is None else store
    if CUDA and sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if CUDA and sync:
        torch.cuda.synchronize()
    store[name].append(time.perf_counter() - t0)


def print_profile(population, n_candidates, eval_size):
    """計測結果を表にして表示する。"""
    def mean(k):
        v = PROF.get(k, [])
        return sum(v) / len(v) if v else 0.0

    def total(k):
        return sum(PROF.get(k, []))

    n_gen = len(PROF.get(K_C, [])) or 1     # 計測できた世代数
    n_g = len(PROF.get(K_G, [])) or 1
    gen_t = mean(K_GEN)

    def line(label, sec, indent=0):
        pct = sec / gen_t * 100 if gen_t > 0 else 0.0
        print(f"{' ' * indent}{label:<30s}{sec:8.3f} s  {pct:5.1f}%")

    print("\n" + "=" * 66)
    print("時間の内訳(1世代あたりの平均。世代0はウォームアップなので除外)")
    print("=" * 66)
    print(f"{'1世代の合計':<30s}{gen_t:8.3f} s")

    measured = 0.0
    for k in [K_A1, K_A2, K_A3, K_C, K_D, K_E, K_F, K_G]:
        s = mean(k)
        measured += s
        line(k, s)
        if k == K_C:
            for sub, label in [('C:重み書込', '重み書込'), ('C:forward', '順伝播(forward)'), ('C:loss/acc(.item)', 'loss/acc計算+.item()')]:
                per_gen = total(sub) / n_gen
                print(f"    - {label:<24s}{per_gen:8.3f} s  ({mean(sub) * 1000:7.2f} ms/個体)")
        if k == K_G:
            per_gen = total(K_G1) / n_g
            print(f"    - {'親選択(CPU, tournament)':<24s}{per_gen:8.3f} s")
    line('その他(未計測+平均の誤差)', gen_t - measured)

    fwd = mean('C:forward')
    if fwd > 0:
        print(f"\n順伝播の速さ: {eval_size / fwd:,.0f} 枚/秒 (1個体 {fwd * 1000:.1f} ms / {eval_size}枚)")
    if CUDA:
        print(f"VRAMのピーク使用量: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    # 本番(FULL_GENERATIONS世代)の所要時間の推定
    startup = sum(sum(ONCE.get(k, [])) for k in (K_S1, K_S2, K_S3))
    final_t = sum(ONCE.get(K_H, []))
    est_main = gen_t * (FULL_GENERATIONS + 1)
    est_final = final_t / max(n_candidates, 1) * (population + FULL_GENERATIONS)
    est_total = est_main + est_final + startup
    print(f"\n本番{FULL_GENERATIONS}世代の推定: GA本体 {est_main / 3600:.2f} 時間"
          f" + 最終選択 {est_final / 60:.1f} 分 + 準備 {startup:.0f} 秒"
          f" = 約 {est_total / 3600:.2f} 時間")
    print("=" * 66 + "\n", flush=True)


def evaluate(model, w, x, y, device, tag=None):
    """
    個体の重みをmodel.layersへ設定し、正答率とLossを計算。
    正答率は0～1で返す。
    tag を渡したときだけ、内部(重み書込/forward/loss計算)の時間を記録する。
    """
    def sec(name):
        return section(f'{tag}:{name}') if tag else nullcontext()

    w = w.to(device)
    model.eval()
    with torch.no_grad():
        with sec('重み書込'):
            # 一次元のGA遺伝子をどこまで使ったか
            offset = 0
            # 畳み込み層を一層ずつ処理する
            for layer in model.layers:
                # 現在の層に重みが何個あるか
                size = layer.weight.numel()
                # 現在の層の重みだけを切り出す
                new_weight = w[offset:offset + size].view_as(layer.weight).contiguous()
                layer.weight.data.copy_(new_weight)
                offset += size

        x, y = x.to(device), y.to(device)

        with sec('forward'):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(x)

        with sec('loss/acc(.item)'):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = nn.functional.cross_entropy(outputs, y).item()
                acc = (outputs.argmax(dim=1) == y).float().mean().item()

    return acc, loss

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

    """
    candidates = np.random.choice(len(scores), k, replace=False)
    best_idx = candidates[0]

    for i in candidates:
        if scores[i] > scores[best_idx] or (scores[i] == scores[best_idx] and losses[i] < losses[best_idx]):
            best_idx = i

    return best_idx


def genetic_algorithm(model, dataset, train_indices, final_indices, device, eval_size, x_test, y_test):
    """

    """
    population = 500
    random = 50
    origin = 1
    generations = PROFILE_GENERATIONS      # 元は 500
    elite_size = 50
    mutation_rate = 0.0001
    random_seed = 42

    torch.manual_seed(random_seed)
    np.random.seed (random_seed)
    # 画像抽出専用の乱数生成器
    data_generator = torch.Generator().manual_seed(random_seed)

    # 全対象層の重みを1次元にまとめる
    origin_w = torch.cat([layer.weight.detach().reshape(-1) for layer in model.layers])
    # 二値化
    origin_w = torch.where(origin_w >= 0, 1.0, -1.0)
    # 探索する重み総数を取得
    n_weight = origin_w.numel()

    print(f"[env] torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if CUDA else 'なし(CPU)'}"
          f" | cudnn.benchmark={torch.backends.cudnn.benchmark}")
    print(f"[env] 探索する重み数 {n_weight} | 個体数 {population} | eval_size {eval_size}"
          f" | 計測する世代数 {generations}", flush=True)

    # 初期個体を作成
    with section(K_S3, ONCE):
        population_w = make_initial_population(origin_w, population, random, origin, mutation_rate)

    # 最良個体を元のパラメータで初期化
    best_w = origin_w.clone()
    history = []
    best_candidates = []

    # GA終了後の個体選択に使う固定5,000枚
    with section(K_S2, ONCE):
        x_final = torch.stack([dataset[i][0] for i in final_indices]).to(device)
        y_final = torch.tensor([dataset[i][1] for i in final_indices]).to(device)

    # 全個体を評価
    for gen in range(generations + 1):
        if CUDA:
            torch.cuda.synchronize()
        t_gen0 = time.perf_counter()

        # 前の世代から持ち越した候補を、
        # 今回の画像で評価し直す
        order = torch.randperm(len(train_indices), generator=data_generator)[:eval_size].tolist()

        # CIFAR-10データセット内の画像番号に変換
        indices = [train_indices[i] for i in order]

        # 今回の世代で使う画像とラベル
        with section(K_A1):
            x_cpu = torch.stack([dataset[i][0] for i in indices])
        with section(K_A2):
            x_eval = x_cpu.to(device)
        with section(K_A3):
            # (元のコードのまま: dataset[i][1] は画像の変換をもう一度実行してからラベルだけ返す)
            y_eval = torch.tensor([dataset[i][1] for i in indices], device=device)

        scores = []
        losses = []

        with section(K_C):
            for i in range(population):
                acc, loss = evaluate(model, population_w[i], x_eval, y_eval, device, tag='C')
                scores.append(acc)
                losses.append(loss)

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

        with section(K_D):
            bp_acc, bp_loss = evaluate(model, origin_w, x_eval, y_eval, device)

        with section(K_E):
            test_acc, test_loss = evaluate(model, best_w, x_test, y_test, device)

        # val_acc, val_loss = evaluate(model, best_w, x_final, y_final, device)

        # 変更された個体数
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
        with section(K_F):
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

        #
        if gen == generations:
            break

        with section(K_G):
            # 現在の集団を保存してから次世代を作る
            parent_population = population_w
            next_population = torch.empty_like(population_w)

            # 上位20個体をそのまま次世代へ保存
            elite_indices = ranked_indices[:elite_size]

            for next_i, elite_idx in enumerate(elite_indices):
                next_population[next_i] = parent_population[elite_idx]

            # エリート以外の個体を交叉と突然変異で生成
            for i in range(elite_size, population):
                with section(K_G1, sync=False):
                    parent1_idx = tournament_select(scores, losses, 3)
                    parent2_idx = tournament_select(scores, losses, 3)

                    # 同じ個体同士の交叉を避ける
                    while parent2_idx == parent1_idx:
                        parent2_idx = tournament_select(scores, losses, 3)

                # 2個体による一様交叉
                cross_mask = torch.rand(n_weight, device=device) < 0.5
                child = torch.where(cross_mask, parent_population[parent1_idx], parent_population[parent2_idx])

                # 二値重みの符号を突然変異させる
                # 子個体450体を，突然変異率の異なる3群に分ける
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

        # この世代の合計時間を記録(世代0はウォームアップなので記録後に捨てる)
        if CUDA:
            torch.cuda.synchronize()
        PROF[K_GEN].append(time.perf_counter() - t_gen0)
        if gen == 0:
            PROF.clear()

    # 最終選択(固定5,000枚で候補を評価)
    with section(K_H, ONCE):
        # 元BPを最初の候補にする
        final_best_w = origin_w.clone()
        final_best_acc, final_best_loss = evaluate(model, final_best_w, x_final, y_final, device)

        # 最終世代の全個体を固定5,000枚で評価する

        # 最終世代の個体と，今までの世代の最高スコアの個体を評価
        candidates = list(population_w) + best_candidates[:-1]
        for candidate_w in candidates:
            acc, loss = evaluate(model, candidate_w, x_final, y_final, device)

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

    # 計測結果を表示
    print_profile(population, len(candidates), eval_size)

    return final_best_w, history


def save_csv(history, after_acc=None, after_loss=None):
    """
    GA前後の公式テストAccuracyとLossをCSVに保存する。
    """

    csv_path = ('./output/ga_cifar_noaug_3conv_nobias_128_BPseed44_best_val_300ep_2000batch_0.0001_5000' + OUTPUT_TAG + '.csv')

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
    if CUDA:
        torch.cuda.reset_peak_memory_stats()

    # BPの学習済みモデルを読み込む
    load_path = './output/binaryconnect_cifar_noaug_3conv_nobias_128_seed44_300ep_best_val.pt'
    checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
    model = BinaryConnectCifar10().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # GA進化用45,000枚の画像番号
    ga_train_indices = list(range(45000))
    # 最終選択用5,000枚の画像番号
    ga_final_indices = list(range(45000, 50000))

    # 公式テスト画像は候補選択に使わない
    with section(K_S1, ONCE):
        x_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
        y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).to(device)

    # GA前を評価
    origin_w = torch.cat([layer.weight.detach().cpu().reshape(-1) for layer in model.layers])
    origin_w = torch.where(origin_w >= 0, 1.0, -1.0)
    before_acc, before_loss = evaluate(model, origin_w, x_test, y_test, device)
    print(f"GA前 Test Accuracy: {before_acc * 100:.2f}%")

    # GA実行とGA後の評価
    best_w, history = genetic_algorithm(model, train_dataset, ga_train_indices, ga_final_indices, device, eval_size, x_test, y_test)
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
        './output/ga_cifar_noaug_3conv_nobias_128_BPseed44_best_val_300ep_2000batch_0.0001_5000' + OUTPUT_TAG + '.png',
        dpi=300
    )
    plt.close()

    save_csv(history, after_acc, after_loss)

if __name__ == '__main__':
    main()