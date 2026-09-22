import os
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from binaryconnect_cifar10 import BinaryConnectCifar10

def evaluate(model, w, x, y, device):
    """
    個体の重みをmodel.layersへ設定し、正答率とLossを計算。
    正答率は0～1で返す。
    """
    w = w.to(device)
    model.eval()
    with torch.no_grad():
        # 一次元のGA遺伝子をどこまで使ったか
        offset = 0
        # 畳み込み層を一層ずつ処理する
        for layer in model.layers:
            # 現在の層に重みが何個あるか
            size = layer.weight.numel()
            # 現在の層の重みだけを切り出す
            layer.weight.copy_(w[offset:offset + size].view_as(layer.weight))
            offset += size

        x, y = x.to(device), y.to(device)
        outputs = model(x)
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
        mutation_mask = (torch.rand(n_weight, device=device) < mutation_rate)
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
    generations = 1000
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

    # 初期個体を作成
    population_w = make_initial_population(origin_w, population, random, origin, mutation_rate)

    # 最良個体を元のパラメータで初期化
    best_w = origin_w.clone()
    history = []
    best_candidates = []

    # GA終了後の個体選択に使う固定5,000枚
    x_final = torch.stack([dataset[i][0] for i in final_indices]).to(device)
    y_final = torch.tensor([dataset[i][1] for i in final_indices]).to(device)

    # 全個体を評価
    for gen in range(generations + 1):

        order = torch.randperm(
            len(train_indices),
            generator=data_generator
        )[:eval_size].tolist()

        # CIFAR-10データセット内の画像番号に変換
        indices = [train_indices[i] for i in order]

        # 今回の世代で使う画像とラベル
        x_eval = torch.stack([dataset[i][0] for i in indices]).to(device)
        y_eval = torch.tensor([dataset[i][1] for i in indices], device=device)

        # 前の世代から持ち越した候補を、
        # 今回の画像で評価し直す

        scores = []
        losses = []

        for i in range(population):
            acc, loss = evaluate(model, population_w[i], x_eval, y_eval, device)
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

        bp_acc, bp_loss = evaluate(model, origin_w, x_eval, y_eval, device)

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

        # 現在の集団を保存してから次世代を作る
        parent_population = population_w
        next_population = torch.empty_like(population_w)

        # 上位20個体をそのまま次世代へ保存
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
            mutation_mask = torch.rand(n_weight, device=device) < mutation_rate
            child[mutation_mask] *= -1.0

            next_population[i] = child

        # 新しい集団に更新
        population_w = next_population

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

  

    return final_best_w, history
    
    
def save_csv(history, after_acc=None, after_loss=None):
    """
    GA前後の公式テストAccuracyとLossをCSVに保存する。
    """

    csv_path = ('./output/ga_cifar_noaug_3conv_nobias_128_BPseed44_best_val_300ep_2000batch_0.0001_1000.csv')

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

    eval_size = 2000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

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
    x_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).to(device)

    # GA前を評価
    origin_w = torch.cat([layer.weight.detach().cpu().reshape(-1) for layer in model.layers])
    origin_w = torch.where(origin_w >= 0, 1.0, -1.0)
    before_acc, before_loss = evaluate(model, origin_w, x_test, y_test, device)

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
        './output/ga_cifar_noaug_3conv_nobias_128_BPseed44_best_val_300ep_2000batch_0.0001_1000.png',
        dpi=300
    )
    plt.close()

    save_csv(history, after_acc, after_loss)

if __name__ == '__main__':
    main()
