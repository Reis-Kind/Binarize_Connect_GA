import os
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

    model.eval()
    with torch.no_grad():
        # 一次元のGA遺伝子をどこまで使ったか
        offset = 0
        # 畳み込み層を一層ずつ処理する
        for layer in model.layers:
            # 現在の層に重みが何個あるか
            size = layer.weight.numel()
            # 現在の層の重みだけを切り出す
            layer.weight.copy_(w[offset:offset + size].view_as(layer.weight).to(device))
            offset += size

        if offset != w.numel():
            raise ValueError("モデルと遺伝子の重み数が一致しません。")
        # 正解した画像数
        correct = 0
        total_loss = 0.0
        # 1500枚を一気に処理ぜず
        batch_size = 128
        # バッチサイズづつ処理する
        for start in range(0, len(y), batch_size):
            batch_x = x[start:start + batch_size].to(device)
            batch_y = y[start:start + batch_size].to(device)
            outputs = model(batch_x)
            total_loss += nn.functional.cross_entropy(outputs, batch_y, reduction='sum').item()
            correct += (outputs.argmax(dim=1) == batch_y).sum().item()

    acc = correct / len(y)
    loss = total_loss / len(y)

    return acc, loss


def tournament_select(scores, k):
    """
    
    """
    candidates = np.random.choice(len(scores), k, replace=False)
    best_idx = candidates[0]

    for i in candidates:
        if scores[i] > scores[best_idx]:
            best_idx = i

    return best_idx


def genetic_algorithm(model, dataset, train_indices,  device, eval_size=1500):
    """
    
    """
    islands = 5
    model_per_island = 32
    generations = 100
    migration_interval = 10
    mutation_rate = 0.0001
    ramdom_seed = 42

    torch.manual_seed(ramdom_seed)
    np.random.seed (ramdom_seed)
    # 画像抽出専用の乱数生成器
    data_generator = torch.Generator().manual_seed(ramdom_seed)

    # 全対象層の重みを1次元にまとめる
    origin_w = torch.cat([layer.weight.detach().cpu().reshape(-1) for layer in model.layers])
    # 二値化
    origin_w = torch.where(origin_w >= 0, 1.0, -1.0)
    # 探索する重み総数を取得
    n_weight = origin_w.numel()

    # 初期個体を作成
    island_w = origin_w.repeat(islands, model_per_island, 1)
    for i in range(islands):
        # 元のパラメータも残すため，0番目の個体は変えない
        for j in range(1, model_per_island):
            mask = torch.rand(n_weight) < mutation_rate
            island_w[i, j, mask] *= -1.0

    # 最良個体を元のパラメータで初期化
    best_w = origin_w.clone()
    history = []

    # 全個体を評価
    for gen in range(generations + 1):

        # 45,000枚の中での位置を、重複なしで1,500個選ぶ
        order = torch.randperm(
            len(train_indices),
            generator=data_generator
        )[:eval_size].tolist()

        # CIFAR-10データセット内の画像番号に変換
        indices = [train_indices[i] for i in order]

        # 今回の世代で使う画像とラベル
        x_eval = torch.stack([dataset[i][0] for i in indices])
        y_eval = torch.tensor([dataset[i][1] for i in indices])

        # 前の世代から持ち越した候補を、
        # 今回の画像で評価し直す
        best_acc, best_loss = evaluate(model, best_w, x_eval, y_eval, device)
        best_score = best_acc - 0.01 * best_loss

        island_score = []
        island_best_w = []
        island_best_scores = []
        island_worst = []

        for i in range(islands):
            scores = []

            for j in range(model_per_island):
                acc, loss = evaluate(model, island_w[i, j], x_eval, y_eval, device)
                score = acc - 0.01 * loss
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_acc = acc
                    best_loss = loss
                    best_w = island_w[i, j].clone()

            best_j = np.argmax(scores)
            worst_j = np.argmin(scores)

            island_score.append(scores)
            island_best_w.append(island_w[i, best_j].clone())
            island_best_scores.append(scores[best_j])
            island_worst.append(worst_j)


        # 今回の1,500枚で、元のBPモデルも評価する
        bp_acc, bp_loss = evaluate(
            model, origin_w, x_eval, y_eval, device
        )
        bp_score = bp_acc - 0.01 * bp_loss

        print(
            f"  元BP: {bp_acc * 100:.2f}% | "
            f"選択候補: {best_acc * 100:.2f}% | "
            f"正答率差: {(best_acc - bp_acc) * 100:+.2f}ポイント | "
            f"Score差: {best_score - bp_score:+.5f}"
        )

        changed = (best_w != origin_w).sum().item()

        history.append({
            'generation': gen,
            'accuracy': best_acc,
            'loss': best_loss,
            'score': best_score,
            'changed_weights': changed
        })

        print(
            f"世代 [{gen}/{generations}] | "
            f"Accuracy: {best_acc * 100:.2f}% | "
            f"Loss: {best_loss:.4f} | "
            f"Score: {best_score:.5f} | "
            f"符号変化: {changed}個",
            flush=True
        )

        # 
        if gen == generations:
            break

        # 移住処理
        # migration_intervalは1以上に設定
        if gen > 0 and gen % migration_interval == 0:
            for now_island in range(islands):
                next_island = (now_island + 1) % islands
                target = island_worst[next_island]
                score = island_best_scores[now_island]

                if score > island_score[next_island][target]:
                    island_w[next_island][target] = island_best_w[now_island].clone()
                    island_score[next_island][target] = score

        # エリート保存，交叉，突然変異
        for i in range(islands):
            scores = island_score[i]
            best_j = np.argmax(scores)
            # 親集団をコピーしてから子を作る(変化した親の子を作らないために)
            parent_population = island_w[i].clone()

            for j in range(model_per_island):
                # その島の一番はそのままに
                if j != best_j:
                    # 親をランダムにトーナメントして決める
                    parent_idx = tournament_select(scores, 3)
                    # 交叉
                    cross_mask = torch.rand(n_weight) < 0.5
                    child = torch.where(cross_mask, parent_population[parent_idx], parent_population[j])

                    # 突然変異の対象をランダムに選ぶ
                    mutation_mask = (torch.rand(n_weight) < mutation_rate)
                    child[mutation_mask] *= -1.0
                    island_w[i, j] = child

    # 最良個体を反映し、二値化層も同期
    evaluate(model, best_w, x_eval, y_eval, device)

    return best_w, history
    

def main():

    torch.manual_seed(42)
    eval_size = 1500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # BPの学習済みモデルを読み込む
    checkpoint = torch.load(
        './output/binaryconnect_cifar_aug_3conv_bp_seed42.pt',
        map_location='cpu', weights_only=True
    )
    model = BinaryConnectCifar10().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    ])
    dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=transform
    )

    # 公式テスト画像は候補選択に使わない
    x_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    # GA前を評価
    origin_w = torch.cat([
        layer.weight.detach().cpu().reshape(-1) for layer in model.layers
    ])
    origin_w = torch.where(origin_w >= 0, 1.0, -1.0)
    before_acc, _ = evaluate(model, origin_w, x_test, y_test, device)

    # GA実行とGA後の評価
    best_w, history = genetic_algorithm(model, dataset, checkpoint['train_indices'], device, eval_size=eval_size)
    after_acc, _ = evaluate(model, best_w, x_test, y_test, device)

    print(f"Test Accuracy: {before_acc * 100:.2f}% → {after_acc * 100:.2f}%")
    print(f"変化: {(after_acc - before_acc) * 100:+.2f}ポイント")

    generations = [h['generation'] for h in history]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(
        generations,
        [h['accuracy'] * 100 for h in history]
    )
    plt.xlabel('Generation')
    plt.ylabel('Accuracy (%)')
    plt.title('Sampled Training Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        generations,
        [h['loss'] for h in history]
    )
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.title('Sampled Training Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./output/ga_cifar_3conv_resample_100_result.png')
    plt.close()


if __name__ == '__main__':
    main()

