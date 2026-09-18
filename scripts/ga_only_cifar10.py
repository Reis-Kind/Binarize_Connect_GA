import os
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class GACifar10(nn.Module):
    def __init__(self):
        super().__init__()

        # RGBで3チャネルあるから3（Mnistは白黒だから1）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1,  bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1,  bias=False)
        self.fc = nn.Linear(64 * 4 * 4, 10, bias=False)
        self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.fc])

        # 神の一手
        self.bn1 = nn.BatchNorm2d(16, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(32, track_running_stats=False)
        self.bn3 = nn.BatchNorm2d(64, track_running_stats=False)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.requires_grad_(False)


    def forward(self, x):

        """
        """

        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


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
            layer.weight.copy_(w[offset:offset + size].view_as(layer.weight))
            offset += size

        if offset != w.numel():
            raise ValueError("モデルと遺伝子の重み数が一致しません。")

        outputs = model(x)

        loss = nn.functional.cross_entropy(outputs, y).item()

        acc = (outputs.argmax(dim=1) == y).float().mean().item()
    
    return acc, loss

def make_initial_population(model, population, device):
    """
    初期個体を生成する関数

    """
    n_weight = 0
    for layer in model.layers:
        n_weight += layer.weight.numel()

    population_w = torch.where(torch.rand(population, n_weight, device=device) < 0.5, -1.0, 1.0,)

    return population_w


def tournament_select(scores, k):
    """
    
    """
    candidates = np.random.choice(len(scores), k, replace=False)
    best_idx = candidates[0]

    for i in candidates:
        if scores[i] > scores[best_idx]:
            best_idx = i

    return best_idx


def genetic_algorithm(model, dataset, train_indices, final_indices, device, eval_size):
    """
    
    """
    population = 500
    generations = 5000
    elite_size = 50
    mutation_rate = 0.0001
    random_seed = 42

    torch.manual_seed(random_seed)
    np.random.seed (random_seed)
    # 画像抽出専用の乱数生成器
    data_generator = torch.Generator().manual_seed(random_seed)

    # 初期個体を作成
    population_w = make_initial_population(model, population, device)

    n_weight = 0
    for layer in model.layers:
        n_weight += layer.weight.numel()

    # 最良個体を元のパラメータで初期化
    best_w = 0
    history = []

    # 全個体を評価
    for gen in range(generations + 1):

        # 45,000枚の中での位置を、重複なしで5000個選ぶ
        order = torch.randperm(
            len(train_indices),
            generator=data_generator
        )[:eval_size].tolist()

        # CIFAR-10データセット内の画像番号に変換
        indices = [train_indices[i] for i in order]

        # 今回の世代で使う画像とラベル
        x_eval = torch.stack([dataset[i][0] for i in indices]).to(device)
        y_eval = torch.tensor([dataset[i][1] for i in indices]).to(device)

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
            key=lambda i: (scores[i], -losses[i]),
            reverse=True
        )

        # この世代で最も良い個体
        best_idx = ranked_indices[0]
        best_w = population_w[best_idx].clone()
        best_acc = scores[best_idx]
        best_loss = losses[best_idx]

        history.append({
            'generation': gen,
            'accuracy': best_acc,
            'loss': best_loss,
        })

        print(
            f"世代 [{gen}/{generations}] | "
            f"Accuracy: {best_acc * 100:.2f}% | "
            f"Loss: {best_loss:.4f} | ",
            flush=True
        )

        # 
        if gen == generations:
            break

        # 現在の集団を保存してから次世代を作る
        parent_population = population_w.clone()
        next_population = torch.empty_like(population_w)

        # 上位20個体をそのまま次世代へ保存
        elite_indices = ranked_indices[:elite_size]

        for next_i, elite_idx in enumerate(elite_indices):
            next_population[next_i] = parent_population[elite_idx].clone()

        # 残りの180個体を交叉と突然変異で生成
        for i in range(elite_size, population):
            parent1_idx = tournament_select(scores, 3)
            parent2_idx = tournament_select(scores, 3)

            # 同じ個体同士の交叉を避ける
            while parent2_idx == parent1_idx:
                parent2_idx = tournament_select(scores, 3)

            # 2個体による一様交叉
            cross_mask = torch.rand(n_weight, device=device) < 0.5
            child = torch.where(cross_mask, parent_population[parent1_idx], parent_population[parent2_idx])

            # 二値重みの符号を突然変異させる
            mutation_mask = torch.rand(n_weight, device=device) < mutation_rate
            child[mutation_mask] *= -1.0

            next_population[i] = child

        # 新しい集団に更新
        population_w = next_population


    # 最良個体を反映し、二値化層も同期
    # GA終了後の個体選択に使う固定5,000枚
    x_final = torch.stack([dataset[i][0] for i in final_indices]).to(device)
    y_final = torch.tensor([dataset[i][1] for i in final_indices]).to(device)

    # 元BPを最初の候補にする
    final_best_w = 0
    final_best_acc = -1.0
    final_best_loss = float('inf')

    # 最終世代の全個体を固定5,000枚で評価する
    
        # 最終世代の全200個体を固定5,000枚で評価する
    for i in range(population):
        acc, loss = evaluate(model, population_w[i], x_final, y_final, device)

        # 正答率が高い個体を選ぶ
        # 同率の場合はLossが小さい個体を選ぶ
        if (acc > final_best_acc or (acc == final_best_acc and loss < final_best_loss)):
            final_best_acc = acc
            final_best_loss = loss
            final_best_w = population_w[i].clone()

    print("\n固定5000枚による最終選択")
    print(
        f"Accuracy: {final_best_acc * 100:.2f}% | "
        f"Loss: {final_best_loss:.4f} | ",
    )

    # 選択した個体をモデルに反映
    evaluate(model, final_best_w, x_final, y_final, device)

  

    return final_best_w, history
    
    
def save_csv(after_acc, after_loss):
    """
    GA前後の公式テストAccuracyとLossをCSVに保存する。
    """

    csv_path = ('./output/ga_only_cifar_3conv_200.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as file:

        writer = csv.writer(file)

        writer.writerow([
            'stage',
            'test_accuracy_percent',
            'test_loss'
        ])

        writer.writerow([
            'after_ga',
            after_acc * 100,
            after_loss
        ])


def main():

    eval_size = 1500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    model = GACifar10().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

   # GA進化用45,000枚の画像番号
    ga_train_indices = list(range(45000))
    # GA最終選択用5,000枚の画像番号
    ga_final_indices = list(range(45000, 50000))

    x_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).to(device)

    # GA実行とGA後の評価
    best_w, history = genetic_algorithm(model, train_dataset, ga_train_indices, ga_final_indices, device, eval_size)
    after_acc, after_loss = evaluate(model, best_w, x_test, y_test, device)

    print(f"Test Accuracy: {after_acc * 100:.2f}%")

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
    plt.savefig('./output/ga_only_cifar_3conv_200.png')
    plt.close()

    save_csv(after_acc, after_loss)

if __name__ == '__main__':
    main()

