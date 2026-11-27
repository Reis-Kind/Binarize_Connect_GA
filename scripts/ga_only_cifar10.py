import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RUN_NAME = "ga_only_cifar_pop1000_eval20000"


def format_duration(seconds):
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class GACifar10(nn.Module):
    def __init__(self):
        super().__init__()

        # RGBで3チャネルあるから3（Mnistは白黒だから1）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1,  bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,  bias=False)
        self.fc = nn.Linear(128 * 4 * 4, 10, bias=False)
        self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.fc])

        # 神の一手
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn3 = nn.BatchNorm2d(128, track_running_stats=False)

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

def load_cifar_tensors(dataset, device):
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.2023, 0.1994, 0.2010), device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(dataset.data).to(device).permute(0, 3, 1, 2).float().div(255.0)
    x = ((x - mean) / std).half().contiguous()
    y = torch.tensor(dataset.targets, device=device)
    return x, y


def evaluate(model, w, x, y, eval_batch, device):
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

        if offset != w.numel():
            raise ValueError("モデルと遺伝子の重み数が一致しません。")

        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)

        for s in range(0, len(x), eval_batch):
            xb, yb = x[s:s + eval_batch], y[s:s + eval_batch]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(xb)
                loss_sum += nn.functional.cross_entropy(outputs, yb, reduction='sum')
                correct += (outputs.argmax(dim=1) == yb).sum()

        loss = (loss_sum / len(x)).item()
        acc = (correct / len(x)).item()
    
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


def tournament_select(scores, losses, k):
    """
    
    """
    candidates = np.random.choice(len(scores), k, replace=False)
    best_idx = candidates[0]

    for i in candidates:
        if scores[i] > scores[best_idx] or (scores[i] == scores[best_idx] and losses[i] < losses[best_idx]):
            best_idx = i

    return best_idx


def genetic_algorithm(
    model, x_train_all, y_train_all,
    train_indices, final_indices, device, eval_size, x_test, y_test
):
    """
    
    """
    population = 1000
    generations = 10000
    elite_size = 50
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
    best_candidates = []

    # 全個体を評価
    for gen in range(generations + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        generation_start = time.perf_counter()

        # GA用データから重複なしで評価画像を選ぶ
        order = torch.randperm(
            len(train_indices),
            generator=data_generator
        )[:eval_size].tolist()

        # CIFAR-10データセット内の画像番号に変換
        indices = [train_indices[i] for i in order]

        # 今回の世代で使う画像とラベル
        sel = torch.tensor(indices, device=device)
        x_eval = x_train_all[sel]
        y_eval = y_train_all[sel]

        x_eval_folds = torch.tensor_split(x_eval, 4, dim=0)
        y_eval_folds = torch.tensor_split(y_eval, 4, dim=0)

        # 前の世代から持ち越した候補を、
        # 今回の画像で評価し直す

        scores = []
        mean_accuracies = []
        accuracy_stds = []
        losses = []

        for i in range(population):
            fold_accs = []
            fold_losses = []

            for x_fold, y_fold in zip(x_eval_folds, y_eval_folds):
                acc, loss = evaluate(model, population_w[i], x_fold, y_fold, 500, device)
                fold_accs.append(acc)
                fold_losses.append(loss)

            mean_acc = float(np.mean(fold_accs))
            accuracy_std = float(np.std(fold_accs))
            mean_loss = float(np.mean(fold_losses))
            robust_score = mean_acc - 0.5 * accuracy_std

            scores.append(robust_score)
            mean_accuracies.append(mean_acc)
            accuracy_stds.append(accuracy_std)
            losses.append(mean_loss)

        # 安定性スコアの降順、同率なら平均Lossの昇順
        ranked_indices = sorted(
            range(population),
            key=lambda i: (scores[i], -losses[i]),
            reverse=True
        )

        # この世代で最も良い個体
        best_idx = ranked_indices[0]
        best_w = population_w[best_idx].clone()

        best_candidates.append(best_w.clone())

        best_acc = mean_accuracies[best_idx]
        best_std = accuracy_stds[best_idx]
        best_score = scores[best_idx]
        best_loss = losses[best_idx]

        test_acc, test_loss = evaluate(model, best_w, x_test, y_test, 500, device)

        history.append({
            'generation': gen,
            'accuracy': best_acc,
            'accuracy_std': best_std,
            'stability_score': best_score,
            'loss': best_loss,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
        })

        # 
        if gen == generations:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - generation_start
            history[-1]['generation_seconds'] = generation_seconds
            save_csv(history)
            print(
                f"世代 [{gen}/{generations}] | "
                f"Train Acc: {best_acc * 100:.2f}% | "
                f"Fold SD: {best_std * 100:.3f}pt | "
                f"Stability: {best_score * 100:.2f}% | "
                f"Train Loss: {best_loss:.4f} | "
                f"Test Acc: {test_acc * 100:.2f}% | "
                f"Test Loss: {test_loss:.4f} | "
                f"世代時間: {generation_seconds:.2f}秒",
                flush=True
            )
            break

        # 現在の集団を保存してから次世代を作る
        parent_population = population_w.clone()
        next_population = torch.empty_like(population_w)

        # 上位個体をそのまま次世代へ保存
        elite_indices = ranked_indices[:elite_size]

        for next_i, elite_idx in enumerate(elite_indices):
            next_population[next_i] = parent_population[elite_idx].clone()

        # 交叉と突然変異
        for i in range(elite_size, population):
            parent1_idx = tournament_select(scores, losses, 6)
            parent2_idx = tournament_select(scores, losses, 6)

            # 同じ個体同士の交叉を避ける
            while parent2_idx == parent1_idx:
                parent2_idx = tournament_select(scores, losses, 6)

            # 2個体による一様交叉
            cross_mask = torch.rand(n_weight, device=device) < 0.5
            child = torch.where(cross_mask, parent_population[parent1_idx], parent_population[parent2_idx])

            # 二値重みの符号を突然変異させる
            n_children = population - elite_size
            child_index = i - elite_size

            if child_index < int(n_children * 0.7):
                child_mutation_rate = 0.0001
            elif child_index < int(n_children * 0.9):
                child_mutation_rate = 0.001
            else:
                child_mutation_rate = 0.01

            mutation_mask = torch.rand(n_weight, device=device) < child_mutation_rate
            child[mutation_mask] *= -1.0
            next_population[i] = child

        # 新しい集団に更新
        population_w = next_population

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_start
        history[-1]['generation_seconds'] = generation_seconds
        save_csv(history)
        print(
            f"世代 [{gen}/{generations}] | "
            f"Train Acc: {best_acc * 100:.2f}% | "
            f"Fold SD: {best_std * 100:.3f}pt | "
            f"Stability: {best_score * 100:.2f}% | "
            f"Train Loss: {best_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"世代時間: {generation_seconds:.2f}秒",
            flush=True
        )


    # GA終了後の個体選択に使う固定5,000枚
    final_sel = torch.tensor(final_indices, device=device)
    x_final = x_train_all[final_sel]
    y_final = y_train_all[final_sel]

    final_best_w = None
    final_best_acc = -1.0
    final_best_loss = float('inf')

    # 最終世代の個体と各世代の最良個体を固定5,000枚で比較する
    candidates = list(population_w) + best_candidates[:-1]

    for candidate_w in candidates:
        acc, loss = evaluate(model, candidate_w, x_final, y_final, 500, device)

        if acc > final_best_acc or (acc == final_best_acc and loss < final_best_loss):
            final_best_acc = acc
            final_best_loss = loss
            final_best_w = candidate_w.clone()

    print("\n固定5000枚による最終選択")
    print(
        f"Accuracy: {final_best_acc * 100:.2f}% | "
        f"Loss: {final_best_loss:.4f} | ",
    )

    # 選択した個体をモデルに反映
    evaluate(model, final_best_w, x_final, y_final, 500, device)

  

    return final_best_w, history
    
    
def save_csv(history, after_acc=None, after_loss=None):
    """
    GA前後の公式テストAccuracyとLossをCSVに保存する。
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{RUN_NAME}.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow([
            'stage',
            'generation',
            'train_accuracy_percent',
            'train_accuracy_std_percent',
            'stability_score_percent',
            'train_loss',
            'test_accuracy_percent',
            'test_loss'
        ])
        for h in history:
            writer.writerow([
                'generation',
                h['generation'],
                h['accuracy'] * 100,
                h['accuracy_std'] * 100,
                h['stability_score'] * 100,
                h['loss'],
                h['test_accuracy'] * 100,
                h['test_loss'],
            ])

        # 最終選択後のテスト結果
        if after_acc is not None and after_loss is not None:
            writer.writerow([
                'after_ga', '', '', '', '', '',
                after_acc * 100, after_loss
            ])


def main():
    program_start = time.perf_counter()

    eval_size = 20000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    model = GACifar10().to(device)

    train_dataset = datasets.CIFAR10('./data', train=True, download=True)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True)
    x_train_all, y_train_all = load_cifar_tensors(train_dataset, device)
    x_test, y_test = load_cifar_tensors(test_dataset, device)

   # GA進化用45,000枚の画像番号
    ga_train_indices = list(range(45000))
    # GA最終選択用5,000枚の画像番号
    ga_final_indices = list(range(45000, 50000))

    # GA実行とGA後の評価
    best_w, history = genetic_algorithm(
        model, x_train_all, y_train_all,
        ga_train_indices, ga_final_indices, device, eval_size, x_test, y_test
    )
    after_acc, after_loss = evaluate(model, best_w, x_test, y_test, 500, device)

    print(f"Test Accuracy: {after_acc * 100:.2f}%")

    generations = [h['generation'] for h in history]

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(
        generations,
        [h['accuracy'] * 100 for h in history],
        color='tab:green',
        label='Sampled Training Accuracy'
    )
    plt.plot(
        generations,
        [h['test_accuracy'] * 100 for h in history],
        color='tab:blue',
        label='Test Accuracy'
    )
    plt.xlabel('Generation')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.grid(True)
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(
        generations,
        [h['loss'] for h in history],
        color='tab:red',
        label='Sampled Training Loss'
    )
    plt.plot(
        generations,
        [h['test_loss'] for h in history],
        color='tab:orange',
        label='Test Loss'
    )
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    graph_path = os.path.join(OUTPUT_DIR, f"{RUN_NAME}.png")
    plt.savefig(graph_path, dpi=300)
    plt.close()

    print(f"グラフを '{graph_path}' に保存しました。")

    save_csv(history, after_acc, after_loss)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(
        f"全体の実行時間: {format_duration(time.perf_counter() - program_start)}",
        flush=True
    )

if __name__ == '__main__':
    main()

