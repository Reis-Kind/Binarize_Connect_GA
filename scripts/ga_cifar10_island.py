import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from binaryconnect_cifar10 import BinaryConnectCifar10


# 島モデルの設定。genetic_algorithm() / save_csv() / main() で共通して使う。
NUM_ISLANDS = 5


def format_duration(seconds):
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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
            size = layer.weight.numel()
            new_weight = w[offset:offset + size].view_as(layer.weight).contiguous()
            layer.weight.data.copy_(new_weight)
            offset += size

        x, y = x.to(device), y.to(device)
        n = len(x)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for s in range(0, n, eval_batch):
            xb, yb = x[s:s + eval_batch], y[s:s + eval_batch]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(xb)
                loss_sum += nn.functional.cross_entropy(outputs, yb, reduction='sum')
                correct += (outputs.argmax(dim=1) == yb).sum()
        loss = (loss_sum / n).item()
        acc = (correct / n).item()

    return acc, loss


def make_initial_population(origin_w, island_population, island_random, island_origin, num_islands):
    """
    島ごとに「元個体(island_origin体)+突然変異個体+ランダム個体」の初期集団を作る。
    各島を同じ構成比で始めることで、島ごとの違いがmigrationと偶然のばらつき
    だけから生まれるようにする(片方の島だけランダム個体ばかり、みたいな偏りを避ける)。
    """
    device = origin_w.device
    n_weight = origin_w.numel()
    island_mutated = island_population - island_random - island_origin

    island_tensors = []
    for _ in range(num_islands):
        island_w = origin_w.repeat(island_population, 1).clone()

        # 元個体をもとに突然変異させた個体(先頭island_origin体は元個体のまま残す)
        for i in range(island_origin, island_mutated + island_origin):
            j = i - island_origin

            if j < int(island_mutated * 0.7):
                individual_rate = 0.0001
            elif j < int(island_mutated * 0.9):
                individual_rate = 0.001
            else:
                individual_rate = 0.01

            mutation_mask = (torch.rand(n_weight, device=device) < individual_rate)
            island_w[i, mutation_mask] *= -1.0

        # 残りは事前BPとは無関係な完全ランダム二値重み
        island_w[island_mutated + island_origin:] = torch.where(
            torch.rand(island_random, n_weight, device=device) < 0.5, -1.0, 1.0,
        )

        island_tensors.append(island_w)

    # 島を縦に並べて1本の集団テンソルにする。
    # island k の個体は [k*island_population : (k+1)*island_population) に入る。
    return torch.cat(island_tensors, dim=0)


def tournament_select(scores, losses, k, pool):
    """
    pool: このトーナメントで候補にしてよい個体indexの集まり(島ごとに絞る)。
    """
    candidates = np.random.choice(pool, k, replace=False)
    best_idx = candidates[0]

    for i in candidates:
        if scores[i] > scores[best_idx] or (scores[i] == scores[best_idx] and losses[i] < losses[best_idx]):
            best_idx = i

    return best_idx


def mean_hamming_distance(vectors):
    """
    個体間の平均Hamming距離を0～1(全ビット一致=0, 全ビット不一致=1)で返す。
    島同士がちゃんと分化しているかを見るための目安。
    """
    if len(vectors) < 2:
        return 0.0

    n_weight = vectors[0].numel()
    total = 0.0
    pairs = 0
    for a in range(len(vectors)):
        for b in range(a + 1, len(vectors)):
            total += (vectors[a] != vectors[b]).sum().item() / n_weight
            pairs += 1

    return total / pairs


def genetic_algorithm(model, x_all, y_all, train_indices, final_indices, device, eval_size, x_test, y_test):
    """
    島モデルGA。
    populationをnum_islands個の島に分け、選択(トーナメント)・エリート保存・交叉は
    島の中だけで行う。migration_interval世代ごとに、島iの上位migration_size体を
    島(i+1)(リング状)へ移住させ、移住先の最も変異率が高い(=選択圧の弱い)枠を置き換える。

    population=500, random=50, origin=1, elite_size=50 という元の設定を
    5島ぶんに割り振った値になっている(island_population*num_islands=500,
    island_random*num_islands=50, elite_size*num_islands=50)。
    ただしorigin(元BP個体)だけは島ごとに1体ずつ持たせるため、元の「全体で1体」から
    「全体でnum_islands体」に変わっている点は元の設定との差分として注意。
    """
    num_islands = NUM_ISLANDS
    island_population = 100
    island_random = 10
    island_origin = 1
    elite_size = 10           # 島ごとのエリート数(全体では elite_size * num_islands = 50)
    generations = 500
    random_seed = 42

    # 移住(migration)の設定
    migration_interval = 50    # 何世代ごとに移住するか(0以下で移住なし)
    migration_size = 2         # 1回の移住で島から送り出す個体数

    population = island_population * num_islands

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # 画像抽出専用の乱数生成器
    data_generator = torch.Generator().manual_seed(random_seed)

    # 全対象層の重みを1次元にまとめる
    origin_w = torch.cat([layer.weight.detach().reshape(-1) for layer in model.layers])
    # 二値化
    origin_w = torch.where(origin_w >= 0, 1.0, -1.0)
    # 探索する重み総数を取得
    n_weight = origin_w.numel()

    # 初期個体を島ごとに作成
    population_w = make_initial_population(
        origin_w, island_population, island_random, island_origin, num_islands
    )

    # 最良個体を元のパラメータで初期化
    best_w = origin_w.clone()
    history = []
    best_candidates = []

    # GA終了後の個体選択に使う固定5,000枚
    final_idx_t = torch.tensor(final_indices, device=device)
    x_final = x_all[final_idx_t]
    y_final = y_all[final_idx_t]

    # 全個体を評価
    ga_start = time.perf_counter()
    for gen in range(generations + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        generation_start = time.perf_counter()

        order = torch.randperm(len(train_indices), generator=data_generator)[:eval_size].tolist()

        # CIFAR-10データセット内の画像番号に変換
        indices = [train_indices[i] for i in order]

        # 今回の世代で使う画像とラベル(全個体・全島で共通)
        sel = torch.tensor(indices, device=device)
        x_eval = x_all[sel]
        y_eval = y_all[sel]

        x_eval_folds = torch.tensor_split(x_eval, 4, dim=0)
        y_eval_folds = torch.tensor_split(y_eval, 4, dim=0)

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

            # 平均精度を保ちつつ、群ごとのばらつきを少し減点する
            robust_score = mean_acc - 0.5 * accuracy_std

            scores.append(robust_score)
            mean_accuracies.append(mean_acc)
            accuracy_stds.append(accuracy_std)
            losses.append(mean_loss)

        # 全体での順位(ログ・全体最良個体の記録用。選択自体はこの後、島ごとに絞る)
        ranked_indices = sorted(
            range(population),
            key=lambda i: (scores[i], -losses[i]), reverse=True
        )

        # 全体で最も良い個体(どの島由来かは問わない)
        best_idx = ranked_indices[0]
        best_w = population_w[best_idx].clone()

        best_acc = mean_accuracies[best_idx]
        best_std = accuracy_stds[best_idx]
        best_score = scores[best_idx]
        best_loss = losses[best_idx]
        best_island = best_idx // island_population

        best_candidates.append(best_w.clone())

        bp_acc, bp_loss = evaluate(model, origin_w, x_eval, y_eval, 500, device)
        test_acc, test_loss = evaluate(model, best_w, x_test, y_test, 500, device)

        changed = (best_w != origin_w).sum().item()

        # 島ごとのランキング(選択・エリート・移住で使う)と、島ごとの最良個体
        island_ranked = []
        island_best_accuracies = []
        island_best_vectors = []

        for isl in range(num_islands):
            isl_start = isl * island_population
            isl_end = isl_start + island_population
            isl_indices = list(range(isl_start, isl_end))

            isl_rank = sorted(isl_indices, key=lambda i: (scores[i], -losses[i]), reverse=True)
            island_ranked.append(isl_rank)
            island_best_accuracies.append(mean_accuracies[isl_rank[0]] * 100)
            island_best_vectors.append(population_w[isl_rank[0]])

        # 島の最良個体同士がどれくらい離れているか(多様性が潰れていないかの目安)
        inter_island_diversity = mean_hamming_distance(island_best_vectors) * 100

        print(
            f"  元BP: {bp_acc * 100:.2f}% | "
            f"全体最良: {best_acc * 100:.2f}%(島{best_island}) | "
            f"正答率差: {(best_acc - bp_acc) * 100:+.2f}ポイント | "
            f"4群の標準偏差: {best_std * 100:.3f}ポイント | "
            f"安定性スコア: {best_score * 100:.2f}% | "
            f"島間多様性: {inter_island_diversity:.2f}%"
        )
        print(
            "  島ごとの最良(サブサンプル)正答率: "
            + ", ".join(f"島{i}={acc:.2f}%" for i, acc in enumerate(island_best_accuracies))
        )

        migrated_this_gen = (
            migration_interval > 0
            and gen > 0
            and gen < generations
            and gen % migration_interval == 0
        )

        history.append({
            'generation': gen,
            'accuracy': best_acc,
            'accuracy_std': best_std,
            'stability_score': best_score,
            'loss': best_loss,
            'changed_weights': changed,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'best_island': best_island,
            'inter_island_diversity': inter_island_diversity,
            'migrated': migrated_this_gen,
            'island_best_accuracies': island_best_accuracies,
        })

        # 完了した世代までを毎回保存する
        save_csv(history, num_islands)

        print(
            f"世代 [{gen}/{generations}] | "
            f"Train Acc: {best_acc * 100:.2f}% | "
            f"Fold SD: {best_std * 100:.3f}pt | "
            f"Stability: {best_score * 100:.2f}% | "
            f"Train Loss: {best_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"符号変化: {changed}個" + ("  ※この世代で移住" if migrated_this_gen else ""),
            flush=True
        )

        if gen == generations:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - ga_start
            print(
                f"時間 | 世代: {format_duration(time.perf_counter() - generation_start)} | "
                f"累計: {format_duration(elapsed)}",
                flush=True
            )
            break

        # 現在の集団を保存してから次世代を作る
        parent_population = population_w
        next_population = torch.empty_like(population_w)

        island_migrants = []

        for isl in range(num_islands):
            isl_start = isl * island_population
            isl_end = isl_start + island_population
            isl_candidates = island_ranked[isl]  # 島内の個体を順位順に並べたもの(選択候補プール)

            # 島内エリートをそのまま次世代へ保存
            isl_elite_indices = isl_candidates[:elite_size]
            for next_i, elite_idx in zip(range(isl_start, isl_start + elite_size), isl_elite_indices):
                next_population[next_i] = parent_population[elite_idx]

            # 移住に使う候補(島内上位migration_size体)
            island_migrants.append(
                [parent_population[i].clone() for i in isl_candidates[:migration_size]]
            )

            # エリート以外は島内だけで交叉・突然変異(他の島の個体とは交叉しない)
            for i in range(isl_start + elite_size, isl_end):
                parent1_idx = tournament_select(scores, losses, 3, isl_candidates)
                parent2_idx = tournament_select(scores, losses, 3, isl_candidates)

                # 同じ個体同士の交叉を避ける
                while parent2_idx == parent1_idx:
                    parent2_idx = tournament_select(scores, losses, 3, isl_candidates)

                # 2個体による一様交叉
                cross_mask = torch.rand(n_weight, device=device) < 0.5
                child = torch.where(cross_mask, parent_population[parent1_idx], parent_population[parent2_idx])

                # 二値重みの符号を突然変異させる
                # 子個体を，突然変異率の異なる3群に分ける(島内で完結させる)
                n_children = island_population - elite_size
                child_index = i - (isl_start + elite_size)

                if child_index < int(n_children * 0.7):
                    child_mutation_rate = 0.0001
                elif child_index < int(n_children * 0.9):
                    child_mutation_rate = 0.001
                else:
                    child_mutation_rate = 0.01

                mutation_mask = (torch.rand(n_weight, device=device) < child_mutation_rate)
                child[mutation_mask] *= -1.0
                next_population[i] = child

        # リングmigration: 島iの上位個体が島(i+1)の末尾(変異率が最も高い=選択圧が弱い枠)を置き換える
        if migrated_this_gen:
            for isl in range(num_islands):
                dest = (isl + 1) % num_islands
                dest_start = dest * island_population
                for k, migrant in enumerate(island_migrants[isl]):
                    next_population[dest_start + island_population - 1 - k] = migrant

        # 新しい集団に更新
        population_w = next_population

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - ga_start
        print(
            f"時間 | 世代: {format_duration(time.perf_counter() - generation_start)} | "
            f"累計: {format_duration(elapsed)}",
            flush=True
        )

    # 元BPを最初の候補にする
    final_best_w = origin_w.clone()
    final_best_acc, final_best_loss = evaluate(model, final_best_w, x_final, y_final, 500, device)

    # 最終世代の全個体と，今までの世代の最高スコアの個体を固定5,000枚で評価
    candidates = list(population_w) + best_candidates[:-1]
    for candidate_w in candidates:
        acc, loss = evaluate(model, candidate_w, x_final, y_final, 500, device)

        # 正答率が高い個体を選ぶ。同率の場合はLossが小さい個体を選ぶ
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
    evaluate(model, final_best_w, x_final, y_final, 500, device)

    return final_best_w, history


def save_csv(history, num_islands=NUM_ISLANDS, after_acc=None, after_loss=None):
    """
    GA前後の公式テストAccuracyとLossをCSVに保存する(島モデル版)。
    元のga_cifar_noaug_45000.csvとは別名で保存するので、単一集団版の結果と
    並べて比較できる。
    """

    csv_path = './output/ga_island_noaug_45000.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    island_cols = [f'island{i}_best_accuracy_percent' for i in range(num_islands)]

    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow([
            'stage',
            'generation',
            'sample_accuracy_percent',
            'sample_accuracy_std_percent',
            'stability_score_percent',
            'sample_loss',
            'changed_weights',
            'test_accuracy_percent',
            'test_loss',
            'best_island',
            'inter_island_diversity_percent',
            'migrated',
        ] + island_cols)

        for h in history:
            row = [
                'generation',
                h['generation'],
                h['accuracy'] * 100,
                h['accuracy_std'] * 100,
                h['stability_score'] * 100,
                h['loss'],
                h['changed_weights'],
                h['test_accuracy'] * 100,
                h['test_loss'],
                h['best_island'],
                h['inter_island_diversity'],
                h['migrated'],
            ] + h['island_best_accuracies']
            writer.writerow(row)

        if after_acc is not None and after_loss is not None:
            # generation〜changed_weightsの6列を空欄にしてtest列の位置を合わせ、
            # best_island以降(3+num_islands列)も空欄で埋めている
            writer.writerow(
                ['after_ga'] + [''] * 6 + [after_acc * 100, after_loss] + [''] * (3 + num_islands)
            )


def main():
    program_start = time.perf_counter()

    eval_size = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # BPの学習済みモデルを読み込む
    load_path = './output/binaryconnect_cifar_noaug_3conv_nobias_128_seed44_300ep_best_val.pt'
    checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
    model = BinaryConnectCifar10().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    train_dataset = datasets.CIFAR10('./data', train=True, download=True)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True)
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
    before_acc, before_loss = evaluate(model, origin_w, x_test, y_test, 500, device)

    # GA実行とGA後の評価
    best_w, history = genetic_algorithm(
        model, x_train_all, y_train_all, ga_train_indices, ga_final_indices, device, eval_size, x_test, y_test
    )
    after_acc, after_loss = evaluate(model, best_w, x_test, y_test, 500, device)

    print(f"Test Accuracy: {before_acc * 100:.2f}% → {after_acc * 100:.2f}%")
    print(f"変化: {(after_acc - before_acc) * 100:+.2f}ポイント")

    generations = [h['generation'] for h in history]

    plt.figure(figsize=(12, 5))

    # 正答率
    plt.subplot(1, 2, 1)

    plt.plot(generations, [h['accuracy'] * 100 for h in history], label='Sampled Train Accuracy', color='tab:green')
    plt.plot(generations, [h['test_accuracy'] * 100 for h in history], label='Test Accuracy', color='tab:blue')
    plt.axhline(y=before_acc * 100, label='BP Test Accuracy', color='tab:gray', linestyle='--')

    plt.xlabel('Generation')
    plt.ylabel('Accuracy (%)')
    plt.title('BP + Island-Model GA Accuracy')
    plt.grid(True)
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)

    plt.plot(generations, [h['loss'] for h in history], label='Sampled Train Loss', color='tab:red')
    plt.plot(generations, [h['test_loss'] for h in history], label='Test Loss', color='tab:orange')
    plt.axhline(y=before_loss, label='BP Test Loss', color='tab:gray', linestyle='--')

    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.title('BP + Island-Model GA Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    os.makedirs('./output', exist_ok=True)
    plt.savefig('./output/ga_island_45000.png', dpi=300)
    plt.close()

    save_csv(history, NUM_ISLANDS, after_acc, after_loss)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(f"全体の実行時間: {format_duration(time.perf_counter() - program_start)}", flush=True)


if __name__ == '__main__':
    main()