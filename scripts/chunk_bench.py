"""
chunk(1回にモデルへ渡す画像の枚数)ごとの、1個体あたりの評価時間を比べる。GAは回さない。
使い方: uv run bench_chunk.py   (1分前後)

  - 評価する画像は 20,000枚、評価する個体はランダムな±1の重み(時間の測定用)
  - chunk=20000 が「一括」(今のGAと同じ)。小さくするほど、分けて流す回数が増える。
  - 各chunkを3回測って中央値を出す(ばらつき対策)
  - 結果(正解数・Loss)が一括と同じになることも確認する
"""
import time
import statistics
import torch
import torch.nn.functional as F
from torchvision import datasets
from binaryconnect_cifar10 import BinaryConnectCifar10

LOAD_PATH = './output/binaryconnect_cifar_noaug_3conv_nobias_128_seed44_300ep_best_val.pt'
N_IMAGES = 20000
CHUNKS = [20000, 5000, 2000, 1000, 800, 500, 400, 250, 200, 100, 50]
WARMUP = 5
REPEAT = 10       # 1回の測定で評価する個体数
ROUNDS = 3        # 測定を繰り返す回数(中央値を使う)
HALF = True       # fp16入力にする(CPUで試すときだけFalse)
POPULATION = 500  # 1世代の個体数(世代あたりの時間の換算に使う)

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
CUDA = torch.cuda.is_available()


def sync():
    if CUDA:
        torch.cuda.synchronize()


def set_weights(model, w):
    offset = 0
    for layer in model.layers:
        size = layer.weight.numel()
        layer.weight.data.copy_(w[offset:offset + size].view_as(layer.weight))
        offset += size


@torch.no_grad()
def evaluate_chunked(model, w, x, y, chunk):
    """x を chunk 枚ずつに分けて流し、正解数とLossの合計を返す(.item()なし)。"""
    set_weights(model, w)
    n = x.shape[0]
    correct = torch.zeros((), device=x.device)
    loss_sum = torch.zeros((), device=x.device)
    for s in range(0, n, chunk):
        xb, yb = x[s:s + chunk], y[s:s + chunk]
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(xb)
        out = out.float()
        correct += (out.argmax(dim=1) == yb).sum()
        loss_sum += F.cross_entropy(out, yb, reduction='sum')
    return correct, loss_sum


def time_one(model, x, y, chunk, ws):
    """REPEAT個体ぶんの平均時間(ms/個体)を1回測る。"""
    for i in range(WARMUP):
        evaluate_chunked(model, ws[i], x, y, chunk)
    sync()
    t0 = time.perf_counter()
    for i in range(WARMUP, WARMUP + REPEAT):
        evaluate_chunked(model, ws[i], x, y, chunk)
    sync()
    return (time.perf_counter() - t0) / REPEAT * 1000


def run_bench(model, x32, y):
    n = x32.shape[0]
    device = x32.device
    n_weight = sum(l.weight.numel() for l in model.layers)

    g = torch.Generator().manual_seed(0)
    ws = [torch.where(torch.rand(n_weight, generator=g) < 0.5, -1.0, 1.0).to(device)
          for _ in range(WARMUP + REPEAT)]
    origin = torch.cat([l.weight.detach().reshape(-1) for l in model.layers])
    w_ref = torch.where(origin >= 0, 1.0, -1.0)
    w_ref[(torch.rand(n_weight, generator=g) < 0.001).to(device)] *= -1.0   # 基準個体(元BP+約100個反転)

    x = x32.half().contiguous() if HALF else x32.contiguous()

    rows = []
    for chunk in CHUNKS:
        chunk = min(chunk, n)
        try:
            ms = statistics.median(time_one(model, x, y, chunk, ws) for _ in range(ROUNDS))
            c, l = evaluate_chunked(model, w_ref, x, y, chunk)
            rows.append((chunk, ms, int(c.item()), l.item() / n))
        except Exception as e:
            rows.append((chunk, None, str(e).strip().splitlines()[0][:80], None))

    ok = [r for r in rows if r[1] is not None]
    base = next(r for r in ok if r[0] == min(CHUNKS[0], n))      # chunk=20000(一括)
    best = min(ok, key=lambda r: r[1])

    print("\n" + "=" * 86)
    print(f"{'chunk':>7s}{'分割回数':>9s}{'ms/個体':>10s}{'枚/秒':>12s}{'一括比':>8s}{'正解数差':>10s}{'Loss差':>11s}")
    print("=" * 86)
    for chunk, ms, c, l in rows:
        if ms is None:
            print(f"{chunk:7d}  失敗: {c}")
            continue
        mark = '  ← 最速' if (chunk, ms) == (best[0], best[1]) else ''
        print(f"{chunk:7d}{-(-n // chunk):9d}{ms:10.2f}{n / ms * 1000:12,.0f}{base[1] / ms:7.2f}x"
              f"{c - base[2]:10d}{l - base[3]:11.5f}{mark}")
    print("=" * 86)
    print(f"最速: chunk={best[0]} → {best[1]:.2f} ms/個体 (一括の {base[1] / best[1]:.2f} 倍)")
    print(f"{POPULATION}個体/世代の順伝播時間: 一括 {base[1] * POPULATION / 1000:.1f} s → chunk={best[0]} {best[1] * POPULATION / 1000:.1f} s")
    near = [r[0] for r in ok if r[1] <= best[1] * 1.03]
    print(f"最速の3%以内: chunk = {near}  (この中の大きめの値を選ぶと、環境差に強い)")
    print("正解数差=一括と比べた正解枚数の差(0〜数枚ならfp16の丸め誤差の範囲)", flush=True)


def main():
    device = torch.device('cuda' if CUDA else 'cpu')
    print(f"使用デバイス: {device} | torch {torch.__version__}"
          f" | GPU: {torch.cuda.get_device_name(0) if CUDA else 'なし'}")

    checkpoint = torch.load(LOAD_PATH, map_location='cpu', weights_only=True)
    model = BinaryConnectCifar10().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ds = datasets.CIFAR10('./data', train=True, download=True)
    x = torch.from_numpy(ds.data[:N_IMAGES]).to(device).permute(0, 3, 1, 2).float().div(255.0)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x32 = ((x - mean) / std).contiguous()          # fp32, NCHW
    y = torch.tensor(ds.targets[:N_IMAGES], device=device)

    run_bench(model, x32, y)


if __name__ == '__main__':
    main()