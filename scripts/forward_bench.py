"""
1個体(20,000枚)の順伝播だけを、いろいろな設定で比べるベンチマーク。GAは回さない。
使い方: uv run bench_forward.py   (1〜3分ほど)

比べるもの
  - 入力: fp32(現状) / fp16
  - メモリ配置: NCHW(現状) / channels_last
  - 1回に流す枚数(chunk): 20000(一括, 現状) / 10000 / 5000 / 2000 / 1000 / 500
  - cudnn.benchmark: False(現状) / True
  - (任意) torch.compile
どの設定でも、同じ個体に対する正解数とLossが基準と(ほぼ)一致するかも表示する。
"""
import time
import torch
import torch.nn.functional as F
from torchvision import datasets
from binaryconnect_cifar10 import BinaryConnectCifar10

LOAD_PATH = './output/binaryconnect_cifar_noaug_3conv_nobias_128_seed44_300ep_best_val.pt'
N_IMAGES = 20000
WARMUP = 3
REPEAT = 10
CHUNKS = [None, 10000, 5000, 2000, 1000, 500]   # None = 一括(N_IMAGES枚)
TRY_COMPILE = False   # Trueで最速の設定にtorch.compileも試す(コンパイルに1〜数分。環境によっては失敗する)

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
def evaluate_gpu(model, w, x, y, chunk=None, fwd=None):
    """正解数(枚)とLossの合計をGPUテンソルで返す。.item()は呼ばない。"""
    f = fwd if fwd is not None else model
    set_weights(model, w)
    n = x.shape[0]
    if chunk is None or chunk >= n:
        with torch.autocast('cuda', dtype=torch.float16):
            out = f(x)
        out = out.float()
        return (out.argmax(dim=1) == y).sum().float(), F.cross_entropy(out, y, reduction='sum')
    correct = torch.zeros((), device=x.device)
    loss_sum = torch.zeros((), device=x.device)
    for s in range(0, n, chunk):
        xb, yb = x[s:s + chunk], y[s:s + chunk]
        with torch.autocast('cuda', dtype=torch.float16):
            out = f(xb)
        out = out.float()
        correct += (out.argmax(dim=1) == yb).sum()
        loss_sum += F.cross_entropy(out, yb, reduction='sum')
    return correct, loss_sum


def set_format(model, channels_last):
    model.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)


def prepare_x(x32, channels_last, half=True):
    x = x32.half() if half else x32
    return x.contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)


def measure(model, x, y, chunk, ws, w_ref, fwd=None):
    """1個体あたりの平均時間(ms)と、基準個体の(正解数, 平均Loss)を返す。失敗したらNone。"""
    try:
        for i in range(WARMUP):
            evaluate_gpu(model, ws[i], x, y, chunk, fwd)
        sync()
        t0 = time.perf_counter()
        for i in range(WARMUP, WARMUP + REPEAT):
            evaluate_gpu(model, ws[i], x, y, chunk, fwd)
        sync()
        ms = (time.perf_counter() - t0) / REPEAT * 1000
        correct, loss_sum = evaluate_gpu(model, w_ref, x, y, chunk, fwd)
        return ms, int(correct.item()), loss_sum.item() / x.shape[0]
    except Exception as e:
        return str(e).strip().splitlines()[0][:90]


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

    rows = []   # (名前, ms, correct, loss) または (名前, エラー文字列)

    def add(name, res):
        rows.append((name,) + (res if isinstance(res, tuple) else (res,)))

    # 基準: 現状と同じ(fp32入力, NCHW, 一括, benchmark=False)
    torch.backends.cudnn.benchmark = False
    set_format(model, False)
    add('基準: fp32入力 NCHW 一括 (現状)', measure(model, prepare_x(x32, False, half=False), y, None, ws, w_ref))
    base_ms, base_correct, base_loss = rows[0][1], rows[0][2], rows[0][3]

    configs = []
    for bench in (False, True):
        for cl in (False, True):
            for chunk in CHUNKS:
                configs.append((bench, cl, chunk))

    for bench, cl, chunk in configs:
        torch.backends.cudnn.benchmark = bench
        x = None
        try:
            set_format(model, cl)
            x = prepare_x(x32, cl)
        except Exception as e:
            pass
        name = f"fp16入力 {'channels_last' if cl else 'NCHW        '} chunk={'一括' if chunk is None else chunk:<5} benchmark={bench}"
        add(name, measure(model, x, y, chunk, ws, w_ref))

    best = min((r for r in rows if len(r) == 4), key=lambda r: r[1])

    if TRY_COMPILE:
        # 最速の設定に対して torch.compile を試す
        name_best = best[0]
        bench = 'benchmark=True' in name_best
        cl = 'channels_last' in name_best
        chunk_txt = name_best.split('chunk=')[1].split()[0]
        chunk = None if chunk_txt == '一括' else int(chunk_txt)
        torch.backends.cudnn.benchmark = bench
        set_format(model, cl)
        x = prepare_x(x32, cl)
        try:
            cm = torch.compile(model)
            add('  ↑最速設定 + torch.compile', measure(model, x, y, chunk, ws, w_ref, fwd=cm))
        except Exception as e:
            add('  ↑最速設定 + torch.compile', str(e).strip().splitlines()[0][:90])

    print("\n" + "=" * 112)
    print(f"{'設定':<62s}{'ms/個体':>9s}{'枚/秒':>11s}{'基準比':>8s}{'正解数差':>9s}{'Loss差':>10s}")
    print("=" * 112)
    for r in rows:
        if len(r) == 4:
            name, ms, correct, loss = r
            print(f"{name:<62s}{ms:9.2f}{n / ms * 1000:11,.0f}{base_ms / ms:7.2f}x"
                  f"{correct - base_correct:9d}{loss - base_loss:10.5f}")
        else:
            print(f"{r[0]:<62s}  失敗: {r[1]}")
    print("=" * 112)
    print(f"最速: {best[0].strip()} → {best[1]:.2f} ms/個体 (基準の {base_ms / best[1]:.2f} 倍)")
    print(f"500個体/世代 の順伝播時間: 基準 {base_ms * 500 / 1000:.1f} s → 最速 {best[1] * 500 / 1000:.1f} s")
    print("正解数差=基準個体で正解した枚数の差(0〜数枚ならfp16の丸め誤差の範囲)", flush=True)


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