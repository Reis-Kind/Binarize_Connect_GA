import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

fonts = sorted(set(f.name for f in fm.fontManager.ttflist))

for name in fonts:
    if (
        "Times" in name
        or "Termes" in name
        or "Liberation" in name
        or "STIX" in name
    ):
        print(name)

# ==================================================
# 設定
# ==================================================

# ---------- ファイル ----------
BP_CSV = "./output/ronbun/binaryconnect_cifar_noaug_3conv_nobias_128_seed44_300ep_best_val.csv"

GA_ONLY_CSV = "./output/ronbun/ga_only_cifar_3conv_128_500_0.0001_10000ep.csv"

GA_BC_CSV = "./output/ronbun/ga_cifar_noaug_3conv_nobias_128_BPseed44_best_val_300ep_2000batch_0.0001_1000.csv"


# ---------- 画像 ----------
FIG_WIDTH = 8
FIG_HEIGHT = 5

# PNGの解像度
DPI = 500


# ---------- フォント ----------
FONT_SIZE = 15          # 基本フォントサイズ
LABEL_SIZE = 15         # x軸・y軸ラベル
TICK_SIZE = 15          # 目盛り
LEGEND_SIZE = 15        # 凡例


# ---------- 線 ----------
LINE_WIDTH = 2.0


# ---------- グラフ範囲 ----------
ACCURACY_Y_MAX = 90

# Noneなら自動
BP_ACCURACY_Y_MIN = None
GA_ONLY_ACCURACY_Y_MIN = None
GA_BC_ACCURACY_Y_MIN = 70


# matplotlib全体のフォント
plt.rcParams.update({
    "font.family": "TeX Gyre Termes",
    "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})


# ==================================================
# 保存用
# ==================================================

def save_graph(filename):
    plt.tight_layout()

    # 高解像度PNG
    plt.savefig(
        filename + ".png",
        dpi=DPI,
        bbox_inches="tight"
    )

    # ベクター形式PDF
    plt.savefig(
        filename + ".pdf",
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# BinaryConnect / 誤差逆伝播法
# ==================================================

bp_df = pd.read_csv(BP_CSV)

if "stage" in bp_df.columns:
    bp_df = bp_df[bp_df["stage"] == "epoch"]


# ---------- Accuracy ----------
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plt.plot(
    bp_df["epoch"],
    bp_df["train_accuracy_percent"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Accuracy"
)

plt.plot(
    bp_df["epoch"],
    bp_df["test_accuracy_percent"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.xlim(left=0)

if BP_ACCURACY_Y_MIN is not None:
    plt.ylim(
        bottom=BP_ACCURACY_Y_MIN,
        top=ACCURACY_Y_MAX
    )
else:
    plt.ylim(top=ACCURACY_Y_MAX)

plt.margins(x=0)
plt.grid(True)
plt.legend()

save_graph("binaryconnect_accuracy")


# ---------- Loss ----------
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plt.plot(
    bp_df["epoch"],
    bp_df["train_loss"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Loss"
)

plt.plot(
    bp_df["epoch"],
    bp_df["test_loss"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.legend()

save_graph("binaryconnect_loss")


# ==================================================
# GAのみ
# ==================================================

ga_df = pd.read_csv(GA_ONLY_CSV)

if "stage" in ga_df.columns:
    ga_df = ga_df[ga_df["stage"] == "generation"]


# ---------- Accuracy ----------
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plt.plot(
    ga_df["generation"],
    ga_df["train_accuracy_percent"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Accuracy"
)

plt.plot(
    ga_df["generation"],
    ga_df["test_accuracy_percent"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Accuracy"
)

plt.xlabel("Generation")
plt.ylabel("Accuracy (%)")

plt.xlim(left=0)

if GA_ONLY_ACCURACY_Y_MIN is not None:
    plt.ylim(
        bottom=GA_ONLY_ACCURACY_Y_MIN,
        top=ACCURACY_Y_MAX
    )
else:
    plt.ylim(top=ACCURACY_Y_MAX)

plt.margins(x=0)
plt.grid(True)
plt.legend()

save_graph("ga_only_accuracy")


# ---------- Loss ----------
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plt.plot(
    ga_df["generation"],
    ga_df["train_loss"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Loss"
)

plt.plot(
    ga_df["generation"],
    ga_df["test_loss"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Loss"
)

plt.xlabel("Generation")
plt.ylabel("Loss")

plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.legend()

save_graph("ga_only_loss")


# ==================================================
# 誤差逆伝播法 + 追加GA
# ==================================================

ga_bc_df = pd.read_csv(GA_BC_CSV)

if "stage" in ga_bc_df.columns:
    ga_bc_df = ga_bc_df[
        ga_bc_df["stage"] == "generation"
    ]


# ---------- Accuracy ----------
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plt.plot(
    ga_bc_df["generation"],
    ga_bc_df["sample_accuracy_percent"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Accuracy"
)

plt.plot(
    ga_bc_df["generation"],
    ga_bc_df["test_accuracy_percent"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Accuracy"
)

plt.xlabel("Generation")
plt.ylabel("Accuracy (%)")

plt.xlim(left=0)

if GA_BC_ACCURACY_Y_MIN is not None:
    plt.ylim(
        bottom=GA_BC_ACCURACY_Y_MIN,
        top=ACCURACY_Y_MAX
    )
else:
    plt.ylim(top=ACCURACY_Y_MAX)

plt.margins(x=0)
plt.grid(True)
plt.legend()

save_graph("ga_bc_accuracy")


# ---------- Loss ----------
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plt.plot(
    ga_bc_df["generation"],
    ga_bc_df["sample_loss"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Loss"
)

plt.plot(
    ga_bc_df["generation"],
    ga_bc_df["test_loss"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Loss"
)

plt.xlabel("Generation")
plt.ylabel("Loss")

plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.legend()

save_graph("ga_bc_loss")