import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# ファイル
# ==================================================

BP_CSV = (
    "./output/ronbun/"
    "binaryconnect_cifar_noaug_3conv_nobias_128_"
    "seed44_300ep_best_val.csv"
)

GA_ONLY_CSV = (
    "./output/ronbun/"
    "ga_only_cifar_3conv_128_500_0.0001_10000ep.csv"
)

GA_BC_CSV = (
    "./output/ronbun/"
    "ga_cifar_noaug_3conv_nobias_128_"
    "BPseed44_best_val_300ep_2000batch_0.0001_1000.csv"
)


# ==================================================
# 見た目
# ==================================================

# ---------- 図サイズ ----------
FIG_WIDTH = 6
FIG_HEIGHT = 4


# ---------- 解像度 ----------
DPI = 600


# ---------- フォント ----------
FONT_NAME = "STIXGeneral"

FONT_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 17


# ---------- 線 ----------
LINE_WIDTH = 2.2


# ---------- グリッド ----------
GRID_LINEWIDTH = 0.8
GRID_ALPHA = 0.45
GRID_LINESTYLE = "--"


# ---------- 軸 ----------
AXIS_LINEWIDTH = 1.0

TICK_WIDTH = 1.0
TICK_LENGTH = 5

LABEL_PAD = 8
TICK_PAD = 5


# ==================================================
# Accuracy の軸
# ==================================================

ACCURACY_Y_MAX = 90


# ---------- BP ----------
BP_ACCURACY_Y_MIN = 0
BP_ACCURACY_Y_TICK = 10


# ---------- GA only ----------
GA_ONLY_ACCURACY_Y_MIN = 0
GA_ONLY_ACCURACY_Y_TICK = 10


# ---------- Additional GA ----------
# 追加GAだけ細かい変化を見るためズーム
GA_BC_ACCURACY_Y_MIN = 70
GA_BC_ACCURACY_Y_TICK = 5


# ==================================================
# Loss の軸
# ==================================================

# ---------- BP ----------
BP_LOSS_Y_MIN = 0
BP_LOSS_Y_MAX = 2.5
BP_LOSS_Y_TICK = 0.5


# ---------- GA only ----------
# GA-onlyはy軸を自動調整
GA_ONLY_LOSS_Y_MIN = None
GA_ONLY_LOSS_Y_MAX = None
GA_ONLY_LOSS_Y_TICK = None


# ---------- Additional GA ----------
GA_BC_LOSS_Y_MIN = 0
GA_BC_LOSS_Y_MAX = 2.5
GA_BC_LOSS_Y_TICK = 0.5


# ==================================================
# x軸
# ==================================================

# ---------- BP ----------
BP_X_MIN = 0
BP_X_MAX = 300
BP_X_TICK = 50


# ---------- GA only ----------
GA_ONLY_X_MIN = 0
GA_ONLY_X_MAX = 10000
GA_ONLY_X_TICK = 2000


# ---------- Additional GA ----------
GA_BC_X_MIN = 0
GA_BC_X_MAX = 1000
GA_BC_X_TICK = 200


# ==================================================
# matplotlib 共通設定
# ==================================================

plt.rcParams.update({
    "font.family": FONT_NAME,

    "font.size": FONT_SIZE,

    "axes.labelsize": LABEL_SIZE,

    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,

    "legend.fontsize": LEGEND_SIZE,

    "axes.linewidth": AXIS_LINEWIDTH,

    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,

    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,

    "xtick.major.pad": TICK_PAD,
    "ytick.major.pad": TICK_PAD,

    "axes.labelpad": LABEL_PAD,

    # 数式を使った場合もSTIX系に揃える
    "mathtext.fontset": "stix",
})


# ==================================================
# 共通関数
# ==================================================

def make_figure():
    """
    固定サイズの図を作成する．
    タイトルは付けない．
    """

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        ),
        constrained_layout=True
    )

    return fig, ax


def set_common_style(ax):
    """
    グリッドと凡例の共通設定．
    """

    ax.grid(
        True,
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA
    )

    ax.legend(
        loc="best",
        frameon=True
    )


def set_accuracy_axis(
    ax,
    x_min,
    x_max,
    x_tick,
    y_min,
    y_tick
):
    """
    Accuracyグラフの軸設定．
    """

    # x軸
    ax.set_xlim(
        x_min,
        x_max
    )

    ax.set_xticks(
        np.arange(
            x_min,
            x_max + x_tick,
            x_tick
        )
    )

    # y軸
    ax.set_ylim(
        y_min,
        ACCURACY_Y_MAX
    )

    ax.set_yticks(
        np.arange(
            y_min,
            ACCURACY_Y_MAX + y_tick,
            y_tick
        )
    )


def set_loss_axis(
    ax,
    x_min,
    x_max,
    x_tick,
    y_min=None,
    y_max=None,
    y_tick=None
):
    """
    Lossグラフの軸設定．
    y軸をNoneにすると自動調整．
    """

    # x軸
    ax.set_xlim(
        x_min,
        x_max
    )

    ax.set_xticks(
        np.arange(
            x_min,
            x_max + x_tick,
            x_tick
        )
    )

    # y軸
    if (
        y_min is not None
        and y_max is not None
    ):
        ax.set_ylim(
            y_min,
            y_max
        )

    if (
        y_min is not None
        and y_max is not None
        and y_tick is not None
    ):
        ax.set_yticks(
            np.arange(
                y_min,
                y_max + y_tick,
                y_tick
            )
        )


def save_graph(
    fig,
    filename
):
    """
    PNGとPDFを保存する．
    """

    # PNG
    fig.savefig(
        filename + ".png",
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.05
    )

    # PDF
    fig.savefig(
        filename + ".pdf",
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.close(fig)


# ==================================================
# BinaryConnect / 誤差逆伝播法
# ==================================================

bp_df = pd.read_csv(
    BP_CSV
)

if "stage" in bp_df.columns:
    bp_df = bp_df[
        bp_df["stage"] == "epoch"
    ]


# --------------------------------------------------
# BP Accuracy
# --------------------------------------------------

fig, ax = make_figure()

ax.plot(
    bp_df["epoch"],
    bp_df["train_accuracy_percent"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Accuracy"
)

ax.plot(
    bp_df["epoch"],
    bp_df["test_accuracy_percent"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Accuracy"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Accuracy (%)"
)

set_accuracy_axis(
    ax,
    BP_X_MIN,
    BP_X_MAX,
    BP_X_TICK,
    BP_ACCURACY_Y_MIN,
    BP_ACCURACY_Y_TICK
)

set_common_style(
    ax
)

save_graph(
    fig,
    "binaryconnect_accuracy"
)


# --------------------------------------------------
# BP Loss
# --------------------------------------------------

fig, ax = make_figure()

ax.plot(
    bp_df["epoch"],
    bp_df["train_loss"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Loss"
)

ax.plot(
    bp_df["epoch"],
    bp_df["test_loss"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Loss"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Loss"
)

set_loss_axis(
    ax,
    BP_X_MIN,
    BP_X_MAX,
    BP_X_TICK,
    BP_LOSS_Y_MIN,
    BP_LOSS_Y_MAX,
    BP_LOSS_Y_TICK
)

set_common_style(
    ax
)

save_graph(
    fig,
    "binaryconnect_loss"
)


# ==================================================
# GA only
# ==================================================

ga_df = pd.read_csv(
    GA_ONLY_CSV
)

if "stage" in ga_df.columns:
    ga_df = ga_df[
        ga_df["stage"] == "generation"
    ]


# --------------------------------------------------
# GA only Accuracy
# --------------------------------------------------

fig, ax = make_figure()

ax.plot(
    ga_df["generation"],
    ga_df["train_accuracy_percent"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Accuracy"
)

ax.plot(
    ga_df["generation"],
    ga_df["test_accuracy_percent"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Accuracy"
)

ax.set_xlabel(
    "Generation"
)

ax.set_ylabel(
    "Accuracy (%)"
)

set_accuracy_axis(
    ax,
    GA_ONLY_X_MIN,
    GA_ONLY_X_MAX,
    GA_ONLY_X_TICK,
    GA_ONLY_ACCURACY_Y_MIN,
    GA_ONLY_ACCURACY_Y_TICK
)

set_common_style(
    ax
)

save_graph(
    fig,
    "ga_only_accuracy"
)


# --------------------------------------------------
# GA only Loss
# --------------------------------------------------

fig, ax = make_figure()

ax.plot(
    ga_df["generation"],
    ga_df["train_loss"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Loss"
)

ax.plot(
    ga_df["generation"],
    ga_df["test_loss"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Loss"
)

ax.set_xlabel(
    "Generation"
)

ax.set_ylabel(
    "Loss"
)

# GA-onlyだけy軸は自動調整
set_loss_axis(
    ax,
    GA_ONLY_X_MIN,
    GA_ONLY_X_MAX,
    GA_ONLY_X_TICK,
    GA_ONLY_LOSS_Y_MIN,
    GA_ONLY_LOSS_Y_MAX,
    GA_ONLY_LOSS_Y_TICK
)

set_common_style(
    ax
)

save_graph(
    fig,
    "ga_only_loss"
)


# ==================================================
# 誤差逆伝播法 + Additional GA
# ==================================================

ga_bc_df = pd.read_csv(
    GA_BC_CSV
)

if "stage" in ga_bc_df.columns:
    ga_bc_df = ga_bc_df[
        ga_bc_df["stage"] == "generation"
    ]


# --------------------------------------------------
# Additional GA Accuracy
# --------------------------------------------------

fig, ax = make_figure()

ax.plot(
    ga_bc_df["generation"],
    ga_bc_df["sample_accuracy_percent"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Accuracy"
)

ax.plot(
    ga_bc_df["generation"],
    ga_bc_df["test_accuracy_percent"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Accuracy"
)

ax.set_xlabel(
    "Generation"
)

ax.set_ylabel(
    "Accuracy (%)"
)

set_accuracy_axis(
    ax,
    GA_BC_X_MIN,
    GA_BC_X_MAX,
    GA_BC_X_TICK,
    GA_BC_ACCURACY_Y_MIN,
    GA_BC_ACCURACY_Y_TICK
)

set_common_style(
    ax
)

save_graph(
    fig,
    "ga_bc_accuracy"
)


# --------------------------------------------------
# Additional GA Loss
# --------------------------------------------------

fig, ax = make_figure()

ax.plot(
    ga_bc_df["generation"],
    ga_bc_df["sample_loss"],
    color="black",
    linewidth=LINE_WIDTH,
    label="Train Loss"
)

ax.plot(
    ga_bc_df["generation"],
    ga_bc_df["test_loss"],
    color="tab:red",
    linewidth=LINE_WIDTH,
    label="Test Loss"
)

ax.set_xlabel(
    "Generation"
)

ax.set_ylabel(
    "Loss"
)

set_loss_axis(
    ax,
    GA_BC_X_MIN,
    GA_BC_X_MAX,
    GA_BC_X_TICK,
    GA_BC_LOSS_Y_MIN,
    GA_BC_LOSS_Y_MAX,
    GA_BC_LOSS_Y_TICK
)

set_common_style(
    ax
)

save_graph(
    fig,
    "ga_bc_loss"
)


print(
    "グラフの保存が完了しました．"
)