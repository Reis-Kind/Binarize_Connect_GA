import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "./output/ronbun/binaryconnect_cifar_noaug_3conv_nobias_128_seed44_300ep_best_val.csv"

df = pd.read_csv(CSV_FILE)

# epochデータだけ取得
bp_df = df[df["stage"] == "epoch"].copy()

epochs = bp_df["epoch"]


# =========================
# Accuracy
# =========================
plt.figure(figsize=(6, 4))

plt.plot(
    epochs,
    bp_df["train_accuracy_percent"],
    label="Train Accuracy",
    color="black"
)

plt.plot(
    epochs,
    bp_df["test_accuracy_percent"],
    label="Test Accuracy",
    color="tab:red"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.ylim(top=90)
plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "./output/binaryconnect_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "./output/binaryconnect_accuracy.pdf",
    bbox_inches="tight"
)

plt.close()


# =========================
# Loss
# =========================
plt.figure(figsize=(6, 4))

plt.plot(
    epochs,
    bp_df["train_loss"],
    label="Train Loss",
    color="black"
)

plt.plot(
    epochs,
    bp_df["test_loss"],
    label="Test Loss",
    color="tab:red"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "./output/binaryconnect_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "./output/binaryconnect_loss.pdf",
    bbox_inches="tight"
)

plt.close()