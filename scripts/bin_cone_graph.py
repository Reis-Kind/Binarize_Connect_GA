import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "./output/ronbun/binaryconnect_cifar_noaug_3conv_nobias_128_seed44_300ep_best_val.csv"

df = pd.read_csv(CSV_FILE)

# 世代データだけ取得
ga_df = df[df["stage"] == "epoch"].copy()

epochs = ga_df["epoch"]


plt.figure(figsize=(12, 5))


# =========================
# 正答率
# =========================
plt.subplot(1, 2, 1)

plt.plot(
    epochs,
    ga_df["train_accuracy_percent"],
    label="Train Accuracy",
    color="black"
)

plt.plot(
    epochs,
    ga_df["test_accuracy_percent"],
    label="Test Accuracy",
    color="tab:red"
)

plt.xlabel("epoch")
plt.ylabel("Accuracy (%)")

# 縦軸を0〜80%に固定
plt.ylim(top=90)
plt.xlim(left=0)
plt.margins(x=0)


plt.grid(True)
plt.legend()


# =========================
# Loss
# =========================
plt.subplot(1, 2, 2)

plt.plot(
    epochs,
    ga_df["train_loss"],
    label="Train Loss",
    color="black"
)

plt.plot(
    epochs,
    ga_df["test_loss"],
    label="Test Loss",
    color="tab:red"
)

plt.xlabel("epoch")
plt.ylabel("Loss")

plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.legend()


plt.tight_layout()

plt.savefig("./output/bin_cone_ronbun.png", dpi=300)

plt.show()