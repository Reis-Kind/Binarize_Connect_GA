import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "./output/ga_only_cifar_3conv_128_500_0.0001_10000ep.csv"

df = pd.read_csv(CSV_FILE)

# 世代データだけ取得
ga_df = df[df["stage"] == "generation"].copy()

generations = ga_df["generation"]


plt.figure(figsize=(12, 5))


# =========================
# 正答率
# =========================
plt.subplot(1, 2, 1)

plt.plot(
    generations,
    ga_df["train_accuracy_percent"],
    label="Sampled Train Accuracy",
    color="tab:green"
)

plt.plot(
    generations,
    ga_df["test_accuracy_percent"],
    label="Test Accuracy",
    color="tab:blue"
)

plt.xlabel("Generation")
plt.ylabel("Accuracy (%)")

# 縦軸を0〜80%に固定
plt.ylim(0, 90)

plt.grid(True)
plt.legend()


# =========================
# Loss
# =========================
plt.subplot(1, 2, 2)

plt.plot(
    generations,
    ga_df["train_loss"],
    label="Sampled Train Loss",
    color="tab:red"
)

plt.plot(
    generations,
    ga_df["test_loss"],
    label="Test Loss",
    color="tab:orange"
)

plt.xlabel("Generation")
plt.ylabel("Loss")

plt.grid(True)
plt.legend()


plt.tight_layout()

plt.savefig("./output/ga_only_ronbun.png", dpi=300)

plt.show()