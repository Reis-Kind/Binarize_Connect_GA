import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "./output/ronbun/ga_cifar_noaug_3conv_nobias_128_BPseed44_best_val_300ep_2000batch_0.0001_1000.csv"

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
    ga_df["sample_accuracy_percent"],
    label="Train Accuracy",
    color="black"
)

plt.plot(
    generations,
    ga_df["test_accuracy_percent"],
    label="Test Accuracy",
    color="tab:red"
)

plt.xlabel("Generation")
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
    generations,
    ga_df["sample_loss"],
    label="Train Loss",
    color="black"
)

plt.plot(
    generations,
    ga_df["test_loss"],
    label="Test Loss",
    color="tab:red"
)

plt.xlabel("Generation")
plt.ylabel("Loss")

plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.legend()


plt.tight_layout()

plt.savefig("./output/ga_bc_ronbun.png", dpi=300)

plt.show()