import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "./output/ronbun/ga_cifar_noaug_3conv_nobias_128_BPseed44_best_val_300ep_2000batch_0.0001_1000.csv"

df = pd.read_csv(CSV_FILE)

# 世代データだけ取得
ga_df = df[df["stage"] == "generation"].copy()

generations = ga_df["generation"]


# =========================
# Accuracy
# =========================
plt.figure(figsize=(6, 4))

plt.plot(
    generations,
    ga_df["sample_accuracy_percent"],
    color="black"
)

plt.plot(
    generations,
    ga_df["test_accuracy_percent"],
    color="tab:red"
)

plt.xlabel("Generation")
plt.ylabel("Accuracy (%)")

# 横軸を0から開始
plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "./output/ga_bc_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "./output/ga_bc_accuracy.pdf",
    bbox_inches="tight"
)

plt.close()


# =========================
# Loss
# =========================
plt.figure(figsize=(6, 4))

plt.plot(
    generations,
    ga_df["sample_loss"],
    color="black"
)

plt.plot(
    generations,
    ga_df["test_loss"],
    color="tab:red"
)

plt.xlabel("Generation")
plt.ylabel("Loss")

# 横軸を0から開始
plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "./output/ga_bc_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "./output/ga_bc_loss.pdf",
    bbox_inches="tight"
)

plt.close()