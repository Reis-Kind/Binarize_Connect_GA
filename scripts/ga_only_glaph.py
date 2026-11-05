import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "./output/ronbun/ga_only_cifar_3conv_128_500_0.0001_10000ep.csv"

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
    ga_df["train_accuracy_percent"],
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

plt.ylim(top=90)
plt.xlim(left=0)
plt.margins(x=0)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "./output/ga_only_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "./output/ga_only_accuracy.pdf",
    bbox_inches="tight"
)

plt.close()


# =========================
# Loss
# =========================
plt.figure(figsize=(6, 4))

plt.plot(
    generations,
    ga_df["train_loss"],
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

plt.savefig(
    "./output/ga_only_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "./output/ga_only_loss.pdf",
    bbox_inches="tight"
)

plt.close()