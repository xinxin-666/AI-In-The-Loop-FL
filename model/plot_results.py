import matplotlib.pyplot as plt

metrics = {
    "Accuracy": 95.83,
    "Precision": 92.31,
    "Recall": 100.00,
    "F1-score": 96.00
}

names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(8, 5))
plt.bar(names, values)

plt.ylim(0, 105)

plt.title("FedGuard Model Evaluation")
plt.ylabel("Percentage (%)")

for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.2f}%", ha="center")

plt.tight_layout()

plt.savefig("results/evaluation_results.png")

print("Chart Saved: results/evaluation_results.png")

plt.show()