import matplotlib.pyplot as plt

rounds = [1, 2, 3]
losses = [0.10, 0.10, 0.10]

plt.figure(figsize=(6,4))
plt.plot(rounds, losses, marker="o")

plt.title("Federated Learning Loss")
plt.xlabel("Round")
plt.ylabel("Loss")

plt.grid(True)

plt.savefig("loss_curve.png")

plt.show()