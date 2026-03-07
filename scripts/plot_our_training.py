import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/our_training_log.csv")

plt.figure(figsize=(8, 6))
plt.plot(df["Episode"], df["Mean_Game_Cooperation"], marker="o")
plt.xlabel("Episode")
plt.ylabel("Mean Game Cooperation")
plt.title("Our GraphNet+A2C Training Curve")
plt.grid(True)
plt.savefig("figs/our_training_curve.png", dpi=300)
plt.show()