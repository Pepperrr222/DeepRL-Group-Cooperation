import pandas as pd
import matplotlib.pyplot as plt

official = pd.read_csv("data/training_data.csv")
ours = pd.read_csv("results/our_training_log.csv")

official_curve = (
    official.groupby("Training_Round")["Mean_Game_Cooperation"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 6))
plt.plot(
    official_curve["Training_Round"],
    official_curve["Mean_Game_Cooperation"],
    label="Official training curve"
)

plt.plot(
    ours["Episode"],
    ours["Mean_Game_Cooperation"],
    marker="o",
    label="Our lightweight run"
)

plt.xlabel("Training Round / Episode")
plt.ylabel("Mean Game Cooperation")
plt.title("Official vs Our Training Curve")
plt.grid(True)
plt.legend()
plt.savefig("figs/training_curve_comparison.png", dpi=300)
plt.show()