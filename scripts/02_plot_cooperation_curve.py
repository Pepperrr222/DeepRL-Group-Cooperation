import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("data/validation_cooperation_data.csv")

# 按 condition + round 计算平均合作率
coop_rate = (
    df.groupby(["Condition", "Round"])["Cooperation"]
    .mean()
    .reset_index()
)

# 画图
plt.figure(figsize=(8,6))

for condition in coop_rate["Condition"].unique():
    subset = coop_rate[coop_rate["Condition"] == condition]
    plt.plot(
        subset["Round"],
        subset["Cooperation"],
        marker="o",
        label=condition
    )

plt.xlabel("Round")
plt.ylabel("Cooperation Rate")
plt.title("Cooperation Rate Over Time")
plt.legend()
plt.grid(True)

plt.savefig("figs/cooperation_curve.png", dpi=300)

plt.show()