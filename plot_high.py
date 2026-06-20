"""
读取 0llm/high/ 下所有 llm_test_results.csv，
取平均后画出 coop_rate、avg_capital、high_risk_ratio、gini 关于回合的变化图。
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).parent / "0llm" / "high"
OUT_DIR = Path(__file__).parent / "0llm" / "plots_high"
OUT_DIR.mkdir(exist_ok=True)

METRICS = {
    "coop_rate":        "Cooperation Rate",
    "avg_capital":      "Average Capital",
    "high_risk_ratio":  "High Ratio",
    "gini":             "Gini Coefficient",
}

STRATEGIES = ["static", "statichigh", "random", "reactive", "graphnet"]
STRATEGY_COLORS = {
    "static":     "#1f77b4",
    "statichigh": "#9467bd",
    "random":     "#ff7f0e",
    "reactive":   "#2ca02c",
    "graphnet":   "#d62728",
}

# ── 收集数据 ──────────────────────────────────────────────
def load_all() -> list[pd.DataFrame]:
    dfs = []
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir():
            continue
        csv_path = folder / "llm_test_results.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
    return dfs

# ── 按策略+回合取平均 ────────────────────────────────────
def avg_by_round(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(dfs, ignore_index=True)
    return merged.groupby(["strategy", "round"]).mean(numeric_only=True).reset_index()

# ── 画图 ────────────────────────────────────────────────
def plot_all(avg: pd.DataFrame, n_files: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    title = f"deepseek-chat (0llm/high/) — {n_files} runs averaged"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, (metric, label) in zip(axes.flat, METRICS.items()):
        for strat in STRATEGIES:
            sub = avg[avg["strategy"] == strat]
            if sub.empty:
                continue
            ax.plot(sub["round"], sub[metric],
                    label=strat, color=STRATEGY_COLORS[strat],
                    marker="o", markersize=4, linewidth=1.5)
        ax.set_xlabel("Round")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / "deepseek_chat.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path}")

# ── 主程序 ────────────────────────────────────────────────
if __name__ == "__main__":
    print("正在读取数据...")
    dfs = load_all()
    print(f"  共 {len(dfs)} 次运行")

    if not dfs:
        print("  没有数据文件，退出。")
        exit()

    avg = avg_by_round(dfs)
    print("正在绘图...")
    plot_all(avg, len(dfs))
    print(f"\n全部完成！图片保存在: {OUT_DIR}")
