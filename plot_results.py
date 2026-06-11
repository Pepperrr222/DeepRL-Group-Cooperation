"""
读取 0llm/static/ 下所有 llm_test_results.csv，
按类别（纯数字/glm/qwen/minimax）分组取平均，
画出 coop_rate、avg_capital、high_risk_ratio、gini 关于回合的变化图。
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).parent / "0llm" / "static"
OUT_DIR = Path(__file__).parent / "0llm" / "plots"
OUT_DIR.mkdir(exist_ok=True)

METRICS = {
    "coop_rate":        "Cooperation Rate",
    "avg_capital":      "Average Capital",
    "high_risk_ratio":  "High Risk Ratio",
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

# ── 分类 ──────────────────────────────────────────────────
def classify(folder_name: str) -> str | None:
    """返回文件夹所属类别，无法识别则返回 None"""
    if re.fullmatch(r'\d+', folder_name):
        return "plain"          # 纯数字
    if folder_name.endswith("glm"):
        return "glm"
    if folder_name.endswith("qwen"):
        return "qwen"
    if folder_name.endswith("minimax"):
        return "minimax"
    return None

CATEGORY_LABELS = {
    "plain":   "Plain (numeric)",
    "glm":     "GLM",
    "qwen":    "Qwen",
    "minimax": "MiniMax",
}

# ── 收集数据 ──────────────────────────────────────────────
def load_all() -> dict[str, list[pd.DataFrame]]:
    """按类别收集所有 CSV 的 DataFrame"""
    groups: dict[str, list[pd.DataFrame]] = {}
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir():
            continue
        cat = classify(folder.name)
        if cat is None:
            continue
        csv_path = folder / "llm_test_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        groups.setdefault(cat, []).append(df)
    return groups

# ── 按策略+回合取平均 ────────────────────────────────────
def avg_by_round(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """合并多个 DataFrame，按 strategy+round 取平均"""
    merged = pd.concat(dfs, ignore_index=True)
    avg = merged.groupby(["strategy", "round"]).mean(numeric_only=True).reset_index()
    return avg

# ── 画单类别图 ────────────────────────────────────────────
def plot_category(cat: str, avg: pd.DataFrame):
    """为一个类别画 4 个子图（每个指标一张）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Category: {CATEGORY_LABELS[cat]}  (n={len(groups[cat])} files averaged)",
                 fontsize=14, fontweight="bold")

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
    path = OUT_DIR / f"{cat}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path}")

# ── 画总图 ────────────────────────────────────────────────
def plot_combined(all_avgs: dict[str, pd.DataFrame]):
    """所有类别画在同一张图上，按指标×类别分色，按策略分线型"""
    LINESTYLE = {"static": "-", "statichigh": (0, (5, 1)), "random": "--", "reactive": "-.", "graphnet": ":"}
    CAT_COLORS = {
        "plain": "#1f77b4",
        "glm":   "#ff7f0e",
        "qwen":  "#2ca02c",
        "minimax":"#d62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("All Categories Comparison", fontsize=15, fontweight="bold")

    for ax, (metric, label) in zip(axes.flat, METRICS.items()):
        for cat, avg in all_avgs.items():
            for strat in STRATEGIES:
                sub = avg[avg["strategy"] == strat]
                if sub.empty:
                    continue
                ax.plot(sub["round"], sub[metric],
                        label=f"{CATEGORY_LABELS[cat]}-{strat}",
                        color=CAT_COLORS[cat],
                        linestyle=LINESTYLE[strat],
                        marker="o", markersize=3, linewidth=1.2)
        ax.set_xlabel("Round")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT_DIR / "combined.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path}")

# ── 主程序 ────────────────────────────────────────────────
if __name__ == "__main__":
    print("正在读取数据...")
    groups = load_all()
    for cat, dfs in groups.items():
        print(f"  {CATEGORY_LABELS[cat]:>15s}: {len(dfs)} 个文件")

    all_avgs: dict[str, pd.DataFrame] = {}
    for cat, dfs in groups.items():
        avg = avg_by_round(dfs)
        all_avgs[cat] = avg
        print(f"\n正在绘制 {CATEGORY_LABELS[cat]} ...")
        plot_category(cat, avg)

    print("\n正在绘制总图...")
    plot_combined(all_avgs)
    print("\n全部完成！图片保存在:", OUT_DIR)
