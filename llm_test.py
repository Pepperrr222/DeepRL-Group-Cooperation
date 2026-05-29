# llm_test.py
import torch
import numpy as np
import argparse
import os
import gc
import csv
from env.game import PublicGoodsGame
from config import GameConfig, MODE
from planners.baselines import StaticPlanner, RandomPlanner, ReactivePlanner
from planners.graphnet import GraphNetPlanner

try:
    from env.llm_bots import LLMBots
except ImportError:
    LLMBots = None


def batch_gini(capital_matrix):
    """批量计算 Gini 系数，capital_matrix shape: (B, N)"""
    cap_clamped = torch.clamp(capital_matrix, min=0.0)
    sorted_cap, _ = torch.sort(cap_clamped, dim=1)
    B, N = sorted_cap.shape
    index = torch.arange(1, N + 1, device=capital_matrix.device).float()
    numerator = torch.sum((2 * index - N - 1) * sorted_cap, dim=1)
    denominator = N * torch.sum(sorted_cap, dim=1)
    gini_per_game = torch.where(denominator == 0, torch.zeros_like(numerator), numerator / denominator)
    return gini_per_game


def get_planner(strategy_name, device, model_path=None):
    if strategy_name == "static":
        return StaticPlanner()
    elif strategy_name == "random":
        return RandomPlanner()
    elif strategy_name == "reactive":
        return ReactivePlanner()
    elif strategy_name == "graphnet":
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"[错误] 未找到模型文件: {model_path}")
        return GraphNetPlanner(model_path=model_path, device=device)
    else:
        raise ValueError(f"未知策略: {strategy_name}")


def run_strategy(strategy_name, n_games, chunk_size, device, model_path, use_llm, llm_kwargs):
    """运行指定策略的 n_games 局游戏，返回每局每轮的指标列表。"""
    planner = get_planner(strategy_name, device, model_path)

    # 每条记录: (strategy, game_id, round, coop_rate, avg_capital, high_risk_ratio, gini)
    records = []
    num_chunks = int(np.ceil(n_games / chunk_size))
    game_offset = 0

    for chunk_idx in range(num_chunks):
        current_bs = min(chunk_size, n_games - game_offset)
        env = PublicGoodsGame(batch_size=current_bs, device=device)

        if use_llm:
            if LLMBots is None:
                raise ImportError("缺少 env.llm_bots 或 openai 库")
            env.bots = LLMBots(
                batch_size=current_bs, device=device,
                api_key=llm_kwargs["api_key"],
                base_url=llm_kwargs.get("base_url"),
                model=llm_kwargs.get("model"),
            )

        with torch.no_grad():
            capital, prev_decisions, edge_features = env.reset()

            for t in range(GameConfig.EPISODE_LENGTH):
                if t > 0:
                    logits = planner.get_logits(capital, prev_decisions, edge_features, t)
                    next_state, _, _, _ = env.step(logits)
                    capital, prev_decisions, edge_features = next_state

                # 逐局记录
                coop_batch = prev_decisions.float().mean(dim=1)          # (B,)
                cap_batch = capital.mean(dim=1)                          # (B,)
                gini_batch = batch_gini(capital)                         # (B,)

                if MODE == 1:
                    adj = edge_features[..., 0]
                    game_modes = edge_features[..., 1]
                    active_edges = adj.sum(dim=(1, 2)) / 2
                    high_risk_edges = (game_modes * adj).sum(dim=(1, 2)) / 2
                    hr_ratio = torch.where(
                        active_edges > 0, high_risk_edges / active_edges, torch.zeros_like(active_edges)
                    )
                else:
                    hr_ratio = torch.zeros(current_bs, device=device)

                for b in range(current_bs):
                    records.append((
                        strategy_name,
                        game_offset + b,
                        t + 1,
                        coop_batch[b].item(),
                        cap_batch[b].item(),
                        hr_ratio[b].item(),
                        gini_batch[b].item(),
                    ))

            game_offset += current_bs

        del env, capital, prev_decisions, edge_features
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  [{strategy_name}] chunk {chunk_idx + 1}/{num_chunks} done ({current_bs} games)")

    return records


def main():
    parser = argparse.ArgumentParser(description="四种 Planner 批量模拟并导出 CSV")
    parser.add_argument("--n", type=int, default=500, help="每种策略运行的游戏局数")
    parser.add_argument("--chunk_size", type=int, default=500, help="单次并行局数 (防 OOM)")
    parser.add_argument("--model_path", type=str, default="checkpoints/replicate_0/final_model.pth")
    parser.add_argument("--output", type=str, default="llm_test_results.csv", help="输出 CSV 文件名")
    # LLM 相关
    parser.add_argument("--use_llm", action="store_true", help="启用 LLM Bot 替代模拟 Bot")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    if args.use_llm:
        if LLMBots is None:
            print("[错误] 缺少依赖，请先 pip install openai")
            return
        if not args.api_key:
            print("[错误] --use_llm 需要 --api_key 或设置 OPENAI_API_KEY")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 模式: {'V2' if MODE == 1 else 'V1'} | LLM: {'ON' if args.use_llm else 'OFF'}")
    print(f"每策略 {args.n} 局, 输出: {args.output}\n")

    llm_kwargs = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
    } if args.use_llm else None

    strategies = ["static", "random", "reactive", "graphnet"]
    all_records = []

    for strat in strategies:
        print(f"▶ 运行策略: {strat.upper()}")
        try:
            records = run_strategy(
                strategy_name=strat,
                n_games=args.n,
                chunk_size=args.chunk_size,
                device=device,
                model_path=args.model_path,
                use_llm=args.use_llm,
                llm_kwargs=llm_kwargs,
            )
            all_records.extend(records)
        except Exception as e:
            print(f"  [跳过] {strat} 出错: {e}")

    # 写入 CSV
    header = ["strategy", "game_id", "round", "coop_rate", "avg_capital", "high_risk_ratio", "gini"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_records)

    print(f"\n完成! 共 {len(all_records)} 条记录已保存至 {args.output}")


if __name__ == "__main__":
    main()
