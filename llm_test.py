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


def run_strategy(strategy_name, n_games, chunk_size, device, model_path, use_llm, llm_kwargs, summary_csv, summary_file):
    """运行指定策略的 n_games 局游戏。边跑边写两个 CSV：LLM响应 + 汇总指标。"""
    planner = get_planner(strategy_name, device, model_path)

    all_coop_rates = []
    num_chunks = int(np.ceil(n_games / chunk_size))
    game_offset = 0

    # 如果启用 LLM，打开当前策略的 LLM 响应日志 CSV
    resp_file = None
    resp_writer = None
    resp_lock = None
    if use_llm:
        resp_lock = __import__('threading').Lock()
        resp_path = f"llm_{strategy_name}.csv"
        resp_file = open(resp_path, "w", newline="", encoding="utf-8")
        resp_writer = csv.writer(resp_file)
        resp_writer.writerow(["strategy", "game_id", "round", "player_id", "prompt_preview", "raw_reply", "decision"])
        resp_file.flush()
        print(f"  📝 LLM 响应日志: {resp_path}")

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
                resp_writer=resp_writer, resp_file=resp_file, resp_lock=resp_lock,
            )
            env.bots._strategy_name = strategy_name
            env.bots._game_offset = game_offset

        chunk_rows = 0

        with torch.no_grad():
            capital, prev_decisions, edge_features = env.reset()

            for t in range(GameConfig.EPISODE_LENGTH):
                if t > 0:
                    logits = planner.get_logits(capital, prev_decisions, edge_features, t)
                    next_state, _, _, _ = env.step(logits)
                    capital, prev_decisions, edge_features = next_state

                coop_batch = prev_decisions.float().mean(dim=1)
                cap_batch = capital.mean(dim=1)
                gini_batch = batch_gini(capital)

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

                # 写入汇总 CSV（每轮一批）
                for b in range(current_bs):
                    summary_csv.writerow((
                        strategy_name, game_offset + b, t + 1,
                        coop_batch[b].item(), cap_batch[b].item(),
                        hr_ratio[b].item(), gini_batch[b].item(),
                    ))
                    all_coop_rates.append(coop_batch[b].item())
                    chunk_rows += 1
                summary_file.flush()

            game_offset += current_bs

        last_coop = coop_batch.mean().item()
        del env, capital, prev_decisions, edge_features, coop_batch, cap_batch, gini_batch
        torch.cuda.empty_cache()
        gc.collect()

        games_done = min((chunk_idx + 1) * chunk_size, n_games)
        pct = games_done / n_games * 100
        bar_len = 30
        filled = int(bar_len * (chunk_idx + 1) / num_chunks)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  [{bar}] {games_done}/{n_games} ({pct:.0f}%) | last_coop={last_coop:.4f} | +{chunk_rows}行", flush=True)

    if resp_file:
        resp_file.close()
    return all_coop_rates


def main():
    parser = argparse.ArgumentParser(description="四种 Planner 批量模拟并导出 CSV")
    parser.add_argument("--n", type=int, default=1, help="每种策略运行的游戏局数")
    parser.add_argument("--chunk_size", type=int, default=1, help="单次并行局数 (防 OOM)")
    parser.add_argument("--model_path", type=str, default="checkpoints/replicate_0/final_model.pth")
    parser.add_argument("--output", type=str, default="llm_test_results.csv", help="输出 CSV 文件名")
    # LLM 相关
    parser.add_argument("--use_llm", action="store_true", help="启用 LLM Bot 替代模拟 Bot")
    parser.add_argument("--api_key", type=str, default="",
                        help="API Key，默认从环境变量自动检测")
    parser.add_argument("--base_url", type=str, default=None,
                        help="API 地址，默认使用 config.LLMConfig.BASE_URL")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    # --- 智能 API Key 检测 ---
    if args.use_llm and not args.api_key:
        # 根据 base_url 自动选择对应的 env key
        resolved_url = args.base_url or __import__('config').LLMConfig.BASE_URL
        if "deepseek.com" in resolved_url:
            args.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        elif "sjtu" in resolved_url:
            args.api_key = os.environ.get("SJTU_API_KEY", "")
        if not args.api_key:
            args.api_key = os.environ.get("OPENAI_API_KEY", "")

    if args.use_llm:
        if LLMBots is None:
            print("[错误] 缺少依赖，请先 pip install openai")
            return
        if not args.api_key:
            print("[错误] 请设置环境变量或 --api_key")
            print("  交大:  export SJTU_API_KEY=sk-xxx")
            print("  官方:  export DEEPSEEK_API_KEY=sk-xxx")
            print("  通用:  export OPENAI_API_KEY=sk-xxx")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual_url = args.base_url or (__import__('config').LLMConfig.BASE_URL if args.use_llm else "N/A")
    print(f"设备: {device} | 模式: {'V2' if MODE == 1 else 'V1'} | LLM: {'ON' if args.use_llm else 'OFF'}")
    if args.use_llm:
        print(f"API: {actual_url} | 模型: {args.model}")
    print(f"每策略 {args.n} 局, 输出: {args.output}\n")

    llm_kwargs = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
    } if args.use_llm else None

    strategies = ["static", "random", "reactive", "graphnet"]
    final_summary = {}
    total_rows = 0

    # 汇总 CSV
    summary_header = ["strategy", "game_id", "round", "coop_rate", "avg_capital", "high_risk_ratio", "gini"]

    with open(args.output, "w", newline="", encoding="utf-8") as sf:
        summary_csv = csv.writer(sf)
        summary_csv.writerow(summary_header)
        sf.flush()

        for idx, strat in enumerate(strategies):
            print(f"\n{'─'*60}")
            print(f"[{idx+1}/{len(strategies)}] 正在运行 {strat.upper()} 策略 ...")
            print(f"{'─'*60}")
            try:
                coop_list = run_strategy(
                    strategy_name=strat,
                    n_games=args.n,
                    chunk_size=args.chunk_size,
                    device=device,
                    model_path=args.model_path,
                    use_llm=args.use_llm,
                    llm_kwargs=llm_kwargs,
                    summary_csv=summary_csv,
                    summary_file=sf,
                )
                strat_coop = np.mean(coop_list) if coop_list else 0.0
                final_summary[strat] = strat_coop
                total_rows += len(coop_list)
                print(f"  ✅ {strat.upper()} 完成 | 平均合作率: {strat_coop:.4f}")
            except Exception as e:
                print(f"  [跳过] {strat} 出错: {e}")
                final_summary[strat] = None

    # 汇总表
    print(f"\n{'='*60}")
    print(f"{'策略':<20} {'平均合作率':>15}")
    print(f"{'-'*60}")
    for strat in strategies:
        val = final_summary.get(strat)
        if val is not None:
            print(f"{strat.upper():<20} {val:>15.4f}")
        else:
            print(f"{strat.upper():<20} {'ERROR':>15}")
    print(f"{'='*60}")

    print(f"\n完成! 共 {total_rows} 条记录已保存至 {args.output}")


if __name__ == "__main__":
    main()
