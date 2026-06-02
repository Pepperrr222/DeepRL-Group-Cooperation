import torch
import re
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from config import BotConfig, GameConfig, LLMConfig


class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.last_request = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.time()


class LLMBots:
    def __init__(self, batch_size, device, api_key=None, base_url=None, model=None,
                 resp_writer=None, resp_file=None, resp_lock=None):
        self.bs = batch_size
        self.device = device
        self.n_players = GameConfig.N_PLAYERS
        self.model_name = model or LLMConfig.MODEL

        resolved_base_url = base_url or LLMConfig.BASE_URL
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**client_kwargs)

        self._rate_limiter = RateLimiter(LLMConfig.RPM)
        self._first_success_lock = threading.Lock()
        self._first_success_logged = False

        # LLM 响应日志（由外部打开文件句柄控制）
        self._resp_writer = resp_writer
        self._resp_file = resp_file
        self._resp_lock = resp_lock or threading.Lock()
        self._strategy_name = ""   # 由调用方设置
        self._game_offset = 0      # 由调用方设置

        self.theta = torch.normal(
            BotConfig.MU_THETA,
            BotConfig.SIGMA_THETA,
            size=(self.bs, self.n_players),
            device=self.device
        )

    def _call_llm(self, prompt, fallback=None, log_meta=None):
        """
        log_meta: (round_num, game_id, player_id) 可选，提供则每次成功回复立即写入 CSV
        """
        if fallback is None:
            fallback = LLMConfig.FALLBACK
        delay = LLMConfig.RETRY_DELAY
        for attempt in range(1, LLMConfig.MAX_RETRIES + 1):
            try:
                self._rate_limiter.wait()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": (
                            "你正在参加一场重复合作博弈游戏。"
                            "你只能回复一个数字：1（合作）或 0（背叛），不要输出任何其他文字。"
                        )},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=LLMConfig.TEMPERATURE,
                    max_tokens=5
                )
                reply = response.choices[0].message.content.strip()
                match = re.search(r'[01]', reply)
                decision = float(match.group(0)) if match else float(fallback)

                if match:
                    with self._first_success_lock:
                        if not self._first_success_logged:
                            self._first_success_logged = True
                            print(f"[LLM] ✅ 首次调用成功！模型回复: '{reply}' → {match.group(0)}", flush=True)

                # 动态写入响应日志
                if log_meta is not None:
                    rnd, gid, pid = log_meta
                    with self._resp_lock:
                        if self._resp_writer is not None:
                            self._resp_writer.writerow((
                                self._strategy_name,
                                self._game_offset + gid,
                                rnd, pid,
                                prompt[:80].replace("\n", " "),
                                reply[:200].replace("\n", " "),
                                int(decision),
                            ))
                            self._resp_file.flush()

                return decision

            except Exception as e:
                if attempt < LLMConfig.MAX_RETRIES:
                    print(f"[LLM] attempt {attempt} failed: {e}. retry in {delay:.1f}s")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"[LLM] all retries failed: {e}. fallback={fallback}")
                    return float(fallback)

    def _build_prompt(self, round_num, neighbor_actions, neighbor_payoffs):
        if round_num == 0:
            return (
                "你正在参加一场重复合作博弈游戏，这是第 1 轮，尚无历史信息。"
                "请做出你的选择，输出 0（背叛）或 1（合作）。"
            )

        coop_payoffs = [f"{p:.2f}" for a, p in zip(neighbor_actions, neighbor_payoffs) if a == 1]
        defect_payoffs = [f"{p:.2f}" for a, p in zip(neighbor_actions, neighbor_payoffs) if a == 0]
        n_coop = len(coop_payoffs)
        n_defect = len(defect_payoffs)

        coop_str = "{" + ",".join(coop_payoffs) + "}" if coop_payoffs else "{}"
        defect_str = "{" + ",".join(defect_payoffs) + "}" if defect_payoffs else "{}"

        return (
            f"你在参加一场重复合作博弈游戏，"
            f"上一轮，邻居中选择合作的有{n_coop}人，他们的收益分别为{coop_str}；"
            f"选择背叛的有{n_defect}人，他们的收益分别是{defect_str}，"
            f"基于以上信息，请做出你的选择，输出 0（背叛）或 1（合作）。"
        )

    def decide_cooperation(self, round_num, adj_matrix, prev_decisions,
                           current_capital, edge_games, delta=10.0):
        B, N = self.bs, self.n_players

        # Round 0: 与 bots.py 相同的初始合作生成逻辑
        if round_num == 0:
            logits = BotConfig.BETA_PRIME_0 + BotConfig.BETA_PRIME_1 * self.theta
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, 0.0, 1.0)
            return torch.bernoulli(probs)

        # 计算每个玩家的收益
        my_acts = prev_decisions.unsqueeze(2).expand(-1, -1, N).long()
        opp_acts = prev_decisions.unsqueeze(1).expand(-1, N, -1).long()

        payoff_low = torch.zeros(B, N, N, device=self.device)
        payoff_high = torch.zeros(B, N, N, device=self.device)
        for i in [0, 1]:
            for j in [0, 1]:
                mask = (my_acts == i) & (opp_acts == j)
                payoff_low[mask] = GameConfig.LOW_RISK_MATRIX[i][j]
                payoff_high[mask] = GameConfig.HIGH_RISK_MATRIX[i][j]

        actual_payoff = payoff_high * edge_games + payoff_low * (1.0 - edge_games)
        player_payoffs = (actual_payoff * adj_matrix).sum(dim=2)  # (B, N)

        # 构造 (prompt, log_meta) 列表
        tasks = []
        for b in range(B):
            for i in range(N):
                neighbors = adj_matrix[b, i].nonzero(as_tuple=True)[0]
                n_actions = [int(prev_decisions[b, j].item()) for j in neighbors]
                n_payoffs = [player_payoffs[b, j].item() for j in neighbors]
                prompt = self._build_prompt(round_num, n_actions, n_payoffs)
                tasks.append((prompt, (round_num, b, i)))

        with ThreadPoolExecutor(max_workers=LLMConfig.MAX_WORKERS) as executor:
            results = list(executor.map(
                lambda t: self._call_llm(t[0], log_meta=t[1]), tasks
            ))

        decisions = torch.tensor(results, device=self.device).view(B, N)
        return decisions

    def decide_acceptance(self, recommendations, prev_decisions):
        return torch.ones_like(recommendations, dtype=torch.float)
