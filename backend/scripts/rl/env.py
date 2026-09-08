"""把K线做成一局游戏 —— 每根K线做一次决策: 空仓 / 半仓 / 满仓。

与前面所有监督学习版本的根本区别:
    监督学习   模型只答"这只票好不好", 什么时候卖、买多少是我写死的
               (+15%止盈 / -8%止损 / 20天到期 / 等权满仓)
    这里       出场和仓位都由策略自己学, 我不再拍规则

⚠️ 必须堵死的陷阱: 如果奖励只是"别亏", 最优策略就是永远空仓。
   解法与监督版一致 —— 奖励用【超额收益】(减去当日全市场等权均值):
       空仓        = 0 分
       牛市满仓    = 0 分附近(大家都涨, 超额为零)
       选对票      才有分
   这一招同时堵死"牛市就买"这条捷径, 与去 regime 记忆是同一个机制。

⚠️ 状态里绝不能出现: 绝对价格、日期、股票代码、市值、指数水平。
   只用已经做过当日横截面分位的特征 + 自己的持仓状态。

⚠️ 成本必须进奖励, 否则策略会学成每天来回换手。
"""
from __future__ import annotations

import numpy as np

# A股实际成本: 佣金双边万2.5 + 印花税卖出万5 + 过户费双边万0.1 + 单边滑点千1
COST_BUY = 0.00025 + 0.00001 + 0.001
COST_SELL = 0.00025 + 0.0005 + 0.00001 + 0.001

POSITIONS = np.array([0.0, 0.5, 1.0], dtype=np.float32)   # 空仓 / 半仓 / 满仓


class Episode:
    """一只股票的一段行情 = 一局游戏。

    feats: (T, F) 已做当日横截面分位的特征
    exret: (T,)   当日超额收益(个股收益 - 当日全市场等权均值)
    """

    __slots__ = ("feats", "exret", "T", "t", "pos", "equity", "peak", "n_feat")

    def __init__(self, feats: np.ndarray, exret: np.ndarray):
        self.feats = feats
        self.exret = exret
        self.T = len(exret)
        self.n_feat = feats.shape[1]
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.pos = 0.0        # 当前仓位 0/0.5/1.0
        self.equity = 1.0     # 超额口径的净值
        self.peak = 1.0
        return self.obs()

    def obs(self) -> np.ndarray:
        """状态 = 市场特征 + 自己的持仓状态。

        持仓状态必须给 —— 不给的话策略无法知道"我现在拿着没有",
        也就学不出"什么时候卖"。三个量都是无量纲的。
        """
        return np.concatenate([
            self.feats[self.t],
            [self.pos,                                   # 当前仓位
             np.clip(self.equity - 1.0, -0.5, 0.5),      # 本局累计超额
             np.clip(self.equity / self.peak - 1.0, -0.5, 0.0)],  # 当前水下深度
        ]).astype(np.float32)

    def step(self, action: int, dd_penalty: float) -> tuple[np.ndarray, float, bool]:
        new_pos = float(POSITIONS[action])
        # 换手成本: 加仓按买入费, 减仓按卖出费
        d = new_pos - self.pos
        cost = d * COST_BUY if d > 0 else (-d) * COST_SELL
        self.pos = new_pos

        r = float(self.exret[self.t])          # 当日超额
        gain = self.pos * r - cost             # 本步净值增量(超额口径)
        before = self.equity
        self.equity *= (1.0 + gain)
        self.peak = max(self.peak, self.equity)

        # 奖励 = 净值增量 − λ×新增回撤。
        # ⚠️ 只罚【新增】的水下深度, 不罚存量 —— 罚存量会让策略在深套时
        #    每一步都被罚, 从而学会一亏就割, 而不是学会别亏。
        dd_now = max(0.0, 1.0 - self.equity / self.peak)
        dd_prev = max(0.0, 1.0 - before / max(self.peak, before))
        reward = (self.equity - before) - dd_penalty * max(0.0, dd_now - dd_prev)

        self.t += 1
        done = self.t >= self.T
        return (self.obs() if not done else np.zeros(self.n_feat + 3, np.float32),
                float(reward), done)
