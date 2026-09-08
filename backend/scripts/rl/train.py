"""PPO —— 让策略自己学 观望/半仓/满仓, 出场和仓位都不再由我拍。

训练 2016-2020, 测试 2021-2026 全样本外(与监督版同一划分, 可比)。

⚠️ 三条纪律与监督版一致:
   1. 特征已做当日横截面分位 -> 状态里没有 regime 信息
   2. 奖励用【超额】收益 -> 空仓 0 分、牛市满仓也≈0 分, 只有选对票才有分
   3. 成本进奖励 -> 否则学成每天换手

⚠️ 评估必须报【平均仓位】。学成几乎不交易不是稳健, 是奖励设计失败
   (监督版 v2 就栽在"只罚亏不奖赚"上, 用户当场点破)。
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rl")
OUT = "/app/data/research/rl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

COST_BUY = 0.00025 + 0.00001 + 0.001
COST_SELL = 0.00025 + 0.0005 + 0.00001 + 0.001
POS = np.array([0.0, 0.5, 1.0], dtype=np.float32)


class AC(nn.Module):
    """网络故意做小 —— 数据信噪比极低, 大网络只会把噪声记下来。"""

    def __init__(self, n_in: int, h: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, h), nn.LayerNorm(h), nn.Tanh(),
                                  nn.Linear(h, h), nn.LayerNorm(h), nn.Tanh())
        self.pi = nn.Linear(h, 3)
        self.v = nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)


class Batch:
    """一批并行的"游戏局"。向量化跑 —— 逐局 Python 循环会慢到没法训。"""

    def __init__(self, feats, ret, starts, L):
        self.f = feats          # (N, F)
        self.r = ret            # (N,)
        self.s = starts         # (B,) 每局起点在全局数组里的下标
        self.L = L
        self.B = len(starts)
        self.reset()

    def reset(self):
        self.t = 0
        self.pos = np.zeros(self.B, np.float32)
        self.eq = np.ones(self.B, np.float32)
        self.peak = np.ones(self.B, np.float32)
        return self.obs()

    def obs(self):
        idx = self.s + self.t
        return np.concatenate([
            self.f[idx],
            self.pos[:, None],
            np.clip(self.eq - 1.0, -0.5, 0.5)[:, None],
            np.clip(self.eq / self.peak - 1.0, -0.5, 0.0)[:, None],
        ], axis=1).astype(np.float32)

    def step(self, act, dd_pen):
        new = POS[act]
        d = new - self.pos
        cost = np.where(d > 0, d * COST_BUY, -d * COST_SELL)
        self.pos = new
        r = self.r[self.s + self.t]
        gain = self.pos * r - cost
        before = self.eq
        self.eq = self.eq * (1.0 + gain)
        self.peak = np.maximum(self.peak, self.eq)
        # ⚠️ 只罚【新增】水下, 不罚存量 —— 罚存量会让策略一亏就割,
        #    学到的是"别被套住"而不是"别亏"。
        dd_now = np.maximum(0.0, 1.0 - self.eq / self.peak)
        dd_pre = np.maximum(0.0, 1.0 - before / np.maximum(self.peak, before))
        # ⚠️ 奖励放大 100 倍(即以"百分点"为单位)。原始日超额量级 1e-3,
        #    与网络初始化和学习率不匹配, 梯度太小学不动。
        # ⚠️ 回撤惩罚不能大。日超额 0.1~3%, 惩罚系数 2.0 时一次 3% 回撤扣 0.06,
        #    把任何持仓的期望值都压成负的 -> 策略学成永远空仓(实测 400 轮
        #    仓位恒为 0.00)。这与监督版 v2"只罚亏不奖赚"是同一个病。
        rew = ((self.eq - before) - dd_pen * np.maximum(0.0, dd_now - dd_pre)) * 100.0
        self.t += 1
        return self.obs(), rew.astype(np.float32), self.t >= self.L


def valid_starts(cid, day, L, rng, n):
    """随机抽 n 个"同一只票、连续 L 根"的起点。⚠️ 必须同票且日期连号,
    否则一局里会跨股票或跨长假, 学到的是拼接处的假信号。"""
    N = len(cid)
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        i = rng.integers(0, N - L - 1, size=min(n * 2, 100000))
        # ⚠️ L 个【交易日】≈ L×1.4 个自然日(周末), 判据写成 L+12 会一个都选不出来。
        #    留到 1.6 倍以容纳长假, 超过就说明中间停牌或跨了年, 不要。
        ok = (cid[i] == cid[i + L]) & (day[i + L] - day[i] <= int(L * 1.6) + 5)
        out.extend(i[ok].tolist())
        tries += len(i)
    got = np.array(out[:n], np.int64)
    if len(got) == 0:
        raise RuntimeError("一个合法起点都没抽到 —— 检查连续性判据")
    return got


def evaluate(net, feats, ret, cid, day, L, rng, n_ep=4000):
    st = valid_starts(cid, day, L, rng, n_ep)
    b = Batch(feats, ret, st, L)
    o = b.reset()
    poss, eqs = [], None
    with torch.no_grad():
        done = False
        while not done:
            logits, _ = net(torch.as_tensor(o, device=DEV))
            act = logits.argmax(-1).cpu().numpy()      # 评估用贪心
            o, _, done = b.step(act, 0.0)
            poss.append(b.pos.mean())
    eqs = b.eq
    return float(np.mean(eqs) - 1.0) * 100, float(np.mean(poss)), float(np.mean(eqs < 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--L", type=int, default=40, help="一局多少根K线")
    ap.add_argument("--B", type=int, default=1024, help="并行局数")
    ap.add_argument("--dd", type=float, default=0.0, help="回撤惩罚权重")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--bc", type=int, default=2000, help="行为克隆轮数, 0=不热启动")
    ap.add_argument("--bc-in", type=float, default=0.99, help="教师进场分位")
    ap.add_argument("--bc-out", type=float, default=0.80, help="教师出场分位")
    ap.add_argument("--ent", type=float, default=0.02, help="熵奖励, 防过早收敛到空仓")
    a = ap.parse_args()

    tr = np.load(f"{OUT}/train.npz", allow_pickle=True)
    te = np.load(f"{OUT}/test.npz", allow_pickle=True)
    n_in = tr["feats"].shape[1] + 3
    log.info("设备 %s | 训练 %s 步 测试 %s 步 | 状态维 %d",
             DEV, f"{len(tr['ret']):,}", f"{len(te['ret']):,}", n_in)
    log.info("局长 %d 并行 %d 回撤惩罚 %.1f 熵 %.3f", a.L, a.B, a.dd, a.ent)

    net = AC(n_in).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    rng = np.random.default_rng(42)

    # ---- 行为克隆热启动 ----
    # ⚠️ 不热启动的话 PPO 必然收敛到"永远空仓": 随机换手 40 天要亏 3.86%
    #    (几乎全是手续费), 成本梯度把 alpha 信号完全盖住, 策略在发现
    #    "选对票能赚钱"之前就先学会了"别动"。实测 dd=0 也一样。
    # 教师策略: 监督模型分位 >=0.99 时满仓, 否则空仓 —— 这正是虚拟盘
    #    形态模型账户在跑的规则, 是一个已知及格(而非最优)的起点。
    if a.bc > 0:
        sup_rank = tr["feats"][:, -2]              # addscore 补的分位列
        # ⚠️ 教师必须带【滞回】: "分位>=0.99 满仓, 否则空仓"看着对, 实则是
        #    天天换手 —— 分数通常只在前1%待一天, 次日掉出就卖。实测这个
        #    教师热启动后样本外 -0.039%/局, 全被手续费吃掉。
        #    改成 进场0.99 / 跌破0.80 才出, 自然形成多日持有, 不需要计数器。
        log.info("行为克隆热启动 %d 轮 (教师: 进场分位>=%.2f, 跌破%.2f 出)",
                 a.bc, a.bc_in, a.bc_out)
        for k in range(a.bc):
            idx = rng.integers(0, len(sup_rank), size=8192)
            r = sup_rank[idx]
            # 随机给一个"当前是否持仓"的状态, 让策略学会两种情形下都做对
            held = (rng.random(len(idx)) < 0.5).astype(np.float32)
            y = np.where((r >= a.bc_in) | ((held > 0) & (r >= a.bc_out)), 2, 0)
            x = np.concatenate([
                tr["feats"][idx],
                held[:, None],                      # pos
                np.zeros((len(idx), 2), np.float32),  # 累计超额 / 水下
            ], axis=1).astype(np.float32)
            logits, _ = net(torch.as_tensor(x, device=DEV))
            loss = F.cross_entropy(logits, torch.as_tensor(y, device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            if (k + 1) % 500 == 0:
                acc = float((logits.argmax(-1).cpu().numpy() == y).mean())
                log.info("  BC %4d 损失 %.4f 复现率 %.3f", k + 1,
                         float(loss.detach()), acc)
        eq, pos, lose = evaluate(net, te["feats"], te["ret"], te["cid"],
                                 te["day"], a.L, rng)
        log.info("热启动后(样本外): 局均超额 %+.3f%% 仓位 %.2f 亏损局 %.0f%%",
                 eq, pos, lose * 100)

    for it in range(1, a.iters + 1):
        st = valid_starts(tr["cid"], tr["day"], a.L, rng, a.B)
        b = Batch(tr["feats"], tr["ret"], st, a.L)
        o = b.reset()
        O, A, LP, R, V = [], [], [], [], []
        done = False
        with torch.no_grad():
            while not done:
                ot = torch.as_tensor(o, device=DEV)
                logits, v = net(ot)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                O.append(ot); A.append(act); LP.append(dist.log_prob(act)); V.append(v)
                o, rew, done = b.step(act.cpu().numpy(), a.dd)
                R.append(torch.as_tensor(rew, device=DEV))
        # 折扣回报(γ=1, 一局就是完整持仓周期, 不需要再打折)
        G = torch.zeros_like(R[0])
        rets = []
        for r in reversed(R):
            G = r + G
            rets.append(G.clone())
        rets = torch.stack(list(reversed(rets)))
        Ob = torch.cat(O); Ab = torch.cat(A); LPb = torch.cat(LP)
        Vb = torch.cat(V); Rb = rets.reshape(-1)
        adv = Rb - Vb
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(4):                       # PPO 多轮复用同一批
            logits, v = net(Ob)
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(Ab) - LPb).exp()
            l_pi = -torch.min(ratio * adv,
                              ratio.clamp(1 - a.clip, 1 + a.clip) * adv).mean()
            l_v = F.mse_loss(v, Rb)
            loss = l_pi + 0.5 * l_v - a.ent * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()

        if it % 25 == 0 or it == 1:
            eq_tr = float(np.mean(b.eq) - 1.0) * 100
            pos_tr = float(b.pos.mean())
            eq_te, pos_te, lose_te = evaluate(net, te["feats"], te["ret"],
                                              te["cid"], te["day"], a.L, rng)
            log.info("iter %3d | 训练 超额%+6.3f%% 仓位%.2f | "
                     "测试 超额%+6.3f%% 仓位%.2f 亏损局%.0f%%",
                     it, eq_tr, pos_tr, eq_te, pos_te, lose_te * 100)
            torch.save(net.state_dict(), f"{OUT}/ppo.pt")

    log.info("训练结束, 模型存 %s/ppo.pt", OUT)


if __name__ == "__main__":
    main()
