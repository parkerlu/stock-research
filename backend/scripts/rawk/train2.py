"""裸K模型 v2 —— 只改方法, 不改输入。

输入仍是【裸K + 量 + 一条MA + 真实大盘K线】, 与 v1 完全相同。
改的是三个方法缺陷(2026-09-08 从 v1 的日志里诊断出来的):

⚠️ 缺陷一: 用测试集挑轮次。
   v1 逐轮在 +0.12 ~ +0.47 之间震荡, 而我每次报的是最高那轮 ——
   这是【用测试集做模型选择】, 是泄漏。真实水平接近震荡均值(约+0.30)。
   也因此 "h=128 比 h=48 差" 这个对比从一开始就不公平: h=128 只跑了6轮,
   还没机会挑出它的幸运轮。
   → 改成三段: 训练 ≤2019 / 验证 2020 / 测试 ≥2021。
     用【验证集】挑轮次, 测试集只在最后碰一次。

⚠️ 缺陷二: 有效样本量只有名义的 1/20。
   标签是未来20日, 相邻两天的标签重叠95% —— 400万名义样本里只有约20万
   是独立的。这解释了"样本从80万加到400万没用"和"容量一放大就过拟合"。
   → 净化采样: 同一只票的训练样本间隔 >= H 天, 标签互不重叠。

⚠️ 缺陷三: 训练目标与使用方式不一致。
   用 Huber 拟合数值, 但实际只用【当日排名】。
   → 主指标改用 RankIC(与用法对齐, 也与文献可比), 并保留分层价差。

另外加了多种子集成 —— 直接消灭那个震荡, 比挑幸运轮诚实。
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rawk.train import DEV, NBAR, Net, make_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rawk2")
DIR = "/root/data"


def purged_sample(idx_pos: np.ndarray, cid: np.ndarray, day: np.ndarray,
                  h: int, rng) -> np.ndarray:
    """净化采样: 同一只票内, 相邻被选样本的间隔 >= h 天。

    ⚠️ 不做这一步的话, 名义样本量是有效样本量的 h 倍, 容量/数据量的对比
       全部失真 —— 加的"新数据"其实是同一段行情的重复标注。
    """
    order = np.lexsort((day[idx_pos], cid[idx_pos]))
    s = idx_pos[order]
    c, d = cid[s], day[s]
    keep = np.zeros(len(s), bool)
    last_c, last_d = -1, -10**9
    # 每只票随机起一个相位, 避免总是取每月同一天
    phase = rng.integers(0, h)
    for i in range(len(s)):
        if c[i] != last_c:
            last_c, last_d = c[i], d[i] - h + phase
        if d[i] - last_d >= h:
            keep[i] = True
            last_d = d[i]
    return s[keep]


def rank_ic(day: np.ndarray, score: np.ndarray, y: np.ndarray) -> float:
    """逐日 RankIC 的均值 —— 与"每天在全市场里排名"这个用法对齐。"""
    d = pd.DataFrame({"day": day, "s": score, "y": y})
    g = d.groupby("day")
    ic = g.apply(lambda x: x["s"].corr(x["y"], method="spearman")
                 if len(x) > 20 else np.nan, include_groups=False)
    return float(np.nanmean(ic.to_numpy()))


PRED_BS = 4096


@torch.no_grad()
def predict(net, ohlcv, idx, day_all, mkt, day_min, bs=None, nbar=NBAR):
    bs = bs or PRED_BS
    net.eval()
    out = None
    for i in range(0, len(idx), bs):
        x = make_batch(ohlcv, idx[i:i + bs], nbar=nbar, day=day_all, mkt=mkt, day_min=day_min)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            o = net(torch.as_tensor(x, device=DEV))
        if o.ndim == 2:                       # 三分类: 输出概率 (B,3)
            o = torch.softmax(o.float(), dim=1)
        o = o.float().cpu().numpy()
        if out is None:
            out = np.empty((len(idx),) + o.shape[1:], np.float32)
        out[i:i + bs] = o
    net.train()
    return out


def report(tag, day, score, y):
    d = pd.DataFrame({"day": day, "y": y, "s": score})
    d["rk"] = d.groupby("day")["s"].rank(pct=True)
    t1 = d.loc[d.rk >= 0.99, "y"].mean() * 100
    t5 = d.loc[d.rk >= 0.95, "y"].mean() * 100
    b20 = d.loc[d.rk < 0.20, "y"].mean() * 100
    ic = rank_ic(day, score, y)
    yr = pd.to_datetime(d["day"], unit="D").dt.year
    ys = d[d.rk >= 0.95].groupby(yr[d.rk >= 0.95])["y"].mean().mul(100)
    log.info("%s  RankIC %+.4f | Top1%% %+.2f Top5%% %+.2f Bot20%% %+.2f | 负年 %d",
             tag, ic, t1, t5, b20, int((ys < 0).sum()))
    log.info("      逐年 %s", {int(k): round(float(v), 2) for k, v in ys.items()})
    return ic


def report_cls(tag, day, p, y3, ret3, vol20=None, up=0.10, dn=0.05):
    """三分类的评价 —— 直接回答用户的问题: 模型说"会涨到+10%"的票里, 真到的有多少。
    每笔期望 = +up*涨率 - dn*跌率 + 平均平盘收益, 这就是盈利标准。"""
    d = pd.DataFrame({"day": day, "pu": p[:, 1], "y": y3, "r": ret3})
    d["rk"] = d.groupby("day")["pu"].rank(pct=True)
    rows = []
    for lo, hi, nm in ((0.99, 1.01, "Top1%"), (0.95, 1.01, "Top5%"), (0.90, 1.01, "Top10%"),
                       (0.0, 0.20, "Bot20%"), (0.0, 1.01, "全部")):
        g = d[(d.rk >= lo) & (d.rk < hi)]
        rows.append(f"{nm} 涨{(g.y == 1).mean() * 100:.1f}% 跌{(g.y == 2).mean() * 100:.1f}% "
                    f"期望{g.r.mean() * 100:+.2f}%")
    g5 = d[d.rk >= 0.95]
    yr = pd.to_datetime(g5["day"], unit="D").dt.year
    ys = g5.groupby(yr)["r"].mean().mul(100)
    ev5 = float(g5.r.mean() * 100)
    log.info("%s\n      %s | 负年 %d", tag, " | ".join(rows), int((ys < 0).sum()))
    log.info("      Top5%% 逐年期望%% %s", {int(k): round(float(v), 2) for k, v in ys.items()})
    # 另一种排法: 按期望值 up*P(涨)-dn*P(跌) 排 —— 同一个模型, 只是不单看 P(涨)
    d["ev"] = up * p[:, 1] - dn * p[:, 2]
    d["rk2"] = d.groupby("day")["ev"].rank(pct=True)
    g = d[d.rk2 >= 0.95]
    log.info("      按期望值排 Top5%%: 涨%.1f%% 跌%.1f%% 期望%+.2f%%",
             (g.y == 1).mean() * 100, (g.y == 2).mean() * 100, g.r.mean() * 100)
    if vol20 is not None:
        d["v"] = vol20
        rho = d.groupby("day").apply(lambda x: x["pu"].corr(x["v"], method="spearman")
                                     if len(x) > 20 else np.nan, include_groups=False)
        log.info("      P(涨) 与 20日波动率 日内Spearman %+.3f  (接近 0 才说明没在学波动率)",
                 float(np.nanmean(rho.to_numpy())))
    return ev5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5, help="集成几个种子")
    ap.add_argument("--hold", type=int, default=20, help="标签持有期, 用于净化间隔")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rank-loss", action="store_true",
                    help="用【当日横截面排序】当目标, 而不是拟合数值")
    ap.add_argument("--no-mkt", action="store_true",
                    help="不给大盘K线 —— 对照实验, 判断优势是选股还是择时")
    ap.add_argument("--cls", action="store_true",
                    help="三分类: 次日开盘买, 10日内先到+10%%=涨 / 先到-5%%=跌 / 都没=平 (label3.npz)")
    ap.add_argument("--label", default="label3.npz", help="三分类标签文件(rawk.label3 生成)")
    ap.add_argument("--smoke", action="store_true", help="只取几千样本跑通流程")
    ap.add_argument("--nbar", type=int, default=NBAR, help="输入几根K线(对照 100/200/400)")
    ap.add_argument("--min-hist", type=int, default=0,
                    help="只用有 >=N 根同股历史的样本 —— nbar 对照时三组用同一批样本才公平")
    a = ap.parse_args()
    global PRED_BS
    PRED_BS = min(4096, a.bs * 2)

    d = np.load(f"{DIR}/panel.npz")
    ohlcv, idx, y, lday = d["ohlcv"], d["idx"], d["y"], d["lab_day"]
    day_all, mkt, day_min = d["day"], d["mkt"], int(d["day_min"])
    if a.no_mkt:
        mkt = mkt[:0]
        log.info("⚠️ 对照组: 不给大盘K线")
    n_ch = 6 + mkt.shape[0] * 4
    yr = pd.to_datetime(pd.Series(lday), unit="D").dt.year.to_numpy()
    y3 = ret3 = vol20 = None
    UP, DN = 0.10, 0.05
    valid = np.ones(len(y), bool)
    if a.cls:
        l3 = np.load(f"{DIR}/{a.label}")
        y3, ret3, vol20 = l3["y3"], l3["ret3"], l3["vol20"]
        valid = y3 >= 0
        y = ret3                                  # 报告里的"收益"换成这套出场规则下的真实收益
        a.hold = int(l3["hold"])                  # 净化间隔跟标签持有期走
        UP, DN = float(l3["up"]), float(l3["dn"])
        log.info("⚠️ 三分类模式: +%.0f%%/-%.0f%%/%d日, 有效 %s | 基准 涨 %.1f%% 跌 %.1f%%",
                 float(l3["up"]) * 100, float(l3["dn"]) * 100, a.hold, f"{valid.sum():,}",
                 (y3[valid] == 1).mean() * 100, (y3[valid] == 2).mean() * 100)

    # ⚠️ nbar 对照: 三组样本必须完全一样, 否则 400 根组少了新股, 不可比
    min_hist = max(a.min_hist, a.nbar)
    if min_hist > NBAR:
        bounds = np.searchsorted(d["cid"], np.arange(int(d["cid"].max()) + 2))
        enough = idx - min_hist + 1 >= bounds[d["lab_cid"]]
        log.info("⚠️ 要求 >=%d 根同股历史: 样本 %s -> %s", min_hist,
                 f"{valid.sum():,}", f"{(valid & enough).sum():,}")
        valid = valid & enough
    dil = (1, 2, 4, 8, 16, 32, 64) if a.nbar >= 256 else (1, 2, 4, 8, 16, 32)
    log.info("输入 %d 根K线, 膨胀 %s (感受野 %d)", a.nbar, dil, 5 + 2 * sum(dil))

    # ⚠️ 三段划分: 验证集只用来挑轮次, 测试集只在最后碰一次
    tr_all = np.flatnonzero((yr <= 2019) & valid)
    va_all = np.flatnonzero((yr == 2020) & valid)
    te_all = np.flatnonzero((yr >= 2021) & valid)
    rng = np.random.default_rng(0)
    tr = purged_sample(tr_all, d["lab_cid"], lday, a.hold, rng)
    va = purged_sample(va_all, d["lab_cid"], lday, a.hold, rng)
    if a.smoke:
        tr, va, te_all = tr[:4000], va[:4000], te_all[:20000]
    log.info("训练 %s(净化前 %s) / 验证 %s / 测试 %s | 通道 %d",
             f"{len(tr):,}", f"{len(tr_all):,}", f"{len(va):,}", f"{len(te_all):,}", n_ch)
    log.info("⚠️ 净化后训练样本降到 %.1f%% —— 这才是【有效】样本量",
             len(tr) / max(len(tr_all), 1) * 100)

    # ⚠️ 排序损失: 把标签换成【当日横截面分位】再拟合。
    #    真正稳定的评价指标是 RankIC(逐日排序), 而原来用 Huber 拟合原始数值 ——
    #    数值受当日全市场波动影响, 大涨大跌的日子标签整体偏移, 模型被迫去
    #    拟合那个偏移。换成分位后, 每天的标签分布都一样, 模型只学"谁排前面"。
    y_fit = y
    if a.rank_loss:
        s_ = pd.DataFrame({"d": lday, "y": y})
        y_fit = (s_.groupby("d")["y"].rank(pct=True).to_numpy(np.float32) - 0.5)
        log.info("⚠️ 用排序损失: 标签换成当日分位(居中 ±0.5)")

    te_scores, va_scores = [], []
    for si in range(a.seeds):
        torch.manual_seed(si); np.random.seed(si)
        net = Net(ch=n_ch, h=a.h, dilations=dil, n_rep=a.rep, n_out=3 if a.cls else 1).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
        # ⚠️ delta 要和标签尺度匹配: 原始超额在 ±0.05 量级, 分位在 ±0.5
        lossf = nn.CrossEntropyLoss() if a.cls else nn.HuberLoss(delta=0.5 if a.rank_loss else 0.05)
        best_ic, best_state, bad = -9e9, None, 0
        r2 = np.random.default_rng(si)
        for ep in range(1, a.epochs + 1):
            t0 = time.time()
            perm = r2.permutation(len(tr))
            for i in range(0, len(tr), a.bs):
                b = tr[perm[i:i + a.bs]]
                x = torch.as_tensor(make_batch(ohlcv, idx[b], nbar=a.nbar, day=day_all,
                                               mkt=mkt, day_min=day_min), device=DEV)
                t = (torch.as_tensor(y3[b].astype(np.int64), device=DEV) if a.cls
                     else torch.as_tensor(y_fit[b], device=DEV))
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                    loss = lossf(net(x), t)
                opt.zero_grad(); scaler.scale(loss).backward()
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            sv = predict(net, ohlcv, idx[va], day_all, mkt, day_min, nbar=a.nbar)
            if a.cls:
                # ⚠️ 挑轮次的标准 = 验证集 Top5% 每笔期望收益 —— 就是盈利标准本身
                dv = pd.DataFrame({"d": lday[va], "p": sv[:, 1], "r": ret3[va]})
                dv["rk"] = dv.groupby("d")["p"].rank(pct=True)
                g = dv[dv.rk >= 0.95]
                ic = float(g.r.mean() * 100)
                log.info("  种子%d ep%-2d 验证Top5%% 期望%+.2f%% 涨%.1f%% (%.0fs)", si, ep, ic,
                         (y3[va][dv.rk.to_numpy() >= 0.95] == 1).mean() * 100, time.time() - t0)
            else:
                ic = rank_ic(lday[va], sv, y[va])
                log.info("  种子%d ep%-2d 验证RankIC %+.4f (%.0fs)", si, ep, ic, time.time() - t0)
            # ⚠️ 只看验证集挑最好的一轮, 测试集这时候一次都不碰
            if ic > best_ic:
                best_ic, bad = ic, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                best_val = sv
            else:
                bad += 1
                if bad >= 3:
                    log.info("  种子%d 连续3轮没提升, 提前停", si); break
        net.load_state_dict(best_state)
        st = predict(net, ohlcv, idx[te_all], day_all, mkt, day_min, nbar=a.nbar)
        te_scores.append(st); va_scores.append(best_val)
        if a.cls:
            report_cls(f"种子{si} 单模型(验证最优 Top5%期望 {best_ic:+.2f}%)", lday[te_all], st,
                       y3[te_all], ret3[te_all], vol20[te_all], UP, DN)
        else:
            report(f"种子{si} 单模型(验证最优 RankIC {best_ic:+.4f})", lday[te_all], st, y[te_all])
        torch.save(net.state_dict(), f"{DIR}/cnn2{a.tag}_s{si}.pt")

    # ⚠️ 集成: 把各种子的【当日排名】平均, 而不是原始分数平均 ——
    #    不同种子的分数尺度不同, 直接平均会被尺度大的那个主导。
    log.info("=" * 70)
    if a.cls:
        # 概率直接平均(同一尺度), 排序按 P(涨)
        pm = np.mean(te_scores, axis=0).astype(np.float32)
        report_cls(f"★ {a.seeds}种子集成", lday[te_all], pm, y3[te_all], ret3[te_all], vol20[te_all], UP, DN)
        pv = np.mean(va_scores, axis=0).astype(np.float32)
        # ⚠️ 同时存 2020 验证集的概率 —— 买入阈值只能在这上面定, 不能在测试集上挑
        np.savez(f"{DIR}/oos_v2{a.tag}.npz", score=pm[:, 1], p_up=pm[:, 1], p_dn=pm[:, 2],
                 lab_day=lday[te_all], lab_cid=d["lab_cid"][te_all],
                 va_p_up=pv[:, 1], va_p_dn=pv[:, 2], va_day=lday[va], va_idx=va, te_idx=te_all)
        log.info("样本外分数已存"); return
    ranks = []
    for s in te_scores:
        ranks.append(pd.DataFrame({"d": lday[te_all], "s": s})
                     .groupby("d")["s"].rank(pct=True).to_numpy())
    ens = np.mean(ranks, axis=0).astype(np.float32)
    report(f"★ {a.seeds}种子集成", lday[te_all], ens, y[te_all])
    np.savez(f"{DIR}/oos_v2{a.tag}.npz", score=ens, lab_day=lday[te_all],
             lab_cid=d["lab_cid"][te_all])
    log.info("样本外分数已存")


if __name__ == "__main__":
    main()
