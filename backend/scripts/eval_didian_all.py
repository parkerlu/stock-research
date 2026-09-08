"""低点组合的【全部】信号与连续线, 逐个测一遍。

⚠️ 用户 2026-09-08 指出: 这个指标有 6 个信号 + 5 条连续线, 而 build_didian
   只用了其中一个(阶段底部 = 动力线上穿0.2)。DIBU 当初被弃用, 理由是
   "单独用跑输随机" —— 但那是【单独用】的结论, 没试过组合。

判据: 持有 H 日的【超额】收益(减当日全市场等权均值) —— 与本项目其它
      指标口径一致, 牛市普涨不给分。
⚠️ 同时看【与基准的差】和【逐年一致性】, 只看总量会被一两年拉起来。
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("didian_all")
PANEL = "/app/data/research/panel.npz"


def _sma_tdx(x, n, m):
    """TDX 的 SMA(x,n,m) = 递推 y = (m*x + (n-m)*y_prev)/n"""
    out = np.full_like(x, np.nan)
    prev = None
    for i in range(x.shape[0]):
        cur = x[i]
        if prev is None:
            prev = np.where(np.isfinite(cur), cur, 0.0)
        else:
            prev = np.where(np.isfinite(cur), (m * cur + (n - m) * prev) / n, prev)
        out[i] = prev
    return out


def _kdj(C, Hh, L, period, sw):
    hh = pd.DataFrame(Hh).rolling(period, min_periods=period).max().to_numpy()
    ll = pd.DataFrame(L).rolling(period, min_periods=period).min().to_numpy()
    rsv = (C - ll) / np.maximum(hh - ll, 1e-9) * 100
    k = _sma_tdx(rsv, sw, 1)
    d = _sma_tdx(k, sw, 1)
    return k, d


async def main() -> None:
    z = np.load(PANEL, allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    m0 = dates >= pd.Timestamp("2018-01-01")
    dates = dates[m0]
    C, Hh, L, O = (z[k][m0] for k in ["close", "high", "low", "open"])
    T, N = C.shape
    log.info("面板 %d 天 × %d 只", T, N)

    # ---- 动力线(与指标实现一致) ----
    var2 = pd.DataFrame(L).rolling(34, min_periods=34).min().to_numpy()
    var33 = pd.DataFrame(Hh).rolling(34, min_periods=34).max().to_numpy()
    dongli = _sma_tdx((C - var2) / np.maximum(var33 - var2, 1e-9) * 4, 4, 2)
    dprev = np.vstack([np.full((1, N), np.nan), dongli[:-1]])

    # ---- 四周期 KDJ ----
    ks = {p: _kdj(C, Hh, L, p, sw)[0] for p, sw in ((13, 3), (21, 3), (34, 3), (55, 5))}
    k13, d14 = _kdj(C, Hh, L, 13, 3)
    k55, d55 = _kdj(C, Hh, L, 55, 5)

    def edge_off(bar, w):
        """状态【结束】的那一天。

        ⚠️ 用户 2026-09-08 指出的关键: 红柱(阶段底部)本身是单日 CROSS 事件,
           但它成簇出现 —— 动力线在 0.2 附近反复穿越, 说明还在底部区挣扎。
           真正的买点是【簇结束】那一刻, 即"红柱出现过、然后不再出现"。
           这与买卖很准的"买线从非0熄灭才是买点"完全同构:
           状态持续 = 还在跌; 状态结束 = 跌不动了。
        """
        on = pd.DataFrame(bar.astype(float)).rolling(w, min_periods=1).max().to_numpy() > 0
        prev = np.vstack([np.zeros((1, on.shape[1]), bool), on[:-1]])
        return prev & (~on)

    sb = (dprev <= 0.2) & (dongli > 0.2)
    sw_ = (dprev <= 0.5) & (dongli > 0.5)
    dibu = np.all([ks[p] < 20 for p in ks], axis=0)

    SIGS = {
        "阶段底部 动力线上穿0.2": (dprev <= 0.2) & (dongli > 0.2),
        "阶段关注 动力线上穿0.5": (dprev <= 0.5) & (dongli > 0.5),
        "清仓 动力线下穿3.5":     (dprev >= 3.5) & (dongli < 3.5),
        "短线卖出 动力线下穿3.2": (dprev >= 3.2) & (dongli < 3.2),
        "DIBU 四周期KDJ全<20":    np.all([ks[p] < 20 for p in ks], axis=0),
        "TOBU 四周期KDJ全>80":    np.all([ks[p] > 80 for p in ks], axis=0),
        "K13上穿D14(金叉)":       (np.vstack([np.full((1,N),np.nan), (k13-d14)[:-1]]) <= 0) & ((k13-d14) > 0),
        "K55上穿D55(慢金叉)":     (np.vstack([np.full((1,N),np.nan), (k55-d55)[:-1]]) <= 0) & ((k55-d55) > 0),
        # 组合: 用户的提示 —— 单独弃用不代表组合无效
        "阶段底部 × DIBU":        ((dprev <= 0.2) & (dongli > 0.2)) & np.all([ks[p] < 30 for p in ks], axis=0),
        "阶段底部 × K55低位":     ((dprev <= 0.2) & (dongli > 0.2)) & (k55 < 30),
        "阶段关注 × K13金叉":     ((dprev <= 0.5) & (dongli > 0.5)) &
                                  (((np.vstack([np.full((1,N),np.nan),(k13-d14)[:-1]])<=0) & ((k13-d14)>0))),
        # ---- 用户提出的【状态结束】口径 ----
        "★红柱消失 w=3":          edge_off(sb, 3),
        "★红柱消失 w=5":          edge_off(sb, 5),
        "★红柱消失 w=8":          edge_off(sb, 8),
        "★红柱消失 w=12":         edge_off(sb, 12),
        "★品红消失 w=5":          edge_off(sw_, 5),
        "★DIBU状态结束 w=5":      edge_off(dibu, 5),
        "★红柱消失w5 × DIBU过":   edge_off(sb, 5) & (pd.DataFrame(dibu.astype(float)).rolling(20, min_periods=1).max().to_numpy() > 0),
    }

    yr = dates.year.to_numpy()
    log.info("%-26s %8s %9s %9s %8s %6s", "信号", "样本数", "超额%", "基准%", "超出pp", "负年")
    for H in (10, 20):
        log.info("=== 持有 %d 日 ===", H)
        entry = np.full((T, N), np.nan); entry[:-1] = O[1:]
        exit_ = np.full((T, N), np.nan); exit_[:-(H+1)] = O[H+1:]
        ret = exit_ / entry - 1
        mkt = np.nanmean(ret, axis=1, keepdims=True)
        ex = ret - mkt                      # 超额
        base = float(np.nanmean(ex))
        for nm, sg in SIGS.items():
            v = sg & np.isfinite(ex)
            n = int(v.sum())
            if n < 500:
                log.info("%-26s %8d  样本太少, 跳过", nm, n); continue
            e = float(np.nanmean(ex[v])) * 100
            neg = 0
            for y in np.unique(yr):
                mask = v & (yr[:, None] == y)
                if mask.sum() > 50 and np.nanmean(ex[mask]) < 0:
                    neg += 1
            log.info("%-26s %8d %+9.3f %+9.3f %+8.3f %6d",
                     nm, n, e, base * 100, e - base * 100, neg)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
