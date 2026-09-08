"""低点组合 v2 —— 全市场信号计算 + 模型评分入库。

用法: python -m app.commands.build_didian

原指标「低点组合」有两类买入信号, 实测差异很大:
  阶段底部(动力线上穿0.2): H=20 超出随机 +0.557pp, 胜率 51.8%  ← 采用
  DIBU(13/21/34/55四周期KDJ全<20): H=5 反而【输给随机】-0.074pp  ← 弃用
条件更苛刻的 DIBU 更差 —— 深度超卖多出现在下跌趋势中段, 接的是下落的刀。

过滤器特征 = 价量 + 筹码。样本外(2020-2026) Top10%: +4.983%, 胜率 57.7%,
按天 t=3.88; 波动层内三档 t 全部 >2.5。
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DidianSignal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("build_didian")

PANEL = "/app/data/research/panel.npz"
CYQ = "/app/data/research/cyq_panel.npz"
HOLD = 20          # baseline 在 H=20 最强, 与买卖很准的 H=10 不同


def _roll(a, w, fn):
    out = np.full_like(a, np.nan)
    for i in range(w - 1, a.shape[0]):
        out[i] = fn(a[i - w + 1:i + 1], axis=0)
    return out


def _sh(a, k):
    o = np.full_like(a, np.nan)
    o[k:] = a[:-k]
    return o


def _sma_tdx(x, n, m):
    """TDX 的 SMA(X,N,M): Y = (M*X + (N-M)*Y')/N。

    ⚠️ 必须用递推式, 不能拿 pandas ewm 近似 —— 买卖很准那次自己重写 rolling
    导致 42% 信号错位, 教训还热着。
    """
    a = m / n
    out = np.full_like(x, np.nan)
    prev = None
    for i in range(x.shape[0]):
        cur = x[i]
        if prev is None:
            prev = np.where(np.isfinite(cur), cur, np.nan)
        else:
            prev = np.where(np.isfinite(cur),
                            a * cur + (1 - a) * np.where(np.isfinite(prev), prev, cur),
                            prev)
        out[i] = prev
    return out


def build():
    import xgboost as xgb

    z = np.load(PANEL, allow_pickle=True)
    dates_all = pd.to_datetime(z["dates"])
    codes = z["codes"]
    m0 = dates_all >= pd.Timestamp("2018-01-01")
    dates = dates_all[m0]
    C, Hh, L, O, V, A = (z[k][m0] for k in ["close", "high", "low", "open", "vol", "amount"])
    T, N = C.shape
    P = np.load(CYQ)
    log.info("面板 %d 天 × %d 只", T, N)

    # ---- 候选信号: 阶段底部 × DIBU ----
    # ⚠️ 窗口必须与图上画的一致(34/34)。原来这里写的是 LLV(low,10)/HHV(high,25),
    #    也就是【训练用的"阶段底部"和用户在图上看到的红柱不是同一个东西】——
    #    2026-09-08 发现, 属于长期存在的静默不一致。
    var2 = _roll(L, 34, np.nanmin)
    var33 = _roll(Hh, 34, np.nanmax)
    dongli = _sma_tdx((C - var2) / np.maximum(var33 - var2, 1e-9) * 4, 4, 2)
    dl_prev = np.vstack([np.full((1, N), np.nan), dongli[:-1]])
    stage_bottom = (dl_prev <= 0.2) & (dongli > 0.2)

    # ⚠️ DIBU(四周期KDJ同时超卖)当初被弃用, 理由是"H=5 时输给随机"——
    #    那是【短持有期】的结论。H=20 时 DIBU 单独就有 +0.708pp(百万级样本),
    #    与阶段底部组合后 +0.954pp, 比单用阶段底部(+0.843pp)高 13%,
    #    负年同样只有 1 个。2026-09-08 用户指出"这个指标很多信号没用到"后复测。
    def _kdj_k(period, sw):
        hh = _roll(Hh, period, np.nanmax)
        ll = _roll(L, period, np.nanmin)
        rsv = (C - ll) / np.maximum(hh - ll, 1e-9) * 100
        return _sma_tdx(rsv, sw, 1)

    k13 = _kdj_k(13, 3); k21 = _kdj_k(21, 3)
    k34 = _kdj_k(34, 3); k55 = _kdj_k(55, 5)
    d14 = _sma_tdx(k13, 3, 1)
    d55 = _sma_tdx(k55, 5, 1)
    dibu = (k13 < 30) & (k21 < 30) & (k34 < 30) & (k55 < 30)

    sig = stage_bottom & dibu
    log.info("阶段底部 %d, DIBU %d, 组合信号 %d",
             int(stage_bottom.sum()), int(dibu.sum()), int(sig.sum()))

    ret1 = np.full_like(C, np.nan)
    ret1[1:] = C[1:] / C[:-1] - 1
    logv = np.log1p(np.nan_to_num(V, nan=0.0))
    loga = np.log1p(np.nan_to_num(A, nan=0.0))
    vol20 = _roll(ret1, 20, np.nanstd)
    wr = P["winner_rate"]; c50 = P["cost_50pct"]
    c15 = P["cost_15pct"]; c85 = P["cost_85pct"]
    F = {
        "动量5": C / _sh(C, 5) - 1, "动量20": C / _sh(C, 20) - 1, "动量60": C / _sh(C, 60) - 1,
        "超卖深度": C / _roll(C, 20, np.nanmin) - 1,
        "距高点": C / _roll(C, 60, np.nanmax) - 1,
        "量比5": _roll(logv, 5, np.nanmean) - _roll(logv, 60, np.nanmean),
        "换手强度": loga - _roll(loga, 60, np.nanmean),
        "连跌天数": _roll((ret1 < 0).astype(np.float32), 10, np.nansum),
        "价格位置60": (C - _roll(C, 60, np.nanmin)) /
                      np.maximum(_roll(C, 60, np.nanmax) - _roll(C, 60, np.nanmin), 1e-9),
        "下影强度": (np.minimum(C, O) - L) / np.maximum(Hh - L, 1e-9),
        "收盘位置": (C - L) / np.maximum(Hh - L, 1e-9),
        "获利盘": wr,
        "获利盘变化20": wr - _sh(wr, 20),
        "获利盘背离": (wr - _sh(wr, 20)) - (C / np.maximum(_sh(C, 20), 1e-9) - 1) * 100,
        "现价比中位成本": C / np.maximum(c50, 1e-9) - 1,
        "筹码宽度": (c85 - c15) / np.maximum(c50, 1e-9),   # 本指标里重要性第一
        "成本上移20": c50 / np.maximum(_sh(c50, 20), 1e-9) - 1,
        # ---- 指标自身的连续线(2026-09-08 补) ----
        # ⚠️ 之前一条都没进特征 —— 模型只知道"信号触发了", 不知道触发时
        #    KDJ 有多低、动力线走到哪、快慢周期是否背离。这些正是人眼在图上
        #    看的东西。
        "动力线": dongli,
        "动力线变化5": dongli - _sh(dongli, 5),
        "K13": k13,
        "D14": d14,
        "K55": k55,
        "D55": d55,
        "K13减D14": k13 - d14,              # 快线金叉/死叉的连续版
        "K55减D55": k55 - d55,              # 慢线趋势
        "K13减K55": k13 - k55,              # 快慢周期背离
    }
    names = list(F)
    X_all = np.full((T, N, len(names)), np.nan, dtype=np.float32)
    for k, n in enumerate(names):
        f = F[n].astype(np.float32)
        mu = np.nanmean(f, axis=1, keepdims=True)
        sd = np.nanstd(f, axis=1, keepdims=True)
        X_all[:, :, k] = (f - mu) / np.maximum(sd, 1e-9)
    del F

    entry = np.full((T, N), np.nan); entry[:-1] = O[1:]
    exit_ = np.full((T, N), np.nan); exit_[:-(HOLD + 1)] = O[HOLD + 1:]
    ret = exit_ / entry - 1
    lim = np.zeros((T, N), bool)
    lim[:-1] = (O[1:] == Hh[1:]) & (O[1:] == L[1:])

    trainable = sig & np.isfinite(ret) & (~lim) & np.isfinite(X_all).all(axis=2) & np.isfinite(vol20)
    scorable = sig & np.isfinite(X_all).all(axis=2) & np.isfinite(vol20)
    tr_r, tr_c = np.where(trainable)
    sc_r, sc_c = np.where(scorable)
    log.info("可训练 %d, 待打分 %d", len(tr_r), len(sc_r))

    Xtr = X_all[tr_r, tr_c]; ytr = ret[tr_r, tr_c]; vtr = vol20[tr_r, tr_c]
    yr_tr = dates[tr_r].year
    df = pd.DataFrame({"d": tr_r, "y": ytr, "v": vtr})
    df["vq"] = df.groupby("d")["v"].transform(
        lambda x: pd.qcut(x.rank(method="first"), min(5, max(1, len(x))), labels=False)
        if len(x) >= 5 else 0)
    y_rel = (ytr - df.groupby(["d", "vq"])["y"].transform("mean").values).astype(np.float32)
    Xsc = X_all[sc_r, sc_c]
    yr_sc = dates[sc_r].year
    del X_all

    score = np.full(len(sc_r), np.nan, dtype=np.float32)
    for Y in sorted(set(yr_sc)):
        tr = yr_tr < Y
        if tr.sum() < 15000:
            continue
        m = xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=100,
                             reg_lambda=2.0, tree_method="hist", n_jobs=8, verbosity=0,
                                # ⚠️ 必须固定种子: 每日 15:30 都会全量重训,
                                # 不固定的话 subsample/colsample 的随机性会让同一个历史
                                # 信号的 rank_pct 天天漂, 昨天"强"今天可能变"中"。
                                random_state=42)
        m.fit(Xtr[tr], y_rel[tr])
        te = yr_sc == Y
        if te.sum():
            score[te] = m.predict(Xsc[te])
        log.info("  %d 年打分 %d", Y, int(te.sum()))

    ok = np.isfinite(score)
    d = pd.DataFrame({"row": sc_r[ok], "col": sc_c[ok], "score": score[ok]})
    d["rank_pct"] = d.groupby("row")["score"].rank(pct=True)
    d["grade"] = np.where(d.rank_pct >= 0.9, "强",
                          np.where(d.rank_pct >= 0.7, "中", "弱"))
    d = d[d.grade != "弱"].copy()
    d["ts_code"] = codes[d.col.values]
    d["trade_date"] = dates[d.row.values].date
    log.info("入库 %d 条 (中/强档)", len(d))
    return d[["ts_code", "trade_date", "score", "rank_pct", "grade"]]


async def main() -> None:
    d = build()
    eng = create_async_engine(settings.database_url)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        await db.execute(delete(DidianSignal))
        await db.commit()
    rows = d.to_dict("records")
    CH = 5000
    async with Session() as db:
        for i in range(0, len(rows), CH):
            chunk = [{"ts_code": r["ts_code"], "trade_date": r["trade_date"],
                      "score": float(r["score"]), "rank_pct": float(r["rank_pct"]),
                      "grade": r["grade"]} for r in rows[i:i + CH]]
            stmt = insert(DidianSignal).values(chunk)
            await db.execute(stmt.on_conflict_do_update(
                index_elements=["ts_code", "trade_date"],
                set_={"score": stmt.excluded.score, "rank_pct": stmt.excluded.rank_pct,
                      "grade": stmt.excluded.grade}))
            await db.commit()
            if (i // CH) % 10 == 0:
                log.info("  写入 %d/%d", i, len(rows))
    log.info("完成 %d 条", len(rows))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
