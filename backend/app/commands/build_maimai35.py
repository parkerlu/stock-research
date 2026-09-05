"""买卖很准 v3.5 —— 参数重扫版 + 模型过滤。

用法: python -m app.commands.build_maimai35

原指标参数(MA5/LLV10/连续5)是移植 TDX 时照抄的, 36 组网格扫描后只排 9/33。
本版用 MA8/LLV20/连续10:

    全市场 (H=20, vs 同期随机对照)
    原参数  28.1万信号  胜率 50.9%  超出 +0.190pp
    v3.5     9.5万信号  胜率 56.2%  超出 +0.602pp

⚠️ 信号计算用 pandas rolling 逐股票做, 不做向量化重写 ——
v3 那次为性能自己重写 rolling, min_periods 处理与 pandas 不一致,
648 天里 59 天买线值不同, 漏掉 42% 信号。宁可慢 30 秒。
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
from app.models.schema import Maimai35Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("build_maimai35")

PANEL = "/app/data/research/panel.npz"
CYQ = "/app/data/research/cyq_panel.npz"
MA_N, LLV_N, RUN_N = 8, 20, 10     # 扫描出来的最优组合
HOLD = 20


def _roll(a, w, fn):
    out = np.full_like(a, np.nan)
    for i in range(w - 1, a.shape[0]):
        out[i] = fn(a[i - w + 1:i + 1], axis=0)
    return out


def _sh(a, k):
    o = np.full_like(a, np.nan)
    o[k:] = a[:-k]
    return o


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
    log.info("面板 %d 天 × %d 只, 参数 MA%d/LLV%d/连续%d", T, N, MA_N, LLV_N, RUN_N)

    # ---- 信号: 逐股票用 pandas rolling(与 compute_lines 同语义) ----
    lao = (L + Hh + C) / 3.0
    sig = np.zeros((T, N), dtype=bool)
    for j in range(N):
        col = lao[:, j]
        if np.isfinite(col).sum() < MA_N + LLV_N + RUN_N:
            continue
        s_lao = pd.Series(col)
        ban = s_lao.rolling(MA_N).mean()
        thr = ban.rolling(LLV_N).min()
        cond = (pd.Series(C[:, j]) < thr).fillna(False)
        # LLV(cond, RUN_N) > 0 —— 过去 RUN_N 天全部满足
        line = cond.rolling(RUN_N).min().fillna(0) > 0
        sig[:, j] = (line.shift(1).fillna(False) & ~line).values
        if (j + 1) % 1500 == 0:
            log.info("  信号 %d/%d", j + 1, N)
    log.info("v3.5 信号 %d", int(sig.sum()))

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
        "筹码宽度": (c85 - c15) / np.maximum(c50, 1e-9),
        "成本上移20": c50 / np.maximum(_sh(c50, 20), 1e-9) - 1,
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
        if tr.sum() < 10000:
            continue
        m = xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=100,
                             reg_lambda=2.0, tree_method="hist", n_jobs=8, verbosity=0)
        m.fit(Xtr[tr], y_rel[tr])
        te = yr_sc == Y
        if te.sum():
            score[te] = m.predict(Xsc[te])
        log.info("  %d 年打分 %d", Y, int(te.sum()))

    ok = np.isfinite(score)
    d = pd.DataFrame({"row": sc_r[ok], "col": sc_c[ok], "score": score[ok]})
    d["rank_pct"] = d.groupby("row")["score"].rank(pct=True)
    d["grade"] = np.where(d.rank_pct >= 0.8, "强",
                          np.where(d.rank_pct >= 0.5, "中", "弱"))
    d["ts_code"] = codes[d.col.values]
    d["trade_date"] = dates[d.row.values].date
    log.info("入库 %d 条", len(d))
    return d[["ts_code", "trade_date", "score", "rank_pct", "grade"]]


async def main() -> None:
    d = build()
    eng = create_async_engine(settings.database_url)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        await db.execute(delete(Maimai35Signal))
        await db.commit()
    rows = d.to_dict("records")
    CH = 5000
    async with Session() as db:
        for i in range(0, len(rows), CH):
            chunk = [{"ts_code": r["ts_code"], "trade_date": r["trade_date"],
                      "score": float(r["score"]), "rank_pct": float(r["rank_pct"]),
                      "grade": r["grade"]} for r in rows[i:i + CH]]
            stmt = insert(Maimai35Signal).values(chunk)
            await db.execute(stmt.on_conflict_do_update(
                index_elements=["ts_code", "trade_date"],
                set_={"score": stmt.excluded.score, "rank_pct": stmt.excluded.rank_pct,
                      "grade": stmt.excluded.grade}))
            await db.commit()
    log.info("完成 %d 条", len(rows))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
