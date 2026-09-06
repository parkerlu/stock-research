"""主力吸筹指标 —— 全市场评分入库。

用法: python -m app.commands.build_pump

预测目标: 未来10个交易日内出现【单日涨幅>7% 且 成交量>过去20日均量2倍】。
价量齐升才算拉升 —— 只涨不放量拉不动。

核心信息来自筹码分布(cyq_perf)的获利盘族, 占模型重要性 60%:
  获利盘背离 = 获利盘变化 − 价格涨幅 → 价格没动但获利盘上升 = 低位换手
在【价格位置 × 量能】双重控制的 3×3 九宫格里, 九格 Q10/Q1 全部 >1.63,
说明它带来的是价格和成交量里【没有】的信息。

样本外(2020-2026 walk-forward): Q10/Q1=4.81, 按天t=77.4, 逐年 1.70~2.02 倍。

⚠️ 评分用 walk-forward 逐年重训生成, 每年的分只用该年之前的数据训出。
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
from app.models.schema import PumpSignal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("build_pump")

PANEL = "/app/data/research/panel.npz"
CYQ = "/app/data/research/cyq_panel.npz"
FWD = 10


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
    C, Hh, L, V = (z[k][m0] for k in ["close", "high", "low", "vol"])
    T, N = C.shape
    P = np.load(CYQ)
    log.info("面板 %d 天 × %d 只", T, N)

    ret1 = np.full_like(C, np.nan)
    ret1[1:] = C[1:] / C[:-1] - 1
    vol20 = _roll(V, 20, np.nanmean)
    pump_day = (ret1 > 0.07) & (V > 2 * vol20)
    y_mat = np.zeros((T, N), bool)
    for i in range(T - 1):
        y_mat[i] = pump_day[i + 1:min(i + 1 + FWD, T)].any(axis=0)
    log.info("拉升日 %d", int(pump_day.sum()))

    wr = P["winner_rate"]; wa = P["weight_avg"]
    c50 = P["cost_50pct"]; c15 = P["cost_15pct"]; c85 = P["cost_85pct"]
    logv = np.log1p(np.nan_to_num(V, nan=0.0))
    F = {
        "获利盘": wr,
        "获利盘变化5": wr - _sh(wr, 5),
        "获利盘变化20": wr - _sh(wr, 20),
        "获利盘背离": (wr - _sh(wr, 20)) - (C / np.maximum(_sh(C, 20), 1e-9) - 1) * 100,
        "获利盘背离5": (wr - _sh(wr, 5)) - (C / np.maximum(_sh(C, 5), 1e-9) - 1) * 100,
        "现价比平均成本": C / np.maximum(wa, 1e-9) - 1,
        "现价比中位成本": C / np.maximum(c50, 1e-9) - 1,
        "筹码宽度": (c85 - c15) / np.maximum(c50, 1e-9),
        "成本上移20": c50 / np.maximum(_sh(c50, 20), 1e-9) - 1,
        "价格位置60": (C - _roll(C, 60, np.nanmin)) /
                      np.maximum(_roll(C, 60, np.nanmax) - _roll(C, 60, np.nanmin), 1e-9),
        "量能萎缩": _roll(V, 5, np.nanmean) / np.maximum(_roll(V, 60, np.nanmean), 1),
        "量比20": _roll(logv, 5, np.nanmean) - _roll(logv, 20, np.nanmean),
        "动量20": C / np.maximum(_sh(C, 20), 1e-9) - 1,
        "波动20": _roll(ret1, 20, np.nanstd),
    }
    names = list(F)
    X_all = np.full((T, N, len(names)), np.nan, dtype=np.float32)
    for k, n in enumerate(names):
        f = F[n].astype(np.float32)
        mu = np.nanmean(f, axis=1, keepdims=True)
        sd = np.nanstd(f, axis=1, keepdims=True)
        X_all[:, :, k] = (f - mu) / np.maximum(sd, 1e-9)
    del F

    ok = np.isfinite(C) & np.isfinite(wr) & np.isfinite(X_all).all(axis=2)
    rows, cols = np.where(ok)
    X = X_all[rows, cols]
    y = y_mat[rows, cols].astype(np.float32)
    yr = dates[rows].year
    del X_all
    log.info("样本 %d, 基础概率 %.2f%%", len(rows), y.mean() * 100)

    # 训练集降采样: 每 3 个交易日取 1 天。
    # 相邻交易日的特征高度重叠(都用 20/60 日窗口), 全量训练只是把同一批信息
    # 重复喂三遍 —— 800 万样本训一轮要几十分钟, 降到 1/3 精度几乎无损。
    # ⚠️ 只降训练集, 打分仍覆盖全部交易日。
    train_pool = (rows % 3) == 0
    log.info("训练集降采样: %d → %d", len(rows), int(train_pool.sum()))

    prob = np.full(len(rows), np.nan, dtype=np.float32)
    for Y in sorted(set(yr)):
        tr = yr < Y
        te = yr == Y
        if tr.sum() < 200000:
            continue
        m = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.06,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=200,
                              reg_lambda=2.0, tree_method="hist", n_jobs=8,
                              verbosity=0, eval_metric="logloss", max_bin=128,
                              # ⚠️ 必须固定种子: 每日 15:30 全量重训, 不固定的话
                              # subsample/colsample 的随机性会让同一个历史信号的
                              # rank_pct 天天漂, 昨天"强"今天可能变"中"。
                              random_state=42)
        m.fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
        log.info("  %d 年: 训练 %d, 打分 %d", Y, int(tr.sum()), int(te.sum()))

    has = np.isfinite(prob)
    d = pd.DataFrame({"row": rows[has], "col": cols[has], "prob": prob[has]})
    d["rank_pct"] = d.groupby("row")["prob"].rank(pct=True)
    d["grade"] = np.where(d.rank_pct >= 0.95, "强",
                          np.where(d.rank_pct >= 0.8, "中", "弱"))
    # 只入库中/强 —— 弱档占 80%, 存全量意义不大且拖慢查询
    d = d[d.grade != "弱"].copy()
    d["ts_code"] = codes[d.col.values]
    d["trade_date"] = dates[d.row.values].date
    log.info("入库 %d 条 (仅中/强档)", len(d))
    return d[["ts_code", "trade_date", "prob", "rank_pct", "grade"]]


async def main() -> None:
    d = build()
    eng = create_async_engine(settings.database_url)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        await db.execute(delete(PumpSignal))
        await db.commit()
    rows = d.to_dict("records")
    CH = 5000
    async with Session() as db:
        for i in range(0, len(rows), CH):
            chunk = [{"ts_code": r["ts_code"], "trade_date": r["trade_date"],
                      "prob": float(r["prob"]), "rank_pct": float(r["rank_pct"]),
                      "grade": r["grade"]} for r in rows[i:i + CH]]
            stmt = insert(PumpSignal).values(chunk)
            await db.execute(stmt.on_conflict_do_update(
                index_elements=["ts_code", "trade_date"],
                set_={"prob": stmt.excluded.prob, "rank_pct": stmt.excluded.rank_pct,
                      "grade": stmt.excluded.grade}))
            await db.commit()
            if (i // CH) % 20 == 0:
                log.info("  写入 %d/%d", i, len(rows))
    log.info("完成 %d 条", len(rows))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
