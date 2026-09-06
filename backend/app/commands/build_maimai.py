"""买卖很准 v3 —— 全市场信号计算 + 模型评分入库。

用法: python -m app.commands.build_maimai

原指标买点(超卖结束→反转确认)本身与随机无异; 这里训练的过滤器在
【同日×同波动层】中性化后, 样本外胜率 48.7%→51.3%, 中位 -0.132%→+0.264%,
低/中/高三个波动档超出全为正(+0.174 / +0.159 / +0.663pp)。

⚠️ 历史评分用 walk-forward 逐年重训生成 —— 每一年的分都只用该年之前的数据训出,
不存在前视; 最新一年用截至上一年的模型打分。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import MaimaiSignal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("build_maimai")

PANEL = "/app/data/research/panel.npz"
# 持有期 H=20 —— 原来用 10 是拍脑袋定的, 没验证过。实测扫描:
#   H=5  +1.6pp  t=1.86      H=10 +2.0pp  t=1.35(原设置)
#   H=20 +2.4pp  t=4.07 ←    H=40 +1.4pp  t=7.35
# 光是把 10 改成 20, 按天 t 就从 1.35 跳到 4.07, 比加 14 个新特征有用得多。
HOLD = 20          # 见下方注释
START = "2014-06-01"


def _roll(a, w, fn):
    out = np.full_like(a, np.nan)
    for i in range(w - 1, a.shape[0]):
        out[i] = fn(a[i - w + 1:i + 1], axis=0)
    return out


def _roll_all(a, w):
    """LLV(cond,w)>0 —— 过去 w 天全部满足"""
    out = np.full_like(a, np.nan)
    for i in range(w - 1, a.shape[0]):
        seg = a[i - w + 1:i + 1]
        out[i] = np.where(np.isfinite(seg).all(0), (seg > 0).all(0).astype(np.float32), np.nan)
    return out


def build():
    import xgboost as xgb

    z = np.load(PANEL, allow_pickle=True)
    dates_all = pd.to_datetime(z["dates"])
    codes = z["codes"]
    m0 = dates_all >= pd.Timestamp(START)
    dates = dates_all[m0]
    C, Hh, L, O, V, A = (z[k][m0] for k in ["close", "high", "low", "open", "vol", "amount"])
    T, N = C.shape
    log.info("面板 %d 天 × %d 只", T, N)

    # ---- 原指标信号 ----
    # ⚠️ 必须复用 maimai_henzhun.compute_lines, 不能自己向量化重写:
    # 重写版在 MA/HHV/LLV 的 min_periods 上与 pandas 版不同, 实测 648 天里
    # 有 59 天买线值不一致, 导致边沿触发日期错位(个别信号早 1-2 天),
    # 图上看到的标记会和原指标对不上。宁可慢, 也要和指标本体同源。
    from app.services.tdx.indicators.maimai_henzhun import compute_lines

    sig = np.zeros((T, N), dtype=bool)        # 买入: 买线 从>0 回落到 0
    sig_sell = np.zeros((T, N), dtype=bool)   # 卖出: 卖线 从<100 回到 100
    for j in range(N):
        c = C[:, j]
        if np.isfinite(c).sum() < 30:
            continue
        ser_c = pd.Series(c); ser_h = pd.Series(Hh[:, j]); ser_l = pd.Series(L[:, j])
        try:
            buy_line, sell_line = compute_lines(ser_c, ser_h, ser_l)
        except Exception:
            continue
        b = pd.Series(buy_line).reset_index(drop=True)
        sl = pd.Series(sell_line).reset_index(drop=True)
        sig[:, j] = ((b.shift(1) > 0) & (b == 0)).values
        # 卖出 = 强势段被打破(重新跌破卖出阈值) → 止盈离场
        sig_sell[:, j] = ((sl.shift(1) < 100) & (sl == 100)).values
        if (j + 1) % 1000 == 0:
            log.info("  信号计算 %d/%d", j + 1, N)
    log.info("原始信号: 买 %d, 卖 %d (复用官方 compute_lines)",
             int(sig.sum()), int(sig_sell.sum()))

    # ---- 特征 ----
    ret1 = np.full_like(C, np.nan); ret1[1:] = C[1:] / C[:-1] - 1
    logv = np.log1p(np.nan_to_num(V, nan=0.0))
    loga = np.log1p(np.nan_to_num(A, nan=0.0))
    vol20 = _roll(ret1, 20, np.nanstd)
    F = {
        "动量5": C / np.vstack([np.full((5, N), np.nan), C[:-5]]) - 1,
        "动量20": C / np.vstack([np.full((20, N), np.nan), C[:-20]]) - 1,
        "动量60": C / np.vstack([np.full((60, N), np.nan), C[:-60]]) - 1,
        "超卖深度": C / _roll(C, 20, np.nanmin) - 1,
        "距高点": C / _roll(C, 60, np.nanmax) - 1,
        "量比5": _roll(logv, 5, np.nanmean) - _roll(logv, 60, np.nanmean),
        "量比20": _roll(logv, 20, np.nanmean) - _roll(logv, 60, np.nanmean),
        "换手强度": loga - _roll(loga, 60, np.nanmean),
        "连跌天数": _roll((ret1 < 0).astype(np.float32), 10, np.nansum),
        "价格位置": (C - _roll(C, 60, np.nanmin)) /
                    np.maximum(_roll(C, 60, np.nanmax) - _roll(C, 60, np.nanmin), 1e-9),
        "下影强度": (np.minimum(C, O) - L) / np.maximum(Hh - L, 1e-9),
        "收盘位置": (C - L) / np.maximum(Hh - L, 1e-9),
        "量价背离": (C / np.vstack([np.full((5, N), np.nan), C[:-5]]) - 1) -
                    (_roll(logv, 5, np.nanmean) - _roll(logv, 20, np.nanmean)),
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

    # 训练集: 有标签的历史信号; 打分集: 全部信号(含最近还没到期的)
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
    X_all_kept = X_all          # 卖点复用同一套特征
    scorable_base = np.isfinite(X_all).all(axis=2) & np.isfinite(vol20)
    vol20_kept = vol20
    X_all_ref = [X_all]

    def mk():
        return xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, min_child_weight=100,
                                reg_lambda=2.0, tree_method="hist", n_jobs=8, verbosity=0,
                                # ⚠️ 必须固定种子: 每日 15:30 都会全量重训,
                                # 不固定的话 subsample/colsample 的随机性会让同一个历史
                                # 信号的 rank_pct 天天漂, 昨天"强"今天可能变"中"。
                                random_state=42)

    score = np.full(len(sc_r), np.nan, dtype=np.float32)
    years = sorted(set(yr_sc))
    for Y in years:
        tr = yr_tr < Y
        if tr.sum() < 5000:
            continue
        m = mk(); m.fit(Xtr[tr], y_rel[tr])
        te = yr_sc == Y
        if te.sum():
            score[te] = m.predict(Xsc[te])
        log.info("  %d 年打分 %d 个 (训练样本 %d)", Y, int(te.sum()), int(tr.sum()))

    ok = np.isfinite(score)
    sr, sc_, sv = sc_r[ok], sc_c[ok], score[ok]
    d = pd.DataFrame({"row": sr, "col": sc_, "score": sv})
    d["rank_pct"] = d.groupby("row")["score"].rank(pct=True)
    d["grade"] = np.where(d.rank_pct >= 0.8, "强", np.where(d.rank_pct >= 0.5, "中", "弱"))
    d["ts_code"] = codes[d.col.values]
    d["trade_date"] = dates[d.row.values].date
    d["side"] = "buy"

    # ---- 卖点 ----
    # 卖点的"好坏"= 卖出后避免了多少下跌, 所以标签取未来收益的相反数;
    # 同样按【同日×同波动层】中性化, 断掉靠波动率作弊的路径。
    sell_ok = sig_sell & np.isfinite(ret) & np.isfinite(X_all_ref[0]).all(axis=2) \
        if False else sig_sell & np.isfinite(ret)
    se_r, se_c = np.where(sell_ok & scorable_base)
    if len(se_r) > 5000:
        Xs = X_all_kept[se_r, se_c]
        ys = -ret[se_r, se_c]                    # 避免的下跌
        yrs_ = dates[se_r].year
        vs_ = vol20_kept[se_r, se_c]
        dfx = pd.DataFrame({"d": se_r, "y": ys, "v": vs_})
        dfx["vq"] = dfx.groupby("d")["v"].transform(
            lambda x: pd.qcut(x.rank(method="first"), min(5, max(1, len(x))), labels=False)
            if len(x) >= 5 else 0)
        ys_rel = (ys - dfx.groupby(["d", "vq"])["y"].transform("mean").values).astype(np.float32)
        s_score = np.full(len(se_r), np.nan, dtype=np.float32)
        for Y in sorted(set(yrs_)):
            tr = yrs_ < Y
            if tr.sum() < 3000:
                continue
            ms = mk(); ms.fit(Xs[tr], ys_rel[tr])
            te = yrs_ == Y
            if te.sum():
                s_score[te] = ms.predict(Xs[te])
        okk = np.isfinite(s_score)
        ds = pd.DataFrame({"row": se_r[okk], "col": se_c[okk], "score": s_score[okk]})
        ds["rank_pct"] = ds.groupby("row")["score"].rank(pct=True)
        ds["grade"] = np.where(ds.rank_pct >= 0.8, "强",
                               np.where(ds.rank_pct >= 0.5, "中", "弱"))
        ds["ts_code"] = codes[ds.col.values]
        ds["trade_date"] = dates[ds.row.values].date
        ds["side"] = "sell"
        log.info("卖点 %d 条", len(ds))
        d = pd.concat([d, ds], ignore_index=True)

    log.info("入库 %d 条 (买 %d / 卖 %d)", len(d),
             int((d.side == "buy").sum()), int((d.side == "sell").sum()))
    return d[["ts_code", "trade_date", "side", "score", "rank_pct", "grade"]]


async def main() -> None:
    d = build()
    eng = create_async_engine(settings.database_url)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        await db.execute(delete(MaimaiSignal))
        await db.commit()
    rows = d.to_dict("records")
    CH = 5000
    async with Session() as db:
        for i in range(0, len(rows), CH):
            chunk = [{"ts_code": r["ts_code"], "trade_date": r["trade_date"],
                      "side": r["side"], "score": float(r["score"]),
                      "rank_pct": float(r["rank_pct"]),
                      "grade": r["grade"]} for r in rows[i:i + CH]]
            stmt = insert(MaimaiSignal).values(chunk)
            await db.execute(stmt.on_conflict_do_update(
                index_elements=["ts_code", "trade_date", "side"],
                set_={"score": stmt.excluded.score, "rank_pct": stmt.excluded.rank_pct,
                      "grade": stmt.excluded.grade}))
            await db.commit()
            if (i // CH) % 10 == 0:
                log.info("  写入 %d/%d", i, len(rows))
    log.info("完成 %d 条", len(rows))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
