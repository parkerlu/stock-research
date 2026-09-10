"""起爆模型每日全市场打分 —— 写 boom_score 表。

这是目前【通过全部防伪检验】的最好策略（2026-09-10）:
    标签: 次日开盘买入, 20 个交易日内最高价触及 +30% 记涨 / 先碰 −8% 记跌
    特征: 筹码 13 + 换手率 9 = 22 个
    walk-forward 滚动选参 -> 比值 2.11, 六年零负年, 六年都选中 MA10 择时
    加流动性约束(20日均额 >2000万, 92% 信号满足) -> 比值 2.31 / 年化 +41.9% / 回撤 −18.2%

⚠️ 四道防伪检验都过了（形态模型 v4 的假 1.70 就是没做这些）:
   1. 随机对照: 同规则同仓位随机选票只有 0.43 -> 2.x 是选股挣的, 不是出场规则凑的
   2. 一字涨停: 买入日开盘涨停仅 0.16%（v4 当年是 22%, 那批买不到的票贡献了全部收益）
   3. 流动性: 信号的 20 日均额中位 5717 万, 86% 在 2000 万以上
   4. walk-forward: 参数逐年滚动选, 每个测试点都是真样本外

⚠️ 为什么它比 +10% 那版好: 方向判别力从 +2.3pt 提到 +4.3pt。
   起爆前的形态比"涨10%前的形态"更有特征 —— 后者太普通, 模型学不出方向。

⚠️ 择时是这个策略的一部分, 不是可选项:
   不择时 1.15 / MA10 择时 3.37。起爆票是高波动小盘股, 与中证1000 高度同步,
   所以大盘开关对它特别灵（与"突破预警"当年的发现一致）。
   ⚠️ 但择时在信号层【不】体现 —— 这里只管打分, 择时在 sync_boom_signals 里做。

⚠️ 特征算法必须走 scripts.chips.features —— 与训练同一份实现。
   本项目"同一套东西抄两遍"已经栽过五次。

⚠️ 模型与特征都在挂载卷 /app/data/research/。放 /app 别处重建镜像就没了。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models.schema import BoomScore
from scripts.chips.features import (CYQ_COLS, DB_COLS, FEAT_NAMES, TURNOVER_NAMES,
                                    add_turnover_feats, bad_rows, chip_feats)

log = logging.getLogger("boom_score")
MDIR = "/app/data/research/chips"
RES = "/app/data/research"
PREFIX = "model_boom30_prod"
SEEDS = 3
UP, DN, COST = 0.30, 0.08, 0.003      # ⚠️ 必须与训练标签 label3_boom30 一致
WARMUP_DAYS = 90                       # 日历日; 换手率最长窗口 60 个交易日
CH = 3000
ALL_NAMES = FEAT_NAMES + TURNOVER_NAMES


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--date", default=None, help="只打某一天, 用来对账")
    ap.add_argument("--until", default=None,
                    help="从这一天【往前】数 --days 个交易日 —— 分段补历史用。"
                         "⚠️ 一次性 --days 800 以上会 OOM(三表合并 500 万行), "
                         "补长历史必须分段: --until 2025-01-14 --days 400 这样倒着推")
    a = ap.parse_args()

    models = []
    for si in range(SEEDS):
        m = xgb.XGBClassifier()
        m.load_model(f"{MDIR}/{PREFIX}_s{si}.json")
        models.append(m)
    cols = list(models[0].get_booster().feature_names or [])
    if cols != ALL_NAMES:
        raise SystemExit(f"模型特征名与 features 不一致: 模型 {len(cols)} 个, "
                         f"代码 {len(ALL_NAMES)} 个")
    log.info("载入 %d 个种子模型, 特征 %d 个", len(models), len(cols))

    eng = create_async_engine(settings.database_url)
    async with eng.connect() as c:
        end = (await c.execute(text("select max(trade_date) from daily_candle"))).scalar()
        if a.until:
            end = date.fromisoformat(a.until)
        # ⚠️ ST 在算分位【之前】剔除, 不是展示时过滤 —— 否则 Top1% 被 ST 占掉
        #    (筹码模型上实测过 32.7%), 真正可买的只剩三分之二
        st = {r[0] for r in (await c.execute(text(
            "select ts_code from stock_basic where name like '%ST%'"))).fetchall()}
    days_back = 1 if a.date else a.days
    start = end - timedelta(days=WARMUP_DAYS + int(days_back * 1.6) + 5)
    log.info("排除 ST %d 只; 取数 %s ~ %s", len(st), start, end)

    cyq = pd.read_parquet(CYQ_PATH := f"{RES}/cyq.parquet",
                          columns=["ts_code", "trade_date"] + CYQ_COLS)
    cyq["trade_date"] = pd.to_datetime(cyq["trade_date"]).dt.date
    cyq = cyq[(cyq["trade_date"] >= start) & (~cyq["ts_code"].isin(st))]
    db = pd.read_parquet(f"{RES}/daily_basic.parquet",
                         columns=["ts_code", "trade_date"] + DB_COLS)
    db["trade_date"] = pd.to_datetime(db["trade_date"]).dt.date
    db = db[db["trade_date"] >= start]
    log.info("筹码 %s 行, daily_basic %s 行", f"{len(cyq):,}", f"{len(db):,}")
    if cyq.empty or db.empty:
        raise SystemExit("筹码或 daily_basic 为空 —— 检查每日增量任务")

    codes = sorted(set(cyq["ts_code"]) & set(db["ts_code"]))
    parts = []
    for i in range(0, len(codes), 400):
        async with eng.connect() as c:
            rows = (await c.execute(text("""
                select ts_code, trade_date, close*adj_factor
                from daily_candle
                where ts_code = any(:cs) and trade_date between :a and :b
                order by ts_code, trade_date
            """), {"cs": codes[i:i + 400], "a": start, "b": end})).fetchall()
        if rows:
            parts.append(pd.DataFrame(rows, columns=["ts_code", "trade_date", "close"]))
    px = pd.concat(parts, ignore_index=True)
    px["close"] = pd.to_numeric(px["close"], errors="coerce").astype("float64")

    df = (px.merge(cyq, on=["ts_code", "trade_date"], how="inner")
            .merge(db, on=["ts_code", "trade_date"], how="inner")
            .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))
    df["gid"] = df["ts_code"]
    log.info("三表合并后 %s 行, %d 只", f"{len(df):,}", df.ts_code.nunique())

    X = pd.concat([chip_feats(df, "gid"), add_turnover_feats(df, "gid")], axis=1)[ALL_NAMES]
    good = np.isfinite(X.to_numpy()).all(1) & ~bad_rows(df).to_numpy()

    if a.date:
        want = [date.fromisoformat(a.date)]
    else:
        want = sorted(df["trade_date"].unique())[-a.days:]
    sel = df["trade_date"].isin(want).to_numpy() & good
    log.info("待打分 %s 行, %d 个交易日 (%s ~ %s)",
             f"{int(sel.sum()):,}", len(want), want[0], want[-1])
    if sel.sum() == 0:
        raise SystemExit("没有可打分的行 —— 检查三张表的日期是否对得上")

    p = np.mean([m.predict_proba(X[sel]) for m in models], axis=0)
    out = df.loc[sel, ["ts_code", "trade_date"]].copy()
    out["p_up"] = p[:, 1]
    out["p_dn"] = p[:, 2]
    out["ev"] = UP * out.p_up - DN * out.p_dn - COST
    out["rank_pct"] = out.groupby("trade_date")["ev"].rank(pct=True)

    # ⚠️ 分批【构造】再写, 不要一次性做出整个 payload ——
    #    200 万行 × 6 字段的 dict 列表约占 1GB, 补长历史时会 OOM(实测 exit 137)。
    #    现在只在每一批时把那 3000 行转成 dict, 峰值内存与批大小成正比。
    n_written = 0
    recs = out[["ts_code", "trade_date", "p_up", "p_dn", "ev", "rank_pct"]]
    async with eng.begin() as c:
        for i in range(0, len(recs), CH):
            chunk = recs.iloc[i:i + CH]
            batch = [{"ts_code": t, "trade_date": d,
                      "p_up": round(float(u), 5), "p_dn": round(float(w), 5),
                      "ev": round(float(e), 6), "rank_pct": round(float(rk), 5)}
                     for t, d, u, w, e, rk in chunk.itertuples(index=False, name=None)]
            st_ = pg_insert(BoomScore).values(batch)
            await c.execute(st_.on_conflict_do_update(
                index_elements=["ts_code", "trade_date"],
                set_={"p_up": st_.excluded.p_up, "p_dn": st_.excluded.p_dn,
                      "ev": st_.excluded.ev, "rank_pct": st_.excluded.rank_pct}))
            n_written += len(batch)
        n = (await c.execute(text(
            "select count(*), min(trade_date), max(trade_date) from boom_score"))).fetchone()
    log.info("写入 %s 条; 表内合计 %s, %s ~ %s",
             f"{n_written:,}", f"{n[0]:,}", n[1], n[2])
    last = out[out.trade_date == want[-1]]
    top = last[last.rank_pct >= 0.99].nlargest(8, "ev")
    log.info("%s 的 Top: %s", want[-1], ", ".join(
        f"{r.ts_code}(P起爆{r.p_up * 100:.0f}%)" for r in top.itertuples()))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
