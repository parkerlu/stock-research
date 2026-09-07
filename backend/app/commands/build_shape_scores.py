"""形态模型每日全市场打分 —— 供各策略当【排除过滤器】用。

⚠️ 用途仅限过滤, 不是选股信号。带成交约束的组合比值只有 0.24;
   有价值的是空头端 —— rank_pct < 0.2 的那批六年一致跑输。
   叠在周线版上: 胜率 45.8% -> 49.7%, 比值 0.19 -> 0.22。

⚠️ 打分必须做【当日横截面排名】才有意义。模型输出的绝对分数没有可比性,
   全市场一起排名后的分位才是"这只票今天在所有票里排第几"。
   所以这个命令一次要算全市场, 不能只算单只。

⚠️ 特征最长窗口 120 日(vol_pct120), 所以要多取约 200 个日历日的历史,
   否则新算出来的特征是 NaN, 打分全废。
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
from app.models.schema import ShapeScore
from scripts.shape.build_features import _feats

log = logging.getLogger("shape_score")
MODEL = "/app/data/research/shape/shape_v5_y_0.15_0.08_h20.json"
WARMUP_DAYS = 260          # 日历日, 覆盖 120 个交易日的特征窗口
CH = 3000                  # PostgreSQL 绑定参数上限 32767


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, help="回补最近几个交易日")
    a = ap.parse_args()

    m = xgb.XGBRegressor()
    m.load_model(MODEL)
    cols = list(m.get_booster().feature_names or [])
    if not cols:
        raise SystemExit("模型里没有特征名, 无法保证列顺序一致")

    eng = create_async_engine(settings.database_url)
    async with eng.connect() as c:
        end = (await c.execute(text("select max(trade_date) from daily_candle"))).scalar()
        codes = [r[0] for r in (await c.execute(text(
            "select distinct ts_code from daily_candle where trade_date > :s"),
            {"s": end - timedelta(days=30)})).fetchall()]
    start = end - timedelta(days=WARMUP_DAYS)
    log.info("全市场 %d 只, 取 %s ~ %s 算特征", len(codes), start, end)

    parts = []
    for i in range(0, len(codes), 500):
        async with eng.connect() as c:
            rows = (await c.execute(text("""
                select ts_code, trade_date, open*adj_factor, high*adj_factor,
                       low*adj_factor, close*adj_factor, vol, amount
                from daily_candle
                where ts_code = any(:cs) and trade_date between :a and :b
                order by ts_code, trade_date
            """), {"cs": codes[i:i + 500], "a": start, "b": end})).fetchall()
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["ts_code", "trade_date", "o", "h",
                                         "l", "c", "v", "amt"])
        for col in ("o", "h", "l", "c", "v", "amt"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df = df.dropna(subset=["o", "h", "l", "c"])
        for _, g in df.groupby("ts_code", sort=False):
            if len(g) > 130:
                parts.append(_feats(g.reset_index(drop=True)))
    feat = pd.concat(parts, ignore_index=True)

    # 只给最近 N 个交易日打分
    days = sorted(feat["trade_date"].unique())[-a.days:]
    feat = feat[feat["trade_date"].isin(days)].copy()
    feat = feat.dropna(subset=cols[:1])
    log.info("待打分 %s 行, 覆盖 %d 个交易日", f"{len(feat):,}", len(days))

    # ⚠️ 先做当日横截面分位, 再喂模型 —— 训练时就是这么做的, 顺序不能变
    g = feat.groupby("trade_date", sort=False)
    X = pd.DataFrame(index=feat.index)
    for col in cols:
        X[col] = g[col].rank(pct=True).astype(np.float32)
    feat["score"] = m.predict(X[cols]).astype(np.float32)
    feat["rank_pct"] = feat.groupby("trade_date")["score"].rank(pct=True)

    payload = [{"ts_code": r.ts_code, "trade_date": r.trade_date,
                "score": round(float(r.score), 6),
                "rank_pct": round(float(r.rank_pct), 5)}
               for r in feat.itertuples()]
    async with eng.begin() as c:
        for i in range(0, len(payload), CH):
            st = pg_insert(ShapeScore).values(payload[i:i + CH])
            await c.execute(st.on_conflict_do_update(
                index_elements=["ts_code", "trade_date"],
                set_={"score": st.excluded.score,
                      "rank_pct": st.excluded.rank_pct}))
        n = (await c.execute(text(
            "select count(*), min(trade_date), max(trade_date) from shape_score"))).fetchone()
    log.info("写入 %s 条; 表内合计 %s, %s ~ %s",
             f"{len(payload):,}", f"{n[0]:,}", n[1], n[2])
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
