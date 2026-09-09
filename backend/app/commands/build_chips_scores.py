"""筹码模型每日全市场打分 —— 写 chips_score 表。

与形态模型(build_shape_scores)的关键差别: 这个【能】当选股信号用。
    形态模型组合比值只有 0.24, 价值只在空头端当排除过滤器;
    筹码模型 0.89(walk-forward + 资金池 + 真实周转), 而且是【纯选股】——
    择时贡献为 0, 99% 的交易日都出信号, 所以回撤只有 −17.9%
    (裸K 那个 85% 的交易挤在 10% 的日子里, 回撤 −31%)。

⚠️ 特征算法必须走 scripts.chips.features.chip_feats —— 与训练同一份实现。
   本项目"同一套东西抄两遍"已经栽过五次; 特征算不一致的后果是模型拿到另一个
   分布, 结果变差【且不报错】。

⚠️ close 必须乘 adj_factor(后复权), 与训练侧 panel.npz 的口径一致。
   跨除权日时只有复权价的涨幅才是真实收益率, 而"获利盘背离"要减这个涨幅。
   但 cyq 的成本价是【原始价】, 所以绝不能拿 close 去除成本价 —— 那算出来的
   是 adj_factor 本身(2026-09-10 第一版就这么错过, 中位数 3.602 而应在 1~2)。

⚠️ 模型文件在挂载卷 /app/data/research/chips/。放 /app 别处重建镜像就没了。

⚠️ 必须算全市场: rank_pct 是当日横截面分位, 单只算没有意义。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models.schema import ChipsScore
from scripts.chips.features import CYQ_COLS, FEAT_NAMES, bad_rows, chip_feats

log = logging.getLogger("chips_score")
MDIR = "/app/data/research/chips"
CYQ = "/app/data/research/cyq.parquet"
SEEDS = 3
UP, DN, COST = 0.10, 0.08, 0.003
WARMUP_DAYS = 60           # 日历日; 特征最长窗口 20 个交易日
CH = 3000                  # PostgreSQL 绑定参数上限 32767


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, help="回补最近几个交易日")
    a = ap.parse_args()

    models = []
    for si in range(SEEDS):
        m = xgb.XGBClassifier()
        m.load_model(f"{MDIR}/model_chips_v1_s{si}.json")
        models.append(m)
    cols = list(models[0].get_booster().feature_names or [])
    if cols != FEAT_NAMES:
        raise SystemExit(f"模型特征名与 features.FEAT_NAMES 不一致: {cols[:3]}...")
    log.info("载入 %d 个种子模型, 特征 %d 个", len(models), len(cols))

    eng = create_async_engine(settings.database_url)
    async with eng.connect() as c:
        end = (await c.execute(text("select max(trade_date) from daily_candle"))).scalar()
        # ⚠️ ST 必须在【算分位之前】剔除, 不是在展示时过滤:
        #    rank_pct 是当日横截面分位, 如果 ST 还在池子里, Top1% 会被它们占掉
        #    (实测 32.7%!), 真正可买的只剩三分之二。模型偏好 ST 是因为它们
        #    波动大、获利盘变化剧烈, 正好是这套特征的强信号区。
        #    虚拟盘那边 paper_trading._universe_ok 的白名单已经挡住了 ST,
        #    所以账户结果不受影响; 但选股页和 strategy_signal 会被污染。
        # ⚠️ 这里用【当前】的 stock_basic.name 过滤: 对每日生产是正确的
        #    (算今天的分就该用今天的状态), 但回补历史时会有轻微幸存者偏差 ——
        #    2024 年才戴帽的票, 它 2023 年的分也被剔掉了。影响可控(ST 状态
        #    变化不频繁), 但回测【绝不能】这么做, 见 paper_trading.py:143。
        st = {r[0] for r in (await c.execute(text(
            "select ts_code from stock_basic where name like '%ST%'"))).fetchall()}
    log.info("排除 ST %d 只", len(st))
    # ⚠️ 取数窗口 = 打分范围 + 预热。只减 WARMUP 的话, --days 一旦超过窗口内的
    #    交易日数(约40), 早几天就【静默少打分】而不是报错。
    start = end - timedelta(days=WARMUP_DAYS + int(a.days * 1.6) + 5)

    cyq = pd.read_parquet(CYQ, columns=["ts_code", "trade_date"] + CYQ_COLS)
    cyq["trade_date"] = pd.to_datetime(cyq["trade_date"]).dt.date
    cyq = cyq[cyq["trade_date"] >= start]
    log.info("筹码 %s 行 (%s ~ %s)", f"{len(cyq):,}",
             cyq.trade_date.min(), cyq.trade_date.max())
    if cyq.empty:
        raise SystemExit("筹码数据为空 —— 检查 update_indicators 的 fetch_cyq_recent")

    # ⚠️ 按 ts_code 分批走主键索引。不能按 trade_date 范围查 ——
    #    索引是 (ts_code, trade_date), 按日期范围要全表扫描(实测卡死 10 分钟)
    cyq = cyq[~cyq["ts_code"].isin(st)]
    codes = sorted(cyq["ts_code"].unique())
    parts = []
    for i in range(0, len(codes), 500):
        async with eng.connect() as c:
            rows = (await c.execute(text("""
                select ts_code, trade_date, close*adj_factor
                from daily_candle
                where ts_code = any(:cs) and trade_date between :a and :b
                order by ts_code, trade_date
            """), {"cs": codes[i:i + 500], "a": start, "b": end})).fetchall()
        if rows:
            parts.append(pd.DataFrame(rows, columns=["ts_code", "trade_date", "close"]))
    px = pd.concat(parts, ignore_index=True)
    px["close"] = pd.to_numeric(px["close"], errors="coerce").astype("float64")
    log.info("日线 %s 行, %d 只", f"{len(px):,}", px.ts_code.nunique())

    df = px.merge(cyq, on=["ts_code", "trade_date"], how="inner")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["gid"] = df["ts_code"]
    log.info("合并后 %s 行, %d 只", f"{len(df):,}", df.ts_code.nunique())

    X = chip_feats(df, gid="gid")
    good = np.isfinite(X.to_numpy()).all(1) & ~bad_rows(df).to_numpy()

    days = sorted(df["trade_date"].unique())[-a.days:]
    sel = df["trade_date"].isin(days).to_numpy() & good
    log.info("待打分 %s 行, 覆盖 %d 个交易日 (%s ~ %s)",
             f"{int(sel.sum()):,}", len(days), days[0], days[-1])
    if sel.sum() == 0:
        raise SystemExit("没有可打分的行 —— 检查筹码与日线的日期是否对得上")

    Xs = X[sel]
    p = np.mean([m.predict_proba(Xs) for m in models], axis=0)
    out = df.loc[sel, ["ts_code", "trade_date"]].copy()
    out["p_up"] = p[:, 1]
    out["p_dn"] = p[:, 2]
    out["ev"] = UP * out.p_up - DN * out.p_dn - COST
    out["rank_pct"] = out.groupby("trade_date")["ev"].rank(pct=True)

    payload = [{"ts_code": r.ts_code, "trade_date": r.trade_date,
                "p_up": round(float(r.p_up), 5), "p_dn": round(float(r.p_dn), 5),
                "ev": round(float(r.ev), 6), "rank_pct": round(float(r.rank_pct), 5)}
               for r in out.itertuples()]
    async with eng.begin() as c:
        for i in range(0, len(payload), CH):
            st = pg_insert(ChipsScore).values(payload[i:i + CH])
            await c.execute(st.on_conflict_do_update(
                index_elements=["ts_code", "trade_date"],
                set_={"p_up": st.excluded.p_up, "p_dn": st.excluded.p_dn,
                      "ev": st.excluded.ev, "rank_pct": st.excluded.rank_pct}))
        n = (await c.execute(text(
            "select count(*), min(trade_date), max(trade_date) from chips_score"))).fetchone()
    log.info("写入 %s 条; 表内合计 %s, %s ~ %s",
             f"{len(payload):,}", f"{n[0]:,}", n[1], n[2])
    last = out[out.trade_date == days[-1]]
    top = last[last.rank_pct >= 0.99].nlargest(8, "ev")
    log.info("%s 的 Top: %%s" % days[-1], ", ".join(
        f"{r.ts_code}({r.ev * 100:+.2f}%)" for r in top.itertuples()))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
