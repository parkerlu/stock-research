"""训练指标每日增量更新 —— 收盘后跑, 只算最近几天。

用法: python -m app.commands.update_indicators [--days 10]

与 build_* 的区别:
  build_*        全历史重建 + walk-forward 逐年重训, 几分钟, 用于首次/重训
  update(本命令)  只拉当日新数据 + 用【已保存的模型】给最近 N 天打分, 秒级

⚠️ 模型不在这里重训。重训要拿新一年的完整数据做 walk-forward, 每天重训既慢
又会让评分口径天天变(同一个信号今天 0.7 明天 0.6), 没法对照。模型按年重训即可。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DidianSignal, MaimaiSignal, PumpSignal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("update_indicators")

RESEARCH = "/app/data/research"
MODEL_DIR = f"{RESEARCH}/models"


def fetch_cyq_recent(days: int) -> int:
    """拉最近几个交易日的筹码分布, 追加进 cyq.parquet。"""
    import tushare as ts

    pro = ts.pro_api(settings.tushare_token)
    path = f"{RESEARCH}/cyq.parquet"
    old = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    have = set(old.trade_date.dt.strftime("%Y%m%d")) if len(old) else set()

    end = date.today()
    cal = pro.trade_cal(exchange="SSE",
                        start_date=(end - timedelta(days=days * 2)).strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"), is_open="1")
    want = [d for d in sorted(cal.cal_date.tolist())[-days:] if d not in have]
    if not want:
        log.info("筹码数据已是最新")
        return 0
    rows = []
    for d in want:
        for _ in range(4):
            try:
                x = pro.cyq_perf(trade_date=d)
                if x is not None and len(x):
                    rows.append(x)
                break
            except Exception as e:  # noqa: BLE001
                if "超限" in str(e) or "频率" in str(e):
                    import time
                    time.sleep(12)
                    continue
                break
    if not rows:
        return 0
    new = pd.concat(rows, ignore_index=True)
    for c in ["his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct",
              "cost_85pct", "cost_95pct", "weight_avg", "winner_rate"]:
        new[c] = pd.to_numeric(new[c], errors="coerce").astype(np.float32)
    new["trade_date"] = pd.to_datetime(new["trade_date"])
    out = pd.concat([old, new], ignore_index=True).drop_duplicates(
        ["ts_code", "trade_date"], keep="last").sort_values(["ts_code", "trade_date"])
    out.to_parquet(path, index=False)
    log.info("筹码新增 %d 行 (%s)", len(new), ",".join(want))
    return len(new)


def fetch_top_list_recent(days: int) -> int:
    """拉最近几个交易日的龙虎榜, 直接写进 top_list 表。

    ⚠️ 这一步不能省: v5 要求信号前 7 日内有龙虎榜净买入, 龙虎榜停更超过一周,
    v5 就再也出不了新信号 —— 而且是静默的, 界面上只表现为"最近没信号"。
    (v3.5 就是因为没接进每日更新, 悄悄过期了才被发现。)
    """
    import time

    import tushare as ts
    from sqlalchemy import create_engine, text as _text

    pro = ts.pro_api(settings.tushare_token)
    sync_url = settings.database_url.replace("+asyncpg", "").replace("+psycopg", "")
    eng = create_engine(sync_url)
    with eng.begin() as conn:
        have = {r[0].strftime("%Y%m%d") for r in conn.execute(_text(
            "select distinct trade_date from top_list "
            "where trade_date >= current_date - :d"), {"d": days * 2})}

    end = date.today()
    cal = pro.trade_cal(exchange="SSE",
                        start_date=(end - timedelta(days=days * 2)).strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"), is_open="1")
    want = [d for d in sorted(cal.cal_date.tolist())[-days:] if d not in have]
    if not want:
        log.info("龙虎榜已是最新")
        return 0

    rows = []
    for d in want:
        for _ in range(4):
            try:
                x = pro.top_list(trade_date=d)
                if x is not None and len(x):
                    rows.append(x)
                break
            except Exception as e:  # noqa: BLE001
                if "超限" in str(e) or "频率" in str(e):
                    time.sleep(12)
                    continue
                log.warning("龙虎榜 %s 拉取失败: %s", d, e)
                break
    if not rows:
        return 0

    new = pd.concat(rows, ignore_index=True)
    keep = ["ts_code", "trade_date", "close", "pct_change", "turnover_rate",
            "l_buy", "l_sell", "net_amount", "reason"]
    new = new[[c for c in keep if c in new.columns]].copy()
    for c in ["close", "pct_change", "turnover_rate", "l_buy", "l_sell", "net_amount"]:
        if c in new:
            new[c] = pd.to_numeric(new[c], errors="coerce")
    new["trade_date"] = pd.to_datetime(new["trade_date"]).dt.date
    # 同一票同一天可能有多条(不同上榜原因), 合并成一条: 金额相加, 原因拼接
    new = (new.groupby(["ts_code", "trade_date"], as_index=False)
              .agg({"close": "first", "pct_change": "first", "turnover_rate": "first",
                    "l_buy": "sum", "l_sell": "sum", "net_amount": "sum",
                    "reason": lambda x: "; ".join(sorted(set(map(str, x))))[:200]}))

    with eng.begin() as conn:
        for r in new.to_dict("records"):
            conn.execute(_text("""
                insert into top_list (ts_code, trade_date, close, pct_change,
                                      turnover_rate, l_buy, l_sell, net_amount, reason)
                values (:ts_code, :trade_date, :close, :pct_change, :turnover_rate,
                        :l_buy, :l_sell, :net_amount, :reason)
                on conflict (ts_code, trade_date) do update set
                  net_amount = excluded.net_amount, l_buy = excluded.l_buy,
                  l_sell = excluded.l_sell, reason = excluded.reason
            """), r)
    log.info("龙虎榜新增 %d 行 (%s)", len(new), ",".join(want))
    return len(new)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10, help="回溯几个交易日")
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()

    if not args.skip_fetch:
        try:
            fetch_cyq_recent(args.days)
        except Exception as exc:  # noqa: BLE001
            log.warning("筹码拉取失败(继续): %s", exc)
        try:
            fetch_top_list_recent(args.days)
        except Exception as exc:  # noqa: BLE001
            log.warning("龙虎榜拉取失败(继续): %s", exc)
        # 指数 —— 突破预警的择时基准。不更新的话基准停在导入那天, 而"停更"
        # 在界面上和"大盘不好"长得一样, 会静默把信号全挡掉。
        try:
            from app.commands.sync_index import main as sync_index
            import sys as _s
            _a = _s.argv; _s.argv = ["sync_index", "--start", "20250101"]
            try:
                await sync_index()
            finally:
                _s.argv = _a
        except Exception as exc:  # noqa: BLE001
            log.warning("指数拉取失败(继续): %s", exc)

    # 面板重建 → 三个指标重算。
    # 目前直接调 build_*(全量), 因为单次几分钟可接受, 且能保证
    # 特征的 rolling 窗口(最长60日)完整 —— 增量算特征要回看60天,
    # 复杂度和全量差不多, 不值得为此引入两套代码路径。
    from app.commands import (build_breakout, build_didian, build_dongli,
                              build_maimai, build_maimai_weekly, build_pump,
                              build_sar)

    # ⚠️ build_dongli 必须在这里跑: 拉升预警 = 动力线 × 吸筹, 动力线不更新
    # 就再也出不了新信号(v3.5 和龙虎榜都因为漏接每日更新静默过期过)。
    for name, mod in [("买卖很准", build_maimai), ("主力吸筹", build_pump),
                      ("低点组合", build_didian), ("动力线", build_dongli),
                      ("突破预警", build_breakout),
                      ("SAR预警", build_sar),
                      ("买卖很准周线", build_maimai_weekly)]:
        try:
            log.info("--- %s 重算 ---", name)
            await mod.main()
        except Exception as exc:  # noqa: BLE001
            log.exception("%s 重算失败: %s", name, exc)

    # 形态模型每日打分 + 各策略的信号同步。
    # ⚠️ 顺序不能颠倒: 打分要先有, mmweek_f(过滤版)才 join 得上 shape_score。
    #    build_shape_scores 依赖上面刚算完的 daily_candle, 也依赖挂载卷上的
    #    模型文件 /app/data/research/shape/*.json —— 那个目录必须是挂载的,
    #    放 /app 别处重建镜像就没了。
    from app.commands import (build_shape_scores, sync_breakout_signals,
                              sync_mmweek_signals, sync_sar_signals,
                              sync_shape_signals)
    import sys as _sys

    async def _run(name, mod, argv):
        old_argv = _sys.argv
        try:
            _sys.argv = argv
            log.info("--- %s ---", name)
            await mod.main()
        except Exception as exc:  # noqa: BLE001
            log.exception("%s 失败: %s", name, exc)
        finally:
            _sys.argv = old_argv

    await _run("形态模型打分", build_shape_scores, ["x", "--days", "5"])
    await _run("突破预警信号", sync_breakout_signals, ["x"])
    await _run("SAR预警信号", sync_sar_signals, ["x"])
    await _run("周线版信号", sync_mmweek_signals, ["x"])
    await _run("周线版信号(过滤)", sync_mmweek_signals, ["x", "--filtered"])
    # 形态模型自己当日线策略 —— Top1%。回测判据没过(0.24), 建账户是为了
    # 向前验; 回测已经骗过我一次(v4 的 1.70 全来自买不到的一字涨停)。
    await _run("形态模型信号", sync_shape_signals, ["x"])

    eng = create_async_engine(settings.database_url)
    async with eng.connect() as c:
        r = (await c.execute(text(
            "select count(*), max(trade_date) from shape_score"))).fetchone()
        log.info("形态模型打分: %s 条, 最新 %s", r[0], r[1])
        for tbl, lbl in [("maimai_signal", "买卖很准"), ("pump_signal", "主力吸筹"),
                         ("didian_signal", "低点组合"), ("dongli_signal", "动力线"),
                         ("breakout_signal", "突破预警"),
                         ("sar_signal", "SAR预警"),
                         ("maimai_weekly", "买卖很准周线")]:
            r = (await c.execute(text(
                f"select count(*), max(trade_date) from {tbl}"))).fetchone()
            log.info("%s: %s 条, 最新 %s", lbl, r[0], r[1])
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
