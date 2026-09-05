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

    # 面板重建 → 三个指标重算。
    # 目前直接调 build_*(全量), 因为单次几分钟可接受, 且能保证
    # 特征的 rolling 窗口(最长60日)完整 —— 增量算特征要回看60天,
    # 复杂度和全量差不多, 不值得为此引入两套代码路径。
    from app.commands import build_didian, build_maimai, build_pump

    for name, mod in [("买卖很准", build_maimai), ("主力吸筹", build_pump),
                      ("低点组合", build_didian)]:
        try:
            log.info("--- %s 重算 ---", name)
            await mod.main()
        except Exception as exc:  # noqa: BLE001
            log.exception("%s 重算失败: %s", name, exc)

    eng = create_async_engine(settings.database_url)
    async with eng.connect() as c:
        for tbl, lbl in [("maimai_signal", "买卖很准"), ("pump_signal", "主力吸筹"),
                         ("didian_signal", "低点组合")]:
            r = (await c.execute(text(
                f"select count(*), max(trade_date) from {tbl}"))).fetchone()
            log.info("%s: %s 条, 最新 %s", lbl, r[0], r[1])
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
