import calendar
import logging
import time
from datetime import date, timedelta
from itertools import groupby

from sqlalchemy import func, select, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.datasources.manager import DataSourceManager
from app.models.schema import DailyCandle

log = logging.getLogger(__name__)

# 全市场最新交易日 —— 判断"这只票的K线是不是落后了"的基准。
# 用全市场而不是日历: 不必维护交易日历, 也天然躲开周末和长假
# (休市日全市场的 max 不会前进, 于是不会去空拉一遍)。
_MKT_DATE: tuple[date | None, float] = (None, 0.0)
_MKT_TTL = 300.0

# 已经尝试过但源里确实没有新数据的 (ts_code, 基准日) —— 停牌票会永远
# 落后于全市场, 不记下来的话每次开图都要白跑一次数据源。
_TRIED: set[tuple[str, date]] = set()


async def _market_last_date(db: AsyncSession) -> date | None:
    global _MKT_DATE
    d, at = _MKT_DATE
    if d is not None and time.monotonic() - at < _MKT_TTL:
        return d
    d = (await db.execute(select(func.max(DailyCandle.trade_date)))).scalar()
    _MKT_DATE = (d, time.monotonic())
    return d


def _iso_week_key(d: date) -> tuple[int, int]:
    """Return (iso_year, iso_week) for grouping."""
    cal = d.isocalendar()
    return (cal[0], cal[1])


def _month_key(d: date) -> tuple[int, int]:
    return (d.year, d.month)


def _aggregate(rows: list[dict], key_fn) -> list[dict]:
    """Generic aggregation: group rows by key_fn, produce OHLCV bars."""
    sorted_rows = sorted(rows, key=lambda r: r["trade_date"])
    result = []
    for _, group in groupby(sorted_rows, key=lambda r: key_fn(r["trade_date"])):
        candles = list(group)
        bar = {
            "timestamp": calendar.timegm(candles[0]["trade_date"].timetuple()) * 1000,
            "open": candles[0]["open"],
            "close": candles[-1]["close"],
            "high": max(c["high"] for c in candles),
            "low": min(c["low"] for c in candles),
            "volume": sum(c["vol"] for c in candles),
            "amount": sum(c["amount"] for c in candles),
        }
        result.append(bar)
    return result


def aggregate_weekly(rows: list[dict]) -> list[dict]:
    return _aggregate(rows, _iso_week_key)


def aggregate_monthly(rows: list[dict]) -> list[dict]:
    return _aggregate(rows, _month_key)


async def get_candles(
    db: AsyncSession,
    manager: DataSourceManager,
    ts_code: str,
    tf: str,
    start: date,
    end: date,
) -> list[dict]:
    """Fetch candles with cache-through. Returns list of dicts for JSON response."""
    # 1. Check what's cached
    stmt = (
        select(DailyCandle)
        .where(and_(DailyCandle.ts_code == ts_code, DailyCandle.trade_date >= start, DailyCandle.trade_date <= end))
        .order_by(DailyCandle.trade_date)
    )
    result = await db.execute(stmt)
    cached = result.scalars().all()
    cached_dates = {c.trade_date for c in cached}

    # 2. If no cached data, fetch full range from source
    if not cached_dates:
        df = await manager.fetch_daily(ts_code, start, end)
        if not df.empty:
            rows = df.to_dict("records")
            stmt_upsert = pg_insert(DailyCandle).values(rows)
            stmt_upsert = stmt_upsert.on_conflict_do_nothing(index_elements=["ts_code", "trade_date"])
            await db.execute(stmt_upsert)
            await db.commit()
            result = await db.execute(
                select(DailyCandle)
                .where(and_(DailyCandle.ts_code == ts_code, DailyCandle.trade_date >= start, DailyCandle.trade_date <= end))
                .order_by(DailyCandle.trade_date)
            )
            cached = result.scalars().all()
    else:
        # 3. Check if we need incremental update (forward and backward)
        first_cached = min(cached_dates)
        last_cached = max(cached_dates)
        need_reload = False

        # Backward fill: fetch earlier history if requested range starts before cache
        # Skip if gap is small (<=5 days covers weekends/holidays)
        if first_cached > start and (first_cached - start).days > 5:
            try:
                fetch_end = first_cached - timedelta(days=1)
                df = await manager.fetch_daily(ts_code, start, fetch_end)
                if not df.empty:
                    rows = df.to_dict("records")
                    stmt_upsert = pg_insert(DailyCandle).values(rows)
                    stmt_upsert = stmt_upsert.on_conflict_do_nothing(index_elements=["ts_code", "trade_date"])
                    await db.execute(stmt_upsert)
                    need_reload = True
            except Exception:
                pass  # Use existing cache if data source fails

        # 前向补数 —— 只要落后于【全市场最新交易日】就补, 不留容差。
        # ⚠️ 原本的判据是 (end - last_cached).days > 5, 想躲开周末假期,
        #    副作用是差 1~4 天时什么都不做: 开图看到的就是旧K线, 而且没有
        #    任何提示。改用全市场基准后, 休市日 mkt 不前进 -> 不会空拉,
        #    开市日只要缺一根就补, 两个目的不再打架。
        mkt = await _market_last_date(db)
        stale = mkt is not None and last_cached < mkt and end >= mkt
        if stale and (ts_code, mkt) not in _TRIED:
            try:
                # ⚠️ 往前多要 10 天: 只请求缺的那一两天会被数据源掐断连接
                #    (实测 9-04~9-07 报 RemoteDisconnected, 9-01 起同一只票正常)。
                #    重叠部分走 on_conflict_do_nothing, 多取无害。
                fetch_start = last_cached - timedelta(days=10)
                df = await manager.fetch_daily(ts_code, fetch_start, end)
                if not df.empty:
                    rows = df.to_dict("records")
                    stmt_upsert = pg_insert(DailyCandle).values(rows)
                    stmt_upsert = stmt_upsert.on_conflict_do_nothing(index_elements=["ts_code", "trade_date"])
                    await db.execute(stmt_upsert)
                    need_reload = True
                    log.info("补K线 %s: %s ~ %s, %d 根", ts_code, fetch_start, end, len(rows))
                else:
                    # 停牌 / 已退市 —— 记下来, 同一个基准日内不再重试
                    _TRIED.add((ts_code, mkt))
            except Exception as e:
                _TRIED.add((ts_code, mkt))
                log.warning("补K线失败 %s: %s", ts_code, e)

        if need_reload:
            await db.commit()
            result = await db.execute(
                select(DailyCandle)
                .where(and_(DailyCandle.ts_code == ts_code, DailyCandle.trade_date >= start, DailyCandle.trade_date <= end))
                .order_by(DailyCandle.trade_date)
            )
            cached = result.scalars().all()

    # 4. Apply adj_factor for backward-adjusted prices.
    # IMPORTANT: latest_adj must be the GLOBAL latest for this symbol, not the
    # latest within the requested date range — otherwise charts show different
    # adjusted prices for the same bar depending on what window is queried.
    latest_row = (await db.execute(
        select(DailyCandle.adj_factor)
        .where(DailyCandle.ts_code == ts_code)
        .order_by(DailyCandle.trade_date.desc())
        .limit(1)
    )).scalar_one_or_none()
    latest_adj = float(latest_row) if latest_row else 1.0

    daily_rows = []
    for c in cached:
        adj = float(c.adj_factor) if c.adj_factor else 1.0
        factor = adj / latest_adj if latest_adj != 0 else 1.0
        daily_rows.append({
            "trade_date": c.trade_date,
            "open": round(float(c.open) * factor, 4),
            "high": round(float(c.high) * factor, 4),
            "low": round(float(c.low) * factor, 4),
            "close": round(float(c.close) * factor, 4),
            "vol": c.vol,
            "amount": float(c.amount),
        })

    # 4.5 今天这根 —— 盘中从实时快照拼出来。
    # ⚠️ 历史接口在收盘前【没有】当天的日线(实测源里最新只到上一个交易日),
    #    所以光靠 fetch_daily 补不出来: 盘中切过去看到的永远是昨天收盘那根。
    # ⚠️ 只出现在返回值里, 不写库 —— 这根没走完, 落库会被 on_conflict_do_nothing
    #    永久冻住半截数据, 而所有训练指标都读这张表。
    # 前复权系数用的是该股最新的 adj_factor, 最新那根的 factor 恒为 1,
    # 所以快照的原始价与调整后的历史序列本来就在同一标度上, 不用再换算。
    today = date.today()
    if daily_rows and end >= today and daily_rows[-1]["trade_date"] < today:
        try:
            snap = await manager.fetch_snapshot(ts_code)
        except Exception:
            snap = None
        qt = (snap or {}).get("quote_time")
        qdate = None
        if isinstance(qt, str) and len(qt) >= 10:
            try:
                qdate = date.fromisoformat(qt[:10])
            except ValueError:
                qdate = None
        elif hasattr(qt, "date"):
            qdate = qt.date()
        # quote_time 必须是今天: 周末/假日快照给的是上一个交易日的数据,
        # 照拼会平白多出一根重复的K线。
        # ⚠️ 必须要求 vol > 0: 盘前(9:20)快照给的是昨收、量为 0, 照拼会画出
        #    一根 开=高=低=收 的平K线, 看着像今天没波动。停牌同理。
        if (snap and qdate == today and (snap.get("price") or 0) > 0
                and (snap.get("vol") or 0) > 0):
            px = float(snap["price"])
            daily_rows.append({
                "trade_date": today,
                "open": round(float(snap.get("open") or px), 4),
                "high": round(float(snap.get("high") or px), 4),
                "low": round(float(snap.get("low") or px), 4),
                "close": round(px, 4),
                "vol": int(snap.get("vol") or 0),
                # 快照的成交额是元, daily_candle.amount 是千元
                "amount": float(snap.get("amount") or 0) / 1000.0,
            })

    # 5. Aggregate if needed
    if tf == "1w":
        return aggregate_weekly(daily_rows)
    elif tf == "1m":
        return aggregate_monthly(daily_rows)

    return [
        {
            "timestamp": calendar.timegm(r["trade_date"].timetuple()) * 1000,
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["vol"],
            "amount": r["amount"],
        }
        for r in daily_rows
    ]
