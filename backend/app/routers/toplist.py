"""龙虎榜 —— 交易所每日公布的异动股买卖席位汇总。

数据来源 tushare `top_list`, 2018-01 起。⚠️ 覆盖率仅 1.2%(只有异动才上榜),
所以它不是选股工具, 是"这只票今天被谁买了"的旁证。

它在本项目里的另一个身份: v5 三重共振的第三层过滤。最有价值的一档是
**当日涨幅 −2~2% 却上了榜** —— 价格几乎没动却有大额资金进出, 与"获利盘背离"
同一逻辑, 但这是实名席位的硬记录, 不是从价格反推的。
"""
from __future__ import annotations

import math
from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db

router = APIRouter(prefix="/api/toplist", tags=["toplist"])


def _f(v, nd: int = 2, div: float = 1.0):
    """安全转 float —— PostgreSQL numeric 可以存 'NaN', 直接 json.dumps 会
    抛 'Out of range float values are not JSON compliant'(实测 turnover_rate)。"""
    if v is None:
        return None
    x = float(v)
    return round(x / div, nd) if math.isfinite(x) else None


@router.get("/days")
async def recent_days(limit: int = Query(30, ge=1, le=120),
                      db: AsyncSession = Depends(get_db)):
    """最近有龙虎榜的交易日 + 当日上榜家数/净买入合计。"""
    rows = (await db.execute(text("""
        select trade_date, count(*) n,
               sum(net_amount) net,
               count(*) filter (where net_amount > 0) buy_n
        from top_list group by trade_date order by trade_date desc limit :l
    """), {"l": limit})).fetchall()
    return {"days": [{"date": str(r[0]), "count": r[1],
                      "net_wan": _f(r[2], 1, 1e4) or 0.0,
                      "buy_count": r[3]} for r in rows]}


@router.get("/list")
async def top_list(
    day: str | None = Query(None, description="YYYY-MM-DD, 缺省取最新交易日"),
    side: str = Query("all", description="all / buy(净买入) / sell(净卖出)"),
    db: AsyncSession = Depends(get_db),
):
    """某一天的龙虎榜明细, 按净买入额降序。"""
    if day:
        try:
            d = date.fromisoformat(day)
        except ValueError:
            d = None
    else:
        d = None
    if d is None:
        d = (await db.execute(text("select max(trade_date) from top_list"))).scalar()

    cond = {"buy": "and t.net_amount > 0", "sell": "and t.net_amount < 0"}.get(side, "")
    rows = (await db.execute(text(f"""
        select t.ts_code, coalesce(b.name,'') name, t.close, t.pct_change,
               t.turnover_rate, t.l_buy, t.l_sell, t.net_amount, t.reason
        from top_list t
        left join stock_basic b on b.ts_code = t.ts_code
        where t.trade_date = :d {cond}
        -- 按净额【绝对值】排序: 净卖 5 亿和净买 5 亿都是大资金动作, 都该排在
        -- 前面。按原值排, "全部"视图里净卖出会全被挤到末尾, 反而看不到。
        order by abs(t.net_amount) desc nulls last
    """), {"d": d})).fetchall()
    return {"date": str(d), "side": side, "count": len(rows), "items": [
        {"ts_code": r[0], "name": r[1],
         "close": _f(r[2]), "pct_change": _f(r[3]), "turnover_rate": _f(r[4]),
         "buy_wan": _f(r[5], 1, 1e4) or 0.0,
         "sell_wan": _f(r[6], 1, 1e4) or 0.0,
         "net_wan": _f(r[7], 1, 1e4) or 0.0,
         "reason": r[8] or ""} for r in rows]}


@router.get("/stock/{ts_code}")
async def stock_history(ts_code: str, limit: int = Query(50, ge=1, le=200),
                        db: AsyncSession = Depends(get_db)):
    """单只股票的历史上榜记录 —— 看它多久上一次榜、每次是谁在买。"""
    rows = (await db.execute(text("""
        select trade_date, close, pct_change, net_amount, l_buy, l_sell, reason
        from top_list where ts_code = :c order by trade_date desc limit :l
    """), {"c": ts_code, "l": limit})).fetchall()
    return {"ts_code": ts_code, "count": len(rows), "items": [
        {"date": str(r[0]),
         "close": _f(r[1]), "pct_change": _f(r[2]),
         "net_wan": _f(r[3], 1, 1e4) or 0.0,
         "buy_wan": _f(r[4], 1, 1e4) or 0.0,
         "sell_wan": _f(r[5], 1, 1e4) or 0.0,
         "reason": r[6] or ""} for r in rows]}
