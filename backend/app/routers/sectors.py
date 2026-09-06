"""概念板块 —— 个股↔板块双向查询 + 板块热度(盘后口径)。

热度全部基于当日收盘(daily_candle), 不拉实时行情:
  - 上涨占比 = 板块内上涨家数 / 有当日行情的成分数
  - 平均涨幅 = 成分股涨跌幅的等权平均
等权而非市值加权 —— 看的是"这个主题有多少票在动"(广度),
市值加权会被一两只权重股主导, 掩盖真实广度。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db

router = APIRouter(prefix="/api")

NON_THEME = ("成份股", "样本股", "指数", "融资融券", "沪股通", "深股通",
             "标准券", "MSCI", "富时", "转债标的", "股权转让", "举牌")

# 当日涨跌幅 = 复权后 close/前一交易日 close - 1。
# 用复权价而非原始价 —— 除权日用原始价会算出巨大假跌幅。
_PCT_CTE = """
with last2 as (
  select ts_code, trade_date, close, adj_factor,
         row_number() over (partition by ts_code order by trade_date desc) rn
  from daily_candle
  where trade_date > (select max(trade_date) - interval '20 days' from daily_candle)
),
pct as (
  select a.ts_code,
         a.trade_date,
         a.close as price,
         (a.close * a.adj_factor) / nullif(b.close * b.adj_factor, 0) - 1 as chg
  from last2 a join last2 b on b.ts_code = a.ts_code and b.rn = 2
  where a.rn = 1
)
"""


@router.get("/sectors")
async def list_sectors(
    theme_only: bool = Query(True, description="只看主题概念, 剔除宽基指数样本股等"),
    db: AsyncSession = Depends(get_db),
):
    """板块列表 + 最新收盘日的热度。"""
    rows = (await db.execute(text(_PCT_CTE + """
        select s.ts_code, s.name, s.count, s.list_date,
               count(p.ts_code)                                    as quoted,
               count(*) filter (where p.chg > 0)                   as up,
               count(*) filter (where p.chg < 0)                   as down,
               count(*) filter (where p.chg = 0)                   as flat,
               avg(p.chg) * 100                                    as avg_pct,
               percentile_cont(0.5) within group (order by p.chg) * 100 as median_pct,
               max(p.chg) * 100                                    as max_pct,
               min(p.chg) * 100                                    as min_pct,
               max(p.trade_date)                                   as trade_date
        from concept_sector s
        left join concept_member m on m.sector_code = s.ts_code
        left join pct p on p.ts_code = m.ts_code
        group by s.ts_code, s.name, s.count, s.list_date
    """))).fetchall()

    out, td = [], None
    for r in rows:
        (code, name, cnt, list_date, quoted, up, down, flat,
         avg_pct, median_pct, max_pct, min_pct, trade_date) = r
        if theme_only and any(k in (name or "") for k in NON_THEME):
            continue
        if trade_date and (td is None or trade_date > td):
            td = trade_date
        out.append({
            "ts_code": code, "name": name, "count": cnt or 0,
            "list_date": str(list_date) if list_date else None,
            "quoted": quoted or 0, "up": up or 0, "down": down or 0, "flat": flat or 0,
            "up_ratio": round(up / quoted * 100, 2) if quoted else None,
            "avg_pct": round(float(avg_pct), 3) if avg_pct is not None else None,
            "median_pct": round(float(median_pct), 3) if median_pct is not None else None,
            "max_pct": round(float(max_pct), 2) if max_pct is not None else None,
            "min_pct": round(float(min_pct), 2) if min_pct is not None else None,
        })
    out.sort(key=lambda x: (x["avg_pct"] is None, -(x["avg_pct"] or 0)))
    # 覆盖率 = 有当日行情的成分 / 总成分。收盘后 K 线渐进入库, 最新交易日常只有
    # 三分之一到位 —— 这时的板块均值不可信, 前端据此提示。
    tot = sum(x["count"] or 0 for x in out)
    quoted = sum(x["quoted"] for x in out)
    return {"sectors": out, "trade_date": str(td) if td else None,
            "coverage": round(quoted / tot * 100, 1) if tot else None}


@router.get("/sectors/hot")
async def hot_sectors(
    days: int = Query(5, ge=1, le=60, description="回看几个交易日"),
    limit: int = Query(30, ge=1, le=100),
    theme_only: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    """N 日持续热度榜 —— 哪些主题在连续走强, 而不是只看今天涨了多少。

    单日榜首常是一两只涨停票把均值拉起来的, 换个日子就掉出去。这里给三个量:
      cum_pct   N 日累计平均涨幅(等权)
      up_days   N 日里板块均涨为正的天数 —— 持续性
      up_ratio  最新一日成分股上涨占比 —— 广度

    ⚠️ 这是信息工具, 不是信号。板块动量、换手热度、Top-N 轮动都做过样本外
    检验, t 值都在 ±0.1 量级, 不构成可交易的边(见 docs/训练指标.md)。
    排在前面只说明"最近这个主题在动", 不代表明天还会动。
    """
    rows = (await db.execute(text("""
        with cal as (
          select distinct trade_date from daily_candle
          order by trade_date desc limit :d + 1
        ),
        px as (
          select ts_code, trade_date, close * adj_factor as adj,
                 row_number() over (partition by ts_code order by trade_date) rn
          from daily_candle where trade_date in (select trade_date from cal)
        ),
        chg as (
          select a.ts_code, a.trade_date,
                 a.adj / nullif(b.adj, 0) - 1 as c
          from px a join px b on b.ts_code = a.ts_code and b.rn = a.rn - 1
        ),
        sec_day as (
          select m.sector_code, chg.trade_date, avg(chg.c) as avg_c,
                 count(*) as n,
                 count(*) filter (where chg.c > 0)::float / nullif(count(*), 0) as up_r
          from concept_member m join chg on chg.ts_code = m.ts_code
          group by m.sector_code, chg.trade_date
        ),
        agg as (
          select sector_code,
                 exp(sum(ln(1 + avg_c))) - 1                as cum,
                 count(*) filter (where avg_c > 0)          as up_days,
                 count(*)                                   as n_days,
                 max(trade_date)                            as last_day
          from sec_day where avg_c > -0.99 group by sector_code
        )
        select s.ts_code, s.name, s.count,
               agg.cum * 100 as cum_pct, agg.up_days, agg.n_days,
               (select up_r from sec_day sd
                 where sd.sector_code = s.ts_code and sd.trade_date = agg.last_day) * 100 as up_ratio,
               (select avg_c from sec_day sd
                 where sd.sector_code = s.ts_code and sd.trade_date = agg.last_day) * 100 as last_pct,
               agg.last_day
        from agg join concept_sector s on s.ts_code = agg.sector_code
        where (:theme = false or s.name not like any(array['%指数%','%成份%','%样本%']))
        order by agg.cum desc nulls last
        limit :l
    """), {"d": days, "l": limit, "theme": theme_only})).fetchall()

    return {"days": days, "count": len(rows), "items": [
        {"code": r[0], "name": r[1], "members": r[2],
         "cum_pct": round(float(r[3]), 2) if r[3] is not None else None,
         "up_days": r[4], "n_days": r[5],
         "up_ratio": round(float(r[6]), 1) if r[6] is not None else None,
         "last_pct": round(float(r[7]), 2) if r[7] is not None else None,
         "last_day": str(r[8])} for r in rows]}


@router.get("/sectors/{sector_code}/members")
async def sector_members(sector_code: str, db: AsyncSession = Depends(get_db)):
    """板块成分股 + 当日收盘行情, 按涨幅排序。"""
    sec = (await db.execute(text(
        "select name, count, list_date from concept_sector where ts_code = :c"),
        {"c": sector_code})).fetchone()
    if sec is None:
        raise HTTPException(status_code=404, detail="板块不存在")
    rows = (await db.execute(text(_PCT_CTE + """
        select m.ts_code, coalesce(b.name, m.name) as name,
               p.price, p.chg * 100 as pct, p.trade_date
        from concept_member m
        left join stock_basic b on b.ts_code = m.ts_code
        left join pct p on p.ts_code = m.ts_code
        where m.sector_code = :c
    """), {"c": sector_code})).fetchall()
    items = [{
        "ts_code": r[0], "name": r[1],
        "price": round(float(r[2]), 2) if r[2] is not None else None,
        "pct": round(float(r[3]), 2) if r[3] is not None else None,
        "amount": None, "vol": None, "turnover": None,
    } for r in rows]
    items.sort(key=lambda x: (x["pct"] is None, -(x["pct"] or 0)))
    return {"sector": {"ts_code": sector_code, "name": sec[0],
                       "count": sec[1] or len(items),
                       "list_date": str(sec[2]) if sec[2] else None},
            "members": items}


@router.get("/stocks/{ts_code}/sectors")
async def stock_sectors(ts_code: str, db: AsyncSession = Depends(get_db)):
    """这只股票属于哪些板块 + 各板块当日热度 —— K线页据此显示"它在哪些主题里"。

    按板块平均涨幅降序: 最靠前的就是今天最热的那条线。
    """
    rows = (await db.execute(text(_PCT_CTE + """
        select s.ts_code, s.name, s.count, s.list_date,
               count(p.ts_code)                      as quoted,
               count(*) filter (where p.chg > 0)     as up,
               avg(p.chg) * 100                      as avg_pct
        from concept_member m
        join concept_sector s on s.ts_code = m.sector_code
        join concept_member m2 on m2.sector_code = s.ts_code
        left join pct p on p.ts_code = m2.ts_code
        where m.ts_code = :c
        group by s.ts_code, s.name, s.count, s.list_date
        order by avg(p.chg) desc nulls last
    """), {"c": ts_code})).fetchall()
    return {"ts_code": ts_code, "sectors": [
        {"ts_code": r[0], "name": r[1], "count": r[2],
         "list_date": str(r[3]) if r[3] else None,
         "quoted": r[4] or 0, "up": r[5] or 0,
         "up_ratio": round(r[5] / r[4] * 100, 1) if r[4] else None,
         "avg_pct": round(float(r[6]), 2) if r[6] is not None else None,
         "is_theme": not any(k in (r[1] or "") for k in NON_THEME)}
        for r in rows]}


@router.get("/sectors/{sector_code}/history")
async def sector_history(
    sector_code: str,
    days: int = Query(30, ge=5, le=120),
    db: AsyncSession = Depends(get_db),
):
    """板块近 N 日的【平均涨幅 / 上涨占比】曲线。

    两条线要一起看: 平均涨幅说的是"涨了多少", 上涨占比说的是"多少票在涨"。
    平均 +2% 但只有 40% 上涨 = 少数权重股拉的; 平均 +2% 且 90% 上涨 = 真整体在动。
    """
    rows = (await db.execute(text("""
        with mem as (select ts_code from concept_member where sector_code = :c),
        px as (
          select d.ts_code, d.trade_date, d.close, d.adj_factor,
                 lag(d.close) over (partition by d.ts_code order by d.trade_date) pc,
                 lag(d.adj_factor) over (partition by d.ts_code order by d.trade_date) pa
          from daily_candle d
          where d.ts_code in (select ts_code from mem)
            and d.trade_date > (select max(trade_date) from daily_candle)
                               - make_interval(days => :d * 2)
        )
        , daily as (
          select trade_date,
                 count(*)                                        as n,
                 count(*) filter (where close*adj_factor > pc*pa) as up,
                 avg((close*adj_factor)/nullif(pc*pa,0) - 1)*100  as avg_pct
          from px
          where pc is not null
          group by trade_date
        )
        -- ⚠️ 剔除覆盖不足的交易日: 当日 K 线入库是渐进的, 最新一两天常只有
        -- 三分之一成分股到位(实测 9-04 只有 13/34)。用不完整的样本算板块均值
        -- 会严重失真 —— 13 只算出的 +1.59%, 代表不了 34 只的真实水平。
        select trade_date, n, up, avg_pct from daily
        where n >= (select max(n) * 0.8 from daily)
        order by trade_date desc
        limit :d
    """), {"c": sector_code, "d": days})).fetchall()
    items = [{"date": str(r[0]), "n": r[1], "up": r[2],
              "up_ratio": round(r[2] / r[1] * 100, 1) if r[1] else None,
              "avg_pct": round(float(r[3]), 3) if r[3] is not None else None}
             for r in reversed(rows)]
    # 累计收益 —— 看这段时间这个主题整体走了多少
    cum = 1.0
    for it in items:
        cum *= 1 + (it["avg_pct"] or 0) / 100
        it["cum_pct"] = round((cum - 1) * 100, 2)
    return {"sector_code": sector_code, "days": days, "items": items}
