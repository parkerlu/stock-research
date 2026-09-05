from dataclasses import asdict
from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.datasources.manager import DataSourceManager
from app.db import get_db
from app.routers.quotes import get_manager
from app.services.quote_service import get_candles
from app.services.tdx import registry
from app.services.tdx.indicators.base import IndicatorResult

router = APIRouter(prefix="/api/indicators")


@router.get("")
def list_indicators():
    """List all registered TDX indicators."""
    return [
        {"name": m.name, "label": m.label, "pane": m.pane, "min_bars": m.min_bars}
        for m in registry.all_indicators()
    ]


@router.get("/{name}")
async def get_indicator(
    name: str,
    ts_code: str = Query(...),
    tf: str = Query("1d", pattern="^(1d|1w|1m)$"),
    start: date = Query(alias="from", default=None),
    end: date = Query(alias="to", default=None),
    db: AsyncSession = Depends(get_db),
    manager: DataSourceManager = Depends(get_manager),
):
    impl = registry.get(name)
    if impl is None:
        raise HTTPException(status_code=404, detail=f"Unknown indicator: {name}")

    if start is None:
        start = date(date.today().year - 2, date.today().month, date.today().day)
    if end is None:
        end = date.today()

    try:
        candles = await get_candles(db, manager, ts_code, tf, start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not candles:
        return asdict(IndicatorResult(
            name=impl.name, label=impl.label, pane=impl.pane,
            warnings=["该股票在所选区间无 K 线数据"],
        ))

    df = pd.DataFrame(candles)

    if len(df) < impl.min_bars:
        return asdict(IndicatorResult(
            name=impl.name, label=impl.label, pane=impl.pane,
            warnings=[
                f"数据不足，{impl.label} 需要至少 {impl.min_bars} 个 bar，当前仅 {len(df)} 个"
            ],
        ))

    result = impl.compute(df)
    return asdict(result)

@router.get("/maimai/{ts_code}")
async def maimai_signals(
    ts_code: str,
    start: str | None = Query(None, description="YYYY-MM-DD"),
    min_grade: str = Query("弱", description="弱=全部 / 中 / 强"),
    side: str = Query("buy", description="buy 买点(默认) / sell 卖点 / all 两者"),
    db: AsyncSession = Depends(get_db),
):
    """买卖很准 v3 —— 原始买点 + 模型评分。

    原指标买点(超卖结束→反转确认)本身与随机无异(32万信号: 胜率48.7%, 中位-0.132%)。
    模型在【同日×同波动层】中性化标签上训练, 样本外把 Top20% 的胜率提到 51.3%、
    中位数翻正到 +0.264%, 低/中/高三个波动档超出全为正。

    grade: 强(当日分位≥80%) / 中(≥50%) / 弱 —— 弱档即模型判定的噪声买点。
    """
    from sqlalchemy import text

    order = {"弱": 0.0, "中": 0.5, "强": 0.8}
    floor = order.get(min_grade, 0.0)
    sql = ("select trade_date, score, rank_pct, grade, side from maimai_signal "
           "where ts_code = :c and rank_pct >= :f")
    params: dict = {"c": ts_code, "f": floor}
    if side in ("buy", "sell"):
        sql += " and side = :sd"
        params["sd"] = side
    if start:
        # asyncpg 不接受字符串日期, 必须转成 date 对象
        try:
            params["s"] = date.fromisoformat(start)
            sql += " and trade_date >= :s"
        except ValueError:
            raise HTTPException(status_code=400, detail="start 需为 YYYY-MM-DD")
    sql += " order by trade_date"
    rows = (await db.execute(text(sql), params)).fetchall()
    return {"ts_code": ts_code, "count": len(rows), "signals": [
        {"date": str(r[0]), "score": float(r[1]),
         "rank_pct": round(float(r[2]), 4), "grade": r[3],
         "side": r[4] if len(r) > 4 else "buy"} for r in rows]}


@router.get("/maimai/scan/recent")
async def maimai_latest(
    days: int = Query(5, ge=1, le=30),
    grade: str = Query("强"),
    db: AsyncSession = Depends(get_db),
):
    """最近几个交易日的高分买点 —— 用于盘后扫描。

    路径特意用三段: 本路由文件里 `/{name}` 是单段通配, 会抢走任何单段路径。
    """
    from sqlalchemy import text

    rows = (await db.execute(text("""
        select m.trade_date, m.ts_code, coalesce(b.name, '') as name,
               m.score, m.rank_pct, m.grade
        from maimai_signal m
        left join stock_basic b on b.ts_code = m.ts_code
        where m.grade = :g
          and m.trade_date > (select max(trade_date) from maimai_signal) - make_interval(days => :d)
        order by m.trade_date desc, m.rank_pct desc
        limit 300"""), {"g": grade, "d": days})).fetchall()
    return {"signals": [
        {"date": str(r[0]), "ts_code": r[1], "name": r[2],
         "score": float(r[3]), "rank_pct": round(float(r[4]), 4), "grade": r[5]}
        for r in rows]}


@router.get("/pump/{ts_code}")
async def pump_signals(
    ts_code: str,
    start: str | None = Query(None, description="YYYY-MM-DD"),
    min_grade: str = Query("中", description="中 / 强"),
    db: AsyncSession = Depends(get_db),
):
    """主力吸筹 —— 预测未来10日内出现拉升(单日涨幅>7% 且量>20日均量2倍)的概率。

    核心信息来自筹码分布的【获利盘背离】= 获利盘变化 − 价格涨幅:
    价格没动但获利盘上升 = 有人在低位持续接走浮筹, 这是价格图上看不出来的。
    在【价格位置 × 量能】双重控制的九宫格里预测力几乎不衰减, 说明是独立信息。

    样本外(2020-2026 walk-forward): Q10/Q1=4.81, 按天t=77.4, 逐年 1.70~2.02 倍。
    强档(当日分位≥95%) 拉升概率约 20~30%, 基础概率仅 9.8%。
    """
    from sqlalchemy import text

    floor = 0.95 if min_grade == "强" else 0.8
    sql = ("select trade_date, prob, rank_pct, grade from pump_signal "
           "where ts_code = :c and rank_pct >= :f")
    params: dict = {"c": ts_code, "f": floor}
    if start:
        try:
            params["s"] = date.fromisoformat(start)
            sql += " and trade_date >= :s"
        except ValueError:
            raise HTTPException(status_code=400, detail="start 需为 YYYY-MM-DD")
    sql += " order by trade_date"
    rows = (await db.execute(text(sql), params)).fetchall()
    return {"ts_code": ts_code, "count": len(rows), "signals": [
        {"date": str(r[0]), "prob": round(float(r[1]), 4),
         "rank_pct": round(float(r[2]), 4), "grade": r[3]} for r in rows]}


@router.get("/pump/scan/recent")
async def pump_scan(
    days: int = Query(5, ge=1, le=30),
    grade: str = Query("强"),
    db: AsyncSession = Depends(get_db),
):
    """盘后扫描 —— 最近几日吸筹嫌疑最大的股票。"""
    from sqlalchemy import text

    rows = (await db.execute(text("""
        select p.trade_date, p.ts_code, coalesce(b.name, '') as name,
               p.prob, p.rank_pct, p.grade
        from pump_signal p
        left join stock_basic b on b.ts_code = p.ts_code
        where p.grade = :g
          and p.trade_date > (select max(trade_date) from pump_signal)
                             - make_interval(days => :d)
        order by p.trade_date desc, p.prob desc
        limit 300"""), {"g": grade, "d": days})).fetchall()
    return {"signals": [
        {"date": str(r[0]), "ts_code": r[1], "name": r[2], "prob": round(float(r[3]), 4),
         "rank_pct": round(float(r[4]), 4), "grade": r[5]} for r in rows]}


# ---- 训练指标选股 ----
# 信号已在库中(build_maimai / build_pump 盘后算好), 这里直接查, 不跑扫描任务。
TRAINED_META = [
    {"key": "maimai_buy", "label": "买卖很准 v3",
     "desc": "动能参考 · 超卖反转买点。过滤后胜率 48.9%→50.9%, 是改进不是答案",
     "grades": ["强", "中", "弱"], "default_grade": "强"},
    {"key": "v5", "label": "★★ v5 (三重共振)",
     "desc": "v3强 × 吸筹强 × 近7日龙虎榜机构净买入。胜率 62.1%, 平均收益 +6.86%(持有20日)。"
             "全市场一年仅约 7.5 个 —— 建议看 90 日窗口, 出现时值得认真看一眼",
     "grades": ["强"], "default_grade": "强", "default_days": 90},
    {"key": "maimai35", "label": "买卖很准 v3.5 (参数重扫)",
     "desc": "动能参考 · 参数从 MA5/LLV10/5 改为 MA8/LLV20/10(36组扫描最优)。"
             "胜率 56.2%(原 50.9%), 信号少三分之二, 普涨行情会跑输",
     "grades": ["强", "中"], "default_grade": "强"},
    {"key": "combo", "label": "★ 买卖很准 v4 (共振)",
     "desc": "v3 + 主力吸筹共振(近5日内先后触发)。胜率 58.3%, 超同日全市场 +1.93pp。"
             "每天约 1.2 个",
     "grades": ["强"], "default_grade": "强"},
    {"key": "didian", "label": "低点组合 v2",
     "desc": "动能参考 · 阶段底部 + 模型过滤。Top10% 胜率 57.7%(原 53.8%), 持有20日",
     "grades": ["强", "中"], "default_grade": "强"},
    {"key": "pump", "label": "主力吸筹",
     "desc": "动能参考 · 强档未来10日拉升率 19.6%(基础 8.6%)。仍有八成不发生, 用于缩小候选池",
     "grades": ["强", "中"], "default_grade": "强"},
]


@router.get("/trained/list")
async def trained_list():
    """可用于选股的训练指标清单。"""
    return {"indicators": TRAINED_META}


@router.get("/trained/screen")
async def screen_by_trained(
    indicator: str = Query("pump", description="maimai_buy / maimai_sell / pump"),
    grade: str = Query("强"),
    days: int = Query(5, ge=1, le=180, description="最近几个交易日内出现过信号"),
    limit: int = Query(200, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """按训练指标选股 —— 返回最近 N 日内出信号的股票 + 当日收盘行情。"""
    from sqlalchemy import text

    if indicator == "v5":
        # 三重共振: v3强档 × 吸筹强档 × 近5日龙虎榜机构净买入。
        # ⚠️ 时间窗只能向后看。原来写成 mm.trade_date ± 7 是前视: T 日的信号要求
        # 知道未来 7 天会不会上龙虎榜, 而龙虎榜是事后才公布的。改成因果口径后
        # 信号从 609 个掉到 58 个(九成靠未来才成立), 胜率 78.0% → 62.1%。
        rows = (await db.execute(text("""
            with mm as (
              select ts_code, trade_date, rank_pct from maimai_signal
              where side='buy' and rank_pct >= 0.8
                and trade_date > (select max(trade_date) from maimai_signal)
                                 - make_interval(days => :d)
            ),
            pp as (select ts_code, trade_date from pump_signal where rank_pct >= 0.95),
            lb as (select ts_code, trade_date, net_amount from top_list where net_amount > 0),
            hit as (
              select mm.ts_code, mm.trade_date, mm.rank_pct,
                     max(lb.net_amount) as net_amt
              from mm
              join pp on pp.ts_code = mm.ts_code
                     and pp.trade_date between mm.trade_date - 5 and mm.trade_date
              join lb on lb.ts_code = mm.ts_code
                     and lb.trade_date between mm.trade_date - 7 and mm.trade_date
              group by mm.ts_code, mm.trade_date, mm.rank_pct
            ),
            px as (
              select d.ts_code, d.trade_date, d.close, d.adj_factor,
                     row_number() over (partition by d.ts_code order by d.trade_date desc) rn
              from daily_candle d
              where d.trade_date > (select max(trade_date) - interval '20 days' from daily_candle)
            ),
            last2 as (
              select ts_code,
                     max(close) filter (where rn=1) c1, max(adj_factor) filter (where rn=1) a1,
                     max(close) filter (where rn=2) c2, max(adj_factor) filter (where rn=2) a2
              from px where rn <= 2 group by ts_code
            )
            select h.ts_code, coalesce(b.name,'') as name, h.trade_date,
                   h.net_amt as score, h.rank_pct, '强' as grade,
                   l.c1 as price, (l.c1*l.a1)/nullif(l.c2*l.a2,0)-1 as chg
            from hit h
            left join stock_basic b on b.ts_code = h.ts_code
            left join last2 l on l.ts_code = h.ts_code
            order by h.trade_date desc, h.net_amt desc nulls last
            limit :lim
        """), {"d": days, "lim": limit})).fetchall()
        return {"indicator": indicator, "grade": "强", "days": days,
                "count": len(rows), "items": [
            {"ts_code": r[0], "name": r[1], "date": str(r[2]),
             "score": round(float(r[3]) / 1e4, 1) if r[3] is not None else None,  # 万元
             "rank_pct": round(float(r[4]), 4), "grade": r[5],
             "price": round(float(r[6]), 2) if r[6] is not None else None,
             "chg": round(float(r[7]) * 100, 2) if r[7] is not None else None}
            for r in rows]}

    if indicator == "combo":
        # 共振: 买卖很准强档(rank>=0.8) 与 主力吸筹强档(rank>=0.95) 在 ±3 交易日内同时出现。
        # 不要求同一天 —— 两个指标描述的是同一段行情的不同阶段, 触发时点天然错开,
        # 严格同日会漏掉绝大多数真共振(实测同日只剩不到三成)。
        rows = (await db.execute(text("""
            with mm as (
              select ts_code, trade_date, rank_pct
              from maimai_signal
              where side='buy' and rank_pct >= 0.8
                and trade_date > (select max(trade_date) from maimai_signal)
                                 - make_interval(days => :d)
            ),
            pp as (select ts_code, trade_date from pump_signal where rank_pct >= 0.95),
            hit as (
              select mm.ts_code, mm.trade_date, mm.rank_pct
              from mm join pp on pp.ts_code = mm.ts_code
                   and pp.trade_date between mm.trade_date - 5 and mm.trade_date
              group by mm.ts_code, mm.trade_date, mm.rank_pct
            ),
            px as (
              select d.ts_code, d.trade_date, d.close, d.adj_factor,
                     row_number() over (partition by d.ts_code order by d.trade_date desc) rn
              from daily_candle d
              where d.trade_date > (select max(trade_date) - interval '20 days' from daily_candle)
            ),
            last2 as (
              select ts_code,
                     max(close) filter (where rn=1) c1, max(adj_factor) filter (where rn=1) a1,
                     max(close) filter (where rn=2) c2, max(adj_factor) filter (where rn=2) a2
              from px where rn <= 2 group by ts_code
            )
            select h.ts_code, coalesce(b.name,'') as name, h.trade_date,
                   h.rank_pct as score, h.rank_pct, '强' as grade,
                   l.c1 as price, (l.c1*l.a1)/nullif(l.c2*l.a2,0)-1 as chg
            from hit h
            left join stock_basic b on b.ts_code = h.ts_code
            left join last2 l on l.ts_code = h.ts_code
            order by h.trade_date desc, h.rank_pct desc
            limit :lim
        """), {"d": days, "lim": limit})).fetchall()
        return {"indicator": indicator, "grade": "强", "days": days,
                "count": len(rows), "items": [
            {"ts_code": r[0], "name": r[1], "date": str(r[2]),
             "score": round(float(r[3]), 5), "rank_pct": round(float(r[4]), 4),
             "grade": r[5],
             "price": round(float(r[6]), 2) if r[6] is not None else None,
             "chg": round(float(r[7]) * 100, 2) if r[7] is not None else None}
            for r in rows]}

    if indicator == "pump":
        sig_sql = ("select ts_code, trade_date, prob as score, rank_pct, grade "
                   "from pump_signal where grade = :g")
        anchor = "select max(trade_date) from pump_signal"
    elif indicator == "maimai35":
        sig_sql = ("select ts_code, trade_date, score, rank_pct, grade "
                   "from maimai35_signal where grade = :g")
        anchor = "select max(trade_date) from maimai35_signal"
    elif indicator == "didian":
        sig_sql = ("select ts_code, trade_date, score, rank_pct, grade "
                   "from didian_signal where grade = :g")
        anchor = "select max(trade_date) from didian_signal"
    elif indicator in ("maimai_buy", "maimai_sell"):
        side = "sell" if indicator == "maimai_sell" else "buy"
        sig_sql = ("select ts_code, trade_date, score, rank_pct, grade "
                   f"from maimai_signal where grade = :g and side = '{side}'")
        anchor = "select max(trade_date) from maimai_signal"
    else:
        # ⚠️ 不要用 else 兜底: 写错指标名会静默返回买卖很准的结果, 看起来"能用"
        # 但选的根本不是那个指标。加指标时忘了加分支也是同样的静默错误。
        raise HTTPException(status_code=400, detail=f"未知指标: {indicator}")

    rows = (await db.execute(text(f"""
        with sig as ({sig_sql}
                     and trade_date > ({anchor}) - make_interval(days => :d)),
        px as (
          select d.ts_code, d.trade_date, d.close, d.adj_factor,
                 row_number() over (partition by d.ts_code order by d.trade_date desc) rn
          from daily_candle d
          where d.trade_date > (select max(trade_date) - interval '20 days' from daily_candle)
        ),
        last2 as (
          select ts_code,
                 max(close) filter (where rn = 1) as c1,
                 max(adj_factor) filter (where rn = 1) as a1,
                 max(close) filter (where rn = 2) as c2,
                 max(adj_factor) filter (where rn = 2) as a2
          from px where rn <= 2 group by ts_code
        )
        select s.ts_code, coalesce(b.name, '') as name, s.trade_date,
               s.score, s.rank_pct, s.grade,
               l.c1 as price,
               (l.c1 * l.a1) / nullif(l.c2 * l.a2, 0) - 1 as chg
        from sig s
        left join stock_basic b on b.ts_code = s.ts_code
        left join last2 l on l.ts_code = s.ts_code
        order by s.trade_date desc, s.rank_pct desc
        limit :lim
    """), {"g": grade, "d": days, "lim": limit})).fetchall()

    return {"indicator": indicator, "grade": grade, "days": days,
            "count": len(rows), "items": [
        {"ts_code": r[0], "name": r[1], "date": str(r[2]),
         "score": round(float(r[3]), 5), "rank_pct": round(float(r[4]), 4),
         "grade": r[5],
         "price": round(float(r[6]), 2) if r[6] is not None else None,
         "chg": round(float(r[7]) * 100, 2) if r[7] is not None else None}
        for r in rows]}


@router.get("/didian/{ts_code}")
async def didian_signals(
    ts_code: str,
    start: str | None = Query(None, description="YYYY-MM-DD"),
    min_grade: str = Query("中", description="中 / 强"),
    db: AsyncSession = Depends(get_db),
):
    """低点组合 v2 —— 原指标「阶段底部」+ 模型过滤。

    原指标两类买入信号里, DIBU(四周期KDJ共振)实测【输给随机】(H=5 −0.074pp),
    只有「阶段底部」(动力线上穿0.2)有效, 所以这里只用后者。
    样本外 Top10%: +4.983%(H=20), 胜率 57.7%, 按天 t=3.88, 波动层内三档 t 全>2.5。
    """
    from sqlalchemy import text

    floor = 0.9 if min_grade == "强" else 0.7
    sql = ("select trade_date, score, rank_pct, grade from didian_signal "
           "where ts_code = :c and rank_pct >= :f")
    params: dict = {"c": ts_code, "f": floor}
    if start:
        try:
            params["s"] = date.fromisoformat(start)
            sql += " and trade_date >= :s"
        except ValueError:
            raise HTTPException(status_code=400, detail="start 需为 YYYY-MM-DD")
    sql += " order by trade_date"
    rows = (await db.execute(text(sql), params)).fetchall()
    return {"ts_code": ts_code, "count": len(rows), "signals": [
        {"date": str(r[0]), "score": round(float(r[1]), 5),
         "rank_pct": round(float(r[2]), 4), "grade": r[3]} for r in rows]}


@router.get("/combo/{ts_code}")
async def combo_signals(
    ts_code: str,
    start: str | None = Query(None, description="YYYY-MM-DD"),
    db: AsyncSession = Depends(get_db),
):
    """买卖很准 v4 —— v3 强档 与 主力吸筹强档 在 ±3 交易日内共振。

    单独看时 主力吸筹是负面的(H=20 胜率 43.5%, 中位 −2.06%) —— 它预测的是
    "会不会拉升", 拉升前的票往往还在跌。但叠加到一个已有 alpha 的信号上,
    它是最大的增益来源: v3 强档单独 50.9% → 共振后 58.3%(因果口径)。
    它抓的信息真实存在, 只是不能单独当买入信号。

    三关验证: 逐年八年全部>54% / 同日随机对照 z=22.1 / 波动×市值九格超出全为正。
    """
    from sqlalchemy import text

    sql = """
        with mm as (
          select ts_code, trade_date, rank_pct from maimai_signal
          where side='buy' and rank_pct >= 0.8 and ts_code = :c
        ),
        pp as (select ts_code, trade_date from pump_signal
               where rank_pct >= 0.95 and ts_code = :c)
        select mm.trade_date, mm.rank_pct
        from mm join pp on pp.trade_date between mm.trade_date - 5 and mm.trade_date
        group by mm.trade_date, mm.rank_pct
    """
    params: dict = {"c": ts_code}
    if start:
        try:
            params["s"] = date.fromisoformat(start)
            sql += " having mm.trade_date >= :s" if False else ""
        except ValueError:
            raise HTTPException(status_code=400, detail="start 需为 YYYY-MM-DD")
    sql += " order by mm.trade_date"
    rows = (await db.execute(text(sql), params)).fetchall()
    items = [{"date": str(r[0]), "score": round(float(r[1]), 5),
              "rank_pct": round(float(r[1]), 4), "grade": "强"} for r in rows]
    if start:
        items = [x for x in items if x["date"] >= start]
    return {"ts_code": ts_code, "count": len(items), "signals": items}


@router.get("/maimai35/{ts_code}")
async def maimai35_signals(
    ts_code: str,
    start: str | None = Query(None, description="YYYY-MM-DD"),
    min_grade: str = Query("中", description="中 / 强"),
    db: AsyncSession = Depends(get_db),
):
    """买卖很准 v3.5 —— 原指标参数重扫版(MA8/LLV20/连续10)。

    移植的参数是照抄 TDX 的, 36 组网格扫描后只排 9/33。换参数后:
      原参数 28.1万信号 胜率 50.9% 超出随机 +0.190pp
      v3.5    9.5万信号 胜率 56.2% 超出随机 +0.602pp

    ⚠️ 它是"放大器": 好年份 +1.0pp、坏年份 −1.0pp(原参数只有 ±0.5pp)。
    2020/2023/2025 跑输随机 —— 普涨行情里严格筛选反而错过。
    """
    from sqlalchemy import text

    floor = 0.8 if min_grade == "强" else 0.5
    sql = ("select trade_date, score, rank_pct, grade from maimai35_signal "
           "where ts_code = :c and rank_pct >= :f")
    params: dict = {"c": ts_code, "f": floor}
    if start:
        try:
            params["s"] = date.fromisoformat(start)
            sql += " and trade_date >= :s"
        except ValueError:
            raise HTTPException(status_code=400, detail="start 需为 YYYY-MM-DD")
    sql += " order by trade_date"
    rows = (await db.execute(text(sql), params)).fetchall()
    return {"ts_code": ts_code, "count": len(rows), "signals": [
        {"date": str(r[0]), "score": round(float(r[1]), 5),
         "rank_pct": round(float(r[2]), 4), "grade": r[3]} for r in rows]}


@router.get("/v5/{ts_code}")
async def v5_signals(
    ts_code: str,
    start: str | None = Query(None, description="YYYY-MM-DD"),
    db: AsyncSession = Depends(get_db),
):
    """v5 三重共振 —— v3强 × 主力吸筹强 × 近5日龙虎榜机构净买入。

    龙虎榜覆盖率仅 1.2%(只有异动才上榜), 单独用没意义, 但作为第三层过滤:
      v4 (v3强×吸筹强)      胜率 58.3%  收益 +5.57%  超全市场 +1.93pp
      v5 (+龙虎榜净买入)     胜率 62.1%  收益 +6.86%  超全市场 +2.49pp
    ⚠️ v5 因果口径下 7.7 年只有 58 个信号(约 7.5 个/年), 样本量小, 别当策略用。

    ⚠️ 因果口径下全市场一年仅约 7.5 个。K线菜单里已撤掉(常年空白), 只在选股页出现。
    """
    from sqlalchemy import text

    sql = """
        with mm as (select ts_code, trade_date, rank_pct from maimai_signal
                    where side='buy' and rank_pct >= 0.8 and ts_code = :c),
        pp as (select trade_date from pump_signal where rank_pct >= 0.95 and ts_code = :c),
        lb as (select trade_date, net_amount from top_list
               where net_amount > 0 and ts_code = :c)
        select mm.trade_date, mm.rank_pct, max(lb.net_amount) as net_amt
        from mm
        join pp on pp.trade_date between mm.trade_date - 5 and mm.trade_date
        join lb on lb.trade_date between mm.trade_date - 7 and mm.trade_date
        group by mm.trade_date, mm.rank_pct
        order by mm.trade_date
    """
    rows = (await db.execute(text(sql), {"c": ts_code})).fetchall()
    items = [{"date": str(r[0]), "score": round(float(r[1]), 5),
              "rank_pct": round(float(r[1]), 4), "grade": "强",
              "net_amount_wan": round(float(r[2]) / 1e4, 1) if r[2] is not None else None}
             for r in rows]
    if start:
        items = [x for x in items if x["date"] >= start]
    return {"ts_code": ts_code, "count": len(items), "signals": items}
