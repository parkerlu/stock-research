from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.datasources.akshare_provider import AKShareProvider
from app.datasources.manager import DataSourceManager
from app.datasources.tencent_provider import TencentProvider
from app.datasources.tushare_provider import TuShareProvider
from app.db import get_db
from app.services.quote_service import get_candles
from app.services.t0_indicator import compute_signals
from app.services.search_service import (
    add_favorite,
    add_search_history,
    get_favorites,
    get_search_history,
    remove_favorite,
    search_stocks,
)

router = APIRouter(prefix="/api")

_manager: DataSourceManager | None = None


def get_manager() -> DataSourceManager:
    global _manager
    if _manager is None:
        # 腾讯放最前: 实时快照走它 (毫秒级, 支持批量);
        # 它的 fetch_daily 返回空, 历史类调用会无开销穿透到 TuShare。
        _manager = DataSourceManager(
            TencentProvider(),
            TuShareProvider(token=settings.tushare_token),
            AKShareProvider(),
        )
    return _manager


def get_tencent() -> TencentProvider:
    p = get_manager().provider_of(TencentProvider)
    if p is None:
        raise HTTPException(503, "Tencent provider unavailable")
    return p


@router.get("/quotes/search")
async def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    return await search_stocks(db, q.strip(), limit)


@router.get("/quotes/{symbol}/candles")
async def candles(
    symbol: str,
    tf: str = Query("1d", pattern="^(1d|1w|1m)$"),
    start: date = Query(alias="from", default=None),
    end: date = Query(alias="to", default=None),
    db: AsyncSession = Depends(get_db),
    manager: DataSourceManager = Depends(get_manager),
):
    if start is None:
        start = date(date.today().year - 10, date.today().month, date.today().day)
    if end is None:
        end = date.today()

    await add_search_history(db, symbol)

    try:
        data = await get_candles(db, manager, symbol, tf, start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"symbol": symbol, "tf": tf, "candles": data}


@router.get("/quotes/{symbol}/snapshot")
async def snapshot(
    symbol: str,
    manager: DataSourceManager = Depends(get_manager),
):
    try:
        data = await manager.fetch_snapshot(symbol)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    if data is None:
        raise HTTPException(status_code=404, detail="Snapshot not available")
    return data


@router.get("/quotes/{symbol}/minute")
async def minute(
    symbol: str,
    tencent: TencentProvider = Depends(get_tencent),
):
    """当日分时 — 价格线 / 均价线 / 分钟量柱 + 实时快照 + 上证开闭市状态。

    前端据此画分时图, 并用 market_open 决定要不要继续轮询。
    """
    data = await tencent.fetch_minute(symbol)
    if data is None:
        raise HTTPException(status_code=404, detail="分时数据不可用")
    return data


@router.get("/quotes/{symbol}/t0")
async def t0_signals(
    symbol: str,
    threshold: float = Query(0.7, ge=0.5, le=0.95, description="信号阈值, 越高越少越准"),
    tencent: TencentProvider = Depends(get_tencent),
    db: AsyncSession = Depends(get_db),
):
    """日内做 T 信号 — 在分时图上标出"到收盘还有 3% 空间"的买卖点。

    卖点(高抛): 用底仓卖出, 跌 3% 或收盘前接回。
    买点(低吸): 买入, 涨 3% 或收盘前卖出等量底仓。
    样本外(2024-2026, 阈值0.7): 卖点精确率 64.7%, 买点 54.4%, 中位收益约 +3%。
    """
    from sqlalchemy import text

    data = await tencent.fetch_minute(symbol)
    if data is None:
        raise HTTPException(status_code=404, detail="分时数据不可用")

    # 昨日日线特征 — 与训练口径一致(振幅用 (high-low)/open, 收益用复权价)
    rows = (await db.execute(text(
        "select open, high, low, close, vol, adj_factor from daily_candle "
        "where ts_code = :c order by trade_date desc limit 2"
    ), {"c": symbol})).fetchall()
    if len(rows) < 2:
        return {"symbol": symbol, "signals": [], "reason": "日线历史不足"}
    y, y2 = rows[0], rows[1]
    o, h, l, c_, v, adj = (float(y[0]), float(y[1]), float(y[2]),
                           float(y[3]), float(y[4] or 0), float(y[5] or 1))
    c2, adj2 = float(y2[3]), float(y2[5] or 1)
    prev_amp = (h - l) / o if o else 0.0
    prev_ret = (c_ * adj) / (c2 * adj2) - 1 if c2 and adj2 else 0.0

    signals = compute_signals(
        symbol, data.get("bars") or [], float(data.get("prev_close") or 0),
        prev_amp, prev_ret, v, threshold=threshold,
    )
    return {"symbol": symbol, "trade_date": data.get("trade_date"),
            "threshold": threshold, "signals": signals}


@router.get("/quotes/snapshots")
async def snapshots(
    codes: str = Query(..., description="逗号分隔的 ts_code, 如 603319.SH,000001.SZ"),
    tencent: TencentProvider = Depends(get_tencent),
):
    """批量实时快照 — 股票池/选股结果列表用, 一次请求拿全部。"""
    wanted = [c.strip() for c in codes.split(",") if c.strip()]
    if not wanted:
        raise HTTPException(status_code=400, detail="codes 不能为空")
    if len(wanted) > 500:
        raise HTTPException(status_code=400, detail="一次最多 500 只")
    return await tencent.fetch_snapshots(wanted)


@router.get("/search-history")
async def history(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    return await get_search_history(db, limit)


@router.post("/favorites/{ts_code}")
async def add_fav(ts_code: str, db: AsyncSession = Depends(get_db)):
    await add_favorite(db, ts_code)
    return {"ok": True}


@router.delete("/favorites/{ts_code}")
async def remove_fav(ts_code: str, db: AsyncSession = Depends(get_db)):
    await remove_favorite(db, ts_code)
    return {"ok": True}


@router.get("/favorites")
async def list_favs(db: AsyncSession = Depends(get_db)):
    return await get_favorites(db)
