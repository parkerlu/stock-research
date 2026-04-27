from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.routers.quotes import get_manager
from app.datasources.manager import DataSourceManager
from app.services.pool_service import (
    add_stock_to_pool,
    create_pool,
    delete_pool,
    get_pool_detail,
    list_pools,
    remove_stock_from_pool,
    update_pool,
)

router = APIRouter(prefix="/api/pools", tags=["pools"])


class CreatePoolRequest(BaseModel):
    name: str
    description: str | None = None


class UpdatePoolRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class CreateAndPopulateRequest(BaseModel):
    name: str
    description: str | None = None
    ts_codes: list[str]


class AddStockRequest(BaseModel):
    ts_code: str


@router.post("")
async def create(body: CreatePoolRequest, db: AsyncSession = Depends(get_db)):
    return await create_pool(db, body.name, body.description)


@router.post("/batch")
async def create_and_populate(
    body: CreateAndPopulateRequest,
    db: AsyncSession = Depends(get_db),
    manager: DataSourceManager = Depends(get_manager),
):
    """One-shot: create a pool and add all given ts_codes to it."""
    pool = await create_pool(db, body.name, body.description)
    pool_id = pool["id"]
    added = 0
    for ts in body.ts_codes:
        if await add_stock_to_pool(db, manager, pool_id, ts):
            added += 1
    return {"id": pool_id, "name": body.name, "added": added,
            "total_requested": len(body.ts_codes)}


@router.get("")
async def list_all(db: AsyncSession = Depends(get_db)):
    return await list_pools(db)


@router.get("/{pool_id}")
async def detail(pool_id: int, db: AsyncSession = Depends(get_db)):
    result = await get_pool_detail(db, pool_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Pool not found")
    return result


@router.put("/{pool_id}")
async def update(pool_id: int, body: UpdatePoolRequest, db: AsyncSession = Depends(get_db)):
    result = await update_pool(db, pool_id, body.name, body.description)
    if result is None:
        raise HTTPException(status_code=404, detail="Pool not found")
    return result


@router.delete("/{pool_id}")
async def remove(pool_id: int, db: AsyncSession = Depends(get_db)):
    ok = await delete_pool(db, pool_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Pool not found")
    return {"ok": True}


@router.post("/{pool_id}/stocks")
async def add_stock(
    pool_id: int,
    body: AddStockRequest,
    db: AsyncSession = Depends(get_db),
    manager: DataSourceManager = Depends(get_manager),
):
    ok = await add_stock_to_pool(db, manager, pool_id, body.ts_code)
    if not ok:
        raise HTTPException(status_code=404, detail="Pool not found")
    return {"ok": True}


@router.delete("/{pool_id}/stocks/{ts_code}")
async def remove_stock(
    pool_id: int, ts_code: str, db: AsyncSession = Depends(get_db)
):
    ok = await remove_stock_from_pool(db, pool_id, ts_code)
    if not ok:
        raise HTTPException(status_code=404, detail="Stock not in pool")
    return {"ok": True}
