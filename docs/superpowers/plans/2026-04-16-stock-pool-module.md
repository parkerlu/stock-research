# 股票池模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现多股票池 CRUD、池内股票管理与当日快照展示，点击池内股票联动 K 线图。

**Architecture:** 后端新增 StockPool/StockPoolItem 两个 ORM 模型，pool_service 提供 CRUD 和快照查询，pools router 提供 RESTful API。前端新增 PoolPanel 组件组、poolStore 状态管理、NavBar 导航切换。双栏布局：左侧面板按 mode 切换（行情/股票池），右侧 K 线始终存在并联动。

**Tech Stack:** FastAPI, SQLAlchemy 2.0 async, Alembic, PostgreSQL, React 18, TypeScript, Zustand, KLineChart v10

---

## File Structure

### Backend (create)
- `backend/app/services/pool_service.py` — 池 CRUD + 股票增删 + 快照查询
- `backend/app/routers/pools.py` — 池 API 路由
- `backend/alembic/versions/<auto>_create_pool_tables.py` — 数据库迁移

### Backend (modify)
- `backend/app/models/schema.py` — 新增 StockPool, StockPoolItem 模型
- `backend/app/main.py` — 注册 pools router

### Frontend (create)
- `frontend/src/types/pool.ts` — Pool 相关 TypeScript 类型
- `frontend/src/api/pools.ts` — 池 API 调用函数
- `frontend/src/stores/poolStore.ts` — Zustand 池状态管理
- `frontend/src/components/NavBar.tsx` — 顶部导航栏
- `frontend/src/components/PoolPanel/index.tsx` — 池面板容器
- `frontend/src/components/PoolPanel/PoolList.tsx` — 池列表（创建/删除）
- `frontend/src/components/PoolPanel/PoolDetail.tsx` — 池内股票快照表格
- `frontend/src/components/PoolPanel/AddStockDialog.tsx` — 搜索并添加股票弹窗

### Frontend (modify)
- `frontend/src/pages/QuotePage.tsx` — 加入 NavBar + mode 切换
- `frontend/src/App.css` — 新增池相关样式

---

### Task 1: Backend Models + Alembic Migration

**Files:**
- Modify: `backend/app/models/schema.py`
- Create: `backend/alembic/versions/<auto>_create_pool_tables.py`

- [ ] **Step 1: Add StockPool and StockPoolItem models to schema.py**

在 `schema.py` 文件末尾（`Favorite` class 之后）追加：

```python
class StockPool(Base):
    __tablename__ = "stock_pool"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(30), unique=True)
    description: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class StockPoolItem(Base):
    __tablename__ = "stock_pool_item"
    __table_args__ = (
        Index("ix_pool_item_unique", "pool_id", "ts_code", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pool_id: Mapped[int] = mapped_column(index=True)
    ts_code: Mapped[str] = mapped_column(String(12))
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
```

注意：`ForeignKey` 不加 ORM relationship，保持与现有代码风格一致（现有 model 都没用 relationship）。`pool_id` 的外键约束在迁移中通过 SQL 手动指定 cascade。

- [ ] **Step 2: Generate Alembic migration**

Run: `cd backend && alembic revision --autogenerate -m "create pool tables"`

- [ ] **Step 3: Edit generated migration to add ForeignKey with CASCADE**

打开生成的迁移文件，在 `stock_pool_item` 的 `pool_id` 列上确保有 `sa.ForeignKeyConstraint(['pool_id'], ['stock_pool.id'], ondelete='CASCADE')`。如果 autogenerate 没生成外键，手动在 `op.create_table('stock_pool_item', ...)` 调用末尾加：

```python
sa.ForeignKeyConstraint(['pool_id'], ['stock_pool.id'], ondelete='CASCADE'),
```

- [ ] **Step 4: Run migration locally to verify**

Run: `alembic upgrade head`

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/schema.py backend/alembic/versions/
git commit -m "feat(pool): add StockPool and StockPoolItem models + migration"
```

---

### Task 2: Backend Pool Service

**Files:**
- Create: `backend/app/services/pool_service.py`

- [ ] **Step 1: Create pool_service.py with all service functions**

```python
from datetime import date, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.datasources.manager import DataSourceManager
from app.models.schema import DailyCandle, StockBasic, StockPool, StockPoolItem


async def create_pool(db: AsyncSession, name: str, description: str | None = None) -> dict:
    pool = StockPool(name=name, description=description)
    db.add(pool)
    await db.commit()
    await db.refresh(pool)
    return {
        "id": pool.id,
        "name": pool.name,
        "description": pool.description,
        "created_at": pool.created_at.isoformat(),
    }


async def list_pools(db: AsyncSession) -> list[dict]:
    stmt = (
        select(
            StockPool,
            func.count(StockPoolItem.id).label("stock_count"),
        )
        .outerjoin(StockPoolItem, StockPool.id == StockPoolItem.pool_id)
        .group_by(StockPool.id)
        .order_by(StockPool.created_at.desc())
    )
    result = await db.execute(stmt)
    return [
        {
            "id": row.StockPool.id,
            "name": row.StockPool.name,
            "description": row.StockPool.description,
            "stock_count": row.stock_count,
            "created_at": row.StockPool.created_at.isoformat(),
        }
        for row in result.all()
    ]


async def get_pool_detail(db: AsyncSession, pool_id: int) -> dict | None:
    pool = await db.get(StockPool, pool_id)
    if not pool:
        return None

    # Get pool items with stock info
    stmt = (
        select(StockPoolItem, StockBasic.name, StockBasic.symbol)
        .join(StockBasic, StockPoolItem.ts_code == StockBasic.ts_code, isouter=True)
        .where(StockPoolItem.pool_id == pool_id)
        .order_by(StockPoolItem.added_at.desc())
    )
    result = await db.execute(stmt)
    items = result.all()

    stocks = []
    for row in items:
        item = row.StockPoolItem
        # Get latest 2 candles for change_pct calculation
        candle_stmt = (
            select(DailyCandle)
            .where(DailyCandle.ts_code == item.ts_code)
            .order_by(DailyCandle.trade_date.desc())
            .limit(2)
        )
        candle_result = await db.execute(candle_stmt)
        candles = candle_result.scalars().all()

        if candles:
            latest = candles[0]
            prev_close = float(candles[1].close) if len(candles) > 1 else None
            change_pct = (
                round((float(latest.close) - prev_close) / prev_close * 100, 2)
                if prev_close
                else None
            )
            stocks.append({
                "ts_code": item.ts_code,
                "name": row.name or "",
                "symbol": row.symbol or "",
                "close": round(float(latest.close), 2),
                "change_pct": change_pct,
                "volume": latest.vol,
                "turnover_rate": None,
                "trade_date": latest.trade_date.isoformat(),
            })
        else:
            stocks.append({
                "ts_code": item.ts_code,
                "name": row.name or "",
                "symbol": row.symbol or "",
                "close": None,
                "change_pct": None,
                "volume": None,
                "turnover_rate": None,
                "trade_date": None,
            })

    return {
        "id": pool.id,
        "name": pool.name,
        "description": pool.description,
        "stocks": stocks,
    }


async def update_pool(
    db: AsyncSession, pool_id: int, name: str | None = None, description: str | None = None
) -> dict | None:
    pool = await db.get(StockPool, pool_id)
    if not pool:
        return None
    if name is not None:
        pool.name = name
    if description is not None:
        pool.description = description
    pool.updated_at = datetime.now()
    await db.commit()
    await db.refresh(pool)
    return {
        "id": pool.id,
        "name": pool.name,
        "description": pool.description,
        "updated_at": pool.updated_at.isoformat(),
    }


async def delete_pool(db: AsyncSession, pool_id: int) -> bool:
    pool = await db.get(StockPool, pool_id)
    if not pool:
        return False
    await db.delete(pool)
    await db.commit()
    return True


async def add_stock_to_pool(
    db: AsyncSession, manager: DataSourceManager, pool_id: int, ts_code: str
) -> bool:
    """Add a stock to a pool. Triggers candle fetch if no cached data exists."""
    pool = await db.get(StockPool, pool_id)
    if not pool:
        return False

    # Check if already in pool
    stmt = select(StockPoolItem).where(
        StockPoolItem.pool_id == pool_id, StockPoolItem.ts_code == ts_code
    )
    existing = await db.execute(stmt)
    if existing.scalar_one_or_none():
        return True  # already exists, idempotent

    db.add(StockPoolItem(pool_id=pool_id, ts_code=ts_code))
    await db.commit()

    # Ensure candle cache exists — fetch last 365 days if empty
    candle_check = await db.execute(
        select(DailyCandle.ts_code)
        .where(DailyCandle.ts_code == ts_code)
        .limit(1)
    )
    if not candle_check.scalar_one_or_none():
        try:
            end = date.today()
            start = end - timedelta(days=365)
            df = await manager.fetch_daily(ts_code, start, end)
            if not df.empty:
                rows = df.to_dict("records")
                stmt_upsert = pg_insert(DailyCandle).values(rows)
                stmt_upsert = stmt_upsert.on_conflict_do_nothing(
                    index_elements=["ts_code", "trade_date"]
                )
                await db.execute(stmt_upsert)
                await db.commit()
        except Exception:
            pass  # non-critical: snapshot will show nulls until user views chart

    return True


async def remove_stock_from_pool(db: AsyncSession, pool_id: int, ts_code: str) -> bool:
    result = await db.execute(
        delete(StockPoolItem).where(
            StockPoolItem.pool_id == pool_id, StockPoolItem.ts_code == ts_code
        )
    )
    await db.commit()
    return result.rowcount > 0
```

- [ ] **Step 2: Verify import and syntax**

Run: `cd backend && python -c "from app.services.pool_service import create_pool, list_pools, get_pool_detail, update_pool, delete_pool, add_stock_to_pool, remove_stock_from_pool; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/pool_service.py
git commit -m "feat(pool): add pool service with CRUD and snapshot logic"
```

---

### Task 3: Backend Pool Router + Registration

**Files:**
- Create: `backend/app/routers/pools.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Create pools.py router**

```python
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


class AddStockRequest(BaseModel):
    ts_code: str


@router.post("")
async def create(body: CreatePoolRequest, db: AsyncSession = Depends(get_db)):
    return await create_pool(db, body.name, body.description)


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
```

- [ ] **Step 2: Register pools router in main.py**

在 `backend/app/main.py` 中，在 `from app.routers.quotes import router as quotes_router` 之后加一行：

```python
from app.routers.pools import router as pools_router
```

在 `app.include_router(quotes_router)` 之后加一行：

```python
    app.include_router(pools_router)
```

- [ ] **Step 3: Run migration in Docker and test API**

```bash
docker compose up -d --build backend
# Wait for startup
sleep 5
# Test create pool
curl -s -X POST http://localhost/api/pools -H "Content-Type: application/json" -d '{"name":"测试池"}' | python3 -m json.tool
# Test list pools
curl -s http://localhost/api/pools | python3 -m json.tool
# Test delete pool
curl -s -X DELETE http://localhost/api/pools/1
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/routers/pools.py backend/app/main.py
git commit -m "feat(pool): add pools REST API router"
```

---

### Task 4: Frontend Types + API Client

**Files:**
- Create: `frontend/src/types/pool.ts`
- Create: `frontend/src/api/pools.ts`

- [ ] **Step 1: Create pool types**

```typescript
// frontend/src/types/pool.ts

export interface Pool {
  id: number;
  name: string;
  description: string | null;
  stock_count: number;
  created_at: string;
}

export interface PoolStockSnapshot {
  ts_code: string;
  name: string;
  symbol: string;
  close: number | null;
  change_pct: number | null;
  volume: number | null;
  turnover_rate: number | null;
  trade_date: string | null;
}

export interface PoolDetail {
  id: number;
  name: string;
  description: string | null;
  stocks: PoolStockSnapshot[];
}
```

- [ ] **Step 2: Create pools API client**

```typescript
// frontend/src/api/pools.ts

import type { Pool, PoolDetail } from "../types/pool";

const BASE = "/api/pools";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function createPool(
  name: string,
  description?: string
): Promise<{ id: number }> {
  return json(`${BASE}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description }),
  });
}

export async function listPools(): Promise<Pool[]> {
  return json(`${BASE}`);
}

export async function getPoolDetail(poolId: number): Promise<PoolDetail> {
  return json(`${BASE}/${poolId}`);
}

export async function updatePool(
  poolId: number,
  data: { name?: string; description?: string }
): Promise<void> {
  await json(`${BASE}/${poolId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function deletePool(poolId: number): Promise<void> {
  await json(`${BASE}/${poolId}`, { method: "DELETE" });
}

export async function addStockToPool(
  poolId: number,
  tsCode: string
): Promise<void> {
  await json(`${BASE}/${poolId}/stocks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ts_code: tsCode }),
  });
}

export async function removeStockFromPool(
  poolId: number,
  tsCode: string
): Promise<void> {
  await json(`${BASE}/${poolId}/stocks/${tsCode}`, { method: "DELETE" });
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/pool.ts frontend/src/api/pools.ts
git commit -m "feat(pool): add frontend pool types and API client"
```

---

### Task 5: Frontend Pool Store

**Files:**
- Create: `frontend/src/stores/poolStore.ts`

- [ ] **Step 1: Create poolStore.ts**

```typescript
// frontend/src/stores/poolStore.ts
import { create } from "zustand";
import type { Pool, PoolDetail } from "../types/pool";
import {
  addStockToPool,
  createPool as apiCreate,
  deletePool as apiDelete,
  getPoolDetail,
  listPools,
  removeStockFromPool,
  updatePool as apiUpdate,
} from "../api/pools";

interface PoolState {
  pools: Pool[];
  currentPoolId: number | null;
  poolDetail: PoolDetail | null;
  loading: boolean;

  fetchPools: () => Promise<void>;
  selectPool: (id: number | null) => void;
  fetchDetail: (id: number) => Promise<void>;
  createPool: (name: string, description?: string) => Promise<void>;
  deletePool: (id: number) => Promise<void>;
  updatePool: (id: number, data: { name?: string; description?: string }) => Promise<void>;
  addStock: (tsCode: string) => Promise<void>;
  removeStock: (tsCode: string) => Promise<void>;
}

export const usePoolStore = create<PoolState>((set, get) => ({
  pools: [],
  currentPoolId: null,
  poolDetail: null,
  loading: false,

  fetchPools: async () => {
    try {
      const data = await listPools();
      set({ pools: data });
    } catch {
      /* ignore */
    }
  },

  selectPool: (id) => {
    set({ currentPoolId: id, poolDetail: null });
    if (id !== null) {
      get().fetchDetail(id);
    }
  },

  fetchDetail: async (id) => {
    set({ loading: true });
    try {
      const data = await getPoolDetail(id);
      set({ poolDetail: data, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  createPool: async (name, description) => {
    await apiCreate(name, description);
    await get().fetchPools();
  },

  deletePool: async (id) => {
    await apiDelete(id);
    const { currentPoolId } = get();
    if (currentPoolId === id) {
      set({ currentPoolId: null, poolDetail: null });
    }
    await get().fetchPools();
  },

  updatePool: async (id, data) => {
    await apiUpdate(id, data);
    await get().fetchPools();
    if (get().currentPoolId === id) {
      await get().fetchDetail(id);
    }
  },

  addStock: async (tsCode) => {
    const { currentPoolId } = get();
    if (currentPoolId === null) return;
    await addStockToPool(currentPoolId, tsCode);
    await get().fetchDetail(currentPoolId);
  },

  removeStock: async (tsCode) => {
    const { currentPoolId } = get();
    if (currentPoolId === null) return;
    await removeStockFromPool(currentPoolId, tsCode);
    await get().fetchDetail(currentPoolId);
  },
}));
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/stores/poolStore.ts
git commit -m "feat(pool): add Zustand pool store"
```

---

### Task 6: Frontend PoolPanel Components

**Files:**
- Create: `frontend/src/components/PoolPanel/index.tsx`
- Create: `frontend/src/components/PoolPanel/PoolList.tsx`
- Create: `frontend/src/components/PoolPanel/PoolDetail.tsx`
- Create: `frontend/src/components/PoolPanel/AddStockDialog.tsx`

- [ ] **Step 1: Create PoolList.tsx**

```tsx
// frontend/src/components/PoolPanel/PoolList.tsx
import { useEffect, useState } from "react";
import { usePoolStore } from "../../stores/poolStore";

export function PoolList() {
  const pools = usePoolStore((s) => s.pools);
  const currentPoolId = usePoolStore((s) => s.currentPoolId);
  const fetchPools = usePoolStore((s) => s.fetchPools);
  const selectPool = usePoolStore((s) => s.selectPool);
  const createPool = usePoolStore((s) => s.createPool);
  const deletePool = usePoolStore((s) => s.deletePool);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");

  useEffect(() => {
    fetchPools();
  }, [fetchPools]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    await createPool(newName.trim());
    setNewName("");
    setCreating(false);
  };

  return (
    <div className="pool-list-section">
      <div className="pool-list-header">
        <span>股票池</span>
        <button className="pool-add-btn" onClick={() => setCreating(true)}>+</button>
      </div>
      {creating && (
        <div className="pool-create-row">
          <input
            autoFocus
            placeholder="池名称"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
          />
          <button onClick={handleCreate}>OK</button>
          <button onClick={() => { setCreating(false); setNewName(""); }}>X</button>
        </div>
      )}
      <ul className="pool-list">
        {pools.map((p) => (
          <li
            key={p.id}
            className={currentPoolId === p.id ? "active" : ""}
            onClick={() => selectPool(p.id)}
          >
            <span className="pool-name">{p.name}</span>
            <span className="pool-count">{p.stock_count}</span>
            <button
              className="remove-btn"
              onClick={(e) => { e.stopPropagation(); deletePool(p.id); }}
            >
              x
            </button>
          </li>
        ))}
        {pools.length === 0 && <li className="empty">暂无股票池</li>}
      </ul>
    </div>
  );
}
```

- [ ] **Step 2: Create AddStockDialog.tsx**

```tsx
// frontend/src/components/PoolPanel/AddStockDialog.tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { searchStocks } from "../../api/quotes";
import type { StockInfo } from "../../types/quote";
import { usePoolStore } from "../../stores/poolStore";

interface Props {
  onClose: () => void;
}

export function AddStockDialog({ onClose }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<StockInfo[]>([]);
  const addStock = usePoolStore((s) => s.addStock);
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const doSearch = useCallback(async (q: string) => {
    if (q.length < 1) {
      setResults([]);
      return;
    }
    try {
      const data = await searchStocks(q);
      setResults(data);
    } catch {
      setResults([]);
    }
  }, []);

  useEffect(() => {
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => doSearch(query), 300);
    return () => clearTimeout(timerRef.current);
  }, [query, doSearch]);

  const handleSelect = async (stock: StockInfo) => {
    await addStock(stock.ts_code);
    onClose();
  };

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog-box" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <span>添加股票</span>
          <button onClick={onClose}>X</button>
        </div>
        <input
          autoFocus
          className="dialog-search"
          placeholder="输入代码或名称..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <ul className="dialog-results">
          {results.map((s) => (
            <li key={s.ts_code} onClick={() => handleSelect(s)}>
              <span className="code">{s.symbol}</span>
              <span className="name">{s.name}</span>
              {s.industry && <span className="industry">{s.industry}</span>}
            </li>
          ))}
          {query && results.length === 0 && <li className="empty">无结果</li>}
        </ul>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create PoolDetail.tsx**

```tsx
// frontend/src/components/PoolPanel/PoolDetail.tsx
import { useState } from "react";
import { usePoolStore } from "../../stores/poolStore";
import { useQuoteStore } from "../../stores/quoteStore";
import { AddStockDialog } from "./AddStockDialog";

export function PoolDetail() {
  const poolDetail = usePoolStore((s) => s.poolDetail);
  const loading = usePoolStore((s) => s.loading);
  const removeStock = usePoolStore((s) => s.removeStock);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const [showAdd, setShowAdd] = useState(false);

  if (!poolDetail) {
    return <div className="pool-detail-empty">选择一个股票池</div>;
  }

  if (loading) {
    return <div className="pool-detail-empty">加载中...</div>;
  }

  return (
    <div className="pool-detail">
      <div className="pool-detail-header">
        <span className="pool-detail-name">{poolDetail.name}</span>
        <button className="pool-add-btn" onClick={() => setShowAdd(true)}>+</button>
      </div>
      {poolDetail.description && (
        <div className="pool-detail-desc">{poolDetail.description}</div>
      )}
      <table className="pool-stock-table">
        <thead>
          <tr>
            <th>代码</th>
            <th>名称</th>
            <th>收盘</th>
            <th>涨跌%</th>
            <th>成交量</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {poolDetail.stocks.map((s) => (
            <tr
              key={s.ts_code}
              onClick={() => setCurrentStock(s.ts_code, s.name)}
              className="pool-stock-row"
            >
              <td className="code">{s.symbol}</td>
              <td>{s.name}</td>
              <td>{s.close != null ? s.close.toFixed(2) : "-"}</td>
              <td className={s.change_pct != null ? (s.change_pct >= 0 ? "up" : "down") : ""}>
                {s.change_pct != null ? `${s.change_pct > 0 ? "+" : ""}${s.change_pct.toFixed(2)}%` : "-"}
              </td>
              <td>{s.volume != null ? (s.volume / 10000).toFixed(0) + "万" : "-"}</td>
              <td>
                <button
                  className="remove-btn"
                  onClick={(e) => { e.stopPropagation(); removeStock(s.ts_code); }}
                >
                  x
                </button>
              </td>
            </tr>
          ))}
          {poolDetail.stocks.length === 0 && (
            <tr>
              <td colSpan={6} className="empty-cell">池内暂无股票，点击 + 添加</td>
            </tr>
          )}
        </tbody>
      </table>
      {showAdd && <AddStockDialog onClose={() => setShowAdd(false)} />}
    </div>
  );
}
```

- [ ] **Step 4: Create PoolPanel index.tsx**

```tsx
// frontend/src/components/PoolPanel/index.tsx
import { PoolList } from "./PoolList";
import { PoolDetail } from "./PoolDetail";

export function PoolPanel() {
  return (
    <div className="pool-panel">
      <PoolList />
      <div className="panel-divider" />
      <PoolDetail />
    </div>
  );
}
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd frontend && npx tsc -b --noEmit`

Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/PoolPanel/
git commit -m "feat(pool): add PoolPanel components (PoolList, PoolDetail, AddStockDialog)"
```

---

### Task 7: Frontend NavBar + Layout Integration

**Files:**
- Create: `frontend/src/components/NavBar.tsx`
- Modify: `frontend/src/pages/QuotePage.tsx`

- [ ] **Step 1: Create NavBar.tsx**

```tsx
// frontend/src/components/NavBar.tsx

export type AppMode = "quote" | "pool";

interface Props {
  mode: AppMode;
  onModeChange: (mode: AppMode) => void;
}

export function NavBar({ mode, onModeChange }: Props) {
  return (
    <nav className="app-navbar">
      <div className="navbar-brand">A股研究平台</div>
      <div className="navbar-tabs">
        <button
          className={mode === "quote" ? "active" : ""}
          onClick={() => onModeChange("quote")}
        >
          行情
        </button>
        <button
          className={mode === "pool" ? "active" : ""}
          onClick={() => onModeChange("pool")}
        >
          股票池
        </button>
      </div>
    </nav>
  );
}
```

- [ ] **Step 2: Modify QuotePage.tsx to integrate NavBar + mode switching**

Replace the entire content of `frontend/src/pages/QuotePage.tsx`:

```tsx
import { useState } from "react";
import { NavBar } from "../components/NavBar";
import type { AppMode } from "../components/NavBar";
import { SearchPanel } from "../components/SearchPanel";
import { PoolPanel } from "../components/PoolPanel";
import { ChartArea } from "../components/ChartArea";

export function QuotePage() {
  const [mode, setMode] = useState<AppMode>("quote");

  return (
    <div className="app-layout">
      <NavBar mode={mode} onModeChange={setMode} />
      <div className="app-body">
        {mode === "quote" ? <SearchPanel /> : <PoolPanel />}
        <ChartArea />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd frontend && npx tsc -b --noEmit`

Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/NavBar.tsx frontend/src/pages/QuotePage.tsx
git commit -m "feat(pool): add NavBar and integrate mode switching into QuotePage"
```

---

### Task 8: Frontend CSS Styles

**Files:**
- Modify: `frontend/src/App.css`

- [ ] **Step 1: Update layout styles and add pool-specific styles**

在 `App.css` 中，修改 `.quote-page` 并追加新样式。

将现有的 `.quote-page` 规则替换为：

```css
.app-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.app-navbar {
  display: flex;
  align-items: center;
  height: 40px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 0 16px;
  flex-shrink: 0;
}
.navbar-brand {
  font-size: 14px;
  font-weight: 600;
  color: var(--accent);
  margin-right: 24px;
}
.navbar-tabs { display: flex; gap: 4px; }
.navbar-tabs button {
  background: none;
  border: none;
  color: var(--text-secondary);
  padding: 6px 14px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}
.navbar-tabs button.active {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font-weight: 600;
}
.navbar-tabs button:hover:not(.active) { background: var(--bg-tertiary); }

.app-body {
  display: flex;
  flex: 1;
  min-height: 0;
}
```

然后在文件末尾追加股票池相关样式：

```css
/* Pool Panel */
.pool-panel {
  width: 280px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  overflow-y: auto;
}

.pool-list-section { padding: 0; }
.pool-list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  font-size: 13px;
  font-weight: 600;
}
.pool-add-btn {
  background: var(--bg-tertiary);
  border: none;
  color: var(--accent);
  width: 24px;
  height: 24px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  line-height: 24px;
}
.pool-add-btn:hover { background: var(--accent); color: #000; }

.pool-create-row {
  display: flex;
  gap: 4px;
  padding: 0 12px 8px;
}
.pool-create-row input {
  flex: 1;
  background: var(--bg-tertiary);
  border: none;
  border-radius: 4px;
  padding: 4px 8px;
  color: var(--text-primary);
  font-size: 12px;
  outline: none;
}
.pool-create-row button {
  background: var(--bg-tertiary);
  border: none;
  color: var(--text-secondary);
  padding: 4px 8px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
}

.pool-list {
  list-style: none;
  padding: 0 8px;
}
.pool-list li {
  padding: 6px 8px;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}
.pool-list li:hover { background: var(--bg-tertiary); }
.pool-list li.active { background: var(--bg-tertiary); color: var(--accent); }
.pool-list li.empty { color: var(--text-muted); cursor: default; justify-content: center; }
.pool-name { flex: 1; }
.pool-count {
  color: var(--text-muted);
  font-size: 11px;
  background: var(--bg-primary);
  padding: 1px 6px;
  border-radius: 8px;
}

/* Pool Detail */
.pool-detail {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.pool-detail-empty {
  color: var(--text-muted);
  text-align: center;
  padding: 24px 12px;
  font-size: 12px;
}
.pool-detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
}
.pool-detail-name { font-size: 13px; font-weight: 600; color: var(--accent); }
.pool-detail-desc {
  padding: 0 12px 6px;
  font-size: 11px;
  color: var(--text-muted);
}

.pool-stock-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}
.pool-stock-table th {
  text-align: left;
  padding: 4px 6px;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  font-weight: normal;
  position: sticky;
  top: 0;
  background: var(--bg-secondary);
}
.pool-stock-table td {
  padding: 5px 6px;
  border-bottom: 1px solid var(--bg-primary);
}
.pool-stock-row { cursor: pointer; }
.pool-stock-row:hover { background: var(--bg-tertiary); }
.pool-stock-table .code { color: var(--accent); }
.empty-cell { text-align: center; color: var(--text-muted); padding: 16px 6px; }

/* Add Stock Dialog */
.dialog-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.dialog-box {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  width: 360px;
  max-height: 400px;
  display: flex;
  flex-direction: column;
}
.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  font-weight: 600;
}
.dialog-header button {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 14px;
}
.dialog-search {
  margin: 10px 14px;
  background: var(--bg-tertiary);
  border: none;
  border-radius: 6px;
  padding: 8px 10px;
  color: var(--text-primary);
  font-size: 13px;
  outline: none;
}
.dialog-results {
  list-style: none;
  overflow-y: auto;
  flex: 1;
  padding: 0 8px 8px;
}
.dialog-results li {
  padding: 6px 8px;
  cursor: pointer;
  display: flex;
  gap: 8px;
  align-items: center;
  border-radius: 4px;
  font-size: 12px;
}
.dialog-results li:hover { background: var(--bg-tertiary); }
.dialog-results li.empty { color: var(--text-muted); cursor: default; justify-content: center; }
.dialog-results .code { color: var(--accent); }
.dialog-results .industry { color: var(--text-muted); font-size: 10px; }
```

- [ ] **Step 2: Verify TypeScript compiles and Vite builds**

Run: `cd frontend && npx tsc -b && npx vite build`

Expected: Build succeeds with no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.css
git commit -m "feat(pool): add pool panel and dialog CSS styles"
```

---

### Task 9: Docker Rebuild + E2E Verification

**Files:** No new files — integration testing only.

- [ ] **Step 1: Rebuild all Docker images**

```bash
cd /Users/chunyuanlu/WebApp/stock
docker compose down
docker compose build
docker compose up -d
```

- [ ] **Step 2: Wait for backend startup and verify migration**

```bash
sleep 8
docker compose logs backend | head -10
# Should see: "Running upgrade ... -> ..., create pool tables"
```

- [ ] **Step 3: Test pool APIs end-to-end**

```bash
# Create pool
curl -s -X POST http://localhost/api/pools \
  -H "Content-Type: application/json" \
  -d '{"name":"核心持仓","description":"长期看好"}' | python3 -m json.tool

# List pools
curl -s http://localhost/api/pools | python3 -m json.tool

# Add stock (should trigger candle fetch if no cache)
curl -s -X POST http://localhost/api/pools/1/stocks \
  -H "Content-Type: application/json" \
  -d '{"ts_code":"000001.SZ"}' | python3 -m json.tool

# Get pool detail with snapshot
curl -s http://localhost/api/pools/1 | python3 -m json.tool

# Remove stock
curl -s -X DELETE http://localhost/api/pools/1/stocks/000001.SZ

# Delete pool
curl -s -X DELETE http://localhost/api/pools/1
```

- [ ] **Step 4: Open browser at http://localhost and verify UI**

Check:
1. NavBar shows 行情/股票池 tabs
2. 行情 tab shows existing SearchPanel + chart (regression check)
3. 股票池 tab shows PoolPanel
4. Can create a pool, add stocks, see snapshot data
5. Clicking a stock in pool updates the K-line chart on the right

- [ ] **Step 5: Final commit if any hotfixes needed**

```bash
git add -A
git commit -m "fix(pool): integration fixes from E2E testing"
```
