# 股票池模块设计文档

## 目标

为 A 股单股 AI 策略研究平台实现股票池模块：支持多个自定义股票池的增删改查，池内展示当日快照列表（收盘价、涨跌幅、成交量），点击池内股票直接联动 K 线图。

## 页面布局

采用双栏布局，与行情模块共享右侧 K 线区域：

- **顶部导航栏**：「行情」/「股票池」两个 tab，切换左侧面板内容
- **左侧面板**：行情模式 → SearchPanel（现有）；股票池模式 → PoolPanel（新增）
- **右侧 K 线图**：始终存在，点击池内股票即联动更新

不引入 react-router，用顶层 mode 状态控制左侧面板切换。

## 数据模型

### StockPool

| 字段 | 类型 | 约束 |
|------|------|------|
| id | int | PK, auto |
| name | varchar(30) | unique, not null |
| description | varchar(200) | nullable |
| created_at | datetime | default now |
| updated_at | datetime | default now, on update |

### StockPoolItem

| 字段 | 类型 | 约束 |
|------|------|------|
| id | int | PK, auto |
| pool_id | int | FK → stock_pool.id, cascade delete |
| ts_code | varchar(12) | not null |
| added_at | datetime | default now |
| | | UNIQUE(pool_id, ts_code) |

删除池时级联删除池内所有条目。

## 后端 API

### 池管理

| 方法 | 路径 | 请求体/参数 | 返回 |
|------|------|-------------|------|
| POST | `/api/pools` | `{name, description?}` | `{id, name, description, created_at}` |
| GET | `/api/pools` | - | `[{id, name, description, stock_count, created_at}]` |
| GET | `/api/pools/{id}` | - | 池详情 + 池内股票快照列表（见下文） |
| PUT | `/api/pools/{id}` | `{name?, description?}` | `{id, name, description, updated_at}` |
| DELETE | `/api/pools/{id}` | - | `{ok: true}` |

### 池内股票管理

| 方法 | 路径 | 请求体 | 返回 |
|------|------|--------|------|
| POST | `/api/pools/{id}/stocks` | `{ts_code}` | `{ok: true}` |
| DELETE | `/api/pools/{id}/stocks/{ts_code}` | - | `{ok: true}` |

### 池详情返回格式

```json
{
  "id": 1,
  "name": "核心持仓",
  "description": "长期看好",
  "stocks": [
    {
      "ts_code": "000001.SZ",
      "name": "平安银行",
      "close": 11.09,
      "change_pct": -0.98,
      "volume": 680914,
      "turnover_rate": null,
      "trade_date": "2026-04-15"
    }
  ]
}
```

## 快照数据策略

- 数据来源：本地 `daily_candle` 表最新一条记录，零外部 API 调用
- 涨跌幅计算：`(today.close - yesterday.close) / yesterday.close * 100`，从最新两条日线计算
- 添加股票时，若该股票无本地日线缓存，触发一次 TuShare 拉取并存入 DB
- 换手率：当前无流通股本数据，暂返回 null（二期可补充）

## 后端分层

遵循现有 router → service → model 模式：

- `backend/app/models/schema.py` — 新增 StockPool、StockPoolItem 模型
- `backend/app/services/pool_service.py` — 池 CRUD + 股票增删 + 快照查询逻辑
- `backend/app/routers/pools.py` — API 路由
- `backend/app/main.py` — 注册 pools router
- `backend/alembic/versions/` — 新 migration

## 前端结构

```
src/
├── components/
│   ├── NavBar.tsx              ← 新增：顶部「行情/股票池」导航
│   └── PoolPanel/              ← 新增
│       ├── index.tsx           ← 面板容器
│       ├── PoolList.tsx        ← 池列表（创建/删除操作）
│       ├── PoolDetail.tsx      ← 池内股票快照表格
│       └── AddStockDialog.tsx  ← 搜索并添加股票弹窗
├── api/
│   └── pools.ts                ← 新增：池 API 调用
├── stores/
│   └── poolStore.ts            ← 新增：Zustand 池状态管理
├── types/
│   └── pool.ts                 ← 新增：Pool/PoolStockSnapshot 类型
└── pages/
    └── QuotePage.tsx           ← 修改：加入 NavBar + mode 切换
```

### 状态管理（poolStore）

- `pools`: 池列表
- `currentPoolId`: 当前选中的池 ID
- `poolStocks`: 当前池内股票快照数据
- `fetchPools()`: 加载池列表
- `fetchPoolDetail(id)`: 加载池详情 + 快照
- `createPool(name, description?)`: 创建池
- `deletePool(id)`: 删除池
- `addStock(poolId, tsCode)`: 添加股票到池
- `removeStock(poolId, tsCode)`: 从池移除股票

### 交互流程

1. 用户点击顶部「股票池」tab → 左侧切换到 PoolPanel
2. PoolPanel 显示池列表，用户选择一个池 → 加载池内股票快照
3. 股票表格展示：代码、名称、收盘价、涨跌幅、成交量
4. 点击表格中某只股票 → 调用 `quoteStore.setCurrentStock()` → 右侧 K 线联动
5. 用户可通过「+」按钮打开 AddStockDialog，搜索并添加股票到池
6. 用户可在池列表中创建新池、删除池；在股票表格中移除单只股票

## 技术要点

- 前端不引入 react-router，用 mode 状态切换
- 后端池 CRUD 使用 Pydantic schema 做请求/响应验证
- 快照批量查询用一条 SQL 完成（JOIN stock_basic + 子查询取最新两条日线）
- 添加股票时确保日线缓存存在，必要时触发拉取
