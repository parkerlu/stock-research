# 行情模块设计文档

## 概述

行情模块是 A 股单股 AI 策略研究平台的第一个功能模块，提供股票搜索、K 线图表展示、技术指标分析和画线标注能力。单用户本地使用，数据范围仅限单股走势（OHLCV）。

## 架构

```
Frontend (React + KLineChart + Zustand)
  │
  │ HTTP REST
  ▼
Backend (FastAPI)
  │
  ▼
Data Layer (DataSourceManager → PostgreSQL 缓存)
  ├── TuShareProvider (主数据源)
  └── AKShareProvider (兜底数据源)
```

- 技术指标在前端由 KLineChart 内置引擎计算，后端只提供原始 OHLCV 数据
- 周/月线由后端基于日线 SQL 聚合，保证数据口径一致
- 数据缓存策略：首次请求从远端拉取并存入 PostgreSQL，后续走库，增量补最新交易日

## 数据库设计

### stock_basic（股票基本信息）

| 字段 | 类型 | 说明 |
|---|---|---|
| ts_code | VARCHAR(12) PK | 如 000001.SZ |
| symbol | VARCHAR(6) | 如 000001 |
| name | VARCHAR(20) | 如 平安银行 |
| area | VARCHAR(10) | 如 深圳 |
| industry | VARCHAR(20) | 如 银行 |
| market | VARCHAR(10) | 如 主板 |
| list_date | DATE | 上市日期 |
| is_active | BOOLEAN | 默认 TRUE |

### daily_candle（日线行情，核心表）

| 字段 | 类型 | 说明 |
|---|---|---|
| ts_code | VARCHAR(12) | 联合主键 |
| trade_date | DATE | 联合主键 |
| open | DECIMAL(12,4) | 开盘价 |
| high | DECIMAL(12,4) | 最高价 |
| low | DECIMAL(12,4) | 最低价 |
| close | DECIMAL(12,4) | 收盘价 |
| vol | BIGINT | 成交量（手）|
| amount | DECIMAL(18,4) | 成交额（千元）|
| adj_factor | DECIMAL(12,6) | 复权因子 |
| source | VARCHAR(10) | tushare 或 akshare |

索引：`(ts_code, trade_date DESC)` 用于范围查询。

### search_history（搜索历史）

| 字段 | 类型 | 说明 |
|---|---|---|
| id | SERIAL PK | 自增 |
| ts_code | VARCHAR(12) | 股票代码 |
| searched_at | TIMESTAMP | 默认 NOW() |

### favorite（收藏）

| 字段 | 类型 | 说明 |
|---|---|---|
| id | SERIAL PK | 自增 |
| ts_code | VARCHAR(12) UNIQUE | 股票代码 |
| created_at | TIMESTAMP | 默认 NOW() |

### 周/月线聚合

不单独存表，基于 `daily_candle` SQL 聚合：

- 周线：按自然周（周一至周五）分组，open 取周一，close 取周五，high/low 取极值，vol/amount 求和
- 月线：按自然月分组，同理
- 聚合前先用 adj_factor 计算前复权价格

## 双数据源切换机制

```
请求 K 线数据
  ├─ 1. 查 PostgreSQL 缓存
  │     ├─ 完整 → 直接返回
  │     └─ 缺失/需更新 ↓
  ├─ 2. TuShareProvider
  │     ├─ 成功 → 入库 source=tushare，返回
  │     └─ 失败 ↓
  └─ 3. AKShareProvider（兜底）
        ├─ 成功 → 入库 source=akshare，返回
        └─ 失败 → 返回 502 错误
```

- TuShare 频率限制：按积分档位每分钟 200-500 次
- AKShare 基于东方财富爬虫，无官方 SLA，仅作兜底
- source 字段标记来源，便于审计

## API 设计

### GET /api/quotes/search?q={keyword}&limit=10

股票搜索联想，按代码或名称模糊匹配。

返回：`[{ ts_code, symbol, name, industry }]`

### GET /api/quotes/{symbol}/candles?tf=1d|1w|1m&from=YYYY-MM-DD&to=YYYY-MM-DD

K 线数据（含前复权）。若缓存缺失自动触发数据源拉取。

返回：`{ symbol, tf, candles: [{ timestamp, open, high, low, close, volume, amount }] }`

### GET /api/quotes/{symbol}/snapshot

当日实时快照。交易日未拉取时实时从数据源获取。

返回：`{ symbol, name, price, change, change_pct, open, high, low, vol, amount, turnover }`

### GET /api/search-history?limit=20

最近搜索记录。

返回：`[{ ts_code, name, searched_at }]`

### POST /api/favorites/{ts_code}

添加收藏。

### DELETE /api/favorites/{ts_code}

取消收藏。

### GET /api/favorites

收藏列表，附带最新价格和涨跌幅。

返回：`[{ ts_code, name, price, change_pct }]`

### 错误码

| 状态码 | 场景 |
|---|---|
| 200 | 正常返回 |
| 400 | 参数错误（无效代码、无效周期等）|
| 404 | 股票代码不存在 |
| 502 | 上游数据源均不可用 |
| 503 | 数据源限流，稍后重试 |

## 前端设计

### 页面布局

左面板 (240px) + 右图表区的双栏布局。

**左面板 SearchPanel：**

- SearchBar：防抖 300ms，最少 1 字符触发联想，下拉展示匹配结果（代码、名称、行业）
- RecentList / FavoriteList：标签页切换，最近查看最多 20 条，收藏列表带涨跌幅
- SnapshotCard：跟随当前股票刷新，展示开/高/低/量/额/换手率

**右侧 ChartArea：**

- Toolbar：日/周/月切换 | 指标下拉管理（已激活指标以标签展示）| 画线工具栏 | 多周期联动开关
- MainChart：KLineChart 实例，K 线主图 + 叠加指标（MA/EMA/BOLL），左上角显示当前数值
- SubCharts：副图指标区（MACD/KDJ/RSI 等），可切换
- VolumeChart：底部成交量柱状图
- DataZoom：底部拖拽缩放条

### 技术指标

全部由 KLineChart 前端计算：

- 均线类：MA、EMA、BOLL
- 趋势类：MACD、DMI、SAR
- 摆动类：KDJ、RSI、WR
- 量能类：OBV、VOL-MA

### 画线工具

趋势线、水平线、垂直线、平行通道、斐波那契回撤、矩形区域、文字标注、箭头标记。画线数据持久化到 localStorage。

### 多周期联动

点击「多周期联动」按钮后，三个 KLineChart 实例（日/周/月）并排显示，十字光标和时间轴同步。

### 状态管理

Zustand store 管理：

- 当前股票代码和名称
- 当前周期（日/周/月）
- 搜索历史列表
- 收藏列表
- 快照数据
- 联动模式开关

## 工程结构

```
stock/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI 入口
│   │   ├── config.py                # 配置
│   │   ├── routers/
│   │   │   └── quotes.py            # 行情 API 路由
│   │   ├── services/
│   │   │   ├── quote_service.py     # 行情业务逻辑
│   │   │   └── search_service.py    # 搜索、历史、收藏
│   │   ├── datasources/
│   │   │   ├── manager.py           # DataSourceManager
│   │   │   ├── tushare_provider.py  # TuShare 接入
│   │   │   └── akshare_provider.py  # AKShare 接入
│   │   ├── models/
│   │   │   └── schema.py            # SQLAlchemy 模型
│   │   └── db.py                    # 数据库连接
│   ├── alembic/                     # 数据库迁移
│   ├── tests/
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   └── QuotePage.tsx
│   │   ├── components/
│   │   │   ├── SearchPanel/
│   │   │   │   ├── SearchBar.tsx
│   │   │   │   ├── RecentList.tsx
│   │   │   │   ├── FavoriteList.tsx
│   │   │   │   └── SnapshotCard.tsx
│   │   │   └── ChartArea/
│   │   │       ├── Toolbar.tsx
│   │   │       ├── MainChart.tsx
│   │   │       ├── DrawingManager.ts
│   │   │       └── LinkedView.tsx
│   │   ├── stores/
│   │   │   └── quoteStore.ts
│   │   ├── api/
│   │   │   └── quotes.ts
│   │   └── types/
│   │       └── quote.ts
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── docs/
└── docker-compose.yml               # PostgreSQL
```

### 技术选型

| 层 | 选型 |
|---|---|
| 前端框架 | React 18 + TypeScript |
| 图表库 | KLineChart |
| 状态管理 | Zustand |
| 构建工具 | Vite |
| 后端框架 | FastAPI + Uvicorn |
| ORM | SQLAlchemy 2.0 + Alembic |
| 数据库 | PostgreSQL（Docker 本地运行）|
| 数据源 | TuShare Pro + AKShare |
