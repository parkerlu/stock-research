# 策略工厂 V1 + 回测引擎 设计文档

## 目标

为单只股票自动生成技术指标策略候选集，通过回测引擎评估筛选，保留 Top 50 入策略库。用户可手动运行回测、查看报告、Pin 冻结策略。

## 架构概览

```
用户触发工厂运行
       ↓
┌─────────────────────┐
│   策略工厂服务       │
│  1. 模板 × 参数扫描  │  → 生成 300-800 候选策略
│  2. 逐个调用回测引擎  │
│  3. 硬过滤 + 排序     │  → 净收益>0 且 回撤<=35%
│  4. 保留 Top50 入库   │  → 按年化收益率排序
└─────────────────────┘
       ↓
┌─────────────────────┐
│     回测引擎         │
│  输入: K线 + 信号     │
│  执行: 三次分仓交易   │
│  输出: 指标 + 曲线    │
└─────────────────────┘
```

## 一、回测引擎

### 1.1 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| candles | list[dict] | OHLCV 日线数据 |
| signals | list[dict] | `{date, action}` action = "buy" / "sell" |
| initial_capital | float | 初始资金，默认 10000 |
| position_ratios | list[int] | 三次分仓比例，如 [40,30,30]，合计 100 |

### 1.2 三次分仓逻辑

- **状态机**：`空仓(0)` → `1/3仓(1)` → `2/3仓(2)` → `满仓(3)`
- **买入信号**：
  - 空仓时收到 buy → 用 `ratio[0]%` 的初始资金买入 → 进入 1/3 仓
  - 1/3 仓时收到 buy → 用 `ratio[1]%` 的初始资金买入 → 进入 2/3 仓
  - 2/3 仓时收到 buy → 用 `ratio[2]%` 的初始资金买入 → 满仓
  - 满仓时收到 buy → 忽略
- **卖出信号**：
  - 任何持仓时收到 sell → 全部平仓 → 回到空仓
  - 空仓时收到 sell → 忽略
- **买入价格**：信号当日收盘价
- **卖出价格**：信号当日收盘价
- **摩擦成本**：V1 默认无摩擦（PRD 一期要求）

### 1.3 输出指标

| 指标 | 计算方式 |
|------|---------|
| net_profit | 最终资产 - 初始资金 |
| net_profit_pct | net_profit / initial_capital × 100 |
| annualized_return | (final / initial) ^ (252/交易天数) - 1 |
| max_drawdown | max((peak - trough) / peak) |
| win_rate | 盈利交易次数 / 总交易次数 |
| total_trades | 完整买卖周期数 |
| profit_factor | 总盈利 / 总亏损 |
| final_capital | 回测结束时的总资产 |
| trades | 交易明细 list[{entry_date, exit_date, entry_price, exit_price, shares, pnl}] |
| equity_curve | list[{date, equity}] 每日资产净值 |

### 1.4 API

```
POST /api/backtests/run
Body: {
  "ts_code": "600519.SH",
  "strategy_id": 1,        // 可选，指定已有策略
  "start_date": "2016-01-01",
  "end_date": "2026-01-01",
  "initial_capital": 10000,
  "position_ratios": [40, 30, 30]
}
Response: { "id": "uuid", "status": "running" }

GET /api/backtests/{id}/report
Response: { "status": "completed", "metrics": {...}, "trades": [...], "equity_curve": [...] }
```

## 二、策略工厂 V1

### 2.1 策略模板

每个模板是一个 Python 函数：输入 OHLCV DataFrame，输出 buy/sell 信号序列。

| 模板 | 参数 | 候选数量 |
|------|------|---------|
| MA 双均线交叉 | fast_period(3-30), slow_period(10-120) | ~100 |
| RSI 超买超卖 | period(6-28), overbought(65-85), oversold(15-35) | ~80 |
| MACD 信号交叉 | fast(8-16), slow(20-30), signal(5-12) | ~60 |
| 布林带突破 | period(10-30), std_dev(1.5-3.0) | ~40 |
| KDJ 金叉死叉 | k_period(5-21), d_period(3-9) | ~40 |
| MA + RSI 组合 | ma_period + rsi_period + thresholds | ~120 |
| MACD + 成交量 | macd_params + vol_ma_period | ~80 |
| 布林带 + RSI | bb_params + rsi_params | ~80 |

合计约 **600 候选**，满足 PRD 300-800 要求。

### 2.2 参数扫描

- 每个模板定义参数名、类型、范围、步长
- 使用 `itertools.product` 生成全组合
- 去除无意义组合（如 fast_period >= slow_period）

### 2.3 工厂运行流程

```
1. 接收请求 (ts_code, cutoff_date, position_ratios)
2. 从 DB 获取该股票 cutoff_date 向前 10 年的日线数据
3. 生成全部候选策略 (模板 × 参数)
4. 创建 FactoryJob 记录 (status=running)
5. 后台执行:
   a. 对每个候选: 生成信号 → 回测 → 收集结果
   b. 定期更新 job.evaluated 计数
6. 硬过滤: net_profit > 0 AND max_drawdown <= 35%
7. 按 annualized_return 降序排序
8. 保留 Top50:
   - 已 pin 的策略不被淘汰
   - 删除该股票旧的非 pin 策略
   - 插入新 Top50
9. 更新 job.status = completed
```

### 2.4 异步执行

- V1 使用 `asyncio.to_thread` + `concurrent.futures.ProcessPoolExecutor` 并行回测
- 通过 `FactoryJob` 表追踪进度（不引入 Redis/Celery）
- 前端轮询 job 状态

### 2.5 API

```
POST /api/strategy-factory/run
Body: {
  "ts_code": "600519.SH",
  "cutoff_date": "2026-04-01",
  "position_ratios": [40, 30, 30]
}
Response: { "job_id": "uuid", "status": "running", "total_candidates": 600 }

GET /api/strategy-factory/jobs/{job_id}
Response: {
  "job_id": "uuid",
  "status": "running|completed|failed",
  "total_candidates": 600,
  "evaluated": 350,
  "passed": 42,
  "error": null
}

GET /api/strategies?ts_code=600519.SH
Response: [
  { "id": 1, "name": "MA_5_20", "template": "ma_crossover",
    "annualized_return": 0.32, "max_drawdown": 0.18, "win_rate": 0.65,
    "is_pinned": false, ... },
  ...
]

POST /api/strategies/{id}/pin
DELETE /api/strategies/{id}/pin
```

## 三、数据库模型

### 3.1 新增表

```python
class FactoryJob(Base):
    __tablename__ = "factory_job"
    id = Column(String(36), primary_key=True)          # UUID
    ts_code = Column(String(20), nullable=False, index=True)
    status = Column(String(20), default="pending")     # pending/running/completed/failed
    total_candidates = Column(Integer, default=0)
    evaluated = Column(Integer, default=0)
    passed = Column(Integer, default=0)
    config = Column(JSON)                              # position_ratios, cutoff_date, etc.
    error = Column(Text)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)

class Strategy(Base):
    __tablename__ = "strategy"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    template = Column(String(50), nullable=False)
    parameters = Column(JSON, nullable=False)
    is_pinned = Column(Boolean, default=False)
    annualized_return = Column(Float)
    net_profit = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    profit_factor = Column(Float)
    final_capital = Column(Float)
    metrics = Column(JSON)                             # 完整指标快照
    job_id = Column(String(36))                        # 关联工厂任务
    created_at = Column(DateTime, default=func.now())

class BacktestRun(Base):
    __tablename__ = "backtest_run"
    id = Column(String(36), primary_key=True)          # UUID
    ts_code = Column(String(20), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategy.id", ondelete="SET NULL"))
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(Float, default=10000)
    position_ratios = Column(JSON)
    status = Column(String(20), default="pending")
    metrics = Column(JSON)
    trades = Column(JSON)
    equity_curve = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
```

## 四、后端文件结构

```
backend/app/
├── routers/
│   ├── quotes.py          (已有)
│   ├── pools.py           (已有)
│   ├── strategies.py      (新增) — 策略 CRUD + pin
│   ├── factory.py         (新增) — 工厂运行 + 状态查询
│   └── backtests.py       (新增) — 回测运行 + 报告查询
├── services/
│   ├── quote_service.py   (已有)
│   ├── pool_service.py    (已有)
│   ├── backtest_engine.py (新增) — 核心回测执行逻辑
│   ├── factory_service.py (新增) — 工厂编排逻辑
│   └── strategy_templates/(新增) — 策略模板目录
│       ├── base.py        — 模板基类
│       ├── ma_crossover.py
│       ├── rsi.py
│       ├── macd.py
│       ├── bollinger.py
│       ├── kdj.py
│       └── combined.py    — 组合策略
├── models/
│   └── schema.py          (扩展) — 新增 3 张表
└── main.py                (扩展) — 注册新路由
```

## 五、前端页面（最小可用）

V1 前端聚焦功能验证，不做复杂 UI：

- **策略列表页**：在 NavBar 新增"策略"标签，展示当前股票的 Top50 策略表格
- **工厂运行面板**：按钮 + 进度条（轮询 job 状态）
- **回测报告弹窗**：点击策略行 → 展示核心指标 + 资金曲线图

## 六、关键约束

- 每只股票最多 50 条策略（非 pin），pin 不计入上限
- 策略入库门槛：`net_profit > 0` 且 `max_drawdown <= 0.35`
- 排序：`annualized_return` 降序
- 回测数据窗口：cutoff_date 向前 10 年
- 初始资金默认 10000
- V1 无交易摩擦（手续费、滑点）
