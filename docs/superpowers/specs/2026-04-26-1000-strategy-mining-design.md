# 1000-Strategy Mining Pipeline — Design

- 日期：2026-04-26
- 状态：草稿
- 数据：1004 只 A 股 × ~7 年（2019-01-02 → 2026-04-24，1772 个交易日）

## 目标

挖出 1000 个**结构与参数都不同**的策略，每个独立通过以下硬性指标：

| # | 指标 | 阈值 |
|---|---|---|
| 1 | 胜率 (win_rate) | ≥ 75% |
| 2 | 平均单笔获利 | ≥ 6% |
| 3 | 单笔最大亏损 | ≤ 10%（硬止损） |
| 4 | 累计总收益 | ≥ 10% |

附加质量控制：
- **OOS gate**：策略必须在留出集（2024-09 → 2026-04）独立通过 1/2/4 三项（3 是硬止损天然满足）
- **最小交易数**：每策略全周期 ≥ 20 笔（防过拟合到几笔幸运单）
- **股票分散度**：至少在 5 只不同股票上有交易（防策略只能在 1-2 只上work）

## 非目标

- 不做 portfolio-level（多策略组合）回测；每策略独立评估。
- 不优化交易费率/滑点（统一 0.05% 单边）。
- 不写实盘对接，只产出 JSON 配置 + 回测报告。

## 架构（C 方案）

```
Data Layer (cache once)
  ├─ 1004 stocks OHLCV (adj-corrected)
  ├─ ML score cache (39 features, 5-seed ensemble)
  ├─ GP factor cache (V1/V2/V3 top-50 each)
  └─ TDX indicator cache (买卖很准, 动力线, KDJ multi-period, DMI, etc)

Strategy Layer
  └─ 50 families × 20 param variants = 1000 candidates

Backtest Engine (existing)
  └─ Per strategy → per stock → trades → aggregate metrics

Filter & Output
  ├─ Hard gate: 4 user criteria + OOS gate + min trades + diversity
  ├─ Rank by composite quality score (信息比率 + 胜率 + 累计收益)
  └─ Top-1000 → docs/strategy_mine/1000_strategies.json
```

## 50 个策略族

### A. ML 模型 + 退出变体（10 个）
基础：5-seed XGBoost ensemble (39 特征 + 动力线 + 买卖很准 + CSF)。买入触发统一 `score ≥ buy_threshold`，区别在退出。

| ID | 退出逻辑 |
|---|---|
| A1 | classic trail：peak × (1 - trail_pct) |
| A2 | ATR trail：peak − atr_mult × ATR |
| A3 | Chandelier exit：HHV(high,N) − atr_mult × ATR |
| A4 | Step trail：每涨 5% 移止盈线 |
| A5 | Time stop only：固定 N 日强平 |
| A6 | 急买奇准 sell signal exit |
| A7 | 动力线 cross down 3.5 exit |
| A8 | DMI bear cross exit |
| A9 | Multi-period KDJ death cross exit |
| A10 | Score decay exit：score 跌破 sell_threshold |

### B. GP 因子（10 个）
利用现有 GP V1/V2/V3 挖矿器输出的 top-50 公式池。

| ID | 进场逻辑 |
|---|---|
| B1 | Top-1 V1 IC 因子，分位数 ≥ q |
| B2 | Top-1 V2 cross-sectional 因子 rank ≥ k |
| B3 | Top-1 V3 niching 因子 |
| B4 | V1+V2+V3 加权 composite |
| B5 | GP rank z-score ≥ z |
| B6 | GP factor momentum (5d 因子变化率) |
| B7 | GP cross-up (因子穿越自身均值) |
| B8 | GP percentile filter (跨股票) |
| B9 | GP factor + ATR exit 组合 |
| B10 | GP factor + classic trail 组合 |

### C. TDX 指标组合（10 个）
基础：本仓库已实现的 7 大 TDX 指标（买卖很准、动力线、KDJ 多周期、DMI、底点组合等）。

| ID | 进场逻辑 |
|---|---|
| C1 | 急买奇准 (买卖很准) |
| C2 | 动力线 stage_bottom (cross-up 0.2) |
| C3 | 动力线 stage_watch (cross-up 0.5) |
| C4 | 动力线 + 买卖很准 confluence |
| C5 | 点点结合 buy signal |
| C6 | DMI bull cross (PDI > MDI) + ADX > 20 |
| C7 | 多周期 KDJ resonance (日+周+月共振) |
| C8 | 动力线 + KDJ confluence |
| C9 | 多周期 KDJ + DMI |
| C10 | 动力线 stage_bottom + 急买奇准 sequential |

### D. 经典 momentum/breakout（10 个）

| ID | 进场逻辑 |
|---|---|
| D1 | 20-day 高点突破 |
| D2 | 50-day 高点突破 |
| D3 | Donchian channel breakout (N=20/55) |
| D4 | Bollinger Band squeeze + breakout |
| D5 | MACD bull cross + signal > 0 |
| D6 | RSI(14) cross above 30 from oversold |
| D7 | SMA(10/30) golden cross |
| D8 | Ichimoku cloud breakout |
| D9 | VWAP bounce (回踩 VWAP 收阳) |
| D10 | Volume spike (vol > 2× MA20) + 价格收阳 |

### E. 横截面排序（10 个）
每日横截面排序，取 Top-K 持有，到下个调仓日。

| ID | 排序信号 |
|---|---|
| E1 | Top-1 by ML score |
| E2 | Top-3 by ML score |
| E3 | Top-5 by ML score |
| E4 | Top-1 by GP composite |
| E5 | Top-3 by ML+GP 复合 |
| E6 | Bottom-K mean revert (反向) |
| E7 | Sector rotation (60d 相对强度 Top-N) |
| E8 | 60d return rank Top-K (earnings momentum proxy) |
| E9 | Low-vol Top-K (年化波动率 Bottom-K) |
| E10 | Multi-factor screener (ML+GP+TDX 投票) |

## 20 个参数变体

每族用 LHS（拉丁超立方）抽样 20 组参数：

| 参数 | 范围 |
|---|---|
| buy_threshold | 0.30 ~ 0.65 |
| exit_threshold | 0.15 ~ 0.40 |
| stop_loss | 固定 -10%（满足约束 3） |
| trail_pct | 0.02 ~ 0.05 |
| atr_mult | 1.0 ~ 3.5 |
| time_stop | 30 ~ 120 日 |
| trail_activation | 0.05 ~ 0.15 |

不适用的参数（如 D 族不用 buy_threshold）按结构内置默认。

## 回测协议

- **手续费**：双边各 0.05%（千万 5 印花税 + 万一佣金 + 万一过户费近似）
- **滑点**：0%（next-bar open 进场，已是保守口径）
- **止损**：硬止损 -10%（约束 3 强制）
- **资金**：每笔等权 ¥10,000，无杠杆，不复利（避免规模偏置）
- **回测窗口**：
  - In-sample (IS)：2019-01-02 → 2024-08-31（5.7 年）
  - Out-of-sample (OOS)：2024-09-01 → 2026-04-24（1.7 年）
  - 全周期合并报告

## 数据流

1. **Pre-cache**（一次性）
   - `cache/stocks.parquet` (1004 × 1772 bars OHLCV+adj)
   - `cache/ml_scores.parquet` (1004 × 1772 score)
   - `cache/gp_factors.parquet` (top-50 V1/V2/V3 因子值)
   - `cache/tdx_indicators.parquet` (买卖很准/动力线/KDJ/DMI/...)

2. **Strategy execution**：每策略对 1004 只股票产出 trades，agg 4 项指标 + OOS gate。

3. **Filter**：
   - 硬过滤 → 保留通过 4 项 + OOS gate + 最小 20 笔 + ≥5 只股票分散
   - 软排序（如果通过的策略 > 1000 个）：按 `composite_score = win_rate × log(total_return) × sqrt(n_trades) / max_drawdown`

4. **Output**：
   ```json
   {
     "id": "A1_v07",
     "family": "ML+ClassicTrail",
     "params": {"buy_threshold": 0.45, "trail_pct": 0.03, "stop_loss": -0.10, "time_stop": 75},
     "metrics_full": {"win_rate": 0.78, "avg_trade": 0.082, "max_loss": -0.099, "total_return": 1.34, "n_trades": 142},
     "metrics_oos": {"win_rate": 0.76, "avg_trade": 0.071, "total_return": 0.18, "n_trades": 28},
     "stocks_traded": 87
   }
   ```

## 风险与限制

1. **过拟合**：1000 个策略中可能多数是局部过拟合。OOS gate + 最小交易数 + 股票分散度三重过滤已防御，但仍需用户警惕。
2. **数据偏移**：1004 只股票在 2019-01 起步，存在生存者偏差（已上市的）。无法完全规避。
3. **TuShare 数据质量**：复权因子由本仓库 backward-adjust 计算，已与图表一致。
4. **执行假设**：next-bar open 进场假设流动性充足，对小盘股或停牌期可能失真。
5. **算力**：50 族 × 20 参数 × 1004 股 × 1772 bars = ~1.78 亿次回测 bar 评估。
   - 用 numpy 向量化 + score 缓存预计 1-3 小时完成。
   - 若慢可分批存中间结果（按族分批 commit）。

## Top-5 输出（用户要求）

挖矿完成后，按 `composite_score` 排序选 **Top 5**，注册到 `TEMPLATE_REGISTRY`：

| Template ID | Composite 排名 | 备注 |
|---|---|---|
| `426-1` | #1 | 综合最优 |
| `426-2` | #2 | |
| `426-3` | #3 | |
| `426-4` | #4 | |
| `426-5` | #5 | |

`composite_score = win_rate × log(1 + total_return) × sqrt(n_trades) / (max_drawdown + 0.01)`

为防同族霸榜，**同一 family 在 Top 5 里最多 2 个**（diversity tie-break）。

前端选 `426-1` 即可在任意股票上回测。

## 验收

1. `python -m scripts.mine_1000_strategies` 单命令完成全流程，输出 `docs/strategy_mine/1000_strategies.json`。
2. 输出文件至少含 1000 条记录（如不足 1000，报告警告并保留最佳 N 条）。
3. 每条记录都满足 4 项硬指标 + OOS gate + 最小交易数 + 股票分散度。
4. 报告 markdown：`docs/strategy_mine/REPORT.md`，含族分布、参数分布、性能直方图。
5. **Top 5 注册为 `426-1` ~ `426-5`，前端可见可回测。**
