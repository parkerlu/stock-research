# Strategy Factory V1 + Backtest Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strategy factory that auto-generates ~600 technical indicator strategies via parameter sweep, evaluates them through a backtest engine with three-tier position sizing, and surfaces the Top 50 per stock in a minimal frontend UI.

**Architecture:** Pure-Python backtest engine (no external backtest lib) with a state-machine position model. Strategy templates are pluggable Python classes registered via a simple registry. Factory service orchestrates parameter sweep + parallel backtest via `concurrent.futures.ProcessPoolExecutor`. Frontend adds a "策略" tab with strategy list table, factory run panel, and backtest report modal.

**Tech Stack:** FastAPI + SQLAlchemy 2.0 async (existing), pandas/numpy for indicator calculation, ProcessPoolExecutor for parallelism, Zustand + React for frontend.

**Spec:** `docs/superpowers/specs/2026-04-17-strategy-backtest-design.md`

---

## File Structure

### Backend — New Files
| File | Responsibility |
|------|---------------|
| `backend/app/models/schema.py` | Extend: add `FactoryJob`, `Strategy`, `BacktestRun` models |
| `backend/alembic/versions/xxxx_create_strategy_tables.py` | Migration for 3 new tables |
| `backend/app/services/backtest_engine.py` | Pure-Python backtest engine (no DB dependency) |
| `backend/app/services/strategy_templates/__init__.py` | Template registry + `generate_all_candidates()` |
| `backend/app/services/strategy_templates/base.py` | `StrategyTemplate` abstract base class |
| `backend/app/services/strategy_templates/ma_crossover.py` | MA dual crossover template |
| `backend/app/services/strategy_templates/rsi.py` | RSI overbought/oversold template |
| `backend/app/services/strategy_templates/macd.py` | MACD signal crossover template |
| `backend/app/services/strategy_templates/bollinger.py` | Bollinger Band breakout template |
| `backend/app/services/strategy_templates/kdj.py` | KDJ golden/dead cross template |
| `backend/app/services/strategy_templates/combined.py` | MA+RSI, MACD+Vol, BB+RSI combo templates |
| `backend/app/services/factory_service.py` | Factory orchestration (sweep, parallel backtest, filter, rank, persist) |
| `backend/app/routers/backtests.py` | `POST /api/backtests/run`, `GET /api/backtests/{id}/report` |
| `backend/app/routers/strategies.py` | `GET /api/strategies`, `POST/DELETE /api/strategies/{id}/pin` |
| `backend/app/routers/factory.py` | `POST /api/strategy-factory/run`, `GET /api/strategy-factory/jobs/{id}` |
| `backend/app/main.py` | Extend: register 3 new routers |
| `backend/tests/test_backtest_engine.py` | Backtest engine unit tests |
| `backend/tests/test_strategy_templates.py` | Strategy template unit tests |
| `backend/tests/test_factory_service.py` | Factory service unit tests |

### Frontend — New Files
| File | Responsibility |
|------|---------------|
| `frontend/src/types/strategy.ts` | TypeScript interfaces for Strategy, FactoryJob, BacktestReport |
| `frontend/src/api/strategy.ts` | API client functions |
| `frontend/src/stores/strategyStore.ts` | Zustand store for strategies + factory jobs |
| `frontend/src/components/StrategyPanel/index.tsx` | Main strategy panel layout |
| `frontend/src/components/StrategyPanel/StrategyList.tsx` | Top 50 strategy table |
| `frontend/src/components/StrategyPanel/FactoryControl.tsx` | Factory run button + progress bar |
| `frontend/src/components/StrategyPanel/BacktestReport.tsx` | Report modal with metrics + equity curve |

### Frontend — Modified Files
| File | Change |
|------|--------|
| `frontend/src/components/NavBar.tsx` | Add "策略" to `AppMode` union |
| `frontend/src/pages/QuotePage.tsx` | Render `StrategyPanel` when mode === "strategy" |
| `frontend/src/App.css` | Add strategy panel CSS |

---

### Task 1: Database Models + Migration

**Files:**
- Modify: `backend/app/models/schema.py:78` (append after StockPoolItem)
- Create: `backend/alembic/versions/` (new migration)

- [ ] **Step 1: Add 3 new models to schema.py**

Append to `backend/app/models/schema.py` after line 78 (end of `StockPoolItem`):

```python
class FactoryJob(Base):
    __tablename__ = "factory_job"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ts_code: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    total_candidates: Mapped[int] = mapped_column(default=0)
    evaluated: Mapped[int] = mapped_column(default=0)
    passed: Mapped[int] = mapped_column(default=0)
    config: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)


class Strategy(Base):
    __tablename__ = "strategy"
    __table_args__ = (
        Index("ix_strategy_ts_code", "ts_code"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(20))
    name: Mapped[str] = mapped_column(String(100))
    template: Mapped[str] = mapped_column(String(50))
    parameters: Mapped[dict] = mapped_column(JSON)
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    annualized_return: Mapped[float | None] = mapped_column(Numeric(12, 6))
    net_profit: Mapped[float | None] = mapped_column(Numeric(14, 4))
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(8, 6))
    win_rate: Mapped[float | None] = mapped_column(Numeric(6, 4))
    total_trades: Mapped[int | None] = mapped_column()
    profit_factor: Mapped[float | None] = mapped_column(Numeric(10, 4))
    final_capital: Mapped[float | None] = mapped_column(Numeric(14, 4))
    metrics: Mapped[dict | None] = mapped_column(JSON)
    job_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class BacktestRun(Base):
    __tablename__ = "backtest_run"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ts_code: Mapped[str] = mapped_column(String(20), index=True)
    strategy_id: Mapped[int | None] = mapped_column()
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    initial_capital: Mapped[float] = mapped_column(Numeric(14, 4), default=10000)
    position_ratios: Mapped[list | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    metrics: Mapped[dict | None] = mapped_column(JSON)
    trades: Mapped[list | None] = mapped_column(JSON)
    equity_curve: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
```

Also add these imports at the top of schema.py:
```python
from sqlalchemy import JSON, Text
```

- [ ] **Step 2: Generate and review Alembic migration**

```bash
cd backend && alembic revision --autogenerate -m "create strategy factory tables"
```

Review the generated migration file to ensure it creates `factory_job`, `strategy`, and `backtest_run` tables correctly.

- [ ] **Step 3: Run migration**

```bash
cd backend && alembic upgrade head
```

- [ ] **Step 4: Verify tables exist**

```bash
docker exec -it stock-db-1 psql -U stock -d stock_db -c "\dt"
```

Expected: `factory_job`, `strategy`, `backtest_run` appear in the table list.

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/schema.py backend/alembic/versions/*strategy*
git commit -m "feat: add factory_job, strategy, backtest_run tables"
```

---

### Task 2: Backtest Engine (Core Logic)

**Files:**
- Create: `backend/app/services/backtest_engine.py`
- Create: `backend/tests/test_backtest_engine.py`

- [ ] **Step 1: Write failing tests for backtest engine**

Create `backend/tests/test_backtest_engine.py`:

```python
"""Tests for the backtest engine with three-tier position sizing."""
from datetime import date
import pytest
from app.services.backtest_engine import run_backtest


def _candle(d: str, close: float, open_: float = 0, high: float = 0, low: float = 0, vol: int = 1000):
    """Helper to build a candle dict."""
    return {
        "trade_date": date.fromisoformat(d),
        "open": open_ or close,
        "high": high or close,
        "low": low or close,
        "close": close,
        "vol": vol,
        "amount": close * vol,
    }


class TestBasicBacktest:
    """Test basic buy/sell cycle."""

    def test_single_buy_sell_profit(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 10.0),  # buy
            _candle("2020-01-03", 11.0),
            _candle("2020-01-04", 12.0),  # sell
            _candle("2020-01-05", 13.0),
        ]
        signals = [
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])

        # First buy: 40% of 10000 = 4000, at price 10 → 400 shares
        # Sell at 12 → 400 * 12 = 4800, profit = 800
        assert result["total_trades"] == 1
        assert result["net_profit"] == pytest.approx(800.0)
        assert result["final_capital"] == pytest.approx(10800.0)
        assert result["win_rate"] == pytest.approx(1.0)

    def test_single_buy_sell_loss(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 10.0),  # buy
            _candle("2020-01-03", 9.0),
            _candle("2020-01-04", 8.0),  # sell
        ]
        signals = [
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])

        # Buy 40% at 10 → 400 shares, sell at 8 → 3200, loss = -800
        assert result["total_trades"] == 1
        assert result["net_profit"] == pytest.approx(-800.0)
        assert result["win_rate"] == pytest.approx(0.0)

    def test_no_signals_no_trades(self):
        candles = [_candle("2020-01-01", 10.0), _candle("2020-01-02", 11.0)]
        result = run_backtest(candles, [], initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 0
        assert result["net_profit"] == pytest.approx(0.0)
        assert result["final_capital"] == pytest.approx(10000.0)


class TestThreeTierPosition:
    """Test the three-tier position sizing state machine."""

    def test_three_buys_then_sell(self):
        candles = [
            _candle("2020-01-01", 10.0),  # buy1
            _candle("2020-01-02", 10.0),  # buy2
            _candle("2020-01-03", 10.0),  # buy3
            _candle("2020-01-04", 12.0),  # sell
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 3), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])

        # Buy1: 4000/10=400 shares, Buy2: 3000/10=300 shares, Buy3: 3000/10=300 shares
        # Total: 1000 shares, sell at 12 → 12000, cost = 10000, profit = 2000
        assert result["total_trades"] == 1
        assert result["net_profit"] == pytest.approx(2000.0)
        assert result["final_capital"] == pytest.approx(12000.0)

    def test_buy_at_full_position_ignored(self):
        candles = [
            _candle("2020-01-01", 10.0),  # buy1
            _candle("2020-01-02", 10.0),  # buy2
            _candle("2020-01-03", 10.0),  # buy3
            _candle("2020-01-04", 10.0),  # buy4 (ignored, already full)
            _candle("2020-01-05", 12.0),  # sell
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 3), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "buy"},
            {"date": date(2020, 1, 5), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        # Same as three_buys: 1000 shares at 10, sell at 12
        assert result["net_profit"] == pytest.approx(2000.0)

    def test_sell_at_empty_position_ignored(self):
        candles = [
            _candle("2020-01-01", 10.0),  # sell (ignored)
            _candle("2020-01-02", 11.0),  # buy
            _candle("2020-01-03", 12.0),  # sell
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "sell"},
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 3), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        # Only 1 trade: buy 40% at 11, sell at 12
        assert result["total_trades"] == 1
        # 4000/11 ≈ 363.636 shares * 12 = 4363.636, profit ≈ 363.636
        assert result["net_profit"] == pytest.approx(4000 / 11 * 12 - 4000, rel=1e-4)

    def test_partial_position_sell_clears_all(self):
        candles = [
            _candle("2020-01-01", 10.0),  # buy1
            _candle("2020-01-02", 10.0),  # buy2
            _candle("2020-01-03", 12.0),  # sell (clears both batches)
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 3), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        # Buy1: 400 shares, Buy2: 300 shares = 700 total
        # Sell at 12: 700*12=8400, cost=7000, profit=1400
        assert result["total_trades"] == 1
        assert result["net_profit"] == pytest.approx(1400.0)


class TestMetrics:
    """Test metric calculations."""

    def test_equity_curve_length(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 11.0),
            _candle("2020-01-03", 12.0),
        ]
        result = run_backtest(candles, [], initial_capital=10000, position_ratios=[40, 30, 30])
        assert len(result["equity_curve"]) == 3

    def test_max_drawdown(self):
        candles = [
            _candle("2020-01-01", 10.0),  # buy
            _candle("2020-01-02", 12.0),  # peak
            _candle("2020-01-03", 8.0),   # trough
            _candle("2020-01-04", 11.0),  # sell
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[100, 0, 0])
        # Buy all at 10: 1000 shares
        # Day 2 equity: 12000, Day 3 equity: 8000
        # Drawdown = (12000 - 8000) / 12000 = 0.3333
        assert result["max_drawdown"] == pytest.approx(1 / 3, rel=1e-3)

    def test_profit_factor(self):
        candles = [
            _candle("2020-01-01", 10.0),  # buy
            _candle("2020-01-02", 12.0),  # sell (win +800)
            _candle("2020-01-03", 12.0),  # buy
            _candle("2020-01-04", 11.0),  # sell (loss -400ish)
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "sell"},
            {"date": date(2020, 1, 3), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 2
        assert result["win_rate"] == pytest.approx(0.5)
        # Trade 1: buy 400@10, sell@12 → profit 800
        # Trade 2: buy 40% of 10000=4000 at 12 → 333.33 shares, sell@11 → loss ≈ -333.33
        gross_profit = 800.0
        gross_loss = abs(4000 / 12 * 11 - 4000)
        assert result["profit_factor"] == pytest.approx(gross_profit / gross_loss, rel=1e-3)

    def test_trades_detail(self):
        candles = [
            _candle("2020-01-01", 10.0),  # buy
            _candle("2020-01-02", 12.0),  # sell
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert len(result["trades"]) == 1
        trade = result["trades"][0]
        assert trade["entry_date"] == "2020-01-01"
        assert trade["exit_date"] == "2020-01-02"
        assert trade["pnl"] == pytest.approx(800.0)

    def test_annualized_return(self):
        candles = [_candle(f"2020-01-{str(i+1).zfill(2)}", 10.0 + i * 0.1) for i in range(20)]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 20), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert "annualized_return" in result
        assert isinstance(result["annualized_return"], float)


class TestMultipleCycles:
    """Test multiple buy-sell cycles."""

    def test_two_full_cycles(self):
        candles = [
            _candle("2020-01-01", 10.0),  # buy
            _candle("2020-01-02", 12.0),  # sell → profit
            _candle("2020-01-03", 12.0),  # buy
            _candle("2020-01-04", 15.0),  # sell → profit
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "sell"},
            {"date": date(2020, 1, 3), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 2
        assert result["net_profit"] > 0

    def test_open_position_at_end(self):
        """Unsold position should be valued at last close for equity curve but not counted as trade."""
        candles = [
            _candle("2020-01-01", 10.0),  # buy
            _candle("2020-01-02", 12.0),  # no sell
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        # No completed trade
        assert result["total_trades"] == 0
        # But equity reflects unrealized gain
        assert result["final_capital"] == pytest.approx(6000 + 400 * 12)  # cash 6000 + 400 shares * 12
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_backtest_engine.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement backtest engine**

Create `backend/app/services/backtest_engine.py`:

```python
"""Backtest engine with three-tier position sizing.

Pure Python, no database dependency. Can be called from factory or standalone.
"""
from __future__ import annotations

from datetime import date


def run_backtest(
    candles: list[dict],
    signals: list[dict],
    initial_capital: float = 10000.0,
    position_ratios: list[int] | None = None,
) -> dict:
    """Run a backtest with three-tier position sizing.

    Args:
        candles: List of dicts with keys: trade_date, open, high, low, close, vol, amount.
                 Must be sorted by trade_date ascending.
        signals: List of dicts with keys: date (date object), action ("buy"/"sell").
        initial_capital: Starting capital.
        position_ratios: Three-element list like [40, 30, 30] summing to 100.

    Returns:
        Dict with metrics: net_profit, net_profit_pct, annualized_return, max_drawdown,
        win_rate, total_trades, profit_factor, final_capital, trades, equity_curve.
    """
    if position_ratios is None:
        position_ratios = [40, 30, 30]

    # Build signal lookup: date → action
    signal_map: dict[date, str] = {}
    for sig in signals:
        signal_map[sig["date"]] = sig["action"]

    # State
    cash = initial_capital
    position_level = 0  # 0=空仓, 1=1/3仓, 2=2/3仓, 3=满仓
    holdings: list[dict] = []  # [{shares, entry_price, entry_date}]

    completed_trades: list[dict] = []
    equity_curve: list[dict] = []

    for candle in candles:
        td = candle["trade_date"]
        close = float(candle["close"])
        action = signal_map.get(td)

        if action == "buy" and position_level < 3:
            amount = initial_capital * position_ratios[position_level] / 100.0
            if amount > 0 and close > 0:
                shares = amount / close
                cash -= amount
                holdings.append({
                    "shares": shares,
                    "entry_price": close,
                    "entry_date": td,
                })
                position_level += 1

        elif action == "sell" and position_level > 0:
            total_shares = sum(h["shares"] for h in holdings)
            total_cost = sum(h["shares"] * h["entry_price"] for h in holdings)
            proceeds = total_shares * close
            pnl = proceeds - total_cost
            first_entry = min(h["entry_date"] for h in holdings)

            completed_trades.append({
                "entry_date": first_entry.isoformat() if isinstance(first_entry, date) else str(first_entry),
                "exit_date": td.isoformat() if isinstance(td, date) else str(td),
                "entry_price": round(total_cost / total_shares, 4) if total_shares else 0,
                "exit_price": close,
                "shares": round(total_shares, 4),
                "pnl": round(pnl, 4),
            })

            cash += proceeds
            holdings.clear()
            position_level = 0

        # Record daily equity
        equity = cash + sum(h["shares"] * close for h in holdings)
        equity_curve.append({
            "date": td.isoformat() if isinstance(td, date) else str(td),
            "equity": round(equity, 4),
        })

    # Final capital (mark-to-market)
    if candles:
        last_close = float(candles[-1]["close"])
        final_capital = cash + sum(h["shares"] * last_close for h in holdings)
    else:
        final_capital = initial_capital

    # Compute metrics
    net_profit = final_capital - initial_capital
    net_profit_pct = (net_profit / initial_capital * 100) if initial_capital else 0.0

    # Annualized return
    trading_days = len(candles)
    if trading_days > 1 and final_capital > 0 and initial_capital > 0:
        annualized_return = (final_capital / initial_capital) ** (252 / trading_days) - 1
    else:
        annualized_return = 0.0

    # Max drawdown
    max_drawdown = 0.0
    peak = 0.0
    for point in equity_curve:
        eq = point["equity"]
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_drawdown:
                max_drawdown = dd

    # Win rate
    total_trades = len(completed_trades)
    winning = sum(1 for t in completed_trades if t["pnl"] > 0)
    win_rate = winning / total_trades if total_trades > 0 else 0.0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in completed_trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in completed_trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    return {
        "net_profit": round(net_profit, 4),
        "net_profit_pct": round(net_profit_pct, 4),
        "annualized_return": round(annualized_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "win_rate": round(win_rate, 4),
        "total_trades": total_trades,
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 9999.0,
        "final_capital": round(final_capital, 4),
        "trades": completed_trades,
        "equity_curve": equity_curve,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_backtest_engine.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/backtest_engine.py backend/tests/test_backtest_engine.py
git commit -m "feat: implement backtest engine with three-tier position sizing"
```

---

### Task 3: Strategy Template Base + MA Crossover

**Files:**
- Create: `backend/app/services/strategy_templates/__init__.py`
- Create: `backend/app/services/strategy_templates/base.py`
- Create: `backend/app/services/strategy_templates/ma_crossover.py`
- Create: `backend/tests/test_strategy_templates.py`

- [ ] **Step 1: Write failing tests for templates**

Create `backend/tests/test_strategy_templates.py`:

```python
"""Tests for strategy templates."""
from datetime import date, timedelta
import pandas as pd
import pytest

from app.services.strategy_templates.base import StrategyTemplate
from app.services.strategy_templates.ma_crossover import MACrossover


def _make_df(n: int = 100, start_price: float = 10.0) -> pd.DataFrame:
    """Generate a simple trending OHLCV DataFrame."""
    import numpy as np
    np.random.seed(42)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    closes = start_price + np.cumsum(np.random.randn(n) * 0.5)
    closes = np.maximum(closes, 1.0)  # no negative prices
    return pd.DataFrame({
        "trade_date": dates,
        "open": closes * 0.99,
        "high": closes * 1.02,
        "low": closes * 0.98,
        "close": closes,
        "vol": [1000000] * n,
        "amount": closes * 1000000,
    })


class TestMACrossover:
    def test_generate_signals_returns_list(self):
        df = _make_df(100)
        template = MACrossover(fast_period=5, slow_period=20)
        signals = template.generate_signals(df)
        assert isinstance(signals, list)
        for sig in signals:
            assert "date" in sig
            assert "action" in sig
            assert sig["action"] in ("buy", "sell")

    def test_signals_have_valid_dates(self):
        df = _make_df(100)
        template = MACrossover(fast_period=5, slow_period=20)
        signals = template.generate_signals(df)
        df_dates = set(df["trade_date"])
        for sig in signals:
            assert sig["date"] in df_dates

    def test_name_format(self):
        template = MACrossover(fast_period=5, slow_period=20)
        assert template.name == "MA_5_20"
        assert template.template_id == "ma_crossover"

    def test_parameter_candidates(self):
        candidates = MACrossover.parameter_candidates()
        assert len(candidates) > 0
        for params in candidates:
            assert params["fast_period"] < params["slow_period"]

    def test_no_signals_for_short_data(self):
        df = _make_df(5)  # too short for MA(20)
        template = MACrossover(fast_period=5, slow_period=20)
        signals = template.generate_signals(df)
        assert signals == []


class TestTemplateRegistry:
    def test_registry_has_ma_crossover(self):
        from app.services.strategy_templates import TEMPLATE_REGISTRY
        assert "ma_crossover" in TEMPLATE_REGISTRY

    def test_generate_all_candidates_non_empty(self):
        from app.services.strategy_templates import generate_all_candidates
        candidates = generate_all_candidates()
        assert len(candidates) > 0
        for c in candidates:
            assert "template_id" in c
            assert "name" in c
            assert "params" in c
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_strategy_templates.py -v
```

- [ ] **Step 3: Implement base class**

Create `backend/app/services/strategy_templates/base.py`:

```python
"""Base class for strategy templates."""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class StrategyTemplate(ABC):
    """Abstract base for all strategy templates.

    Subclasses implement generate_signals() and parameter_candidates().
    """

    template_id: str = ""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name, e.g. 'MA_5_20'."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        """Generate buy/sell signals from OHLCV DataFrame.

        Args:
            df: DataFrame with columns: trade_date, open, high, low, close, vol, amount.
                Sorted by trade_date ascending.

        Returns:
            List of dicts: [{"date": date, "action": "buy"|"sell"}, ...]
        """

    @staticmethod
    @abstractmethod
    def parameter_candidates() -> list[dict]:
        """Return all valid parameter combinations for this template."""

    def to_dict(self) -> dict:
        """Serialize template instance to dict for DB storage."""
        return {
            "template_id": self.template_id,
            "name": self.name,
        }
```

- [ ] **Step 4: Implement MA Crossover template**

Create `backend/app/services/strategy_templates/ma_crossover.py`:

```python
"""MA dual crossover strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class MACrossover(StrategyTemplate):
    template_id = "ma_crossover"

    def __init__(self, fast_period: int, slow_period: int):
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def name(self) -> str:
        return f"MA_{self.fast_period}_{self.slow_period}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.slow_period + 1:
            return []

        fast_ma = df["close"].rolling(window=self.fast_period).mean()
        slow_ma = df["close"].rolling(window=self.slow_period).mean()

        signals = []
        for i in range(1, len(df)):
            if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                continue
            if pd.isna(fast_ma.iloc[i - 1]) or pd.isna(slow_ma.iloc[i - 1]):
                continue

            # Golden cross: fast crosses above slow
            if fast_ma.iloc[i - 1] <= slow_ma.iloc[i - 1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            # Death cross: fast crosses below slow
            elif fast_ma.iloc[i - 1] >= slow_ma.iloc[i - 1] and fast_ma.iloc[i] < slow_ma.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        fast_values = list(range(3, 31, 3))   # [3,6,9,...,30]
        slow_values = list(range(10, 121, 10)) # [10,20,...,120]
        candidates = []
        for fast, slow in product(fast_values, slow_values):
            if fast < slow:
                candidates.append({"fast_period": fast, "slow_period": slow})
        return candidates
```

- [ ] **Step 5: Implement template registry**

Create `backend/app/services/strategy_templates/__init__.py`:

```python
"""Strategy template registry.

All templates register themselves here. The factory service uses
generate_all_candidates() to produce the full candidate list.
"""
from __future__ import annotations

from .ma_crossover import MACrossover

# Registry: template_id → template class
TEMPLATE_REGISTRY: dict[str, type] = {
    "ma_crossover": MACrossover,
}


def generate_all_candidates() -> list[dict]:
    """Generate all strategy candidates across all registered templates.

    Returns:
        List of dicts: [{"template_id": str, "name": str, "params": dict}, ...]
    """
    candidates = []
    for template_id, cls in TEMPLATE_REGISTRY.items():
        for params in cls.parameter_candidates():
            instance = cls(**params)
            candidates.append({
                "template_id": template_id,
                "name": instance.name,
                "params": params,
            })
    return candidates
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_strategy_templates.py -v
```

Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/strategy_templates/ backend/tests/test_strategy_templates.py
git commit -m "feat: add strategy template base + MA crossover template"
```

---

### Task 4: Remaining Strategy Templates (RSI, MACD, Bollinger, KDJ)

**Files:**
- Create: `backend/app/services/strategy_templates/rsi.py`
- Create: `backend/app/services/strategy_templates/macd.py`
- Create: `backend/app/services/strategy_templates/bollinger.py`
- Create: `backend/app/services/strategy_templates/kdj.py`
- Modify: `backend/app/services/strategy_templates/__init__.py`
- Modify: `backend/tests/test_strategy_templates.py`

- [ ] **Step 1: Add tests for all new templates**

Append to `backend/tests/test_strategy_templates.py`:

```python
from app.services.strategy_templates.rsi import RSIOverboughtOversold
from app.services.strategy_templates.macd import MACDCrossover
from app.services.strategy_templates.bollinger import BollingerBreakout
from app.services.strategy_templates.kdj import KDJCrossover


class TestRSI:
    def test_generate_signals(self):
        df = _make_df(200)
        t = RSIOverboughtOversold(period=14, overbought=70, oversold=30)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_name(self):
        t = RSIOverboughtOversold(period=14, overbought=70, oversold=30)
        assert t.name == "RSI_14_70_30"
        assert t.template_id == "rsi"

    def test_candidates(self):
        c = RSIOverboughtOversold.parameter_candidates()
        assert len(c) >= 40
        for p in c:
            assert p["oversold"] < p["overbought"] - 15


class TestMACD:
    def test_generate_signals(self):
        df = _make_df(200)
        t = MACDCrossover(fast=12, slow=26, signal=9)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_name(self):
        t = MACDCrossover(fast=12, slow=26, signal=9)
        assert t.name == "MACD_12_26_9"

    def test_candidates(self):
        c = MACDCrossover.parameter_candidates()
        assert len(c) >= 30
        for p in c:
            assert p["fast"] < p["slow"]


class TestBollinger:
    def test_generate_signals(self):
        df = _make_df(200)
        t = BollingerBreakout(period=20, std_dev=2.0)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_name(self):
        t = BollingerBreakout(period=20, std_dev=2.0)
        assert t.name == "BB_20_2.0"

    def test_candidates(self):
        c = BollingerBreakout.parameter_candidates()
        assert len(c) >= 20


class TestKDJ:
    def test_generate_signals(self):
        df = _make_df(200)
        t = KDJCrossover(k_period=9, d_period=3)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_name(self):
        t = KDJCrossover(k_period=9, d_period=3)
        assert t.name == "KDJ_9_3"

    def test_candidates(self):
        c = KDJCrossover.parameter_candidates()
        assert len(c) >= 20
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
cd backend && python -m pytest tests/test_strategy_templates.py -v -k "RSI or MACD or Bollinger or KDJ"
```

- [ ] **Step 3: Implement RSI template**

Create `backend/app/services/strategy_templates/rsi.py`:

```python
"""RSI overbought/oversold strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class RSIOverboughtOversold(StrategyTemplate):
    template_id = "rsi"

    def __init__(self, period: int, overbought: int, oversold: int):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    @property
    def name(self) -> str:
        return f"RSI_{self.period}_{self.overbought}_{self.oversold}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.period + 2:
            return []

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = []
        for i in range(1, len(df)):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 1]):
                continue
            # Cross below oversold → buy signal
            if rsi.iloc[i - 1] >= self.oversold and rsi.iloc[i] < self.oversold:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            # Cross above overbought → sell signal
            elif rsi.iloc[i - 1] <= self.overbought and rsi.iloc[i] > self.overbought:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        periods = list(range(6, 29, 2))
        overboughts = list(range(65, 86, 5))
        oversolds = list(range(15, 36, 5))
        candidates = []
        for period, ob, os_ in product(periods, overboughts, oversolds):
            if os_ < ob - 15:
                candidates.append({"period": period, "overbought": ob, "oversold": os_})
        return candidates
```

- [ ] **Step 4: Implement MACD template**

Create `backend/app/services/strategy_templates/macd.py`:

```python
"""MACD signal crossover strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class MACDCrossover(StrategyTemplate):
    template_id = "macd"

    def __init__(self, fast: int, slow: int, signal: int):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return f"MACD_{self.fast}_{self.slow}_{self.signal}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.slow + self.signal + 1:
            return []

        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()

        signals = []
        for i in range(self.slow + self.signal, len(df)):
            prev_diff = macd_line.iloc[i - 1] - signal_line.iloc[i - 1]
            curr_diff = macd_line.iloc[i] - signal_line.iloc[i]

            if prev_diff <= 0 and curr_diff > 0:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif prev_diff >= 0 and curr_diff < 0:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        fasts = list(range(8, 17, 2))
        slows = list(range(20, 31, 2))
        signal_vals = list(range(5, 13))
        candidates = []
        for f, s, sig in product(fasts, slows, signal_vals):
            if f < s:
                candidates.append({"fast": f, "slow": s, "signal": sig})
        return candidates
```

- [ ] **Step 5: Implement Bollinger template**

Create `backend/app/services/strategy_templates/bollinger.py`:

```python
"""Bollinger Band breakout strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class BollingerBreakout(StrategyTemplate):
    template_id = "bollinger"

    def __init__(self, period: int, std_dev: float):
        self.period = period
        self.std_dev = std_dev

    @property
    def name(self) -> str:
        return f"BB_{self.period}_{self.std_dev}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.period + 1:
            return []

        ma = df["close"].rolling(window=self.period).mean()
        std = df["close"].rolling(window=self.period).std()
        upper = ma + self.std_dev * std
        lower = ma - self.std_dev * std

        signals = []
        for i in range(1, len(df)):
            if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                continue
            # Price breaks below lower band → buy (mean reversion)
            if df["close"].iloc[i - 1] >= lower.iloc[i - 1] and df["close"].iloc[i] < lower.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            # Price breaks above upper band → sell
            elif df["close"].iloc[i - 1] <= upper.iloc[i - 1] and df["close"].iloc[i] > upper.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        periods = list(range(10, 31, 2))
        std_devs = [1.5, 2.0, 2.5, 3.0]
        return [
            {"period": p, "std_dev": s}
            for p, s in product(periods, std_devs)
        ]
```

- [ ] **Step 6: Implement KDJ template**

Create `backend/app/services/strategy_templates/kdj.py`:

```python
"""KDJ golden/dead cross strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class KDJCrossover(StrategyTemplate):
    template_id = "kdj"

    def __init__(self, k_period: int, d_period: int):
        self.k_period = k_period
        self.d_period = d_period

    @property
    def name(self) -> str:
        return f"KDJ_{self.k_period}_{self.d_period}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.k_period + self.d_period + 1:
            return []

        low_min = df["low"].rolling(window=self.k_period, min_periods=self.k_period).min()
        high_max = df["high"].rolling(window=self.k_period, min_periods=self.k_period).max()

        rsv = (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10) * 100

        k = rsv.ewm(com=self.d_period - 1, adjust=False).mean()
        d = k.ewm(com=self.d_period - 1, adjust=False).mean()

        signals = []
        for i in range(self.k_period + self.d_period, len(df)):
            if pd.isna(k.iloc[i]) or pd.isna(d.iloc[i]):
                continue
            # K crosses above D → golden cross → buy
            if k.iloc[i - 1] <= d.iloc[i - 1] and k.iloc[i] > d.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            # K crosses below D → dead cross → sell
            elif k.iloc[i - 1] >= d.iloc[i - 1] and k.iloc[i] < d.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        k_values = list(range(5, 22, 2))
        d_values = list(range(3, 10))
        return [
            {"k_period": k, "d_period": d}
            for k, d in product(k_values, d_values)
            if k > d
        ]
```

- [ ] **Step 7: Update registry with new templates**

Update `backend/app/services/strategy_templates/__init__.py`:

```python
"""Strategy template registry."""
from __future__ import annotations

from .ma_crossover import MACrossover
from .rsi import RSIOverboughtOversold
from .macd import MACDCrossover
from .bollinger import BollingerBreakout
from .kdj import KDJCrossover

TEMPLATE_REGISTRY: dict[str, type] = {
    "ma_crossover": MACrossover,
    "rsi": RSIOverboughtOversold,
    "macd": MACDCrossover,
    "bollinger": BollingerBreakout,
    "kdj": KDJCrossover,
}


def generate_all_candidates() -> list[dict]:
    candidates = []
    for template_id, cls in TEMPLATE_REGISTRY.items():
        for params in cls.parameter_candidates():
            instance = cls(**params)
            candidates.append({
                "template_id": template_id,
                "name": instance.name,
                "params": params,
            })
    return candidates
```

- [ ] **Step 8: Run tests**

```bash
cd backend && python -m pytest tests/test_strategy_templates.py -v
```

Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add backend/app/services/strategy_templates/ backend/tests/test_strategy_templates.py
git commit -m "feat: add RSI, MACD, Bollinger, KDJ strategy templates"
```

---

### Task 5: Combined Strategy Templates

**Files:**
- Create: `backend/app/services/strategy_templates/combined.py`
- Modify: `backend/app/services/strategy_templates/__init__.py`
- Modify: `backend/tests/test_strategy_templates.py`

- [ ] **Step 1: Add tests for combined templates**

Append to `backend/tests/test_strategy_templates.py`:

```python
from app.services.strategy_templates.combined import MARSICombined, MACDVolumeCombined, BollingerRSICombined


class TestMARSI:
    def test_signals(self):
        df = _make_df(200)
        t = MARSICombined(ma_period=20, rsi_period=14, overbought=70, oversold=30)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_candidates_count(self):
        c = MARSICombined.parameter_candidates()
        assert 60 <= len(c) <= 200


class TestMACDVolume:
    def test_signals(self):
        df = _make_df(200)
        t = MACDVolumeCombined(fast=12, slow=26, signal=9, vol_ma=10)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_candidates_count(self):
        c = MACDVolumeCombined.parameter_candidates()
        assert 30 <= len(c) <= 150


class TestBollingerRSI:
    def test_signals(self):
        df = _make_df(200)
        t = BollingerRSICombined(bb_period=20, bb_std=2.0, rsi_period=14, rsi_ob=70, rsi_os=30)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_candidates_count(self):
        c = BollingerRSICombined.parameter_candidates()
        assert 30 <= len(c) <= 150


class TestTotalCandidates:
    def test_total_candidates_in_range(self):
        from app.services.strategy_templates import generate_all_candidates
        candidates = generate_all_candidates()
        assert 300 <= len(candidates) <= 800, f"Got {len(candidates)} candidates, expected 300-800"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_strategy_templates.py -v -k "MARSI or MACDVolume or BollingerRSI or TotalCandidates"
```

- [ ] **Step 3: Implement combined templates**

Create `backend/app/services/strategy_templates/combined.py`:

```python
"""Combined strategy templates: MA+RSI, MACD+Volume, Bollinger+RSI."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class MARSICombined(StrategyTemplate):
    """Buy when price above MA AND RSI crosses below oversold. Sell when RSI crosses above overbought."""
    template_id = "ma_rsi"

    def __init__(self, ma_period: int, rsi_period: int, overbought: int, oversold: int):
        self.ma_period = ma_period
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    @property
    def name(self) -> str:
        return f"MA{self.ma_period}_RSI{self.rsi_period}_{self.overbought}_{self.oversold}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        n = max(self.ma_period, self.rsi_period) + 2
        if len(df) < n:
            return []

        ma = df["close"].rolling(window=self.ma_period).mean()

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = []
        for i in range(1, len(df)):
            if pd.isna(ma.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 1]):
                continue
            # Buy: RSI crosses below oversold AND price above MA
            if (rsi.iloc[i - 1] >= self.oversold and rsi.iloc[i] < self.oversold
                    and df["close"].iloc[i] > ma.iloc[i]):
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            # Sell: RSI crosses above overbought
            elif rsi.iloc[i - 1] <= self.overbought and rsi.iloc[i] > self.overbought:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        ma_periods = [5, 10, 20, 30, 60]
        rsi_periods = [6, 10, 14, 20]
        obs = [70, 75, 80]
        oss = [20, 25, 30]
        candidates = []
        for ma, rsi, ob, os_ in product(ma_periods, rsi_periods, obs, oss):
            if os_ < ob - 20:
                candidates.append({"ma_period": ma, "rsi_period": rsi, "overbought": ob, "oversold": os_})
        return candidates


class MACDVolumeCombined(StrategyTemplate):
    """MACD crossover confirmed by volume above its MA."""
    template_id = "macd_vol"

    def __init__(self, fast: int, slow: int, signal: int, vol_ma: int):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.vol_ma = vol_ma

    @property
    def name(self) -> str:
        return f"MACD{self.fast}_{self.slow}_{self.signal}_V{self.vol_ma}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        n = self.slow + self.signal + 1
        if len(df) < n:
            return []

        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        vol_ma = df["vol"].rolling(window=self.vol_ma).mean()

        signals = []
        for i in range(n, len(df)):
            if pd.isna(vol_ma.iloc[i]):
                continue
            prev_diff = macd_line.iloc[i - 1] - signal_line.iloc[i - 1]
            curr_diff = macd_line.iloc[i] - signal_line.iloc[i]
            vol_confirm = df["vol"].iloc[i] > vol_ma.iloc[i]

            if prev_diff <= 0 and curr_diff > 0 and vol_confirm:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif prev_diff >= 0 and curr_diff < 0:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        fasts = [8, 10, 12]
        slows = [20, 24, 26, 30]
        sigs = [7, 9, 11]
        vol_mas = [5, 10, 20]
        candidates = []
        for f, s, sig, vm in product(fasts, slows, sigs, vol_mas):
            if f < s:
                candidates.append({"fast": f, "slow": s, "signal": sig, "vol_ma": vm})
        return candidates


class BollingerRSICombined(StrategyTemplate):
    """Bollinger Band breakout confirmed by RSI extremes."""
    template_id = "bb_rsi"

    def __init__(self, bb_period: int, bb_std: float, rsi_period: int, rsi_ob: int, rsi_os: int):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_ob = rsi_ob
        self.rsi_os = rsi_os

    @property
    def name(self) -> str:
        return f"BB{self.bb_period}_{self.bb_std}_RSI{self.rsi_period}_{self.rsi_ob}_{self.rsi_os}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        n = max(self.bb_period, self.rsi_period) + 2
        if len(df) < n:
            return []

        ma = df["close"].rolling(window=self.bb_period).mean()
        std = df["close"].rolling(window=self.bb_period).std()
        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = []
        for i in range(1, len(df)):
            if pd.isna(upper.iloc[i]) or pd.isna(rsi.iloc[i]):
                continue
            # Buy: price below lower band AND RSI < oversold
            if df["close"].iloc[i] < lower.iloc[i] and rsi.iloc[i] < self.rsi_os:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            # Sell: price above upper band AND RSI > overbought
            elif df["close"].iloc[i] > upper.iloc[i] and rsi.iloc[i] > self.rsi_ob:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        bb_periods = [10, 15, 20, 25]
        bb_stds = [1.5, 2.0, 2.5]
        rsi_periods = [6, 10, 14]
        obs = [70, 80]
        oss = [20, 30]
        return [
            {"bb_period": bp, "bb_std": bs, "rsi_period": rp, "rsi_ob": ob, "rsi_os": os_}
            for bp, bs, rp, ob, os_ in product(bb_periods, bb_stds, rsi_periods, obs, oss)
        ]
```

- [ ] **Step 4: Update registry**

Update `backend/app/services/strategy_templates/__init__.py` to include the combined templates:

```python
"""Strategy template registry."""
from __future__ import annotations

from .ma_crossover import MACrossover
from .rsi import RSIOverboughtOversold
from .macd import MACDCrossover
from .bollinger import BollingerBreakout
from .kdj import KDJCrossover
from .combined import MARSICombined, MACDVolumeCombined, BollingerRSICombined

TEMPLATE_REGISTRY: dict[str, type] = {
    "ma_crossover": MACrossover,
    "rsi": RSIOverboughtOversold,
    "macd": MACDCrossover,
    "bollinger": BollingerBreakout,
    "kdj": KDJCrossover,
    "ma_rsi": MARSICombined,
    "macd_vol": MACDVolumeCombined,
    "bb_rsi": BollingerRSICombined,
}


def generate_all_candidates() -> list[dict]:
    candidates = []
    for template_id, cls in TEMPLATE_REGISTRY.items():
        for params in cls.parameter_candidates():
            instance = cls(**params)
            candidates.append({
                "template_id": template_id,
                "name": instance.name,
                "params": params,
            })
    return candidates
```

- [ ] **Step 5: Run tests**

```bash
cd backend && python -m pytest tests/test_strategy_templates.py -v
```

Expected: ALL PASS, total candidates in 300-800 range.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/strategy_templates/ backend/tests/test_strategy_templates.py
git commit -m "feat: add combined strategy templates (MA+RSI, MACD+Vol, BB+RSI)"
```

---

### Task 6: Factory Service

**Files:**
- Create: `backend/app/services/factory_service.py`
- Create: `backend/tests/test_factory_service.py`

- [ ] **Step 1: Write factory service tests**

Create `backend/tests/test_factory_service.py`:

```python
"""Tests for the factory service (unit tests with mocked DB)."""
from datetime import date, timedelta
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.factory_service import (
    evaluate_single_candidate,
    filter_and_rank,
)


def _make_candle_dicts(n: int = 500) -> list[dict]:
    """Generate test candle data."""
    import numpy as np
    np.random.seed(42)
    base = 100.0
    candles = []
    for i in range(n):
        d = date(2015, 1, 1) + timedelta(days=i)
        close = base + np.cumsum(np.random.randn(1))[0] * 2
        base = max(close, 10)
        candles.append({
            "trade_date": d,
            "open": base * 0.99,
            "high": base * 1.02,
            "low": base * 0.98,
            "close": base,
            "vol": 1000000,
            "amount": base * 1000000,
        })
    return candles


class TestEvaluateSingleCandidate:
    def test_returns_result_dict(self):
        candles = _make_candle_dicts(200)
        candidate = {
            "template_id": "ma_crossover",
            "name": "MA_5_20",
            "params": {"fast_period": 5, "slow_period": 20},
        }
        result = evaluate_single_candidate(candles, candidate, [40, 30, 30])
        assert "name" in result
        assert "metrics" in result
        assert "net_profit" in result["metrics"]

    def test_returns_none_for_no_signals(self):
        candles = _make_candle_dicts(5)  # too short
        candidate = {
            "template_id": "ma_crossover",
            "name": "MA_5_20",
            "params": {"fast_period": 5, "slow_period": 20},
        }
        result = evaluate_single_candidate(candles, candidate, [40, 30, 30])
        # Should still return a result, just with 0 trades
        assert result["metrics"]["total_trades"] == 0


class TestFilterAndRank:
    def test_filters_negative_profit(self):
        results = [
            {"name": "a", "metrics": {"net_profit": 100, "max_drawdown": 0.1, "annualized_return": 0.2}},
            {"name": "b", "metrics": {"net_profit": -50, "max_drawdown": 0.1, "annualized_return": -0.1}},
        ]
        filtered = filter_and_rank(results)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "a"

    def test_filters_high_drawdown(self):
        results = [
            {"name": "a", "metrics": {"net_profit": 100, "max_drawdown": 0.5, "annualized_return": 0.2}},
            {"name": "b", "metrics": {"net_profit": 100, "max_drawdown": 0.3, "annualized_return": 0.15}},
        ]
        filtered = filter_and_rank(results)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "b"

    def test_top_50_limit(self):
        results = [
            {"name": f"s{i}", "metrics": {"net_profit": 100 + i, "max_drawdown": 0.1, "annualized_return": 0.01 * i}}
            for i in range(100)
        ]
        filtered = filter_and_rank(results)
        assert len(filtered) == 50
        # Should be sorted by annualized_return descending
        returns = [r["metrics"]["annualized_return"] for r in filtered]
        assert returns == sorted(returns, reverse=True)

    def test_sorted_by_annualized_return(self):
        results = [
            {"name": "a", "metrics": {"net_profit": 100, "max_drawdown": 0.1, "annualized_return": 0.1}},
            {"name": "b", "metrics": {"net_profit": 200, "max_drawdown": 0.2, "annualized_return": 0.3}},
            {"name": "c", "metrics": {"net_profit": 150, "max_drawdown": 0.15, "annualized_return": 0.2}},
        ]
        filtered = filter_and_rank(results)
        assert [r["name"] for r in filtered] == ["b", "c", "a"]
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && python -m pytest tests/test_factory_service.py -v
```

- [ ] **Step 3: Implement factory service**

Create `backend/app/services/factory_service.py`:

```python
"""Strategy factory service: generate candidates, backtest, filter, persist."""
from __future__ import annotations

import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import delete, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import DailyCandle, FactoryJob, Strategy
from app.services.backtest_engine import run_backtest
from app.services.strategy_templates import TEMPLATE_REGISTRY, generate_all_candidates


def evaluate_single_candidate(
    candles: list[dict],
    candidate: dict,
    position_ratios: list[int],
) -> dict:
    """Evaluate a single strategy candidate (runs in worker process).

    Returns dict with candidate info + backtest metrics.
    """
    template_cls = TEMPLATE_REGISTRY[candidate["template_id"]]
    template = template_cls(**candidate["params"])

    df = pd.DataFrame(candles)
    signals = template.generate_signals(df)
    metrics = run_backtest(candles, signals, initial_capital=10000, position_ratios=position_ratios)

    return {
        "template_id": candidate["template_id"],
        "name": candidate["name"],
        "params": candidate["params"],
        "metrics": metrics,
    }


def filter_and_rank(results: list[dict], top_n: int = 50) -> list[dict]:
    """Apply hard filters and return top N by annualized return.

    Filters: net_profit > 0 AND max_drawdown <= 0.35
    Sort: annualized_return descending
    """
    passed = [
        r for r in results
        if r["metrics"]["net_profit"] > 0 and r["metrics"]["max_drawdown"] <= 0.35
    ]
    passed.sort(key=lambda r: r["metrics"]["annualized_return"], reverse=True)
    return passed[:top_n]


async def get_candle_dicts(db: AsyncSession, ts_code: str, start: date, end: date) -> list[dict]:
    """Fetch candle data from DB as list of dicts for backtest."""
    stmt = (
        select(DailyCandle)
        .where(and_(
            DailyCandle.ts_code == ts_code,
            DailyCandle.trade_date >= start,
            DailyCandle.trade_date <= end,
        ))
        .order_by(DailyCandle.trade_date)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [
        {
            "trade_date": r.trade_date,
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "vol": r.vol,
            "amount": float(r.amount),
        }
        for r in rows
    ]


async def run_factory(
    db: AsyncSession,
    ts_code: str,
    cutoff_date: date,
    position_ratios: list[int] | None = None,
) -> str:
    """Start a factory job: generate candidates, backtest in parallel, persist top 50.

    Returns job_id. Updates FactoryJob row with progress.
    """
    if position_ratios is None:
        position_ratios = [40, 30, 30]

    job_id = str(uuid.uuid4())
    candidates = generate_all_candidates()

    job = FactoryJob(
        id=job_id,
        ts_code=ts_code,
        status="running",
        total_candidates=len(candidates),
        evaluated=0,
        passed=0,
        config={
            "cutoff_date": cutoff_date.isoformat(),
            "position_ratios": position_ratios,
        },
    )
    db.add(job)
    await db.commit()

    # Fetch candle data: cutoff_date minus 10 years
    start_date = date(cutoff_date.year - 10, cutoff_date.month, cutoff_date.day)
    candles = await get_candle_dicts(db, ts_code, start_date, cutoff_date)

    if not candles:
        job.status = "failed"
        job.error = "No candle data available"
        job.completed_at = datetime.now()
        await db.commit()
        return job_id

    # Run backtests in parallel using ProcessPoolExecutor
    results = []
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(evaluate_single_candidate, candles, c, position_ratios)
                for c in candidates
            ]
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception:
                    pass  # Skip failed candidates

                # Update progress periodically
                if (i + 1) % 50 == 0 or i == len(futures) - 1:
                    job.evaluated = i + 1
                    await db.commit()

        # Filter and rank
        top_results = filter_and_rank(results)
        job.evaluated = len(candidates)
        job.passed = len(top_results)

        # Delete old non-pinned strategies for this stock
        await db.execute(
            delete(Strategy).where(
                and_(Strategy.ts_code == ts_code, Strategy.is_pinned == False)
            )
        )

        # Insert new top strategies
        for r in top_results:
            m = r["metrics"]
            strategy = Strategy(
                ts_code=ts_code,
                name=r["name"],
                template=r["template_id"],
                parameters=r["params"],
                annualized_return=m["annualized_return"],
                net_profit=m["net_profit"],
                max_drawdown=m["max_drawdown"],
                win_rate=m["win_rate"],
                total_trades=m["total_trades"],
                profit_factor=m["profit_factor"],
                final_capital=m["final_capital"],
                metrics=m,
                job_id=job_id,
            )
            db.add(strategy)

        job.status = "completed"
        job.completed_at = datetime.now()
        await db.commit()

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now()
        await db.commit()

    return job_id
```

- [ ] **Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_factory_service.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/factory_service.py backend/tests/test_factory_service.py
git commit -m "feat: implement strategy factory service with parallel backtest"
```

---

### Task 7: API Routers + Main Registration

**Files:**
- Create: `backend/app/routers/backtests.py`
- Create: `backend/app/routers/strategies.py`
- Create: `backend/app/routers/factory.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Implement backtests router**

Create `backend/app/routers/backtests.py`:

```python
"""Backtest API: run backtests, retrieve reports."""
from __future__ import annotations

import uuid
from datetime import date, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.schema import BacktestRun, Strategy
from app.services.backtest_engine import run_backtest
from app.services.factory_service import get_candle_dicts
from app.services.strategy_templates import TEMPLATE_REGISTRY

router = APIRouter(prefix="/api/backtests", tags=["backtests"])


class RunBacktestRequest(BaseModel):
    ts_code: str
    strategy_id: int | None = None
    start_date: date
    end_date: date
    initial_capital: float = 10000.0
    position_ratios: list[int] = [40, 30, 30]


async def _execute_backtest(run_id: str, request: RunBacktestRequest):
    """Background task to run a backtest."""
    from app.db import async_session

    async with async_session() as db:
        bt = await db.get(BacktestRun, run_id)
        if not bt:
            return

        try:
            candles = await get_candle_dicts(db, request.ts_code, request.start_date, request.end_date)
            if not candles:
                bt.status = "failed"
                bt.completed_at = datetime.now()
                await db.commit()
                return

            # Get signals from strategy template
            signals = []
            if request.strategy_id:
                strategy = await db.get(Strategy, request.strategy_id)
                if strategy and strategy.template in TEMPLATE_REGISTRY:
                    import pandas as pd
                    template_cls = TEMPLATE_REGISTRY[strategy.template]
                    template = template_cls(**strategy.parameters)
                    df = pd.DataFrame(candles)
                    signals = template.generate_signals(df)

            result = run_backtest(
                candles, signals,
                initial_capital=request.initial_capital,
                position_ratios=request.position_ratios,
            )

            bt.status = "completed"
            bt.metrics = {
                k: v for k, v in result.items()
                if k not in ("trades", "equity_curve")
            }
            bt.trades = result["trades"]
            bt.equity_curve = result["equity_curve"]
            bt.completed_at = datetime.now()
            await db.commit()

        except Exception as e:
            bt.status = "failed"
            bt.completed_at = datetime.now()
            await db.commit()


@router.post("/run")
async def run(
    body: RunBacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    run_id = str(uuid.uuid4())
    bt = BacktestRun(
        id=run_id,
        ts_code=body.ts_code,
        strategy_id=body.strategy_id,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_capital=body.initial_capital,
        position_ratios=body.position_ratios,
        status="running",
    )
    db.add(bt)
    await db.commit()

    background_tasks.add_task(_execute_backtest, run_id, body)
    return {"id": run_id, "status": "running"}


@router.get("/{run_id}/report")
async def report(run_id: str, db: AsyncSession = Depends(get_db)):
    bt = await db.get(BacktestRun, run_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest run not found")
    return {
        "id": bt.id,
        "status": bt.status,
        "ts_code": bt.ts_code,
        "strategy_id": bt.strategy_id,
        "metrics": bt.metrics,
        "trades": bt.trades,
        "equity_curve": bt.equity_curve,
        "created_at": bt.created_at.isoformat() if bt.created_at else None,
        "completed_at": bt.completed_at.isoformat() if bt.completed_at else None,
    }
```

- [ ] **Step 2: Implement strategies router**

Create `backend/app/routers/strategies.py`:

```python
"""Strategy API: list strategies, pin/unpin."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.schema import Strategy

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


@router.get("")
async def list_strategies(
    ts_code: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Strategy)
        .where(Strategy.ts_code == ts_code)
        .order_by(Strategy.annualized_return.desc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "template": s.template,
            "parameters": s.parameters,
            "is_pinned": s.is_pinned,
            "annualized_return": float(s.annualized_return) if s.annualized_return else None,
            "net_profit": float(s.net_profit) if s.net_profit else None,
            "max_drawdown": float(s.max_drawdown) if s.max_drawdown else None,
            "win_rate": float(s.win_rate) if s.win_rate else None,
            "total_trades": s.total_trades,
            "profit_factor": float(s.profit_factor) if s.profit_factor else None,
            "final_capital": float(s.final_capital) if s.final_capital else None,
            "job_id": s.job_id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in rows
    ]


@router.post("/{strategy_id}/pin")
async def pin(strategy_id: int, db: AsyncSession = Depends(get_db)):
    strategy = await db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    strategy.is_pinned = True
    await db.commit()
    return {"ok": True}


@router.delete("/{strategy_id}/pin")
async def unpin(strategy_id: int, db: AsyncSession = Depends(get_db)):
    strategy = await db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    strategy.is_pinned = False
    await db.commit()
    return {"ok": True}
```

- [ ] **Step 3: Implement factory router**

Create `backend/app/routers/factory.py`:

```python
"""Strategy factory API: run factory, check job status."""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.schema import FactoryJob
from app.services.factory_service import run_factory
from app.services.strategy_templates import generate_all_candidates

router = APIRouter(prefix="/api/strategy-factory", tags=["factory"])


class RunFactoryRequest(BaseModel):
    ts_code: str
    cutoff_date: date
    position_ratios: list[int] = [40, 30, 30]


async def _run_factory_bg(ts_code: str, cutoff_date: date, position_ratios: list[int], job_id: str):
    """Background wrapper: run_factory needs its own session since the request session is closed."""
    from app.db import async_session

    async with async_session() as db:
        # Update the existing job to running
        job = await db.get(FactoryJob, job_id)
        if not job:
            return

        try:
            await run_factory(db, ts_code, cutoff_date, position_ratios, job_id=job_id)
        except Exception as e:
            job = await db.get(FactoryJob, job_id)
            if job:
                job.status = "failed"
                job.error = str(e)
                await db.commit()


@router.post("/run")
async def run(
    body: RunFactoryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    import uuid

    job_id = str(uuid.uuid4())
    candidates = generate_all_candidates()

    job = FactoryJob(
        id=job_id,
        ts_code=body.ts_code,
        status="pending",
        total_candidates=len(candidates),
        config={
            "cutoff_date": body.cutoff_date.isoformat(),
            "position_ratios": body.position_ratios,
        },
    )
    db.add(job)
    await db.commit()

    background_tasks.add_task(
        _run_factory_bg, body.ts_code, body.cutoff_date, body.position_ratios, job_id
    )
    return {"job_id": job_id, "status": "pending", "total_candidates": len(candidates)}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(FactoryJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "total_candidates": job.total_candidates,
        "evaluated": job.evaluated,
        "passed": job.passed,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
```

- [ ] **Step 4: Update factory_service to accept job_id parameter**

Modify `run_factory` in `backend/app/services/factory_service.py` to accept an optional `job_id` parameter instead of always creating a new job — since the router creates the job first and passes the ID:

```python
async def run_factory(
    db: AsyncSession,
    ts_code: str,
    cutoff_date: date,
    position_ratios: list[int] | None = None,
    job_id: str | None = None,
) -> str:
    if position_ratios is None:
        position_ratios = [40, 30, 30]

    candidates = generate_all_candidates()

    # Use existing job or create new one
    if job_id:
        job = await db.get(FactoryJob, job_id)
        if job:
            job.status = "running"
            await db.commit()
        else:
            return job_id
    else:
        job_id = str(uuid.uuid4())
        job = FactoryJob(
            id=job_id,
            ts_code=ts_code,
            status="running",
            total_candidates=len(candidates),
            config={
                "cutoff_date": cutoff_date.isoformat(),
                "position_ratios": position_ratios,
            },
        )
        db.add(job)
        await db.commit()

    # ... rest of the function stays the same
```

- [ ] **Step 5: Register routers in main.py**

Modify `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.quotes import router as quotes_router
from app.routers.pools import router as pools_router
from app.routers.backtests import router as backtests_router
from app.routers.strategies import router as strategies_router
from app.routers.factory import router as factory_router


def create_app() -> FastAPI:
    app = FastAPI(title="Stock Quote API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(quotes_router)
    app.include_router(pools_router)
    app.include_router(backtests_router)
    app.include_router(strategies_router)
    app.include_router(factory_router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/backtests.py backend/app/routers/strategies.py backend/app/routers/factory.py backend/app/main.py backend/app/services/factory_service.py
git commit -m "feat: add backtest, strategy, factory API routers"
```

---

### Task 8: Frontend Types + API Client

**Files:**
- Create: `frontend/src/types/strategy.ts`
- Create: `frontend/src/api/strategy.ts`

- [ ] **Step 1: Create TypeScript types**

Create `frontend/src/types/strategy.ts`:

```typescript
export interface StrategyItem {
  id: number;
  name: string;
  template: string;
  parameters: Record<string, number>;
  is_pinned: boolean;
  annualized_return: number | null;
  net_profit: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
  total_trades: number | null;
  profit_factor: number | null;
  final_capital: number | null;
  job_id: string | null;
  created_at: string | null;
}

export interface FactoryJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  total_candidates: number;
  evaluated: number;
  passed: number;
  error: string | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface BacktestReport {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  ts_code: string;
  strategy_id: number | null;
  metrics: BacktestMetrics | null;
  trades: TradeDetail[] | null;
  equity_curve: EquityPoint[] | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface BacktestMetrics {
  net_profit: number;
  net_profit_pct: number;
  annualized_return: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  profit_factor: number;
  final_capital: number;
}

export interface TradeDetail {
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  shares: number;
  pnl: number;
}

export interface EquityPoint {
  date: string;
  equity: number;
}
```

- [ ] **Step 2: Create API client**

Create `frontend/src/api/strategy.ts`:

```typescript
import type {
  BacktestReport,
  FactoryJob,
  StrategyItem,
} from "../types/strategy";

const BASE = "/api";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

// Strategies
export async function listStrategies(tsCode: string): Promise<StrategyItem[]> {
  return json(`${BASE}/strategies?ts_code=${encodeURIComponent(tsCode)}`);
}

export async function pinStrategy(strategyId: number): Promise<void> {
  await json(`${BASE}/strategies/${strategyId}/pin`, { method: "POST" });
}

export async function unpinStrategy(strategyId: number): Promise<void> {
  await json(`${BASE}/strategies/${strategyId}/pin`, { method: "DELETE" });
}

// Factory
export async function runFactory(
  tsCode: string,
  cutoffDate: string,
  positionRatios: number[] = [40, 30, 30]
): Promise<{ job_id: string; status: string; total_candidates: number }> {
  return json(`${BASE}/strategy-factory/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ts_code: tsCode,
      cutoff_date: cutoffDate,
      position_ratios: positionRatios,
    }),
  });
}

export async function getFactoryJob(jobId: string): Promise<FactoryJob> {
  return json(`${BASE}/strategy-factory/jobs/${jobId}`);
}

// Backtests
export async function runBacktest(params: {
  ts_code: string;
  strategy_id?: number;
  start_date: string;
  end_date: string;
  initial_capital?: number;
  position_ratios?: number[];
}): Promise<{ id: string; status: string }> {
  return json(`${BASE}/backtests/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function getBacktestReport(runId: string): Promise<BacktestReport> {
  return json(`${BASE}/backtests/${runId}/report`);
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/strategy.ts frontend/src/api/strategy.ts
git commit -m "feat: add strategy frontend types and API client"
```

---

### Task 9: Frontend Strategy Store

**Files:**
- Create: `frontend/src/stores/strategyStore.ts`

- [ ] **Step 1: Create Zustand store**

Create `frontend/src/stores/strategyStore.ts`:

```typescript
import { create } from "zustand";
import type { BacktestReport, FactoryJob, StrategyItem } from "../types/strategy";
import {
  getBacktestReport,
  getFactoryJob,
  listStrategies,
  pinStrategy as apiPin,
  runBacktest as apiRunBacktest,
  runFactory as apiRunFactory,
  unpinStrategy as apiUnpin,
} from "../api/strategy";

interface StrategyState {
  strategies: StrategyItem[];
  loading: boolean;
  fetchStrategies: (tsCode: string) => Promise<void>;

  pinStrategy: (id: number) => Promise<void>;
  unpinStrategy: (id: number) => Promise<void>;

  // Factory
  factoryJob: FactoryJob | null;
  factoryRunning: boolean;
  startFactory: (tsCode: string, cutoffDate: string) => Promise<void>;
  pollFactoryJob: (jobId: string, tsCode: string) => void;
  stopPolling: () => void;

  // Backtest report
  selectedReport: BacktestReport | null;
  reportLoading: boolean;
  runAndShowReport: (tsCode: string, strategyId: number) => Promise<void>;
  clearReport: () => void;
}

let pollTimer: ReturnType<typeof setInterval> | null = null;

export const useStrategyStore = create<StrategyState>((set, get) => ({
  strategies: [],
  loading: false,
  fetchStrategies: async (tsCode) => {
    set({ loading: true });
    try {
      const data = await listStrategies(tsCode);
      set({ strategies: data, loading: false });
    } catch {
      set({ strategies: [], loading: false });
    }
  },

  pinStrategy: async (id) => {
    await apiPin(id);
    set((s) => ({
      strategies: s.strategies.map((st) =>
        st.id === id ? { ...st, is_pinned: true } : st
      ),
    }));
  },

  unpinStrategy: async (id) => {
    await apiUnpin(id);
    set((s) => ({
      strategies: s.strategies.map((st) =>
        st.id === id ? { ...st, is_pinned: false } : st
      ),
    }));
  },

  factoryJob: null,
  factoryRunning: false,
  startFactory: async (tsCode, cutoffDate) => {
    set({ factoryRunning: true, factoryJob: null });
    try {
      const res = await apiRunFactory(tsCode, cutoffDate);
      set({
        factoryJob: {
          job_id: res.job_id,
          status: "pending",
          total_candidates: res.total_candidates,
          evaluated: 0,
          passed: 0,
          error: null,
          created_at: null,
          completed_at: null,
        },
      });
      get().pollFactoryJob(res.job_id, tsCode);
    } catch {
      set({ factoryRunning: false });
    }
  },

  pollFactoryJob: (jobId, tsCode) => {
    get().stopPolling();
    pollTimer = setInterval(async () => {
      try {
        const job = await getFactoryJob(jobId);
        set({ factoryJob: job });
        if (job.status === "completed" || job.status === "failed") {
          get().stopPolling();
          set({ factoryRunning: false });
          if (job.status === "completed") {
            get().fetchStrategies(tsCode);
          }
        }
      } catch {
        get().stopPolling();
        set({ factoryRunning: false });
      }
    }, 2000);
  },

  stopPolling: () => {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  },

  selectedReport: null,
  reportLoading: false,
  runAndShowReport: async (tsCode, strategyId) => {
    set({ reportLoading: true, selectedReport: null });
    try {
      const res = await apiRunBacktest({
        ts_code: tsCode,
        strategy_id: strategyId,
        start_date: `${new Date().getFullYear() - 10}-01-01`,
        end_date: new Date().toISOString().slice(0, 10),
      });

      // Poll for completion
      const poll = setInterval(async () => {
        try {
          const report = await getBacktestReport(res.id);
          if (report.status === "completed" || report.status === "failed") {
            clearInterval(poll);
            set({ selectedReport: report, reportLoading: false });
          }
        } catch {
          clearInterval(poll);
          set({ reportLoading: false });
        }
      }, 1000);
    } catch {
      set({ reportLoading: false });
    }
  },

  clearReport: () => set({ selectedReport: null }),
}));
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/stores/strategyStore.ts
git commit -m "feat: add strategy Zustand store"
```

---

### Task 10: Frontend UI — NavBar + StrategyPanel

**Files:**
- Modify: `frontend/src/components/NavBar.tsx`
- Modify: `frontend/src/pages/QuotePage.tsx`
- Create: `frontend/src/components/StrategyPanel/index.tsx`
- Create: `frontend/src/components/StrategyPanel/StrategyList.tsx`
- Create: `frontend/src/components/StrategyPanel/FactoryControl.tsx`
- Create: `frontend/src/components/StrategyPanel/BacktestReport.tsx`
- Modify: `frontend/src/App.css`

- [ ] **Step 1: Update NavBar**

Replace `frontend/src/components/NavBar.tsx`:

```tsx
export type AppMode = "quote" | "pool" | "strategy";

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
        <button
          className={mode === "strategy" ? "active" : ""}
          onClick={() => onModeChange("strategy")}
        >
          策略
        </button>
      </div>
    </nav>
  );
}
```

- [ ] **Step 2: Update QuotePage**

Replace `frontend/src/pages/QuotePage.tsx`:

```tsx
import { useState } from "react";
import { NavBar } from "../components/NavBar";
import type { AppMode } from "../components/NavBar";
import { SearchPanel } from "../components/SearchPanel";
import { PoolPanel } from "../components/PoolPanel";
import { StrategyPanel } from "../components/StrategyPanel";
import { ChartArea } from "../components/ChartArea";

export function QuotePage() {
  const [mode, setMode] = useState<AppMode>("quote");

  const renderSidePanel = () => {
    switch (mode) {
      case "pool":
        return <PoolPanel />;
      case "strategy":
        return <StrategyPanel />;
      default:
        return <SearchPanel />;
    }
  };

  return (
    <div className="app-layout">
      <NavBar mode={mode} onModeChange={setMode} />
      <div className="app-body">
        {renderSidePanel()}
        <ChartArea />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create StrategyPanel index**

Create `frontend/src/components/StrategyPanel/index.tsx`:

```tsx
import { useEffect } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import { useStrategyStore } from "../../stores/strategyStore";
import { FactoryControl } from "./FactoryControl";
import { StrategyList } from "./StrategyList";
import { BacktestReport } from "./BacktestReport";

export function StrategyPanel() {
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const currentName = useQuoteStore((s) => s.currentName);
  const { strategies, loading, fetchStrategies, selectedReport, clearReport } =
    useStrategyStore();

  useEffect(() => {
    if (currentSymbol) {
      fetchStrategies(currentSymbol);
    }
  }, [currentSymbol, fetchStrategies]);

  return (
    <div className="strategy-panel">
      <div className="strategy-panel-header">
        <div className="strategy-panel-title">
          {currentSymbol ? (
            <>
              <span className="stock-name">{currentName}</span>
              <span className="stock-code">{currentSymbol}</span>
            </>
          ) : (
            <span className="stock-code">请先选择股票</span>
          )}
        </div>
      </div>

      {currentSymbol && <FactoryControl tsCode={currentSymbol} />}

      <div className="strategy-panel-body">
        {loading ? (
          <div className="strategy-loading">加载中...</div>
        ) : strategies.length === 0 ? (
          <div className="strategy-empty">
            {currentSymbol ? "暂无策略，请运行策略工厂" : "请先选择股票"}
          </div>
        ) : (
          <StrategyList strategies={strategies} tsCode={currentSymbol} />
        )}
      </div>

      {selectedReport && (
        <BacktestReport report={selectedReport} onClose={clearReport} />
      )}
    </div>
  );
}
```

- [ ] **Step 4: Create StrategyList**

Create `frontend/src/components/StrategyPanel/StrategyList.tsx`:

```tsx
import type { StrategyItem } from "../../types/strategy";
import { useStrategyStore } from "../../stores/strategyStore";

interface Props {
  strategies: StrategyItem[];
  tsCode: string;
}

function pct(v: number | null): string {
  if (v === null) return "-";
  return (v * 100).toFixed(2) + "%";
}

function num(v: number | null, decimals = 2): string {
  if (v === null) return "-";
  return v.toFixed(decimals);
}

export function StrategyList({ strategies, tsCode }: Props) {
  const { pinStrategy, unpinStrategy, runAndShowReport } = useStrategyStore();

  return (
    <div className="strategy-list-wrap">
      <table className="strategy-table">
        <thead>
          <tr>
            <th>Pin</th>
            <th>策略名称</th>
            <th>年化收益</th>
            <th>最大回撤</th>
            <th>胜率</th>
            <th>交易次数</th>
            <th>盈亏比</th>
            <th>净利润</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {strategies.map((s) => (
            <tr key={s.id} className={s.is_pinned ? "pinned-row" : ""}>
              <td>
                <button
                  className="pin-btn"
                  onClick={() =>
                    s.is_pinned ? unpinStrategy(s.id) : pinStrategy(s.id)
                  }
                >
                  {s.is_pinned ? "★" : "☆"}
                </button>
              </td>
              <td className="strategy-name-cell">{s.name}</td>
              <td className={s.annualized_return && s.annualized_return > 0 ? "up" : "down"}>
                {pct(s.annualized_return)}
              </td>
              <td>{pct(s.max_drawdown)}</td>
              <td>{pct(s.win_rate)}</td>
              <td>{s.total_trades ?? "-"}</td>
              <td>{num(s.profit_factor)}</td>
              <td className={s.net_profit && s.net_profit > 0 ? "up" : "down"}>
                {num(s.net_profit)}
              </td>
              <td>
                <button
                  className="report-btn"
                  onClick={() => runAndShowReport(tsCode, s.id)}
                >
                  回测
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 5: Create FactoryControl**

Create `frontend/src/components/StrategyPanel/FactoryControl.tsx`:

```tsx
import { useStrategyStore } from "../../stores/strategyStore";

interface Props {
  tsCode: string;
}

export function FactoryControl({ tsCode }: Props) {
  const { factoryJob, factoryRunning, startFactory } = useStrategyStore();

  const handleRun = () => {
    const today = new Date().toISOString().slice(0, 10);
    startFactory(tsCode, today);
  };

  const progress =
    factoryJob && factoryJob.total_candidates > 0
      ? Math.round((factoryJob.evaluated / factoryJob.total_candidates) * 100)
      : 0;

  return (
    <div className="factory-control">
      <button
        className="factory-run-btn"
        onClick={handleRun}
        disabled={factoryRunning}
      >
        {factoryRunning ? "运行中..." : "运行策略工厂"}
      </button>

      {factoryJob && (
        <div className="factory-status">
          <div className="factory-progress-bar">
            <div
              className="factory-progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="factory-progress-text">
            {factoryJob.status === "completed" ? (
              <span>完成 — 通过 {factoryJob.passed} 个策略</span>
            ) : factoryJob.status === "failed" ? (
              <span className="factory-error">失败: {factoryJob.error}</span>
            ) : (
              <span>
                已评估 {factoryJob.evaluated}/{factoryJob.total_candidates} ({progress}%)
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Create BacktestReport modal**

Create `frontend/src/components/StrategyPanel/BacktestReport.tsx`:

```tsx
import type { BacktestReport as ReportType } from "../../types/strategy";

interface Props {
  report: ReportType;
  onClose: () => void;
}

function pct(v: number): string {
  return (v * 100).toFixed(2) + "%";
}

export function BacktestReport({ report, onClose }: Props) {
  const m = report.metrics;

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div
        className="report-dialog"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-header">
          <span>回测报告</span>
          <button onClick={onClose}>✕</button>
        </div>

        {report.status === "failed" ? (
          <div className="report-error">回测失败</div>
        ) : !m ? (
          <div className="report-loading">计算中...</div>
        ) : (
          <div className="report-body">
            <div className="report-metrics">
              <div className="metric-card">
                <span className="metric-label">净利润</span>
                <span className={`metric-value ${m.net_profit > 0 ? "up" : "down"}`}>
                  {m.net_profit.toFixed(2)}
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">年化收益</span>
                <span className={`metric-value ${m.annualized_return > 0 ? "up" : "down"}`}>
                  {pct(m.annualized_return)}
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">最大回撤</span>
                <span className="metric-value">{pct(m.max_drawdown)}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">胜率</span>
                <span className="metric-value">{pct(m.win_rate)}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">交易次数</span>
                <span className="metric-value">{m.total_trades}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">盈亏比</span>
                <span className="metric-value">{m.profit_factor.toFixed(2)}</span>
              </div>
            </div>

            {report.trades && report.trades.length > 0 && (
              <div className="report-trades">
                <h4>交易明细</h4>
                <table className="trades-table">
                  <thead>
                    <tr>
                      <th>买入日期</th>
                      <th>卖出日期</th>
                      <th>买入价</th>
                      <th>卖出价</th>
                      <th>数量</th>
                      <th>盈亏</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.trades.map((t, i) => (
                      <tr key={i}>
                        <td>{t.entry_date}</td>
                        <td>{t.exit_date}</td>
                        <td>{t.entry_price.toFixed(2)}</td>
                        <td>{t.exit_price.toFixed(2)}</td>
                        <td>{t.shares.toFixed(0)}</td>
                        <td className={t.pnl > 0 ? "up" : "down"}>
                          {t.pnl.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {report.equity_curve && report.equity_curve.length > 0 && (
              <div className="report-equity">
                <h4>资金曲线</h4>
                <div className="equity-chart-placeholder">
                  起始: {report.equity_curve[0].equity.toFixed(2)} →
                  结束: {report.equity_curve[report.equity_curve.length - 1].equity.toFixed(2)}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Add CSS for strategy panel**

Append to `frontend/src/App.css`:

```css
/* Strategy Panel */
.strategy-panel {
  width: 100%;
  max-width: 100%;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
  overflow: hidden;
}

.strategy-panel-header {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
}
.strategy-panel-title {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.strategy-panel-body {
  flex: 1;
  overflow: auto;
  min-height: 0;
}

.strategy-loading,
.strategy-empty {
  color: var(--text-muted);
  text-align: center;
  padding: 24px;
  font-size: 12px;
}

/* Factory Control */
.factory-control {
  padding: 8px 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 12px;
}
.factory-run-btn {
  background: var(--accent);
  color: #000;
  border: none;
  padding: 6px 16px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
}
.factory-run-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.factory-status {
  flex: 1;
  min-width: 0;
}
.factory-progress-bar {
  height: 6px;
  background: var(--bg-tertiary);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 4px;
}
.factory-progress-fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.3s ease;
}
.factory-progress-text {
  font-size: 11px;
  color: var(--text-secondary);
}
.factory-error {
  color: var(--up);
}

/* Strategy Table */
.strategy-list-wrap {
  overflow: auto;
}
.strategy-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
  white-space: nowrap;
}
.strategy-table th {
  text-align: left;
  padding: 6px 8px;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  font-weight: normal;
  position: sticky;
  top: 0;
  background: var(--bg-secondary);
}
.strategy-table td {
  padding: 5px 8px;
  border-bottom: 1px solid var(--bg-primary);
}
.strategy-table tr:hover {
  background: var(--bg-tertiary);
}
.pinned-row {
  background: rgba(76, 201, 240, 0.05);
}
.strategy-name-cell {
  color: var(--accent);
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.pin-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  color: var(--accent);
  padding: 0 2px;
}
.report-btn {
  background: var(--bg-tertiary);
  border: none;
  color: var(--text-primary);
  padding: 2px 8px;
  border-radius: 3px;
  cursor: pointer;
  font-size: 11px;
}
.report-btn:hover {
  background: var(--accent);
  color: #000;
}

/* Report Dialog */
.report-dialog {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  width: 680px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.report-body {
  padding: 14px;
  overflow-y: auto;
}
.report-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-bottom: 16px;
}
.metric-card {
  background: var(--bg-tertiary);
  padding: 8px 10px;
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.metric-label {
  font-size: 10px;
  color: var(--text-muted);
}
.metric-value {
  font-size: 16px;
  font-weight: 600;
}
.report-error,
.report-loading {
  padding: 24px;
  text-align: center;
  color: var(--text-muted);
}

.report-trades h4,
.report-equity h4 {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}
.trades-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}
.trades-table th {
  text-align: left;
  padding: 4px 6px;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  font-weight: normal;
}
.trades-table td {
  padding: 4px 6px;
  border-bottom: 1px solid var(--bg-primary);
}
.equity-chart-placeholder {
  background: var(--bg-tertiary);
  padding: 12px;
  border-radius: 4px;
  font-size: 12px;
  color: var(--text-secondary);
  text-align: center;
}
```

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/NavBar.tsx frontend/src/pages/QuotePage.tsx frontend/src/components/StrategyPanel/ frontend/src/App.css
git commit -m "feat: add strategy panel UI with factory control and backtest report"
```

---

### Task 11: Integration Testing + Fixes

- [ ] **Step 1: Add pandas/numpy to requirements**

Add to `backend/requirements.txt`:
```
pandas>=2.0
numpy>=1.24
```

- [ ] **Step 2: Run all backend tests**

```bash
cd backend && python -m pytest tests/ -v
```

Fix any failures.

- [ ] **Step 3: Start backend and verify API endpoints**

```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

Test endpoints:
```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/strategies?ts_code=600519.SH
```

- [ ] **Step 4: Start frontend and verify UI**

```bash
cd frontend && npm run dev
```

Navigate to http://localhost:5173, click "策略" tab, verify it renders.

- [ ] **Step 5: Run migration on Docker (if using Docker)**

```bash
docker compose exec backend alembic upgrade head
```

- [ ] **Step 6: Commit final integration**

```bash
git add -A
git commit -m "feat: complete strategy factory V1 + backtest engine integration"
```
