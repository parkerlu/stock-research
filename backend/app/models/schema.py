from datetime import date, datetime

from sqlalchemy import Boolean, BigInteger, Date, DateTime, Index, Integer, JSON, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class StockBasic(Base):
    __tablename__ = "stock_basic"

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(6), index=True)
    name: Mapped[str] = mapped_column(String(20), index=True)
    area: Mapped[str | None] = mapped_column(String(10))
    industry: Mapped[str | None] = mapped_column(String(20))
    market: Mapped[str | None] = mapped_column(String(10))
    list_date: Mapped[date | None] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class DailyCandle(Base):
    __tablename__ = "daily_candle"
    __table_args__ = (
        Index("ix_daily_candle_lookup", "ts_code", "trade_date"),
    )

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Numeric(12, 4))
    high: Mapped[float] = mapped_column(Numeric(12, 4))
    low: Mapped[float] = mapped_column(Numeric(12, 4))
    close: Mapped[float] = mapped_column(Numeric(12, 4))
    vol: Mapped[int] = mapped_column(BigInteger)
    amount: Mapped[float] = mapped_column(Numeric(18, 4))
    adj_factor: Mapped[float] = mapped_column(Numeric(12, 6), default=1.0)
    source: Mapped[str] = mapped_column(String(10), default="tushare")


class SearchHistory(Base):
    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(12), index=True)
    searched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Favorite(Base):
    __tablename__ = "favorite"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(12), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


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


class FactoryJob(Base):
    __tablename__ = "factory_job"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ts_code: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    total_candidates: Mapped[int] = mapped_column(Integer, default=0)
    evaluated: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[int] = mapped_column(Integer, default=0)
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
    total_trades: Mapped[int | None] = mapped_column(Integer)
    profit_factor: Mapped[float | None] = mapped_column(Numeric(10, 4))
    final_capital: Mapped[float | None] = mapped_column(Numeric(14, 4))
    metrics: Mapped[dict | None] = mapped_column(JSON)
    job_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class BacktestRun(Base):
    __tablename__ = "backtest_run"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ts_code: Mapped[str] = mapped_column(String(20), index=True)
    strategy_id: Mapped[int | None] = mapped_column(Integer)
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
