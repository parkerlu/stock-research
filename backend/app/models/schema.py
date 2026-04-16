from datetime import date, datetime

from sqlalchemy import Boolean, BigInteger, Date, DateTime, Index, Numeric, String
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
