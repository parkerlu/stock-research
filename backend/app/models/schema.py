from datetime import date, datetime

from sqlalchemy import Boolean, BigInteger, Date, DateTime, Index, Integer, JSON, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class StockBasic(Base):
    __tablename__ = "stock_basic"

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(6), index=True)
    # 64 而非 20: ETF/LOF 简称远长于股票简称，实测 tushare fund_basic 有 113/2879
    # 个超过 20 字符，最长 32 (如 "广发道琼斯美国石油开发与生产指数(QDII-LOF)-A-CNY")。
    # 之前 20 会让整批 upsert 直接回滚，ETF 元数据一条都写不进去。
    name: Mapped[str] = mapped_column(String(64), index=True)
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
    actions: Mapped[list | None] = mapped_column(JSON)
    equity_curve: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)


# =========================================================================
# 虚拟盘 (Paper Trading) — 按验证过的 chan-2buy 最优配置实盘跟踪
# =========================================================================

class PaperAccount(Base):
    """一个虚拟账户。config 存策略参数, 改参数应新建账户而非改旧的 ——
    否则历史成交和当前规则对不上, 跟踪记录就失去意义。"""
    __tablename__ = "paper_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)
    initial_capital: Mapped[float] = mapped_column(Numeric(16, 2))
    cash: Mapped[float] = mapped_column(Numeric(16, 2))
    slots: Mapped[int] = mapped_column(Integer, default=10)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    started_on: Mapped[date] = mapped_column(Date)
    # 已推进到哪一天 —— 防止同一天重复结算
    last_run_date: Mapped[date | None] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PaperPosition(Base):
    """一笔持仓。分批止盈后 shares 递减, 清零即 status=closed。"""
    __tablename__ = "paper_position"
    __table_args__ = (Index("ix_paper_pos_acct", "account_id", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    ts_code: Mapped[str] = mapped_column(String(12), index=True)
    name: Mapped[str | None] = mapped_column(String(64))
    open_date: Mapped[date] = mapped_column(Date)
    open_price: Mapped[float] = mapped_column(Numeric(12, 4))
    init_shares: Mapped[int] = mapped_column(Integer)
    shares: Mapped[int] = mapped_column(Integer)          # 当前剩余
    stop_price: Mapped[float] = mapped_column(Numeric(12, 4))
    # 移动止盈用: 持仓期间见过的最高价
    peak_price: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    tier1_done: Mapped[bool] = mapped_column(Boolean, default=False)
    tier2_done: Mapped[bool] = mapped_column(Boolean, default=False)
    realized_pnl: Mapped[float] = mapped_column(Numeric(16, 2), default=0)
    status: Mapped[str] = mapped_column(String(10), default="open")   # open | closed
    close_date: Mapped[date | None] = mapped_column(Date)
    close_reason: Mapped[str | None] = mapped_column(String(20))


class PaperTrade(Base):
    """所有动作的流水 —— 买入/分批止盈/止损/清仓, 一条不落。"""
    __tablename__ = "paper_trade"
    __table_args__ = (Index("ix_paper_trade_acct", "account_id", "trade_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    position_id: Mapped[int | None] = mapped_column(Integer, index=True)
    ts_code: Mapped[str] = mapped_column(String(12), index=True)
    name: Mapped[str | None] = mapped_column(String(64))
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    action: Mapped[str] = mapped_column(String(16))   # buy|tier1|tier2|stop|timeout|close
    price: Mapped[float] = mapped_column(Numeric(12, 4))
    shares: Mapped[int] = mapped_column(Integer)
    amount: Mapped[float] = mapped_column(Numeric(16, 2))
    fee: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    pnl: Mapped[float | None] = mapped_column(Numeric(16, 2))
    pnl_pct: Mapped[float | None] = mapped_column(Numeric(10, 4))
    note: Mapped[str | None] = mapped_column(Text)


class PaperEquity(Base):
    """每日净值快照 —— 画净值曲线用。"""
    __tablename__ = "paper_equity"
    __table_args__ = (Index("ix_paper_eq", "account_id", "trade_date", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    trade_date: Mapped[date] = mapped_column(Date)
    cash: Mapped[float] = mapped_column(Numeric(16, 2))
    market_value: Mapped[float] = mapped_column(Numeric(16, 2))
    equity: Mapped[float] = mapped_column(Numeric(16, 2))
    n_positions: Mapped[int] = mapped_column(Integer, default=0)


class ChanSignal(Base):
    """缠论买点的预计算缓存.

    虚拟盘回放需要"点一下走一天", 而每天现算 4400 只票的缠论要 75 秒 ——
    根本没法交互。但整段历史一次性算完只要几十秒(信号本身不依赖回放进度),
    所以把结果落表, 回放时变成一次索引查询。

    trade_date 存的是**可操作日**(分型 + CONFIRM_LAG), 不是分型日 ——
    直接对应下单日的前一天, 消费方不用再关心 lag。
    """
    __tablename__ = "chan_signal"
    __table_args__ = (Index("ix_chan_signal_date", "trade_date", "kind"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(12), index=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    kind: Mapped[str] = mapped_column(String(4))          # "1" | "2"
    fractal_date: Mapped[date] = mapped_column(Date)      # 分型日(事后位置), 供核对
