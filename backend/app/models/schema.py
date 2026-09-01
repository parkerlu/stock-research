from datetime import date, datetime

from sqlalchemy import Boolean, BigInteger, Date, DateTime, Index, Integer, JSON, Numeric, String, Text, UniqueConstraint
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


class SignalSnapshot(Base):
    """每日收盘后把"今天算出来的信号"原样封存, 之后不再改动.

    为什么需要
    ----------
    缠论的笔用的是 ZigZag 式端点: 连续同类分型只保留更极端的那个。一个底分型
    可能在若干根之后被更低的底分型顶掉, 原来的 2 类买点就消失。chan_signal 表
    若用全历史重算, 表里只剩"事后仍然成立"的信号 —— 实测约一半信号被这样删掉,
    而且删掉的正是输家 (被推翻组后续 12 日 -1.81%/胜率 34.6%, 幸存组 +3.60%/56.4%)。
    回测读这张表就等于预知了哪些信号不会被推翻。

    这张快照表是防线: 当天算出什么就存什么, 只增不改。日后拿它和重算结果比对,
    就能用**真实的前进数据**量出漂移率, 不依赖任何截断复算的模拟 (那类脚本
    自己也可能写错 —— 我就写错过一次, 把幸存者偏差写进了检验本身)。
    """

    __tablename__ = "signal_snapshot"
    __table_args__ = (
        UniqueConstraint("snapshot_date", "ts_code", "trade_date", "kind",
                         name="uq_signal_snapshot"),
        Index("ix_signal_snapshot_date", "snapshot_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 封存这一批信号的日期 = 当天收盘, 用截至当天的数据算出来的
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    ts_code: Mapped[str] = mapped_column(String(12), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    fractal_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    kind: Mapped[str] = mapped_column(String(2), nullable=False)
    # 当时的价与流动性, 用来算漂移后的价格差
    close_at_signal: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    amount_20d_k: Mapped[float | None] = mapped_column(Numeric(16, 2), nullable=True)
    # 后续核对结果: NULL=还没查, true=重算后仍在, false=被重绘掉了
    still_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    checked_on: Mapped[date | None] = mapped_column(Date, nullable=True)


class StrategySignal(Base):
    """策略买点的预计算缓存 —— 任何策略共用一张表, 用 strategy 列区分.

    为什么要缓存: 虚拟盘要"点一下走一天", 每天现算全市场 4400 只要几十秒,
    交互完全不可用。买点只依赖该票自身历史、与回放进度无关, 整段一次算完即可。

    ⚠️ 纪律(chan-2buy 的教训): 只做每日增量 append, 不要全历史重建。
    ZigZag 类指标会在重建时抹掉"当时成立、后来被更低的低点撤销"的信号,
    而抹掉的正是输家 —— 回测读表就等于预知哪些信号不会被推翻。
    换用任何新策略前
    必须用 scripts/screening/causal_check.py 重新验一遍。
    """

    __tablename__ = "strategy_signal"
    __table_args__ = (
        UniqueConstraint("strategy", "ts_code", "trade_date", name="uq_strategy_signal"),
        Index("ix_strategy_signal_lookup", "strategy", "trade_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    ts_code: Mapped[str] = mapped_column(String(12), nullable=False)
    # 可操作日: 信号成立的交易日, 次日开盘成交
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
