from datetime import date, datetime

from sqlalchemy import Boolean, BigInteger, Column, Date, DateTime, Index, Integer, JSON, Numeric, String, Text, UniqueConstraint
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


class ConceptSector(Base):
    """概念板块 —— 同花顺口径。list_date 是板块被创建的日期,
    本身就是"市场正式承认这个主题"的时间戳。"""

    __tablename__ = "concept_sector"

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)  # 885728.TI
    name: Mapped[str] = mapped_column(String(64), index=True)
    count: Mapped[int | None] = mapped_column(Integer)          # 成分股数
    exchange: Mapped[str | None] = mapped_column(String(10))
    list_date: Mapped[date | None] = mapped_column(Date, index=True)
    type: Mapped[str | None] = mapped_column(String(4))          # N=概念 I=行业 R=地域
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)


class ConceptMember(Base):
    """板块成分股 —— 多对多。⚠️ 上游只给当前快照, 没有历史进出记录,
    因此这张表能回答"现在谁属于哪个板块", 但不能用来做历史回测。"""

    __tablename__ = "concept_member"
    __table_args__ = (
        Index("ix_concept_member_stock", "ts_code"),
        Index("ix_concept_member_sector", "sector_code"),
        UniqueConstraint("sector_code", "ts_code", name="uq_concept_member"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sector_code: Mapped[str] = mapped_column(String(12))
    ts_code: Mapped[str] = mapped_column(String(12))
    name: Mapped[str | None] = mapped_column(String(64))


class MaimaiSignal(Base):
    """买卖很准 v3 —— 原始信号 + 模型评分。

    原指标的买点(超卖结束→反转确认)本身与随机无异(32万信号实测胜率48.7%,
    中位-0.132%)。这里用 XGBoost 在【同日×同波动层】中性化标签上训练过滤器,
    样本外把胜率提到 51.3%、中位数翻正到 +0.264%, 三个波动档超出全为正。

    score: 模型原始打分(同波动层内超额的预测值)
    rank_pct: 当日所有信号中的分位 (1.0 = 最好)。前端据此分强/中/弱三档。
    """

    __tablename__ = "maimai_signal"
    __table_args__ = (
        Index("ix_maimai_lookup", "ts_code", "trade_date"),
        Index("ix_maimai_date", "trade_date"),
    )

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    side: Mapped[str] = mapped_column(String(4), primary_key=True, default="buy")  # buy/sell
    score: Mapped[float] = mapped_column(Numeric(12, 6))
    rank_pct: Mapped[float] = mapped_column(Numeric(6, 4))
    grade: Mapped[str] = mapped_column(String(4))       # 强 / 中 / 弱


class PumpSignal(Base):
    """主力吸筹 —— 预测未来10日内出现拉升(单日涨幅>7% 且量>20日均量2倍)的概率。

    核心信息来自筹码分布(cyq_perf)的【获利盘族】特征, 占模型重要性 60%:
      - 获利盘变化: 低位筹码换手的痕迹
      - 获利盘背离: 价格没动但获利盘上升 = 主力在低位接货, 价格图上看不出来
    「筹码集中度」反而无效(Q10/Q1=1.10) —— 吸筹不一定让筹码变集中,
    但一定会改变获利盘结构。

    样本外(2020-2026 walk-forward, 160万+样本):
      Q10/Q1 = 4.81, 按天 t = 77.4, 逐年 1.70~2.02 倍无失效。
      Top10% 拉升概率 17.5% (基础 9.8%), Top1% 达 29.7%。
    """

    __tablename__ = "pump_signal"
    __table_args__ = (
        Index("ix_pump_lookup", "ts_code", "trade_date"),
        Index("ix_pump_date_rank", "trade_date", "rank_pct"),
    )

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    prob: Mapped[float] = mapped_column(Numeric(8, 5))
    rank_pct: Mapped[float] = mapped_column(Numeric(6, 4))
    grade: Mapped[str] = mapped_column(String(4))


class DongliSignal(Base):
    """动力线上穿 0.2 的日子 —— 即原「低点组合」里的阶段底部信号。

    单独落表是为了能和别的信号做 SQL join(与 maimai_signal 同样的用法)。
    它自己几乎没有预测力(「10日涨10%」命中率 18.7%, 基准 17.45%),
    价值在于给主力吸筹做最后一道收紧: 吸筹强 25.7% → 叠上动力线 32.6%。

    ⚠️ 无前视: LLV(10)/HHV(25)/EMA(4) 全是向后看的窗口, 309 次截断重算零不一致。
    """
    __tablename__ = "dongli_signal"
    __table_args__ = (Index("ix_dongli_date", "trade_date"),)

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    dl_value: Mapped[float] = mapped_column(Numeric(8, 4))


class IndexDaily(Base):
    """指数日线 —— 中证1000 / 中证500 / 沪深300 / 上证指数。

    主要用途是【择时基准】: 突破预警离了大盘择时不能用(裸跑比值 0.17),
    而原来的择时口径是"全市场等权指数"——每次都要现算一遍。落表后可以直接查,
    而且中证1000 是真实可交易的小盘指数, 比等权口径更贴近这些信号的标的。
    """
    __tablename__ = "index_daily"
    __table_args__ = (Index("ix_index_lookup", "ts_code", "trade_date"),)

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Numeric(12, 4))
    high: Mapped[float] = mapped_column(Numeric(12, 4))
    low: Mapped[float] = mapped_column(Numeric(12, 4))
    close: Mapped[float] = mapped_column(Numeric(12, 4))
    vol: Mapped[float] = mapped_column(Numeric(20, 2))
    amount: Mapped[float] = mapped_column(Numeric(20, 2))


class IndexBar(Base):
    """指数多周期 K 线 —— 日线之外的周/月/120/60/30 分钟。

    日线在 index_daily(择时基准用), 这张表放其他周期供看图。
    分钟级只有 tushare 的 stk_mins 能给指数数据(baostock 对指数返回 0 行),
    而它【限速 1 次/分钟】—— 导入必须节流, 别指望一次拉完全部历史。
    """
    __tablename__ = "index_bar"
    __table_args__ = (Index("ix_ibar_lookup", "ts_code", "freq", "bar_time"),)

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    freq: Mapped[str] = mapped_column(String(8), primary_key=True)   # 1w/1m/120min/60min/30min
    bar_time: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    open: Mapped[float] = mapped_column(Numeric(12, 4))
    high: Mapped[float] = mapped_column(Numeric(12, 4))
    low: Mapped[float] = mapped_column(Numeric(12, 4))
    close: Mapped[float] = mapped_column(Numeric(12, 4))
    vol: Mapped[float] = mapped_column(Numeric(20, 2))


class MaimaiWeekly(Base):
    """买卖很准 周线版 —— 周线买线 > 0 的那些周(超卖【状态】, 不是事件)。

    ⚠️ 与日线版是两个东西, 不要混:
      日线版取【边沿】(买线 >0→0, 状态刚结束), 事件标签 -1.00pp, 七年为负
      周线版取【状态】(买线 >0 的每一周), 持有8周 +3.39pp, 按周 t=5.56, 九格全正(下周一开盘口径)

    为什么是"每一周"而不是"起始周"(实测, 持有8周):
      买线>0 每周   +3.45pp  t=5.05  九格 +0.38  ✅
      状态起始周    +2.51pp  t=2.56  九格 -0.01
      连续第2周     +2.67pp  t=1.32  九格 -0.79
    价值在"处于超卖状态"这个持续条件本身, 不在"刚进入"那一刻 —— 所以它是
    状态指标(像均线多头排列), 不是事件信号(像 SAR 翻多), 前端画成副图色带。

    持有期单峰在 8 周(2周 +1.68 / 4周 +3.20 / 8周 +3.45 / 12周 +2.55 /
    16周 +1.86 / 26周 +1.03) —— 单峰形状比孤立高点可信。

    ⚠️ 买入价用【下周一开盘】而非周五收盘 —— 信号在周五收盘才成立, 那一刻买不到。
    实测两种口径几乎无差(8周 +3.45pp → +3.39pp, 跳空均值 -0.006%), 因为它是
    状态指标, 没有"消息公布"引发的抢跑; 龙虎榜那种盘后信号跳空是 +1.31%,
    换算下来命中率要掉 4.31pp。口径这件事不能靠运气, 每个新指标都要单独验。

    两个负年 2020(-1.7pp)/2023(-1.3pp) 都是极端抱团行情(核心资产 / AI 独走),
    资金集中在少数龙头, 而它抓的是超卖反弹的普通股。失效模式可解释, 且幅度浅。
    """
    __tablename__ = "maimai_weekly"
    __table_args__ = (Index("ix_mmwk_date", "week_end"),)

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    week_end: Mapped[date] = mapped_column(Date, primary_key=True)
    buy_line: Mapped[float] = mapped_column(Numeric(10, 4))


class SarSignal(Base):
    """SAR 预警 —— SAR 由空翻多 × 近5日内主力吸筹强档。

    与突破预警是同一族(都是"趋势确认 × 吸筹"), 差别在性格:
      突破预警  比值 2.62, 年化 +39.96% / 回撤 15.24%, 胜率约 35%
      SAR预警   比值 1.74, 年化 +34.65% / 回撤 19.96%, 胜率  52.5%
    突破版更狠但靠少数大赢家; SAR 版胜率高 17 个点, 连亏的串更短, 好拿住。

    ⚠️ 同样必须配大盘择时(中证1000 > 自身MA20): 裸跑比值只有 0.29(回撤 40.6%)。

    信号级(标签=10个交易日内触及+10%, 基准 17.55%):
      命中 29.3% · 超基准 +10.80pp · 逐年 0 负年 · 按天 t=20.40 ·
      九格最小 +3.93pp —— 九格最小值是所有组合里最高的, 各波动/市值格子都均匀。
    """
    __tablename__ = "sar_signal"
    __table_args__ = (Index("ix_sar_date", "trade_date"),)

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    prob: Mapped[float] = mapped_column(Numeric(8, 5))


class BreakoutSignal(Base):
    """突破预警 —— 收盘创 60 日新高 × 近5日内主力吸筹强档。

    与「拉升预警」(动力线×吸筹)是一对反向的东西:
      拉升预警  抄底型, 信号日 89.1% 处于空头排列, 组合天花板 0.71
      突破预警  趋势型, 信号日  0.0% 处于空头排列, 组合可达 1.21~1.91

    信号级两者几乎一样(命中 32.0% vs 32.6%), 差别全在【相关性】:
    突破型信号高度同步, 裸跑一起崩(回撤 59.7%), 但也因此大盘择时能整批挡住
    (回撤降到 21.5%, 年化反而从 10.0% 升到 41.0%)。
    ⚠️ 必须配大盘择时用 —— 不带择时比值只有 0.17。

    验证(2018-06 起, 2227 只抽样, 标签=10个交易日内触及+10%):
      命中率 32.0%(基准 17.52%) · 逐年 0 负年 · 按天 t=41.53 · 九格最小 +2.68pp
    """
    __tablename__ = "breakout_signal"
    __table_args__ = (Index("ix_breakout_date", "trade_date"),)

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    prob: Mapped[float] = mapped_column(Numeric(8, 5))       # 吸筹概率
    hh60: Mapped[float] = mapped_column(Numeric(12, 4))      # 被突破的60日高点


class DidianSignal(Base):
    """低点组合 v2 —— 原指标「阶段底部」(动力线上穿0.2) + 模型过滤。

    三个训练指标里最强的一个:
      原信号本身 +2.924%(H=20), 胜率 53.8%, 按天 t=2.87 —— 起点就优于买卖很准 2.7 倍
      过滤后 Top10%: +4.983%, 中位 +2.563%, 胜率 57.7%, **按天 t=3.88**
      波动层内三档超出全为正且 t 全部 >2.5 (2.55 / 2.80 / 3.74)

    特征里 `筹码宽度` 排第一(0.130) —— 而同一特征在主力吸筹里完全无效(Q10/Q1=1.10)。
    筹码数据源通用, 但具体哪个筹码特征有效, 要按目标重新筛。

    ⚠️ 持有期 H=20(baseline 在此窗口最强), 与买卖很准的 H=10 不同。
    """

    __tablename__ = "didian_signal"
    __table_args__ = (
        Index("ix_didian_lookup", "ts_code", "trade_date"),
        Index("ix_didian_date_rank", "trade_date", "rank_pct"),
    )

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    score: Mapped[float] = mapped_column(Numeric(12, 6))
    rank_pct: Mapped[float] = mapped_column(Numeric(6, 4))
    grade: Mapped[str] = mapped_column(String(4))


class Maimai35Signal(Base):
    """买卖很准 v3.5 —— 原指标【参数重扫】后的版本。

    移植过来的参数(MA5/LLV10/连续5)是照抄 TDX 公式的, 从没针对 A 股验证过。
    36 组网格扫描后它只排 9/33。改用 MA8/LLV20/连续10:

        全市场对比 (H=20, vs 同期随机对照)
        原参数  28.1万信号  胜率 50.9%  超出 +0.190pp
        v3.5     9.5万信号  胜率 56.2%  超出 +0.602pp   ← +5.3pp

    ⚠️ 代价: 它是"放大器"。好年份 +1.0pp, 坏年份 -1.0pp(原参数只有 ±0.5pp),
    信号量少三分之二。2020/2023/2025 三年跑输随机 —— 超卖反转在普涨行情里
    天然吃亏, 那时候随便买什么都涨, 严格筛选反而错过。
    """

    __tablename__ = "maimai35_signal"
    __table_args__ = (
        Index("ix_mm35_lookup", "ts_code", "trade_date"),
        Index("ix_mm35_date_rank", "trade_date", "rank_pct"),
    )

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    score: Mapped[float] = mapped_column(Numeric(12, 6))
    rank_pct: Mapped[float] = mapped_column(Numeric(6, 4))
    grade: Mapped[str] = mapped_column(String(4))


class TopList(Base):
    """龙虎榜每日明细 —— 交易所公布的实名席位成交, 是"谁在买"的硬记录。

    覆盖率仅 1.2% 的股票日(只有异动才上榜), 所以不能当主要特征,
    但作为共振的第三个条件很有效:
      v4 (v3强×吸筹强)          胜率 63.7%  3.0个/天
      + 近5日机构净买入          胜率 76.2%  0.3个/天

    同等涨幅下对比(排除"异动本身")最有价值的是【−2~2%】那一档: +4.95pp。
    价格几乎没动却上了龙虎榜 = 大资金在换手但价格没反应, 与「获利盘背离」同一逻辑,
    只是这里是实名席位的硬记录, 不是从价格反推的。

    net_amount 分五档: Q1净卖 14.93% → Q4 28.63% → Q5净买 25.20%,
    **不是越多越好** —— 极端净买入(Q5)反而回落, 可能是游资对倒或已拉过高。
    """

    __tablename__ = "top_list"
    __table_args__ = (
        Index("ix_toplist_lookup", "ts_code", "trade_date"),
        Index("ix_toplist_date", "trade_date"),
    )

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    close: Mapped[float | None] = mapped_column(Numeric(12, 4))
    pct_change: Mapped[float | None] = mapped_column(Numeric(10, 4))
    turnover_rate: Mapped[float | None] = mapped_column(Numeric(10, 4))
    l_buy: Mapped[float | None] = mapped_column(Numeric(18, 2))
    l_sell: Mapped[float | None] = mapped_column(Numeric(18, 2))
    net_amount: Mapped[float | None] = mapped_column(Numeric(18, 2))
    reason: Mapped[str | None] = mapped_column(String(128))


class ChipsScore(Base):
    """筹码模型每日全市场打分 —— 与形态模型不同, 这个【能】当选股信号用。

    题目与裸K 完全一致: 次日开盘买入, 10 根K线内先碰 +10% 记涨 / 先碰 -8% 记跌。
    特征只用 tushare cyq_perf 的获利盘族(前6名重要性合计 60.0%)。

    ⚠️ 与形态模型的关键差别 —— 它是【纯选股】:
       择时 vs 选股拆解显示它的择时贡献是 0(同一批交易日买全市场 +0.37%,
       等于所有交易日的 +0.37%), 99% 的交易日都在出信号。
       裸K 那个模型 85% 的交易挤在 10% 的日子里, 靠挑日子挣钱, 所以回撤 −31%;
       筹码这个仓位不堆在少数几天, 回撤只有 −17.9%。

    ⚠️ 实测(walk-forward + 资金池 + 真实周转, 仓位20):
       比值 0.89, 年化 +15.9%, 回撤 −17.9%, 六年 2 个负年(都在 2% 以内)。
       与裸K 共振后 1.07 —— 但共振要跑 CNN 推理, 生产机内存不够, 暂未上线。

    ⚠️ 筹码数据 2018-01 才有, 训练集只有 2018-2019 两年。实测训练样本量存在
       悬崖(231K 尚可 / 53K 崩到 −0.20), 而这个模型是 200,627, 已经贴着线 ——
       不要再缩训练窗口。
    """
    __tablename__ = "chips_score"
    ts_code = Column(String(12), primary_key=True)
    trade_date = Column(Date, primary_key=True)
    p_up = Column(Numeric(8, 5), nullable=False)      # P(10根K线内先碰+10%)
    p_dn = Column(Numeric(8, 5), nullable=False)      # P(先碰-8%)
    ev = Column(Numeric(10, 6), nullable=False)       # 0.10*p_up − 0.08*p_dn − 0.003
    rank_pct = Column(Numeric(8, 5), nullable=False)  # 当日横截面分位
    __table_args__ = (Index("ix_chips_date", "trade_date"),)


class ShapeScore(Base):
    """形态模型每日全市场打分。

    ⚠️ 这个模型【不能】当选股信号用 —— 带成交约束的组合比值只有 0.24。
       它的价值在空头端: Bot20%(rank_pct<0.2) 六年一致跑输, 拿来做
       【排除过滤器】。叠在周线版上胜率 45.8%->49.7%。
    """
    __tablename__ = "shape_score"
    ts_code = Column(String(12), primary_key=True)
    trade_date = Column(Date, primary_key=True)
    score = Column(Numeric(12, 6), nullable=False)
    rank_pct = Column(Numeric(8, 5), nullable=False)
    __table_args__ = (Index("ix_shape_date", "trade_date"),)
