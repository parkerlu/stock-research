export interface StockInfo {
  ts_code: string;
  symbol: string;
  name: string;
  industry: string | null;
}

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  amount: number;
}

export interface CandleResponse {
  symbol: string;
  tf: string;
  candles: Candle[];
}

export interface Snapshot {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_pct: number;
  open: number;
  high: number;
  low: number;
  vol: number;
  amount: number;
  turnover: number;
  // 腾讯源附加字段 (tushare/akshare 回落时可能缺失)
  prev_close?: number;
  high_limit?: number;
  low_limit?: number;
  amplitude?: number;
  pe?: number;
  pb?: number;
  total_cap?: number;
  float_cap?: number;
  quote_time?: string | null;
  bids?: OrderLevel[];  // 买一..买五
  asks?: OrderLevel[];  // 卖一..卖五
  source?: string;
}

/** 盘口一档。vol 单位为手。 */
export interface OrderLevel {
  price: number;
  vol: number;
}

/** 分时图的一分钟数据点。vol/amount 已由后端差分成"当分钟"增量。 */
export interface MinuteBar {
  time: string;       // "0930"
  price: number;
  avg_price: number;  // 均价 = 累计成交额 / 累计股数
  vol: number;        // 手
  amount: number;     // 元
}

export interface MinuteData {
  symbol: string;
  trade_date: string;    // "20260828"
  prev_close: number;    // 分时图的基准线
  market_open: boolean;  // 上证是否在交易 — 决定要不要继续轮询
  snapshot: Snapshot | null;
  bars: MinuteBar[];
}

/** 一次完整的日内做 T —— 入场 + 接回配对。
 *  side='sell': 用底仓高抛, 跌到 target 或收盘前接回。
 *  side='buy' : 低吸买入, 涨到 target 或收盘前卖出等量底仓。 */
export interface MaimaiSignal {
  date: string;
  score: number;
  rank_pct: number;
  grade: string;
  /** buy = 超卖反转买点(画在下方红三角); sell = 强势段被打破(上方绿三角) */
  side?: "buy" | "sell";
}

export interface T0Trade {
  side: "sell" | "buy";
  prob: number;
  entry_time: string;
  entry_bar: number;
  entry_price: number;
  target: number;
  exit_time: string;
  exit_bar: number;
  exit_price: number;
  exit_reason: string; // 达标 / 收盘平 / 持有中
  ret: number;
  /** 持仓期最大浮亏 (≤0) —— 只看 ret 会低估这次 T 的煎熬程度 */
  mae: number;
  worst_price: number | null;
}

export interface T0Response {
  symbol: string;
  trade_date?: string;
  threshold: number;
  signals: T0Trade[];
}

export interface SearchHistoryItem {
  ts_code: string;
  name: string;
  searched_at: string;
}

export interface FavoriteItem {
  ts_code: string;
  name: string;
}

export type Timeframe = "1d" | "1w" | "1m";
