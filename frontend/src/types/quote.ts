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
