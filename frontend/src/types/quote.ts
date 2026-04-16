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
