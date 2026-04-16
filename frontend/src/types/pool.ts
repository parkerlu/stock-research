export interface Pool {
  id: number;
  name: string;
  description: string | null;
  stock_count: number;
  created_at: string;
}

export interface PoolStockSnapshot {
  ts_code: string;
  name: string;
  symbol: string;
  close: number | null;
  change_pct: number | null;
  volume: number | null;
  turnover_rate: number | null;
  trade_date: string | null;
}

export interface PoolDetail {
  id: number;
  name: string;
  description: string | null;
  stocks: PoolStockSnapshot[];
}
