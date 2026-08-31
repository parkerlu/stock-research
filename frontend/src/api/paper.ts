const BASE = "/api/paper";

export interface Holding {
  ts_code: string;
  name: string | null;
  open_date: string;
  open_price: number;
  shares: number;
  init_shares: number;
  last_price: number;
  as_of: string | null;
  market_value: number;
  float_pnl: number;
  float_pnl_pct: number;
  realized_pnl: number;
  stop_price: number;
  tier1_price: number;
  tier2_price: number;
  tier1_done: boolean;
  hold_days: number;
}

export interface ClosedPosition {
  ts_code: string;
  name: string | null;
  open_date: string;
  close_date: string;
  open_price: number;
  reason: string | null;
  realized_pnl: number;
}

export interface PaperStatus {
  exists: boolean;
  name: string;
  slots: number;
  initial_capital: number;
  cash: number;
  market_value: number;
  equity: number;
  total_pnl: number;
  total_pnl_pct: number;
  realized_pnl: number;
  float_pnl: number;
  started_on: string;
  last_run_date: string | null;
  n_open: number;
  n_closed: number;
  holdings: Holding[];
  closed: ClosedPosition[];
}

export interface PaperTrade {
  date: string;
  ts_code: string;
  name: string | null;
  action: string;
  price: number;
  shares: number;
  amount: number;
  fee: number;
  pnl: number | null;
  pnl_pct: number | null;
  note: string | null;
}

export interface PaperConfig {
  default_account: string;
  strategy: string;
  rules: Record<string, number | string | boolean>;
  backtest: {
    period: string; capital: string; cagr: number;
    max_drawdown: number; sharpe: number; trades: number; note: string;
  };
}

export interface EquityPoint { date: string; equity: number; cash: number; n: number }

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return r.json();
}

export const getPaperConfig = () => json<PaperConfig>(`${BASE}/config`);
export const getPaperStatus = () => json<PaperStatus>(`${BASE}/status`);
export const getPaperTrades = (limit = 200) =>
  json<{ exists: boolean; trades: PaperTrade[] }>(`${BASE}/trades?limit=${limit}`);
export const getPaperEquity = () =>
  json<{ exists: boolean; initial: number; points: EquityPoint[] }>(`${BASE}/equity`);
export const runPaper = () =>
  json<{ job_id: string }>(`${BASE}/run`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
  });
export const getPaperRun = (id: string) =>
  json<{ status: string; progress: number; total: number; error: string | null; elapsed_sec: number }>(
    `${BASE}/run/${id}`);
