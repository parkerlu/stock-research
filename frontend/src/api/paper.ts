const BASE = "/api/paper";

// 两个独立账户: 实操盘只能向前(定时任务推进), 演示盘可任意重置回放
export const LIVE_ACCOUNT = "live-2026";
export const DEMO_ACCOUNT = "demo";

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
  as_of: string;
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
export const getPaperStatus = (name = LIVE_ACCOUNT) =>
  json<PaperStatus>(`${BASE}/status?name=${name}`);
export const getPaperTrades = (name = LIVE_ACCOUNT, limit = 200) =>
  json<{ exists: boolean; trades: PaperTrade[] }>(`${BASE}/trades?name=${name}&limit=${limit}`);
export const getPaperEquity = (name = LIVE_ACCOUNT) =>
  json<{ exists: boolean; initial: number; points: EquityPoint[] }>(`${BASE}/equity?name=${name}`);

/** 演示盘: 推进 N 个交易日, 同步返回 —— 走预计算信号表, 一天约 0.5s */
export const stepPaper = (name: string, days = 1) =>
  json<{ advanced: number; as_of: string; done: boolean; message?: string }>(
    `${BASE}/step`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, days }),
    });

/** 演示盘: 清空并从指定日期重新开始 */
export const resetPaper = (name: string, start: string, capital = 100000, slots = 10) =>
  json<{ name: string; started_on: string }>(`${BASE}/reset`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, start, capital, slots }),
  });

export interface SignalPick {
  ts_code: string; name: string | null; signal_date: string; buy_date: string;
  next_open: number | null; amount_20d_wan: number; held: boolean;
}
export const getPaperSignals = (name: string, limit = 20) =>
  json<{ exists: boolean; as_of: string; picks: SignalPick[] }>(
    `${BASE}/signals?name=${name}&limit=${limit}`);
export const runPaper = (name = LIVE_ACCOUNT) =>
  json<{ job_id: string }>(`${BASE}/run`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
export const getPaperRun = (id: string) =>
  json<{ status: string; progress: number; total: number; error: string | null; elapsed_sec: number }>(
    `${BASE}/run/${id}`);
