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
  /** 该账户不设止损时(stop_pct>=0.5)为 true —— 此时 stop_price 是买价的 1%,
   *  打不到, 直接显示横杠而不是印一个吓人的 0.29 */
  no_stop?: boolean;
  /** null = 该账户不设这一档止盈(如周线版只按持有期出场) */
  tier1_price: number | null;
  tier2_price: number | null;
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
/** 演示盘可选的策略 + 各自信号起点 —— 起点决定能回放到哪年 */
export interface DemoStrategy { key: string; label: string; since: string }
export const getDemoStrategies = () =>
  json<{ min_date: string; items: DemoStrategy[] }>(`${BASE}/strategies`,
                                                    { method: "GET" });

/** ⚠️ strategy 必须传: 不传的话账户继承 DEFAULT_CONFIG 的 strategy=None,
 *  点"下一日"日期会走但一笔都不买, 而且不报错。 */
export const resetPaper = (name: string, start: string, strategy: string,
                           capital = 100000, slots = 10) =>
  json<{ name: string; started_on: string; strategy: string }>(`${BASE}/reset`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, start, capital, slots, strategy }),
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


export interface PaperAccountRow {
  id: number;
  name: string;
  strategy: string;
  started_on: string | null;
  last_run_date: string | null;
  slots: number;
  trades: number;
  pnl_pct: number;
}

/** 在跑的账户列表。⚠️ 后端只返回 is_active 的 —— 被证伪的策略已停用,
 *  不该再出现在选择器里让人误以为还在参考。 */
export const getPaperAccounts = () =>
  json<{ accounts: PaperAccountRow[] }>(`${BASE}/accounts`);
