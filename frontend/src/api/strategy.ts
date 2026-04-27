import type {
  BacktestReport,
  FactoryJob,
  StrategyItem,
} from "../types/strategy";

const BASE = "/api";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function listStrategies(tsCode: string): Promise<StrategyItem[]> {
  return json(`${BASE}/strategies?ts_code=${encodeURIComponent(tsCode)}`);
}

export async function pinStrategy(strategyId: number): Promise<void> {
  await json(`${BASE}/strategies/${strategyId}/pin`, { method: "POST" });
}

export async function unpinStrategy(strategyId: number): Promise<void> {
  await json(`${BASE}/strategies/${strategyId}/pin`, { method: "DELETE" });
}

export async function runFactory(
  tsCode: string,
  cutoffDate: string,
  positionRatios: number[] = [40, 30, 30]
): Promise<{ job_id: string; status: string; total_candidates: number }> {
  return json(`${BASE}/strategy-factory/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ts_code: tsCode,
      cutoff_date: cutoffDate,
      position_ratios: positionRatios,
    }),
  });
}

export async function getFactoryJob(jobId: string): Promise<FactoryJob> {
  return json(`${BASE}/strategy-factory/jobs/${jobId}`);
}

export async function runBacktest(params: {
  ts_code: string;
  strategy_id?: number;
  start_date: string;
  end_date: string;
  initial_capital?: number;
  position_ratios?: number[];
}): Promise<{ id: string; status: string }> {
  return json(`${BASE}/backtests/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function getBacktestReport(runId: string): Promise<BacktestReport> {
  return json(`${BASE}/backtests/${runId}/report`);
}

export interface TemplateInfo {
  template_id: string;
  name: string;
}

export async function listTemplates(): Promise<TemplateInfo[]> {
  return json(`${BASE}/backtests/templates`);
}

export interface TryTemplateResult {
  ts_code: string;
  template_id: string;
  start_date: string;
  end_date: string;
  metrics: {
    net_profit_pct: number;
    annualized_return: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
    profit_factor: number;
    final_capital: number;
  };
  trades: Array<{
    entry_date: string;
    exit_date: string;
    entry_price: number;
    exit_price: number;
    shares: number;
    pnl: number;
  }>;
  actions: Array<{
    date: string;
    type: "buy" | "sell";
    price: number;
    shares: number;
    amount: number;
    position_level: number;
    pnl?: number;
    pnl_pct?: number;
  }>;
}

export async function tryTemplate(params: {
  ts_code: string;
  template_id: string;
  params?: Record<string, number>;
  start_date?: string;
  end_date?: string;
}): Promise<TryTemplateResult> {
  return json(`${BASE}/backtests/try`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}
