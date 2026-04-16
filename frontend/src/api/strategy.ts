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
