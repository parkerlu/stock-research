const BASE = "/api";

export interface StockHit {
  ts_code: string;
  name: string | null;
  signal_date: string;
  latest_date: string;
  latest_close: number;
}

export interface JobState {
  job_id: string;
  status: "pending" | "running" | "completed" | "cancelled" | "error";
  progress: number;
  total: number;
  hits: StockHit[];
  error: string | null;
  elapsed_sec: number;
}

export async function startScan(params: {
  template_id: string;
  lookback_days?: number;
}): Promise<{ job_id: string }> {
  const res = await fetch(`${BASE}/screening/scan/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`startScan ${res.status}`);
  return res.json();
}

export async function getScanStatus(jobId: string): Promise<JobState> {
  const res = await fetch(`${BASE}/screening/scan/${jobId}`);
  if (!res.ok) throw new Error(`getScanStatus ${res.status}`);
  return res.json();
}

export async function cancelScan(jobId: string): Promise<void> {
  const res = await fetch(`${BASE}/screening/scan/${jobId}/cancel`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`cancelScan ${res.status}`);
}

// ===== 组合信号 — 回测验证过的三源组合 =====

export interface PortfolioPick {
  rank: number;
  ts_code: string;
  name: string | null;
  signal_date: string;
  latest_date: string;
  latest_close: number;
  amount_20d_wan: number;
  strategies: string;
}

export interface PortfolioJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "error";
  progress: number;
  total: number;
  as_of: string | null;
  scanned: number;
  picks: PortfolioPick[];
  elapsed_sec: number;
  error: string | null;
}

export interface PortfolioConfig {
  strategies: { id: string; label: string }[];
  slots: number;
  rank_rule: string;
  min_amount_wan: number;
  backtest: {
    period: string;
    cagr: number;
    max_drawdown: number;
    longest_drawdown_days: number;
    final: string;
    benchmark: string;
    capacity_wan: number;
  };
}

export async function getPortfolioConfig(): Promise<PortfolioConfig> {
  const r = await fetch(`${BASE}/screening/portfolio/config`);
  if (!r.ok) throw new Error(`getPortfolioConfig ${r.status}`);
  return r.json();
}

export async function startPortfolioScan(
  lookbackDays = 5,
  topN = 20
): Promise<{ job_id: string }> {
  const r = await fetch(`${BASE}/screening/portfolio/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lookback_days: lookbackDays, top_n: topN }),
  });
  if (!r.ok) throw new Error(`startPortfolioScan ${r.status}`);
  return r.json();
}

export async function getPortfolioStatus(jobId: string): Promise<PortfolioJob> {
  const r = await fetch(`${BASE}/screening/portfolio/${jobId}`);
  if (!r.ok) throw new Error(`getPortfolioStatus ${r.status}`);
  return r.json();
}
