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
