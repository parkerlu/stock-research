const BASE = "/api/system";

export interface DataStatus {
  market_latest: string | null;
  totals: {
    total_stocks: number;
    up_to_date: number;
    stale_7d: number;
    stale_30d: number;
    inactive: number;
  };
  stale: Array<{
    ts_code: string;
    name: string | null;
    latest_date: string;
    bars: number;
    days_stale: number;
    active: boolean;
  }>;
}

export interface BackfillJobState {
  job_id: string;
  status: "pending" | "running" | "completed" | "cancelled" | "error";
  progress: number;
  total: number;
  rows_inserted: number;
  elapsed_sec: number;
  error: string | null;
  failures: Array<{ ts_code: string; error: string }>;
}

export async function getDataStatus(): Promise<DataStatus> {
  const r = await fetch(`${BASE}/data-status`);
  if (!r.ok) throw new Error(`getDataStatus ${r.status}`);
  return r.json();
}

export async function startBackfill(opts?: {
  include_inactive?: boolean;
  only_stale?: boolean;
}): Promise<{ job_id: string }> {
  const r = await fetch(`${BASE}/backfill/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      include_inactive: opts?.include_inactive ?? false,
      only_stale: opts?.only_stale ?? true,
    }),
  });
  if (!r.ok) throw new Error(`startBackfill ${r.status}`);
  return r.json();
}

export async function getBackfillStatus(jobId: string): Promise<BackfillJobState> {
  const r = await fetch(`${BASE}/backfill/${jobId}`);
  if (!r.ok) throw new Error(`getBackfillStatus ${r.status}`);
  return r.json();
}

export async function cancelBackfill(jobId: string): Promise<void> {
  const r = await fetch(`${BASE}/backfill/${jobId}/cancel`, { method: "POST" });
  if (!r.ok) throw new Error(`cancelBackfill ${r.status}`);
}

export interface SymbolSyncResult {
  ts_code: string;
  inserted: number;
  latest: string | null;
  /** 非 null 表示没打数据源: "cooldown" 冷却期内, "up-to-date" 本地已到今天 */
  skipped: "cooldown" | "up-to-date" | null;
}

/** 补齐单只标的的日线到今天 — 实时页切股时调用, 周/月K 由日线聚合故一并补齐。 */
export async function syncSymbol(
  tsCode: string,
  force = false
): Promise<SymbolSyncResult> {
  const r = await fetch(
    `${BASE}/backfill/symbol/${tsCode}${force ? "?force=true" : ""}`,
    { method: "POST" }
  );
  if (!r.ok) throw new Error(`syncSymbol ${r.status}`);
  return r.json();
}
