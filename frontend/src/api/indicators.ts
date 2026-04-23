import type { IndicatorMeta, IndicatorResult } from "../types/indicator";
import type { Timeframe } from "../types/quote";

const BASE = "/api/indicators";

async function json<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function listIndicators(): Promise<IndicatorMeta[]> {
  return json(BASE);
}

export async function getIndicator(
  name: string,
  tsCode: string,
  tf: Timeframe = "1d",
  from?: string,
  to?: string
): Promise<IndicatorResult> {
  const params = new URLSearchParams({ ts_code: tsCode, tf });
  if (from) params.set("from", from);
  if (to) params.set("to", to);
  return json(`${BASE}/${name}?${params}`);
}
