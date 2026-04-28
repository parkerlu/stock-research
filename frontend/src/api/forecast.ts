const BASE = "/api/forecast";

export interface ForecastDay {
  day: number;
  close: number;
  low: number;
  high: number;
  log_return: number;
}

export interface ForecastResult {
  ts_code: string;
  anchor_close: number;
  anchor_date: string;
  forecast: ForecastDay[];
}

export async function getForecast(tsCode: string): Promise<ForecastResult> {
  const res = await fetch(`${BASE}/${encodeURIComponent(tsCode)}`);
  if (!res.ok) throw new Error(`forecast ${res.status}`);
  return res.json();
}
