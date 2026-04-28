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

export async function getForecast(
  tsCode: string,
  asOf?: string,
): Promise<ForecastResult> {
  const url = asOf
    ? `${BASE}/${encodeURIComponent(tsCode)}?as_of=${asOf}`
    : `${BASE}/${encodeURIComponent(tsCode)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`forecast ${res.status}`);
  return res.json();
}
