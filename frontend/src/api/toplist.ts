const BASE = "/api";
async function json<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export interface TopDay {
  date: string; count: number; net_wan: number; buy_count: number;
}
export interface TopItem {
  ts_code: string; name: string;
  close: number | null; pct_change: number | null; turnover_rate: number | null;
  buy_wan: number; sell_wan: number; net_wan: number; reason: string;
}

export const getTopDays = (limit = 30) =>
  json<{ days: TopDay[] }>(`${BASE}/toplist/days?limit=${limit}`);

export const getTopList = (day?: string, side: "all" | "buy" | "sell" = "all") =>
  json<{ date: string; count: number; items: TopItem[] }>(
    `${BASE}/toplist/list?side=${side}${day ? `&day=${day}` : ""}`
  );

export const getStockTopList = (tsCode: string) =>
  json<{ ts_code: string; count: number; items: any[] }>(
    `${BASE}/toplist/stock/${tsCode}`
  );
