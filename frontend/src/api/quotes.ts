import type {
  CandleResponse,
  FavoriteItem,
  MinuteData,
  SearchHistoryItem,
  Snapshot,
  StockInfo,
  Timeframe,
} from "../types/quote";

const BASE = "/api";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function searchStocks(
  q: string,
  limit = 10
): Promise<StockInfo[]> {
  return json(`${BASE}/quotes/search?q=${encodeURIComponent(q)}&limit=${limit}`);
}

export async function getCandles(
  symbol: string,
  tf: Timeframe = "1d",
  from?: string,
  to?: string
): Promise<CandleResponse> {
  const params = new URLSearchParams({ tf });
  if (from) params.set("from", from);
  if (to) params.set("to", to);
  return json(`${BASE}/quotes/${symbol}/candles?${params}`);
}

export async function getSnapshot(symbol: string): Promise<Snapshot> {
  return json(`${BASE}/quotes/${symbol}/snapshot`);
}

/** 当日分时 (腾讯) — 含实时快照与上证开闭市状态。 */
export async function getMinute(symbol: string): Promise<MinuteData> {
  return json(`${BASE}/quotes/${symbol}/minute`);
}

/** 批量实时快照 — 列表页一次拿全部。 */
export async function getSnapshots(
  codes: string[]
): Promise<Record<string, Snapshot>> {
  if (codes.length === 0) return {};
  return json(`${BASE}/quotes/snapshots?codes=${encodeURIComponent(codes.join(","))}`);
}

export async function getSearchHistory(
  limit = 20
): Promise<SearchHistoryItem[]> {
  return json(`${BASE}/search-history?limit=${limit}`);
}

export async function addFavorite(tsCode: string): Promise<void> {
  await json(`${BASE}/favorites/${tsCode}`, { method: "POST" });
}

export async function removeFavorite(tsCode: string): Promise<void> {
  await json(`${BASE}/favorites/${tsCode}`, { method: "DELETE" });
}

export async function getFavorites(): Promise<FavoriteItem[]> {
  return json(`${BASE}/favorites`);
}
