import type {
  CandleResponse,
  FavoriteItem,
  MinuteData,
  SearchHistoryItem,
  Snapshot,
  StockInfo,
  Timeframe,
  T0Response,
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

/** 日内做 T 信号 — 分时图上标出买卖点。阈值越高信号越少越准。 */
export async function getT0Signals(
  symbol: string,
  threshold = 0.6
): Promise<T0Response> {
  return json(`${BASE}/quotes/${symbol}/t0?threshold=${threshold}`);
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

/** 买卖很准 v3 —— 原始买点 + 模型评分。
 *  原指标买点与随机无异; 模型在同日×同波动层中性化后训练,
 *  样本外把 Top20% 胜率 48.7%→51.3%、中位 -0.132%→+0.264%。 */
export async function getMaimaiSignals(
  tsCode: string,
  minGrade: "弱" | "中" | "强" = "弱",
  start?: string
): Promise<{
  ts_code: string;
  count: number;
  signals: { date: string; score: number; rank_pct: number; grade: string }[];
}> {
  const q = new URLSearchParams({ min_grade: minGrade });
  if (start) q.set("start", start);
  return json(`${BASE}/indicators/maimai/${tsCode}?${q}`);
}

/** 主力吸筹 —— 预测未来10日内出现拉升的概率。
 *  核心是筹码分布的「获利盘背离」(价格没动但获利盘上升 = 低位接货),
 *  样本外 Q10/Q1=4.81, 七年逐年 1.70~2.02 倍无失效。 */
export async function getPumpSignals(
  tsCode: string,
  minGrade: "中" | "强" = "中",
  start?: string
): Promise<{
  ts_code: string;
  count: number;
  signals: { date: string; prob: number; rank_pct: number; grade: string }[];
}> {
  const q = new URLSearchParams({ min_grade: minGrade });
  if (start) q.set("start", start);
  return json(`${BASE}/indicators/pump/${tsCode}?${q}`);
}

/** 低点组合 v2 —— 阶段底部 + 模型过滤。
 *  三个训练指标里最强: Top10% 胜率 57.7%(原 53.8%), 按天 t=3.88。 */
export async function getDidianSignals(
  tsCode: string,
  minGrade: "中" | "强" = "中",
  start?: string
): Promise<{
  ts_code: string;
  count: number;
  signals: { date: string; score: number; rank_pct: number; grade: string }[];
}> {
  const q = new URLSearchParams({ min_grade: minGrade });
  if (start) q.set("start", start);
  return json(`${BASE}/indicators/didian/${tsCode}?${q}`);
}

/** 买卖很准 v4 —— v3强档 与 主力吸筹强档 ±3日内共振。
 *  胜率 62.5%(同日随机 53.2%, z=22.1), 八年全部>54%。 */
export async function getComboSignals(
  tsCode: string,
  start?: string
): Promise<{
  ts_code: string;
  count: number;
  signals: { date: string; score: number; rank_pct: number; grade: string }[];
}> {
  const q = start ? `?start=${start}` : "";
  return json(`${BASE}/indicators/combo/${tsCode}${q}`);
}

