const BASE = "/api";

async function json<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export interface TrainedMeta {
  key: string;
  label: string;
  desc: string;
  grades: string[];
  default_grade: string;
  /** 该指标建议的回看天数。信号极稀疏的(如 v5)需要长窗口, 否则常年空列表。 */
  default_days?: number;
}

export interface ScreenItem {
  ts_code: string;
  name: string;
  date: string;
  score: number;
  rank_pct: number;
  grade: string;
  price: number | null;
  chg: number | null;
}

/** 可用于选股的训练指标清单 */
export async function listTrained(): Promise<{ indicators: TrainedMeta[] }> {
  return json(`${BASE}/indicators/trained/list`);
}

/** 按训练指标选股 —— 信号盘后已算好入库, 直接查, 无需扫描任务 */
export async function screenByTrained(
  indicator: string,
  grade: string,
  days: number
): Promise<{ indicator: string; grade: string; days: number; count: number; items: ScreenItem[] }> {
  const q = new URLSearchParams({ indicator, grade, days: String(days) });
  return json(`${BASE}/indicators/trained/screen?${q}`);
}
