import type { SectorItem, SectorMember, StockSector } from "../types/sector";

const BASE = "/api";

async function json<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

/** 板块列表 + 今日热度 (上涨占比 / 平均涨幅)。 */
export async function getSectors(themeOnly = true): Promise<{
  sectors: SectorItem[];
  trade_date: string | null;
  /** 成分股行情覆盖率 % —— 低于 80 说明当日K线还没入库完, 均值不可信 */
  coverage: number | null;
}> {
  return json(`${BASE}/sectors?theme_only=${themeOnly}`);
}

/** 板块成分股 + 各自实时行情, 按涨幅排序。 */
export async function getSectorMembers(code: string): Promise<{
  sector: { ts_code: string; name: string; count: number; list_date: string | null };
  members: SectorMember[];
}> {
  return json(`${BASE}/sectors/${code}/members`);
}

/** 个股所属的全部板块 (一对多)。 */
export async function getStockSectors(tsCode: string): Promise<{
  ts_code: string;
  sectors: StockSector[];
}> {
  return json(`${BASE}/stocks/${tsCode}/sectors`);
}

export interface SectorHistPoint {
  date: string;
  n: number;
  up: number;
  up_ratio: number | null;
  avg_pct: number | null;
  cum_pct: number;
}

/** 板块近 N 日的平均涨幅 / 上涨占比曲线 */
export async function getSectorHistory(
  code: string,
  days = 30
): Promise<{ sector_code: string; days: number; items: SectorHistPoint[] }> {
  return json(`${BASE}/sectors/${code}/history?days=${days}`);
}

export interface StockSectorHot {
  ts_code: string;
  name: string;
  count: number | null;
  list_date: string | null;
  quoted: number;
  up: number;
  up_ratio: number | null;
  avg_pct: number | null;
  is_theme: boolean;
}

/** 个股所属板块 + 各板块当日热度(按涨幅降序) */
export async function getStockSectorsHot(
  tsCode: string
): Promise<{ ts_code: string; sectors: StockSectorHot[] }> {
  return json(`${BASE}/stocks/${tsCode}/sectors`);
}
