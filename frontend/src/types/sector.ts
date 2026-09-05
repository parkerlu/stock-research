export interface SectorItem {
  ts_code: string;
  name: string;
  count: number;
  list_date: string | null;
  quoted: number;      // 有实时行情的成分数
  up: number;
  down: number;
  flat: number;
  up_ratio: number | null;   // 上涨家数占比 %
  avg_pct: number | null;    // 成分股涨跌幅等权平均 %
  median_pct: number | null;
  max_pct: number | null;
  min_pct: number | null;
}

export interface SectorMember {
  ts_code: string;
  name: string;
  price: number | null;
  pct: number | null;
  amount: number | null;
  vol: number | null;
  turnover: number | null;
}

export interface StockSector {
  ts_code: string;
  name: string;
  count: number | null;
  list_date: string | null;
  is_theme: boolean;
}
