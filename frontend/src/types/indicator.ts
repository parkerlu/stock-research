export interface IndicatorLine {
  name: string;
  values: (number | null)[];
  color: string;
  thickness: number;
  /** "bar" renders as solid columns from 0 (TDX 柱状副图); defaults to "line". */
  render?: "line" | "bar";
}

export interface IndicatorHLine {
  name: string;
  value: number;
  color: string;
  dashed: boolean;
}

export interface IndicatorBand {
  timestamps: number[];
  y1: number;
  y2: number;
  color: string;
  opacity: number;
}

export interface IndicatorMarker {
  timestamp: number;
  value: number;
  color: string;
  icon: "dot" | "triangle_up" | "triangle_down";
}

export interface IndicatorResult {
  name: string;
  label: string;
  pane: "main" | "sub";
  y_axis_range: [number, number] | null;
  timestamps: number[];
  lines: IndicatorLine[];
  hlines: IndicatorHLine[];
  bands: IndicatorBand[];
  markers: IndicatorMarker[];
  warnings: string[];
}

export interface IndicatorMeta {
  name: string;
  label: string;
  pane: "main" | "sub";
  min_bars: number;
}
