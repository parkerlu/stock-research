const BASE = "/api";

export interface KBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface KronosPrediction {
  ts_code: string;
  model: string;
  anchor_date: string;
  anchor_close: number;
  lookback: number;
  pred_len: number;
  samples: number;
  predicted: KBar[];
  /** 历史起点时非空 —— 同期真实走势, 用来对照 */
  actual: KBar[];
  is_historical: boolean;
}

/** Kronos 预测某日之后的 K 线。
 *  ⚠️ 模型是自回归采样的, 有随机性。samples>1 取多条路径均值;
 *     单条只能看方向, 不要当精确预测。 */
export async function predictKronos(params: {
  tsCode: string;
  end: string;
  predLen?: number;
  lookback?: number;
  samples?: number;
  model?: "Kronos-mini" | "Kronos-small";
}): Promise<KronosPrediction> {
  const q = new URLSearchParams({
    end: params.end,
    pred_len: String(params.predLen ?? 30),
    lookback: String(params.lookback ?? 400),
    samples: String(params.samples ?? 1),
    model: params.model ?? "Kronos-mini",
  });
  const r = await fetch(`${BASE}/kronos/predict/${params.tsCode}?${q}`);
  if (!r.ok) throw new Error((await r.json()).detail ?? "预测失败");
  return r.json();
}
