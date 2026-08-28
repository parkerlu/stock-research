// TDX 指标挂载 hook — 把 ChartArea 里那套"取数 → 注册 → 建 pane → 换股刷新"
// 的逻辑抽出来, 供多个图表实例复用 (ChartArea 主图 / 实时页日K)。
//
// 每个使用处传各自的 storageKey, 这样两张图的指标选择互不干扰。
import { useCallback, useEffect, useRef, useState } from "react";
import type { Chart } from "klinecharts";
import { getIndicator, listIndicators } from "../api/indicators";
import type { IndicatorMeta } from "../types/indicator";
import {
  getKlineIndicatorName,
  setIndicatorData,
} from "../components/ChartArea/TdxIndicatorManager";

interface Options {
  /** 惰性取图表实例 —— 图表可能晚于 hook 初始化 */
  getChart: () => Chart | null;
  symbol: string;
  timeframe?: "1d" | "1w" | "1m";
  storageKey: string;
}

function readLS(key: string): string[] {
  try {
    const raw = localStorage.getItem(key);
    const parsed = raw ? JSON.parse(raw) : null;
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

export function useTdxIndicators({
  getChart,
  symbol,
  timeframe = "1d",
  storageKey,
}: Options) {
  const [available, setAvailable] = useState<IndicatorMeta[]>([]);
  const [active, setActive] = useState<string[]>(() => readLS(storageKey));
  const [busy, setBusy] = useState<string | null>(null);
  const paneIds = useRef<Record<string, string>>({});

  useEffect(() => {
    listIndicators().then(setAvailable).catch(() => setAvailable([]));
  }, []);

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(active));
  }, [active, storageKey]);

  /** 取数并在图上建/更新一个指标。create=false 时只刷新数据。 */
  const apply = useCallback(
    async (name: string, create: boolean) => {
      const chart = getChart();
      if (!chart || !symbol) return false;
      const today = new Date().toISOString().slice(0, 10);
      const result = await getIndicator(name, symbol, timeframe, "1990-01-01", today);
      if (result.warnings.length > 0) {
        alert(result.warnings.join("\n"));
        return false;
      }
      setIndicatorData(result);
      const kcName = getKlineIndicatorName(name);
      if (create) {
        const paneId = result.pane === "main" ? "candle_pane" : `tdx_${name}_pane`;
        chart.createIndicator(kcName, true, { id: paneId });
        paneIds.current[name] = paneId;
      } else {
        chart.overrideIndicator({ name: kcName });
      }
      return true;
    },
    [getChart, symbol, timeframe]
  );

  const toggle = useCallback(
    async (name: string) => {
      const chart = getChart();
      if (!chart || !symbol) return;
      if (active.includes(name)) {
        chart.removeIndicator({ name: getKlineIndicatorName(name) });
        delete paneIds.current[name];
        setActive((p) => p.filter((n) => n !== name));
        return;
      }
      setBusy(name);
      try {
        if (await apply(name, true)) setActive((p) => [...p, name]);
      } catch (err) {
        console.error("加载 TDX 指标失败:", err);
        alert("加载 TDX 指标失败");
      } finally {
        setBusy(null);
      }
    },
    [active, apply, getChart, symbol]
  );

  // 换股 / 换周期后重建已选指标。图表实例可能还没就绪, 轮询几次再放弃。
  useEffect(() => {
    if (!symbol || active.length === 0) return;
    let cancelled = false;
    let tries = 0;
    const run = async () => {
      if (cancelled) return;
      if (!getChart()) {
        if (tries++ < 20) setTimeout(run, 200);
        return;
      }
      for (const name of active) {
        if (cancelled) return;
        try {
          // 换股后 pane 已被图表重建流程清掉, 这里统一按"创建"处理
          await apply(name, !paneIds.current[name]);
        } catch (err) {
          console.error(`刷新 TDX 指标 ${name} 失败:`, err);
        }
      }
    };
    run();
    return () => {
      cancelled = true;
    };
    // active 变化由 toggle 自己处理, 这里只跟随 symbol/timeframe
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, timeframe]);

  return { available, active, toggle, busy };
}
