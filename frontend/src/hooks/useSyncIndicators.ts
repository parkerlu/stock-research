// 把一张图表同步到给定的指标选择 —— 多周期联动视图用。
//
// 与 useTdxIndicators 的分工: 那个 hook 自己持有"选了哪些指标"(受工具栏驱动),
// 这个 hook 不持有选择, 只负责把外面传进来的选择应用到自己这张图上, 并且用
// 自己的周期取数。三张联动图各挂一个, 就得到"每个周期都显示指标, 且是该周期
// 自己算出来的指标"。
import { useEffect, useRef } from "react";
import type { Chart } from "klinecharts";
import { getIndicator } from "../api/indicators";
import type { Timeframe } from "../types/quote";
import { MAIN_PANE_INDICATORS } from "../components/ChartArea/indicatorPanes";
import {
  getKlineIndicatorName,
  setIndicatorData,
} from "../components/ChartArea/TdxIndicatorManager";

interface Options {
  /** 惰性取图表实例 —— 图表可能晚于 hook 初始化 */
  getChart: () => Chart | null;
  symbol: string;
  timeframe: Timeframe;
  /** klinecharts 内置指标 (MA/MACD/KDJ…) */
  std: string[];
  /** TDX 自定义指标 */
  tdx: string[];
}

/** 图表实例可能还没就绪, 轮询几次再放弃 (与 useTdxIndicators 同一策略)。 */
function whenChartReady(
  getChart: () => Chart | null,
  isCancelled: () => boolean,
  fn: (chart: Chart) => void
) {
  let tries = 0;
  const run = () => {
    if (isCancelled()) return;
    const chart = getChart();
    if (!chart) {
      if (tries++ < 20) setTimeout(run, 200);
      return;
    }
    fn(chart);
  };
  run();
}

export function useSyncIndicators({
  getChart,
  symbol,
  timeframe,
  std,
  tdx,
}: Options) {
  const appliedStd = useRef<Set<string>>(new Set());
  const appliedTdx = useRef<Set<string>>(new Set());

  // 内置指标: klinecharts 按本图自己的 K 线算, 天然就是本周期的口径
  useEffect(() => {
    let cancelled = false;
    whenChartReady(getChart, () => cancelled, (chart) => {
      const want = new Set(std);
      for (const name of [...appliedStd.current]) {
        if (want.has(name)) continue;
        chart.removeIndicator(
          MAIN_PANE_INDICATORS.has(name) ? { paneId: "candle_pane", name } : { name }
        );
        appliedStd.current.delete(name);
      }
      for (const name of want) {
        if (appliedStd.current.has(name)) continue;
        if (MAIN_PANE_INDICATORS.has(name)) {
          chart.createIndicator(name, true, { id: "candle_pane" });
        } else {
          chart.createIndicator(name, false, { id: `kc_${name}_pane` });
        }
        appliedStd.current.add(name);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [std, getChart]);

  // TDX 指标: 每张图按自己的周期各取一份, 注册名带周期所以互不覆盖
  useEffect(() => {
    if (!symbol) return;
    let cancelled = false;
    whenChartReady(getChart, () => cancelled, (chart) => {
      void (async () => {
        const want = new Set(tdx);
        for (const name of [...appliedTdx.current]) {
          if (want.has(name)) continue;
          chart.removeIndicator({ name: getKlineIndicatorName(name, timeframe) });
          appliedTdx.current.delete(name);
        }

        const today = new Date().toISOString().slice(0, 10);
        for (const name of want) {
          if (cancelled) return;
          try {
            const result = await getIndicator(
              name, symbol, timeframe, "1990-01-01", today
            );
            if (cancelled) return;
            if (result.warnings.length > 0) {
              // 周/月线的 bar 数天然比日线少一个数量级, "数据不足"是常态。
              // 三张图各弹一次窗没法用, 这里只记日志, 该周期不画就是了。
              console.warn(`[${timeframe}] ${name}: ${result.warnings.join("; ")}`);
              continue;
            }
            setIndicatorData(result, timeframe);
            const kcName = getKlineIndicatorName(name, timeframe);
            if (appliedTdx.current.has(name)) {
              chart.overrideIndicator({ name: kcName });
            } else {
              chart.createIndicator(kcName, true, {
                id: result.pane === "main" ? "candle_pane" : `tdx_${name}_pane`,
              });
              appliedTdx.current.add(name);
            }
          } catch (err) {
            console.error(`[${timeframe}] 加载 TDX 指标 ${name} 失败:`, err);
          }
        }
      })();
    });
    return () => {
      cancelled = true;
    };
  }, [tdx, symbol, timeframe, getChart]);
}
