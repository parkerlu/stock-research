import { useCallback, useEffect, useRef, useState } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import type { TradeAction } from "../../types/strategy";
import { getIndicator } from "../../api/indicators";
import { MainChart } from "./MainChart";
import type { MainChartHandle } from "./MainChart";
import { Toolbar } from "./Toolbar";
import { LinkedView } from "./LinkedView";
import { getKlineIndicatorName, setIndicatorData } from "./TdxIndicatorManager";
import { getForecast } from "../../api/forecast";
import type { ForecastResult } from "./MainChart";

interface Props {
  tradeActions?: TradeAction[] | null;
}

const MAIN_PANE_INDICATORS = new Set(["MA", "EMA", "BOLL", "SAR"]);
const LS_STD_KEY = "chart.activeIndicators.v1";
const LS_TDX_KEY = "chart.activeTdxIndicators.v1";

function readLS(key: string, fallback: string[]): string[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : fallback;
  } catch {
    return fallback;
  }
}

export function ChartArea({ tradeActions }: Props = {}) {
  const linkedMode = useQuoteStore((s) => s.linkedMode);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const timeframe = useQuoteStore((s) => s.timeframe);
  const [activeIndicators, setActiveIndicators] = useState<string[]>(() =>
    readLS(LS_STD_KEY, ["MA"])
  );
  const [activeTdxIndicators, setActiveTdxIndicators] = useState<string[]>(() =>
    readLS(LS_TDX_KEY, [])
  );
  const mainChartRef = useRef<MainChartHandle>(null);
  const tdxPaneIds = useRef<Record<string, string>>({});
  const [tradeIdx, setTradeIdx] = useState(-1);
  const [forecast, setForecast] = useState<ForecastResult | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);

  const indicatorPaneIds = useRef<Record<string, string>>({});

  // Persist whenever state changes
  useEffect(() => {
    localStorage.setItem(LS_STD_KEY, JSON.stringify(activeIndicators));
  }, [activeIndicators]);
  useEffect(() => {
    localStorage.setItem(LS_TDX_KEY, JSON.stringify(activeTdxIndicators));
  }, [activeTdxIndicators]);

  useEffect(() => {
    setTradeIdx(-1);
  }, [tradeActions]);

  // Apply persisted standard indicators once the chart is initialized.
  useEffect(() => {
    const t = setTimeout(() => {
      const chart = mainChartRef.current?.getChart();
      if (!chart) return;
      for (const name of activeIndicators) {
        const isMain = MAIN_PANE_INDICATORS.has(name);
        if (isMain) {
          chart.createIndicator(name, true, { id: "candle_pane" });
        } else {
          const paneId = `kc_${name}_pane`;
          chart.createIndicator(name, false, { id: paneId });
          indicatorPaneIds.current[name] = paneId;
        }
      }
    }, 100);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Apply persisted TDX indicators once chart + symbol are ready.
  // Runs only on first symbol load (the existing refresh effect re-applies
  // them on subsequent symbol/timeframe changes).
  const tdxBootstrapped = useRef(false);
  useEffect(() => {
    if (tdxBootstrapped.current) return;
    if (!currentSymbol || activeTdxIndicators.length === 0) return;
    const chart = mainChartRef.current?.getChart();
    if (!chart) return;
    tdxBootstrapped.current = true;
    (async () => {
      const today = new Date().toISOString().slice(0, 10);
      for (const name of activeTdxIndicators) {
        try {
          const result = await getIndicator(
            name, currentSymbol, timeframe, "1990-01-01", today
          );
          setIndicatorData(result);
          const kcName = getKlineIndicatorName(name);
          const paneId =
            result.pane === "main" ? "candle_pane" : `tdx_${name}_pane`;
          chart.createIndicator(kcName, true, { id: paneId });
          tdxPaneIds.current[name] = paneId;
        } catch (err) {
          console.error(`恢复 TDX 指标 ${name} 失败:`, err);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSymbol]);

  const goToTrade = useCallback(
    (idx: number) => {
      if (!tradeActions || idx < 0 || idx >= tradeActions.length) return;
      setTradeIdx(idx);
      const action = tradeActions[idx];
      const ts = new Date(action.date + "T00:00:00").getTime();
      mainChartRef.current?.scrollToTimestamp(ts);
      // Auto-fetch forecast as of the trade date (only on buy actions to avoid double-fetch)
      if (currentSymbol && action.type === "buy") {
        getForecast(currentSymbol, action.date)
          .then((f) => setForecast(f))
          .catch(() => {/* swallow — forecast may not be available pre-2019 */});
      }
    },
    [tradeActions, currentSymbol]
  );

  const handleToggleIndicator = useCallback(
    (name: string, isMainPane: boolean) => {
      const chart = mainChartRef.current?.getChart();
      setActiveIndicators((prev) => {
        if (prev.includes(name)) {
          if (chart) {
            if (isMainPane) {
              chart.removeIndicator({ paneId: "candle_pane", name });
            } else {
              chart.removeIndicator({ name });
              delete indicatorPaneIds.current[name];
            }
          }
          return prev.filter((n) => n !== name);
        } else {
          if (chart) {
            if (isMainPane) {
              // isStack=true → overlay onto candle pane without wiping it
              chart.createIndicator(name, true, { id: "candle_pane" });
            } else {
              const paneId = `kc_${name}_pane`;
              chart.createIndicator(name, false, { id: paneId });
              indicatorPaneIds.current[name] = paneId;
            }
          }
          return [...prev, name];
        }
      });
    },
    []
  );

  const handleToggleTdxIndicator = useCallback(
    async (name: string) => {
      const chart = mainChartRef.current?.getChart();
      if (!chart || !currentSymbol) return;

      const kcName = getKlineIndicatorName(name);

      if (activeTdxIndicators.includes(name)) {
        // Remove by name — klinecharts will drop the pane when empty
        chart.removeIndicator({ name: kcName });
        delete tdxPaneIds.current[name];
        setActiveTdxIndicators((prev) => prev.filter((n) => n !== name));
        return;
      }

      // Add: fetch, register, create
      try {
        const result = await getIndicator(
          name,
          currentSymbol,
          timeframe,
          "1990-01-01",
          new Date().toISOString().slice(0, 10)
        );

        if (result.warnings.length > 0) {
          alert(result.warnings.join("\n"));
          return;
        }

        setIndicatorData(result);
        // For "main" pane indicators (e.g. MA, BOLL) overlay onto candle pane;
        // "sub" pane indicators get their own bottom pane.
        const paneId =
          result.pane === "main" ? "candle_pane" : `tdx_${name}_pane`;
        chart.createIndicator(kcName, true, { id: paneId });
        tdxPaneIds.current[name] = paneId;
        setActiveTdxIndicators((prev) => [...prev, name]);
      } catch (err) {
        console.error("加载 TDX 指标失败:", err);
        alert("加载 TDX 指标失败");
      }
    },
    [currentSymbol, timeframe, activeTdxIndicators]
  );

  // When stock or timeframe changes, refresh active TDX indicators with new data
  useEffect(() => {
    if (!currentSymbol || activeTdxIndicators.length === 0) return;
    const chart = mainChartRef.current?.getChart();
    if (!chart) return;

    (async () => {
      const today = new Date().toISOString().slice(0, 10);
      for (const name of activeTdxIndicators) {
        try {
          const result = await getIndicator(
            name,
            currentSymbol,
            timeframe,
            "1990-01-01",
            today
          );
          setIndicatorData(result);
          // Force re-draw by overriding (filter by name since paneId is auto-generated)
          chart.overrideIndicator({ name: getKlineIndicatorName(name) });
        } catch (err) {
          console.error(`刷新 TDX 指标 ${name} 失败:`, err);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSymbol, timeframe]);

  const handleSelectOverlay = useCallback((type: string) => {
    mainChartRef.current?.getChart()?.createOverlay(type);
  }, []);

  // Clear forecast when symbol changes
  useEffect(() => {
    setForecast(null);
  }, [currentSymbol]);

  const handleToggleForecast = useCallback(async () => {
    if (forecast) {
      setForecast(null);
      return;
    }
    if (!currentSymbol) return;
    setForecastLoading(true);
    try {
      const f = await getForecast(currentSymbol);
      setForecast(f);
    } catch (e: any) {
      alert(`预测失败: ${e?.message ?? "unknown"}`);
    } finally {
      setForecastLoading(false);
    }
  }, [currentSymbol, forecast]);

  const hasActions = tradeActions && tradeActions.length > 0;

  return (
    <div className="chart-area">
      <Toolbar
        activeIndicators={activeIndicators}
        activeTdxIndicators={activeTdxIndicators}
        onToggleIndicator={handleToggleIndicator}
        onToggleTdxIndicator={handleToggleTdxIndicator}
        onSelectOverlay={handleSelectOverlay}
        forecastActive={!!forecast}
        forecastLoading={forecastLoading}
        onToggleForecast={handleToggleForecast}
      />
      {hasActions && (
        <div className="trade-nav">
          <button
            className="trade-nav-btn"
            disabled={tradeIdx <= 0}
            onClick={() => goToTrade(tradeIdx <= 0 ? 0 : tradeIdx - 1)}
          >
            ◀ 前交易点
          </button>
          <span className="trade-nav-info">
            {tradeIdx >= 0
              ? `${tradeIdx + 1} / ${tradeActions!.length}`
              : `共 ${tradeActions!.length} 个交易点`}
          </span>
          <button
            className="trade-nav-btn"
            disabled={tradeIdx >= tradeActions!.length - 1}
            onClick={() => goToTrade(tradeIdx + 1)}
          >
            后交易点 ▶
          </button>
        </div>
      )}
      <div className="chart-body">
        {linkedMode ? (
          <LinkedView />
        ) : (
          <MainChart ref={mainChartRef} tradeActions={tradeActions} forecast={forecast} />
        )}
      </div>
    </div>
  );
}
