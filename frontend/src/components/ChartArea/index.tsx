import { useCallback, useEffect, useRef, useState } from "react";
import type { Chart } from "klinecharts";
import { useQuoteStore } from "../../stores/quoteStore";
import type { TradeAction } from "../../types/strategy";
import { getIndicator } from "../../api/indicators";
import { MainChart } from "./MainChart";
import type { MainChartHandle } from "./MainChart";
import { Toolbar } from "./Toolbar";
import { LinkedView } from "./LinkedView";
import { getKlineIndicatorName, setIndicatorData } from "./TdxIndicatorManager";

interface Props {
  tradeActions?: TradeAction[] | null;
}

export function ChartArea({ tradeActions }: Props = {}) {
  const linkedMode = useQuoteStore((s) => s.linkedMode);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const timeframe = useQuoteStore((s) => s.timeframe);
  const [activeIndicators, setActiveIndicators] = useState<string[]>(["MA"]);
  const [activeTdxIndicators, setActiveTdxIndicators] = useState<string[]>([]);
  const chartRef = useRef<Chart | null>(null);
  const mainChartRef = useRef<MainChartHandle>(null);
  const tdxPaneIds = useRef<Record<string, string>>({});
  const [tradeIdx, setTradeIdx] = useState(-1);

  const indicatorPaneIds = useRef<Record<string, string>>({});

  useEffect(() => {
    setTradeIdx(-1);
  }, [tradeActions]);

  const goToTrade = useCallback(
    (idx: number) => {
      if (!tradeActions || idx < 0 || idx >= tradeActions.length) return;
      setTradeIdx(idx);
      const ts = new Date(tradeActions[idx].date + "T00:00:00").getTime();
      mainChartRef.current?.scrollToTimestamp(ts);
    },
    [tradeActions]
  );

  const handleToggleIndicator = useCallback(
    (name: string, isMainPane: boolean) => {
      setActiveIndicators((prev) => {
        const chart = chartRef.current;
        if (prev.includes(name)) {
          if (chart) {
            if (isMainPane) {
              chart.removeIndicator({ paneId: "candle_pane", name });
            } else if (indicatorPaneIds.current[name]) {
              chart.removeIndicator({ paneId: indicatorPaneIds.current[name], name });
              delete indicatorPaneIds.current[name];
            }
          }
          return prev.filter((n) => n !== name);
        } else {
          if (chart) {
            if (isMainPane) {
              chart.createIndicator(name, false, { id: "candle_pane" });
            } else {
              const paneId = chart.createIndicator(name, true);
              if (paneId) {
                indicatorPaneIds.current[name] = paneId;
              }
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

      if (activeTdxIndicators.includes(name)) {
        // Remove
        const paneId = tdxPaneIds.current[name];
        if (paneId) {
          chart.removeIndicator({ paneId, name: getKlineIndicatorName(name) });
          delete tdxPaneIds.current[name];
        }
        setActiveTdxIndicators((prev) => prev.filter((n) => n !== name));
        return;
      }

      // Add: fetch, register, create
      try {
        const now = new Date();
        const from = new Date(now.getFullYear() - 2, now.getMonth(), now.getDate());
        const result = await getIndicator(
          name,
          currentSymbol,
          timeframe,
          from.toISOString().slice(0, 10),
          now.toISOString().slice(0, 10)
        );

        if (result.warnings.length > 0) {
          alert(result.warnings.join("\n"));
          return;
        }

        const kcName = setIndicatorData(result);
        const paneId = chart.createIndicator(kcName, true);
        if (paneId) {
          tdxPaneIds.current[name] = paneId;
          setActiveTdxIndicators((prev) => [...prev, name]);
        }
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
      const now = new Date();
      const from = new Date(now.getFullYear() - 2, now.getMonth(), now.getDate());
      for (const name of activeTdxIndicators) {
        try {
          const result = await getIndicator(
            name,
            currentSymbol,
            timeframe,
            from.toISOString().slice(0, 10),
            now.toISOString().slice(0, 10)
          );
          setIndicatorData(result);
          // Force re-draw by overriding
          const paneId = tdxPaneIds.current[name];
          if (paneId) {
            chart.overrideIndicator({ paneId, name: getKlineIndicatorName(name) });
          }
        } catch (err) {
          console.error(`刷新 TDX 指标 ${name} 失败:`, err);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSymbol, timeframe]);

  const handleSelectOverlay = useCallback((type: string) => {
    chartRef.current?.createOverlay(type);
  }, []);

  const hasActions = tradeActions && tradeActions.length > 0;

  return (
    <div className="chart-area">
      <Toolbar
        activeIndicators={activeIndicators}
        activeTdxIndicators={activeTdxIndicators}
        onToggleIndicator={handleToggleIndicator}
        onToggleTdxIndicator={handleToggleTdxIndicator}
        onSelectOverlay={handleSelectOverlay}
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
          <MainChart ref={mainChartRef} tradeActions={tradeActions} />
        )}
      </div>
    </div>
  );
}
