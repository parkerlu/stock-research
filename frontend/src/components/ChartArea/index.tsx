import { useCallback, useEffect, useRef, useState } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import type { TradeAction } from "../../types/strategy";
import { MainChart } from "./MainChart";
import type { MainChartHandle } from "./MainChart";
import { Toolbar } from "./Toolbar";
import { LinkedView } from "./LinkedView";
import { MAIN_PANE_INDICATORS } from "./indicatorPanes";
import { useTdxIndicators } from "../../hooks/useTdxIndicators";

interface Props {
  tradeActions?: TradeAction[] | null;
}

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
  const mainChartRef = useRef<MainChartHandle>(null);
  const [tradeIdx, setTradeIdx] = useState(-1);

  const indicatorPaneIds = useRef<Record<string, string>>({});

  // TDX 指标的选择/挂载统一交给 hook —— 换股、换周期、重建 pane 都在里面。
  // 多周期联动模式下主图未挂载, hook 的 toggle 仍会更新选择, 联动视图据此
  // 让三张图各按自己的周期取数。
  const getMainChart = useCallback(
    () => mainChartRef.current?.getChart() ?? null,
    []
  );
  const tdx = useTdxIndicators({
    getChart: getMainChart,
    symbol: currentSymbol,
    timeframe,
    storageKey: LS_TDX_KEY,
  });

  // Persist whenever state changes
  useEffect(() => {
    localStorage.setItem(LS_STD_KEY, JSON.stringify(activeIndicators));
  }, [activeIndicators]);

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

  const handleSelectOverlay = useCallback((type: string) => {
    mainChartRef.current?.getChart()?.createOverlay(type);
  }, []);


  const hasActions = tradeActions && tradeActions.length > 0;

  return (
    <div className="chart-area">
      <Toolbar
        activeIndicators={activeIndicators}
        activeTdxIndicators={tdx.active}
        onToggleIndicator={handleToggleIndicator}
        onToggleTdxIndicator={tdx.toggle}
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
          <LinkedView
            activeIndicators={activeIndicators}
            activeTdxIndicators={tdx.active}
          />
        ) : (
          <MainChart ref={mainChartRef} tradeActions={tradeActions} />
        )}
      </div>
    </div>
  );
}
