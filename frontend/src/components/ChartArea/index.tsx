import { useCallback, useRef, useState } from "react";
import type { Chart } from "klinecharts";
import { useQuoteStore } from "../../stores/quoteStore";
import { MainChart } from "./MainChart";
import { Toolbar } from "./Toolbar";
import { LinkedView } from "./LinkedView";

export function ChartArea() {
  const linkedMode = useQuoteStore((s) => s.linkedMode);
  const [activeIndicators, setActiveIndicators] = useState<string[]>(["MA"]);
  const chartRef = useRef<Chart | null>(null);

  const indicatorPaneIds = useRef<Record<string, string>>({});

  const handleToggleIndicator = useCallback(
    (name: string, isMainPane: boolean) => {
      setActiveIndicators((prev) => {
        const chart = chartRef.current;
        if (prev.includes(name)) {
          if (chart) {
            if (isMainPane) {
              chart.removeIndicator("candle_pane", name);
            } else if (indicatorPaneIds.current[name]) {
              chart.removeIndicator(indicatorPaneIds.current[name], name);
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

  const handleSelectOverlay = useCallback((type: string) => {
    chartRef.current?.createOverlay(type);
  }, []);

  return (
    <div className="chart-area">
      <Toolbar
        activeIndicators={activeIndicators}
        onToggleIndicator={handleToggleIndicator}
        onSelectOverlay={handleSelectOverlay}
      />
      <div className="chart-body">
        {linkedMode ? (
          <LinkedView />
        ) : (
          <MainChart />
        )}
      </div>
    </div>
  );
}
