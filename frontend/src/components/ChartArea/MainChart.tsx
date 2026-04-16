// frontend/src/components/ChartArea/MainChart.tsx
import { useEffect, useRef } from "react";
import { init, dispose } from "klinecharts";
import type { Chart } from "klinecharts";
import { getCandles } from "../../api/quotes";
import { useQuoteStore } from "../../stores/quoteStore";
import type { Timeframe } from "../../types/quote";

interface Props {
  timeframe?: Timeframe;
  className?: string;
}

export function MainChart({ timeframe: tfOverride, className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const storeTimeframe = useQuoteStore((s) => s.timeframe);
  const tf = tfOverride ?? storeTimeframe;

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = init(containerRef.current, {
      styles: {
        grid: {
          horizontal: { color: "#1e1e30" },
          vertical: { color: "#1e1e30" },
        },
        candle: {
          priceMark: { last: { show: true } },
        },
      },
    });
    chartRef.current = chart ?? null;

    // Default indicators
    chart?.createIndicator("VOL", false, { id: "volume_pane" });

    return () => {
      if (containerRef.current) {
        dispose(containerRef.current);
      }
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!currentSymbol || !chartRef.current) return;
    let cancelled = false;

    (async () => {
      try {
        const resp = await getCandles(currentSymbol, tf);
        if (cancelled || !chartRef.current) return;
        chartRef.current.applyNewData(resp.candles);
      } catch (err) {
        console.error("Failed to load candles:", err);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [currentSymbol, tf]);

  return (
    <div
      ref={containerRef}
      className={`main-chart ${className ?? ""}`}
      style={{ width: "100%", height: "100%" }}
    />
  );
}
