// frontend/src/components/ChartArea/MainChart.tsx
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { init, dispose } from "klinecharts";
import type { Chart, KLineData, Period } from "klinecharts";
import { getCandles } from "../../api/quotes";
import { useQuoteStore } from "../../stores/quoteStore";
import type { Candle, Timeframe } from "../../types/quote";
import type { TradeAction } from "../../types/strategy";
import { buildTradeLabel } from "./tradeOverlays";

interface Props {
  timeframe?: Timeframe;
  className?: string;
  tradeActions?: TradeAction[] | null;
}

const TF_TO_PERIOD: Record<Timeframe, Period> = {
  "1d": { type: "day", span: 1 },
  "1w": { type: "week", span: 1 },
  "1m": { type: "month", span: 1 },
};

function periodToTf(period: Period): Timeframe {
  if (period.type === "week") return "1w";
  if (period.type === "month") return "1m";
  return "1d";
}

function fmtDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function mapCandle(c: Candle): KLineData {
  return {
    timestamp: c.timestamp,
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
    volume: c.volume,
    turnover: c.amount,
  };
}

// Module-level cache: stores all loaded candles per symbol+tf
const candleCache = new Map<string, KLineData[]>();
// Tracks whether we've hit the earliest available data
const noMoreHistory = new Set<string>();

export interface MainChartHandle {
  scrollToTimestamp: (timestamp: number) => void;
  getChart: () => Chart | null;
}

export const MainChart = forwardRef<MainChartHandle, Props>(function MainChart(
  { timeframe: tfOverride, className, tradeActions },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const storeTimeframe = useQuoteStore((s) => s.timeframe);
  const tf = tfOverride ?? storeTimeframe;
  const [loading, setLoading] = useState(false);

  useImperativeHandle(ref, () => ({
    scrollToTimestamp: (ts: number) => {
      chartRef.current?.scrollToTimestamp(ts, 300);
    },
    getChart: () => chartRef.current,
  }));

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = init(containerRef.current, {
      styles: {
        grid: {
          horizontal: { color: "#1e1e30" },
          vertical: { color: "#1e1e30" },
        },
        candle: {
          bar: {
            upColor: "#e94560",
            downColor: "#4caf50",
            noChangeColor: "#888",
            upBorderColor: "#e94560",
            downBorderColor: "#4caf50",
            noChangeBorderColor: "#888",
            upWickColor: "#e94560",
            downWickColor: "#4caf50",
            noChangeWickColor: "#888",
          },
          priceMark: { last: { show: true } },
        },
      },
    });
    chartRef.current = chart ?? null;

    chart?.createIndicator("VOL", false, { id: "volume_pane" });

    // klinecharts v10 DataLoader convention:
    //   "forward" = user scrolled LEFT past oldest data → fetch OLDER history → PREPEND
    //   "backward" = user scrolled RIGHT past newest data → fetch NEWER data → APPEND
    chart?.setDataLoader({
      getBars: async ({ type, symbol: symInfo, period, callback }) => {
        if (!symInfo) {
          callback([], false);
          return;
        }

        const tfVal = periodToTf(period);
        const cacheKey = `${symInfo.ticker}|${tfVal}`;

        if (type === "init") {
          const cached = candleCache.get(cacheKey);
          if (cached && cached.length > 0) {
            const hasMore = !noMoreHistory.has(cacheKey);
            callback(cached, { forward: hasMore, backward: false });
            return;
          }

          setLoading(true);
          const now = new Date();
          const oneYearAgo = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());

          try {
            const resp = await getCandles(symInfo.ticker, tfVal, fmtDate(oneYearAgo), fmtDate(now));
            const data = resp.candles.map(mapCandle);
            candleCache.set(cacheKey, data);
            setLoading(false);
            callback(data, { forward: true, backward: false });
          } catch (err) {
            console.error("Failed to load candles:", err);
            setLoading(false);
            callback([], false);
          }
        } else if (type === "forward") {
          // User scrolled left past the oldest candle → load older history
          const cached = candleCache.get(cacheKey);
          if (!cached || cached.length === 0 || noMoreHistory.has(cacheKey)) {
            callback([], { forward: false });
            return;
          }

          const earliestTs = cached[0].timestamp;
          const endDate = new Date(earliestTs);
          endDate.setDate(endDate.getDate() - 1);
          const startDate = new Date(endDate);
          startDate.setFullYear(startDate.getFullYear() - 2);

          try {
            const resp = await getCandles(symInfo.ticker, tfVal, fmtDate(startDate), fmtDate(endDate));
            const newData = resp.candles.map(mapCandle);

            if (newData.length > 0) {
              candleCache.set(cacheKey, [...newData, ...cached]);
              // klinecharts prepends forward data correctly
              callback(newData, { forward: true });
            } else {
              noMoreHistory.add(cacheKey);
              callback([], { forward: false });
            }
          } catch {
            callback([], { forward: false });
          }
        } else {
          // "backward" (newer data) or "update" — not needed for now
          callback([], false);
        }
      },
    });

    // Resize chart when container size changes (needed for flex layouts)
    const ro = new ResizeObserver(() => {
      chartRef.current?.resize();
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      if (containerRef.current) {
        dispose(containerRef.current);
      }
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.setPeriod(TF_TO_PERIOD[tf]);
  }, [tf]);

  useEffect(() => {
    if (!currentSymbol || !chartRef.current) return;
    chartRef.current.setSymbol({ ticker: currentSymbol });
  }, [currentSymbol]);

  // Render trade markers on chart
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    // Clear previous trade markers
    chart.removeOverlay({ groupId: "trade-markers" });

    if (!tradeActions || tradeActions.length === 0) return;

    // Wait for chart data to be ready, then add overlays
    const timer = setTimeout(() => {
      const overlays = tradeActions.map((action) => {
        const ts = new Date(action.date + "T00:00:00").getTime();
        const label = buildTradeLabel(
          action.type,
          action.position_level,
          action.pnl,
          action.pnl_pct
        );
        const isBuy = action.type === "buy";
        return {
          name: "simpleAnnotation",
          groupId: "trade-markers",
          lock: true,
          points: [{ timestamp: ts, value: action.price }],
          extendData: label,
          styles: {
            line: { color: isBuy ? "#e94560" : "#4caf50" },
            polygon: { color: isBuy ? "#e94560" : "#4caf50" },
            text: { color: isBuy ? "#e94560" : "#4caf50", size: 11 },
          },
        };
      });
      chart.createOverlay(overlays);
    }, 500);

    return () => clearTimeout(timer);
  }, [tradeActions]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div
        ref={containerRef}
        className={`main-chart ${className ?? ""}`}
        style={{ width: "100%", height: "100%" }}
      />
      {loading && (
        <div className="chart-loading">
          <div className="chart-loading-spinner" />
          <span>K线加载中...</span>
        </div>
      )}
    </div>
  );
});
