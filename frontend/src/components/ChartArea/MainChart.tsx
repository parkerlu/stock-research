// frontend/src/components/ChartArea/MainChart.tsx
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { init, dispose } from "klinecharts";
import type { Chart, KLineData, Period } from "klinecharts";
import { getCandles } from "../../api/quotes";
import { useQuoteStore } from "../../stores/quoteStore";
import type { Candle, Timeframe } from "../../types/quote";
import type { TradeAction } from "../../types/strategy";
import { buildTradeLabel, registerTradeMarker } from "./tradeOverlays";

registerTradeMarker();

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
      locale: "zh-CN",
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
          priceMark: { last: { show: false } },
          tooltip: {
            legend: {
              template: ({ current, prev }: any) => {
                const prevClose = Number(prev?.close ?? current?.close ?? 0);
                const close = Number(current?.close ?? 0);
                const open = Number(current?.open ?? 0);
                const change = prevClose > 0 ? (close - prevClose) / prevClose * 100 : 0;
                const changeColor = change > 0 ? "#e94560" : change < 0 ? "#4caf50" : "#888";
                const closeColor = close > open ? "#e94560" : close < open ? "#4caf50" : "#888";
                return [
                  { title: "时间", value: "{time}" },
                  { title: "开", value: "{open}" },
                  { title: "高", value: { text: "{high}", color: "#e94560" } },
                  { title: "低", value: { text: "{low}", color: "#4caf50" } },
                  { title: "收", value: { text: "{close}", color: closeColor } },
                  {
                    title: "涨幅",
                    value: {
                      text: `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`,
                      color: changeColor,
                    },
                  },
                  { title: "成交量", value: "{volume}" },
                ];
              },
            },
          },
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

  // Click anywhere in the chart selects the nearest K-line as the anchor;
  // ArrowLeft/Right move from that anchor; ArrowUp/Down zoom in/out around it.
  useEffect(() => {
    const chart = chartRef.current;
    const container = containerRef.current;
    if (!chart || !container) return;

    let selectedIndex: number | undefined;

    const moveCrosshairTo = (idx: number) => {
      const dataList = chart.getDataList();
      const bar = dataList[idx] as { close?: number } | undefined;
      const point = chart.convertToPixel(
        { dataIndex: idx, value: bar?.close },
        { paneId: "candle_pane", absolute: false }
      );
      const coord = Array.isArray(point) ? point[0] : point;
      if (coord?.x === undefined) return;
      chart.executeAction("onCrosshairChange", {
        x: coord.x,
        y: coord.y ?? 0,
        paneId: "candle_pane",
      });
    };

    // Mouse-down anywhere in the chart container → resolve nearest bar via pixel→data
    const onClick = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const result = chart.convertFromPixel(
        [{ x, y: 0 }],
        { paneId: "candle_pane", absolute: false }
      );
      const point = Array.isArray(result) ? result[0] : result;
      const ts = (point as { timestamp?: number } | undefined)?.timestamp;
      if (ts === undefined) return;
      const dataList = chart.getDataList() as { timestamp?: number }[];
      const idx = dataList.findIndex((b) => b.timestamp === ts);
      if (idx >= 0) {
        selectedIndex = idx;
        moveCrosshairTo(idx);
      }
    };
    container.addEventListener("mousedown", onClick);

    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (document.activeElement?.tagName ?? "").toUpperCase();
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      const dataList = chart.getDataList();
      if (dataList.length === 0) return;

      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const start = selectedIndex ?? dataList.length - 1;
        const next =
          e.key === "ArrowLeft"
            ? Math.max(0, start - 1)
            : Math.min(dataList.length - 1, start + 1);
        if (next === start) return;
        e.preventDefault();
        selectedIndex = next;
        moveCrosshairTo(next);
      } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        // Zoom around the currently selected bar (or last bar if none selected)
        const anchorIdx = selectedIndex ?? dataList.length - 1;
        // ArrowUp zooms IN (scale > 1), ArrowDown zooms OUT (scale < 1)
        const scale = e.key === "ArrowUp" ? 1.25 : 0.8;
        e.preventDefault();
        chart.zoomAtDataIndex(scale, anchorIdx, 100);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      container.removeEventListener("mousedown", onClick);
    };
  }, []);

  // Render trade markers on chart
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    // Clear previous trade markers
    chart.removeOverlay({ groupId: "trade-markers" });

    if (!tradeActions || tradeActions.length === 0) return;

    // Wait for chart data to be ready, then add overlays
    const timer = setTimeout(() => {
      // Look up actual K-line bars to anchor markers to bar low (buy) or high (sell)
      const dataList = chart.getDataList() as Array<{
        timestamp: number;
        low: number;
        high: number;
      }>;
      const overlays = tradeActions
        .map((action) => {
          // IMPORTANT: parse as UTC, since backend candle timestamps are UTC midnight
          const ts = new Date(action.date + "T00:00:00Z").getTime();
          const bar = dataList.find((b) => b.timestamp === ts);
          const anchor =
            action.type === "buy"
              ? bar?.low ?? action.price
              : bar?.high ?? action.price;
          const label = buildTradeLabel(
            action.type,
            action.price,
            action.pnl_pct,
          );
          return {
            name: "tradeMarker",
            groupId: "trade-markers",
            lock: true,
            points: [{ timestamp: ts, value: anchor }],
            extendData: { type: action.type, text: label },
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
