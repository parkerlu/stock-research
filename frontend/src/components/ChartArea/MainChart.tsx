// frontend/src/components/ChartArea/MainChart.tsx
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { init, dispose } from "klinecharts";
import type { Chart, KLineData, Period } from "klinecharts";
import { getCandles } from "../../api/quotes";
import { useQuoteStore } from "../../stores/quoteStore";
import type { Candle, Timeframe } from "../../types/quote";
import type { TradeAction } from "../../types/strategy";
import { buildTradeLabel, registerReplayDivider, registerTradeMarker } from "./tradeOverlays";

registerTradeMarker();
registerReplayDivider();

export interface ForecastDay {
  day: number;
  close: number;
  low: number;
  high: number;
  log_return: number;
}
export interface ForecastResult {
  ts_code: string;
  anchor_close: number;
  anchor_date: string;
  forecast: ForecastDay[];
}

/** 两根K线之间的测量结果。 */
export interface MeasureResult {
  fromDate: string;
  toDate: string;
  bars: number;          // 含首尾的K线根数
  calendarDays: number;  // 自然日跨度
  fromClose: number;
  toClose: number;
  changePct: number;     // A收盘 -> B收盘
  /** 上涨看区间最高、下跌看区间最低, 相对 A 收盘的幅度 */
  extremeLabel: "最高" | "最低";
  extremeValue: number;
  extremePct: number;
  extremeDate: string;
}

interface Props {
  timeframe?: Timeframe;
  className?: string;
  tradeActions?: TradeAction[] | null;
  /** 回放已推进到的日期 (YYYY-MM-DD)。画一条竖线区分"已回放"和"尚未到达的未来"。 */
  replayDate?: string | null;
  forecast?: ForecastResult | null;
  onBarSelected?: (date: string | null) => void;
  /** 测量模式: 依次点两根K线出结果, 再点一次开始新一轮 */
  measuring?: boolean;
  onMeasure?: (r: MeasureResult | null) => void;
}

// 图表字体。klinecharts 默认 12px, 在高分屏上读起来费劲。
const TOOLTIP_FONT_SIZE = 15;   // K线图例 + 指标图例 (最常读)
const AXIS_FONT_SIZE = 13;      // 坐标轴刻度 + 十字光标标签

// 首屏加载多长的历史 —— 按周期给, 不能一律 1 年: 1 年只有 52 根周线 / 12 根
// 月线, MA60 之类的长周期指标直接算不出来 (图例显示 n/a)。
const INIT_YEARS: Record<Timeframe, number> = { "1d": 1, "1w": 5, "1m": 15 };

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
  /** 推送一根实时K线 (盘中更新当日末根)。klinecharts v10 走 subscribeBar 回调。 */
  pushBar: (bar: KLineData) => void;
  /** 丢弃缓存重新拉K线 (数据补齐后调用)。加载中则排队到本次加载结束再执行。 */
  reload: () => void;
  /** 跳到某个历史日期 (YYYY-MM-DD)。初始只加载 1 年日线, 回放 2022 年时那天
   *  根本不在图上, 所以要先按目标日期重新取数, 加载完再定位。 */
  focusDate: (dateStr: string) => void;
}

/** scrollToTimestamp 会把目标K线顶到最右边, 标记和分割线就贴着边缘看不清。
 *  往后挪 ~25 根再滚, 目标就落在偏右但留有余量的位置。 */
function anchorTs(list: { timestamp: number }[], ts: number): number {
  const i = list.findIndex((b) => b.timestamp >= ts);
  if (i < 0) return ts;
  return list[Math.min(i + 25, list.length - 1)]?.timestamp ?? ts;
}

export const MainChart = forwardRef<MainChartHandle, Props>(function MainChart(
  { timeframe: tfOverride, className, tradeActions, replayDate, forecast, onBarSelected,
    measuring = false, onMeasure },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const storeTimeframe = useQuoteStore((s) => s.timeframe);
  const tf = tfOverride ?? storeTimeframe;
  const [loading, setLoading] = useState(false);

  // klinecharts 通过 subscribeBar 交给我们一个推送回调, 存起来供 pushBar 使用
  const livePushRef = useRef<((bar: KLineData) => void) | null>(null);

  // 测量状态。点击 effect 只在挂载时绑一次, 所以这些要走 ref 而不是闭包捕获
  const measuringRef = useRef(measuring);
  const onMeasureRef = useRef(onMeasure);
  const measureAnchor = useRef<number | null>(null);
  measuringRef.current = measuring;
  onMeasureRef.current = onMeasure;

  // reload 与 init 取数的竞态: klinecharts 的 resetData 会强制 _loading=false 再
  // 发一次 init, 但它不会作废在途的旧回调 —— 旧回调后落地就会用补齐前的数据盖掉
  // 新数据。所以加载中时把 reload 挂起, 等本次加载结束再执行。
  const loadingRef = useRef(false);
  const pendingReloadRef = useRef(false);
  // 要定位的历史日期: 影响 init 取数的起点, 数据落地后再滚过去
  const focusFromRef = useRef<string | null>(null);
  const focusToRef = useRef<string | null>(null);
  const focusTsRef = useRef<number | null>(null);

  const doReload = () => {
    const chart = chartRef.current;
    const ticker = chart?.getSymbol()?.ticker;
    if (!chart || !ticker) return;
    // 周/月K 由日线聚合, 补齐日线后三个周期的缓存一起作废
    (["1d", "1w", "1m"] as const).forEach((t) => {
      candleCache.delete(`${ticker}|${t}`);
      noMoreHistory.delete(`${ticker}|${t}`);
    });
    chart.resetData();
  };

  useImperativeHandle(ref, () => ({
    scrollToTimestamp: (ts: number) => {
      chartRef.current?.scrollToTimestamp(ts, 300);
    },
    getChart: () => chartRef.current,
    // 若图表尚未订阅 (还没初始化完), 静默丢弃 —— 下一轮轮询会再推一次
    pushBar: (bar: KLineData) => livePushRef.current?.(bar),
    reload: () => {
      if (loadingRef.current) {
        pendingReloadRef.current = true;
        return;
      }
      doReload();
    },
    focusDate: (dateStr: string) => {
      const ts = new Date(`${dateStr}T00:00:00`).getTime();
      if (Number.isNaN(ts)) return;
      focusTsRef.current = ts;
      const cached = candleCache.get(
        `${chartRef.current?.getSymbol()?.ticker}|${periodToTf(
          chartRef.current?.getPeriod() ?? TF_TO_PERIOD["1d"])}`);
      // 已经加载到这一天就直接滚, 免得白重取一次
      if (cached && cached.length > 0 && cached[0].timestamp <= ts) {
        chartRef.current?.scrollToTimestamp(anchorTs(cached, ts), 300);
        setTimeout(() => chartRef.current?.resize(), 360);
        return;
      }
      // 只取目标日附近的窗口, 前 9 个月 + 后 3 个月。
      // 回放 2022 年不该把 2026 年的走势整段铺在眼前, 顺带也少拉几千根K线。
      const from = new Date(ts);
      from.setMonth(from.getMonth() - 9);
      const to = new Date(ts);
      to.setMonth(to.getMonth() + 3);
      const today = new Date();
      focusFromRef.current = fmtDate(from);
      focusToRef.current = fmtDate(to < today ? to : today);
      if (loadingRef.current) pendingReloadRef.current = true;
      else doReload();
    },
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
              // 默认 12px 在高分屏上读起来吃力, 图例是最常看的信息, 放大加粗
              size: TOOLTIP_FONT_SIZE,
              weight: "bold",
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
        indicator: {
          tooltip: {
            legend: { size: TOOLTIP_FONT_SIZE, weight: "bold" },
          },
        },
        xAxis: { tickText: { size: AXIS_FONT_SIZE } },
        yAxis: { tickText: { size: AXIS_FONT_SIZE } },
        crosshair: {
          horizontal: { text: { size: AXIS_FONT_SIZE } },
          vertical: { text: { size: AXIS_FONT_SIZE } },
        },
      },
    });
    chartRef.current = chart ?? null;
    (window as unknown as Record<string, unknown>).__kc = chart;

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
          loadingRef.current = true;
          const now = new Date();
          const from = new Date(
            now.getFullYear() - INIT_YEARS[tfVal], now.getMonth(), now.getDate()
          );
          // focusDate 指定了更早的起点就用它 —— 否则那一天不在数据里, 滚不过去
          const fromStr = focusFromRef.current && focusFromRef.current < fmtDate(from)
            ? focusFromRef.current : fmtDate(from);
          const toStr = focusToRef.current ?? fmtDate(now);
          focusFromRef.current = null;
          focusToRef.current = null;

          try {
            const resp = await getCandles(symInfo.ticker, tfVal, fromStr, toStr);
            const data = resp.candles.map(mapCandle);
            candleCache.set(cacheKey, data);
            setLoading(false);
            callback(data, { forward: true, backward: false });
            // ⚠️ 图表可能是在面板刚打开、还没有 symbol 时初始化的, 那时纵轴定在
            // 默认的 0~10 且**不会**因为后来数据到位而重算 —— 结果价格 5.5~7.2
            // 的股票被压成贴着 6.00 的一条线。数据首次落地后强制重算一次。
            // ⚠️ 未解决的已知问题: 从虚拟盘切股后, 纵轴渲染出来是 0~11,
            // 而 K 线实际只在 3.16~4.29 这一小段, 被压成一条。
            // 已经排除的: 不是 auto-calc 被关 (getAutoCalcTickFlag()===true),
            // 不是区间算错 (getRange() 与可视数据一致, pixelToValue 也对得上),
            // 不是刻度算错 (getTicks() 返回的正是 17.00/18.00/... 这类正确值),
            // 不是实例泄漏 (canvas 恰好 10 个), 不是 DPR (DPR=1 同样复现),
            // 不是加载了太多历史 (把取数窗口收到 12 个月后依旧复现)。
            // 试过 resize()、setPaneOptions({axis})、buildTicks(true) 以及
            // layout({measureWidth,update,buildYAxisTick}) 都推不动画布。
            // 即模型层全对、只有绘制不跟随。留待专门排查 klinecharts 的绘制层。
            requestAnimationFrame(() => {
              const c = chartRef.current as unknown as {
                layout?: (o: Record<string, boolean>) => void; resize: () => void;
              } | null;
              if (!c) return;
              if (typeof c.layout === "function") {
                c.layout({ measureWidth: true, update: true, buildYAxisTick: true });
              } else {
                c.resize();     // 库版本变了就退回去, 至少不报错
              }
            });
            const ft = focusTsRef.current;
            if (ft != null) {
              focusTsRef.current = null;
              // 等 klinecharts 把数据画完再滚, 否则 scrollToTimestamp 找不到那根
              setTimeout(() => {
                const c = chartRef.current;
                const list = (c?.getDataList() ?? []) as { timestamp: number }[];
                c?.scrollToTimestamp(anchorTs(list, ft), 300);
                // 跳转会重新取数, 但 Y 轴不会自己跟着重算 —— 实测数据区间
                // 17.7~29.5 时坐标轴还停在 0~10, K线整个飞出画面。resize()
                // 会强制重算纵轴。
                setTimeout(() => c?.resize(), 360);
              }, 60);
            }
          } catch (err) {
            console.error("Failed to load candles:", err);
            setLoading(false);
            callback([], false);
          } finally {
            loadingRef.current = false;
            if (pendingReloadRef.current) {
              pendingReloadRef.current = false;
              doReload();
            }
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
      // 实时推送通道: klinecharts 把回调交给我们, LiveView 通过 pushBar 喂当日末根。
      // 时间戳等于当日已有K线时它会原地更新, 不存在则追加一根。
      subscribeBar: ({ callback }) => {
        livePushRef.current = callback;
      },
      unsubscribeBar: () => {
        livePushRef.current = null;
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

  // 关掉测量模式 / 换股换周期 -> 清除测量痕迹
  useEffect(() => {
    measureAnchor.current = null;
    chartRef.current?.removeOverlay({ groupId: "measure" });
    if (!measuring) onMeasureRef.current?.(null);
  }, [measuring, currentSymbol, tf]);

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

    const fmtDate = (ts: number) => {
      const d = new Date(ts);
      return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(d.getUTCDate()).padStart(2, "0")}`;
    };

    /** 第一次点定 A, 第二次点定 B 并算结果, 第三次点重新开始。 */
    const handleMeasureClick = (idx: number) => {
      const bars = chart.getDataList() as Array<{
        timestamp: number; high: number; low: number; close: number;
      }>;
      const a = measureAnchor.current;
      if (a === null || a === idx) {
        measureAnchor.current = idx;
        chart.removeOverlay({ groupId: "measure" });
        chart.createOverlay({
          name: "verticalStraightLine", groupId: "measure", lock: true,
          points: [{ timestamp: bars[idx].timestamp, value: bars[idx].close }],
          styles: { line: { color: "#e8b54d", size: 1, style: "dashed" } },
        });
        onMeasureRef.current?.(null);
        return;
      }
      const [i1, i2] = a < idx ? [a, idx] : [idx, a];
      const seg = bars.slice(i1, i2 + 1);
      const fromClose = seg[0].close;
      const toClose = seg[seg.length - 1].close;
      const up = toClose >= fromClose;
      // 上涨看区间最高, 下跌看区间最低 —— 衡量这一段走出去多远
      let ext = seg[0];
      for (const b of seg) {
        if (up ? b.high > ext.high : b.low < ext.low) ext = b;
      }
      const extVal = up ? ext.high : ext.low;
      const ms = seg[seg.length - 1].timestamp - seg[0].timestamp;

      chart.removeOverlay({ groupId: "measure" });
      chart.createOverlay({
        name: "segment", groupId: "measure", lock: true,
        points: [
          { timestamp: seg[0].timestamp, value: fromClose },
          { timestamp: seg[seg.length - 1].timestamp, value: toClose },
        ],
        styles: {
          line: { color: up ? "#e94560" : "#4caf50", size: 2 },
          point: { color: up ? "#e94560" : "#4caf50", borderColor: "#171c26" },
        },
      });

      onMeasureRef.current?.({
        fromDate: fmtDate(seg[0].timestamp),
        toDate: fmtDate(seg[seg.length - 1].timestamp),
        bars: seg.length,
        calendarDays: Math.round(ms / 86400000),
        fromClose, toClose,
        changePct: (toClose / fromClose - 1) * 100,
        extremeLabel: up ? "最高" : "最低",
        extremeValue: extVal,
        extremePct: (extVal / fromClose - 1) * 100,
        extremeDate: fmtDate(ext.timestamp),
      });
      measureAnchor.current = null;
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
        if (measuringRef.current) {
          handleMeasureClick(idx);
          return;
        }
        selectedIndex = idx;
        moveCrosshairTo(idx);
        // Notify parent of selection (for forecast-from-bar)
        if (onBarSelected) {
          const d = new Date(ts);
          const yyyy = d.getUTCFullYear();
          const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
          const dd = String(d.getUTCDate()).padStart(2, "0");
          onBarSelected(`${yyyy}-${mm}-${dd}`);
        }
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

  // Render LSTM forecast as dashed lines + band on the right side of the chart
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.removeOverlay({ groupId: "lstm-forecast" });
    if (!forecast || forecast.forecast.length === 0) return;

    const timer = setTimeout(() => {
      const dataList = chart.getDataList() as Array<{ timestamp: number }>;
      if (dataList.length === 0) return;
      // Anchor at the bar matching forecast.anchor_date if present;
      // otherwise default to the last bar in the chart.
      let anchorTs = dataList[dataList.length - 1].timestamp;
      if (forecast.anchor_date) {
        const wantTs = new Date(forecast.anchor_date + "T00:00:00Z").getTime();
        const match = dataList.find((b) => b.timestamp === wantTs);
        if (match) anchorTs = match.timestamp;
      }

      // Project N business days into future (skip Sat/Sun)
      const projectDays = (anchor: number, n: number): number => {
        const d = new Date(anchor);
        let added = 0;
        while (added < n) {
          d.setUTCDate(d.getUTCDate() + 1);
          const wd = d.getUTCDay();
          if (wd !== 0 && wd !== 6) added += 1;
        }
        return d.getTime();
      };

      const anchorClose = forecast.anchor_close;
      const overlays: any[] = [];
      let prevTs = anchorTs;
      let prevClose = anchorClose;
      let prevHigh = anchorClose;
      let prevLow = anchorClose;
      for (const fd of forecast.forecast) {
        const futureTs = projectDays(anchorTs, fd.day);
        // Predicted close polyline (red dashed)
        overlays.push({
          name: "segment",
          groupId: "lstm-forecast",
          lock: true,
          points: [
            { timestamp: prevTs, value: prevClose },
            { timestamp: futureTs, value: fd.close },
          ],
          styles: {
            line: { color: "#e94560", style: "dashed", size: 2, dashedValue: [4, 4] },
            point: { color: "transparent", borderColor: "transparent" },
          },
          extendData: { day: fd.day, label: `日+${fd.day}: ${fd.close.toFixed(2)}` },
        });
        // Upper band (gray dashed)
        overlays.push({
          name: "segment",
          groupId: "lstm-forecast",
          lock: true,
          points: [
            { timestamp: prevTs, value: prevHigh },
            { timestamp: futureTs, value: fd.high },
          ],
          styles: {
            line: { color: "#888", style: "dashed", size: 1, dashedValue: [3, 3] },
            point: { color: "transparent", borderColor: "transparent" },
          },
        });
        // Lower band (gray dashed)
        overlays.push({
          name: "segment",
          groupId: "lstm-forecast",
          lock: true,
          points: [
            { timestamp: prevTs, value: prevLow },
            { timestamp: futureTs, value: fd.low },
          ],
          styles: {
            line: { color: "#888", style: "dashed", size: 1, dashedValue: [3, 3] },
            point: { color: "transparent", borderColor: "transparent" },
          },
        });
        prevTs = futureTs;
        prevClose = fd.close;
        prevHigh = fd.high;
        prevLow = fd.low;
      }
      chart.createOverlay(overlays);
    }, 300);
    return () => clearTimeout(timer);
  }, [forecast]);

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
            action.type === "buy" || action.type === "signal"
              ? bar?.low ?? action.price
              : bar?.high ?? action.price;
          const label = action.label ?? buildTradeLabel(
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

  // 回放分割线
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.removeOverlay({ groupId: "replay-divider" });
    if (!replayDate) return;
    const ts = new Date(replayDate + "T00:00:00Z").getTime();
    if (Number.isNaN(ts)) return;

    // ⚠️ 锚点的 value 会被算进 Y 轴范围。跳转会触发一次重新取数, 固定延时里
    // getDataList() 还是空的, 取不到价就退成 0 —— 整个价格轴被拉到 0~最高价,
    // K线全挤在顶部。所以要等数据真的到位, 且**宁可不画也不用 0**。
    let tries = 0;
    let timer: ReturnType<typeof setTimeout>;
    const attempt = () => {
      const list = chart.getDataList() as Array<{ timestamp: number; close: number }>;
      const bar = list.find((b) => b.timestamp === ts);
      if (!bar) {
        if (++tries < 20) timer = setTimeout(attempt, 250);
        return;
      }
      chart.createOverlay({
        name: "replayDivider",
        groupId: "replay-divider",
        lock: true,
        points: [{ timestamp: ts, value: bar.close }],
        extendData: { text: `回放至 ${replayDate}` },
      });
    };
    timer = setTimeout(attempt, 500);
    return () => clearTimeout(timer);
  }, [replayDate]);

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
