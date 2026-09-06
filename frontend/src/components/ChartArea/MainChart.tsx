// frontend/src/components/ChartArea/MainChart.tsx
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { init, dispose } from "klinecharts";
import type { Chart, KLineData, Period } from "klinecharts";
import { getCandles } from "../../api/quotes";
import { useQuoteStore } from "../../stores/quoteStore";
import type { Candle, Timeframe } from "../../types/quote";
import type { TradeAction } from "../../types/strategy";
import { buildTradeLabel, registerReplayDivider, registerTradeMarker } from "./tradeOverlays";
import { PaneCloseButtons } from "./PaneCloseButtons";
import type { PaneBtn } from "./PaneCloseButtons";
import { registerTrainedMarker, paintTrainedMarkers,
         registerSelectedSignal, paintSelectedSignal } from "./trainedOverlays";
import { createTrainedPane, removeTrainedPane, setTrainedData } from "./TrainedPaneManager";

registerTradeMarker();
registerReplayDivider();
registerTrainedMarker();
registerSelectedSignal();

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

  /** 训练指标信号。主图标记类: 颜色即等级(金 v4 > 紫 v3.5 > 红 v3)。
   *  v5 不在这里 —— 因果口径下一年仅约 7.5 个, 放 K 线上常年空白, 只在选股页出现。 */
  maimaiSignals?: { date: string; side?: string; score: number; rank_pct: number; grade: string }[];
  comboSignals?: { date: string; score: number; rank_pct: number; grade: string }[];
  /** 副图类: 信号密集时主图会糊成一片, 画成副图能看出评分随时间的变化。 */
  pumpSignals?: { date: string; prob: number; rank_pct: number; grade: string }[];
  didianSignals?: { date: string; score: number; rank_pct: number; grade: string }[];
  /** 选股页点中的信号日期 —— 在那根K线上打青色高亮带 */
  signalMark?: string | null;
  /** 副图关闭按钮: 每个副图右上角一个 ×, 点了从对应的开关状态里移除 */
  panes?: PaneBtn[];
}

// 图表字体。klinecharts 默认 12px, 在高分屏上读起来费劲。
const TOOLTIP_FONT_SIZE = 15;   // K线图例 + 指标图例 (最常读)
const AXIS_FONT_SIZE = 13;      // 坐标轴刻度 + 十字光标标签

// 首屏加载多长的历史 —— 按周期给, 不能一律 1 年: 1 年只有 52 根周线 / 12 根
// 月线, MA60 之类的长周期指标直接算不出来 (图例显示 n/a)。
// 日线首屏加载年数。训练指标的标记只能画在已加载的K线上, 给 1 年的话
// v4 这种每票 1.6 个信号的指标基本看不到东西 —— 3 年才够看出规律。
const INIT_YEARS: Record<Timeframe, number> = { "1d": 3, "1w": 5, "1m": 15 };

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

/** K线时间戳 → 日期串。时间戳是 UTC 零点, 必须用 UTC getter 取。
 *  ⚠️ 别跟 fmtDate 混用: fmtDate 是给本地构造的 Date(如"今天往前9个月")算日历用的。 */
function fmtBarDate(ts: number): string {
  return new Date(ts).toISOString().slice(0, 10);
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
    measuring = false, onMeasure,
    maimaiSignals, comboSignals, pumpSignals, didianSignals, signalMark, panes },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const storeTimeframe = useQuoteStore((s) => s.timeframe);

  // ⚠️ 往回滚动补历史后必须重画标记: paintTrainedMarkers 画完就停, 新加载的
  // 旧K线上不会自己补出标记 —— 表现就是"滚回去标记没了", 和指标坏掉分不清。
  // 这里记住当前各组信号, 由 DataLoader 在补完数据后调用 repaintTrained()。
  const trainedRef = useRef<Record<string, { sig: any; color?: string }>>({});
  /** 首屏可视窗口固定约 1 年(日线 243 根)。
   *
   *  数据取的是 3 年(见 INIT_YEARS) —— 数据窗口和可视窗口是两回事:
   *  取 3 年是为了往回滚有东西看、标记有 K 线可落; 只显示 1 年是因为
   *  726 根铺满后一根 K 线不到 2px, 看不出形态。往回滚会自动补历史,
   *  补完 repaintTrained() 会把那段的标记画上。 */
  const VISIBLE_BARS: Record<Timeframe, number> = { "1d": 243, "1w": 120, "1m": 120 };
  const fitWindow = useRef((tfKey: Timeframe) => {
    const c = chartRef.current;
    const n = (c?.getDataList() ?? []).length;
    if (!c || n === 0) return;
    const w = containerRef.current?.clientWidth ?? 0;
    const want = Math.min(VISIBLE_BARS[tfKey] ?? 243, n);
    if (w > 0 && want > 0) c.setBarSpace(Math.max(w / want, 0.8));
  });

  const selMarkRef = useRef<string | null | undefined>(null);
  const repaintTrained = useRef(() => {
    for (const [gid, v] of Object.entries(trainedRef.current)) {
      paintTrainedMarkers(chartRef.current, gid, v.sig, v.color);
    }
    paintSelectedSignal(chartRef.current, selMarkRef.current);
  });

  const tf = tfOverride ?? storeTimeframe;
  // ===== 训练指标 =====
  // 主图标记: 一律走 paintTrainedMarkers(内含重试)。
  // ⚠️ 不要改回"固定 setTimeout 后读 getDataList": 从选股列表点进来时信号接口
  // 比 K 线快, 定时到点时 K 线还是空的, overlay 一个都建不出来, 而 effect 只依赖
  // signals 不会因 K 线到位重跑 —— 标记就静默消失了(实测 603997.SH 09-03)。
  useEffect(() => {
    selMarkRef.current = signalMark;
    return paintSelectedSignal(chartRef.current, signalMark);
  }, [signalMark]);

  useEffect(() => {
    trainedRef.current["maimai"] = { sig: maimaiSignals };
    return paintTrainedMarkers(chartRef.current, "maimai", maimaiSignals);
  }, [maimaiSignals]);
  useEffect(() => {
    trainedRef.current["combo"] = { sig: comboSignals, color: "#f0a020" };
    return paintTrainedMarkers(chartRef.current, "combo", comboSignals, "#f0a020");
  }, [comboSignals]);

  // 副图: 主力吸筹 / 低点组合。
  // ⚠️ createTrainedPane 内部必须 isStack=true, 传 false 时 klinecharts 静默不建面板。
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (!pumpSignals || pumpSignals.length === 0) {
      removeTrainedPane(chart, "pump", tf);
      return;
    }
    setTrainedData("pump", tf, pumpSignals.map((p) => ({ date: p.date, value: p.prob, grade: p.grade })));
    createTrainedPane(chart, "pump", tf);
    return () => removeTrainedPane(chart, "pump", tf);
  }, [pumpSignals, tf]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (!didianSignals || didianSignals.length === 0) {
      removeTrainedPane(chart, "didian", tf);
      return;
    }
    setTrainedData("didian", tf, didianSignals.map((p) => ({ date: p.date, value: p.rank_pct, grade: p.grade })));
    createTrainedPane(chart, "didian", tf);
    return () => removeTrainedPane(chart, "didian", tf);
  }, [didianSignals, tf]);

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
      // ⚠️ 必须按 UTC 解析: 后端 timestamp 由 calendar.timegm() 生成, K线落在 UTC 零点。
      // 用本地解析(UTC+8)会得到前一天 16:00Z, 定位就偏一根。
      const ts = Date.parse(`${dateStr}T00:00:00Z`);
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
          // ⚠️ 成交量柱必须跟 K 线同色板(A股: 涨红 #e94560 / 跌绿 #4caf50)。
          // klinecharts 默认是欧美色板(涨绿跌红), 不改的话量和K线颜色相反。
          bars: [{ style: "fill", upColor: "#e94560", downColor: "#4caf50", noChangeColor: "#888888" }],
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
            setTimeout(() => { fitWindow.current(tfVal); repaintTrained.current(); }, 120);
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
            setTimeout(() => { fitWindow.current(tfVal); repaintTrained.current(); }, 120);
            // ⚠️ 图表可能是在面板刚打开、还没有 symbol 时初始化的, 那时纵轴定在
            // 默认的 0~10 且**不会**因为后来数据到位而重算 —— 结果价格 5.5~7.2
            // 的股票被压成贴着 6.00 的一条线。数据首次落地后强制重算一次。
            // 【已定位】"切股后纵轴渲染成 0~11、K线压成一条"其实是两个问题:
            // 1) 模型层竞态(真 bug, 偶发): scrollToTimestamp 之后可视窗口可能
            //    整个滑出数据范围, 空窗口让 klinecharts 回落到默认 0~10 区间
            //    (createRangeImp 里 min/max 无数据时取 0/10, 加 gap 后 -1~12)。
            //    下面 focusTs 链的末尾加了"落点校验"兜底。
            // 2) 合成器旧帧(占绝大多数复现, 非 klinecharts bug): 逐帧 hook
            //    updateMain 证明切股后每次绘制的 range 都正确, canvas 光栅
            //    (toDataURL) 也正确, 但 Chromium 在页面静止 ~1.5s 后按需出帧
            //    (无头截图/被遮挡窗口)会回退到切股瞬间 Y 轴宽度 51→66→51 抖动
            //    重建 backing store 之前的旧纹理 —— 看起来就是"停在 0~11"。
            //    直接往 ctx fillRect 都上不了屏, 但 DOM 改动能上屏, 即只有
            //    canvas 纹理被回退。靠 init useEffect 里的 rAF 心跳规避。
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
                // 落点校验: 滚动动画 + 补历史取数的竞态偶发把可视窗口甩出
                // 数据范围(窗口里 0 根K线 → 纵轴回落默认 0~10)。等动画和
                // resize 都结束后检查一次, 空了就直接跳回锚点。
                setTimeout(() => {
                  const cc = chartRef.current;
                  if (!cc) return;
                  const vr = cc.getVisibleRange();
                  const n = cc.getDataList().length;
                  if (n > 0 && (vr.to <= 0 || vr.from >= n)) {
                    const l2 = cc.getDataList() as { timestamp: number }[];
                    cc.scrollToTimestamp(anchorTs(l2, ft), 0);
                    cc.resize();
                  }
                }, 500);
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
              setTimeout(() => repaintTrained.current(), 120);   // 补历史后补画标记
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

    // ⚠️ workaround (Chromium 合成器旧帧, 详见 getBars init 回调里的注释):
    // 切股时 Y 轴宽度 51→66→51 抖动导致 canvas backing store 两次重建;
    // 此后页面一旦静止 ~1.5s, Chromium 按需出帧(无头截图、被遮挡/后台窗口)
    // 会回退到重建前的旧纹理 —— 模型和光栅全对, 唯独屏幕/截图停在旧刻度。
    // 实测: 任何一次性补救(updatePane 全量重绘、el.width 重建纹理、
    // --disable-accelerated-2d-canvas)都会在下次闲置后复现; 只有渲染器持续
    // 产帧时合成器才始终持有最新纹理。空 rAF 心跳: 无绘制、无布局, 只是让
    // 渲染器保持出帧。klinecharts v10.0.0-beta1 自身绘制已验证无误。
    let heartbeatId = requestAnimationFrame(function beat() {
      heartbeatId = requestAnimationFrame(beat);
    });

    return () => {
      cancelAnimationFrame(heartbeatId);
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
        fromDate: fmtBarDate(seg[0].timestamp),
        toDate: fmtBarDate(seg[seg.length - 1].timestamp),
        bars: seg.length,
        calendarDays: Math.round(ms / 86400000),
        fromClose, toClose,
        changePct: (toClose / fromClose - 1) * 100,
        extremeLabel: up ? "最高" : "最低",
        extremeValue: extVal,
        extremePct: (extVal / fromClose - 1) * 100,
        extremeDate: fmtBarDate(ext.timestamp),
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
      <PaneCloseButtons getChart={() => chartRef.current} panes={panes ?? []} />
      {loading && (
        <div className="chart-loading">
          <div className="chart-loading-spinner" />
          <span>K线加载中...</span>
        </div>
      )}
    </div>
  );
});
