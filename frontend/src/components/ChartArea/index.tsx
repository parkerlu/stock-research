import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import type { TradeAction } from "../../types/strategy";
import { MainChart } from "./MainChart";
import type { MainChartHandle, MeasureResult } from "./MainChart";
import { Toolbar } from "./Toolbar";
import { LinkedView } from "./LinkedView";
import { MAIN_PANE_INDICATORS } from "./indicatorPanes";
import { StockSectors } from "./StockSectors";
import { TopListBadge } from "./TopListBadge";
import { getMaimaiSignals, getPumpSignals, getDidianSignals, getComboSignals , getLiftAlertSignals, getBreakoutSignals, getSarSignals, getMmWeekSignals } from "../../api/quotes";
import { useTdxIndicators } from "../../hooks/useTdxIndicators";

interface Props {
  tradeActions?: TradeAction[] | null;
}

const LS_STD_KEY = "chart.activeIndicators.v1";

// MA 周期。klinecharts 默认 5/10/30/60, 这里按需要补上 20。
const MA_PERIODS = [5, 10, 20, 30, 60];
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

function MeasureBar({ result }: { result: MeasureResult | null }) {
  if (!result) {
    return (
      <div className="measure-bar measure-hint">
        📏 测量中 —— 点第一根 K 线定起点，再点第二根出结果
      </div>
    );
  }
  const cls = result.changePct >= 0 ? "up" : "down";
  const sign = (v: number) => (v > 0 ? "+" : "");
  return (
    <div className="measure-bar">
      <span className="m-range">{result.fromDate} → {result.toDate}</span>
      <span className="m-item"><i>{result.bars}</i> 根K</span>
      <span className="m-item"><i>{result.calendarDays}</i> 自然日</span>
      <span className="m-sep" />
      <span className="m-item">起 <i>{result.fromClose.toFixed(2)}</i></span>
      <span className="m-sep" />
      {/* A收盘 → B那根的 最高/最低/收盘 三个口径。
          最高 = 这段最多能赚到多少(若在B当天最高点卖)
          最低 = 中途最多要扛多少(若在B当天最低点割)
          收盘 = 老老实实持有到B收盘的结果 */}
      <span className="m-item">B最高 <i>{result.toHigh.toFixed(2)}</i></span>
      <span className={`m-pct ${result.toHighPct >= 0 ? "up" : "down"}`}>
        {sign(result.toHighPct)}{result.toHighPct.toFixed(2)}%
      </span>
      <span className="m-item">B最低 <i>{result.toLow.toFixed(2)}</i></span>
      <span className={`m-pct ${result.toLowPct >= 0 ? "up" : "down"}`}>
        {sign(result.toLowPct)}{result.toLowPct.toFixed(2)}%
      </span>
      <span className="m-item">B收盘 <i>{result.toClose.toFixed(2)}</i></span>
      <span className={`m-pct ${cls}`}>
        {sign(result.changePct)}{result.changePct.toFixed(2)}%
      </span>
      <span className="m-sep" />
      <span className="m-item">
        区间{result.extremeLabel} <i>{result.extremeValue.toFixed(2)}</i>
        <em>({result.extremeDate})</em>
      </span>
      <span className={`m-pct ${cls}`}>
        {sign(result.extremePct)}{result.extremePct.toFixed(2)}%
      </span>
    </div>
  );
}

export function ChartArea({ tradeActions }: Props = {}) {
  const linkedMode = useQuoteStore((s) => s.linkedMode);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const timeframe = useQuoteStore((s) => s.timeframe);
  const focusDate = useQuoteStore((s) => s.focusDate);
  const focusSeq = useQuoteStore((s) => s.focusSeq);
  const signalMark = useQuoteStore((s) => s.signalMark);

  const clearFocusDate = useQuoteStore((s) => s.clearFocusDate);
  const paperMarks = useQuoteStore((s) => s.paperMarks);
  const replayDate = useQuoteStore((s) => s.replayDate);
  // 策略回测的标记优先; 没有时才画虚拟盘的成交
  const marks = tradeActions && tradeActions.length ? tradeActions : paperMarks;
  const [activeIndicators, setActiveIndicators] = useState<string[]>(() =>
    readLS(LS_STD_KEY, ["MA"])
  );

  // 自训练指标 —— 目前只有买卖很准v3, 后续新增的训练指标都挂这里
  const [lhb, setLhb] = useState<{ date: string; net_wan: number }[]>([]);
  // 龙虎榜默认开 —— 它是基础信息, 但用户可以在「训练」菜单里关掉。
  const [activeTrained, setActiveTrained] = useState<string[]>(() => {
    try {
      const raw = localStorage.getItem("chart.trained.v1");
      const p = raw ? JSON.parse(raw) : null;
      if (!Array.isArray(p)) return ["toplist"];
      // 一次性迁移: 龙虎榜标记以前是常驻的(没有开关), 加开关后老用户的存档里
      // 没有这一项, 直接按"关"处理会让标记莫名消失。补一次, 之后完全听用户的。
      const MIG = "chart.trained.toplist.migrated";
      if (!localStorage.getItem(MIG)) {
        localStorage.setItem(MIG, "1");
        if (!p.includes("toplist")) return [...p, "toplist"];
      }
      return p;
    } catch {
      return ["toplist"];
    }
  });
  const [mmwkSig, setMmwkSig] = useState<
    { date: string; value: number; grade: string }[]
  >([]);
  const [sarSig, setSarSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [boSig, setBoSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [liftSig, setLiftSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [comboSig, setComboSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [didianSig, setDidianSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [pumpSignals, setPumpSignals] = useState<
    { date: string; prob: number; rank_pct: number; grade: string }[]
  >([]);
  const [maimaiSignals, setMaimaiSignals] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  useEffect(() => {
    localStorage.setItem("chart.trained.v1", JSON.stringify(activeTrained));
  }, [activeTrained]);
  useEffect(() => {
    if (!currentSymbol || !activeTrained.includes("maimai_v3")) {
      setMaimaiSignals([]);
      return;
    }
    let live = true;
    getMaimaiSignals(currentSymbol, "弱")
      .then((r) => live && setMaimaiSignals(r.signals ?? []))
      .catch(() => live && setMaimaiSignals([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, activeTrained]);

  useEffect(() => {
    if (!currentSymbol || !activeTrained.includes("pump")) {
      setPumpSignals([]);
      return;
    }
    let live = true;
    getPumpSignals(currentSymbol, "中")
      .then((r) => live && setPumpSignals(r.signals ?? []))
      .catch(() => live && setPumpSignals([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, activeTrained]);

  useEffect(() => {
    if (!currentSymbol || !activeTrained.includes("didian")) {
      setDidianSig([]);
      return;
    }
    let live = true;
    getDidianSignals(currentSymbol, "中")
      .then((r) => live && setDidianSig(r.signals ?? []))
      .catch(() => live && setDidianSig([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, activeTrained]);

  useEffect(() => {
    if (!currentSymbol || !activeTrained.includes("combo")) {
      setComboSig([]);
      return;
    }
    let live = true;
    getComboSignals(currentSymbol)
      .then((r) => live && setComboSig(r.signals ?? []))
      .catch(() => live && setComboSig([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, activeTrained]);

  useEffect(() => {
    if (!currentSymbol || !activeTrained.includes("liftalert")) {
      setLiftSig([]);
      return;
    }
    let live = true;
    getLiftAlertSignals(currentSymbol)
      .then((r) => live && setLiftSig(r.signals ?? []))
      .catch(() => live && setLiftSig([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, activeTrained]);

  useEffect(() => {
    if (!currentSymbol || !activeTrained.includes("breakout")) {
      setBoSig([]);
      return;
    }
    let live = true;
    getBreakoutSignals(currentSymbol)
      .then((r) => live && setBoSig(r.signals ?? []))
      .catch(() => live && setBoSig([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, activeTrained]);

  useEffect(() => {
    if (!currentSymbol || !activeTrained.includes("sar")) {
      setSarSig([]);
      return;
    }
    let live = true;
    getSarSignals(currentSymbol)
      .then((r) => live && setSarSig(r.signals ?? []))
      .catch(() => live && setSarSig([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, activeTrained]);

  useEffect(() => {
    if (!currentSymbol || !activeTrained.includes("mmweek")) {
      setMmwkSig([]);
      return;
    }
    let live = true;
    getMmWeekSignals(currentSymbol)
      .then((r) => live && setMmwkSig(r.signals ?? []))
      .catch(() => live && setMmwkSig([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, activeTrained]);


  const mainChartRef = useRef<MainChartHandle>(null);
  const [tradeIdx, setTradeIdx] = useState(-1);
  const [measuring, setMeasuring] = useState(false);
  const [measure, setMeasure] = useState<MeasureResult | null>(null);

  const indicatorPaneIds = useRef<Record<string, string>>({});

  // 虚拟盘点持仓 -> 跳到该股该日的K线。换股后图表要重新初始化, 所以给一点
  // 延时再定位; 依赖 focusSeq 而非 focusDate, 连点同一天也能再次触发。
  useEffect(() => {
    if (!focusDate) return;
    const t = setTimeout(() => {
      mainChartRef.current?.focusDate(focusDate);
      clearFocusDate();
    }, 260);
    return () => clearTimeout(t);
  }, [focusSeq, focusDate, clearFocusDate]);

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

  // 副图关闭按钮。三类副图各自的开关状态不同, 这里统一映射成 paneId → 关闭动作。
  // ⚠️ 主图(candle_pane)和成交量不给按钮 —— 关掉没意义。
  const panes = useMemo(() => {
    const out: { paneId: string; label: string; onClose: () => void }[] = [];
    for (const n of activeIndicators) {
      if (MAIN_PANE_INDICATORS.has(n)) continue;
      out.push({ paneId: `kc_${n}_pane`, label: n,
                 onClose: () => setActiveIndicators((p) => p.filter((x) => x !== n)) });
    }
    for (const n of tdx.active) {
      // 画在主图上的 TDX 指标 paneId 是 candle_pane, 不能给关闭按钮
      // (按钮会叠在主图右上角, 而且点了会误关整个主图区域的标注)。
      out.push({ paneId: `tdx_${n}_pane`, label: n, onClose: () => tdx.toggle(n) });
    }
    for (const n of activeTrained) {
      if (n !== "pump" && n !== "didian" && n !== "mmweek") continue;   // 副图类
      out.push({ paneId: `trained_${n}_pane`, label: n,
                 onClose: () => setActiveTrained((p) => p.filter((x) => x !== n)) });
    }
    return out;
  }, [activeIndicators, tdx.active, activeTrained]);

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
          chart.createIndicator(
            name === "MA" ? { name, calcParams: MA_PERIODS } : name,
            true, { id: "candle_pane" });
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
      // UTC 基准, 与后端 timestamp 一致(见 MainChart 的时区注释)
      const ts = Date.parse(`${tradeActions[idx].date}T00:00:00Z`);
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
              chart.createIndicator(
            name === "MA" ? { name, calcParams: MA_PERIODS } : name,
            true, { id: "candle_pane" });
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
        activeTrained={activeTrained}
        onToggleTrained={(n) =>
          setActiveTrained((prev) =>
            prev.includes(n) ? prev.filter((x) => x !== n) : [...prev, n]
          )
        }
        activeIndicators={activeIndicators}
        activeTdxIndicators={tdx.active}
        onToggleIndicator={handleToggleIndicator}
        onToggleTdxIndicator={tdx.toggle}
        onSelectOverlay={handleSelectOverlay}
        measuring={measuring}
        onToggleMeasure={() => setMeasuring((v) => !v)}
      />
      <div className="chart-info-row">
        <StockSectors symbol={currentSymbol} />
        <TopListBadge symbol={currentSymbol} onData={setLhb} />
      </div>
      {measuring && <MeasureBar result={measure} />}
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
          <MainChart
        comboSignals={activeTrained.includes("combo") ? comboSig : undefined}
        didianSignals={activeTrained.includes("didian") ? didianSig : undefined}
        pumpSignals={activeTrained.includes("pump") ? pumpSignals : undefined}
        signalMark={signalMark}
        liftSignals={activeTrained.includes("liftalert") ? liftSig : undefined}
        breakoutSignals={activeTrained.includes("breakout") ? boSig : undefined}
        sarSignals={activeTrained.includes("sar") ? sarSig : undefined}
        mmweekSignals={activeTrained.includes("mmweek") ? mmwkSig : undefined}
        panes={panes}
        topList={activeTrained.includes("toplist") ? lhb : undefined}
        maimaiSignals={activeTrained.includes("maimai_v3") ? maimaiSignals : undefined}
            ref={mainChartRef}
            tradeActions={marks}
            replayDate={replayDate}
            measuring={measuring}
            onMeasure={setMeasure}
          />
        )}
      </div>
    </div>
  );
}
