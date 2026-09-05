// 实时看盘 — 左分时 / 右日K, 开市时自动刷新。
//
// 刷新策略: 只轮询 /minute 一个接口 (它同时返回分时明细 + 实时快照 +
// 上证开闭市状态)。收市后自动停止轮询, 不空转。
// 日K 的最后一根由快照实时打补丁 (今开/最高/最低/现价/成交量),
// 这样盘中日K末根跟着走, 不必等收盘入库。
import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import { getMinute, getT0Signals, getMaimaiSignals, getPumpSignals, getDidianSignals, getComboSignals, getMaimai35Signals } from "../../api/quotes";
import { syncSymbol } from "../../api/system";
import type { MinuteData, T0Trade, Timeframe } from "../../types/quote";
import { MinuteChart } from "./MinuteChart";
import { OrderBook } from "./OrderBook";
import { useTdxIndicators } from "../../hooks/useTdxIndicators";
import { MainChart } from "./MainChart";
import { IndicatorMenus } from "./IndicatorMenus";
import { StockSectors } from "./StockSectors";
import { FavButton } from "./FavButton";
import type { MainChartHandle } from "./MainChart";

const POLL_MS = 5000;        // 开市中
const IDLE_POLL_MS = 60000;  // 收市后偶尔探一次, 以便开市自动恢复

const SPLIT_KEY = "live.split";
const SPLIT_DEFAULT = 0.5;
const SPLIT_MIN = 0.2;
const SPLIT_MAX = 0.8;

const clampSplit = (v: number) => Math.min(SPLIT_MAX, Math.max(SPLIT_MIN, v));

export function LiveView() {
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const currentName = useQuoteStore((s) => s.currentName);

  const [minute, setMinute] = useState<MinuteData | null>(null);
  // 做 T 信号 — 5min bar 才变一次, 单独低频轮询, 不跟着 5s 的分时刷
  const [signals, setSignals] = useState<T0Trade[]>([]);
  const [t0On, setT0On] = useState<boolean>(
    () => localStorage.getItem("live.t0") === "1"
  );
  // 阈值: 高 = 信号少但准 (0.7 精确率约 68%), 低 = 信号多 (0.5 约 52%)
  // 买卖很准 v3 —— K线上的买点标记
  const [mmSignals, setMmSignals] = useState<
    { date: string; score: number; rank_pct: number; grade: string; side?: "buy" | "sell" }[]
  >([]);
  const [mm35Sig, setMm35Sig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [mm35On, setMm35On] = useState(false);
  const [comboSig, setComboSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [comboOn, setComboOn] = useState(false);
  const [didianSig, setDidianSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [didianOn, setDidianOn] = useState<boolean>(
    () => localStorage.getItem("live.didian") === "1"
  );
  const [pumpSig, setPumpSig] = useState<
    { date: string; prob: number; rank_pct: number; grade: string }[]
  >([]);
  const [pumpOn, setPumpOn] = useState<boolean>(
    () => localStorage.getItem("live.pump") === "1"
  );
  const [mmOn, setMmOn] = useState<boolean>(
    () => localStorage.getItem("live.mm") === "1"
  );
  const [t0Th, setT0Th] = useState<number>(() => {
    const v = parseFloat(localStorage.getItem("live.t0th") ?? "");
    return [0.5, 0.6, 0.7].includes(v) ? v : 0.6;
  });
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  const dailyRef = useRef<MainChartHandle>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const alive = useRef(true);

  // 左右分栏比例 —— 拖分割线调整, 记住上次
  const panesRef = useRef<HTMLDivElement>(null);
  const [split, setSplit] = useState<number>(() => {
    const v = parseFloat(localStorage.getItem(SPLIT_KEY) ?? "");
    return Number.isFinite(v) ? clampSplit(v) : SPLIT_DEFAULT;
  });
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    localStorage.setItem(SPLIT_KEY, String(split));
  }, [split]);

  const startDrag = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const el = panesRef.current;
    if (!el) return;
    // 窄屏下 .live-panes 变成上下排列, 拖拽轴随之改为纵向
    const vertical = getComputedStyle(el).flexDirection === "column";
    setDragging(true);

    const onMove = (ev: MouseEvent) => {
      const r = el.getBoundingClientRect();
      const ratio = vertical
        ? (ev.clientY - r.top) / r.height
        : (ev.clientX - r.left) / r.width;
      setSplit(clampSplit(ratio));
    };
    const onUp = () => {
      setDragging(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, []);

  // 右侧K线周期 (日/周/月), 记住上次选择
  const [tf, setTf] = useState<Timeframe>(() => {
    const v = localStorage.getItem("live.tf");
    return v === "1w" || v === "1m" ? v : "1d";
  });
  useEffect(() => {
    localStorage.setItem("live.tf", tf);
  }, [tf]);
  // tick 是稳定回调, 用 ref 读当前周期避免把 tf 塞进它的依赖里重启轮询
  const tfRef = useRef(tf);
  useEffect(() => {
    tfRef.current = tf;
  }, [tf]);

  // 右侧K线的 TDX 指标 —— 独立 storageKey, 与主图表页互不干扰;
  // 周期变化时 hook 会自动按新周期重新取数
  const tdx = useTdxIndicators({
    getChart: () => dailyRef.current?.getChart() ?? null,
    symbol: currentSymbol,
    timeframe: tf,
    storageKey: "live.tdxIndicators.v1",
  });

  // 做 T 信号轮询 (60s)。收市后也拉一次 — 复盘时能看到当天全部信号点。
  useEffect(() => {
    if (!currentSymbol || !t0On) {
      setSignals([]);
      return;
    }
    let live = true;
    const pull = async () => {
      try {
        const r = await getT0Signals(currentSymbol, t0Th);
        if (live) setSignals(r.signals ?? []);
      } catch {
        if (live) setSignals([]);
      }
    };
    pull();
    const id = setInterval(pull, 60000);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [currentSymbol, t0On, t0Th]);

  useEffect(() => {
    localStorage.setItem("live.t0", t0On ? "1" : "0");
  }, [t0On]);

  useEffect(() => {
    localStorage.setItem("live.t0th", String(t0Th));
  }, [t0Th]);

  // 买卖很准信号 —— 盘后数据, 切股票时拉一次即可
  useEffect(() => {
    localStorage.setItem("live.mm", mmOn ? "1" : "0");
    if (!currentSymbol || !mmOn) {
      setMmSignals([]);
      return;
    }
    let live = true;
    getMaimaiSignals(currentSymbol, "弱")
      .then((r) => live && setMmSignals(r.signals ?? []))
      .catch(() => live && setMmSignals([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, mmOn]);

  useEffect(() => {
    localStorage.setItem("live.pump", pumpOn ? "1" : "0");
    if (!currentSymbol || !pumpOn) {
      setPumpSig([]);
      return;
    }
    let live = true;
    getPumpSignals(currentSymbol, "中")
      .then((r) => live && setPumpSig(r.signals ?? []))
      .catch(() => live && setPumpSig([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, pumpOn]);

  useEffect(() => {
    localStorage.setItem("live.didian", didianOn ? "1" : "0");
    if (!currentSymbol || !didianOn) {
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
  }, [currentSymbol, didianOn]);

  useEffect(() => {
    if (!currentSymbol || !comboOn) {
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
  }, [currentSymbol, comboOn]);


  useEffect(() => {
    if (!currentSymbol || !mm35On) {
      setMm35Sig([]);
      return;
    }
    let live = true;
    getMaimai35Signals(currentSymbol)
      .then((r) => live && setMm35Sig(r.signals ?? []))
      .catch(() => live && setMm35Sig([]));
    return () => {
      live = false;
    };
  }, [currentSymbol, mm35On]);

  const tick = useCallback(async () => {
    if (!currentSymbol) return;
    try {
      const d = await getMinute(currentSymbol);
      if (!alive.current) return;
      setMinute(d);
      setErr(null);
      setUpdatedAt(new Date().toLocaleTimeString("zh-CN", { hour12: false }));

      // 用快照给日K最后一根打补丁 —— 盘中也能看到今天的实时K线。
      // 仅日线适用: 周/月线的末根时间戳是周初/月初, 用当日时间戳会多插一根。
      const snap = d.snapshot;
      const day = d.trade_date; // "20260828"
      if (tfRef.current === "1d" && snap && snap.price > 0 && day && day.length === 8) {
        const ts = new Date(
          `${day.slice(0, 4)}-${day.slice(4, 6)}-${day.slice(6, 8)}T00:00:00`
        ).getTime();
        dailyRef.current?.pushBar({
          timestamp: ts,
          open: snap.open || snap.price,
          high: snap.high || snap.price,
          low: snap.low || snap.price,
          close: snap.price,
          volume: snap.vol,
          turnover: snap.amount,
        });
      }
      return d.market_open;
    } catch (e: unknown) {
      if (!alive.current) return;
      setErr(e instanceof Error ? e.message : "分时获取失败");
      return false;
    }
  }, [currentSymbol]);

  // 自增式轮询: 开市 5s, 收市 60s。用 setTimeout 链而非 setInterval,
  // 避免请求慢时堆积。
  useEffect(() => {
    alive.current = true;
    if (!currentSymbol) {
      setMinute(null);
      return;
    }
    setLoading(true);
    let stopped = false;

    const loop = async () => {
      const open = await tick();
      if (!alive.current || stopped) return;
      setLoading(false);
      timer.current = setTimeout(loop, open ? POLL_MS : IDLE_POLL_MS);
    };
    loop();

    return () => {
      stopped = true;
      alive.current = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [currentSymbol, tick]);

  // 切股时按需补齐该标的的日线 —— 全量 backfill 要跑几千只, 只看一只不该等它。
  // 周/月K 由日线聚合, 补日线即三个周期一起补齐。补到新数据才重拉K线, 否则白闪。
  // 结果按标的存, 这样"补齐中…"是渲染时算出来的, 不必在 effect 里同步 setState
  const [syncDone, setSyncDone] = useState<{ symbol: string; text: string | null } | null>(null);
  const syncMsg = syncDone?.symbol === currentSymbol ? syncDone.text : "补齐中…";

  useEffect(() => {
    if (!currentSymbol) return;
    let cancelled = false;
    syncSymbol(currentSymbol)
      .then((r) => {
        if (cancelled) return;
        if (r.inserted > 0) dailyRef.current?.reload();
        setSyncDone({
          symbol: currentSymbol,
          text: r.inserted > 0 ? `已补 ${r.inserted} 根` : null,
        });
      })
      .catch(() => {
        // 补齐失败不影响看盘 —— 图上仍是本地已有的数据
        if (!cancelled) setSyncDone({ symbol: currentSymbol, text: "补齐失败" });
      });
    return () => {
      cancelled = true;
    };
  }, [currentSymbol]);

  const snap = minute?.snapshot ?? null;
  const up = (snap?.change_pct ?? 0) >= 0;
  const color = up ? "#eb5454" : "#26a69a";

  if (!currentSymbol) {
    return <div className="live-empty">请先搜索并选择一只股票</div>;
  }

  return (
    <div className="live-view">
      {/* 顶部实时报价条 */}
      <div className="live-header">
        <div className="live-title">
          <span className="live-name">{snap?.name || currentName}</span>
          <span className="live-code">{currentSymbol}</span>
          <FavButton symbol={currentSymbol} />
        </div>
        {snap && (
          <div className="live-quote">
            <span className="live-price" style={{ color }}>
              {snap.price.toFixed(2)}
            </span>
            <span style={{ color }}>
              {up ? "+" : ""}{snap.change.toFixed(2)}
            </span>
            <span style={{ color }}>
              {up ? "+" : ""}{snap.change_pct.toFixed(2)}%
            </span>
            <span className="live-dim">昨收 {(snap.prev_close ?? 0).toFixed(2)}</span>
            <span className="live-dim">开 {snap.open.toFixed(2)}</span>
            <span className="live-dim">高 {snap.high.toFixed(2)}</span>
            <span className="live-dim">低 {snap.low.toFixed(2)}</span>
            <span className="live-dim">换手 {snap.turnover.toFixed(2)}%</span>
            <span className="live-dim">
              额 {(snap.amount / 1e8).toFixed(2)}亿
            </span>
          </div>
        )}
        <div className="live-status">
          {minute?.market_open ? (
            <span className="live-badge live-on">● 交易中 · 5秒刷新</span>
          ) : (
            <span className="live-badge live-off">● 已收市</span>
          )}
          {updatedAt && <span className="live-dim">{updatedAt}</span>}
          {syncMsg && <span className="live-dim">{syncMsg}</span>}
          {err && <span className="live-err">{err}</span>}
        </div>
      </div>

      <StockSectors symbol={currentSymbol} />

      {/* 左分时 / 右日K, 中间可拖动的分割线 */}
      <div
        ref={panesRef}
        className={`live-panes${dragging ? " dragging" : ""}`}
        style={{ "--live-split": `${(split * 100).toFixed(2)}%` } as CSSProperties}
      >
        <div className="live-pane live-pane-first">
          <div className="live-pane-title">
            <span>分时</span>
            <span className="t0-bar">
              {t0On && (
                <span className="t0-th">
                  {[0.5, 0.6, 0.7].map((v) => (
                    <button
                      key={v}
                      type="button"
                      className={t0Th === v ? "on" : ""}
                      onClick={() => setT0Th(v)}
                      title={
                        v === 0.7 ? "少而准 — 样本外精确率约 68%"
                        : v === 0.6 ? "均衡 — 约 60%"
                        : "多而糙 — 约 52%"
                      }
                    >
                      {v.toFixed(1)}
                    </button>
                  ))}
                </span>
              )}
              <button
                type="button"
                className={`t0-toggle${t0On ? " on" : ""}`}
                onClick={() => setT0On((v) => !v)}
                title="做T指标 — 标出到收盘还有 3% 空间的买卖点"
              >
                做T{t0On && signals.length > 0 ? ` ${signals.length}` : ""}
              </button>
            </span>
          </div>
          <div className="live-pane-split">
            <div className="live-pane-body">
              <MinuteChart data={minute} loading={loading} signals={t0On ? signals : []} />
            </div>
            <div className="live-orderbook">
              <div className="live-pane-title">五档盘口</div>
              <OrderBook snapshot={snap} />
            </div>
          </div>
        </div>
        <div
          className="live-splitter"
          role="separator"
          title="拖动调整左右宽度, 双击复位"
          onMouseDown={startDrag}
          onDoubleClick={() => setSplit(SPLIT_DEFAULT)}
        />
        <div className="live-pane">
          <div className="live-pane-title live-pane-title-row">
            <div className="live-tf">
              {([["1d", "日"], ["1w", "周"], ["1m", "月"]] as const).map(
                ([v, label]) => (
                  <button
                    key={v}
                    className={tf === v ? "active" : ""}
                    onClick={() => setTf(v)}
                  >
                    {label}
                  </button>
                )
              )}
            </div>
            <IndicatorMenus
              tdxAvailable={tdx.available}
              tdxActive={tdx.active}
              onToggleTdx={tdx.toggle}
              tdxBusy={tdx.busy}
              trainedActive={[...(mmOn ? ["maimai_v3"] : []), ...(pumpOn ? ["pump"] : []),
                              ...(didianOn ? ["didian"] : []), ...(comboOn ? ["combo"] : []), ...(mm35On ? ["maimai35"] : [])]}
              onToggleTrained={(n) =>
                n === "pump" ? setPumpOn((v) => !v)
                : n === "didian" ? setDidianOn((v) => !v)
                : n === "maimai35" ? setMm35On((v) => !v)
                : n === "combo" ? setComboOn((v) => !v)
                : setMmOn((v) => !v)
              }
              trainedCounts={{ maimai_v3: mmSignals.length, pump: pumpSig.length,
                               didian: didianSig.length,
                               combo: comboSig.length, maimai35: mm35Sig.length }}
            />
          </div>
          <div className="live-pane-body">
            <MainChart
              maimaiSignals={mmOn ? mmSignals : undefined}
              pumpSignals={pumpOn ? pumpSig : undefined}
              didianSignals={didianOn ? didianSig : undefined}
              comboSignals={comboOn ? comboSig : undefined}
              maimai35Signals={mm35On ? mm35Sig : undefined} ref={dailyRef} timeframe={tf} className="live-daily" />
          </div>
        </div>
      </div>
    </div>
  );
}
