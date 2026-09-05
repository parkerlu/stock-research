// 板块 —— 列表 + 热图。看的是"今天哪个主题在动", 不做任何预测。
//
// 热度口径: 上涨占比 = 板块内上涨家数/有行情家数; 平均涨幅 = 成分股等权平均。
// 用等权而非市值加权 —— 市值加权会被一两只权重股主导, 掩盖主题的真实广度。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getSectors, getSectorMembers } from "../api/sectors";
import type { SectorItem, SectorMember } from "../types/sector";
import { useQuoteStore } from "../stores/quoteStore";
import { MainChart } from "./ChartArea/MainChart";
import { SectorTrend } from "./SectorTrend";
import { FavButton } from "./ChartArea/FavButton";
import type { MainChartHandle } from "./ChartArea/MainChart";
import { useTdxIndicators } from "../hooks/useTdxIndicators";
import { IndicatorMenus } from "./ChartArea/IndicatorMenus";
import { getMaimaiSignals, getPumpSignals, getDidianSignals, getComboSignals, getV5Signals, getMaimai35Signals } from "../api/quotes";

type SortKey = "avg_pct" | "up_ratio" | "count" | "max_pct";
// 盘后口径 —— 数据来自当日收盘, 不做轮询

function heatColor(pct: number | null): string {
  if (pct === null || Number.isNaN(pct)) return "rgba(120,125,135,0.18)";
  const v = Math.max(-6, Math.min(6, pct)) / 6;      // ±6% 饱和
  if (v >= 0) return `rgba(235, 84, 84, ${0.12 + v * 0.72})`;   // 红涨
  return `rgba(38, 166, 154, ${0.12 + -v * 0.72})`;              // 绿跌
}

export function SectorView() {
  const [sectors, setSectors] = useState<SectorItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("avg_pct");
  const [kw, setKw] = useState("");
  const [sel, setSel] = useState<SectorItem | null>(null);
  const [members, setMembers] = useState<SectorMember[]>([]);
  const [mLoading, setMLoading] = useState(false);
  // 点成分股 → 右侧看日/周/月三周期 K 线
  const [pick, setPick] = useState<{ code: string; name: string } | null>(null);
  // 单周期显示 + 按钮切换 —— 三张图挤在一起谁都看不清
  const [chartTf, setChartTf] = useState<"1d" | "1w" | "1m">("1d");
  const chartRef = useRef<MainChartHandle>(null);
  // 训练指标(买卖很准v3) —— 与行情/实时页同一套数据
  const [mmOn, setMmOn] = useState(false);
  const [pumpOn, setPumpOn] = useState(false);
  const [didianOn, setDidianOn] = useState(false);
  const [mm35Sig, setMm35Sig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [mm35On, setMm35On] = useState(false);
  const [v5Sig, setV5Sig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [v5On, setV5On] = useState(false);
  const [comboSig, setComboSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [comboOn, setComboOn] = useState(false);
  const [didianSig, setDidianSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string }[]
  >([]);
  const [pumpSig, setPumpSig] = useState<
    { date: string; prob: number; rank_pct: number; grade: string }[]
  >([]);
  const [mmSig, setMmSig] = useState<
    { date: string; score: number; rank_pct: number; grade: string; side?: "buy" | "sell" }[]
  >([]);
  useEffect(() => {
    if (!pick || !mmOn) {
      setMmSig([]);
      return;
    }
    let live = true;
    getMaimaiSignals(pick.code, "弱")
      .then((r) => live && setMmSig(r.signals ?? []))
      .catch(() => live && setMmSig([]));
    return () => {
      live = false;
    };
  }, [pick, mmOn]);

  useEffect(() => {
    if (!pick || !pumpOn) {
      setPumpSig([]);
      return;
    }
    let live = true;
    getPumpSignals(pick.code, "中")
      .then((r) => live && setPumpSig(r.signals ?? []))
      .catch(() => live && setPumpSig([]));
    return () => {
      live = false;
    };
  }, [pick, pumpOn]);

  useEffect(() => {
    if (!pick || !didianOn) {
      setDidianSig([]);
      return;
    }
    let live = true;
    getDidianSignals(pick.code, "中")
      .then((r) => live && setDidianSig(r.signals ?? []))
      .catch(() => live && setDidianSig([]));
    return () => {
      live = false;
    };
  }, [pick, didianOn]);

  useEffect(() => {
    if (!pick || !comboOn) {
      setComboSig([]);
      return;
    }
    let live = true;
    getComboSignals(pick.code)
      .then((r) => live && setComboSig(r.signals ?? []))
      .catch(() => live && setComboSig([]));
    return () => {
      live = false;
    };
  }, [pick, comboOn]);

  useEffect(() => {
    if (!pick || !v5On) {
      setV5Sig([]);
      return;
    }
    let live = true;
    getV5Signals(pick.code)
      .then((r) => live && setV5Sig(r.signals ?? []))
      .catch(() => live && setV5Sig([]));
    return () => {
      live = false;
    };
  }, [pick, v5On]);

  useEffect(() => {
    if (!pick || !mm35On) {
      setMm35Sig([]);
      return;
    }
    let live = true;
    getMaimai35Signals(pick.code)
      .then((r) => live && setMm35Sig(r.signals ?? []))
      .catch(() => live && setMm35Sig([]));
    return () => {
      live = false;
    };
  }, [pick, mm35On]);
  const tdx = useTdxIndicators({
    getChart: () => chartRef.current?.getChart() ?? null,
    symbol: pick?.code ?? "",
    timeframe: chartTf,
    storageKey: "sector.tdx.v1",
  });
  const [tradeDate, setTradeDate] = useState<string | null>(null);
  const [coverage, setCoverage] = useState<number | null>(null);
  const alive = useRef(true);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);

  const pull = useCallback(async () => {
    try {
      const r = await getSectors();
      if (!alive.current) return;
      setSectors(r.sectors);
      setTradeDate(r.trade_date ?? null);
      setCoverage(r.coverage ?? null);
      setErr(null);
    } catch (e: unknown) {
      if (alive.current) setErr(e instanceof Error ? e.message : "板块加载失败");
    } finally {
      if (alive.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    setLoading(true);
    pull();
    return () => {
      alive.current = false;
    };
  }, [pull]);

  const openSector = useCallback(async (s: SectorItem) => {
    setSel(s);
    setMLoading(true);
    try {
      const r = await getSectorMembers(s.ts_code);
      setMembers(r.members);
    } catch {
      setMembers([]);
    } finally {
      setMLoading(false);
    }
  }, []);

  const view = useMemo(() => {
    const k = kw.trim();
    let list = k ? sectors.filter((s) => s.name.includes(k)) : sectors;
    list = [...list].sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (av === null) return 1;
      if (bv === null) return -1;
      return (bv as number) - (av as number);
    });
    return list;
  }, [sectors, kw, sortKey]);

  const stat = useMemo(() => {
    const withData = sectors.filter((s) => s.avg_pct !== null);
    const up = withData.filter((s) => (s.avg_pct ?? 0) > 0).length;
    return { total: withData.length, up, down: withData.length - up };
  }, [sectors]);

  return (
    <div className="sector-view">
      <div className="sector-head">
        <input
          className="sector-search"
          placeholder="搜板块名…"
          value={kw}
          onChange={(e) => setKw(e.target.value)}
        />
        <div className="sector-sort">
          {([
            ["avg_pct", "平均涨幅"],
            ["up_ratio", "上涨占比"],
            ["max_pct", "最强个股"],
            ["count", "成分数"],
          ] as [SortKey, string][]).map(([k, label]) => (
            <button
              key={k}
              className={sortKey === k ? "on" : ""}
              onClick={() => setSortKey(k)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="sector-meta">
          {stat.total > 0 && (
            <span>
              <b className="up">{stat.up}</b> 涨 / <b className="down">{stat.down}</b> 跌
              &nbsp;·&nbsp; {view.length} 个板块
            </span>
          )}
          {tradeDate && <span className="dim">{tradeDate} 收盘</span>}
          {coverage !== null && coverage < 80 && (
            <span className="cov-warn" title="当日K线尚未入库完整, 板块均值会失真">
              数据仅 {coverage.toFixed(0)}%
            </span>
          )}
        </div>
      </div>

      {err && <div className="sector-err">{err}</div>}
      {loading && sectors.length === 0 && <div className="sector-empty">加载中…</div>}

      <div className="sector-body">
        <div className="sector-heat">
          {view.map((s) => (
            <button
              key={s.ts_code}
              className={`heat-cell${sel?.ts_code === s.ts_code ? " sel" : ""}`}
              style={{ background: heatColor(s.avg_pct) }}
              onClick={() => openSector(s)}
              title={`${s.name}\n平均 ${s.avg_pct ?? "-"}%  中位 ${s.median_pct ?? "-"}%\n上涨 ${s.up}/${s.quoted} (${s.up_ratio ?? "-"}%)\n最强 ${s.max_pct ?? "-"}%  最弱 ${s.min_pct ?? "-"}%`}
            >
              <span className="heat-name">{s.name}</span>
              <span className="heat-pct">
                {s.avg_pct === null ? "—" : `${s.avg_pct > 0 ? "+" : ""}${s.avg_pct.toFixed(2)}%`}
              </span>
              <span className="heat-ratio">
                {s.up_ratio === null ? "" : `${s.up}/${s.quoted}`}
              </span>
            </button>
          ))}
        </div>

        {sel && (
          <div className="sector-detail">
            <div className="sd-head">
              <div>
                <span className="sd-name">{sel.name}</span>
                <span className="sd-code">{sel.ts_code}</span>
              </div>
              <button className="sd-close" onClick={() => setSel(null)}>×</button>
            </div>
            <div className="sd-stat">
              <span>平均 <b className={(sel.avg_pct ?? 0) >= 0 ? "up" : "down"}>
                {sel.avg_pct?.toFixed(2) ?? "-"}%</b></span>
              <span>上涨 <b>{sel.up}</b>/{sel.quoted}
                {sel.up_ratio !== null && ` (${sel.up_ratio.toFixed(0)}%)`}</span>
              <span>最强 <b className="up">{sel.max_pct?.toFixed(2) ?? "-"}%</b></span>
              <span>最弱 <b className="down">{sel.min_pct?.toFixed(2) ?? "-"}%</b></span>
              {sel.list_date && <span className="dim">成立 {sel.list_date}</span>}
            </div>
            <SectorTrend sectorCode={sel.ts_code} days={30} />
            <div className="sd-list">
              {mLoading ? (
                <div className="sector-empty">成分股加载中…</div>
              ) : (
                members.map((m) => (
                  <button
                    key={m.ts_code}
                    className="sd-row"
                    onClick={() => {
                      setCurrentStock(m.ts_code, m.name);
                      setPick({ code: m.ts_code, name: m.name });
                    }}
                    title="点击查看该股日/周/月K线"
                  >
                    <span className="sd-mname">{m.name}</span>
                    <span className="sd-mcode">{m.ts_code.split(".")[0]}</span>
                    <span className="sd-mpx">{m.price?.toFixed(2) ?? "—"}</span>
                    <span className={`sd-mpct ${(m.pct ?? 0) >= 0 ? "up" : "down"}`}>
                      {m.pct === null ? "—" : `${m.pct > 0 ? "+" : ""}${m.pct.toFixed(2)}%`}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        )}

        {pick && (
          <div className="sector-charts">
            <div className="sc-head">
              <span className="sc-name">{pick.name}</span>
              <span className="sc-code">{pick.code}</span>
              <FavButton symbol={pick.code} />
              <div className="sc-tf">
                {([["1d", "日"], ["1w", "周"], ["1m", "月"]] as const).map(([v, lb]) => (
                  <button
                    key={v}
                    className={chartTf === v ? "on" : ""}
                    onClick={() => setChartTf(v)}
                  >
                    {lb}
                  </button>
                ))}
              </div>
              <div className="sc-ind">
                <IndicatorMenus
                  tdxAvailable={tdx.available}
                  tdxActive={tdx.active}
                  onToggleTdx={tdx.toggle}
                  tdxBusy={tdx.busy}
                  trainedActive={[...(mmOn ? ["maimai_v3"] : []), ...(pumpOn ? ["pump"] : []),
                                  ...(didianOn ? ["didian"] : []), ...(comboOn ? ["combo"] : []), ...(v5On ? ["v5"] : []), ...(mm35On ? ["maimai35"] : [])]}
                  onToggleTrained={(n) =>
                    n === "pump" ? setPumpOn((v) => !v)
                    : n === "didian" ? setDidianOn((v) => !v)
                    : n === "maimai35" ? setMm35On((v) => !v)
                : n === "v5" ? setV5On((v) => !v)
                : n === "combo" ? setComboOn((v) => !v)
                    : setMmOn((v) => !v)
                  }
                  trainedCounts={{ maimai_v3: mmSig.length, pump: pumpSig.length,
                                   didian: didianSig.length,
                               combo: comboSig.length, v5: v5Sig.length, maimai35: mm35Sig.length }}
                />
              </div>
              <button className="sd-close" onClick={() => setPick(null)}>×</button>
            </div>
            <div className="sc-body">
              <MainChart ref={chartRef} timeframe={chartTf} className="sc-chart"
                         maimaiSignals={mmOn ? mmSig : undefined}
                         pumpSignals={pumpOn ? pumpSig : undefined}
                         didianSignals={didianOn ? didianSig : undefined}
              comboSignals={comboOn ? comboSig : undefined}
              v5Signals={v5On ? v5Sig : undefined}
              maimai35Signals={mm35On ? mm35Sig : undefined} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
