// 虚拟盘 — chan-2buy 最优配置的实盘跟踪。
// 三块: 账户总览 / 当前持仓(含浮盈与三档价位) / 全部动作流水。
import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEMO_ACCOUNT,
  LIVE_ACCOUNT,
  getPaperConfig,
  getPaperEquity,
  getPaperRun,
  getPaperSignals,
  getPaperStatus,
  getPaperTrades,
  resetPaper,
  runPaper,
  stepPaper,
} from "../../api/paper";
import type {
  EquityPoint,
  PaperConfig,
  PaperStatus,
  PaperTrade,
  SignalPick,
} from "../../api/paper";
import { useQuoteStore } from "../../stores/quoteStore";

const ACTION_LABEL: Record<string, string> = {
  buy: "买入",
  tier1: "止盈½",
  tier2: "清仓",
  stop: "止损",
  timeout: "到期",
  close: "平仓",
};

function money(v: number) {
  return v.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** 净值曲线 — 数据点少，手绘 SVG 比引库划算 */
function EquityChart({ points, initial }: { points: EquityPoint[]; initial: number }) {
  if (points.length < 2) return null;
  const W = 560, H = 90, P = 4;
  const vals = points.map((p) => p.equity);
  const lo = Math.min(initial, ...vals), hi = Math.max(initial, ...vals);
  const span = hi - lo || 1;
  const x = (i: number) => P + (i / (points.length - 1)) * (W - 2 * P);
  const y = (v: number) => P + (1 - (v - lo) / span) * (H - 2 * P);
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(p.equity).toFixed(1)}`).join("");
  const up = vals[vals.length - 1] >= initial;
  const col = up ? "#e94560" : "#4caf50";
  return (
    <svg className="pp-eq" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <line x1={P} x2={W - P} y1={y(initial)} y2={y(initial)}
            stroke="#5c6675" strokeDasharray="3 3" strokeWidth="1" />
      <path d={`${d} L${x(points.length - 1)} ${H - P} L${x(0)} ${H - P} Z`}
            fill={col} opacity="0.10" />
      <path d={d} fill="none" stroke={col} strokeWidth="1.8" />
    </svg>
  );
}

export function PaperPanel() {
  const [cfg, setCfg] = useState<PaperConfig | null>(null);
  const [st, setSt] = useState<PaperStatus | null>(null);
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [eq, setEq] = useState<{ initial: number; points: EquityPoint[] } | null>(null);
  const [tab, setTab] = useState<"holdings" | "trades" | "closed">("holdings");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showRules, setShowRules] = useState(false);
  const [acct, setAcct] = useState<string>(LIVE_ACCOUNT);
  const [signals, setSignals] = useState<SignalPick[]>([]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600);
  const [startDate, setStartDate] = useState("2020-01-02");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const playRef = useRef(false);
  const isDemo = acct === DEMO_ACCOUNT;
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);

  const reload = useCallback(async (name = acct) => {
    try {
      const [s, t, e, g] = await Promise.all([
        getPaperStatus(name), getPaperTrades(name, 200), getPaperEquity(name),
        getPaperSignals(name, 20).catch(() => ({ picks: [] as SignalPick[] })),
      ]);
      setSt(s);
      setTrades(t.trades);
      setSignals(g.picks ?? []);
      if (e.exists) setEq({ initial: e.initial, points: e.points });
      setErr(null);
    } catch (x) {
      setErr(x instanceof Error ? x.message : "加载失败");
    }
  }, [acct]);

  /** 演示盘: 走一个交易日 */
  const stepOnce = useCallback(async () => {
    try {
      const r = await stepPaper(acct, 1);
      await reload(acct);
      if (r.done) { playRef.current = false; setPlaying(false); }
      return !r.done;
    } catch (x) {
      playRef.current = false; setPlaying(false);
      setErr(x instanceof Error ? x.message : "推进失败");
      return false;
    }
  }, [acct, reload]);

  // 自动播放 —— 用 ref 控制, 避免 setInterval 在异步推进未完成时叠加
  useEffect(() => {
    if (!playing) return;
    playRef.current = true;
    let stop = false;
    const loop = async () => {
      while (playRef.current && !stop) {
        const ok = await stepOnce();
        if (!ok) break;
        await new Promise((r) => setTimeout(r, speed));
      }
    };
    void loop();
    return () => { stop = true; playRef.current = false; };
  }, [playing, speed, stepOnce]);

  useEffect(() => {
    getPaperConfig().then(setCfg).catch(() => setCfg(null));
    reload(acct);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [reload]);

  const poll = (id: string) => {
    getPaperRun(id).then((j) => {
      if (j.status === "pending" || j.status === "running") {
        timer.current = setTimeout(() => poll(id), 2000);
      } else {
        setBusy(false);
        if (j.status === "error") setErr(j.error ?? "推进失败");
        reload();
      }
    }).catch(() => setBusy(false));
  };

  const advance = async () => {
    setBusy(true);
    setErr(null);
    try {
      const { job_id } = await runPaper();
      poll(job_id);
    } catch (x) {
      setBusy(false);
      setErr(x instanceof Error ? x.message : "启动失败");
    }
  };

  if (!st) return <div className="paper-panel"><div className="screening-empty">加载中…</div></div>;
  if (!st.exists) {
    return (
      <div className="paper-panel">
        <div className="screening-empty">虚拟盘尚未建立</div>
      </div>
    );
  }

  const up = st.total_pnl >= 0;

  return (
    <div className="paper-panel">
      {/* ---- 总览 ---- */}
      <div className="pp-head">
        <div className="pp-title">
          <span>💼 虚拟盘</span>
          <div className="pp-switch">
            <button className={!isDemo ? "on" : ""}
                    onClick={() => { setPlaying(false); setAcct(LIVE_ACCOUNT); reload(LIVE_ACCOUNT); }}>
              实操盘
            </button>
            <button className={isDemo ? "on" : ""}
                    onClick={() => { setAcct(DEMO_ACCOUNT); reload(DEMO_ACCOUNT); }}>
              演示回放
            </button>
          </div>
          {cfg && (
            <button className="pp-rules-btn" onClick={() => setShowRules((v) => !v)}>
              规则 {showRules ? "▴" : "▾"}
            </button>
          )}
        </div>
        <div className="pp-actions">
          {isDemo ? (
            <span className="pp-asof">
              从
              <input type="date" className="pp-date" value={startDate}
                     onChange={(e) => setStartDate(e.target.value)} />
              <button className="pp-reset-btn" title="清空并从这一天重新开始 (10 万本金 / 10 仓位)"
                      onClick={async () => {
                        setPlaying(false);
                        await resetPaper(DEMO_ACCOUNT, startDate);
                        await reload(DEMO_ACCOUNT);
                      }}>重新开始</button>
              <span className="pp-cur">当前 {st.last_run_date ?? "未开始"}</span>
            </span>
          ) : (
            <span className="pp-asof">
              {st.started_on} 起 · 当前 {st.last_run_date ?? "未开始"}
            </span>
          )}
          {isDemo ? (
            <>
              <button className="pp-run" onClick={() => void stepOnce()} disabled={playing}>
                下一日 ▸
              </button>
              <button className={`pp-run pp-play${playing ? " on" : ""}`}
                      onClick={() => setPlaying((v) => !v)}>
                {playing ? "⏸ 暂停" : "▶ 自动"}
              </button>
              <select className="pp-speed" value={speed}
                      onChange={(e) => setSpeed(Number(e.target.value))}>
                <option value={1200}>慢</option>
                <option value={600}>中</option>
                <option value={150}>快</option>
              </select>
            </>
          ) : (
            <button className="pp-run" onClick={advance} disabled={busy}>
              {busy ? "推进中…" : "推进到最新"}
            </button>
          )}
        </div>
      </div>

      {showRules && cfg && (
        <div className="pp-rules">
          <div><b>策略</b> {cfg.strategy}</div>
          <div>
            <b>规则</b> 本金 {money(st.initial_capital)} · {st.slots} 个等权仓位 ·
            止损 −{(Number(cfg.rules.stop_pct) * 100).toFixed(0)}% ·
            +{(Number(cfg.rules.tier1_pct) * 100).toFixed(0)}% 卖半(止损上移保本) ·
            +{(Number(cfg.rules.tier2_pct) * 100).toFixed(0)}% 清仓 · 低流动性优先
          </div>
          <div><b>回测</b> {cfg.backtest.period}，年化 {cfg.backtest.cagr}%、
            最大回撤 {cfg.backtest.max_drawdown}%、夏普 {cfg.backtest.sharpe}</div>
          <div className="pp-warn">⚠️ {cfg.backtest.note}</div>
        </div>
      )}

      <div className="pp-stats">
        <div className="pp-stat">
          <div className={`v ${up ? "up" : "down"}`}>{money(st.equity)}</div>
          <div className="l">总净值</div>
        </div>
        <div className="pp-stat">
          <div className={`v ${up ? "up" : "down"}`}>
            {up ? "+" : ""}{money(st.total_pnl)}
          </div>
          <div className="l">总盈亏</div>
        </div>
        <div className="pp-stat">
          <div className={`v ${up ? "up" : "down"}`}>
            {up ? "+" : ""}{st.total_pnl_pct.toFixed(2)}%
          </div>
          <div className="l">收益率</div>
        </div>
        <div className="pp-stat">
          <div className="v">{money(st.cash)}</div>
          <div className="l">可用现金</div>
        </div>
        <div className="pp-stat">
          <div className="v">{st.n_open}/{st.slots}</div>
          <div className="l">持仓/仓位</div>
        </div>
      </div>

      {eq && eq.points.length > 1 && (
        <EquityChart points={eq.points} initial={eq.initial} />
      )}
      <div className="pp-split">
        <span>已实现 <b className={st.realized_pnl >= 0 ? "up" : "down"}>
          {st.realized_pnl >= 0 ? "+" : ""}{money(st.realized_pnl)}</b></span>
        <span>浮动 <b className={st.float_pnl >= 0 ? "up" : "down"}>
          {st.float_pnl >= 0 ? "+" : ""}{money(st.float_pnl)}</b></span>
        <span>已平仓 {st.n_closed} 笔</span>
      </div>

      {err && <div className="screening-error">{err}</div>}

      {signals.length > 0 && (
        <div className="pp-signals">
          <div className="pp-signals-t">
            当日买点 {signals.length} 只 · 空仓位 {Math.max(0, st.slots - st.n_open)} 个
            {" → 下一日最多买入 "}
            {Math.min(signals.length, Math.max(0, st.slots - st.n_open))} 只
            <span className="pp-signals-hint">（仅候选，尚未买入；空仓位不足就按流动性从低到高挑，宁可空着也不加仓）</span>
          </div>
          <div className="pp-signals-list">
            {signals.map((p) => (
              <span key={p.ts_code}
                    className={`pp-sig${p.held ? " held" : ""}`}
                    onClick={() => setCurrentStock(p.ts_code, p.name ?? p.ts_code)}
                    title={`${p.amount_20d_wan} 万 · 次日开盘 ${p.next_open}`}>
                {p.name ?? p.ts_code}{p.held && " ✓"}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="pp-tabs">
        <button className={tab === "holdings" ? "active" : ""} onClick={() => setTab("holdings")}>
          持仓 {st.n_open}
        </button>
        <button className={tab === "trades" ? "active" : ""} onClick={() => setTab("trades")}>
          流水 {trades.length}
        </button>
        <button className={tab === "closed" ? "active" : ""} onClick={() => setTab("closed")}>
          已平仓 {st.n_closed}
        </button>
      </div>

      {tab === "holdings" && (
        <div className="pp-table-wrap">
          <table className="pp-table">
            <thead>
              <tr>
                <th>股票</th><th>买入</th><th>买价</th><th>现价</th><th>股数</th>
                <th>浮动盈亏</th><th>止损</th><th>止盈</th>
              </tr>
            </thead>
            <tbody>
              {st.holdings.map((h) => (
                <tr key={h.ts_code} onClick={() => setCurrentStock(h.ts_code, h.name ?? h.ts_code)}>
                  <td className="pp-name">
                    <b>{h.name ?? ""}</b><em>{h.ts_code}</em>
                  </td>
                  <td>{h.open_date.slice(5)}<em>{h.hold_days}天</em></td>
                  <td>{h.open_price.toFixed(2)}</td>
                  <td>{h.last_price.toFixed(2)}</td>
                  <td>{h.shares}{h.tier1_done && <em>已减半</em>}</td>
                  <td className={h.float_pnl >= 0 ? "up" : "down"}>
                    {h.float_pnl >= 0 ? "+" : ""}{money(h.float_pnl)}
                    <em>{h.float_pnl_pct >= 0 ? "+" : ""}{h.float_pnl_pct.toFixed(2)}%</em>
                  </td>
                  <td className="down">{h.stop_price.toFixed(2)}</td>
                  <td className="up">
                    {h.tier1_done ? h.tier2_price.toFixed(2) : h.tier1_price.toFixed(2)}
                  </td>
                </tr>
              ))}
              {st.holdings.length === 0 && (
                <tr><td colSpan={8} className="pp-empty">当前空仓</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {tab === "trades" && (
        <div className="pp-table-wrap pp-scroll">
          <table className="pp-table">
            <thead>
              <tr><th>日期</th><th>股票</th><th>动作</th><th>价格</th><th>股数</th><th>金额</th><th>盈亏</th></tr>
            </thead>
            <tbody>
              {trades.map((t, i) => (
                <tr key={i} onClick={() => setCurrentStock(t.ts_code, t.name ?? t.ts_code)}>
                  <td>{t.date.slice(5)}</td>
                  <td className="pp-name"><b>{t.name ?? ""}</b><em>{t.ts_code}</em></td>
                  <td><span className={`pp-act pp-act-${t.action}`}>{ACTION_LABEL[t.action] ?? t.action}</span></td>
                  <td>{t.price.toFixed(2)}</td>
                  <td>{t.shares}</td>
                  <td>{money(t.amount)}</td>
                  <td className={t.pnl == null ? "" : t.pnl >= 0 ? "up" : "down"}>
                    {t.pnl == null ? "—" : `${t.pnl >= 0 ? "+" : ""}${money(t.pnl)}`}
                    {t.pnl_pct != null && (
                      <em>{t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct.toFixed(2)}%</em>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === "closed" && (
        <div className="pp-table-wrap pp-scroll">
          <table className="pp-table">
            <thead>
              <tr><th>股票</th><th>买入</th><th>卖出</th><th>买价</th><th>方式</th><th>已实现</th></tr>
            </thead>
            <tbody>
              {st.closed.map((c, i) => (
                <tr key={i} onClick={() => setCurrentStock(c.ts_code, c.name ?? c.ts_code)}>
                  <td className="pp-name"><b>{c.name ?? ""}</b><em>{c.ts_code}</em></td>
                  <td>{c.open_date.slice(5)}</td>
                  <td>{c.close_date.slice(5)}</td>
                  <td>{c.open_price.toFixed(2)}</td>
                  <td><span className={`pp-act pp-act-${c.reason}`}>
                    {ACTION_LABEL[c.reason ?? ""] ?? c.reason}</span></td>
                  <td className={c.realized_pnl >= 0 ? "up" : "down"}>
                    {c.realized_pnl >= 0 ? "+" : ""}{money(c.realized_pnl)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
