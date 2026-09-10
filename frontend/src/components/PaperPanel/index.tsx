// 虚拟盘 — chan-2buy 最优配置的实盘跟踪。
// 三块: 账户总览 / 当前持仓(含浮盈与三档价位) / 全部动作流水。
import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEMO_ACCOUNT,
  LIVE_ACCOUNT,
  getPaperAccounts,
  getPaperConfig,
  getPaperEquity,
  getPaperRun,
  getPaperSignals,
  getPaperStatus,
  getPaperTrades,
  getDemoStrategies,
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
import type { TradeAction } from "../../types/strategy";

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

/** 虚拟盘成交 -> K线标记。tier1/tier2/stop/timeout 都是卖, 但标签要分清楚,
 *  复用上面表格用的 ACTION_LABEL, 免得两处文案对不上。 */
function toMarks(rows: PaperTrade[], code: string): TradeAction[] {
  const mine = rows.filter((t) => t.ts_code === code);
  // 信号日单独标一个 —— 信号在收盘后确认, 次日开盘才成交, 两者不是同一根K线。
  // 信号日已经写在买入成交的 note 里 ("chan-2buy 信号 YYYY-MM-DD")。
  const signals: TradeAction[] = [];
  for (const t of mine) {
    const m = t.action === "buy" ? /(\d{4}-\d{2}-\d{2})/.exec(t.note ?? "") : null;
    if (m && m[1] !== t.date) {
      signals.push({
        date: m[1], type: "signal", price: t.price, shares: 0, amount: 0,
        position_level: 0, label: "信号",
      });
    }
  }
  return signals.concat(mine
    .map((t) => ({
      date: t.date,
      type: (t.action === "buy" ? "buy" : "sell") as "buy" | "sell",
      price: t.price,
      shares: t.shares,
      amount: t.amount,
      position_level: 0,
      pnl: t.pnl ?? undefined,
      pnl_pct: t.pnl_pct ?? undefined,
      label: `${ACTION_LABEL[t.action] ?? t.action} ${t.price.toFixed(2)}`
             + (t.pnl_pct != null ? ` ${t.pnl_pct >= 0 ? "+" : ""}${t.pnl_pct.toFixed(1)}%` : ""),
    })));
}

/** 策略 key -> 中文名。⚠️ 与后端 DEMO_STRATEGIES 的 label 对齐, 改一处要改两处;
 *  真要彻底统一应该让前端直接用 /api/paper/strategies 的返回。 */
const STRAT_LABEL: Record<string, string> = {
  chips: "筹码模型", shape: "形态模型", liftalert: "拉升预警",
  mmweek: "周线版", mmweek_f: "周线版(过滤)", breakout: "突破预警",
  sar: "SAR预警", "tdx-dual-kdj": "⚠️已下架",
};

export function PaperPanel() {
  const [cfg, setCfg] = useState<PaperConfig | null>(null);
  const [st, setSt] = useState<PaperStatus | null>(null);
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [eq, setEq] = useState<{ initial: number; points: EquityPoint[] } | null>(null);
  const [tab, setTab] = useState<"holdings" | "trades" | "closed">("holdings");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showRules, setShowRules] = useState(false);
  // ⚠️ 默认账户不能写死 LIVE_ACCOUNT("live-2026") —— 2026-09-10 用户发现:
  //    那个账户用的是【已下架】的 tdx-dual-kdj(周内多周期穿越), 而且
  //    is_active=false 所以【不在下拉列表里】。结果"什么都不选"时页面显示的
  //    是一个本该停用的账户的净值, 看起来却像当前实盘。
  //    改成落到列表里第一个可用账户, 列表为空时才回退到常量。
  const [acct, setAcct] = useState<string>(
    () => localStorage.getItem("paper.acct") ?? LIVE_ACCOUNT);
  // 账户列表从后端拿 —— 以前是两个写死的按钮(live-2026 / demo), 结果新建的
  // 账户在页面上根本点不到。⚠️ 后端只返回 is_active 的。
  const [accts, setAccts] = useState<
    { id: number; name: string; strategy: string; trades: number; pnl_pct: number }[]
  >([]);
  const [acctLoaded, setAcctLoaded] = useState(false);
  useEffect(() => {
    getPaperAccounts()
      .then((r) => {
        setAccts(r.accounts);
        // 选中的账户不在可用列表里(如已停用的 live-2026) -> 挑一个有内容的。
        // ⚠️ 不能直接取 accounts[0]: 那是按 id 排的, 很可能是个 0 笔交易的
        //    账户, 打开页面一片空白。优先选有交易记录的。
        setAcct((cur) => {
          if (!r.accounts.length || r.accounts.some((x) => x.name === cur)) return cur;
          const live = r.accounts.filter((x) => !x.name.startsWith("demo"));
          const pool = live.length ? live : r.accounts;
          return (pool.find((x) => (x.trades ?? 0) > 0) ?? pool[0]).name;
        });
      })
      .catch(() => setAccts([]))
      .finally(() => setAcctLoaded(true));
  }, []);
  const [signals, setSignals] = useState<SignalPick[]>([]);
  const [playing, setPlaying] = useState(false);
  // 推进一天要等后端撮合, 慢的时候几秒 —— 不给反馈会让人以为点没生效
  const [stepping, setStepping] = useState(false);
  // 演示盘的策略选择。⚠️ 以前没有这个: reset 出来的账户继承 DEFAULT_CONFIG 的
  // strategy=None, 点"下一日"日期在走但一笔都不买, 而且不报错
  // (2026-09-10 用户从 2025-01-01 点到 2025-02-12 什么都没有)。
  const [strategies, setStrategies] = useState<
    { key: string; label: string; since: string }[]>([]);
  // 最早可回放日由后端给 —— chips 的历史打分只补到 2024-12, 再往前没有信号
  const [minDate, setMinDate] = useState("2025-01-01");
  const [demoStrat, setDemoStrat] = useState(
    () => localStorage.getItem("paper.demoStrat") ?? "liftalert");
  useEffect(() => {
    getDemoStrategies().then((r) => {
      setStrategies(r.items);
      if (r.min_date) setMinDate(r.min_date);
    }).catch(() => {});
  }, []);
  useEffect(() => { localStorage.setItem("paper.demoStrat", demoStrat); }, [demoStrat]);
  const stratMeta = strategies.find((x) => x.key === demoStrat);
  const [speed, setSpeed] = useState(600);
  const [startDate, setStartDate] = useState("2025-01-01");   // ⚠️ 与后端 DEMO_MIN_DATE 一致
  // 起点早于该策略的信号起点 -> 前面那段必然空跑, 直接提示而不是让人点半天
  const tooEarly = !!stratMeta && startDate < `${stratMeta.since}-01`;
  // 起始日期框跟随账户实际起点 —— 否则重置后框里还留着上次手打的日期,
  // 会出现"框里 2026/01/01, 当前 2022-06-09"这种自相矛盾的显示。
  const syncedFor = useRef<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const playRef = useRef(false);
  const isDemo = acct.startsWith("demo");
  // 当前 tab 下该显示哪些账户 —— 实操盘与演示盘分开, 不能混
  const visibleAccts = accts.filter((x) => x.name.startsWith("demo") === isDemo);
  // ⚠️ 只有主演示盘("demo")能"重新开始"。demo-xxx-回测 是几年期的历史回放
  //    (1097 笔 / 772 笔), 而重置按钮写死 resetPaper(DEMO_ACCOUNT) ——
  //    选中它们时若还显示这个按钮, 点下去会把主演示盘清掉, 而用户以为
  //    重置的是当前这个。
  const canReset = acct === DEMO_ACCOUNT;
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const jumpToDate = useQuoteStore((s) => s.jumpToDate);
  const setReplayDate = useQuoteStore((s) => s.setReplayDate);
  const setPaperMarks = useQuoteStore((s) => s.setPaperMarks);
  // 用 ref 读, 避免把它们塞进 reload 的依赖里导致每次换股都重新拉一遍数据
  const curSymRef = useRef("");
  const hasMarksRef = useRef(false);
  // 标记内容的指纹 —— 每步都无脑重建会清空再画, 图上闪一下
  const markSigRef = useRef("");
  curSymRef.current = useQuoteStore((s) => s.currentSymbol);
  hasMarksRef.current = useQuoteStore((s) => s.paperMarks) !== null;

  const reload = useCallback(async (name = acct) => {
    try {
      const [s, t, e, g] = await Promise.all([
        getPaperStatus(name), getPaperTrades(name, 200), getPaperEquity(name),
        getPaperSignals(name, 20).catch(() => ({ picks: [] as SignalPick[] })),
      ]);
      setSt(s);
      // 回放推进一天, 右侧K线上的"今日"线就跟着移动
      setReplayDate(s.last_run_date ?? null);
      // 这一天如果对当前正在看的票有买卖动作, K线上的标记也要立刻跟着出现,
      // 否则得重新点一次持仓行才看得到。只在图上本来就是虚拟盘标记时才刷新,
      // 免得把从行情页打开的图也接管了。
      if (hasMarksRef.current && curSymRef.current) {
        const next = toMarks(t.trades, curSymRef.current);
        const sig = `${curSymRef.current}|` +
          next.map((m) => `${m.date}${m.type}${m.price}`).join(",");
        if (sig !== markSigRef.current) {
          markSigRef.current = sig;
          setPaperMarks(next);
        }
      }
      // 换账户时把日期框对齐到该账户的实际起点 (同一账户内不覆盖用户正在改的值)
      if (s.started_on && syncedFor.current !== name) {
        syncedFor.current = name;
        setStartDate(s.started_on);
      }
      setTrades(t.trades);
      setSignals(g.picks ?? []);
      if (e.exists) setEq({ initial: e.initial, points: e.points });
      setErr(null);
    } catch (x) {
      setErr(x instanceof Error ? x.message : "加载失败");
    }
  }, [acct, setReplayDate, setPaperMarks]);

  /** 在右侧K线打开某只票并定位到某天, 同时带上它的全部成交标记 */
  const openInChart = useCallback((code: string, nm: string | null, day: string) => {
    markSigRef.current = "";        // 换股/换日期 -> 指纹作废, 下次必定重画
    jumpToDate(code, nm ?? code, day, toMarks(trades, code), st?.last_run_date ?? null);
  }, [jumpToDate, trades, st]);

  /** 演示盘: 走一个交易日 */
  const stepOnce = useCallback(async () => {
    setStepping(true);
    try {
      const r = await stepPaper(acct, 1);
      await reload(acct);
      if (r.done) { playRef.current = false; setPlaying(false); }
      return !r.done;
    } catch (x) {
      playRef.current = false; setPlaying(false);
      setErr(x instanceof Error ? x.message : "推进失败");
      return false;
    } finally {
      setStepping(false);
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
    // ⚠️ 等账户列表回来再取数 —— 否则会先用写死的 LIVE_ACCOUNT 拉一次,
    //    用户会看到那个已停用账户的净值闪一下(甚至停在上面)。
    if (acctLoaded) {
      localStorage.setItem("paper.acct", acct);
      reload(acct);
    }
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
    // ⚠️ 依赖必须带 acctLoaded/acct: 账户列表是异步回来的, 只依赖 reload
    //    的话列表把 acct 换掉之后不会重新取数, 页面停在旧账户上。
  }, [reload, acctLoaded, acct]);

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
          {/* ⚠️ 按当前 tab 分开列: 实操盘只列自动推进的, 演示回放只列 demo。
              以前混在一起, 在"实操盘"下也能选到 demo-xxx-回测(那些是几年回放、
              收益动辄 +165%), 摆在实操盘旁边极易被当成实盘业绩。
              ⚠️ 也不再放空的占位 option —— 它在"当前账户不在列表里"时会被选中
              并打勾(2026-09-10 用户截图就是这个状态), 看起来像选了, 其实页面
              显示的是隐藏的 live-2026。现在 acct 始终落在列表内, 无需占位。 */}
          {visibleAccts.length > 0 && (
            <select
              className="pp-acct-sel"
              value={visibleAccts.some((x) => x.name === acct) ? acct : visibleAccts[0].name}
              onChange={(e) => {
                const v = e.target.value;
                if (!v) return;
                setPlaying(false);
                setAcct(v);
                reload(v);
              }}
              title={isDemo ? "演示账户 —— 手动回放用" : "实操盘 —— 每晚自动推进"}
            >
              {visibleAccts.map((x) => (
                <option key={x.id} value={x.name}>
                  {STRAT_LABEL[x.strategy] ?? x.strategy ?? "无策略"}
                  {" · "}{x.name} · {x.pnl_pct >= 0 ? "+" : ""}{x.pnl_pct}% · {x.trades}笔
                </option>
              ))}
            </select>
          )}
          {cfg && (
            <button className="pp-rules-btn" onClick={() => setShowRules((v) => !v)}>
              规则 {showRules ? "▴" : "▾"}
            </button>
          )}
        </div>
        <div className="pp-actions">
          {isDemo && canReset ? (
            <span className="pp-asof">
              <select className="pp-date" value={demoStrat}
                      title="回放哪个策略 —— 出场规则跟随该策略的训练标签"
                      onChange={(e) => setDemoStrat(e.target.value)}>
                {strategies.map((x) => (
                  <option key={x.key} value={x.key}>{x.label}</option>
                ))}
              </select>
              从
              <input type="date" className="pp-date" value={startDate} min={minDate}
                     title={`最早可回放 ${minDate}`}
                     onChange={(e) => setStartDate(
                       e.target.value < minDate ? minDate : e.target.value)} />
              {tooEarly && (
                <span className="pp-warn" title="该策略在这个日期还没有信号, 回放会空跑">
                  ⚠️ {stratMeta?.label}的信号从 {stratMeta?.since} 才有
                </span>
              )}
              <button className="pp-reset-btn" title="清空并从这一天重新开始 (10 万本金 / 10 仓位)"
                      onClick={async () => {
                        setPlaying(false);
                        await resetPaper(DEMO_ACCOUNT, startDate, demoStrat);
                        await reload(DEMO_ACCOUNT);
                      }}>重新开始</button>
            </span>
          ) : (
            <span className="pp-asof">
              {st.started_on} 起 · 当前 {st.last_run_date ?? "未开始"}
            </span>
          )}
          {isDemo ? (
            <>
              <button className="pp-run" onClick={() => void stepOnce()}
                      disabled={playing || stepping}>
                {stepping ? <span className="pp-spin" /> : null}
                {stepping ? "计算中" : "下一日 ▸"}
              </button>
              <button className={`pp-run pp-play${playing ? " on" : ""}`}
                      onClick={() => setPlaying((v) => !v)}>
                {playing && stepping ? <span className="pp-spin" /> : null}
                {playing ? "⏸ 暂停" : "▶ 自动"}
              </button>
              <select className="pp-speed" value={speed}
                      onChange={(e) => setSpeed(Number(e.target.value))}>
                <option value={1200}>慢</option>
                <option value={600}>中</option>
                <option value={150}>快</option>
              </select>
              {/* 回放当前日期是回放盘最要紧的一个数 —— 放在最后、加粗白字 */}
              <span className="pp-cur">当前 {st.last_run_date ?? "未开始"}</span>
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
                    title={`20日均额 ${p.amount_20d_wan} 万 · ${p.next_open != null ? `次日开盘 ${p.next_open}` : "明日开盘买入"}`}>
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
                <tr key={h.ts_code} title="点击查看该股买入当日的K线"
                    onClick={() => openInChart(h.ts_code, h.name, h.open_date)}>
                  <td className="pp-name">
                    <b>{h.name ?? ""}</b><em>{h.ts_code}</em>
                  </td>
                  <td>{h.open_date.slice(5)}<em>{h.hold_days}天</em></td>
                  <td>{h.open_price.toFixed(2)}</td>
                  <td>{h.last_price.toFixed(2)}</td>
                  <td>{h.shares}{h.tier1_done && <em>已减半</em>}</td>
                  <td className={`pp-pnl ${h.float_pnl >= 0 ? "up" : "down"}`}>
                    <b>{h.float_pnl_pct >= 0 ? "+" : ""}{h.float_pnl_pct.toFixed(2)}%</b>
                    <em>{h.float_pnl >= 0 ? "+" : ""}{money(h.float_pnl)}</em>
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

      {/* 持仓表下面本来是一大片空白 —— 拿来常驻显示流水, 不用切标签页 */}
      {tab === "holdings" && trades.length > 0 && (
        <div className="pp-recent">
          <div className="pp-recent-t">最近动作（点任意一行看当日K线）</div>
          <div className="pp-table-wrap pp-scroll pp-recent-list">
            <table className="pp-table">
              <tbody>
                {trades.slice(0, 60).map((t, i) => (
                  <tr key={i} title="点击查看该笔成交当日的K线"
                      onClick={() => openInChart(t.ts_code, t.name, t.date)}>
                    <td className="pp-rd">{t.date.slice(5)}</td>
                    <td className="pp-name"><b>{t.name ?? t.ts_code}</b></td>
                    <td><span className={`pp-act pp-act-${t.action}`}>
                      {ACTION_LABEL[t.action] ?? t.action}</span></td>
                    <td>{t.price.toFixed(2)}</td>
                    <td>{t.shares}</td>
                    <td className={`pp-pnl ${(t.pnl ?? 0) >= 0 ? "up" : "down"}`}>
                      {t.pnl_pct != null && (
                        <b>{t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct.toFixed(2)}%</b>
                      )}
                      {t.pnl != null && (
                        <em>{t.pnl >= 0 ? "+" : ""}{money(t.pnl)}</em>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
                <tr key={i} title="点击查看该笔成交当日的K线"
                    onClick={() => openInChart(t.ts_code, t.name, t.date)}>
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
                <tr key={i} title="点击查看该股买入当日的K线"
                    onClick={() => openInChart(c.ts_code, c.name, c.open_date)}>
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
