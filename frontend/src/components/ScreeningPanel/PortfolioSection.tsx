// 组合信号 — 回测验证过的三源组合每日买入清单。
//
// 与下方"选择策略"逐个扫描不同: 这里一次跑三个策略并按组合规则(低流动性优先、
// 每源配额)出 20 只, 直接对应回测里那套仓位配置。
import { useEffect, useRef, useState } from "react";
import {
  getPortfolioConfig,
  getPortfolioStatus,
  startPortfolioScan,
} from "../../api/screening";
import type { PortfolioConfig, PortfolioJob } from "../../api/screening";

interface Props {
  onPick: (tsCode: string, name: string | null) => void;
}

export function PortfolioSection({ onPick }: Props) {
  const [cfg, setCfg] = useState<PortfolioConfig | null>(null);
  const [job, setJob] = useState<PortfolioJob | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [lookback, setLookback] = useState(5);
  const [open, setOpen] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    getPortfolioConfig().then(setCfg).catch(() => setCfg(null));
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const poll = (id: string) => {
    getPortfolioStatus(id)
      .then((j) => {
        setJob(j);
        if (j.status === "pending" || j.status === "running") {
          timer.current = setTimeout(() => poll(id), 1200);
        } else if (j.status === "error") {
          setErr(j.error ?? "扫描失败");
        }
      })
      .catch((e) => setErr(e instanceof Error ? e.message : "扫描失败"));
  };

  const run = async () => {
    setErr(null);
    setJob(null);
    try {
      const { job_id } = await startPortfolioScan(lookback, 20);
      poll(job_id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "启动失败");
    }
  };

  const busy = job?.status === "pending" || job?.status === "running";
  const picks = job?.picks ?? [];

  return (
    <div className="screening-section pf-section">
      <div className="screening-section-title pf-title">
        <span>🎯 组合信号</span>
        {cfg && (
          <button className="pf-info-btn" onClick={() => setOpen((v) => !v)}>
            回测 {cfg.backtest.cagr}% / 回撤 {cfg.backtest.max_drawdown}% {open ? "▴" : "▾"}
          </button>
        )}
      </div>

      {open && cfg && (
        <div className="pf-info">
          <div className="pf-info-row">
            <b>信号源</b>
            <span>{cfg.strategies.map((s) => s.label).join(" · ")}</span>
          </div>
          <div className="pf-info-row">
            <b>规则</b>
            <span>
              {cfg.slots} 个并行仓位 · {cfg.rank_rule} · 20日均额 ≥{" "}
              {cfg.min_amount_wan} 万 · 每笔按当前净值 1/{cfg.slots} 下单
            </span>
          </div>
          <div className="pf-info-row">
            <b>回测</b>
            <span>
              {cfg.backtest.period} → {cfg.backtest.final}，年化{" "}
              {cfg.backtest.cagr}%，最大回撤 {cfg.backtest.max_drawdown}%，
              最长回撤 {cfg.backtest.longest_drawdown_days} 个交易日
            </span>
          </div>
          <div className="pf-info-row">
            <b>基准</b>
            <span>{cfg.backtest.benchmark}</span>
          </div>
          <div className="pf-warn">
            ⚠️ 容量上限约 {cfg.backtest.capacity_wan} 万本金，超过后冲击成本会吃掉超额。
            回测不预示未来，据此交易的风险由你自己承担。
          </div>
        </div>
      )}

      <div className="pf-controls">
        <label>
          回看
          <select
            value={lookback}
            onChange={(e) => setLookback(Number(e.target.value))}
            disabled={busy}
          >
            <option value={3}>3 天</option>
            <option value={5}>5 天</option>
            <option value={10}>10 天</option>
          </select>
        </label>
        <button className="pf-run" onClick={run} disabled={busy}>
          {busy ? "扫描中…" : "生成今日清单"}
        </button>
        {job?.status === "completed" && (
          <span className="pf-meta">
            {job.as_of} · 扫描 {job.scanned} 只 · {job.elapsed_sec}s
          </span>
        )}
      </div>

      {busy && (
        <div className="pf-progress">
          <div
            className="pf-progress-bar"
            style={{ width: `${job?.total ? (job.progress / job.total) * 100 : 0}%` }}
          />
        </div>
      )}
      {err && <div className="screening-error">{err}</div>}

      {picks.length > 0 && (
        <div className="pf-list">
          {picks.map((p) => (
            <div key={p.ts_code} className="pf-pick" onClick={() => onPick(p.ts_code, p.name)}>
              <span className="pf-rank">{p.rank}</span>
              <div className="pf-body">
                <div className="pf-row1">
                  <span className="pf-code">{p.ts_code}</span>
                  <span className="pf-name">{p.name ?? ""}</span>
                  <span className="pf-price">¥{p.latest_close.toFixed(2)}</span>
                </div>
                <div className="pf-row2">
                  <span className="pf-src">{p.strategies}</span>
                  <span className="pf-date">{p.signal_date}</span>
                  <span className="pf-amt">{p.amount_20d_wan.toLocaleString()} 万</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
      {job?.status === "completed" && picks.length === 0 && (
        <div className="screening-empty">近 {lookback} 天无符合条件的信号</div>
      )}
    </div>
  );
}
