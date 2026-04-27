import { useEffect, useRef, useState } from "react";
import {
  getDataStatus, startBackfill, getBackfillStatus, cancelBackfill,
} from "../../api/system";
import type { DataStatus, BackfillJobState } from "../../api/system";

export function SystemPanel() {
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [job, setJob] = useState<BackfillJobState | null>(null);
  const [includeInactive, setIncludeInactive] = useState(false);
  const [onlyStale, setOnlyStale] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const currentJobId = useRef<string | null>(null);

  const loadStatus = async () => {
    setLoading(true);
    try {
      setStatus(await getDataStatus());
    } catch (e: any) {
      setErr(e?.message ?? "load failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStatus();
    return () => {
      if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, []);

  const stopPolling = () => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  };

  const handleStart = async () => {
    setErr(null);
    setJob(null);
    try {
      const { job_id } = await startBackfill({
        include_inactive: includeInactive,
        only_stale: onlyStale,
      });
      currentJobId.current = job_id;
      const tick = async () => {
        if (!currentJobId.current) return;
        try {
          const s = await getBackfillStatus(currentJobId.current);
          setJob(s);
          if (s.status === "completed" || s.status === "cancelled" || s.status === "error") {
            stopPolling();
            currentJobId.current = null;
            await loadStatus();   // refresh stats after job done
          }
        } catch (e: any) {
          stopPolling();
          setErr(e?.message ?? "poll failed");
        }
      };
      stopPolling();
      pollTimer.current = setInterval(tick, 1000);
      tick();
    } catch (e: any) {
      setErr(e?.message ?? "start failed");
    }
  };

  const handleCancel = async () => {
    if (!currentJobId.current) return;
    try { await cancelBackfill(currentJobId.current); }
    catch (e: any) { setErr(e?.message ?? "cancel failed"); }
  };

  const isRunning = job?.status === "pending" || job?.status === "running";
  const progressPct = !job
    ? 0
    : job.status === "completed"
    ? 100
    : job.total > 0
    ? (job.progress / job.total) * 100
    : 0;

  return (
    <div className="system-page">
      <div className="system-toolbar">
        <div className="system-title">系统</div>
        <button className="strategy-pool-refresh" onClick={loadStatus} disabled={loading}>
          ↻ {loading ? "加载中" : "刷新"}
        </button>
      </div>

      {err && <div className="strategy-pool-msg">{err}</div>}

      <div className="system-section">
        <div className="system-section-title">数据库状态</div>
        {!status ? (
          <div className="screening-empty">加载中...</div>
        ) : (
          <div className="system-stats">
            <div className="stat-card">
              <div className="stat-label">市场最新</div>
              <div className="stat-value">{status.market_latest ?? "—"}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">总股票</div>
              <div className="stat-value">{status.totals.total_stocks}</div>
            </div>
            <div className="stat-card stat-good">
              <div className="stat-label">已最新</div>
              <div className="stat-value">{status.totals.up_to_date}</div>
            </div>
            <div className="stat-card stat-warn">
              <div className="stat-label">7天未更新</div>
              <div className="stat-value">{status.totals.stale_7d}</div>
            </div>
            <div className="stat-card stat-bad">
              <div className="stat-label">30天未更新</div>
              <div className="stat-value">{status.totals.stale_30d}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">禁用 (ST/停牌)</div>
              <div className="stat-value">{status.totals.inactive}</div>
            </div>
          </div>
        )}
      </div>

      <div className="system-section">
        <div className="system-section-title">K线补齐</div>
        <div className="system-options">
          <label>
            <input
              type="checkbox"
              checked={onlyStale}
              onChange={(e) => setOnlyStale(e.target.checked)}
              disabled={isRunning}
            />
            <span>仅补齐落后于市场最新日的股票（推荐）</span>
          </label>
          <label>
            <input
              type="checkbox"
              checked={includeInactive}
              onChange={(e) => setIncludeInactive(e.target.checked)}
              disabled={isRunning}
            />
            <span>包含已禁用股票（ST/停牌）</span>
          </label>
        </div>
        <div className="system-actions">
          {!isRunning ? (
            <button
              className="screening-btn-primary"
              onClick={handleStart}
              disabled={!status}
            >
              📥 开始补齐
            </button>
          ) : (
            <button className="screening-btn-cancel" onClick={handleCancel}>
              ⛔ 中止
            </button>
          )}
        </div>

        {job && (
          <div className="screening-progress">
            <div className="screening-progress-bar">
              <div
                className={`screening-progress-fill ${
                  job.status === "cancelled" ? "cancelled" : ""
                } ${job.status === "completed" ? "completed" : ""}`}
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <div className="screening-progress-text">
              {job.status === "running" && (
                <>
                  <span>{job.progress}/{job.total} 股</span>
                  <span>+{job.rows_inserted.toLocaleString()} 行</span>
                  <span>{job.elapsed_sec.toFixed(0)}s</span>
                </>
              )}
              {job.status === "completed" && (
                <span>完成 · {job.progress}/{job.total} 股 · 写入 {job.rows_inserted.toLocaleString()} 行 · {job.elapsed_sec.toFixed(0)}s</span>
              )}
              {job.status === "cancelled" && (
                <span>已中止 · {job.progress}/{job.total} · 写入 {job.rows_inserted.toLocaleString()} 行</span>
              )}
              {job.status === "error" && (
                <span className="error-text">错误: {job.error}</span>
              )}
            </div>
          </div>
        )}

        {job?.failures && job.failures.length > 0 && (
          <div className="system-failures">
            <div className="system-failures-title">最近失败 ({job.failures.length})</div>
            {job.failures.map((f, i) => (
              <div key={i} className="system-failure-row">
                <span className="hit-code">{f.ts_code}</span>
                <span className="hit-name">{f.error}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {status && status.stale.length > 0 && (
        <div className="system-section">
          <div className="system-section-title">
            落后股票 ({status.stale.length}{status.stale.length >= 200 ? "+" : ""})
          </div>
          <div className="system-table-wrap">
            <table className="strategy-pool-table">
              <thead>
                <tr>
                  <th>代码</th>
                  <th>名称</th>
                  <th>最新数据</th>
                  <th>落后天数</th>
                  <th>K线条数</th>
                  <th>状态</th>
                </tr>
              </thead>
              <tbody>
                {status.stale.map((s) => (
                  <tr key={s.ts_code} className={s.active ? "" : "inactive"}>
                    <td><span className="cell-tid">{s.ts_code}</span></td>
                    <td>{s.name ?? "—"}</td>
                    <td>{s.latest_date}</td>
                    <td className={s.days_stale >= 30 ? "stat-bad" : "stat-warn"}>
                      {s.days_stale}天
                    </td>
                    <td>{s.bars}</td>
                    <td>
                      <span className="concept-tag">
                        {s.active ? "激活" : "禁用"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
