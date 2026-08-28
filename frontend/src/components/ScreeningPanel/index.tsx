import { useEffect, useRef, useState } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import { useStrategyStore } from "../../stores/strategyStore";
import { startScan, getScanStatus, cancelScan } from "../../api/screening";
import type { JobState, StockHit } from "../../api/screening";
import { createPoolWithStocks } from "../../api/pools";
import { reorderStrategyPool } from "../../api/strategy";

function todayYYMMDD(): string {
  const d = new Date();
  const yy = String(d.getFullYear()).slice(-2);
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yy}${mm}${dd}`;
}

export function ScreeningPanel() {
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const templates = useStrategyStore((s) => s.templates);
  const fetchTemplates = useStrategyStore((s) => s.fetchTemplates);

  const [activeTpl, setActiveTpl] = useState<string | null>(null);
  const [lookbackDays, setLookbackDays] = useState(3);
  const [job, setJob] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const currentJobId = useRef<string | null>(null);

  // Drag-drop reordering of the strategy list (persisted via strategy-pool)
  const dragFrom = useRef<number | null>(null);
  const [dragOverIdx, setDragOverIdx] = useState<number | null>(null);

  const handleDrop = async (to: number) => {
    const from = dragFrom.current;
    dragFrom.current = null;
    setDragOverIdx(null);
    if (from === null || from === to) return;
    const arr = [...templates];
    const [moved] = arr.splice(from, 1);
    arr.splice(to, 0, moved);
    useStrategyStore.setState({ templates: arr });
    try {
      await reorderStrategyPool(arr.map((t) => t.template_id));
    } catch {
      fetchTemplates(); // restore server truth on failure
    }
  };

  useEffect(() => {
    if (templates.length === 0) fetchTemplates();
  }, [templates.length, fetchTemplates]);

  // Cleanup on unmount
  useEffect(() => () => {
    if (pollTimer.current) clearInterval(pollTimer.current);
  }, []);

  const stopPolling = () => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  };

  const handleStockClick = (hit: StockHit) => {
    setCurrentStock(hit.ts_code, hit.name ?? hit.ts_code);
    // Just mark the buy date — no full backtest, no sell point
    useStrategyStore.setState({
      tradeActions: [
        {
          date: hit.signal_date,
          type: "buy" as const,
          price: hit.latest_close,
          shares: 0,
          amount: 0,
          position_level: 1,
        },
      ],
    });
  };

  const handleSearch = async () => {
    if (!activeTpl) return;
    setError(null);
    setJob(null);
    try {
      const { job_id } = await startScan({
        template_id: activeTpl,
        lookback_days: lookbackDays,
      });
      currentJobId.current = job_id;
      // Poll every 800ms
      const tick = async () => {
        if (!currentJobId.current) return;
        try {
          const s = await getScanStatus(currentJobId.current);
          setJob(s);
          if (s.status === "completed" || s.status === "cancelled" || s.status === "error") {
            stopPolling();
            currentJobId.current = null;
          }
        } catch (e: any) {
          stopPolling();
          setError(e?.message ?? "poll failed");
        }
      };
      stopPolling();
      pollTimer.current = setInterval(tick, 250);  // tight poll for smooth progress
      tick();  // immediate first poll
    } catch (e: any) {
      setError(e?.message ?? "start failed");
    }
  };

  const handleCancel = async () => {
    if (!currentJobId.current) return;
    try {
      await cancelScan(currentJobId.current);
    } catch (e: any) {
      setError(e?.message ?? "cancel failed");
    }
  };

  const [poolName, setPoolName] = useState(todayYYMMDD());
  const [savingPool, setSavingPool] = useState(false);
  const [poolMsg, setPoolMsg] = useState<string | null>(null);

  const handleAddToPool = async () => {
    const hits = job?.hits ?? [];
    if (hits.length === 0 || !poolName.trim()) return;
    setSavingPool(true);
    setPoolMsg(null);
    try {
      const r = await createPoolWithStocks(
        poolName.trim(),
        hits.map((h) => h.ts_code),
        activeTpl ? `${activeTpl} 信号 (${lookbackDays}天)` : undefined
      );
      setPoolMsg(`✓ "${r.name}" 已建，加入 ${r.added}/${r.total_requested}`);
    } catch (e: any) {
      setPoolMsg(`✗ ${e?.message ?? "保存失败"}`);
    } finally {
      setSavingPool(false);
    }
  };

  const isRunning = job?.status === "pending" || job?.status === "running";
  const progressPct = !job
    ? 0
    : job.status === "completed"
    ? 100
    : job.total > 0
    ? (job.progress / job.total) * 100
    : 0;
  const hits = job?.hits ?? [];

  return (
    <div className="screening-panel">
      <div className="screening-section">
        <div className="screening-section-title">a) 选择策略</div>
        <div className="screening-strategy-list">
          {templates.length === 0 && <div className="screening-empty">加载策略中...</div>}
          {templates.map((t, i) => (
            <button
              key={t.template_id}
              className={`screening-strategy-btn ${activeTpl === t.template_id ? "active" : ""} ${
                dragOverIdx === i ? "dragover" : ""
              }`}
              disabled={isRunning}
              onClick={() => setActiveTpl(t.template_id)}
              title={`${t.name}（可拖拽排序）`}
              draggable={!isRunning}
              onDragStart={() => { dragFrom.current = i; }}
              onDragOver={(e) => { e.preventDefault(); setDragOverIdx(i); }}
              onDragLeave={() => setDragOverIdx((v) => (v === i ? null : v))}
              onDrop={(e) => { e.preventDefault(); handleDrop(i); }}
              onDragEnd={() => { dragFrom.current = null; setDragOverIdx(null); }}
            >
              <span className="screening-strategy-id">{t.template_id}</span>
              <span className="screening-strategy-name">{t.name}</span>
            </button>
          ))}
        </div>

        <div className="screening-controls">
          <label>
            最近
            <input
              type="number"
              min={1}
              max={30}
              value={lookbackDays}
              disabled={isRunning}
              onChange={(e) => setLookbackDays(parseInt(e.target.value) || 3)}
            />
            天信号
          </label>
        </div>

        <div className="screening-actions">
          {!isRunning ? (
            <button
              className="screening-btn-primary"
              disabled={!activeTpl}
              onClick={handleSearch}
            >
              🔍 搜索股票
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
                  <span>{job.progress}/{job.total}</span>
                  <span>{progressPct.toFixed(0)}%</span>
                  <span>{job.elapsed_sec.toFixed(0)}s</span>
                </>
              )}
              {job.status === "completed" && (
                <span>完成 · 扫描 {job.total} 股 · 耗时 {job.elapsed_sec.toFixed(0)}s</span>
              )}
              {job.status === "cancelled" && (
                <span>已中止 · {job.progress}/{job.total}</span>
              )}
              {job.status === "error" && (
                <span className="error-text">错误: {job.error}</span>
              )}
            </div>
          </div>
        )}

        {error && <div className="screening-error">{error}</div>}
      </div>

      <div className="screening-section screening-section-bottom">
        <div className="screening-section-title">
          b) 找到的股票{" "}
          {hits.length > 0 && <span className="hit-count">({hits.length})</span>}
        </div>

        {hits.length > 0 && (
          <div className="screening-pool-add">
            <input
              type="text"
              className="screening-pool-name"
              value={poolName}
              onChange={(e) => setPoolName(e.target.value)}
              placeholder="股票池名称"
              disabled={savingPool}
            />
            <button
              className="screening-btn-pool"
              onClick={handleAddToPool}
              disabled={savingPool || !poolName.trim()}
              title={`将 ${hits.length} 只股票加入新股票池`}
            >
              {savingPool ? "保存中..." : `📥 加入股票池 (${hits.length})`}
            </button>
            {poolMsg && (
              <div className={`screening-pool-msg ${poolMsg.startsWith("✓") ? "ok" : "err"}`}>
                {poolMsg}
              </div>
            )}
          </div>
        )}
        {!job && !activeTpl && (
          <div className="screening-empty">先点击上方策略选择</div>
        )}
        {!job && activeTpl && (
          <div className="screening-empty">点击「搜索股票」开始扫描</div>
        )}
        {job && hits.length === 0 && job.status !== "running" && (
          <div className="screening-empty">最近 {lookbackDays} 天无信号</div>
        )}
        <div className="screening-hits-list">
          {hits.map((h) => (
            <div
              key={h.ts_code}
              className={`screening-hit ${currentSymbol === h.ts_code ? "active" : ""}`}
              onClick={() => handleStockClick(h)}
            >
              <div className="hit-row1">
                <span className="hit-code">{h.ts_code}</span>
                <span className="hit-name">{h.name ?? ""}</span>
              </div>
              <div className="hit-row2">
                <span className="hit-date">📅 信号 {h.signal_date}</span>
                <span className="hit-price">¥{h.latest_close.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
