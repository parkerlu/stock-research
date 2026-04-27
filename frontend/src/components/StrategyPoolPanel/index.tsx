import { useEffect, useMemo, useRef, useState } from "react";
import {
  deletePoolEntry,
  listConcepts,
  listPool,
  reorderPool,
  updatePoolEntry,
} from "../../api/strategyPool";
import type { PoolEntry } from "../../api/strategyPool";

export function StrategyPoolPanel() {
  const [entries, setEntries] = useState<PoolEntry[]>([]);
  const [concepts, setConcepts] = useState<string[]>([]);
  const [filter, setFilter] = useState<{ concept: string; active: "all" | "active" | "inactive" }>(
    { concept: "all", active: "all" }
  );
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const dragId = useRef<string | null>(null);
  const dragOverId = useRef<string | null>(null);
  const [, force] = useState(0);

  const load = async () => {
    setLoading(true);
    try {
      const [es, cs] = await Promise.all([listPool(), listConcepts()]);
      setEntries(es);
      setConcepts(cs);
    } catch (e: any) {
      setMsg(`✗ ${e?.message ?? "load failed"}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const filtered = useMemo(() => {
    return entries.filter((e) => {
      if (filter.concept !== "all" && e.concept !== filter.concept) return false;
      if (filter.active === "active" && !e.is_active) return false;
      if (filter.active === "inactive" && e.is_active) return false;
      return true;
    });
  }, [entries, filter]);

  const handleToggle = async (tid: string, current: boolean) => {
    setMsg(null);
    try {
      await updatePoolEntry(tid, { is_active: !current });
      setEntries((prev) => prev.map((e) => (e.template_id === tid ? { ...e, is_active: !current } : e)));
    } catch (e: any) {
      setMsg(`✗ ${e?.message ?? "更新失败"}`);
    }
  };

  const handleDelete = async (tid: string, name: string) => {
    if (!confirm(`确认从池子里删除「${name}」(${tid})？\n（不影响代码注册，可重启后台后重新出现）`)) return;
    setMsg(null);
    try {
      await deletePoolEntry(tid);
      setEntries((prev) => prev.filter((e) => e.template_id !== tid));
    } catch (e: any) {
      setMsg(`✗ ${e?.message ?? "删除失败"}`);
    }
  };

  const handleDragStart = (id: string) => () => {
    dragId.current = id;
  };
  const handleDragOver = (id: string) => (e: React.DragEvent) => {
    e.preventDefault();
    if (dragOverId.current !== id) {
      dragOverId.current = id;
      force((x) => x + 1);
    }
  };
  const handleDragEnd = async () => {
    const from = dragId.current;
    const to = dragOverId.current;
    dragId.current = null;
    dragOverId.current = null;
    if (!from || !to || from === to) {
      force((x) => x + 1);
      return;
    }
    // Reorder within the FULL entries list (not just filtered)
    const ordered = [...entries].sort((a, b) => a.sort_order - b.sort_order);
    const fromIdx = ordered.findIndex((e) => e.template_id === from);
    const toIdx = ordered.findIndex((e) => e.template_id === to);
    if (fromIdx < 0 || toIdx < 0) return;
    const [moved] = ordered.splice(fromIdx, 1);
    ordered.splice(toIdx, 0, moved);
    const newOrder = ordered.map((e) => e.template_id);
    try {
      const updated = await reorderPool(newOrder);
      setEntries(updated);
    } catch (e: any) {
      setMsg(`✗ ${e?.message ?? "重排失败"}`);
    }
  };

  return (
    <div className="strategy-pool-page">
      <div className="strategy-pool-toolbar">
        <div className="strategy-pool-title">
          策略池 <span className="strategy-pool-count">({entries.length})</span>
        </div>
        <div className="strategy-pool-filters">
          <label>
            概念:
            <select
              value={filter.concept}
              onChange={(e) => setFilter({ ...filter, concept: e.target.value })}
            >
              <option value="all">全部</option>
              {concepts.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
          <label>
            状态:
            <select
              value={filter.active}
              onChange={(e) =>
                setFilter({ ...filter, active: e.target.value as any })
              }
            >
              <option value="all">全部</option>
              <option value="active">仅激活</option>
              <option value="inactive">仅禁用</option>
            </select>
          </label>
          <button className="strategy-pool-refresh" onClick={load} disabled={loading}>
            ↻ {loading ? "加载中" : "刷新"}
          </button>
        </div>
      </div>

      {msg && <div className="strategy-pool-msg">{msg}</div>}

      <div className="strategy-pool-table-wrap">
        <table className="strategy-pool-table">
          <thead>
            <tr>
              <th style={{ width: 28 }}></th>
              <th style={{ width: 36 }}>#</th>
              <th>策略</th>
              <th>概念</th>
              <th>家族</th>
              <th>IS胜率</th>
              <th>IS平均</th>
              <th>OOS胜率</th>
              <th>OOS平均</th>
              <th>笔数</th>
              <th>状态</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((e) => {
              const m = e.metrics ?? {};
              const isOver = dragOverId.current === e.template_id;
              return (
                <tr
                  key={e.template_id}
                  draggable
                  onDragStart={handleDragStart(e.template_id)}
                  onDragOver={handleDragOver(e.template_id)}
                  onDragEnd={handleDragEnd}
                  onDrop={handleDragEnd}
                  className={`${e.is_active ? "" : "inactive"} ${isOver ? "drag-over" : ""}`}
                >
                  <td className="drag-handle" title="拖动以重排">⋮⋮</td>
                  <td>{e.sort_order}</td>
                  <td>
                    <div className="cell-name">
                      <span className="cell-tid">{e.template_id}</span>
                      <span className="cell-display">{e.display_name}</span>
                    </div>
                  </td>
                  <td><span className="concept-tag">{e.concept}</span></td>
                  <td className="cell-family">{e.family || "—"}</td>
                  <td>{m.is_win != null ? `${m.is_win.toFixed(1)}%` : "—"}</td>
                  <td>{m.is_avg != null ? `+${m.is_avg.toFixed(1)}%` : "—"}</td>
                  <td>{m.oos_win != null ? `${m.oos_win.toFixed(1)}%` : "—"}</td>
                  <td>{m.oos_avg != null ? `+${m.oos_avg.toFixed(1)}%` : "—"}</td>
                  <td>{m.trades ?? "—"}</td>
                  <td>
                    <button
                      className={`toggle-btn ${e.is_active ? "on" : "off"}`}
                      onClick={() => handleToggle(e.template_id, e.is_active)}
                    >
                      {e.is_active ? "● 激活" : "○ 禁用"}
                    </button>
                  </td>
                  <td>
                    <button
                      className="delete-btn"
                      onClick={() => handleDelete(e.template_id, e.display_name)}
                      title="从策略池移除"
                    >
                      ×
                    </button>
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={12} className="empty-row">无匹配策略</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
