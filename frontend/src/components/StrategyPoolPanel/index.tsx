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
  const [sortKey, setSortKey] = useState<string>("score");
  const [sortAsc, setSortAsc] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const dragId = useRef<string | null>(null);
  const dragOverId = useRef<string | null>(null);
  const [, force] = useState(0);

  const toggleSort = (key: string) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);   // most columns most useful in descending order
    }
  };

  const beginRename = (e: PoolEntry) => {
    setEditingId(e.template_id);
    setEditText(e.display_name);
  };
  const commitRename = async () => {
    if (!editingId) return;
    const newName = editText.trim();
    const cur = entries.find((x) => x.template_id === editingId);
    setEditingId(null);
    if (!cur || !newName || newName === cur.display_name) return;
    try {
      await updatePoolEntry(editingId, { display_name: newName });
      setEntries((prev) =>
        prev.map((e) => (e.template_id === editingId ? { ...e, display_name: newName } : e))
      );
    } catch (e: any) {
      setMsg(`✗ ${e?.message ?? "rename failed"}`);
    }
  };

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
    const list = entries.filter((e) => {
      if (filter.concept !== "all" && e.concept !== filter.concept) return false;
      if (filter.active === "active" && !e.is_active) return false;
      if (filter.active === "inactive" && e.is_active) return false;
      return true;
    });
    const get = (e: PoolEntry, k: string): number | string => {
      if (k === "score") return e.score ?? 0;
      if (k === "tid") return e.template_id;
      if (k === "name") return e.display_name;
      if (k === "concept") return e.concept;
      if (k === "family") return e.family;
      if (k === "active") return e.is_active ? 1 : 0;
      const m: any = e.metrics ?? {};
      if (k === "win") return m.win_rate ?? m.is_win ?? 0;
      if (k === "avg") return m.avg_ret ?? m.is_avg ?? 0;
      if (k === "trades") return m.trades ?? 0;
      if (k === "stocks") return m.stocks ?? 0;
      if (k === "mdd") return m.avg_mdd ?? 0;
      return 0;
    };
    list.sort((a, b) => {
      const va = get(a, sortKey);
      const vb = get(b, sortKey);
      const cmp = typeof va === "number" && typeof vb === "number"
        ? (va as number) - (vb as number)
        : String(va).localeCompare(String(vb));
      return sortAsc ? cmp : -cmp;
    });
    return list;
  }, [entries, filter, sortKey, sortAsc]);

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
              <th style={{ width: 60, cursor: "pointer" }} onClick={() => toggleSort("score")}>
                评分 {sortKey === "score" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("name")}>
                策略 {sortKey === "name" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("concept")}>
                概念 {sortKey === "concept" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("family")}>
                家族 {sortKey === "family" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("win")}>
                胜率 {sortKey === "win" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("avg")}>
                平均 {sortKey === "avg" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("trades")}>
                笔数 {sortKey === "trades" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("stocks")}>
                股票 {sortKey === "stocks" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("mdd")}>
                回撤 {sortKey === "mdd" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
              <th style={{ cursor: "pointer" }} onClick={() => toggleSort("active")}>
                状态 {sortKey === "active" ? (sortAsc ? "↑" : "↓") : ""}
              </th>
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
                  <td className="cell-score">
                    {e.score && e.score > 0 ? e.score.toFixed(2) : "—"}
                  </td>
                  <td>
                    <div className="cell-name">
                      <span className="cell-tid">{e.template_id}</span>
                      {editingId === e.template_id ? (
                        <input
                          className="cell-rename-input"
                          value={editText}
                          autoFocus
                          onChange={(ev) => setEditText(ev.target.value)}
                          onBlur={commitRename}
                          onKeyDown={(ev) => {
                            if (ev.key === "Enter") commitRename();
                            else if (ev.key === "Escape") setEditingId(null);
                          }}
                        />
                      ) : (
                        <span
                          className="cell-display cell-display-editable"
                          onClick={() => beginRename(e)}
                          title="点击重命名"
                        >
                          {e.display_name}
                        </span>
                      )}
                    </div>
                  </td>
                  <td><span className="concept-tag">{e.concept}</span></td>
                  <td className="cell-family">{e.family || "—"}</td>
                  <td>{(m as any).win_rate != null ? `${(m as any).win_rate.toFixed(1)}%` : (m.is_win != null ? `${m.is_win.toFixed(1)}%` : "—")}</td>
                  <td>{(m as any).avg_ret != null ? `+${(m as any).avg_ret.toFixed(2)}%` : (m.is_avg != null ? `+${m.is_avg.toFixed(1)}%` : "—")}</td>
                  <td>{(m as any).trades ?? m.trades ?? "—"}</td>
                  <td>{(m as any).stocks ?? "—"}</td>
                  <td className={(m as any).avg_mdd > 15 ? "stat-bad" : "stat-warn"}>{(m as any).avg_mdd != null ? `${(m as any).avg_mdd.toFixed(1)}%` : "—"}</td>
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
                <td colSpan={13} className="empty-row">无匹配策略</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
