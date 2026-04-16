import { useEffect, useState } from "react";
import { usePoolStore } from "../../stores/poolStore";

export function PoolList() {
  const pools = usePoolStore((s) => s.pools);
  const currentPoolId = usePoolStore((s) => s.currentPoolId);
  const fetchPools = usePoolStore((s) => s.fetchPools);
  const selectPool = usePoolStore((s) => s.selectPool);
  const createPool = usePoolStore((s) => s.createPool);
  const deletePool = usePoolStore((s) => s.deletePool);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");

  useEffect(() => {
    fetchPools();
  }, [fetchPools]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    await createPool(newName.trim());
    setNewName("");
    setCreating(false);
  };

  return (
    <div className="pool-list-section">
      <div className="pool-list-header">
        <span>股票池</span>
        <button className="pool-add-btn" onClick={() => setCreating(true)}>+</button>
      </div>
      {creating && (
        <div className="pool-create-row">
          <input
            autoFocus
            placeholder="池名称"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
          />
          <button onClick={handleCreate}>OK</button>
          <button onClick={() => { setCreating(false); setNewName(""); }}>X</button>
        </div>
      )}
      <ul className="pool-list">
        {pools.map((p) => (
          <li
            key={p.id}
            className={currentPoolId === p.id ? "active" : ""}
            onClick={() => selectPool(p.id)}
          >
            <span className="pool-name">{p.name}</span>
            <span className="pool-count">{p.stock_count}</span>
            <button
              className="remove-btn"
              onClick={(e) => { e.stopPropagation(); deletePool(p.id); }}
            >
              x
            </button>
          </li>
        ))}
        {pools.length === 0 && <li className="empty">暂无股票池</li>}
      </ul>
    </div>
  );
}
