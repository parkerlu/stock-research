// 选股结果一键入池 —— 把当前筛出来的股票批量加进某个股票池。
//
// 支持两种: 加进已有池 / 新建一个池。新建时默认名带上指标和日期,
// 免得过几天看到一堆"新建池1/2/3"不知道哪个是哪个。
import { useCallback, useEffect, useState } from "react";
import { addStockToPool, createPoolWithStocks, listPools } from "../../api/pools";
import type { Pool } from "../../types/pool";

interface Props {
  codes: string[];
  /** 用于生成默认池名, 如 "买卖很准v4·强·09-05" */
  label: string;
}

export function AddToPool({ codes, label }: Props) {
  const [pools, setPools] = useState<Pool[]>([]);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    listPools().then(setPools).catch(() => setPools([]));
  }, [open]);

  const addTo = useCallback(async (poolId: number, poolName: string) => {
    if (codes.length === 0 || busy) return;
    setBusy(true);
    setMsg(null);
    let ok = 0;
    for (const c of codes) {
      try {
        await addStockToPool(poolId, c);
        ok += 1;
      } catch {
        /* 已在池中会报错, 跳过即可 */
      }
    }
    setMsg(`已加 ${ok}/${codes.length} 只 → ${poolName}`);
    setBusy(false);
    setTimeout(() => setOpen(false), 1200);
  }, [codes, busy]);

  const createNew = useCallback(async () => {
    if (codes.length === 0 || busy) return;
    setBusy(true);
    setMsg(null);
    const d = new Date();
    const name = `${label}·${String(d.getMonth() + 1).padStart(2, "0")}-${String(
      d.getDate()
    ).padStart(2, "0")}`;
    try {
      await createPoolWithStocks(name, codes);
      setMsg(`已建池「${name}」${codes.length} 只`);
      setTimeout(() => setOpen(false), 1200);
    } catch (e: unknown) {
      setMsg(e instanceof Error ? e.message : "建池失败");
    } finally {
      setBusy(false);
    }
  }, [codes, label, busy]);

  if (codes.length === 0) return null;

  return (
    <div className="add-pool">
      <button className="ap-trigger" onClick={() => setOpen((v) => !v)}>
        + 入池 ({codes.length})
      </button>
      {open && (
        <div className="ap-menu">
          <button className="ap-item ap-new" onClick={createNew} disabled={busy}>
            ＋ 新建池「{label}」
          </button>
          {pools.length > 0 && <div className="ap-sep">加入已有池</div>}
          {pools.map((p) => (
            <button
              key={p.id}
              className="ap-item"
              onClick={() => addTo(p.id, p.name)}
              disabled={busy}
            >
              {p.name}
            </button>
          ))}
          {msg && <div className="ap-msg">{msg}</div>}
        </div>
      )}
    </div>
  );
}
