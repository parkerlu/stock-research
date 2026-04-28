const BASE = "/api/strategy-pool";

export interface PoolEntry {
  template_id: string;
  display_name: string;
  concept: string;
  family: string;
  description: string;
  metrics: {
    is_win?: number;
    is_avg?: number;
    oos_win?: number;
    oos_avg?: number;
    trades?: number;
  };
  is_active: boolean;
  sort_order: number;
  source: string;
  score?: number;
}

export async function listPool(opts?: {
  active_only?: boolean;
  concept?: string;
}): Promise<PoolEntry[]> {
  const q = new URLSearchParams();
  if (opts?.active_only) q.set("active_only", "true");
  if (opts?.concept) q.set("concept", opts.concept);
  const url = q.toString() ? `${BASE}?${q}` : BASE;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`listPool ${r.status}`);
  return r.json();
}

export async function listConcepts(): Promise<string[]> {
  const r = await fetch(`${BASE}/concepts`);
  if (!r.ok) throw new Error(`listConcepts ${r.status}`);
  return r.json();
}

export async function updatePoolEntry(
  template_id: string,
  patch: Partial<Pick<PoolEntry, "display_name" | "concept" | "description" | "is_active">>
): Promise<PoolEntry> {
  const r = await fetch(`${BASE}/${encodeURIComponent(template_id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(`updatePoolEntry ${r.status}`);
  return r.json();
}

export async function deletePoolEntry(template_id: string): Promise<void> {
  const r = await fetch(`${BASE}/${encodeURIComponent(template_id)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`deletePoolEntry ${r.status}`);
}

export async function reorderPool(order: string[]): Promise<PoolEntry[]> {
  const r = await fetch(`${BASE}/reorder`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order }),
  });
  if (!r.ok) throw new Error(`reorderPool ${r.status}`);
  return r.json();
}
