import type { Pool, PoolDetail } from "../types/pool";

const BASE = "/api/pools";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function createPool(
  name: string,
  description?: string
): Promise<{ id: number }> {
  return json(`${BASE}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description }),
  });
}

export async function listPools(): Promise<Pool[]> {
  return json(`${BASE}`);
}

export async function getPoolDetail(poolId: number): Promise<PoolDetail> {
  return json(`${BASE}/${poolId}`);
}

export async function updatePool(
  poolId: number,
  data: { name?: string; description?: string }
): Promise<void> {
  await json(`${BASE}/${poolId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function deletePool(poolId: number): Promise<void> {
  await json(`${BASE}/${poolId}`, { method: "DELETE" });
}

export async function addStockToPool(
  poolId: number,
  tsCode: string
): Promise<void> {
  await json(`${BASE}/${poolId}/stocks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ts_code: tsCode }),
  });
}

export async function removeStockFromPool(
  poolId: number,
  tsCode: string
): Promise<void> {
  await json(`${BASE}/${poolId}/stocks/${tsCode}`, { method: "DELETE" });
}
