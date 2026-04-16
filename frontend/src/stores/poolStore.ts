import { create } from "zustand";
import type { Pool, PoolDetail } from "../types/pool";
import {
  addStockToPool,
  createPool as apiCreate,
  deletePool as apiDelete,
  getPoolDetail,
  listPools,
  removeStockFromPool,
  updatePool as apiUpdate,
} from "../api/pools";

interface PoolState {
  pools: Pool[];
  currentPoolId: number | null;
  poolDetail: PoolDetail | null;
  loading: boolean;

  fetchPools: () => Promise<void>;
  selectPool: (id: number | null) => void;
  fetchDetail: (id: number) => Promise<void>;
  createPool: (name: string, description?: string) => Promise<void>;
  deletePool: (id: number) => Promise<void>;
  updatePool: (id: number, data: { name?: string; description?: string }) => Promise<void>;
  addStock: (tsCode: string) => Promise<void>;
  removeStock: (tsCode: string) => Promise<void>;
}

export const usePoolStore = create<PoolState>((set, get) => ({
  pools: [],
  currentPoolId: null,
  poolDetail: null,
  loading: false,

  fetchPools: async () => {
    try {
      const data = await listPools();
      set({ pools: data });
    } catch {
      /* ignore */
    }
  },

  selectPool: (id) => {
    set({ currentPoolId: id, poolDetail: null });
    if (id !== null) {
      get().fetchDetail(id);
    }
  },

  fetchDetail: async (id) => {
    set({ loading: true });
    try {
      const data = await getPoolDetail(id);
      set({ poolDetail: data, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  createPool: async (name, description) => {
    await apiCreate(name, description);
    await get().fetchPools();
  },

  deletePool: async (id) => {
    await apiDelete(id);
    const { currentPoolId } = get();
    if (currentPoolId === id) {
      set({ currentPoolId: null, poolDetail: null });
    }
    await get().fetchPools();
  },

  updatePool: async (id, data) => {
    await apiUpdate(id, data);
    await get().fetchPools();
    if (get().currentPoolId === id) {
      await get().fetchDetail(id);
    }
  },

  addStock: async (tsCode) => {
    const { currentPoolId } = get();
    if (currentPoolId === null) return;
    await addStockToPool(currentPoolId, tsCode);
    await get().fetchDetail(currentPoolId);
  },

  removeStock: async (tsCode) => {
    const { currentPoolId } = get();
    if (currentPoolId === null) return;
    await removeStockFromPool(currentPoolId, tsCode);
    await get().fetchDetail(currentPoolId);
  },
}));
