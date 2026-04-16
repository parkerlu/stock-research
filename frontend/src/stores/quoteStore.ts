// frontend/src/stores/quoteStore.ts
import { create } from "zustand";
import type {
  FavoriteItem,
  SearchHistoryItem,
  Snapshot,
  Timeframe,
} from "../types/quote";
import {
  addFavorite as apiFav,
  getFavorites,
  getSearchHistory,
  getSnapshot,
  removeFavorite as apiUnfav,
} from "../api/quotes";

interface QuoteState {
  currentSymbol: string;
  currentName: string;
  setCurrentStock: (symbol: string, name: string) => void;

  timeframe: Timeframe;
  setTimeframe: (tf: Timeframe) => void;

  linkedMode: boolean;
  toggleLinkedMode: () => void;

  snapshot: Snapshot | null;
  fetchSnapshot: () => Promise<void>;

  history: SearchHistoryItem[];
  fetchHistory: () => Promise<void>;

  favorites: FavoriteItem[];
  fetchFavorites: () => Promise<void>;
  addFavorite: (tsCode: string) => Promise<void>;
  removeFavorite: (tsCode: string) => Promise<void>;
}

export const useQuoteStore = create<QuoteState>((set, get) => ({
  currentSymbol: "",
  currentName: "",
  setCurrentStock: (symbol, name) => {
    set({ currentSymbol: symbol, currentName: name });
    get().fetchSnapshot();
    get().fetchHistory();
  },

  timeframe: "1d",
  setTimeframe: (tf) => set({ timeframe: tf }),

  linkedMode: false,
  toggleLinkedMode: () => set((s) => ({ linkedMode: !s.linkedMode })),

  snapshot: null,
  fetchSnapshot: async () => {
    const { currentSymbol } = get();
    if (!currentSymbol) return;
    try {
      const data = await getSnapshot(currentSymbol);
      set({ snapshot: data });
    } catch {
      set({ snapshot: null });
    }
  },

  history: [],
  fetchHistory: async () => {
    try {
      const data = await getSearchHistory();
      set({ history: data });
    } catch {
      /* ignore */
    }
  },

  favorites: [],
  fetchFavorites: async () => {
    try {
      const data = await getFavorites();
      set({ favorites: data });
    } catch {
      /* ignore */
    }
  },
  addFavorite: async (tsCode) => {
    await apiFav(tsCode);
    await get().fetchFavorites();
  },
  removeFavorite: async (tsCode) => {
    await apiUnfav(tsCode);
    await get().fetchFavorites();
  },
}));
