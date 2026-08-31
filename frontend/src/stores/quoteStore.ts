// frontend/src/stores/quoteStore.ts
import { create } from "zustand";
import type { TradeAction } from "../types/strategy";
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

  /** 要在K线上定位的日期 (YYYY-MM-DD)。虚拟盘点持仓时设置, ChartArea 消费后清空。
   *  用自增 seq 而不是只看日期 —— 连点同一天也要能再次定位。 */
  focusDate: string | null;
  focusSeq: number;
  /** 该股在虚拟盘里的买卖标记, 交给 MainChart 画在K线上 */
  paperMarks: TradeAction[] | null;
  /** 回放已推进到的日期 —— K线上画竖线, 提醒线右边是回放还没走到的未来 */
  replayDate: string | null;
  jumpToDate: (symbol: string, name: string, date: string,
               marks?: TradeAction[], replayDate?: string | null) => void;
  /** 回放日期单独更新 —— 点"下一日"时线要立刻跟着移动, 不用重新点股票 */
  setReplayDate: (date: string | null) => void;
  clearFocusDate: () => void;

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
    set({ currentSymbol: symbol, currentName: name, paperMarks: null, replayDate: null });
    get().fetchSnapshot();
    get().fetchHistory();
  },

  focusDate: null,
  focusSeq: 0,
  paperMarks: null,
  replayDate: null,
  jumpToDate: (symbol, name, date, marks, replayDate) => {
    const changed = get().currentSymbol !== symbol;
    set((st) => ({ currentSymbol: symbol, currentName: name,
                   focusDate: date, focusSeq: st.focusSeq + 1,
                   paperMarks: marks ?? null, replayDate: replayDate ?? null }));
    if (changed) { get().fetchSnapshot(); get().fetchHistory(); }
  },
  clearFocusDate: () => set({ focusDate: null }),
  setReplayDate: (date) => set({ replayDate: date }),

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
