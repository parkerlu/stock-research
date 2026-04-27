import { create } from "zustand";
import type { BacktestReport, FactoryJob, StrategyItem, TradeAction } from "../types/strategy";
import {
  getBacktestReport,
  getFactoryJob,
  listStrategies,
  listTemplates,
  pinStrategy as apiPin,
  runBacktest as apiRunBacktest,
  runFactory as apiRunFactory,
  tryTemplate,
  unpinStrategy as apiUnpin,
} from "../api/strategy";
import type { TemplateInfo, TryTemplateResult } from "../api/strategy";

interface StrategyState {
  strategies: StrategyItem[];
  loading: boolean;
  fetchStrategies: (tsCode: string) => Promise<void>;

  pinStrategy: (id: number) => Promise<void>;
  unpinStrategy: (id: number) => Promise<void>;

  factoryJob: FactoryJob | null;
  factoryRunning: boolean;
  startFactory: (tsCode: string, cutoffDate: string) => Promise<void>;
  pollFactoryJob: (jobId: string, tsCode: string) => void;
  stopPolling: () => void;

  selectedReport: BacktestReport | null;
  reportLoading: boolean;
  runAndShowReport: (tsCode: string, strategyId: number) => Promise<void>;
  clearReport: () => void;

  tradeActions: TradeAction[] | null;
  clearTradeActions: () => void;

  templates: TemplateInfo[];
  fetchTemplates: () => Promise<void>;
  adhocResult: TryTemplateResult | null;
  adhocRunning: boolean;
  runAdhocTemplate: (tsCode: string, templateId: string) => Promise<void>;
  clearAdhocResult: () => void;
}

let pollTimer: ReturnType<typeof setInterval> | null = null;

export const useStrategyStore = create<StrategyState>((set, get) => ({
  strategies: [],
  loading: false,
  fetchStrategies: async (tsCode) => {
    set({ loading: true });
    try {
      const data = await listStrategies(tsCode);
      set({ strategies: data, loading: false });
    } catch {
      set({ strategies: [], loading: false });
    }
  },

  pinStrategy: async (id) => {
    await apiPin(id);
    set((s) => ({
      strategies: s.strategies.map((st) =>
        st.id === id ? { ...st, is_pinned: true } : st
      ),
    }));
  },

  unpinStrategy: async (id) => {
    await apiUnpin(id);
    set((s) => ({
      strategies: s.strategies.map((st) =>
        st.id === id ? { ...st, is_pinned: false } : st
      ),
    }));
  },

  factoryJob: null,
  factoryRunning: false,
  startFactory: async (tsCode, cutoffDate) => {
    set({ factoryRunning: true, factoryJob: null });
    try {
      const res = await apiRunFactory(tsCode, cutoffDate);
      set({
        factoryJob: {
          job_id: res.job_id,
          status: "pending",
          total_candidates: res.total_candidates,
          evaluated: 0,
          passed: 0,
          error: null,
          created_at: null,
          completed_at: null,
        },
      });
      get().pollFactoryJob(res.job_id, tsCode);
    } catch {
      set({ factoryRunning: false });
    }
  },

  pollFactoryJob: (jobId, tsCode) => {
    get().stopPolling();
    pollTimer = setInterval(async () => {
      try {
        const job = await getFactoryJob(jobId);
        set({ factoryJob: job });
        if (job.status === "completed" || job.status === "failed") {
          get().stopPolling();
          set({ factoryRunning: false });
          if (job.status === "completed") {
            get().fetchStrategies(tsCode);
          }
        }
      } catch {
        get().stopPolling();
        set({ factoryRunning: false });
      }
    }, 2000);
  },

  stopPolling: () => {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  },

  tradeActions: null,
  clearTradeActions: () => set({ tradeActions: null }),

  selectedReport: null,
  reportLoading: false,
  runAndShowReport: async (tsCode, strategyId) => {
    set({ reportLoading: true, selectedReport: null });
    try {
      const res = await apiRunBacktest({
        ts_code: tsCode,
        strategy_id: strategyId,
        start_date: `${new Date().getFullYear() - 10}-01-01`,
        end_date: new Date().toISOString().slice(0, 10),
      });

      const poll = setInterval(async () => {
        try {
          const report = await getBacktestReport(res.id);
          if (report.status === "completed" || report.status === "failed") {
            clearInterval(poll);
            set({ selectedReport: report, tradeActions: report.actions ?? null, reportLoading: false });
          }
        } catch {
          clearInterval(poll);
          set({ reportLoading: false });
        }
      }, 1000);
    } catch {
      set({ reportLoading: false });
    }
  },

  clearReport: () => set({ selectedReport: null }),

  templates: [],
  fetchTemplates: async () => {
    try {
      const data = await listTemplates();
      set({ templates: data });
    } catch {
      set({ templates: [] });
    }
  },

  adhocResult: null,
  adhocRunning: false,
  runAdhocTemplate: async (tsCode, templateId) => {
    set({ adhocRunning: true, adhocResult: null });
    try {
      const result = await tryTemplate({ ts_code: tsCode, template_id: templateId });
      set({
        adhocResult: result,
        tradeActions: result.actions as TradeAction[],
        adhocRunning: false,
      });
    } catch {
      set({ adhocRunning: false });
    }
  },
  clearAdhocResult: () => set({ adhocResult: null, tradeActions: null }),
}));
