export type AppMode =
  | "strategy"
  | "screening"
  | "strategy-pool"
  | "live"
  | "sector"
  | "toplist"
  | "paper"
  | "quote"
  | "pool"
  | "system";

interface Props {
  mode: AppMode;
  onModeChange: (mode: AppMode) => void;
}

export function NavBar({ mode, onModeChange }: Props) {
  return (
    <nav className="app-navbar">
      <div className="navbar-brand">A股研究平台</div>
      <div className="navbar-tabs">
        {/* 顺序按用户 2026-09-08 的要求排:
            实时 / 选股 / 股票池 / 板块 / 行情 —— 常用的在前
            龙虎榜 / 策略 / 策略池 / 虚拟盘 —— 其次 */}
        <button
          className={mode === "live" ? "active" : ""}
          onClick={() => onModeChange("live")}
          title="分时 + 日K 实时看盘 (腾讯行情)"
        >
          实时
        </button>
        <button
          className={mode === "screening" ? "active" : ""}
          onClick={() => onModeChange("screening")}
        >
          选股
        </button>
        <button
          className={mode === "pool" ? "active" : ""}
          onClick={() => onModeChange("pool")}
        >
          股票池
        </button>
        <button
          className={mode === "sector" ? "active" : ""}
          onClick={() => onModeChange("sector")}
          title="概念板块热度 — 上涨占比 / 平均涨幅 / 成分股"
        >
          板块
        </button>
        <button
          className={mode === "quote" ? "active" : ""}
          onClick={() => onModeChange("quote")}
        >
          行情
        </button>
        <button
          className={mode === "toplist" ? "active" : ""}
          onClick={() => onModeChange("toplist")}
        >
          龙虎榜
        </button>
        <button
          className={mode === "strategy" ? "active" : ""}
          onClick={() => onModeChange("strategy")}
        >
          策略
        </button>
        <button
          className={mode === "strategy-pool" ? "active" : ""}
          onClick={() => onModeChange("strategy-pool")}
        >
          策略池
        </button>
        <button
          className={mode === "paper" ? "active" : ""}
          onClick={() => onModeChange("paper")}
          title="虚拟盘 — 各策略的实盘跟踪与历史回放"
        >
          虚拟盘
        </button>
      </div>
      <div className="navbar-spacer" />
      <div className="navbar-tabs">
        <button
          className={mode === "system" ? "active" : ""}
          onClick={() => onModeChange("system")}
          title="数据库状态 / K线补齐"
        >
          ⚙ 系统
        </button>
      </div>
    </nav>
  );
}
