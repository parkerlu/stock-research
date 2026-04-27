export type AppMode =
  | "strategy"
  | "screening"
  | "strategy-pool"
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
        <button
          className={mode === "strategy" ? "active" : ""}
          onClick={() => onModeChange("strategy")}
        >
          策略
        </button>
        <button
          className={mode === "screening" ? "active" : ""}
          onClick={() => onModeChange("screening")}
        >
          选股
        </button>
        <button
          className={mode === "strategy-pool" ? "active" : ""}
          onClick={() => onModeChange("strategy-pool")}
        >
          策略池
        </button>
        <button
          className={mode === "quote" ? "active" : ""}
          onClick={() => onModeChange("quote")}
        >
          行情
        </button>
        <button
          className={mode === "pool" ? "active" : ""}
          onClick={() => onModeChange("pool")}
        >
          股票池
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
