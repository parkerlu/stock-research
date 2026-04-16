export type AppMode = "quote" | "pool";

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
    </nav>
  );
}
