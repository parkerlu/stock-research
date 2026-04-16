import { useState } from "react";
import { NavBar } from "../components/NavBar";
import type { AppMode } from "../components/NavBar";
import { SearchPanel } from "../components/SearchPanel";
import { PoolPanel } from "../components/PoolPanel";
import { StrategyPanel } from "../components/StrategyPanel";
import { ChartArea } from "../components/ChartArea";

export function QuotePage() {
  const [mode, setMode] = useState<AppMode>("quote");

  const renderPanel = () => {
    switch (mode) {
      case "pool":
        return <PoolPanel />;
      case "strategy":
        return <StrategyPanel />;
      default:
        return <SearchPanel />;
    }
  };

  return (
    <div className="app-layout">
      <NavBar mode={mode} onModeChange={setMode} />
      <div className="app-body">
        {renderPanel()}
        {mode !== "strategy" && <ChartArea />}
      </div>
    </div>
  );
}
