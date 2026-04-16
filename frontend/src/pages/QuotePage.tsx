import { useState } from "react";
import { NavBar } from "../components/NavBar";
import type { AppMode } from "../components/NavBar";
import { SearchPanel } from "../components/SearchPanel";
import { PoolPanel } from "../components/PoolPanel";
import { ChartArea } from "../components/ChartArea";

export function QuotePage() {
  const [mode, setMode] = useState<AppMode>("quote");

  return (
    <div className="app-layout">
      <NavBar mode={mode} onModeChange={setMode} />
      <div className="app-body">
        {mode === "quote" ? <SearchPanel /> : <PoolPanel />}
        <ChartArea />
      </div>
    </div>
  );
}
