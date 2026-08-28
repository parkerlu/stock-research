import { useState } from "react";
import { NavBar } from "../components/NavBar";
import type { AppMode } from "../components/NavBar";
import { SearchPanel } from "../components/SearchPanel";
import { PoolPanel } from "../components/PoolPanel";
import { StrategyPanel } from "../components/StrategyPanel";
import { ScreeningPanel } from "../components/ScreeningPanel";
import { StrategyPoolPanel } from "../components/StrategyPoolPanel";
import { SystemPanel } from "../components/SystemPanel";
import { ChartArea } from "../components/ChartArea";
import { LiveView } from "../components/ChartArea/LiveView";
import { useStrategyStore } from "../stores/strategyStore";

export function QuotePage() {
  // 默认落在实时看盘 —— 打开就是看今天的盘
  const [mode, setMode] = useState<AppMode>("live");
  const tradeActions = useStrategyStore((s) => s.tradeActions);
  // tradeActions show on chart in strategy AND screening modes
  const actions = mode === "strategy" || mode === "screening" ? tradeActions : null;

  return (
    <div className="app-layout">
      <NavBar mode={mode} onModeChange={setMode} />
      <div className="app-body">
        {mode === "pool" ? (
          <>
            <PoolPanel />
            <ChartArea />
          </>
        ) : mode === "strategy" ? (
          <>
            <SearchPanel />
            <div className="strategy-main">
              <div className="strategy-chart">
                <ChartArea tradeActions={actions} />
              </div>
              <div className="strategy-table-area">
                <StrategyPanel />
              </div>
            </div>
          </>
        ) : mode === "screening" ? (
          <>
            <ScreeningPanel />
            <ChartArea tradeActions={actions} />
          </>
        ) : mode === "live" ? (
          <>
            <SearchPanel />
            <LiveView />
          </>
        ) : mode === "strategy-pool" ? (
          <StrategyPoolPanel />
        ) : mode === "system" ? (
          <SystemPanel />
        ) : (
          <>
            <SearchPanel />
            <ChartArea />
          </>
        )}
      </div>
    </div>
  );
}
