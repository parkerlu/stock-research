import { useState } from "react";
import { NavBar } from "../components/NavBar";
import type { AppMode } from "../components/NavBar";
import { SearchPanel } from "../components/SearchPanel";
import { PoolPanel } from "../components/PoolPanel";
import { StrategyPoolPanel } from "../components/StrategyPoolPanel";
import { SystemPanel } from "../components/SystemPanel";
import { ChartArea } from "../components/ChartArea";
import { LiveView } from "../components/ChartArea/LiveView";
import { SectorView } from "../components/SectorView";
import { TopListView } from "../components/TopListView";
import { TrainedScreen } from "../components/ScreeningPanel/TrainedScreen";
import { PaperPanel } from "../components/PaperPanel";
import { useStrategyStore } from "../stores/strategyStore";

export function QuotePage() {
  // 默认落在实时看盘 —— 打开就是看今天的盘
  const [mode, setMode] = useState<AppMode>("live");
  const tradeActions = useStrategyStore((s) => s.tradeActions);
  // tradeActions show on chart in strategy AND screening modes
  // ⚠️ "策略"页已于 2026-09-10 移除(策略工厂的单票参数组合与现在这套
  //    全市场模型选股不是一回事), tradeActions 现在只服务选股页。
  const actions = mode === "screening" ? tradeActions : null;

  return (
    <div className="app-layout">
      <NavBar mode={mode} onModeChange={setMode} />
      <div className="app-body">
        {mode === "toplist" ? (
          <TopListView />
        ) : mode === "pool" ? (
          <>
            <PoolPanel />
            <ChartArea />
          </>
        ) : mode === "screening" ? (
          <>
            <TrainedScreen />
            <ChartArea tradeActions={actions} />
          </>
        ) : mode === "live" ? (
          <>
            <SearchPanel />
            <LiveView />
          </>
        ) : mode === "sector" ? (
          <SectorView />
        ) : mode === "paper" ? (
          <>
            <PaperPanel />
            <ChartArea />
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
