// 多周期联动 — 日/周/月三张图并排, 每张都挂上工具栏选中的指标, 且各自按
// 自己的周期取数计算 (TDX 指标的注册名带周期, 三份数据互不覆盖)。
import { useCallback, useRef } from "react";
import { MainChart } from "./MainChart";
import type { MainChartHandle } from "./MainChart";
import { useQuoteStore } from "../../stores/quoteStore";
import { useSyncIndicators } from "../../hooks/useSyncIndicators";
import type { Timeframe } from "../../types/quote";

const LINKED_TFS: { label: string; tf: Timeframe }[] = [
  { label: "日线", tf: "1d" },
  { label: "周线", tf: "1w" },
  { label: "月线", tf: "1m" },
];

interface Props {
  /** klinecharts 内置指标 (工具栏选择, 三张图共用一份选择) */
  activeIndicators?: string[];
  /** TDX 自定义指标 */
  activeTdxIndicators?: string[];
}

interface PaneProps {
  label: string;
  tf: Timeframe;
  std: string[];
  tdx: string[];
}

function LinkedPane({ label, tf, std, tdx }: PaneProps) {
  const chartRef = useRef<MainChartHandle>(null);
  const symbol = useQuoteStore((s) => s.currentSymbol);
  // useCallback 是必须的: 这个函数进了 useSyncIndicators 的依赖, 每次渲染
  // 新建闭包会让指标同步 effect 每帧重跑一遍
  const getChart = useCallback(() => chartRef.current?.getChart() ?? null, []);

  useSyncIndicators({ getChart, symbol, timeframe: tf, std, tdx });

  return (
    <div className="linked-pane">
      <div className="linked-label">{label}</div>
      <MainChart ref={chartRef} timeframe={tf} className="linked-chart" />
    </div>
  );
}

export function LinkedView({
  activeIndicators = [],
  activeTdxIndicators = [],
}: Props) {
  return (
    <div className="linked-view">
      {LINKED_TFS.map(({ label, tf }) => (
        <LinkedPane
          key={tf}
          label={label}
          tf={tf}
          std={activeIndicators}
          tdx={activeTdxIndicators}
        />
      ))}
    </div>
  );
}
