import { MainChart } from "./MainChart";
import type { Timeframe } from "../../types/quote";

const LINKED_TFS: { label: string; tf: Timeframe }[] = [
  { label: "日线", tf: "1d" },
  { label: "周线", tf: "1w" },
  { label: "月线", tf: "1m" },
];

export function LinkedView() {
  return (
    <div className="linked-view">
      {LINKED_TFS.map(({ label, tf }) => (
        <div key={tf} className="linked-pane">
          <div className="linked-label">{label}</div>
          <MainChart timeframe={tf} className="linked-chart" />
        </div>
      ))}
    </div>
  );
}
