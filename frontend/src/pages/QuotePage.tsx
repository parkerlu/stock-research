import { SearchPanel } from "../components/SearchPanel";
import { ChartArea } from "../components/ChartArea";

export function QuotePage() {
  return (
    <div className="quote-page">
      <SearchPanel />
      <ChartArea />
    </div>
  );
}
