import { useEffect } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import { useStrategyStore } from "../../stores/strategyStore";
import { StrategyList } from "./StrategyList";
import { BacktestReport } from "./BacktestReport";
import { AdhocStrategy } from "./AdhocStrategy";

export function StrategyPanel() {
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const currentName = useQuoteStore((s) => s.currentName);
  const { strategies, loading, fetchStrategies, selectedReport, clearReport } =
    useStrategyStore();

  useEffect(() => {
    if (currentSymbol) {
      fetchStrategies(currentSymbol);
    }
  }, [currentSymbol, fetchStrategies]);

  return (
    <div className="strategy-panel">
      <div className="strategy-panel-header">
        <div className="strategy-panel-title">
          {currentSymbol ? (
            <>
              <span className="stock-name">{currentName}</span>
              <span className="stock-code">{currentSymbol}</span>
            </>
          ) : (
            <span className="stock-code">请先选择股票</span>
          )}
        </div>
      </div>

      {currentSymbol && <AdhocStrategy tsCode={currentSymbol} />}

      <div className="strategy-panel-body">
        {loading ? (
          <div className="strategy-loading">加载中...</div>
        ) : strategies.length === 0 ? (
          <div className="strategy-empty">
            {currentSymbol ? "暂无策略" : "请先选择股票"}
          </div>
        ) : (
          <StrategyList strategies={strategies} tsCode={currentSymbol} />
        )}
      </div>

      {selectedReport && (
        <BacktestReport report={selectedReport} onClose={clearReport} />
      )}
    </div>
  );
}
