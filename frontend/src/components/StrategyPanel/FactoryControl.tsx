import { useStrategyStore } from "../../stores/strategyStore";

interface Props {
  tsCode: string;
}

export function FactoryControl({ tsCode }: Props) {
  const { factoryJob, factoryRunning, startFactory } = useStrategyStore();

  const handleRun = () => {
    const today = new Date().toISOString().slice(0, 10);
    startFactory(tsCode, today);
  };

  const progress =
    factoryJob && factoryJob.total_candidates > 0
      ? Math.round((factoryJob.evaluated / factoryJob.total_candidates) * 100)
      : 0;

  return (
    <div className="factory-control">
      <button
        className="factory-run-btn"
        onClick={handleRun}
        disabled={factoryRunning}
      >
        {factoryRunning ? "运行中..." : "运行策略工厂"}
      </button>

      {factoryJob && (
        <div className="factory-status">
          <div className="factory-progress-bar">
            <div
              className="factory-progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="factory-progress-text">
            {factoryJob.status === "completed" ? (
              <span>完成 — 通过 {factoryJob.passed} 个策略</span>
            ) : factoryJob.status === "failed" ? (
              <span className="factory-error">失败: {factoryJob.error}</span>
            ) : (
              <span>
                已评估 {factoryJob.evaluated}/{factoryJob.total_candidates} ({progress}%)
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
