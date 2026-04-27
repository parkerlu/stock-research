import { useEffect, useState } from "react";
import { useStrategyStore } from "../../stores/strategyStore";

interface Props {
  tsCode: string;
}

export function AdhocStrategy({ tsCode }: Props) {
  const templates = useStrategyStore((s) => s.templates);
  const fetchTemplates = useStrategyStore((s) => s.fetchTemplates);
  const adhocRunning = useStrategyStore((s) => s.adhocRunning);
  const adhocResult = useStrategyStore((s) => s.adhocResult);
  const runAdhocTemplate = useStrategyStore((s) => s.runAdhocTemplate);
  const clearAdhocResult = useStrategyStore((s) => s.clearAdhocResult);

  const [selected, setSelected] = useState<string>("");

  useEffect(() => {
    if (templates.length === 0) fetchTemplates();
  }, [fetchTemplates, templates.length]);

  // Auto-select first template once list loads, or if current selection is invalid
  useEffect(() => {
    if (templates.length === 0) return;
    if (!selected || !templates.some((t) => t.template_id === selected)) {
      setSelected(templates[0].template_id);
    }
  }, [templates, selected]);

  const onRun = () => {
    if (!tsCode || !selected) return;
    runAdhocTemplate(tsCode, selected);
  };

  const m = adhocResult?.metrics;
  // Compute ≥10% accuracy client-side for maimai_reversal_v2/v3; ≥6% for v1
  let accuracy: number | null = null;
  let threshold = 6;
  if (adhocResult) {
    if (selected.includes("v2") || selected.includes("v3")) threshold = 10;
    const trades = adhocResult.trades;
    if (trades.length > 0) {
      const acc = trades.filter((t) => {
        const pct = ((t.exit_price - t.entry_price) / t.entry_price) * 100;
        return pct >= threshold;
      }).length;
      accuracy = (acc / trades.length) * 100;
    }
  }

  return (
    <div className="adhoc-strategy">
      <div className="adhoc-strategy-header">快速试策略</div>
      <div className="adhoc-strategy-row">
        <select
          className="adhoc-strategy-select"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={adhocRunning}
        >
          {templates.map((t) => (
            <option key={t.template_id} value={t.template_id}>
              {t.template_id} — {t.name}
            </option>
          ))}
        </select>
        <button
          className="adhoc-strategy-btn"
          onClick={onRun}
          disabled={!tsCode || adhocRunning}
        >
          {adhocRunning ? "运行中..." : "运行"}
        </button>
        {adhocResult && (
          <button
            className="adhoc-strategy-btn adhoc-strategy-btn-clear"
            onClick={clearAdhocResult}
          >
            清除
          </button>
        )}
      </div>

      {adhocResult && m && (
        <div className="adhoc-strategy-result">
          <div className="adhoc-result-row">
            <span>交易数:</span><b>{m.total_trades}</b>
            <span>准确率(≥{threshold}%):</span>
            <b>{accuracy === null ? "—" : `${accuracy.toFixed(1)}%`}</b>
          </div>
          <div className="adhoc-result-row">
            <span>胜率:</span><b>{(m.win_rate * 100).toFixed(1)}%</b>
            <span>总收益:</span>
            <b className={m.net_profit_pct >= 0 ? "profit-up" : "profit-down"}>
              {m.net_profit_pct.toFixed(2)}%
            </b>
          </div>
          <div className="adhoc-result-row">
            <span>年化:</span>
            <b className={m.annualized_return >= 0 ? "profit-up" : "profit-down"}>
              {(m.annualized_return * 100).toFixed(2)}%
            </b>
            <span>最大回撤:</span>
            <b className="profit-down">{(m.max_drawdown * 100).toFixed(2)}%</b>
          </div>
          <div className="adhoc-result-hint">
            买卖点已标注到 K 线图（切到"策略"模式查看）
          </div>
        </div>
      )}
    </div>
  );
}
