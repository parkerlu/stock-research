import type { StrategyItem } from "../../types/strategy";
import { useStrategyStore } from "../../stores/strategyStore";

interface Props {
  strategies: StrategyItem[];
  tsCode: string;
}

function pct(v: number | null): string {
  if (v === null) return "-";
  return (v * 100).toFixed(2) + "%";
}

function num(v: number | null, decimals = 2): string {
  if (v === null) return "-";
  return v.toFixed(decimals);
}

export function StrategyList({ strategies, tsCode }: Props) {
  const { pinStrategy, unpinStrategy, runAndShowReport } = useStrategyStore();

  return (
    <div className="strategy-list-wrap">
      <table className="strategy-table">
        <thead>
          <tr>
            <th>Pin</th>
            <th>策略名称</th>
            <th>年化收益</th>
            <th>最大回撤</th>
            <th>胜率</th>
            <th>交易次数</th>
            <th>盈亏比</th>
            <th>净利润</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {strategies.map((s) => (
            <tr key={s.id} className={s.is_pinned ? "pinned-row" : ""}>
              <td>
                <button
                  className="pin-btn"
                  onClick={() =>
                    s.is_pinned ? unpinStrategy(s.id) : pinStrategy(s.id)
                  }
                >
                  {s.is_pinned ? "★" : "☆"}
                </button>
              </td>
              <td className="strategy-name-cell">{s.name}</td>
              <td className={s.annualized_return && s.annualized_return > 0 ? "up" : "down"}>
                {pct(s.annualized_return)}
              </td>
              <td>{pct(s.max_drawdown)}</td>
              <td>{pct(s.win_rate)}</td>
              <td>{s.total_trades ?? "-"}</td>
              <td>{num(s.profit_factor)}</td>
              <td className={s.net_profit && s.net_profit > 0 ? "up" : "down"}>
                {num(s.net_profit)}
              </td>
              <td>
                <button
                  className="report-btn"
                  onClick={() => runAndShowReport(tsCode, s.id)}
                >
                  回测
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
