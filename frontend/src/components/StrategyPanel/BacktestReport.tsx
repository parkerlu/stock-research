import type { BacktestReport as ReportType } from "../../types/strategy";

interface Props {
  report: ReportType;
  onClose: () => void;
}

function pct(v: number): string {
  return (v * 100).toFixed(2) + "%";
}

export function BacktestReport({ report, onClose }: Props) {
  const m = report.metrics;

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="report-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <span>回测报告</span>
          <button onClick={onClose}>✕</button>
        </div>

        {report.status === "failed" ? (
          <div className="report-error">回测失败</div>
        ) : !m ? (
          <div className="report-loading">计算中...</div>
        ) : (
          <div className="report-body">
            <div className="report-metrics">
              <div className="metric-card">
                <span className="metric-label">净利润</span>
                <span className={`metric-value ${m.net_profit > 0 ? "up" : "down"}`}>
                  {m.net_profit.toFixed(2)}
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">年化收益</span>
                <span className={`metric-value ${m.annualized_return > 0 ? "up" : "down"}`}>
                  {pct(m.annualized_return)}
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">最大回撤</span>
                <span className="metric-value">{pct(m.max_drawdown)}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">胜率</span>
                <span className="metric-value">{pct(m.win_rate)}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">交易次数</span>
                <span className="metric-value">{m.total_trades}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">盈亏比</span>
                <span className="metric-value">{m.profit_factor.toFixed(2)}</span>
              </div>
            </div>

            {report.trades && report.trades.length > 0 && (
              <div className="report-trades">
                <h4>交易明细</h4>
                <table className="trades-table">
                  <thead>
                    <tr>
                      <th>买入日期</th>
                      <th>卖出日期</th>
                      <th>买入价</th>
                      <th>卖出价</th>
                      <th>数量</th>
                      <th>盈亏</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.trades.map((t, i) => (
                      <tr key={i}>
                        <td>{t.entry_date}</td>
                        <td>{t.exit_date}</td>
                        <td>{t.entry_price.toFixed(2)}</td>
                        <td>{t.exit_price.toFixed(2)}</td>
                        <td>{t.shares.toFixed(0)}</td>
                        <td className={t.pnl > 0 ? "up" : "down"}>
                          {t.pnl.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {report.equity_curve && report.equity_curve.length > 0 && (
              <div className="report-equity">
                <h4>资金曲线</h4>
                <div className="equity-chart-placeholder">
                  起始: {report.equity_curve[0].equity.toFixed(2)} →
                  结束: {report.equity_curve[report.equity_curve.length - 1].equity.toFixed(2)}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
