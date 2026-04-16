import { useState } from "react";
import { usePoolStore } from "../../stores/poolStore";
import { useQuoteStore } from "../../stores/quoteStore";
import { AddStockDialog } from "./AddStockDialog";

export function PoolDetail() {
  const poolDetail = usePoolStore((s) => s.poolDetail);
  const loading = usePoolStore((s) => s.loading);
  const removeStock = usePoolStore((s) => s.removeStock);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const [showAdd, setShowAdd] = useState(false);

  if (!poolDetail) {
    return <div className="pool-detail-empty">选择一个股票池</div>;
  }

  if (loading) {
    return <div className="pool-detail-empty">加载中...</div>;
  }

  return (
    <div className="pool-detail">
      <div className="pool-detail-header">
        <span className="pool-detail-name">{poolDetail.name}</span>
        <button className="pool-add-btn" onClick={() => setShowAdd(true)}>+</button>
      </div>
      {poolDetail.description && (
        <div className="pool-detail-desc">{poolDetail.description}</div>
      )}
      <table className="pool-stock-table">
        <thead>
          <tr>
            <th>代码</th>
            <th>名称</th>
            <th>收盘</th>
            <th>涨跌%</th>
            <th>成交量</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {poolDetail.stocks.map((s) => (
            <tr
              key={s.ts_code}
              onClick={() => setCurrentStock(s.ts_code, s.name)}
              className="pool-stock-row"
            >
              <td className="code">{s.symbol}</td>
              <td>{s.name}</td>
              <td>{s.close != null ? s.close.toFixed(2) : "-"}</td>
              <td className={s.change_pct != null ? (s.change_pct >= 0 ? "up" : "down") : ""}>
                {s.change_pct != null ? `${s.change_pct > 0 ? "+" : ""}${s.change_pct.toFixed(2)}%` : "-"}
              </td>
              <td>{s.volume != null ? (s.volume / 10000).toFixed(0) + "万" : "-"}</td>
              <td>
                <button
                  className="remove-btn"
                  onClick={(e) => { e.stopPropagation(); removeStock(s.ts_code); }}
                >
                  x
                </button>
              </td>
            </tr>
          ))}
          {poolDetail.stocks.length === 0 && (
            <tr>
              <td colSpan={6} className="empty-cell">池内暂无股票，点击 + 添加</td>
            </tr>
          )}
        </tbody>
      </table>
      {showAdd && <AddStockDialog onClose={() => setShowAdd(false)} />}
    </div>
  );
}
