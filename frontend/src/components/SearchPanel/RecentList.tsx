import { useEffect } from "react";
import { useQuoteStore } from "../../stores/quoteStore";

export function RecentList() {
  const history = useQuoteStore((s) => s.history);
  const fetchHistory = useQuoteStore((s) => s.fetchHistory);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return (
    <ul className="stock-list">
      {history.map((item, i) => (
        <li
          key={`${item.ts_code}-${i}`}
          onClick={() => setCurrentStock(item.ts_code, item.name)}
        >
          <span className="name">{item.name}</span>
          <span className="code">{item.ts_code}</span>
        </li>
      ))}
      {history.length === 0 && (
        <li className="empty">暂无搜索记录</li>
      )}
    </ul>
  );
}
