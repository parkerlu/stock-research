import { useCallback, useEffect, useRef, useState } from "react";
import { searchStocks } from "../../api/quotes";
import type { StockInfo } from "../../types/quote";
import { usePoolStore } from "../../stores/poolStore";

interface Props {
  onClose: () => void;
}

export function AddStockDialog({ onClose }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<StockInfo[]>([]);
  const addStock = usePoolStore((s) => s.addStock);
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const doSearch = useCallback(async (q: string) => {
    if (q.length < 1) {
      setResults([]);
      return;
    }
    try {
      const data = await searchStocks(q);
      setResults(data);
    } catch {
      setResults([]);
    }
  }, []);

  useEffect(() => {
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => doSearch(query), 300);
    return () => clearTimeout(timerRef.current);
  }, [query, doSearch]);

  const handleSelect = async (stock: StockInfo) => {
    await addStock(stock.ts_code);
    onClose();
  };

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog-box" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <span>添加股票</span>
          <button onClick={onClose}>X</button>
        </div>
        <input
          autoFocus
          className="dialog-search"
          placeholder="输入代码或名称..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <ul className="dialog-results">
          {results.map((s) => (
            <li key={s.ts_code} onClick={() => handleSelect(s)}>
              <span className="code">{s.symbol}</span>
              <span className="name">{s.name}</span>
              {s.industry && <span className="industry">{s.industry}</span>}
            </li>
          ))}
          {query && results.length === 0 && <li className="empty">无结果</li>}
        </ul>
      </div>
    </div>
  );
}
