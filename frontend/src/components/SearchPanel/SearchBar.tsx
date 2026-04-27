import { useCallback, useEffect, useRef, useState } from "react";
import { searchStocks } from "../../api/quotes";
import type { StockInfo } from "../../types/quote";
import { useQuoteStore } from "../../stores/quoteStore";

export function SearchBar() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<StockInfo[]>([]);
  const [open, setOpen] = useState(false);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const doSearch = useCallback(async (q: string) => {
    if (q.length < 1) {
      setResults([]);
      return;
    }
    try {
      const data = await searchStocks(q);
      setResults(data);
      setOpen(true);
    } catch {
      setResults([]);
    }
  }, []);

  useEffect(() => {
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => doSearch(query), 300);
    return () => clearTimeout(timerRef.current);
  }, [query, doSearch]);

  const handleSelect = (stock: StockInfo) => {
    setCurrentStock(stock.ts_code, stock.name);
    setQuery("");
    setOpen(false);
  };

  return (
    <div className="search-bar">
      <input
        type="text"
        placeholder="输入代码或名称..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => results.length > 0 && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
      />
      {open && results.length > 0 && (
        <ul className="search-dropdown">
          {results.map((s) => (
            <li key={s.ts_code} onMouseDown={() => handleSelect(s)}>
              <span className="code">{s.symbol}</span>
              <span className="name">{s.name}</span>
              {s.industry && <span className="industry">{s.industry}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
