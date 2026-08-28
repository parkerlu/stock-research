import { useEffect, useMemo } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import { useLiveQuotes } from "../../hooks/useLiveQuotes";

export function FavoriteList() {
  const favorites = useQuoteStore((s) => s.favorites);
  const fetchFavorites = useQuoteStore((s) => s.fetchFavorites);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const removeFavorite = useQuoteStore((s) => s.removeFavorite);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);

  useEffect(() => {
    fetchFavorites();
  }, [fetchFavorites]);

  const codes = useMemo(() => favorites.map((f) => f.ts_code), [favorites]);
  const quotes = useLiveQuotes(codes);

  return (
    <ul className="stock-list fav-list">
      {favorites.map((item) => {
        const q = quotes[item.ts_code];
        const pct = q?.change_pct ?? 0;
        const cls = !q ? "" : pct > 0 ? "up" : pct < 0 ? "down" : "";
        return (
          <li
            key={item.ts_code}
            className={currentSymbol === item.ts_code ? "active" : ""}
          >
            <span
              className="name clickable"
              onClick={() => setCurrentStock(item.ts_code, item.name)}
              title={item.ts_code}
            >
              {item.name}
            </span>
            <span
              className="fav-quote clickable"
              onClick={() => setCurrentStock(item.ts_code, item.name)}
            >
              {q ? (
                <>
                  <span className={`fav-price ${cls}`}>{q.price.toFixed(2)}</span>
                  <span className={`fav-pct ${cls}`}>
                    {pct >= 0 ? "+" : ""}
                    {pct.toFixed(2)}%
                  </span>
                </>
              ) : (
                <span className="fav-loading">—</span>
              )}
            </span>
            <button
              className="remove-btn"
              onClick={() => removeFavorite(item.ts_code)}
              title="取消收藏"
            >
              ✕
            </button>
          </li>
        );
      })}
      {favorites.length === 0 && <li className="empty">暂无收藏</li>}
    </ul>
  );
}
