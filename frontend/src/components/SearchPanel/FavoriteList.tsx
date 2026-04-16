import { useEffect } from "react";
import { useQuoteStore } from "../../stores/quoteStore";

export function FavoriteList() {
  const favorites = useQuoteStore((s) => s.favorites);
  const fetchFavorites = useQuoteStore((s) => s.fetchFavorites);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const removeFavorite = useQuoteStore((s) => s.removeFavorite);

  useEffect(() => {
    fetchFavorites();
  }, [fetchFavorites]);

  return (
    <ul className="stock-list">
      {favorites.map((item) => (
        <li key={item.ts_code}>
          <span
            className="name clickable"
            onClick={() => setCurrentStock(item.ts_code, item.name)}
          >
            {item.name}
          </span>
          <button
            className="remove-btn"
            onClick={() => removeFavorite(item.ts_code)}
            title="取消收藏"
          >
            ✕
          </button>
        </li>
      ))}
      {favorites.length === 0 && (
        <li className="empty">暂无收藏</li>
      )}
    </ul>
  );
}
