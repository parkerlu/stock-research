// 一键加自选 —— 放在 K 线工具栏, 三个页面共用。
import { useCallback, useEffect, useState } from "react";
import { addFavorite, getFavorites, removeFavorite } from "../../api/quotes";
import { useQuoteStore } from "../../stores/quoteStore";

export function FavButton({ symbol }: { symbol: string }) {
  const [fav, setFav] = useState(false);
  const [busy, setBusy] = useState(false);
  const currentName = useQuoteStore((s) => s.currentName);

  const refresh = useCallback(async () => {
    if (!symbol) return;
    try {
      const list = await getFavorites();
      setFav(list.some((f) => f.ts_code === symbol));
    } catch {
      /* 拿不到就当未收藏, 不阻塞 */
    }
  }, [symbol]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const toggle = async () => {
    if (!symbol || busy) return;
    setBusy(true);
    // 乐观更新: 点击立刻变色, 失败再回滚 —— 网络往返几百毫秒, 等结果会显得很钝
    const next = !fav;
    setFav(next);
    try {
      if (next) await addFavorite(symbol);
      else await removeFavorite(symbol);
      window.dispatchEvent(new CustomEvent("favorites-changed"));
    } catch {
      setFav(!next);
    } finally {
      setBusy(false);
    }
  };

  if (!symbol) return null;
  return (
    <button
      className={`fav-btn${fav ? " on" : ""}`}
      onClick={toggle}
      disabled={busy}
      title={fav ? `已收藏 ${currentName || symbol}，点击移除` : "加入自选"}
    >
      {fav ? "★" : "☆"} 自选
    </button>
  );
}
