import { useQuoteStore } from "../../stores/quoteStore";

export function SnapshotCard() {
  const snapshot = useQuoteStore((s) => s.snapshot);
  const currentName = useQuoteStore((s) => s.currentName);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const addFavorite = useQuoteStore((s) => s.addFavorite);

  if (!snapshot || !currentSymbol) {
    return <div className="snapshot-card empty">选择一只股票查看行情</div>;
  }

  const isUp = snapshot.change >= 0;
  const colorClass = isUp ? "up" : "down";

  return (
    <div className="snapshot-card">
      <div className="snapshot-header">
        <div>
          <div className="stock-name">{currentName}</div>
          <div className="stock-code">{currentSymbol}</div>
        </div>
        <button
          className="fav-btn"
          onClick={() => addFavorite(currentSymbol)}
          title="收藏"
        >
          ⭐
        </button>
      </div>
      <div className={`snapshot-price ${colorClass}`}>
        {snapshot.price.toFixed(2)}
      </div>
      <div className={`snapshot-change ${colorClass}`}>
        {isUp ? "+" : ""}
        {snapshot.change.toFixed(2)} ({isUp ? "+" : ""}
        {snapshot.change_pct.toFixed(2)}%)
      </div>
      <div className="snapshot-grid">
        <div><span className="label">开</span><span>{snapshot.open.toFixed(2)}</span></div>
        <div><span className="label">高</span><span>{snapshot.high.toFixed(2)}</span></div>
        <div><span className="label">低</span><span>{snapshot.low.toFixed(2)}</span></div>
        <div><span className="label">量</span><span>{(snapshot.vol / 10000).toFixed(1)}万</span></div>
        <div><span className="label">额</span><span>{(snapshot.amount / 100000000).toFixed(1)}亿</span></div>
        <div><span className="label">换</span><span>{snapshot.turnover.toFixed(2)}%</span></div>
      </div>
    </div>
  );
}
