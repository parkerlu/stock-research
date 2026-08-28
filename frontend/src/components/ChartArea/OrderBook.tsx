// 五档盘口 — 卖五..卖一 在上, 买一..买五 在下 (与交易软件一致)。
// 每行带一条按挂单量归一化的背景条, 一眼看出哪档挂得厚。
import type { OrderLevel, Snapshot } from "../../types/quote";

interface Props {
  snapshot: Snapshot | null;
}

function fmtVol(v: number): string {
  if (v >= 10000) return `${(v / 10000).toFixed(1)}万`;
  return String(v);
}

export function OrderBook({ snapshot }: Props) {
  const bids = snapshot?.bids ?? [];
  const asks = snapshot?.asks ?? [];
  if (bids.length === 0 && asks.length === 0) {
    return <div className="ob-empty">无盘口数据</div>;
  }

  const prev = snapshot?.prev_close ?? 0;
  // 量条按五档内最大挂单归一化
  const maxVol = Math.max(1, ...bids.map((b) => b.vol), ...asks.map((a) => a.vol));

  const row = (lv: OrderLevel, label: string, side: "ask" | "bid") => {
    // 价格颜色相对昨收: 高于昨收红, 低于绿 (A股惯例)
    const cls = !prev || lv.price === 0 ? "" : lv.price >= prev ? "ob-up" : "ob-down";
    const pctW = (lv.vol / maxVol) * 100;
    return (
      <div className={`ob-row ob-${side}`} key={label}>
        <span
          className="ob-bar"
          style={{ width: `${pctW}%` }}
          aria-hidden="true"
        />
        <span className="ob-label">{label}</span>
        <span className={`ob-price ${cls}`}>
          {lv.price > 0 ? lv.price.toFixed(2) : "—"}
        </span>
        <span className="ob-vol">{lv.vol > 0 ? fmtVol(lv.vol) : "—"}</span>
      </div>
    );
  };

  const askLabels = ["卖五", "卖四", "卖三", "卖二", "卖一"];
  const bidLabels = ["买一", "买二", "买三", "买四", "买五"];

  return (
    <div className="order-book">
      {/* 卖档由高到低往下排 —— 卖五在最上, 卖一贴近中间 */}
      {[...asks].slice(0, 5).reverse().map((lv, i) => row(lv, askLabels[i], "ask"))}
      <div className="ob-mid">
        <span className="ob-label">现价</span>
        <span
          className={`ob-price ${
            (snapshot?.change_pct ?? 0) >= 0 ? "ob-up" : "ob-down"
          }`}
        >
          {snapshot?.price?.toFixed(2) ?? "—"}
        </span>
        <span className="ob-vol">
          {snapshot ? `${(snapshot.change_pct >= 0 ? "+" : "")}${snapshot.change_pct.toFixed(2)}%` : ""}
        </span>
      </div>
      {bids.slice(0, 5).map((lv, i) => row(lv, bidLabels[i], "bid"))}
    </div>
  );
}
