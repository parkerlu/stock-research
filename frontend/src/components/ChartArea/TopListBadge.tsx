import { useEffect, useState } from "react";
import { getStockTopList } from "../../api/toplist";

interface Row {
  date: string; close: number | null; pct_change: number | null;
  net_wan: number; buy_wan: number; sell_wan: number; reason: string;
}

/** K 线页的龙虎榜入口。
 *
 * ⚠️ 定位是**参考**, 不是信号 —— 龙虎榜盘后公布, 这些票次日平均高开 1.31%,
 * 好看的收益你买不到(见 docs/训练指标.md 验证规矩第 8 条)。
 * 所以这里只回答"这只票哪几天有大资金动过、是谁在动", 不给任何买卖提示。 */
export function TopListBadge({ symbol, onData }: {
  symbol: string;
  onData?: (rows: { date: string; net_wan: number }[]) => void;
}) {
  const [rows, setRows] = useState<Row[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setOpen(false);
    if (!symbol) { setRows([]); onData?.([]); return; }
    let live = true;
    getStockTopList(symbol)
      .then((r) => {
        if (!live) return;
        setRows(r.items as Row[]);
        onData?.((r.items as Row[]).map((x) => ({ date: x.date, net_wan: x.net_wan })));
      })
      .catch(() => { if (live) { setRows([]); onData?.([]); } });
    return () => { live = false; };
  }, [symbol]);

  if (rows.length === 0) return null;
  return (
    <div className="lhb-badge-wrap">
      <button className="lhb-badge" onClick={() => setOpen((v) => !v)}
        title="该股历史上榜记录 —— 仅供参考, 不是买卖信号">
        龙虎榜 {rows.length} 次 {open ? "▴" : "▾"}
      </button>
      {open && (
        <div className="lhb-pop">
          <div className="lhb-pop-head">
            <span>上榜记录 · 仅供参考</span>
            <button onClick={() => setOpen(false)}>×</button>
          </div>
          <div className="lhb-pop-body">
            {rows.map((r) => (
              <div key={r.date} className="lhb-item">
                <div className="lhb-line1">
                  <span className="lhb-date">{r.date}</span>
                  <span className={(r.pct_change ?? 0) >= 0 ? "up" : "down"}>
                    {r.pct_change === null ? "—"
                      : `${r.pct_change > 0 ? "+" : ""}${r.pct_change.toFixed(2)}%`}
                  </span>
                  <span className={`lhb-net ${r.net_wan >= 0 ? "up" : "down"}`}>
                    净{r.net_wan >= 0 ? "买" : "卖"} {Math.abs(r.net_wan).toLocaleString()}万
                  </span>
                </div>
                <div className="lhb-line2">
                  买 {r.buy_wan.toLocaleString()}万 · 卖 {r.sell_wan.toLocaleString()}万
                </div>
                <div className="lhb-reason">{r.reason}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
