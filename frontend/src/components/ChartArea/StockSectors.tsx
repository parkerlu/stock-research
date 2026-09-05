// K线页的板块条 —— 这只股票在哪些主题里, 那些主题今天热不热。
//
// 按板块平均涨幅降序: 排最前的就是今天带动它的那条线。
// 只显示主题概念, 宽基指数样本股(沪深300成份股之类)不是"主题", 折叠起来。
import { useEffect, useState } from "react";
import { getStockSectorsHot } from "../../api/sectors";
import type { StockSectorHot } from "../../api/sectors";

interface Props {
  symbol: string;
  onPick?: (sectorCode: string, name: string) => void;
}

export function StockSectors({ symbol, onPick }: Props) {
  const [items, setItems] = useState<StockSectorHot[]>([]);
  const [expand, setExpand] = useState(false);

  useEffect(() => {
    if (!symbol) {
      setItems([]);
      return;
    }
    let live = true;
    getStockSectorsHot(symbol)
      .then((r) => live && setItems(r.sectors ?? []))
      .catch(() => live && setItems([]));
    return () => {
      live = false;
    };
  }, [symbol]);

  const themes = items.filter((s) => s.is_theme);
  if (themes.length === 0) return null;
  const shown = expand ? themes : themes.slice(0, 8);

  return (
    <div className="stock-sectors">
      <span className="ss-label">板块</span>
      {shown.map((s) => (
        <button
          key={s.ts_code}
          className="ss-chip"
          onClick={() => onPick?.(s.ts_code, s.name)}
          title={`${s.name}\n平均 ${s.avg_pct ?? "-"}%  上涨 ${s.up}/${s.quoted}${
            s.up_ratio !== null ? ` (${s.up_ratio}%)` : ""
          }`}
        >
          <span className="ss-name">{s.name}</span>
          <span className={`ss-pct ${(s.avg_pct ?? 0) >= 0 ? "up" : "down"}`}>
            {s.avg_pct === null
              ? "—"
              : `${s.avg_pct > 0 ? "+" : ""}${s.avg_pct.toFixed(2)}%`}
          </span>
          {s.up_ratio !== null && (
            <span className="ss-ratio">{s.up_ratio.toFixed(0)}%</span>
          )}
        </button>
      ))}
      {themes.length > 8 && (
        <button className="ss-more" onClick={() => setExpand((v) => !v)}>
          {expand ? "收起" : `+${themes.length - 8}`}
        </button>
      )}
    </div>
  );
}
