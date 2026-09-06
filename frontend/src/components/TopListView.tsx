import { useCallback, useEffect, useState } from "react";
import { getTopDays, getTopList } from "../api/toplist";
import type { TopDay, TopItem } from "../api/toplist";
import { useQuoteStore } from "../stores/quoteStore";
import { ChartArea } from "./ChartArea";

/** 龙虎榜。左边选交易日, 中间是当日明细, 右边 K 线。
 *
 * 覆盖率只有 1.2%(要异动才上榜), 所以不当选股工具用 —— 它回答的是
 * "这只票今天被谁买了", 是个旁证。
 * 涨幅那一列特意留着: 涨停上榜是常态, **涨幅接近 0 却上榜才是信息量最大的**
 * (价格没动但有大额资金进出), 这也是 v5 里龙虎榜那一层的价值来源。 */
export function TopListView() {
  const [days, setDays] = useState<TopDay[]>([]);
  const [day, setDay] = useState<string | null>(null);
  const [side, setSide] = useState<"all" | "buy" | "sell">("buy");
  const [items, setItems] = useState<TopItem[]>([]);
  const [loading, setLoading] = useState(false);
  const jumpToSignal = useQuoteStore((s) => s.jumpToSignal);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);

  useEffect(() => {
    getTopDays(30).then((r) => {
      setDays(r.days);
      if (r.days.length && !day) setDay(r.days[0].date);
    }).catch(() => setDays([]));
  }, []);

  const load = useCallback(() => {
    setLoading(true);
    getTopList(day ?? undefined, side)
      .then((r) => { setItems(r.items); if (!day) setDay(r.date); })
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, [day, side]);
  useEffect(() => { load(); }, [load]);

  return (
    <>
      <div className="tl-panel">
        <div className="tl-days">
          <div className="tl-title">交易日</div>
          {days.map((d) => (
            <button key={d.date}
              className={`tl-day${day === d.date ? " on" : ""}`}
              onClick={() => setDay(d.date)}>
              <span className="tl-day-date">{d.date.slice(5)}</span>
              <span className="tl-day-n">{d.count} 家</span>
              <span className={`tl-day-net ${d.net_wan >= 0 ? "up" : "down"}`}>
                {d.net_wan >= 0 ? "+" : ""}{(d.net_wan / 10000).toFixed(1)}亿
              </span>
            </button>
          ))}
        </div>

        <div className="tl-main">
          <div className="tl-bar">
            <span className="tl-bar-label">{day ?? "—"}</span>
            {([["buy", "净买入"], ["sell", "净卖出"], ["all", "全部"]] as const).map(
              ([k, lbl]) => (
                <button key={k} className={side === k ? "on" : ""}
                  onClick={() => setSide(k)}>{lbl}</button>
              ))}
            <span className="tl-bar-stat">
              {loading ? "加载中…" : `${items.length} 只`}
            </span>
          </div>

          <div className="tl-table">
            <div className="tl-head">
              <span>名称</span><span>代码</span>
              <span className="r">现价</span><span className="r">涨跌</span>
              <span className="r">换手</span><span className="r">净买(万)</span>
              <span>上榜原因</span>
            </div>
            <div className="tl-rows">
              {items.length === 0 && !loading && (
                <div className="tl-empty">这一天没有符合条件的记录</div>
              )}
              {items.map((x) => (
                <button key={x.ts_code}
                  className={`tl-row${currentSymbol === x.ts_code ? " sel" : ""}`}
                  onClick={() => day && jumpToSignal(x.ts_code, x.name, day)}>
                  <span className="tl-name">{x.name}</span>
                  <span className="tl-code">{x.ts_code.split(".")[0]}</span>
                  <span className="r">{x.close?.toFixed(2) ?? "—"}</span>
                  <span className={`r ${(x.pct_change ?? 0) >= 0 ? "up" : "down"}`}>
                    {x.pct_change === null ? "—"
                      : `${x.pct_change > 0 ? "+" : ""}${x.pct_change.toFixed(2)}%`}
                  </span>
                  <span className="r tl-dim">
                    {x.turnover_rate?.toFixed(1) ?? "—"}
                  </span>
                  <span className={`r tl-net ${x.net_wan >= 0 ? "up" : "down"}`}>
                    {x.net_wan >= 0 ? "+" : ""}{x.net_wan.toFixed(0)}
                  </span>
                  <span className="tl-reason" title={x.reason}>{x.reason}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
      <ChartArea />
    </>
  );
}
