/** 实时页的「今日筹码买点」快捷条 —— 点一下直接切过去看盘。
 *
 * ⚠️ 为什么实时页需要这个: 筹码模型每天只出全市场 Top1%(约 54 只), 落在自选股上
 * 的概率很低 —— 实测 12 只自选里只有 4 只在过去 70 天出现过买点。所以光把标记
 * 接进实时页是不够用的: 勾了看不到东西, 会以为坏了。这里把当天的名单直接摆出来。
 *
 * ⚠️ 信号是【盘后】算的, 对应的动作是【次日开盘买入】—— 与训练标签一致。
 * 所以开市时这里显示的是"昨天收盘选出、今天该买"的票, 不是盘中实时选股。
 * 文案上要说清楚, 否则很容易被当成盘中信号。
 */
import { useEffect, useState } from "react";
import { screenByTrained } from "../../api/trained";
import { useQuoteStore } from "../../stores/quoteStore";

type Item = {
  ts_code: string; name: string; date: string;
  score: number; price: number | null; chg: number | null;
  p_up?: number; p_dn?: number;
};

export function ChipsToday() {
  const [items, setItems] = useState<Item[]>([]);
  const [open, setOpen] = useState(
    () => localStorage.getItem("live.chipsToday") === "1"
  );
  const [err, setErr] = useState(false);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);

  useEffect(() => {
    localStorage.setItem("live.chipsToday", open ? "1" : "0");
    if (!open || items.length) return;
    let live = true;
    screenByTrained("chips", "强", 1, 40)
      .then((r) => live && setItems((r.items ?? []) as Item[]))
      .catch(() => live && setErr(true));
    return () => { live = false; };
  }, [open]);

  const day = items[0]?.date;
  return (
    <div className="chips-today">
      <button className="chips-today-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "▾" : "▸"} 今日筹码买点
        {open && day && <span className="chips-today-day">{day} 收盘选出 · 次日开盘买</span>}
        {!open && <span className="chips-today-day">全市场 Top1%</span>}
      </button>
      {open && (
        <div className="chips-today-list">
          {err && <span className="chips-today-empty">取数失败</span>}
          {!err && !items.length && <span className="chips-today-empty">加载中…</span>}
          {items.map((it) => (
            <button
              key={it.ts_code}
              className={`chips-chip${it.ts_code === currentSymbol ? " on" : ""}`}
              title={`净期望 ${(it.score * 100).toFixed(2)}%　P涨 ${it.p_up ?? "-"}%　P跌 ${it.p_dn ?? "-"}%`}
              onClick={() => setCurrentStock(it.ts_code, it.name)}
            >
              <span className="chips-chip-name">{it.name || it.ts_code}</span>
              <span className="chips-chip-ev">{(it.score * 100).toFixed(1)}%</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
