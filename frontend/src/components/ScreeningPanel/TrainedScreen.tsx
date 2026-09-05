// 训练指标选股 —— 用本项目训出来的指标筛股票。
//
// 信号是盘后批量算好入库的(build_maimai / build_pump), 这里直接查,
// 不跑扫描任务, 所以是秒出。
import { useCallback, useEffect, useState } from "react";
import { listTrained, screenByTrained } from "../../api/trained";
import type { ScreenItem, TrainedMeta } from "../../api/trained";
import { useQuoteStore } from "../../stores/quoteStore";
import { AddToPool } from "./AddToPool";

const DAY_OPTIONS = [1, 3, 5, 10, 20];

export function TrainedScreen() {
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);

  const [metas, setMetas] = useState<TrainedMeta[]>([]);
  const [indicator, setIndicator] = useState<string>("pump");
  const [grade, setGrade] = useState<string>("强");
  const [days, setDays] = useState(5);
  const [items, setItems] = useState<ScreenItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    listTrained()
      .then((r) => {
        setMetas(r.indicators);
        if (r.indicators.length && !r.indicators.some((m) => m.key === indicator)) {
          setIndicator(r.indicators[0].key);
          setGrade(r.indicators[0].default_grade);
        }
      })
      .catch(() => setMetas([]));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const run = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await screenByTrained(indicator, grade, days);
      setItems(r.items);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "选股失败");
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [indicator, grade, days]);

  useEffect(() => {
    run();
  }, [run]);

  const cur = metas.find((m) => m.key === indicator);

  return (
    <div className="trained-screen">
      <div className="ts-side">
        <div className="ts-title">训练指标</div>
        {metas.length === 0 && <div className="ts-empty">加载中…</div>}
        {metas.map((m) => (
          <button
            key={m.key}
            className={`ts-ind${indicator === m.key ? " on" : ""}`}
            onClick={() => {
              setIndicator(m.key);
              setGrade(m.default_grade);
            }}
          >
            <span className="ts-ind-label">{m.label}</span>
            <span className="ts-ind-desc">{m.desc}</span>
          </button>
        ))}
      </div>

      <div className="ts-main">
        <div className="ts-bar">
          <span className="ts-bar-label">档位</span>
          {(cur?.grades ?? ["强", "中"]).map((g) => (
            <button key={g} className={grade === g ? "on" : ""} onClick={() => setGrade(g)}>
              {g}
            </button>
          ))}
          <span className="ts-bar-sep" />
          <span className="ts-bar-label">最近</span>
          {DAY_OPTIONS.map((d) => (
            <button key={d} className={days === d ? "on" : ""} onClick={() => setDays(d)}>
              {d}日
            </button>
          ))}
          <span className="ts-bar-stat">
            {loading ? "查询中…" : `${items.length} 只`}
          </span>
          <AddToPool
            codes={items.map((x) => x.ts_code)}
            label={`${cur?.label ?? indicator}·${grade}`}
          />
        </div>

        {err && <div className="ts-err">{err}</div>}

        <div className="ts-table">
          <div className="ts-head">
            <span>信号日</span>
            <span>名称</span>
            <span>代码</span>
            <span className="r">
              <span
                className="ts-help"
                title="信号强度分位(0~100)。模型给这个信号打分后, 在该指标全部历史信号里的百分位排名 —— 100 表示比历史上 100% 的同类信号都强。它衡量的是「这一次触发有多典型」, 不是预测涨幅。"
              >
                强度
              </span>
            </span>
            <span className="r">现价</span>
            <span className="r">涨跌</span>
          </div>
          <div className="ts-rows">
            {items.length === 0 && !loading && (
              <div className="ts-empty">该条件下没有股票 —— 换个档位或放宽天数</div>
            )}
            {items.map((x) => (
              <button
                key={`${x.ts_code}-${x.date}`}
                className={`ts-row${currentSymbol === x.ts_code ? " sel" : ""}`}
                onClick={() => setCurrentStock(x.ts_code, x.name)}
              >
                <span className="ts-date">{x.date.slice(5)}</span>
                <span className="ts-name">{x.name}</span>
                <span className="ts-code">{x.ts_code.split(".")[0]}</span>
                <span className="r ts-rank">{(x.rank_pct * 100).toFixed(0)}</span>
                <span className="r">{x.price?.toFixed(2) ?? "—"}</span>
                <span className={`r ${(x.chg ?? 0) >= 0 ? "up" : "down"}`}>
                  {x.chg === null ? "—" : `${x.chg > 0 ? "+" : ""}${x.chg.toFixed(2)}%`}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
