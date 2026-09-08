// 训练指标选股 —— 用本项目训出来的指标筛股票。
//
// 信号是盘后批量算好入库的(build_maimai / build_pump), 这里直接查,
// 不跑扫描任务, 所以是秒出。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listTrained, screenByTrained } from "../../api/trained";
import type { ScreenItem, TrainedMeta } from "../../api/trained";
import { useQuoteStore } from "../../stores/quoteStore";
import { AddToPool } from "./AddToPool";

const DAY_OPTIONS = [1, 3, 5, 10, 20, 60, 90];

export function TrainedScreen() {
  const jumpToSignal = useQuoteStore((s) => s.jumpToSignal);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);

  const [metas, setMetas] = useState<TrainedMeta[]>([]);
  const [indicator, setIndicator] = useState<string>("pump");
  const [grade, setGrade] = useState<string>("强");
  const [days, setDays] = useState(5);
  const [items, setItems] = useState<ScreenItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  // 搜索 + 自定义顺序。顺序存 localStorage —— 这是个人偏好, 不该每次重开就丢。
  const [q, setQ] = useState("");
  const [order, setOrder] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem("screen.ind.order") ?? "[]"); }
    catch { return []; }
  });
  const dragKey = useRef<string | null>(null);
  useEffect(() => {
    localStorage.setItem("screen.ind.order", JSON.stringify(order));
  }, [order]);

  // ⚠️ 打开页面默认选【你排在第一位的那个】, 不是后端返回的第一个。
  //    只在首次加载时定一次 —— 之后拖动排序不应该把当前选择跳走,
  //    那样正在看的结果会被无声换掉。
  const pickedOnce = useRef(false);
  useEffect(() => {
    listTrained()
      .then((r) => {
        setMetas(r.indicators);
        if (!r.indicators.length || pickedOnce.current) return;
        pickedOnce.current = true;
        const pos = new Map(order.map((k, i) => [k, i]));
        const first = [...r.indicators].sort(
          (a, b) => (pos.get(a.key) ?? 1e9) - (pos.get(b.key) ?? 1e9)
        )[0];
        setIndicator(first.key);
        setGrade(first.default_grade);
        // 稀疏指标自带建议窗口 —— 不套用的话切过去就是一片空白
        setDays(first.default_days ?? 5);
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

  // 排好序的清单: 自定义顺序在前, 新增的(order 里没有的)排在后面。
  // ⚠️ 不能直接按 order 过滤 —— 后端新增指标时 order 里没有它, 会整个消失。
  const ordered = useMemo(() => {
    const pos = new Map(order.map((k, i) => [k, i]));
    return [...metas].sort(
      (a, b) => (pos.get(a.key) ?? 1e9) - (pos.get(b.key) ?? 1e9)
    );
  }, [metas, order]);

  const shown = useMemo(() => {
    const kw = q.trim().toLowerCase();
    if (!kw) return ordered;
    return ordered.filter(
      (m) => m.label.toLowerCase().includes(kw) ||
             m.key.toLowerCase().includes(kw) ||
             (m.desc ?? "").toLowerCase().includes(kw)
    );
  }, [ordered, q]);

  // 拖拽换位。⚠️ 只在【没有搜索】时允许 —— 过滤状态下拖动, 落点对应的是
  // 过滤后的位置, 写回全量顺序会错乱。
  const canDrag = q.trim() === "";
  const onDrop = (targetKey: string) => {
    const from = dragKey.current;
    dragKey.current = null;
    if (!from || from === targetKey) return;
    const keys = ordered.map((m) => m.key);
    const i = keys.indexOf(from);
    const j = keys.indexOf(targetKey);
    if (i < 0 || j < 0) return;
    keys.splice(j, 0, ...keys.splice(i, 1));
    setOrder(keys);
  };

  return (
    <div className="trained-screen">
      <div className="ts-side">
        <div className="ts-title">训练指标</div>
        <input
          className="ts-search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="搜索指标…"
        />
        {order.length > 0 && (
          <button className="ts-reset" onClick={() => setOrder([])}
                  title="恢复默认顺序">↺ 默认顺序</button>
        )}
        {metas.length === 0 && <div className="ts-empty">加载中…</div>}
        {metas.length > 0 && shown.length === 0 && (
          <div className="ts-empty">没有匹配「{q}」的指标</div>
        )}
        {shown.map((m) => (
          <button
            key={m.key}
            className={`ts-ind${indicator === m.key ? " on" : ""}${canDrag ? " draggable" : ""}`}
            draggable={canDrag}
            onDragStart={() => { dragKey.current = m.key; }}
            onDragOver={(e) => { if (canDrag) e.preventDefault(); }}
            onDrop={(e) => { e.preventDefault(); onDrop(m.key); }}
            title={canDrag ? "拖动可调整顺序" : "搜索时不能拖动"}
            onClick={() => {
              setIndicator(m.key);
              setGrade(m.default_grade);
              // 稀疏指标(v5)自带建议窗口 —— 不套用的话切过去就是一片空白
              setDays(m.default_days ?? 5);
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
                onClick={() => jumpToSignal(x.ts_code, x.name, x.date)}
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
