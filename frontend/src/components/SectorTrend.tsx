// 板块 30 日走势 —— 平均涨幅柱 + 上涨占比线, 叠在一起看。
//
// 为什么两条一起画: 平均涨幅说"涨了多少", 上涨占比说"多少票在涨"。
// 平均 +2% 但占比 40% = 少数权重股拉的; 平均 +2% 且占比 90% = 整体在动。
// 只看其中一条会误判主题的真实广度。
import { useEffect, useMemo, useState } from "react";
import { getSectorHistory } from "../api/sectors";
import type { SectorHistPoint } from "../api/sectors";

interface Props {
  sectorCode: string;
  days?: number;
}

const H = 132;
const PAD = { l: 36, r: 34, t: 12, b: 18 };

export function SectorTrend({ sectorCode, days = 30 }: Props) {
  const [items, setItems] = useState<SectorHistPoint[]>([]);
  const [w, setW] = useState(560);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    if (!sectorCode) return;
    let live = true;
    getSectorHistory(sectorCode, days)
      .then((r) => live && setItems(r.items ?? []))
      .catch(() => live && setItems([]));
    return () => {
      live = false;
    };
  }, [sectorCode, days]);

  const model = useMemo(() => {
    if (items.length === 0) return null;
    const pcts = items.map((x) => x.avg_pct ?? 0);
    const maxAbs = Math.max(1, ...pcts.map(Math.abs));
    return { pcts, maxAbs, last: items[items.length - 1] };
  }, [items]);

  const setRef = (el: HTMLDivElement | null) => {
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      if (r.width > 0) setW(Math.floor(r.width));
    });
    ro.observe(el);
  };

  if (!model) return null;
  const plotW = Math.max(w - PAD.l - PAD.r, 10);
  const plotH = H - PAD.t - PAD.b;
  const zeroY = PAD.t + plotH / 2;
  const bw = Math.max(2, plotW / items.length - 2);
  const xOf = (i: number) => PAD.l + (i + 0.5) * (plotW / items.length);
  const yPct = (v: number) => zeroY - (v / model.maxAbs) * (plotH / 2 - 4);
  const yRatio = (v: number) => PAD.t + plotH - (v / 100) * plotH;

  const ratioPath = items
    .map((x, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yRatio(x.up_ratio ?? 50).toFixed(1)}`)
    .join(" ");
  const hv = hover !== null ? items[hover] : model.last;

  return (
    <div className="sector-trend" ref={setRef}>
      <div className="st-head">
        <span className="st-title">近{days}日</span>
        <span className="st-cum">
          累计<b className={model.last.cum_pct >= 0 ? "up" : "down"}>
            {model.last.cum_pct > 0 ? "+" : ""}{model.last.cum_pct.toFixed(2)}%
          </b>
        </span>
      </div>
      {hv && (
        <div className="st-sub">
          {hv.date.slice(5)} 均
          <b className={(hv.avg_pct ?? 0) >= 0 ? "up" : "down"}>
            {(hv.avg_pct ?? 0) > 0 ? "+" : ""}{(hv.avg_pct ?? 0).toFixed(2)}%
          </b>
          <span className="st-updn">涨 {hv.up}/{hv.n}</span>
        </div>
      )}
      <svg width={w} height={H} onMouseLeave={() => setHover(null)}>
        {/* 上涨占比 50% 参考线 + 涨幅 0 轴 */}
        <line x1={PAD.l} y1={zeroY} x2={w - PAD.r} y2={zeroY}
              stroke="#3a3f4b" strokeWidth={1} />
        <line x1={PAD.l} y1={yRatio(50)} x2={w - PAD.r} y2={yRatio(50)}
              stroke="#3a3f4b" strokeWidth={1} strokeDasharray="3 4" opacity={0.5} />
        <text x={4} y={zeroY + 3} fontSize={9} fill="#8b909a">0%</text>
        <text x={w - PAD.r + 4} y={yRatio(50) + 3} fontSize={9} fill="#8b909a">50%</text>
        <text x={w - PAD.r + 4} y={PAD.t + 8} fontSize={9} fill="#8b909a">100%</text>

        {items.map((x, i) => {
          const v = x.avg_pct ?? 0;
          const y = yPct(v);
          return (
            <rect
              key={x.date}
              x={xOf(i) - bw / 2}
              y={Math.min(y, zeroY)}
              width={bw}
              height={Math.max(1, Math.abs(zeroY - y))}
              fill={v >= 0 ? "#eb5454" : "#26a69a"}
              opacity={hover === null || hover === i ? 0.85 : 0.4}
              onMouseEnter={() => setHover(i)}
            />
          );
        })}
        <path d={ratioPath} fill="none" stroke="#f0a020" strokeWidth={1.4} opacity={0.9} />
        {hover !== null && (
          <line x1={xOf(hover)} y1={PAD.t} x2={xOf(hover)} y2={PAD.t + plotH}
                stroke="#ffffff44" strokeWidth={1} />
        )}
      </svg>
      <div className="st-legend">
        <span><i className="lg-bar" />平均涨幅</span>
        <span><i className="lg-line" />上涨占比</span>
      </div>
    </div>
  );
}
