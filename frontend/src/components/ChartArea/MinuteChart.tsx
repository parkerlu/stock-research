// 分时图 — 自绘 SVG。klinecharts 是 K 线导向的, 分时的"以昨收为基准、
// 上下等幅、固定 242 分钟横轴"用 SVG 更直接也更可控。
import { useMemo, useRef, useState } from "react";
import type { MinuteBar, MinuteData, T0Trade } from "../../types/quote";

interface Props {
  data: MinuteData | null;
  loading?: boolean;
  /** 做 T 信号 — 模型判断该时点到收盘还有 3% 空间 */
  signals?: T0Trade[];
}

// A 股交易时段: 9:30–11:30 + 13:00–15:00 = 242 个点 (含 9:30 与收盘点)
const MORNING = 121; // 0930..1130
const TOTAL = 242;

/** "1013" -> 该点在 242 格中的位置; 非交易时间返回 -1 */
function slotOf(hhmm: string): number {
  const h = parseInt(hhmm.slice(0, 2), 10);
  const m = parseInt(hhmm.slice(2, 4), 10);
  const mins = h * 60 + m;
  const amStart = 9 * 60 + 30;
  const amEnd = 11 * 60 + 30;
  const pmStart = 13 * 60;
  const pmEnd = 15 * 60;
  if (mins >= amStart && mins <= amEnd) return mins - amStart;
  if (mins >= pmStart && mins <= pmEnd) return MORNING + (mins - pmStart);
  return -1;
}

const PAD = { left: 52, right: 52, top: 10, bottom: 18 };
const VOL_RATIO = 0.24; // 量柱占高度比例

export function MinuteChart({ data, loading, signals = [] }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 600, h: 380 });
  const [hover, setHover] = useState<number | null>(null);

  // 跟随容器尺寸
  const setRef = (el: HTMLDivElement | null) => {
    wrapRef.current = el;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) {
        setSize({ w: Math.floor(r.width), h: Math.floor(r.height) });
      }
    });
    ro.observe(el);
  };

  const model = useMemo(() => {
    if (!data || data.bars.length === 0) return null;
    const prev = data.prev_close || data.bars[0].price;

    // 价格轴以昨收为中心上下等幅 — 分时图的标准画法, 这样中线永远是昨收
    let maxDev = 0;
    let maxVol = 0;
    const slots: (MinuteBar | null)[] = new Array(TOTAL).fill(null);
    for (const b of data.bars) {
      const s = slotOf(b.time);
      if (s < 0 || s >= TOTAL) continue;
      slots[s] = b;
      maxDev = Math.max(maxDev, Math.abs(b.price - prev), Math.abs(b.avg_price - prev));
      maxVol = Math.max(maxVol, b.vol);
    }
    if (maxDev <= 0) maxDev = prev * 0.01;
    maxDev *= 1.08; // 留白
    return { prev, maxDev, maxVol, slots, hi: prev + maxDev, lo: prev - maxDev };
  }, [data]);

  const { w, h } = size;
  const plotW = Math.max(w - PAD.left - PAD.right, 10);
  const volH = (h - PAD.top - PAD.bottom) * VOL_RATIO;
  const priceH = Math.max(h - PAD.top - PAD.bottom - volH - 6, 10);

  const xOf = (slot: number) => PAD.left + (slot / (TOTAL - 1)) * plotW;
  const yOf = (price: number) => {
    if (!model) return 0;
    const t = (model.hi - price) / (model.hi - model.lo);
    return PAD.top + t * priceH;
  };

  // 鼠标位置 -> 最近的有数据的分钟
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!model) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const raw = Math.round(((x - PAD.left) / plotW) * (TOTAL - 1));
    const slot = Math.max(0, Math.min(TOTAL - 1, raw));
    setHover(slot);
  };

  if (loading && !model) {
    return <div className="minute-chart-empty">分时加载中…</div>;
  }
  if (!model) {
    return <div className="minute-chart-empty">暂无分时数据</div>;
  }

  // 价格线 / 均价线路径 (中途停牌的空档断开)
  const pricePts: string[] = [];
  const avgPts: string[] = [];
  let lastSlot = -1;
  model.slots.forEach((b, s) => {
    if (!b) return;
    // 首个点必须是 M —— 以 L 开头的 path 不合法, 浏览器会整条静默丢弃。
    // (lastSlot 初值 -1 与 s=0 时的 s-1 相等, 这里必须显式排除)
    const cmd = lastSlot >= 0 && lastSlot === s - 1 ? "L" : "M";
    pricePts.push(`${cmd}${xOf(s).toFixed(1)},${yOf(b.price).toFixed(1)}`);
    avgPts.push(`${cmd}${xOf(s).toFixed(1)},${yOf(b.avg_price).toFixed(1)}`);
    lastSlot = s;
  });

  const last = lastSlot >= 0 ? model.slots[lastSlot]! : null;
  const up = last ? last.price >= model.prev : true;
  const lineColor = up ? "#eb5454" : "#26a69a"; // A股: 红涨绿跌
  const fillColor = up ? "rgba(235,84,84,0.13)" : "rgba(38,166,154,0.13)";

  const areaPath =
    pricePts.length > 0
      ? `${pricePts.join(" ")} L${xOf(lastSlot).toFixed(1)},${(PAD.top + priceH).toFixed(1)} L${xOf(0).toFixed(1)},${(PAD.top + priceH).toFixed(1)} Z`
      : "";

  const pct = (p: number) => ((p - model.prev) / model.prev) * 100;
  const gridLevels = [model.hi, model.prev + model.maxDev / 2, model.prev,
                      model.prev - model.maxDev / 2, model.lo];

  const hoverBar = hover !== null ? model.slots[hover] : null;
  const volTop = PAD.top + priceH + 6;

  // 做 T 配对 → 画布坐标。一次 T = 入场点 + 接回点, 中间连虚线。
  const marks = signals
    .map((sg) => {
      const s1 = slotOf(sg.entry_time);
      const s2 = slotOf(sg.exit_time);
      if (s1 < 0 || s2 < 0) return null;
      return {
        sg,
        x1: xOf(s1), y1: yOf(sg.entry_price),
        x2: xOf(s2), y2: yOf(sg.exit_price),
      };
    })
    .filter((m): m is NonNullable<typeof m> => m !== null);

  return (
    <div className="minute-chart" ref={setRef}>
      <svg
        width={w}
        height={h}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {/* 横向网格 + 左价格轴 + 右百分比轴 */}
        {gridLevels.map((lv, i) => {
          const y = yOf(lv);
          const isMid = i === 2;
          const c = isMid ? "#4a4a68" : "#232338";
          const txt = isMid ? "#c8c8d8" : pct(lv) >= 0 ? "#eb5454" : "#26a69a";
          return (
            <g key={i}>
              <line x1={PAD.left} y1={y} x2={PAD.left + plotW} y2={y}
                    stroke={c} strokeWidth={1}
                    strokeDasharray={isMid ? "4 3" : undefined} />
              <text x={PAD.left - 5} y={y + 3} textAnchor="end"
                    fontSize={10} fill={txt}>{lv.toFixed(2)}</text>
              <text x={PAD.left + plotW + 5} y={y + 3} fontSize={10} fill={txt}>
                {pct(lv) >= 0 ? "+" : ""}{pct(lv).toFixed(2)}%
              </text>
            </g>
          );
        })}

        {/* 午休分隔 + 时间刻度 */}
        <line x1={xOf(MORNING - 1)} y1={PAD.top} x2={xOf(MORNING - 1)}
              y2={volTop + volH} stroke="#232338" strokeWidth={1} />
        {[[0, "09:30"], [MORNING - 1, "11:30/13:00"], [TOTAL - 1, "15:00"]].map(
          ([s, label]) => (
            <text key={label as string} x={xOf(s as number)} y={h - 5}
                  textAnchor={s === 0 ? "start" : s === TOTAL - 1 ? "end" : "middle"}
                  fontSize={10} fill="#7a7a92">{label}</text>
          )
        )}

        {/* 均价线 (黄) */}
        {avgPts.length > 0 && (
          <path d={avgPts.join(" ")} fill="none" stroke="#f5c542" strokeWidth={1} />
        )}
        {/* 价格线 + 填充 */}
        {areaPath && <path d={areaPath} fill={fillColor} stroke="none" />}
        {pricePts.length > 0 && (
          <path d={pricePts.join(" ")} fill="none" stroke={lineColor} strokeWidth={1.4} />
        )}

        {/* 量柱 */}
        {model.slots.map((b, s) =>
          b && model.maxVol > 0 ? (
            <rect
              key={s}
              x={xOf(s) - Math.max(plotW / TOTAL / 2, 0.5)}
              y={volTop + volH - (b.vol / model.maxVol) * volH}
              width={Math.max(plotW / TOTAL, 1)}
              height={Math.max((b.vol / model.maxVol) * volH, 0.5)}
              fill={b.price >= model.prev ? "#eb545488" : "#26a69a88"}
            />
          ) : null
        )}

        {/* 十字光标 */}
        {hover !== null && hoverBar && (
          <g>
            <line x1={xOf(hover)} y1={PAD.top} x2={xOf(hover)} y2={volTop + volH}
                  stroke="#8888aa" strokeWidth={1} strokeDasharray="3 3" />
            <line x1={PAD.left} y1={yOf(hoverBar.price)} x2={PAD.left + plotW}
                  y2={yOf(hoverBar.price)} stroke="#8888aa" strokeWidth={1}
                  strokeDasharray="3 3" />
            <circle cx={xOf(hover)} cy={yOf(hoverBar.price)} r={3} fill={lineColor} />
          </g>
        )}

        {/* 做 T —— 入场(实心)→ 接回(空心), 虚线相连, 中点标收益 */}
        {marks.map((m, i) => {
          const sell = m.sg.side === "sell";
          const color = sell ? "#eb5454" : "#26a69a";
          const win = m.sg.ret > 0;
          const mx = (m.x1 + m.x2) / 2;
          const my = Math.min(m.y1, m.y2) - 10;
          const tri = (x: number, y: number, down: boolean) =>
            down
              ? `${x},${y - 3} ${x - 5},${y - 11} ${x + 5},${y - 11}`
              : `${x},${y + 3} ${x - 5},${y + 11} ${x + 5},${y + 11}`;
          return (
            <g key={`t0-${i}`} className="t0-mark">
              <line
                x1={m.x1} y1={m.y1} x2={m.x2} y2={m.y2}
                stroke={color} strokeWidth={1} strokeDasharray="3 3" opacity={0.6}
              />
              {/* 入场 */}
              <polygon points={tri(m.x1, m.y1, sell)} fill={color} />
              <circle cx={m.x1} cy={m.y1} r={2.8} fill={color} />
              <text x={m.x1} y={sell ? m.y1 - 14 : m.y1 + 20} textAnchor="middle"
                    fontSize={9} fill={color}>
                {sell ? "抛" : "吸"}{(m.sg.prob * 100).toFixed(0)}
              </text>
              {/* 接回 */}
              <polygon points={tri(m.x2, m.y2, !sell)} fill="none"
                       stroke={color} strokeWidth={1.2} />
              <circle cx={m.x2} cy={m.y2} r={2.8} fill="none"
                      stroke={color} strokeWidth={1.2} />
              <text x={m.x2} y={sell ? m.y2 + 20 : m.y2 - 14} textAnchor="middle"
                    fontSize={9} fill={color}>
                {sell ? "接" : "平"}
              </text>
              {/* 收益 */}
              <text x={mx} y={my} textAnchor="middle" fontSize={10}
                    fill={win ? "#eb5454" : "#8b909a"} fontWeight={600}>
                {(m.sg.ret * 100).toFixed(2)}%
              </text>
              {m.sg.mae < -0.005 && (
                <text x={mx} y={my + 11} textAnchor="middle" fontSize={9}
                      fill="#8b909a">
                  最大浮亏{(m.sg.mae * 100).toFixed(1)}%
                </text>
              )}
              <title>
                {`${m.sg.entry_time} ${sell ? "高抛" : "低吸"} ${m.sg.entry_price} (概率${(m.sg.prob * 100).toFixed(1)}%)\n`}
                {`目标 ${m.sg.target}\n`}
                {`${m.sg.exit_time} ${sell ? "接回" : "卖出"} ${m.sg.exit_price} — ${m.sg.exit_reason}\n`}
                {`本次 ${(m.sg.ret * 100).toFixed(2)}%  中途最大浮亏 ${(m.sg.mae * 100).toFixed(2)}%${m.sg.worst_price ? ` (最不利 ${m.sg.worst_price})` : ""}`}
              </title>
            </g>
          );
        })}
      </svg>

      {/* 图例 / 悬停读数 */}
      <div className="minute-legend">
        {hoverBar ? (
          <>
            <span className="ml-time">
              {hoverBar.time.slice(0, 2)}:{hoverBar.time.slice(2)}
            </span>
            <span style={{ color: hoverBar.price >= model.prev ? "#eb5454" : "#26a69a" }}>
              价 {hoverBar.price.toFixed(2)}
            </span>
            <span style={{ color: "#f5c542" }}>均 {hoverBar.avg_price.toFixed(2)}</span>
            <span className="ml-dim">量 {Math.round(hoverBar.vol)} 手</span>
            <span style={{ color: pct(hoverBar.price) >= 0 ? "#eb5454" : "#26a69a" }}>
              {pct(hoverBar.price) >= 0 ? "+" : ""}{pct(hoverBar.price).toFixed(2)}%
            </span>
          </>
        ) : (
          <>
            <span className="ml-dim">昨收 {model.prev.toFixed(2)}</span>
            {last && (
              <span style={{ color: lineColor }}>
                现价 {last.price.toFixed(2)} ({pct(last.price) >= 0 ? "+" : ""}
                {pct(last.price).toFixed(2)}%)
              </span>
            )}
            <span style={{ color: "#f5c542" }}>— 均价</span>
          </>
        )}
      </div>
    </div>
  );
}
