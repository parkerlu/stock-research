import { useEffect, useState } from "react";
import type { Chart } from "klinecharts";

/** 副图右上角的关闭按钮。
 *
 * klinecharts 把所有面板画在 canvas 里, 没有独立 DOM, 所以按钮只能用
 * chart.getSize(paneId) 拿到面板矩形后绝对定位叠上去。
 * 主图(candle_pane)不给按钮 —— 关掉它整个图就没了。 */
export interface PaneBtn {
  paneId: string;
  label: string;
  onClose: () => void;
}

interface Props {
  getChart: () => Chart | null;
  panes: PaneBtn[];
}

export function PaneCloseButtons({ getChart, panes }: Props) {
  const [rects, setRects] = useState<Record<string, { top: number; right: number }>>({});

  useEffect(() => {
    if (panes.length === 0) {
      setRects({});
      return;
    }
    // 面板几何随指标增删/窗口缩放变化, 而 klinecharts 不给变更回调,
    // 只能轮询。300ms 足够跟手, 又不至于每帧算。
    const tick = () => {
      const c = getChart();
      if (!c) return;
      const next: Record<string, { top: number; right: number }> = {};
      for (const p of panes) {
        const b = c.getSize?.(p.paneId);
        if (b && b.height > 0) next[p.paneId] = { top: b.top, right: 0 };
      }
      setRects((prev) =>
        JSON.stringify(prev) === JSON.stringify(next) ? prev : next);
    };
    tick();
    const id = setInterval(tick, 300);
    return () => clearInterval(id);
  }, [getChart, panes]);

  return (
    <>
      {panes.map((p) => {
        const r = rects[p.paneId];
        if (!r) return null;
        return (
          <button
            key={p.paneId}
            className="pane-close"
            style={{ top: r.top + 3, right: 62 }}
            title={`关闭 ${p.label}`}
            onClick={(e) => { e.stopPropagation(); p.onClose(); }}
          >
            ×
          </button>
        );
      })}
    </>
  );
}
