/**
 * 把 Kronos 的预测画到 K 线图上。
 *
 * ⚠️ 用 klinecharts【内置】的 segment 逐段连接, 不注册新的 overlay 模板 ——
 *    实测再注册一个新模板会让 trainedMarker 整体失效(v3 的 22 个 overlay
 *    建出来了但一个都画不出来)。见 trainedOverlays.ts 的同款警告。
 *
 * ⚠️ 画折线而不是蜡烛: 模型是自回归采样的, OHLC 有随机性, 画成蜡烛会让人
 *    误以为形态是确定的。折线只表达"大致往哪走", 与它的实际精度相称。
 */
import type { Chart } from "klinecharts";
import type { KronosPrediction } from "../../api/kronos";

const GROUP = "kronos-pred";

export function clearKronos(chart: Chart | null) {
  chart?.removeOverlay({ groupId: GROUP });
}

export function paintKronos(
  chart: Chart | null,
  pred: KronosPrediction,
  anchorTs: number,
) {
  if (!chart) return;
  chart.removeOverlay({ groupId: GROUP });

  // 预测落在图右侧还没有K线的位置, 按交易日间隔外推坐标;
  // 有真实对照时直接用真实K线的日期, 位置才准。
  const dayMs = 86400000;
  const tsOf = (i: number, b: { date: string }) =>
    pred.is_historical ? Date.parse(`${b.date}T00:00:00Z`) : anchorTs + (i + 1) * dayMs;

  const chain = (
    bars: { date: string; close: number }[],
    color: string,
    dashed: boolean,
  ) => {
    const pts = [
      { timestamp: anchorTs, value: pred.anchor_close },
      ...bars.map((b, i) => ({ timestamp: tsOf(i, b), value: b.close })),
    ];
    const out = [];
    for (let i = 0; i + 1 < pts.length; i++) {
      out.push({
        name: "segment",
        groupId: GROUP,
        lock: true,
        points: [pts[i], pts[i + 1]],
        styles: {
          line: {
            color,
            size: 2,
            style: dashed ? "dashed" : "solid",
            dashedValue: [4, 3],
          },
        },
      });
    }
    return out;
  };

  const ovs = [
    ...chain(pred.predicted, "#ffb020", true),
    ...(pred.is_historical ? chain(pred.actual, "#4aa3ff", false) : []),
  ];
  if (ovs.length) chart.createOverlay(ovs as never);
}
