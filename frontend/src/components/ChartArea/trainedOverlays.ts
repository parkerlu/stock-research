import { registerOverlay } from "klinecharts";

let registered = false;

/**
 * 训练指标的标记 —— 只画三角 + 一段虚线, 不带气泡框。
 *
 * klinecharts 自带的 simpleAnnotation 会渲染一个蓝色圆角气泡, 在密集的 K 线上
 * 又大又抢眼, 所以这里自绘。
 *
 * 约定(与交易方向一致, 一眼可辨):
 *   买入 → 红色实心三角朝上, 画在 K 线【下方】
 *   卖出 → 绿色实心三角朝下, 画在 K 线【上方】
 * 强度用大小 + 透明度表示, 不换符号 —— 换符号会让人误以为是不同类型的信号。
 */
export function registerTrainedMarker(): void {
  if (registered) return;
  registered = true;

  registerOverlay({
    name: "trainedMarker",
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ overlay, coordinates, bounding }) => {
      const d = overlay.extendData as
        | { side: "buy" | "sell"; grade: "强" | "中" | "弱"; color?: string }
        | undefined;
      if (!d) return [];
      const isBuy = d.side === "buy";
      const strong = d.grade === "强";
      const mid = d.grade === "中";
      const base = d.color ?? (isBuy ? "#e94560" : "#26a69a");
      const color = strong ? base : mid ? `${base}dd` : `${base}99`;
      const tri = strong ? 8 : mid ? 6.5 : 5;
      const cx = coordinates[0].x;
      const barY = coordinates[0].y;              // 买=bar low, 卖=bar high
      const paneH = bounding?.height ?? 400;

      // 标记画在【图表边缘】而不是贴着 K 线 —— 贴着画会糊在蜡烛上看不清,
      // 而且高低不一没法快速扫视。固定在边缘 + 虚线牵引, 一眼就知道是哪根。
      const edge = 14;
      const tipY = isBuy ? paneH - edge - tri * 1.5 : edge + tri * 1.5;
      const baseY = isBuy ? paneH - edge : edge;
      // 虚线: 从 K 线端点连到三角尖, 留一点缝隙不碰到蜡烛
      const lineFrom = isBuy ? barY + 4 : barY - 4;
      const lineTo = isBuy ? tipY - 2 : tipY + 2;

      return [
        {
          type: "line",
          attrs: {
            coordinates: [
              { x: cx, y: lineFrom },
              { x: cx, y: lineTo },
            ],
          },
          styles: {
            style: "dashed",
            color: strong ? color : `${base}55`,
            size: 1,
            dashedValue: [3, 3],
          },
          ignoreEvent: true,
        },
        {
          type: "polygon",
          attrs: {
            coordinates: [
              { x: cx, y: tipY },
              { x: cx - tri, y: baseY },
              { x: cx + tri, y: baseY },
            ],
          },
          styles: strong
            ? { style: "stroke_fill", color, borderColor: "#ffffffcc", borderSize: 1 }
            : { style: "fill", color },
          ignoreEvent: true,
        },
      ];
    },
  });
}

/** 画训练指标标记 —— 等 K 线数据就绪再画。
 *
 * ⚠️ 原来每个 effect 各自 setTimeout(320~400ms) 后读 chart.getDataList():
 * 从选股列表点进去时信号接口比 K 线接口快, 400ms 到点时 K 线还是空的,
 * 于是一个 overlay 都不创建, 而 effect 只依赖 signals 不会因 K 线到位而重跑
 * —— 标记就这么静默丢了(实测 603997.SH 09-03)。
 * 改成轮询: 画出至少一个才算成功, 否则最多重试 12 次(约 3.8s)。
 */
export function paintTrainedMarkers(
  chart: any,
  groupId: string,
  signals: { date: string; grade: string; side?: string }[] | undefined,
  color?: string
): () => void {
  chart?.removeOverlay({ groupId });
  if (!chart || !signals || signals.length === 0) return () => {};
  let cancelled = false;
  let tries = 0;
  let timer: ReturnType<typeof setTimeout>;
  const attempt = () => {
    if (cancelled) return;
    const list = chart.getDataList?.() ?? [];
    if (list.length > 0) {
      const low = new Map<string, number>();
      const high = new Map<string, number>();
      for (const b of list) {
        const k = new Date(b.timestamp).toISOString().slice(0, 10); // UTC, 与后端一致
        low.set(k, b.low as number);
        high.set(k, b.high as number);
      }
      const overlays: any[] = [];
      for (const sg of signals) {
        const isSell = sg.side === "sell";
        const anchor = isSell ? high.get(sg.date) : low.get(sg.date);
        if (anchor === undefined) continue;
        overlays.push({
          name: "trainedMarker",
          groupId,
          lock: true,
          points: [{ timestamp: Date.parse(`${sg.date}T00:00:00Z`), value: anchor }],
          extendData: { side: isSell ? "sell" : "buy", grade: sg.grade, color },
        });
      }
      if (overlays.length) {
        chart.createOverlay(overlays);
        return;
      }
    }
    if (++tries < 12) timer = setTimeout(attempt, 300);
  };
  timer = setTimeout(attempt, 150);
  return () => {
    cancelled = true;
    clearTimeout(timer);
  };
}
