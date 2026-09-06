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
        | { side: "buy" | "sell"; grade: "强" | "中" | "弱"; color?: string;
            shape?: "triangle" | "circle"; noLine?: boolean }
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

      // 圆圈变体(龙虎榜用): 贴着 K 线画一个小圆, 不要牵引线 —— 它是参考信息,
      // 不该像买卖信号那样抢视线。
      // ⚠️ 圆用 polygon 近似, 不用 circle: circle 的 per-figure styles 在
      // klinecharts v10 上不生效, 会退回默认蓝。
      if (d.shape === "circle") {
        const r = 5;
        const cy = isBuy ? barY + r + 6 : barY - r - 6;
        const pts = Array.from({ length: 14 }, (_, i) => {
          const a = (i / 14) * Math.PI * 2;
          return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
        });
        return [{
          type: "polygon",
          attrs: { coordinates: pts },
          styles: { style: "stroke_fill", color: `${base}33`,
                    borderColor: base, borderSize: 1.4 },
          ignoreEvent: true,
        }];
      }

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

let selRegistered = false;

/** 选中信号高亮 —— 选股页点某一行时, 在对应那根 K 线上打一道青色高亮带。
 *
 * 和训练指标的三角标记是两回事: 三角标记回答"这只票哪些天出过信号",
 * 高亮带回答"我刚点的是哪一天"。同一只票可能有十几个同色三角, 不标出来
 * 根本认不出点进来的是哪根。
 * 颜色用青色, 避开已有的红(v3)/金(v4)/灰(回放线)。 */
export function registerSelectedSignal(): void {
  if (selRegistered) return;
  selRegistered = true;

  registerOverlay({
    name: "selectedSignal",
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: (params) => {
      const { overlay, coordinates, bounding } = params;
      const d = overlay.extendData as { text?: string } | undefined;
      const cx = coordinates[0].x;
      const h = bounding?.height ?? 400;
      // barSpace 不在公开类型里, 但运行时有 —— 拿不到就退回固定宽度
      const w = Math.max((params as any).barSpace?.bar ?? 6, 3);
      const color = "#22d3ee";
      return [
        // 高亮带: 半透明底色, 让那根 K 线整体亮起来
        {
          type: "rect",
          attrs: { x: cx - w / 2, y: 0, width: w, height: h },
          styles: { style: "fill", color: "rgba(34,211,238,.16)" },
          ignoreEvent: true,
        },
        {
          type: "line",
          attrs: { coordinates: [{ x: cx, y: 0 }, { x: cx, y: h }] },
          styles: { style: "dashed", color, size: 1, dashedValue: [3, 3] },
          ignoreEvent: true,
        },
        {
          type: "text",
          attrs: { x: cx + 5, y: 4, text: d?.text ?? "", align: "left", baseline: "top" },
          styles: {
            color: "#06202a", backgroundColor: color, size: 10, weight: "bold",
            paddingLeft: 5, paddingRight: 5, paddingTop: 2, paddingBottom: 2,
            borderRadius: 2,
          },
          ignoreEvent: true,
        },
      ];
    },
  });
}

/** 画选中信号高亮。与 paintTrainedMarkers 一样等 K 线就绪再画。 */
export function paintSelectedSignal(chart: any, date: string | null | undefined): () => void {
  chart?.removeOverlay({ groupId: "selected-signal" });
  if (!chart || !date) return () => {};
  let cancelled = false;
  let tries = 0;
  let timer: ReturnType<typeof setTimeout>;
  const attempt = () => {
    if (cancelled) return;
    const list = chart.getDataList?.() ?? [];
    const hit = list.find(
      (b: any) => new Date(b.timestamp).toISOString().slice(0, 10) === date
    );
    if (hit) {
      chart.createOverlay({
        name: "selectedSignal",
        groupId: "selected-signal",
        lock: true,
        points: [{ timestamp: hit.timestamp, value: hit.close }],
        extendData: { text: `信号 ${date.slice(5)}` },
      });
      return;
    }
    if (++tries < 12) timer = setTimeout(attempt, 300);
  };
  timer = setTimeout(attempt, 150);
  return () => {
    cancelled = true;
    clearTimeout(timer);
  };
}

/** 画龙虎榜标记 —— 纯参考, 不是买卖信号。
 *
 * ⚠️ 复用 trainedMarker, 不另注册 overlay 模板。实测再注册第三个模板会让
 * trainedMarker 整体失效(v3 的 22 个 overlay 建出来了但一个都画不出来),
 * 原因没查到, 但复用是稳的 —— 别再试着自建。
 *
 * 画成蓝色小圆圈贴在 K 线上方(不带牵引线) —— 与买卖三角(图上下沿, 红/金)
 * 在颜色和形状上都区分开, 一眼就知道不是买卖信号;
 * 详细内容看工具栏右侧的「龙虎榜 N 次」徽章。
 *
 * 为什么只标不做信号: 龙虎榜盘后公布, 这些票次日平均高开 1.31%,
 * 「10日涨10%」命中率从 41.53%(收盘基准)掉到 37.22%(可交易基准)。 */
export function paintTopList(
  chart: any,
  items: { date: string; net_wan: number }[] | undefined
): () => void {
  chart?.removeOverlay({ groupId: "toplist" });
  if (!chart || !items || items.length === 0) return () => {};
  let cancelled = false, tries = 0;
  let timer: ReturnType<typeof setTimeout>;
  const attempt = () => {
    if (cancelled) return;
    const list = chart.getDataList?.() ?? [];
    if (list.length > 0) {
      const hi = new Map<string, number>();
      for (const b of list) {
        hi.set(new Date(b.timestamp).toISOString().slice(0, 10), b.high as number);
      }
      const ovs: any[] = [];
      for (const it of items) {
        const h = hi.get(it.date);
        if (h === undefined) continue;
        ovs.push({
          name: "trainedMarker", groupId: "toplist", lock: true,
          points: [{ timestamp: Date.parse(`${it.date}T00:00:00Z`), value: h }],
          extendData: { side: "sell", grade: "强", color: "#3b9dff",
                        shape: "circle", noLine: true },
        });
      }
      if (ovs.length) { chart.createOverlay(ovs); return; }
    }
    if (++tries < 12) timer = setTimeout(attempt, 300);
  };
  timer = setTimeout(attempt, 150);
  return () => { cancelled = true; clearTimeout(timer); };
}
