import type { Chart, KLineData } from "klinecharts";
import { registerIndicator } from "klinecharts";

/**
 * 训练指标的副图 —— 把模型评分画成独立面板, 而不是往主图上堆标记。
 *
 * 主图标记在信号密集时会糊成一片; 副图能看到评分【随时间的变化】,
 * 什么时候开始升、升到多高、持续多久, 这些在标记上完全看不出来。
 *
 * 与 TdxIndicatorManager 同样的套路: 数据来自后端, 注册名带周期
 * (多周期联动时同一指标会挂在日/周/月三张图上, 名字不带周期会互相覆盖)。
 */
const PREFIX = "TRAINED_";
type Tf = "1d" | "1w" | "1m";

export interface TrainedPoint {
  date: string;      // YYYY-MM-DD
  value: number;     // 概率或评分
  grade: string;     // 强/中/弱
}

interface Spec {
  key: string;
  label: string;
  color: string;
  /** 高于此值算"强", 画满色; 低于画淡色 */
  strongAt: number;
  maxValue?: number;
}

export const TRAINED_SPECS: Record<string, Spec> = {
  pump: {
    key: "pump",
    label: "主力吸筹",
    color: "#f0a020",
    strongAt: 0.2,
  },
  didian: {
    key: "didian",
    label: "低点组合 v2",
    color: "#26a69a",
    strongAt: 0.9,      // 用 rank_pct, ≥0.9 为强档
    maxValue: 1,
  },
  mmweek: {
    key: "mmweek",
    label: "买卖很准 周线",
    color: "#38bdf8",
    // 买线 0~100, 这里 >0 即处于超卖状态; 80 以上算深度
    strongAt: 80,
    maxValue: 100,
  },
  maimai_v3: {
    key: "maimai_v3",
    label: "买卖很准 v3",
    color: "#e94560",
    strongAt: 0.8,     // 这里用 rank_pct
  },
};

const cache = new Map<string, TrainedPoint[]>();
const registered = new Set<string>();

export function paneName(key: string, tf: Tf): string {
  return `${PREFIX}${key}_${tf}`;
}

export function setTrainedData(key: string, tf: Tf, points: TrainedPoint[]): string {
  const name = paneName(key, tf);
  cache.set(name, points);
  if (registered.has(name)) return name;
  registered.add(name);

  const spec = TRAINED_SPECS[key];
  registerIndicator<{ v?: number }>({
    name,
    shortName: spec.label,
    calcParams: [],
    minValue: 0,
    figures: [
      {
        key: "v",
        title: `${spec.label}: `,
        type: "bar",
        baseValue: 0,
        styles: ({ data }) => {
          const v = data?.current?.v ?? 0;
          // 强弱用不透明度区分 —— 一眼看出哪几根是真正值得注意的
          return {
            color: v >= spec.strongAt ? spec.color : `${spec.color}55`,
            style: "fill",
          };
        },
      },
    ],
    calc: (dataList: KLineData[]) => {
      const pts = cache.get(name);
      if (!pts) return dataList.map(() => ({}));
      // 后端 timestamp 是 UTC 基准(calendar.timegm), 用本地时区归一化会整体
      // 偏移一天, 所以统一按 UTC 日期字符串匹配。
      const byDay = new Map<string, number>();
      for (const p of pts) byDay.set(p.date, p.value);
      return dataList.map((bar) => {
        const v = byDay.get(new Date(bar.timestamp).toISOString().slice(0, 10));
        return v === undefined ? {} : { v };
      });
    },
  });
  return name;
}

export function createTrainedPane(chart: Chart, key: string, tf: Tf): string | null {
  const name = paneName(key, tf);
  const id = `trained_${key}_pane`;
  // isStack=true —— 与 TdxIndicatorManager 一致。传 false 时 klinecharts
  // 不会新建副图面板, 表现就是"勾了指标但什么都没出现"。
  return chart.createIndicator(name, true, { id });
}

export function removeTrainedPane(chart: Chart, key: string, tf: Tf): void {
  // 按 name 移除(TDX 也是这么做的); 按 paneId 移除在 v10 上不生效。
  chart.removeIndicator({ name: paneName(key, tf) });
}
