/** 训练指标的数据获取 —— 三个页面(行情/实时/板块)共用一份。
 *
 * ⚠️ 存在的理由: ChartArea / LiveView / SectorView 是三个独立组件, 以前每加一个
 * 训练指标都要手动同步三处。实测漏过五次(板块条漏两页、v4 切换漏两页、v3.5
 * 完全没接、SAR/突破/拉升/周线版只接了行情页)。菜单清单同理 —— Toolbar 与
 * IndicatorMenus 各有一份, 结果实时页的菜单停在四个指标。
 *
 * 现在清单和取数都只有这一份, 新增指标改这里一处即可。
 */
import { useEffect, useState } from "react";
import {
  getBreakoutSignals, getChipsSignals, getComboSignals, getDidianSignals,
  getLiftAlertSignals,
  getMaimaiSignals, getMmWeekSignals, getPumpSignals, getSarSignals,
} from "../api/quotes";
import { getStockTopList } from "../api/toplist";

export interface TrainedMeta {
  name: string;
  label: string;
  desc: string;
  /** 参考项(龙虎榜) —— 不是训练指标, 菜单里用分隔线隔开 */
  ref?: boolean;
}

/** 唯一的训练指标清单。⚠️ 新增指标只改这里, 三个页面自动跟上。 */
export const TRAINED_INDICATORS: TrainedMeta[] = [
  { name: "chips", label: "★★★ 筹码模型",
    desc: "获利盘族×XGBoost · 10根K线内触及+10% · 组合比值0.89(年化+15.9%/回撤-17.9%) · 纯选股, 择时贡献为0" },
  { name: "sar", label: "★★★ SAR预警",
    desc: "SAR翻多×吸筹强 · 命中29.3% · 胜率52.5%(比突破版高17点) · 必须配择时" },
  { name: "breakout", label: "★★★ 突破预警",
    desc: "创60日新高×吸筹强 · 命中32.0% · 必须配大盘择时(裸跑回撤59.7%)" },
  { name: "liftalert", label: "★★ 拉升预警",
    desc: "动力线×吸筹强 · 10日内触及+10% 命中32.6%(基准17.45%) · 各关全过" },
  { name: "mmweek", label: "买卖很准 周线版",
    desc: "周线买线>0 的状态(非买点) · 持有8周 +3.39pp · 按周t=5.56 · 九格全正" },
  { name: "maimai_v3", label: "买卖很准 v3",
    desc: "动能参考 · 超卖反转买点, 胜率 48.9%→50.9%" },
  { name: "combo", label: "买卖很准 v4",
    desc: "v3+吸筹共振 · 胜率 58.3%, 超同日全市场 +1.93pp" },
  { name: "didian", label: "低点组合 v2",
    desc: "动能参考 · 阶段底部+模型过滤, 持有20日" },
  { name: "pump", label: "主力吸筹",
    desc: "筹码换手痕迹 · 「10日涨10%」命中25.7%(基准17.45%), t=67, 九格全正" },
  { name: "toplist", label: "龙虎榜(参考)", ref: true,
    desc: "蓝圈标出上榜日 · 盘后公布, 次日平均高开1.31%, 只作参考不作信号" },
];

/** 画在副图(色带/柱)而非主图三角的指标 */
export const SUBPANE_TRAINED = new Set(["pump", "didian", "mmweek"]);

/** 各端点返回的形状略有差异(吸筹给 prob, 别的给 score), 统一成一个宽类型,
 *  具体字段由各自的绘制函数取用。 */
type Sig = { date: string; grade: string; score?: number; rank_pct?: number;
             value?: number; prob?: number; side?: "buy" | "sell" };

/** 按当前选中的指标取数。返回值直接摊给 MainChart。 */
export function useTrainedSignals(symbol: string, active: string[]) {
  const [sig, setSig] = useState<Record<string, Sig[]>>({});
  const [lhb, setLhb] = useState<{ date: string; net_wan: number }[]>([]);

  const key = active.slice().sort().join(",");
  useEffect(() => {
    if (!symbol) { setSig({}); setLhb([]); return; }
    let live = true;
    const fetchers: Record<string, (c: string) => Promise<{ signals: Sig[] }>> = {
      chips: getChipsSignals as never,
      sar: getSarSignals, breakout: getBreakoutSignals, liftalert: getLiftAlertSignals,
      mmweek: getMmWeekSignals, combo: getComboSignals, didian: getDidianSignals,
      pump: getPumpSignals as never,
      maimai_v3: ((c: string) => getMaimaiSignals(c, "弱")) as never,
    };
    const wanted = active.filter((n) => n in fetchers);
    Promise.all(wanted.map((n) =>
      fetchers[n](symbol).then((r) => [n, (r.signals ?? []) as Sig[]] as const)
                         .catch(() => [n, [] as Sig[]] as const)
    )).then((pairs) => { if (live) setSig(Object.fromEntries(pairs)); });

    // 龙虎榜始终取(徽章要用), 但只在勾选时才画到图上
    getStockTopList(symbol)
      .then((r) => live && setLhb(r.items.map((x: { date: string; net_wan: number }) =>
        ({ date: x.date, net_wan: x.net_wan }))))
      .catch(() => live && setLhb([]));
    return () => { live = false; };
  }, [symbol, key]);

  const pick = (n: string) => (active.includes(n) ? sig[n] : undefined);
  return {
    lhb,
    props: {
      chipsSignals: pick("chips") as never,
      sarSignals: pick("sar") as never,
      breakoutSignals: pick("breakout") as never,
      liftSignals: pick("liftalert") as never,
      mmweekSignals: pick("mmweek") as never,
      comboSignals: pick("combo") as never,
      didianSignals: pick("didian") as never,
      pumpSignals: pick("pump") as never,
      maimaiSignals: pick("maimai_v3") as never,
      topList: active.includes("toplist") ? lhb : undefined,
    },
  };
}
