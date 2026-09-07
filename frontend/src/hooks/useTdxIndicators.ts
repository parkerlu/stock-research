// TDX 指标挂载 hook — 把 ChartArea 里那套"取数 → 注册 → 建 pane → 换股刷新"
// 的逻辑抽出来, 供多个图表实例复用 (ChartArea 主图 / 实时页日K)。
//
// 每个使用处传各自的 storageKey, 这样两张图的指标选择互不干扰。
import { useCallback, useEffect, useRef, useState } from "react";
import type { Chart } from "klinecharts";
import { getIndicator, listIndicators } from "../api/indicators";
import { useQuoteStore } from "../stores/quoteStore";
import type { IndicatorMeta } from "../types/indicator";
import {
  getKlineIndicatorName,
  setIndicatorData,
} from "../components/ChartArea/TdxIndicatorManager";

/** 图上可能出现过的全部周期 —— purge 要按这个逐个清。 */
const ALL_TFS = ["1d", "1w", "1m"] as const;

interface Options {
  /** 惰性取图表实例 —— 图表可能晚于 hook 初始化 */
  getChart: () => Chart | null;
  symbol: string;
  timeframe?: "1d" | "1w" | "1m";
  storageKey: string;
}

function readLS(key: string): string[] {
  try {
    const raw = localStorage.getItem(key);
    const parsed = raw ? JSON.parse(raw) : null;
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

export function useTdxIndicators({
  getChart,
  symbol,
  timeframe = "1d",
  storageKey,
}: Options) {
  const [available, setAvailable] = useState<IndicatorMeta[]>([]);
  const [active, setActive] = useState<string[]>(() => readLS(storageKey));
  const [busy, setBusy] = useState<string | null>(null);
  // 多周期联动开关 —— 切换时主图会卸载/重建, 指标要跟着重挂
  const linkedMode = useQuoteStore((st) => st.linkedMode);
  const paneIds = useRef<Record<string, string>>({});
  // 已挂在图上的那批指标是按哪个周期注册的 —— 注册名带周期, 换周期时得先按
  // 旧名字摘掉, 否则 overrideIndicator 找不到目标, 图上会留着旧周期的数据。
  const appliedTf = useRef<string | null>(null);

  useEffect(() => {
    listIndicators().then(setAvailable).catch(() => setAvailable([]));
  }, []);

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(active));
  }, [active, storageKey]);

  /** 取数并在图上建/更新一个指标。create=false 时只刷新数据。 */
  /** 把某个 TDX 指标【所有周期】的变体从图上摘干净。
   *
   * ⚠️ 必须按全周期清, 不能只清当前周期: klinecharts 的指标名带周期
   * (getKlineIndicatorName), 而面板 id 不带, 且 createIndicator 用的是
   * isStack=true —— 切一次周期就往同一个面板里多叠一层, 而移除只按当前周期
   * 的名字删, 别的周期永远留着。实测叠了四层"买卖很准"。
   * 原来靠 appliedTf 记录上次周期来清理, 但 active 为空时那段 effect 直接
   * return, appliedTf 就停在旧值, 状态一漂就漏。清全部才是幂等的。 */
  const purge = useCallback((chart: Chart, name: string) => {
    for (const tf of ALL_TFS) {
      chart.removeIndicator({ name: getKlineIndicatorName(name, tf) });
    }
  }, []);

  const apply = useCallback(
    async (name: string, create: boolean) => {
      const chart = getChart();
      if (!chart || !symbol) return false;
      const today = new Date().toISOString().slice(0, 10);
      const result = await getIndicator(name, symbol, timeframe, "1990-01-01", today);
      if (result.warnings.length > 0) {
        alert(result.warnings.join("\n"));
        return false;
      }
      setIndicatorData(result, timeframe);
      const kcName = getKlineIndicatorName(name, timeframe);
      if (create) {
        const paneId = result.pane === "main" ? "candle_pane" : `tdx_${name}_pane`;
        purge(chart, name);              // 建之前先清掉所有周期的旧变体
        chart.createIndicator(kcName, true, { id: paneId });
        paneIds.current[name] = paneId;
      } else {
        chart.overrideIndicator({ name: kcName });
      }
      return true;
    },
    [getChart, symbol, timeframe, purge]
  );

  const toggle = useCallback(
    async (name: string) => {
      const chart = getChart();
      if (!symbol) return;
      if (active.includes(name)) {
        if (chart) purge(chart, name);   // 同样清全部周期
        delete paneIds.current[name];
        setActive((p) => p.filter((n) => n !== name));
        return;
      }
      // 多周期联动模式下主图没挂载, 但选择仍要生效 —— 联动视图的三张图订阅
      // 的就是这份 active, 它们各自按自己的周期取数
      if (!chart) {
        setActive((p) => [...p, name]);
        return;
      }
      setBusy(name);
      try {
        if (await apply(name, true)) setActive((p) => [...p, name]);
      } catch (err) {
        console.error("加载 TDX 指标失败:", err);
        alert("加载 TDX 指标失败");
      } finally {
        setBusy(null);
      }
    },
    [active, apply, getChart, symbol, timeframe]
  );

  // ⚠️ 联动开关一变就清空 paneIds —— 必须独立成一个 effect, 不能塞进下面那个。
  // 下面那个要先轮询图表实例, 而切【到】联动时主图已卸载, 轮询 20 次全空后
  // 直接 return, 清空逻辑根本走不到; 等切【回】来时记录还是旧的,
  // apply(name, !paneIds.current[name]) 就走了 overrideIndicator 分支 ——
  // 新图表上没有这个指标, override 什么也不做, 指标就永久消失了(用户实测)。
  useEffect(() => {
    paneIds.current = {};
  }, [linkedMode]);

  // 换股 / 换周期后重建已选指标。图表实例可能还没就绪, 轮询几次再放弃。
  useEffect(() => {
    if (!symbol || active.length === 0) return;
    let cancelled = false;
    let tries = 0;
    const run = async () => {
      if (cancelled) return;
      if (!getChart()) {
        if (tries++ < 20) setTimeout(run, 200);
        return;
      }
      // 换周期: 旧注册名已失效, 先按旧周期把指标摘干净再重建
      const chart = getChart();
      if (appliedTf.current && appliedTf.current !== timeframe && chart) {
        for (const name of Object.keys(paneIds.current)) {
          chart.removeIndicator({
            name: getKlineIndicatorName(name, appliedTf.current as typeof timeframe),
          });
        }
        paneIds.current = {};
      }
      appliedTf.current = timeframe;

      for (const name of active) {
        if (cancelled) return;
        try {
          // 换股后 pane 已被图表重建流程清掉, 这里统一按"创建"处理
          await apply(name, !paneIds.current[name]);
        } catch (err) {
          console.error(`刷新 TDX 指标 ${name} 失败:`, err);
        }
      }
    };
    run();
    return () => {
      cancelled = true;
    };
    // active 变化由 toggle 自己处理, 这里跟随 symbol/timeframe/联动开关。
    // ⚠️ linkedMode 必须在依赖里: 切到多周期联动时主图被卸载, 切回来是全新的
    // 图表实例, 不重挂的话指标就永久消失(用户实测)。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, timeframe, linkedMode]);

  return { available, active, toggle, busy };
}
