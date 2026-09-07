import { useEffect, useState } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import type { Timeframe } from "../../types/quote";
import type { IndicatorMeta } from "../../types/indicator";
import { listIndicators } from "../../api/indicators";
import { MAIN_PANE_INDICATORS } from "./indicatorPanes";
import { FavButton } from "./FavButton";

const TIMEFRAMES: { label: string; value: Timeframe }[] = [
  { label: "日", value: "1d" },
  { label: "周", value: "1w" },
  { label: "月", value: "1m" },
];

// 标准指标 —— 全部走 klinecharts 内置实现。
// 主图/副图的归属见 indicatorPanes.ts 的 MAIN_PANE_INDICATORS;
// 归错了会画到错误的面板里(BOLL/SAR 必须叠在 K 线上才有意义)。
// MA 周期见 MainChart 的 MA_PERIODS。
const INDICATORS = [
  { group: "主图", items: ["MA", "BOLL", "SAR"] },
  { group: "副图", items: ["MACD"] },
];

/** 自训练指标 —— 我们自己训出来的, 与 TDX 移植指标区分开。
 *  买卖很准v3: 原指标买点与随机无异(胜率48.7%), 模型在同日×同波动层中性化后
 *  把 Top20% 胜率提到 51.3%、中位翻正到 +0.264%, 三个波动档超出全为正。 */
export const TRAINED_INDICATORS = [
  { name: "sar", label: "★★★ SAR预警",
    desc: "SAR翻多×吸筹强 · 命中29.3% · 胜率52.5%(比突破版高17点) · 必须配择时" },
  { name: "breakout", label: "★★★ 突破预警",
    desc: "创60日新高×吸筹强 · 命中32.0% · 必须配大盘择时(裸跑回撤59.7%)" },
  { name: "liftalert", label: "★★ 拉升预警",
    desc: "动力线×吸筹强 · 10日内触及+10% 命中32.6%(基准17.45%) · 各关全过" },
  { name: "mmweek", label: "买卖很准 周线版",
    desc: "周线买线>0 的状态(非买点) · 持有8周 +3.39pp · 按周t=5.56 · 九格全正" },
  { name: "maimai_v3", label: "买卖很准 v3", desc: "动能参考 · 超卖反转买点, 胜率 48.9%→50.9%" },
  { name: "combo", label: "买卖很准 v4", desc: "v3+吸筹共振 · 胜率 58.3%, 超同日全市场 +1.93pp" },
  { name: "didian", label: "低点组合 v2", desc: "动能参考 · 阶段底部+模型过滤, 持有20日" },
  { name: "pump", label: "主力吸筹", desc: "动能参考 · 强档拉升率 19.6%(基础 8.6%), 八成不发生" },
  // 参考项 —— 不是训练指标, 但同样是"能叠在图上的东西", 放一起方便开关。
  { name: "toplist", label: "龙虎榜(参考)", ref: true,
    desc: "蓝圈标出上榜日 · 盘后公布, 次日平均高开1.31%, 只作参考不作信号" },
];

// 画线工具。2026-09-06 打开(此前默认隐藏)。
const SHOW_DRAWING = true;

const OVERLAYS = [
  { label: "趋势线", type: "segment" },
  { label: "水平线", type: "horizontalStraightLine" },
  { label: "垂直线", type: "verticalStraightLine" },
  { label: "平行通道", type: "parallelStraightLine" },
  { label: "斐波那契", type: "fibonacciLine" },
  { label: "矩形", type: "rect" },
  { label: "文字", type: "simpleAnnotation" },
  { label: "箭头", type: "arrow" },
];



interface Props {
  activeIndicators: string[];
  activeTdxIndicators: string[];
  onToggleIndicator: (name: string, isMainPane: boolean) => void;
  onToggleTdxIndicator: (name: string) => void;
  onSelectOverlay: (type: string) => void;
  measuring: boolean;
  onToggleMeasure: () => void;
  /** 自训练指标 —— 与 TDX 移植指标分开管理 */
  activeTrained?: string[];
  onToggleTrained?: (name: string) => void;
}

export function Toolbar({
  activeIndicators,
  activeTdxIndicators,
  onToggleIndicator,
  onToggleTdxIndicator,
  onSelectOverlay,
  measuring,
  onToggleMeasure,
  activeTrained,
  onToggleTrained,
}: Props) {
  const timeframe = useQuoteStore((s) => s.timeframe);
  const setTimeframe = useQuoteStore((s) => s.setTimeframe);
  const linkedMode = useQuoteStore((s) => s.linkedMode);
  const toggleLinkedMode = useQuoteStore((s) => s.toggleLinkedMode);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const currentName = useQuoteStore((s) => s.currentName);

  const [tdxIndicators, setTdxIndicators] = useState<IndicatorMeta[]>([]);

  useEffect(() => {
    listIndicators()
      .then(setTdxIndicators)
      .catch(() => setTdxIndicators([]));
  }, []);

  return (
    <div className="chart-toolbar">
      {currentSymbol && (
        <div className="toolbar-stock-label">
          <span className="toolbar-stock-name">{currentName || "—"}</span>
          <span className="toolbar-stock-code">{currentSymbol}</span>
        </div>
      )}
      <div className="toolbar-divider" />

      <div className="toolbar-group">
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf.value}
            className={timeframe === tf.value ? "active" : ""}
            onClick={() => setTimeframe(tf.value)}
          >
            {tf.label}
          </button>
        ))}
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-group dropdown-container">
        <button className="dropdown-trigger">📊 指标 ▾</button>
        <div className="dropdown-menu indicator-menu">
          {INDICATORS.map((group) => (
            <div key={group.group} className="indicator-group">
              <div className="group-label">{group.group}</div>
              {group.items.map((name) => (
                <label key={name} className="indicator-item">
                  <input
                    type="checkbox"
                    checked={activeIndicators.includes(name)}
                    onChange={() =>
                      onToggleIndicator(name, MAIN_PANE_INDICATORS.has(name))
                    }
                  />
                  {name}
                </label>
              ))}
            </div>
          ))}
          {tdxIndicators.length > 0 && (
            <div className="indicator-group">
              <div className="group-label">TDX 指标</div>
              {tdxIndicators.map((ind) => (
                <label key={ind.name} className="indicator-item">
                  <input
                    type="checkbox"
                    checked={activeTdxIndicators.includes(ind.name)}
                    onChange={() => onToggleTdxIndicator(ind.name)}
                  />
                  {ind.label}
                </label>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 训练指标 —— 本项目自己训出来的, 与标准/TDX 移植指标并列而非混在一起 */}
      {onToggleTrained && (
        <div className="toolbar-group dropdown-container">
          <button className="dropdown-trigger trained-trigger">
            🎯 训练{(activeTrained ?? []).length > 0 && ` (${(activeTrained ?? []).length})`} ▾
          </button>
          <div className="dropdown-menu indicator-menu">
            {TRAINED_INDICATORS.map((ind) => (
              // ref 项(龙虎榜)加分隔线 —— 它不是训练指标, 只是能叠在图上的参考
              <label key={ind.name} title={ind.desc}
                className={`indicator-item${(ind as { ref?: boolean }).ref ? " ref-item" : ""}`}>
                <input
                  type="checkbox"
                  checked={(activeTrained ?? []).includes(ind.name)}
                  onChange={() => onToggleTrained(ind.name)}
                />
                <span className="ti-label">
                  {ind.label}
                  <span className="ti-desc">{ind.desc}</span>
                </span>
              </label>
            ))}
          </div>
        </div>
      )}


      {SHOW_DRAWING && <>
      <div className="toolbar-divider" />

      <div className="toolbar-group dropdown-container">
        <button className="dropdown-trigger">📐 画线 ▾</button>
        <div className="dropdown-menu">
          {OVERLAYS.map((o) => (
            <button
              key={o.type}
              className="overlay-item"
              onClick={() => onSelectOverlay(o.type)}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>
      </>}

      <div className="toolbar-group">
        <button
          className={measuring ? "active" : ""}
          onClick={onToggleMeasure}
          title="测量: 依次点两根K线, 算出跨度 / 涨跌幅 / 区间极值"
        >
          📏 测量
        </button>
      </div>

      <div className="toolbar-group" style={{ marginLeft: "auto" }}>
        <FavButton symbol={currentSymbol} />
        {/* LSTM 5-day forecast removed: model exhibited mode collapse
            (returns near-identical path for all stocks). Backend endpoint
            and trained model files retained for future experiments. */}
        <button
          className={linkedMode ? "active" : ""}
          onClick={toggleLinkedMode}
        >
          ⊞ 多周期联动
        </button>
      </div>
    </div>
  );
}
