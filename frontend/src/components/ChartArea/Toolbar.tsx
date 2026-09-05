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

// 只保留 MA —— 其余标准指标平时不看, 留着反而挤占菜单。
// MA 周期见 MainChart 的 MA_PERIODS。
const INDICATORS = [
  { group: "均线", items: ["MA"] },
];

/** 自训练指标 —— 我们自己训出来的, 与 TDX 移植指标区分开。
 *  买卖很准v3: 原指标买点与随机无异(胜率48.7%), 模型在同日×同波动层中性化后
 *  把 Top20% 胜率提到 51.3%、中位翻正到 +0.264%, 三个波动档超出全为正。 */
export const TRAINED_INDICATORS = [
  { name: "maimai_v3", label: "买卖很准 v3", desc: "动能参考 · 超卖反转买点, 胜率 48.9%→50.9%" },
  { name: "combo", label: "买卖很准 v4", desc: "v3+吸筹共振 · 胜率 62.5%(随机 53.2%)" },
  { name: "didian", label: "低点组合 v2", desc: "动能参考 · 胜率 57.7%(原 53.8%), 持有20日" },
  { name: "pump", label: "主力吸筹", desc: "动能参考 · 强档拉升率 19.6%(基础 8.6%), 八成不发生" },
];

// 画线工具暂时隐藏 —— 日常用不到。改成 true 即可恢复, 代码原样保留。
const SHOW_DRAWING = false;

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
              <label key={ind.name} className="indicator-item" title={ind.desc}>
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
