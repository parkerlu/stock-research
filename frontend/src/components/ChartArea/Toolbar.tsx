import { useEffect, useState } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import type { Timeframe } from "../../types/quote";
import type { IndicatorMeta } from "../../types/indicator";
import { listIndicators } from "../../api/indicators";
import { MAIN_PANE_INDICATORS } from "./indicatorPanes";

const TIMEFRAMES: { label: string; value: Timeframe }[] = [
  { label: "日", value: "1d" },
  { label: "周", value: "1w" },
  { label: "月", value: "1m" },
];

// 只保留 MA —— 其余指标平时不看, 留着反而挤占工具栏和视线。
// MA 周期见 MainChart 的 MA_PERIODS。
const INDICATORS = [
  { group: "均线", items: ["MA"] },
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
}

export function Toolbar({
  activeIndicators,
  activeTdxIndicators,
  onToggleIndicator,
  onToggleTdxIndicator,
  onSelectOverlay,
  measuring,
  onToggleMeasure,
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
        </div>
      </div>

      <div className="toolbar-group indicator-tags">
        {activeIndicators.map((name) => (
          <span key={name} className="indicator-tag">
            {name}
          </span>
        ))}
      </div>

      {tdxIndicators.length > 0 && (
        <>
          <div className="toolbar-divider" />
          <div className="toolbar-group dropdown-container">
            <button className="dropdown-trigger">🔮 TDX 指标 ▾</button>
            <div className="dropdown-menu">
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
          </div>
        </>
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
