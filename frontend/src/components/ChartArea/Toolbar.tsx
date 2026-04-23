import { useEffect, useState } from "react";
import { useQuoteStore } from "../../stores/quoteStore";
import type { Timeframe } from "../../types/quote";
import type { IndicatorMeta } from "../../types/indicator";
import { listIndicators } from "../../api/indicators";

const TIMEFRAMES: { label: string; value: Timeframe }[] = [
  { label: "日", value: "1d" },
  { label: "周", value: "1w" },
  { label: "月", value: "1m" },
];

const INDICATORS = [
  { group: "均线", items: ["MA", "EMA", "BOLL"] },
  { group: "趋势", items: ["MACD", "DMI", "SAR"] },
  { group: "摆动", items: ["KDJ", "RSI", "WR"] },
  { group: "量能", items: ["OBV"] },
];

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

const MAIN_PANE_INDICATORS = new Set(["MA", "EMA", "BOLL", "SAR"]);

interface Props {
  activeIndicators: string[];
  activeTdxIndicators: string[];
  onToggleIndicator: (name: string, isMainPane: boolean) => void;
  onToggleTdxIndicator: (name: string) => void;
  onSelectOverlay: (type: string) => void;
}

export function Toolbar({
  activeIndicators,
  activeTdxIndicators,
  onToggleIndicator,
  onToggleTdxIndicator,
  onSelectOverlay,
}: Props) {
  const timeframe = useQuoteStore((s) => s.timeframe);
  const setTimeframe = useQuoteStore((s) => s.setTimeframe);
  const linkedMode = useQuoteStore((s) => s.linkedMode);
  const toggleLinkedMode = useQuoteStore((s) => s.toggleLinkedMode);

  const [tdxIndicators, setTdxIndicators] = useState<IndicatorMeta[]>([]);

  useEffect(() => {
    listIndicators()
      .then(setTdxIndicators)
      .catch(() => setTdxIndicators([]));
  }, []);

  return (
    <div className="chart-toolbar">
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

      <div className="toolbar-group" style={{ marginLeft: "auto" }}>
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
