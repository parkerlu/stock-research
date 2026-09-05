// 「指标」+「训练」两个并排下拉 —— 实时页 / 板块页共用。
// 行情页用的是 Toolbar 里的同名结构(样式不同, 逻辑一致)。
//
// 分开而不是合并成一个菜单: 训练指标是本项目训出来的, 与 TDX 移植指标
// 在来源和可信度上完全不是一回事, 混在一起容易让人以为都是现成公式。
import type { IndicatorMeta } from "../../types/indicator";

/** 已训练好、可挂到 K 线上的指标。新增训练指标往这里加。 */
export const TRAINED = [
  {
    name: "maimai_v3",
    label: "买卖很准 v3",
    desc: "动能参考 · 超卖反转买点, 胜率 48.9%→50.9%",
  },
  {
    name: "combo",
    label: "买卖很准 v4",
    desc: "v3+吸筹共振 · 胜率 62.5%(随机 53.2%), 八年全部>54%",
  },
  {
    name: "didian",
    label: "低点组合 v2",
    desc: "动能参考 · 胜率 57.7%(原 53.8%), 持有20日",
  },
  {
    name: "pump",
    label: "主力吸筹",
    desc: "动能参考 · 强档拉升率 19.6%(基础 8.6%), 八成不发生",
  },
];

interface Props {
  tdxAvailable: IndicatorMeta[];
  tdxActive: string[];
  onToggleTdx: (name: string) => void;
  tdxBusy?: string | null;
  trainedActive: string[];
  onToggleTrained: (name: string) => void;
  /** 每个训练指标当前的信号条数, 显示在名称后面 */
  trainedCounts?: Record<string, number>;
}

export function IndicatorMenus({
  tdxAvailable,
  tdxActive,
  onToggleTdx,
  tdxBusy,
  trainedActive,
  onToggleTrained,
  trainedCounts,
}: Props) {
  return (
    <>
      <div className="live-tdx">
        <span className="live-tdx-btn">
          指标 {tdxActive.length > 0 && `(${tdxActive.length})`} ▾
        </span>
        <div className="live-tdx-menu">
          {tdxAvailable.length === 0 && <div className="live-tdx-empty">加载中…</div>}
          {tdxAvailable.map((m) => (
            <label key={m.name} className="live-tdx-item">
              <input
                type="checkbox"
                checked={tdxActive.includes(m.name)}
                onChange={() => onToggleTdx(m.name)}
              />
              <span>{m.label}</span>
              {tdxBusy === m.name && <span className="live-dim">…</span>}
            </label>
          ))}
        </div>
      </div>

      <div className="live-tdx trained-menu">
        <span className="live-tdx-btn trained-btn">
          🎯 训练 {trainedActive.length > 0 && `(${trainedActive.length})`} ▾
        </span>
        <div className="live-tdx-menu">
          {TRAINED.map((t) => {
            const n = trainedCounts?.[t.name];
            return (
              <label key={t.name} className="live-tdx-item" title={t.desc}>
                <input
                  type="checkbox"
                  checked={trainedActive.includes(t.name)}
                  onChange={() => onToggleTrained(t.name)}
                />
                <span className="ti-label">
                  {t.label}
                  {n ? ` (${n})` : ""}
                  <span className="ti-desc">{t.desc}</span>
                </span>
              </label>
            );
          })}
        </div>
      </div>
    </>
  );
}
