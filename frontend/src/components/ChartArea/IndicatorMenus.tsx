// 「指标」+「训练」两个并排下拉 —— 实时页 / 板块页共用。
// 行情页用的是 Toolbar 里的同名结构(样式不同, 逻辑一致)。
//
// 分开而不是合并成一个菜单: 训练指标是本项目训出来的, 与 TDX 移植指标
// 在来源和可信度上完全不是一回事, 混在一起容易让人以为都是现成公式。
import type { IndicatorMeta } from "../../types/indicator";

// ⚠️ 清单不在这里维护 —— 见 hooks/useTrainedSignals.ts。
// 以前这里和 Toolbar.tsx 各存一份, 结果实时页的菜单停在四个指标,
// 而行情页已经有九个。同一份数据抄两遍, 迟早对不上。
import { TRAINED_INDICATORS } from "../../hooks/useTrainedSignals";
export const TRAINED = TRAINED_INDICATORS;

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
