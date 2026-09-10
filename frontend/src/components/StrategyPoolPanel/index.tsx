/** 策略池 —— 策略的唯一定义源, 也是给人看的说明页。
 *
 * ⚠️ 2026-09-10 重写。这个页面以前显示的是【策略工厂】跑出来的 101 条参数组合
 * (同一个模板在单只票上调参), 与现在这套"全市场模型选股"不是一回事,
 * 留着只会让人以为策略池里有一百多个策略。已连数据一起清掉。
 *
 * ⚠️ 现在它读 strategy_pool 表, 与虚拟盘的策略下拉是【同一个数据源】——
 * 以前策略定义散在三处(演示回放的硬编码、每个账户的 config、各命令的 docstring),
 * 改一处要动三处而且没人知道以哪个为准。
 */
import { useEffect, useState } from "react";
import { getDemoStrategies } from "../../api/paper";
import type { DemoStrategy } from "../../api/paper";

export function StrategyPoolPanel() {
  const [items, setItems] = useState<DemoStrategy[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getDemoStrategies()
      .then((r) => {
        setItems(r.items);
        if (r.items.length) setOpen(r.items[0].key);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  return (
    <div className="pool-page">
      <div className="pool-head">
        <h2>策略池</h2>
        <p>
          这里定义的策略同时供【虚拟盘·实操盘】与【演示回放】使用 —— 改这里一处，两边都跟上。
          <br />
          <b>比值 = 年化 ÷ 最大回撤</b>，是这个项目的硬约束（要求 &gt; 1）。
          所有数字都来自 walk-forward + 资金池 + 真实周转的口径，不是简单回测。
        </p>
      </div>
      {err && <div className="pool-err">读取失败：{err}</div>}
      <div className="pool-list">
        {items.map((s) => {
          const on = open === s.key;
          const pass = s.ratio != null && s.ratio >= 1;
          return (
            <div key={s.key} className={`pool-card${on ? " on" : ""}`}>
              <button className="pool-card-head" onClick={() => setOpen(on ? null : s.key)}>
                <span className="pool-tri">{on ? "▾" : "▸"}</span>
                <span className="pool-name">{s.name}</span>
                {s.ratio != null && (
                  <span className={`pool-ratio${pass ? " pass" : ""}`}>
                    比值 {s.ratio.toFixed(2)}
                    {pass && <em>达标</em>}
                  </span>
                )}
                <span className="pool-since">信号自 {s.since}</span>
                {s.is_live && <span className="pool-live">实操盘</span>}
              </button>
              <div className="pool-summary">{s.summary}</div>
              {on && <pre className="pool-detail">{s.detail}</pre>}
            </div>
          );
        })}
        {!items.length && !err && <div className="pool-empty">加载中…</div>}
      </div>
    </div>
  );
}
