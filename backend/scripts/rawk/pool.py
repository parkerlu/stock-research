"""资金池回测 —— 每笔在【它自己的实际出场日】把钱还回池子, 立刻能买下一只。

⚠️ 为什么不用 decide.py 的 equity():
   那个是"每天投 1/hold 的资金, 收益摊到 hold 天"。它假设每笔都占满 10 天,
   而标签本身是"先碰 +10% 或 -8% 就出场" —— 实测平均只占 7.80 天
   (涨 5.21 天 / 跌 5.37 天 / 平 10 天)。提前出场的钱被当成在场外空等,
   年化被系统性低估, 而"年化 > 回撤"正是这个项目的硬约束。
   2026-09-09 用户提出: "买入到10%就抛, 等下次。盘中过10%也可以抛呀"。

⚠️ 盘中触及即出场是【可成交】的, 不需要额外约束:
   触及 +10% 判据用的是当日 high。A 股涨停时买不进但【卖得掉】——
   涨停价上排着大量买单。买入侧的一字涨停约束已经在 label3.py 里剔除过了。

⚠️ 仍然偏乐观的一处: 出场价按标签的 +10% / −8% 计, 没有考虑跳空穿越
   (开盘直接 −12% 时实际成交价低于 −8%)。所以"跌"那部分的收益偏好。
   要修得给 label3 多存一列"实际触及日的开盘价", 目前没有。

⚠️ 与 decide.py 的关系: 那个是【每笔期望】的评估(信号级), 这个是【资金效率】
   的评估(组合级)。两个都要看 —— 项目里"信号级全过、组合级垮掉"的先例太多了。

用法: python pool.py _cls512_nbar400 --slots 20
需要 holddays_dn8.npz(由 holddays.py 生成), 与标签一一对应。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DIR = "/root/data"
COST = 0.003          # 佣金+印花税+滑点, 单边买卖合计的保守估计


def breadth_mask(te: pd.DataFrame, thr: float, win: int = 60) -> dict[int, bool]:
    """用模型【自己的广度】做择时: 当天有多少只票的 EV 过阈值。

    想法来源: 回撤被诊断为系统性的(20/40/60 仓位下都是 −30%), 而外部择时
    (中证1000 > MA) 在 walk-forward 下站不住。那就换个不依赖外部数据、
    也不引入新阈值的开关 —— 模型在全市场都找不到机会的那天, 本身就是信号。

    ⚠️ 无新参数: 判据是"今天的合格只数 > 过去 win 天的中位数", 中位数是从
       【已经过去的】数据滚动算的, 不是全期分位数(那会偷看未来)。
    ⚠️ win=60 是唯一的自由度, 且不参与选参 —— 固定成三个月, 一个常识值。
    """
    cnt = te[te.ev >= thr].groupby("day").size()
    all_days = np.sort(te["day"].unique())
    cnt = cnt.reindex(all_days, fill_value=0).astype(float)
    med = cnt.shift(1).rolling(win, min_periods=20).median()
    on = (cnt > med).fillna(False)
    return {int(d): bool(v) for d, v in on.items()}


def crash_mask(drop: float, win: int = 20, idx: int = 0) -> dict[int, bool]:
    """尾部保护: 指数近 win 日跌幅超过 drop 时停止建仓, 直到跌幅收窄。

    ⚠️ 与 MA 择时的区别, 也是它存在的理由:
       MA 择时是【常规】开关, 在震荡市频繁误触发, walk-forward 下整体拖累
       (固定 MA20 只有 0.28)。而诊断显示 −31% 回撤集中在 2024-01 那一次
       小微盘踩踏, 属于尾部事件。用只在极端时触发的开关去接尾部, 平时不干预,
       比"天天判断牛熊"更贴合问题。

    ⚠️ 阈值取先验值(−15% / 20日), 不进选参网格 —— 刚得出的结论是"参数越多
       walk-forward 越差"(0.78 -> 0.38), 这里就不能再加一个可调项。
       −15%/20日 是常识性的极端跌幅, 不是扫出来的。
    """
    d = np.load(f"{DIR}/panel.npz")
    mkt, day_min = d["mkt"], int(d["day_min"])
    close = pd.Series(mkt[idx][3].astype(np.float64))
    ret = close / close.shift(win) - 1
    on = (ret > drop).fillna(True).to_numpy()
    return {day_min + i: bool(v) for i, v in enumerate(on)}


def timing_mask(ma: int, idx: int = 0) -> dict[int, bool]:
    """大盘开关: 指数收盘 > 自身 MA(ma) 的交易日才允许建仓。

    ⚠️ 因果性: 判据用【信号日 T 的收盘】, 建仓在 T+1 开盘 —— T 日收盘时这个值
       就已经知道了, 不是未来函数。

    ⚠️ 为什么值得试: 本项目"突破预警"上量过同一件事 —— 信号高度同步, 裸跑
       一起崩(回撤 59.7%), 但也正因为同步, 一个大盘开关就整批挡住了,
       回撤降到 21.5% 而年化反而从 10.0% 升到 41.0%。
       资金池回测显示这个模型的回撤在 20/40/60 仓位下都是 −30%,
       分散化无效 == 回撤是系统性的, 正是择时该管的那部分。

    idx=0 中证1000, idx=1 沪深300。选中证1000 是因为选出来的多是小盘股 ——
    拿蓝筹的脸色判断小盘股死活, 项目里已经证明过是错的(换基准 1.91 -> 2.62)。
    """
    d = np.load(f"{DIR}/panel.npz")
    mkt, day_min = d["mkt"], int(d["day_min"])
    close = pd.Series(mkt[idx][3].astype(np.float64))
    on = (close > close.rolling(ma).mean()).to_numpy()
    return {day_min + i: bool(v) for i, v in enumerate(on)}


def run(te: pd.DataFrame, days_all: np.ndarray, slots: int, thr: float,
        allow: dict[int, bool] | None = None, cap: int = 0) -> dict:
    """事件驱动的资金池模拟。

    每天的顺序固定: 先结算到期(资金回池) -> 再用空闲仓位买入。
    顺序不能颠倒 —— 反过来会让当天到期的钱当天就重复使用一次。
    """
    g = te[te.ev >= thr]
    if len(g) == 0:
        return {}
    by_day = {d: v for d, v in g.groupby("day")}
    all_days = np.sort(te["day"].unique())
    pos = {d: i for i, d in enumerate(all_days)}

    cash = 1.0
    free = slots
    due: dict[int, list[tuple[float, float]]] = {}     # 出场日 -> [(投入金额, 收益率)]
    curve: list[float] = []
    used: list[float] = []
    n_trade = 0
    hold_sum = 0

    for d in all_days:
        for amt, r in due.pop(d, []):
            cash += amt * (1 + r - COST)
            free += 1
        held = sum(amt for lst in due.values() for amt, _ in lst)
        equity = cash + held

        if free > 0 and d in by_day and (allow is None or allow.get(int(d), False)):
            per = equity / slots
            # cap: 单日最多建几仓。timing 分析显示 69% 的交易挤在 10% 的交易日 ——
            # 仓位在时间上极度集中, 同买同卖, 回撤自然大。限制单日建仓数是强制
            # 时间分散, 零成本(不用重训)。
            room = min(free, cap) if cap else free
            for row in by_day[d].nlargest(room, "ev").itertuples():
                if free <= 0 or cash < per * 0.5:
                    break
                amt = min(per, cash)
                cash -= amt
                free -= 1
                hd = int(days_all[row.idxpos])
                out_i = min(pos[d] + hd, len(all_days) - 1)
                due.setdefault(all_days[out_i], []).append((amt, float(row.r)))
                n_trade += 1
                hold_sum += hd
        curve.append(equity)
        used.append((slots - free) / slots)          # 当日仓位利用率

    eq = pd.Series(curve, index=all_days)
    yrs = len(all_days) / 243
    ann = (eq.iloc[-1] ** (1 / yrs) - 1) * 100
    mdd = ((eq / eq.cummax()) - 1).min() * 100
    return {"ann": float(ann), "mdd": float(mdd),
            "ratio": float(ann / abs(mdd)) if mdd else 0.0,
            "n": n_trade, "hold": hold_sum / max(n_trade, 1),
            "turn": n_trade / max(yrs, 1e-9) / slots, "eq": eq,
            # ⚠️ 仓位利用率必须报: 靠"少建仓"把回撤压下去也能让 年化/回撤 过线,
            #    但那是降杠杆不是 alpha。极端情况只用 1% 仓位, 比值也能 >1 而毫无意义。
            "use": float(np.mean(used)) if used else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--label", default="label3_dn8.npz")
    ap.add_argument("--days", default="holddays_dn8.npz")
    ap.add_argument("--slots", type=int, default=20)
    ap.add_argument("--timing", type=int, default=0,
                    help="大盘择时: 指数收盘 > MA(N) 才建仓, 0=不择时")
    ap.add_argument("--bench", type=int, default=0, help="0=中证1000, 1=沪深300")
    ap.add_argument("--fixed", action="store_true",
                    help="对照: 强制每笔都持满 hold 天, 用来量化周转带来的差别")
    ap.add_argument("--crash", type=float, default=0.0,
                    help="尾部保护: 指数20日跌幅超过该值(如 -0.15)就停止建仓")
    ap.add_argument("--breadth", action="store_true",
                    help="用模型自己的广度做择时(合格只数 > 过去60日中位数)")
    ap.add_argument("--no-cap", action="store_true", help="cap 不进网格, 固定为不限")
    ap.add_argument("--cap", type=int, default=0,
                    help="单日最多建几仓(强制时间分散), 0=不限")
    ap.add_argument("--fix-ma", type=int, default=-1,
                    help="walk-forward 时把择时固定成 MA(N), 只滚动选阈值。"
                         "-1=择时也参与选参")
    ap.add_argument("--walk", action="store_true",
                    help="walk-forward: 逐年用【它之前】的数据选参数, 只在该年应用(最严谨)")
    ap.add_argument("--select", action="store_true",
                    help="诚实模式: 在 2020 验证集上网格选 (MA, 阈值), 再拿到测试集跑【一次】")
    a = ap.parse_args()

    o = np.load(f"{DIR}/oos_v2{a.tag}.npz")
    l3 = np.load(f"{DIR}/{a.label}")
    up, dn, hold = float(l3["up"]), float(l3["dn"]), int(l3["hold"])
    y3, ret3 = l3["y3"], l3["ret3"]
    days_all = np.load(f"{DIR}/{a.days}")["days"]
    if a.fixed:
        days_all = np.full_like(days_all, hold)

    va = pd.DataFrame({"pu": o["va_p_up"], "pd": o["va_p_dn"]})
    va["ev"] = up * va.pu - dn * va.pd - COST
    te = pd.DataFrame({"day": o["lab_day"], "pu": o["p_up"], "pd": o["p_dn"],
                       "y": y3[o["te_idx"]], "r": ret3[o["te_idx"]],
                       "idxpos": o["te_idx"]})
    te["ev"] = up * te.pu - dn * te.pd - COST
    te = te.sort_values("day")

    if a.walk:
        # walk-forward 滚动选参 —— 这是【唯一】能同时满足两点的做法:
        #   (1) 每个测试点都是真样本外(参数只用它之前的数据定)
        #   (2) 选参数的窗口能随时间变长, 覆盖到牛熊两种 regime
        #
        # ⚠️ 为什么非要它: --select 那种"只用 2020 定参数"看似诚实, 但 2020 是
        #    单一年份且是牛市 —— 择时在牛市里天然没用, 于是它必然选出"不择时",
        #    而测试期 2021-2026 含 2022/2023 熊市, 那才是择时该发挥的地方。
        #    结果是【诚实但无能】: 0.80。验证集不具代表性时, 诚实流程也会给错答案。
        #
        # 做法: 测试期逐年。评 Y 年时, 用 2020~Y-1 的全部数据网格选 (MA, 阈值),
        #       只把选中那一组用在 Y 年上。第一年(2021)只有 2020 可用, 与 --select 等价。
        va_full = pd.DataFrame({"day": o["va_day"], "pu": o["va_p_up"],
                                "pd": o["va_p_dn"], "y": y3[o["va_idx"]],
                                "r": ret3[o["va_idx"]], "idxpos": o["va_idx"]})
        va_full["ev"] = up * va_full.pu - dn * va_full.pd - COST
        pool_df = pd.concat([va_full, te], ignore_index=True).sort_values("day")
        yr = pd.to_datetime(pool_df["day"], unit="D").dt.year
        pool_df = pool_df.assign(yr=yr.to_numpy())
        # ⚠️ --fix-ma 的理由: 按"历史比值最优"选择时参数是脆弱的 ——
        #    择时的价值是【尾部保护】, 而平均比值衡量不了尾部。实测 2024 年
        #    选参窗口(2020-2023)里择时不占优, 于是那年选了"不择时", 正好撞上
        #    年初小盘股流动性踩踏, 单年回撤 −31%, 把整条曲线的比值从 1.17
        #    拉到 0.78。把择时当风控常开、而不是当收益增强去挑, 更符合它的作用。
        #    MA20 + 中证1000 是项目已有的先验(突破预警用的就是这个), 不是在这里
        #    看着测试集挑的。
        ma_list = (a.fix_ma,) if a.fix_ma >= 0 else (0, 5, 10, 20, 40, 60)
        masks = {ma: (timing_mask(ma, a.bench) if ma else None) for ma in ma_list}
        if a.crash:
            masks = {f"crash{int(a.crash * 100)}": crash_mask(a.crash, 20, a.bench)}
        # ⚠️ cap 也必须进网格。上一轮扫 cap=2/4/8 发现 cap=2 比值 1.57 "过线",
        #    但那是【看着测试集挑的】。凡是能调的都要在选参窗口里定, 否则
        #    就是换个花样重犯 Top5% +0.47 -> +0.02 那个错。
        cap_list = (0,) if a.no_cap else (0, 2, 4, 8)
        if a.breadth:
            # 广度开关按每个阈值单独算(合格只数依赖阈值), 所以放进循环里
            masks = {"breadth": "LAZY"}
        test_years = sorted(int(y) for y in pool_df.yr.unique() if y >= 2021)

        print(f"== walk-forward 滚动选参 ({a.tag}, 仓位 {a.slots}) ==")
        print("年份 | 选参窗口   选中参数        | 该年 笔数  年化     回撤    比值")
        eq_all, picks, use_all, skipped = [], [], [], []
        for y in test_years:
            hist = pool_df[pool_df.yr < y]
            cur = pool_df[pool_df.yr == y]
            if len(hist) == 0 or len(cur) == 0:
                continue
            grid = []
            for ma, am in masks.items():
                for q in (0.90, 0.95, 0.98, 0.99, 0.995):
                    thr = float(hist["ev"].quantile(q))
                    if am == "LAZY":
                        am = breadth_mask(pool_df, thr)
                    for cp in cap_list:
                        r = run(hist, days_all, a.slots, thr, am, cp)
                        if r and r["n"] >= 60:
                            grid.append((r["ratio"], ma, q, cp))
            if not grid:
                continue
            grid.sort(key=lambda x: -x[0])
            _, ma, q, cp = grid[0]
            thr = float(hist["ev"].quantile(q))
            mk = masks[ma]
            if mk == "LAZY":
                mk = breadth_mask(pool_df, thr)
            r = run(cur, days_all, a.slots, thr, mk, cp)
            if not r:
                # ⚠️ 绝不能静默 continue: 选参窗口上最优的阈值可能严到下一年
                #    一笔都触发不了(实测 cls512_dn8 在 2024 选了"前0.5%",
                #    2025 整年零信号)。静默跳过会让"负年 0/5"看起来很干净,
                #    而真相是那一年根本没交易 —— 这正是本项目反复栽的静默失效。
                skipped.append((y, ma, q, cp))
                print(f"{y} | 该年零信号(沿用的阈值太严: MA{ma or 0} 前{(1 - q) * 100:.1f}%) "
                      f"—— 跳过, 但这一年【没有被评估】")
                continue
            picks.append((y, ma, q, cp))
            eq_all.append(r["eq"])
            use_all.append(r["use"])
            # ⚠️ 回撤趋近 0 时比值会炸(实测 2026 年 6 笔交易算出 4.8e13),
            #    那不是"策略完美", 是分母没有意义。笔数太少同样不能当结论。
            rs = "  n/a" if abs(r["mdd"]) < 1.0 else f"{r['ratio']:5.2f}"
            warn = "  ⚠️笔数过少" if r["n"] < 30 else ""
            print(f"{y} | {int(hist.yr.min())}-{y - 1}  "
                  f"MA{ma or '--':<3} 前{(1 - q) * 100:4.1f}% cap{cp or '-'}| "
                  f"{r['n']:5d}笔  {r['ann']:+7.1f}%  {r['mdd']:6.1f}%  {rs}{warn}")
        if eq_all:
            # ⚠️ 必须把逐年曲线【首尾相接】再算总回撤 —— 每年独立从 1.0 重启的话,
            #    跨年的那段回撤就被切断了, 数字会好看得多但不是真的。
            #    同理"逐年比值的平均"也不能用: 2026 只有 85 笔、回撤 0.7%,
            #    比值 49 纯粹是分母太小, 平均进去会把整体拉飞。
            chained, base = [], 1.0
            for e in eq_all:
                seg = e / e.iloc[0] * base
                chained.append(seg)
                base = float(seg.iloc[-1])
            full = pd.concat(chained)
            yrs = len(full) / 243
            ann = (float(full.iloc[-1]) ** (1 / yrs) - 1) * 100
            mdd = float(((full / full.cummax()) - 1).min() * 100)
            yearly = [float(e.iloc[-1] / e.iloc[0] - 1) * 100 for e in eq_all]
            neg = sum(1 for x in yearly if x < 0)
            print(f"\n== 串成一条连续净值曲线 ==")
            print(f"  {yrs:.1f} 年  总收益 {(float(full.iloc[-1]) - 1) * 100:+.1f}%  "
                  f"年化 {ann:+.1f}%  最大回撤 {mdd:.1f}%")
            print(f"  平均仓位利用率 {np.mean([r for r in use_all]) * 100:.0f}%"
                  f"  (100% = 满仓)")
            if skipped:
                print(f"  ⚠️ 有 {len(skipped)} 年因零信号未被评估: "
                      f"{[y for y, *_ in skipped]} —— 下面的数字不覆盖这些年")
            print(f"  比值 {ann / abs(mdd):.2f}  "
                  f"{'✅ 过线' if ann / abs(mdd) >= 1 else '❌ 未过 1.0'}   负年 {neg}/{len(yearly)}")
            print(f"  逐年 {[f'{y}:{v:+.0f}%' for (y, *_), v in zip(picks, yearly)]}")
            print(f"  选中参数 {[f'{y}:MA{m or 0}/cap{c or 0}' for y, m, _, c in picks]}")
        return

    if a.select:
        # ⚠️⚠️ 这段存在的唯一理由: 上面那种"扫一遍 MA10/20/60 报最好的"是
        #    【用测试集挑参数】—— 本项目最贵的一次教训就是这么来的
        #    (裸K Top5% 报 +0.47, 换成诚实流程后 +0.02, 差一个数量级)。
        #    这里改成: 在 2020 验证集上把 (MA, 阈值分位) 网格跑完, 按比值选一组,
        #    然后【只在测试集上跑这一组】。测试集只碰一次。
        va_full = pd.DataFrame({"day": o["va_day"], "pu": o["va_p_up"],
                                "pd": o["va_p_dn"], "y": y3[o["va_idx"]],
                                "r": ret3[o["va_idx"]], "idxpos": o["va_idx"]})
        va_full["ev"] = up * va_full.pu - dn * va_full.pd - COST
        va_full = va_full.sort_values("day")
        grid = []
        for ma in (0, 5, 10, 20, 40, 60):
            am = timing_mask(ma, a.bench) if ma else None
            for q in (0.90, 0.95, 0.98, 0.99, 0.995):
                thr = float(va_full["ev"].quantile(q))
                r = run(va_full, days_all, a.slots, thr, am, a.cap)
                if r and r["n"] >= 60:            # 2020 一年, 少于 60 笔没法比
                    grid.append((r["ratio"], ma, q, r))
        if not grid:
            print("验证集上没有任何一组能凑够 60 笔, 无法选参数")
            return
        grid.sort(key=lambda x: -x[0])
        print("== 2020 验证集网格(按比值排, 前5) ==")
        for ratio, ma, q, r in grid[:5]:
            print(f"  MA{ma or '--':<3} 前{(1 - q) * 100:4.1f}%  "
                  f"{r['n']:4d}笔  年化{r['ann']:+6.1f}% 回撤{r['mdd']:6.1f}%  比值 {ratio:5.2f}")
        _, best_ma, best_q, _ = grid[0]
        best_thr = float(va_full["ev"].quantile(best_q))
        am = timing_mask(best_ma, a.bench) if best_ma else None
        rt = run(te, days_all, a.slots, best_thr, am, a.cap)
        print(f"\n== 搬到 2021+ 测试集(只跑这一组) ==")
        print(f"  选中: MA{best_ma or '不择时'} + 验证前{(1 - best_q) * 100:.1f}% "
              f"(ev>={best_thr * 100:+.2f}%)  仓位 {a.slots}")
        print(f"  {rt['n']}笔  平均持有 {rt['hold']:.2f}天  年周转 {rt['turn']:.1f}次")
        print(f"  年化 {rt['ann']:+.1f}%  回撤 {rt['mdd']:.1f}%  "
              f"比值 {rt['ratio']:.2f}  {'✅ 过线' if rt['ratio'] >= 1 else '❌ 未过 1.0'}")
        return

    allow = timing_mask(a.timing, a.bench) if a.timing else None
    if allow:
        on = sum(1 for v in allow.values() if v)
        print(f"择时: {'中证1000' if a.bench == 0 else '沪深300'} > MA{a.timing}, "
              f"可建仓交易日占 {on / len(allow) * 100:.0f}%")
    print(f"{a.tag}  仓位 {a.slots}  手续费 {COST:.2%}  "
          f"{'【对照: 强制持满 %d 天】' % hold if a.fixed else '实际出场日回收资金'}")
    print("阈值(2020验证集定)  |   笔数  平均持有  年周转 |    年化     回撤   比值")
    for q in (0.90, 0.95, 0.98, 0.99, 0.995):
        thr = float(va["ev"].quantile(q))
        r = run(te, days_all, a.slots, thr, allow, a.cap)
        if not r:
            continue
        print(f"ev>={thr * 100:+6.2f}% (前{(1 - q) * 100:4.1f}%) | {r['n']:6d}  "
              f"{r['hold']:6.2f}天  {r['turn']:5.1f}次 | "
              f"{r['ann']:+7.1f}%  {r['mdd']:7.1f}%  {r['ratio']:5.2f}")


if __name__ == "__main__":
    main()
