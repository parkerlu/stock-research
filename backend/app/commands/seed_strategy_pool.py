"""灌入策略池的定义 —— 策略的唯一定义源。

⚠️ 这个文件是策略定义的【唯一出处】。以前散在三处:
   演示回放的下拉硬编码在 paper.py 的 DEMO_STRATEGIES、实操盘规则散在每个
   账户的 config、说明写在各命令的 docstring 里 —— 改一个策略要动三处,
   而且没人知道该以哪个为准。

⚠️ detail 写的是【验证过什么、短板在哪】, 不是营销话术。
   这个项目里被证伪的东西太多了(v4 的假 1.70、SAR/突破的 2.62、裸K 的 0.75),
   所以每条都要写清"这个数字是在什么口径下得到的"。

⚠️ since 必须是该策略信号表的真实起点。补算历史信号后要回来改这里,
   否则演示回放会让人选一个根本没信号的日期然后点半天。

用法: python -m app.commands.seed_strategy_pool          # 幂等, 可反复跑
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models.schema import StrategyPool

log = logging.getLogger("seed_pool")

# 出场规则的几个常用组合 —— 必须与各自训练标签一致
_BASE = {"lot": 100, "entry_mode": "next_open", "breakeven": False,
         "trail_pct": None, "trail_arm_pct": None, "trail_max_days_to_arm": None,
         "tier2_pct": 9.99, "tier2_frac": 1.0, "commission_pct": 0.0003,
         "commission_min": 5.0, "transfer_pct": 0.00001, "entry_slip_pct": 0.0,
         "min_amount_k": 5000}


def cfg(stop: float, tier1: float, hold: int, **kw) -> dict:
    return {**_BASE, "stop_pct": stop, "tier1_pct": tier1, "tier1_frac": 1.0,
            "max_hold_days": hold, **kw}


POOL = [
    {
        "key": "boom30", "name": "起爆模型", "sort_order": 10, "is_live": True,
        "slots": 20, "since": "2026-08", "ratio": 2.11,
        "summary": "筹码+换手率 22 特征 · 抓 20 个交易日内涨 30% 的票 · 目前唯一通过全部防伪检验的策略",
        "detail": (
            "【抓什么】次日开盘买入, 20 个交易日内最高价触及 +30% 为成功, 先碰 −8% 止损。\n"
            "【为什么有效】方向判别力 +4.3pt, 是 +10% 那版(+2.3pt)的近两倍 —— "
            "起爆前的形态比\"涨10%前的形态\"更有特征, 后者太普通, 模型学不出方向。\n"
            "【验证】walk-forward 滚动选参 比值 2.11 / 年化 +42.2% / 回撤 −20.0% / 六年零负年, "
            "且六年【每一年都选中 MA10 择时】, 参数选择本身稳定。\n"
            "四道防伪检验: 随机对照 0.43(说明是选股挣的不是规则凑的) · "
            "买入日一字涨停 0.16%(形态模型 v4 当年是 22%, 那批买不到的票贡献了全部收益) · "
            "信号 20 日均额中位 5717 万 · 加 >2000万 流动性约束后 2.31。\n"
            "【择时是策略的一部分】不择时只有 1.15, MA10 择时 3.37 —— 同时提高年化并砍掉一半回撤。"
            "起爆票是高波动小盘股与中证1000 高度同步, 一个大盘开关就能整批挡住。\n"
            "【短板】命中率只有 11.8%(基准 7.8%), 八成半的时候是错的, 靠 30:8 的赔率赚钱。"
            "资金曲线水下时间占 45%, 最长连续 8 个月 —— 执行它需要的心理承受力远超普通策略。\n"
            "【容量】门槛与收益: 不限 2.64 / >2000万 2.31 / >5000万 1.88 / >1亿 1.11。"
            "以 5000 万为界、20 仓位, 约能容纳千万级资金。"
        ),
        "config": cfg(0.08, 0.30, 30, min_amount_k=20000),
    },
    {
        "key": "chips", "name": "筹码模型", "sort_order": 20, "is_live": True,
        "slots": 20, "since": "2024-12", "ratio": 0.89,
        "summary": "只用筹码分布的获利盘族 · 抓 10 根K线内涨 10% · 纯选股, 择时贡献为 0",
        "detail": (
            "【抓什么】次日开盘买入, 10 根K线内先碰 +10% 记涨 / 先碰 −8% 记跌。\n"
            "【为什么有效】特征重要性前 6 名全是获利盘族(合计 60.0%), 筹码集中度垫底 —— "
            "独立复现了 2026-09-05 在完全不同目标上得到的结论(\"有效的只有获利盘族\")。\n"
            "【验证】walk-forward + 资金池 + 真实周转, 比值 0.89 / 年化 +15.9% / 回撤 −17.9% / 六年 2 个浅负年。\n"
            "【它是纯选股】择时 vs 选股拆解显示它的择时贡献为 0(同一批交易日买全市场 +0.37%, "
            "等于所有交易日的 +0.37%), 99% 的交易日都出信号。裸K 那个模型 85% 的交易挤在 "
            "10% 的日子里靠挑日子挣钱, 所以回撤 −31%; 这个仓位不堆在少数几天, 回撤只有 −17.9%。\n"
            "【短板】比值未过 1.0。训练集只有 2018-2019 两年(筹码数据 2018 才有), "
            "而实测训练样本存在悬崖(23万尚可 / 5.3万崩溃), 这个模型是 20 万, 已贴着线。"
        ),
        "config": cfg(0.08, 0.10, 15),
    },
    {
        "key": "combo_ck", "name": "共振(筹码×裸K)", "sort_order": 30, "is_live": True,
        "slots": 20, "since": "2026-09", "ratio": 1.07,
        "summary": "筹码模型与裸K CNN 的 EV 等权平均 · 两者当日分位相关仅 +0.021, 几乎正交",
        "detail": (
            "【抓什么】与筹码模型同一个题目(10根K线 +10%/−8%), 但把两个模型的每笔净期望等权平均。\n"
            "【为什么有效】两个模型信息源完全独立 —— 一个看 400 根K线的形态, 一个看持仓成本结构, "
            "当日横截面分位相关只有 +0.021。纯选股超额是【超加】的: 筹码 +0.33pp、裸K +0.37pp, "
            "合起来 +0.45pp。\n"
            "【验证】比值 1.07 / 年化 +27.7% / 回撤 −26.0% / 六年零负年。\n"
            "组合方式过了诚实复核: 在 2020 验证集上比 avg/rank/min = 5.74/2.62/1.66 选出 avg, "
            "测试集只跑一次 —— 不是在测试集上挑的。\n"
            "【短板】要跑 CNN 推理, 全市场一天 3.7 分钟 / 743MB, 是整条链里最慢的一环, "
            "所以历史信号只能补最近的(1700 个交易日要 100 小时)。"
        ),
        "config": cfg(0.08, 0.10, 15),
    },
    {
        "key": "liftalert", "name": "拉升预警", "sort_order": 40, "is_live": True,
        "slots": 5, "since": "2019-01", "ratio": 0.71,
        "summary": "动力线 × 主力吸筹强档 · 全项目第一个逐年/按天t/九宫格全部通过的指标",
        "detail": (
            "【抓什么】动力线上穿 0.2, 且近 5 日内出现过主力吸筹强档(rank>=0.95)。\n"
            "【为什么有效】命中率 32.6%(基准 17.45%, 1.87倍), 逐年 0 负年, 按天 t=4.79, "
            "波动×市值九宫格全为正 —— 目前唯一各关全过的指标。\n"
            "【三个反直觉处】动力线单用几乎无效(18.7% vs 基准 17.45%), 但叠上去把吸筹从 25.7% "
            "收紧到 32.6% —— 它的价值不是当基底, 是当最后一道过滤。\n"
            "【短板】信号极稀疏, 全市场约 2 只/天。组合级 0.71, 信号级各关全过但组合级仍不达标 —— "
            "这正是\"信号级 ≠ 组合级\"最典型的例子。"
        ),
        "config": cfg(0.06, 0.10, 15),
    },
    {
        "key": "mmweek", "name": "买卖很准 周线版", "sort_order": 50, "is_live": True,
        "slots": 8, "since": "2017-02", "ratio": 0.19,
        "summary": "周线买线>0 的【状态】(不是买点) · 持有 8 周 · 日线版是负的, 换到周线才转正",
        "detail": (
            "【抓什么】周线的买线 >0 的每一周都算信号(状态量), 下周一开盘买入、持有 8 周。\n"
            "【为什么有效】同一个信号, 日线版超同日全市场 −1.00pp(负的), 周线版 +3.39pp。"
            "价值在\"处于超卖状态\"这个持续条件本身, 不在\"刚进入\"那一刻 —— "
            "所以取状态而非边沿, 周线天然滤掉了日线的噪声抖动。\n"
            "按周 t=5.56, 九格最小 +0.30pp(全正), 九年只有 2020/2023 两个浅负年(−1.7/−1.3pp), "
            "都是极端抱团行情(资金集中在少数龙头, 而它抓的是超卖反弹的普通股)。\n"
            "【出场规则特殊】不设止损止盈, 只按持有期(35 天)出场 —— 它是状态指标, "
            "中途止盈会破坏\"持有 8 周\"的验证口径。持仓表里止损止盈显示横杠就是这个原因。\n"
            "【短板】组合级只有 0.19, 信号量大但相关性高。"
        ),
        "config": {**_BASE, "stop_pct": 0.99, "tier1_pct": 9.99, "tier1_frac": 0.5,
                   "max_hold_days": 35},
    },
    {
        "key": "mmweek_f", "name": "周线版(形态过滤)", "sort_order": 60, "is_live": True,
        "slots": 8, "since": "2026-06", "ratio": 0.22,
        "summary": "周线版信号剔掉形态模型 Bot20% · 胜率 +3.9pp, 但不是干净的胜利",
        "detail": (
            "【抓什么】周线版的信号, 再剔掉形态模型当日打分最差 20% 的票。\n"
            "【为什么这么做】形态模型的空头端比多头端强得多(它识别\"不该买\"比\"该买\"准), "
            "所以只拿它当排除过滤器, 不当选股信号。\n"
            "【验证】叠加后 胜率 45.8% -> 49.7%, 比值 0.19 -> 0.22, 剔掉 19.6% 的信号。"
            "胜率 +3.9pp 是最扎实的一项。\n"
            "【短板】不是干净的胜利: 最差的几年明显改善(2022 −33→−24, 2023 −12→−5, 2026 −17→+2), "
            "但 2021 +13→−11、2024 +13→+1 反而变差。"
        ),
        "config": {**_BASE, "stop_pct": 0.99, "tier1_pct": 9.99, "tier1_frac": 0.5,
                   "max_hold_days": 35},
    },
    {
        "key": "shape", "name": "形态模型", "sort_order": 70, "is_live": True,
        "slots": 20, "since": "2024-03", "ratio": 0.24,
        "summary": "40 个无量纲形状特征, 构造上杜绝 regime 记忆 · 价值在空头端, 当过滤器用",
        "detail": (
            "【抓什么】三重障碍标签(+15%/−8%/20交易日)下真正落袋的超额, 每日打分取 Top1%。\n"
            "【去记忆是构造上杜绝的】特征全是比值(无绝对价格/日期/代码/行业/指数/市值) + "
            "每日横截面转分位 + 标签取超额。审计: 每日均分与当日大盘涨跌相关 −0.036。\n"
            "【验证】带成交约束的组合比值 0.24 —— 未达标。\n"
            "【真正的价值在空头端】Bot20% 六年一致跑输, 拿来当排除过滤器叠在周线版上: "
            "胜率 45.8%→49.7%。所以它以【过滤器】身份存在, 不作选股信号。\n"
            "【重要教训】它的前身 v4 曾报出组合比值 1.70, 补上涨跌停成交约束后变成 −0.10 —— "
            "22% 的买入是次日一字涨停, 那批【买不到】的单子贡献了全部收益。"
            "这条是本项目\"成交约束必须在第一次回测就加\"这条规矩的来源。"
        ),
        "config": cfg(0.08, 0.15, 18),
    },
]


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    eng = create_async_engine(settings.database_url)
    async with eng.begin() as c:
        for e in POOL:
            st = pg_insert(StrategyPool).values(**e)
            await c.execute(st.on_conflict_do_update(
                index_elements=["key"],
                set_={k: getattr(st.excluded, k) for k in e if k != "key"}))
        rows = (await c.execute(text(
            "select key, name, since, ratio, is_live from strategy_pool "
            "order by sort_order"))).fetchall()
    for r in rows:
        log.info("  %-10s %-16s 起点 %s  比值 %s  %s", r[0], r[1], r[2],
                 f"{float(r[3]):.2f}" if r[3] is not None else "  - ",
                 "实操盘" if r[4] else "")
    log.info("策略池 %d 条", len(rows))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
