"""筹码特征的【唯一】实现 —— 训练(prep.py) 与生产(build_chips_scores.py) 都调这里。

⚠️ 为什么必须只留一份: 本项目已经因为"同一份清单/同一套算法抄了两遍"栽过五次
   (板块条漏两页、v4 切换漏两页、v3.5 完全没接数据、SAR/突破/拉升/周线版只接了
   行情页、实时页训练菜单停在4个)。特征更严重 —— 训练和生产算得不一样,
   模型拿到的是另一个分布, 结果只会变差【而不会报错】。

⚠️ 输入约定(两边都必须满足, 否则结果错而不报错):
     - 按 (股票, 日期) 升序排列
     - gid 列标识股票分组边界(训练用 cid, 生产用 ts_code)
     - close 是【后复权价】(close*adj_factor)。训练侧 panel.npz 存的就是这个,
       生产侧查库时也要乘 adj_factor —— 涨幅跨除权日时只有复权价才是真实收益率
     - cyq 各列是【原始价】, 所以只做表内比值, 绝不与 close 相除
       (2026-09-10 第一版就是这么错的: "现价/低位成本"中位 3.602, 算出来的
        其实是 adj_factor 本身, 而且不报错)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 生产端查库/训练端取列都按这个清单来, 少一列下游就 KeyError 而不是静默出错
CYQ_COLS = ["winner_rate", "cost_5pct", "cost_15pct", "cost_50pct",
            "cost_85pct", "cost_95pct", "weight_avg", "his_low", "his_high"]

# daily_basic 的列(换手率族)。2026-09-10 加 —— 见 add_turnover_feats 的说明
DB_COLS = ["turnover_rate", "turnover_rate_f", "volume_ratio",
           "float_share", "free_share", "circ_mv"]

FEAT_NAMES = [
    "获利盘", "获利盘变化5", "获利盘变化10", "获利盘变化20",
    "获利盘背离10", "获利盘背离20",
    "均价在成本区间位置", "中位/均价", "上半区宽度", "下半区宽度",
    "均价在历史区间位置", "筹码集中度(对照)", "筹码宽度(对照)",
]


TURNOVER_NAMES = [
    "换手率", "自由换手率", "量比",
    "换手相对20日", "换手相对60日", "换手20日趋势",
    "自由换手相对60日", "换手波动20日", "流通市值对数",
]


def add_turnover_feats(df: pd.DataFrame, gid: str = "gid") -> pd.DataFrame:
    """换手率族 —— 2026-09-10 加, 目标是补【方向】信息。

    ⚠️ 为什么要加(问题的量化):
       裸K 与筹码模型的共同瓶颈是【方向判别力只有 ±4pt】—— 在真的会动的票里,
       模型最看多的 Top10% 涨占比 52.7%, 最看空的 Bot10% 44.1%, 基准 48.4%。
       也就是它 70% 的本事在"哪只票要动了", 只有 30% 在"往哪动"。
       而此前喂给模型的量是【绝对成交量】: 只能看出"放量了", 看不出"谁在放量"。
       小盘股换手 20% 与大盘股换手 0.5% 可能是同样的成交额, 含义完全不同。

    ⚠️ 每个指标都要【相对它自己的历史】归一化, 不能直接用原值:
       直接用换手率, 模型学到的会是"小盘股换手天生高"这种【静态】差异 ——
       那是市值的代理变量, 不是方向信号。要的是"这只票今天的换手, 相对它
       自己平时算不算异常"。这与筹码族只用表内比值是同一个道理。

    ⚠️ 市值只保留对数流通市值一个: 它是已知的风格因子, 放进去是为了让模型
       能把"小盘效应"与真正的换手异常区分开, 而不是让它去押小盘。
    """
    g = df.groupby(gid, sort=False)
    tr = df["turnover_rate"]
    out = pd.DataFrame(index=df.index)
    out["换手率"] = tr
    out["自由换手率"] = df["turnover_rate_f"]
    out["量比"] = df["volume_ratio"]
    for k in (20, 60):
        ma = g["turnover_rate"].transform(lambda x: x.rolling(k, min_periods=5).mean())
        out[f"换手相对{k}日"] = tr / ma.clip(lower=1e-6)
    ma20 = g["turnover_rate"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    ma60 = g["turnover_rate"].transform(lambda x: x.rolling(60, min_periods=10).mean())
    out["换手20日趋势"] = ma20 / ma60.clip(lower=1e-6)
    maf60 = g["turnover_rate_f"].transform(lambda x: x.rolling(60, min_periods=10).mean())
    out["自由换手相对60日"] = df["turnover_rate_f"] / maf60.clip(lower=1e-6)
    out["换手波动20日"] = (g["turnover_rate"].transform(
        lambda x: x.rolling(20, min_periods=5).std()) / ma20.clip(lower=1e-6))
    out["流通市值对数"] = np.log(df["circ_mv"].clip(lower=1.0))
    return out[TURNOVER_NAMES].astype(np.float32)


SHAPE_NAMES = [
    "横盘压缩60", "横盘压缩20", "距60日高点", "距20日高点", "距60日低点",
    "均线纠缠度", "站上均线数", "回踩不破深度",
    "涨跌量能比", "缩量回踩", "放量突破", "长下影", "洗盘反身",
]


def add_shape_feats(df: pd.DataFrame, gid: str = "gid") -> pd.DataFrame:
    """经典起爆前形态 —— 2026-09-10 用户提出的几类, 量化成特征。

    用户描述的原型:
      1. 长期横盘, 突然突破上限      -> 横盘压缩 + 距高点距离
      2. 大跌洗盘, 逼出筹码后反身    -> 长下影 + 洗盘反身
      3. 老鸭头                     -> 均线纠缠后发散 + 回踩不破
      4. 放量拉升 -> 缩量回踩 -> 再放量 -> 涨跌量能比 + 缩量回踩 + 放量突破

    ⚠️ 与项目以前做法的关键区别: 以前是把这些形态当【信号】(触发就买),
       实测效果有限(突破预警裸跑比值 0.17)。这里当【特征】喂给模型, 让它
       自己去权衡组合 —— 换手率今天刚验证过这条路(方向判别力 +77%)。

    ⚠️ 全部无量纲: 比值或占比, 不含绝对价格/绝对量。否则模型会学到
       "贵的票"和"大盘股"这类静态差异, 那是市值的代理变量不是形态。

    ⚠️ 输入 df 需含 gid / o,h,l,c,v(后复权价), 按 (gid,日期) 升序。
    """
    g = df.groupby(gid, sort=False)
    o_, h_, l_, c_, v_ = df["o"], df["h"], df["l"], df["c"], df["v"]
    out = pd.DataFrame(index=df.index)

    def roll(col, k, fn):
        return g[col].transform(lambda x: getattr(x.rolling(k, min_periods=max(3, k // 4)), fn)())

    for k in (20, 60):
        hi = roll("h", k, "max")
        lo = roll("l", k, "min")
        ma = roll("c", k, "mean").clip(lower=1e-6)
        # 横盘压缩: 区间宽度/均价。越小说明盘得越紧, 突破时爆发力越强
        out[f"横盘压缩{k}"] = (hi - lo) / ma
        out[f"距{k}日高点"] = c_ / hi.clip(lower=1e-6)
    out["距60日低点"] = c_ / roll("l", 60, "min").clip(lower=1e-6)

    # 老鸭头的基础: 均线先纠缠(发散度低)再发散, 且回踩不破
    ma5, ma10, ma20 = roll("c", 5, "mean"), roll("c", 10, "mean"), roll("c", 20, "mean")
    mmax = pd.concat([ma5, ma10, ma20], axis=1).max(axis=1)
    mmin = pd.concat([ma5, ma10, ma20], axis=1).min(axis=1)
    out["均线纠缠度"] = (mmax - mmin) / c_.clip(lower=1e-6)
    out["站上均线数"] = ((c_ > ma5).astype(np.float32) + (c_ > ma10).astype(np.float32)
                        + (c_ > ma20).astype(np.float32))
    # 回踩不破: 近5日最低点相对 MA10 的位置 —— >1 说明回踩没破均线
    out["回踩不破深度"] = roll("l", 5, "min") / ma10.clip(lower=1e-6)

    # 量价配合: 上涨日的量 vs 下跌日的量。>1 = 放量涨、缩量跌(健康)
    prev_c = g["c"].shift(1)
    up_day = (c_ > prev_c).astype(np.float32)
    vma = roll("v", 20, "mean").clip(lower=1e-6)
    vr = v_ / vma
    tmp = pd.DataFrame({"gid": df[gid], "u": vr * up_day, "d": vr * (1 - up_day)})
    tg = tmp.groupby("gid", sort=False)
    up_v = tg["u"].transform(lambda x: x.rolling(20, min_periods=5).sum())
    dn_v = tg["d"].transform(lambda x: x.rolling(20, min_periods=5).sum())
    out["涨跌量能比"] = up_v / dn_v.clip(lower=1e-6)
    # 缩量回踩: 今天跌 且 量明显小于均量
    out["缩量回踩"] = np.where((c_ < prev_c) & (vr < 0.8), 1.0 - vr, 0.0)
    # 放量突破: 今天涨 且 放量 且 创20日新高
    hi20 = roll("h", 20, "max")
    out["放量突破"] = np.where((c_ > prev_c) & (vr > 1.5) & (h_ >= hi20 * 0.995), vr, 0.0)
    # 长下影: 下影线占全天振幅的比例 —— 洗盘的典型形态
    rng = (h_ - l_).clip(lower=1e-6)
    out["长下影"] = (pd.concat([o_, c_], axis=1).min(axis=1) - l_) / rng
    # 洗盘反身: 当天最低跌破前一日收盘 3% 以上, 但收盘收回来
    out["洗盘反身"] = np.where((l_ < prev_c * 0.97) & (c_ > prev_c * 0.995),
                              (c_ - l_) / rng, 0.0)
    return out[SHAPE_NAMES].astype(np.float32)


ADV_NAMES = [
    "CGO", "筹码偏度代理", "上尾厚度", "腰部集中度",
    "换手斜率5_60", "换手变异系数20",
    "涨停次数20", "低换手涨停20", "炸板次数20", "龙虎榜触发20",
]


def add_advanced_feats(df: pd.DataFrame, gid: str = "gid") -> pd.DataFrame:
    """外部研报/制度性特征 —— 2026-09-10 据网络检索结果实现。

    只收录【有明确公式】或【有回测数据支撑】的, 拒绝自媒体的定性描述。
    检索中明确判定"无法量化"的(长期地量后首次放量、假破位、启动信号的
    绝对换手率阈值)一律不做 —— 那些在中文互联网上没有任何可编码定义,
    所有通达信公式源码都被截断在变量声明处, 且无一有回测。

    各特征来源与依据:

    1. CGO(资本利得突出量) —— 广发证券《行为金融因子研究之一》2017-06
       原式: 权重 w_{t-n} = V_{t-n}·Π(1−V_{t-i}), RP = Σw·P/Σw, CGO=(P−RP)/P
       回测 2007-2017 中证500: 低 CGO 组年均超额 8~15%, 高 CGO 组仅 2~8%。
       ⚠️ 这里用 cyq_perf 的 weight_avg 作 RP 的现成代理 —— tushare 的筹码
          分布同样是换手率衰减算的, 两者机制一致。
       ⚠️ 广发的结论方向是"CGO 越低未来收益越高"(处置效应), 而我们的目标是
          右尾爆发, 方向未必一致 —— 但符号信息本身有价值, 让模型自己学。

    2. 筹码分布形状(偏度/峰度的分位数代理) —— 中信建投《筹码分布因子系统构建》
       报告称 kurtosis 族 IC −3.74% / 年化 13.53%, chip_distri 族 IC +3.96% /
       年化 20.10%, 且【筹码因子在中证1000/500 上显著优于沪深300】——
       正好是起爆模型的股票池。
       ⚠️ 报告未披露因子的数学定义(38页原文在付费墙后), 这里是用 5 个分位点
          做的形状近似, 不是原factor。
       ⚠️ 与已证伪的"筹码集中度"(只用 85/15)不是同一个量: 这里用到 95/5 和
          50, 描述的是分布的尾部与偏斜, 不是腰部宽度。

    3. 换手率斜率与变异系数 —— 华泰《单因子测试之换手率类因子》
       研报明确提示同类不同周期高度相关需去冗余, 所以这里做的是与已有
       "换手相对N日"正交的形式: 斜率(短期/长期之比)与离散度(std/mean)。

    4. 涨停形态族 —— 雪球统计(可信度较低, 但可自行验证):
       换手率<3% 的涨停次日胜率 80.2%, 换手率>50% 的仅 45.5%。
       ⚠️ 这些是【次日】尺度的统计, 我们的标签是 20 日 +30%, 不能直接套用;
          作为"过去 N 日内出现过几次"的特征则合适。
       ⚠️ 涨跌停幅度按板块分: 主板 10%, 创业板/科创板 20%(2020-08 起),
          这里用 9.8% 的宽松判据 + 20% 判据同时覆盖, 不做精确的板块判定 ——
          精确判定要 ts_code 前缀 + 日期, 留待需要时再加。

    5. 龙虎榜触发条件 —— 交易所披露规则本身是公开可复现的:
       日涨跌幅偏离值 ±7% / 换手率 20% / 振幅 15%。
       ⚠️ 这是【免费的制度性特征】: 不需要龙虎榜数据本身, 只要算出"今天是否
          触发披露条件"。上榜是 T 日盘后公布, 无前视问题。
    """
    g = df.groupby(gid, sort=False)
    c_ = df["c"]
    out = pd.DataFrame(index=df.index)

    # --- 1. CGO (用 weight_avg 作参考价格 RP) ---
    wa = df["weight_avg"].clip(lower=1e-6)
    out["CGO"] = (c_ - wa) / c_.clip(lower=1e-6)

    # --- 2. 筹码分布形状 ---
    c5, c50, c85, c95 = (df["cost_5pct"], df["cost_50pct"],
                         df["cost_85pct"], df["cost_95pct"])
    span = (c95 - c5).clip(lower=1e-6)
    # 中位数 vs 加权均值的偏离 = 偏度代理(分布右偏时均值 > 中位数)
    out["筹码偏度代理"] = (c50 - wa) / wa
    # 上尾厚度: 95~85 相对 85~50 —— 高位筹码的尾部有多重
    out["上尾厚度"] = (c95 - c85) / (c85 - c50).clip(lower=1e-6)
    # 腰部集中 vs 全域宽度 = 峰度代理
    out["腰部集中度"] = (c85 - df["cost_15pct"]) / span

    # --- 3. 换手率斜率与离散度 ---
    t5 = g["turnover_rate"].transform(lambda x: x.rolling(5, min_periods=2).mean())
    t60 = g["turnover_rate"].transform(lambda x: x.rolling(60, min_periods=10).mean())
    out["换手斜率5_60"] = t5 / t60.clip(lower=1e-6)
    t20m = g["turnover_rate"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    t20s = g["turnover_rate"].transform(lambda x: x.rolling(20, min_periods=5).std())
    out["换手变异系数20"] = t20s / t20m.clip(lower=1e-6)

    # --- 4. 涨停形态族 ---
    prev_c = g["c"].shift(1)
    pct = c_ / prev_c.clip(lower=1e-6) - 1
    up_limit = ((pct >= 0.098) & (pct < 0.115)) | (pct >= 0.198)   # 主板 / 双创
    intraday_limit = (df["h"] / prev_c.clip(lower=1e-6) - 1 >= 0.098)
    zhaban = intraday_limit & ~up_limit                             # 摸过涨停但没收在涨停
    low_turn_limit = up_limit & (df["turnover_rate"] < 3.0)
    for nm, ser in [("涨停次数20", up_limit), ("低换手涨停20", low_turn_limit),
                    ("炸板次数20", zhaban)]:
        tmp = pd.DataFrame({"g": df[gid], "v": ser.astype(np.float32)})
        out[nm] = tmp.groupby("g")["v"].transform(
            lambda x: x.rolling(20, min_periods=5).sum())

    # --- 5. 龙虎榜披露条件(制度性, 免费) ---
    amp = (df["h"] - df["l"]) / prev_c.clip(lower=1e-6)
    lhb = (pct.abs() >= 0.07) | (df["turnover_rate"] >= 20.0) | (amp >= 0.15)
    tmp = pd.DataFrame({"g": df[gid], "v": lhb.astype(np.float32)})
    out["龙虎榜触发20"] = tmp.groupby("g")["v"].transform(
        lambda x: x.rolling(20, min_periods=5).sum())

    return out[ADV_NAMES].astype(np.float32)


def chip_feats(df: pd.DataFrame, gid: str = "gid") -> pd.DataFrame:
    """算 13 个筹码特征。df 需含 gid / close / CYQ_COLS，按 (gid, 日期) 升序。

    返回与 df 同索引的特征表，列顺序 = FEAT_NAMES。
    """
    g = df.groupby(gid, sort=False)
    wr = df["winner_rate"]
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # --- 获利盘族: 训练里重要性前 6 名合计 60.0% ---
    out["获利盘"] = wr
    for k in (5, 10, 20):
        out[f"获利盘变化{k}"] = wr - g["winner_rate"].shift(k)
    # 获利盘背离 = 获利盘变化 − 价格涨幅。价格没动而获利盘上升 = 有人在低位接浮筹。
    # 文档: Q10/Q1=2.16, 过了价格位置×量能的九宫格控制后仍有 2.09
    for k in (10, 20):
        px_chg = (close / g["close"].shift(k) - 1) * 100
        out[f"获利盘背离{k}"] = out[f"获利盘变化{k}"] - px_chg

    # --- 成本结构: 只用 cyq 表内部比值(见模块 docstring 的复权口径警告) ---
    c5, c15 = df["cost_5pct"], df["cost_15pct"]
    c50, c85, c95 = df["cost_50pct"], df["cost_85pct"], df["cost_95pct"]
    wa = df["weight_avg"]
    span = (c95 - c5).clip(lower=1e-6)
    c50s = c50.clip(lower=1e-6)
    out["均价在成本区间位置"] = (wa - c5) / span
    out["中位/均价"] = c50 / wa.clip(lower=1e-6)
    out["上半区宽度"] = (c95 - c50) / c50s
    out["下半区宽度"] = (c50 - c5) / c50s
    span_h = (df["his_high"] - df["his_low"]).clip(lower=1e-6)
    out["均价在历史区间位置"] = (wa - df["his_low"]) / span_h

    # --- 对照特征: 文档已证伪(集中度 Q10/Q1=1.10)。留着是为了每次训练都能
    #     确认它确实排在末尾, 而不是默默把它当成有用的 ---
    out["筹码集中度(对照)"] = (c85 - c15) / c50s
    out["筹码宽度(对照)"] = span / c50s

    return out[FEAT_NAMES].astype(np.float32)


def bad_rows(df: pd.DataFrame) -> pd.Series:
    """成本价异常的行(停牌/数据错误)。这些行的比值会爆掉 —— 实测未剔时
    "上半区宽度"均值 153.965 而中位只有 0.132。"""
    return (~np.isfinite(df["cost_50pct"]) | (df["cost_50pct"] <= 0)
            | ~np.isfinite(df["weight_avg"]) | (df["weight_avg"] <= 0))
