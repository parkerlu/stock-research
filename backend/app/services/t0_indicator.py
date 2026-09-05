"""日内做 T 指标 —— 在分时图上标出"到收盘还有 3% 空间"的买卖点。

模型: XGBoost 二分类, 5min bar 粒度, 2020-2023 训练 / 2024-2026 样本外验证。
样本外表现(阈值 0.7): 卖点精确率 64.7%, 买点 54.4%; 中位收益约 +3%。

⚠️ 特征口径必须与训练时逐字对齐, 否则模型静默失效(不会报错, 只会给出错误概率):
  - 成交量单位: 腾讯给"手", 训练用"股" → 一律 ×100
  - 昨收基准: 用交易所公布的昨收(已除权调整), 与训练时的前复权昨收同口径
  - VWAP: 用 close×vol 累计计算(与训练一致), 不用上游的 avg_price
  - 5min close: 取整 5 分钟时刻的价格, 与 baostock bar 结束时间戳对齐
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path("/app/models")
BARS_PER_DAY = 48          # 5min bar: 09:35 … 15:00
FIRST_BAR, LAST_BAR = 3, 39  # 只在 09:50–14:35 出信号(过早特征不足, 过晚没空间)
TARGET = 0.03

# 训练时的特征顺序 —— 不可改动
FEATURES = [
    "日内位置", "VWAP偏离", "今日累计涨跌", "相对昨收", "动量3", "动量6",
    "已实现振幅", "单bar量比", "累计量比", "距涨停", "距跌停", "剩余时间",
    "开盘跳空", "昨日振幅", "昨日收益", "涨跌停幅", "涨跌幅占涨跌停", "振幅占涨跌停",
]


def limit_of(ts_code: str) -> float:
    """涨跌停幅度 —— 决定日内波动空间, 是做 T 的量纲基准。"""
    code = ts_code.split(".")[0]
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("4", "8", "920")):
        return 0.30            # 北交所
    return 0.10


def is_supported(ts_code: str) -> bool:
    """模型只在训练覆盖的板块上有效。

    训练集 = 沪深主板 / 创业板 / 科创板(307 只, 已剔除 ST)。北交所既不在训练集内
    (纯外推, 概率不可信), 涨跌停也是 ±30%, 波动结构完全不同 —— 直接不出信号。
    """
    code, _, mkt = ts_code.partition(".")
    if mkt.upper() == "BJ" or code.startswith(("4", "8", "920")):
        return False
    return code.startswith(("600", "601", "603", "605", "688", "689",
                            "000", "001", "002", "003", "300", "301"))


@lru_cache(maxsize=2)
def _load(side: str):
    import xgboost as xgb

    path = MODEL_DIR / f"t0_{side}.json"
    if not path.exists():
        logger.warning("做T模型缺失: %s", path)
        return None
    m = xgb.XGBClassifier()
    m.load_model(str(path))
    return m


def _slot(hhmm: str) -> int:
    """"1035" → 5min bar 序号 (0 = 09:35, 47 = 15:00); 非整 5 分或非交易时段返回 -1。"""
    h, m = int(hhmm[:2]), int(hhmm[2:])
    mins = h * 60 + m
    am0, am1, pm0, pm1 = 9 * 60 + 30, 11 * 60 + 30, 13 * 60, 15 * 60
    if am0 < mins <= am1:
        off = mins - am0
    elif pm0 < mins <= pm1:
        off = (am1 - am0) + (mins - pm0)
    else:
        return -1
    if off % 5:
        return -1
    return off // 5 - 1


def aggregate_5min(bars: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    """腾讯 1min 分时 → 5min close / volume(股)。返回 (close, vol, 今开)。"""
    close = np.full(BARS_PER_DAY, np.nan)
    vol = np.zeros(BARS_PER_DAY)
    open_px = np.nan
    for b in bars:
        t = str(b.get("time", ""))
        if len(t) != 4:
            continue
        if t == "0930" and np.isnan(open_px):
            open_px = float(b.get("price") or np.nan)
        s = _slot(t)
        price = float(b.get("price") or 0)
        if s >= 0 and price > 0:
            close[s] = price
        # 量按 5min 归集。⚠️ 09:30 这一分钟含集合竞价, 量往往是全天最大的几笔之一,
        # 必须计入第一根 bar —— baostock 的 09:35 bar 就是含集合竞价的。
        # 用 mins < am0 排除(而非 <=), 否则 09:30 被整根丢弃, 累计量比会系统性偏低。
        h, m = int(t[:2]), int(t[2:])
        mins = h * 60 + m
        am0, am1, pm0 = 9 * 60 + 30, 11 * 60 + 30, 13 * 60
        if am0 <= mins <= am1:
            off = mins - am0
        elif pm0 <= mins <= 15 * 60:
            off = (am1 - am0) + (mins - pm0)
        else:
            continue
        idx = max((off - 1) // 5, 0)
        if 0 <= idx < BARS_PER_DAY:
            vol[idx] += float(b.get("vol") or 0) * 100.0  # 手 → 股
    if np.isnan(open_px):
        first = np.where(np.isfinite(close))[0]
        open_px = float(close[first[0]]) if len(first) else np.nan
    return close, vol, open_px


def build_features(
    close: np.ndarray, vol: np.ndarray, open_px: float, prev_close: float,
    prev_amp: float, prev_ret: float, prev_vol_shares: float, lim: float,
) -> tuple[np.ndarray, list[int]]:
    """逐个 5min 时点构造特征矩阵。返回 (X, 对应的 bar 序号)。"""
    real = np.isfinite(close)
    if not real.any() or not np.isfinite(prev_close) or prev_close <= 0:
        return np.empty((0, len(FEATURES))), []
    # ⚠️ 盘中只有部分 bar 存在。前值填充【只能填到最后一根真实 bar】,
    # 否则会给尚未发生的时点造出特征(动量恒为 0、量比恒为 0), 盘中实时会出错信号。
    first_real = int(np.where(real)[0][0])
    last_real = int(np.where(real)[0][-1])
    filled = close.copy()
    filled[:first_real] = close[first_real]
    for i in range(first_real + 1, last_real + 1):
        if not np.isfinite(filled[i]):
            filled[i] = filled[i - 1]
    filled[last_real + 1:] = np.nan

    head = filled[: last_real + 1]
    cmax = np.maximum.accumulate(head)
    cmin = np.minimum.accumulate(head)
    vc = np.cumsum(vol[: last_real + 1])
    vwap = np.cumsum(head * vol[: last_real + 1]) / np.maximum(vc, 1)

    rows, idx = [], []
    last = min(LAST_BAR, last_real)
    for t in range(FIRST_BAR, last + 1):
        c = filled[t]
        if not np.isfinite(c) or c <= 0:
            continue
        rng = max(cmax[t] - cmin[t], 1e-9)
        m3 = filled[max(t - 3, 0)]
        m6 = filled[max(t - 6, 0)]
        rows.append([
            (c - cmin[t]) / rng,                                  # 日内位置
            c / vwap[t] - 1 if vwap[t] > 0 else 0.0,              # VWAP偏离
            c / open_px - 1,                                      # 今日累计涨跌
            c / prev_close - 1,                                   # 相对昨收
            c / m3 - 1 if m3 > 0 else 0.0,                        # 动量3
            c / m6 - 1 if m6 > 0 else 0.0,                        # 动量6
            rng / open_px,                                        # 已实现振幅
            vol[t] / max(vc[t] / (t + 1), 1),                     # 单bar量比
            vc[t] / max(prev_vol_shares * (t + 1) / 48, 1),       # 累计量比
            (prev_close * (1 + lim) - c) / c,                     # 距涨停
            (c - prev_close * (1 - lim)) / c,                     # 距跌停
            float(47 - t),                                        # 剩余时间
            open_px / prev_close - 1,                             # 开盘跳空
            prev_amp,                                             # 昨日振幅
            prev_ret,                                             # 昨日收益
            lim,                                                  # 涨跌停幅
            (c / prev_close - 1) / lim,                           # 涨跌幅占涨跌停
            (rng / open_px) / lim,                                # 振幅占涨跌停
        ])
        idx.append(t)
    if not rows:
        return np.empty((0, len(FEATURES))), []
    return np.asarray(rows, dtype=np.float32), idx


BAR_TIME = [
    "0935", "0940", "0945", "0950", "0955", "1000", "1005", "1010", "1015", "1020",
    "1025", "1030", "1035", "1040", "1045", "1050", "1055", "1100", "1105", "1110",
    "1115", "1120", "1125", "1130", "1305", "1310", "1315", "1320", "1325", "1330",
    "1335", "1340", "1345", "1350", "1355", "1400", "1405", "1410", "1415", "1420",
    "1425", "1430", "1435", "1440", "1445", "1450", "1455", "1500",
]


def compute_signals(
    ts_code: str, bars: list[dict], prev_close: float,
    prev_amp: float, prev_ret: float, prev_vol_hands: float,
    threshold: float = 0.7,
) -> list[dict]:
    """算出当日各 5min 时点的做 T 信号。

    prev_vol_hands: 昨日成交量(手, 直接取 daily_candle.vol)
    返回: [{time, bar, side, prob, price}] —— side 为 'sell'(高抛) / 'buy'(低吸)
    """
    if not is_supported(ts_code):
        return []
    up, dn = _load("up"), _load("dn")
    if up is None or dn is None:
        return []
    close, vol, open_px = aggregate_5min(bars)
    X, idx = build_features(
        close, vol, open_px, prev_close, prev_amp, prev_ret,
        float(prev_vol_hands or 0) * 100.0, limit_of(ts_code),
    )
    if len(idx) == 0:
        return []
    p_up = up.predict_proba(X)[:, 1]
    p_dn = dn.predict_proba(X)[:, 1]

    # 原始信号 → 配对成完整的一次 T。
    # 做 T 是一卖一买(或一买一卖)的来回, 只给入场点没有意义。
    # 持仓期间不再开新仓 —— 手上已经没货(或没现金)了, 重复提示是噪声。
    real = np.isfinite(close)
    last_real = int(np.where(real)[0][-1]) if real.any() else -1
    trades: list[dict] = []
    busy_until = -1
    for k, t in enumerate(idx):
        if t <= busy_until or not np.isfinite(close[t]):
            continue
        sell = p_dn[k] >= threshold
        buy = p_up[k] >= threshold
        if not (sell or buy):
            continue
        entry = float(close[t])
        goal = entry * (1 - TARGET) if sell else entry * (1 + TARGET)
        exit_bar, reason = None, None
        for j in range(t + 1, last_real + 1):
            if not np.isfinite(close[j]):
                continue
            if (sell and close[j] <= goal) or (buy and close[j] >= goal):
                exit_bar, reason = j, "达标"
                break
        if exit_bar is None:
            exit_bar = last_real
            reason = "收盘平" if last_real >= BARS_PER_DAY - 1 else "持有中"
        exit_px = float(close[exit_bar]) if np.isfinite(close[exit_bar]) else entry
        ret = (entry / exit_px - 1) if sell else (exit_px / entry - 1)
        # 持仓期最大不利偏移(MAE) —— 这一次 T 中途最多浮亏多少。
        # 只看终值会严重低估风险: 高抛后股价先冲高、低吸后先杀跌, 都可能让人中途扛不住。
        seg = close[t: exit_bar + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            worst = float(seg.max()) if sell else float(seg.min())
            mae = (entry / worst - 1) if sell else (worst / entry - 1)
        else:
            mae = 0.0
        trades.append({
            "side": "sell" if sell else "buy",
            "prob": round(float(p_dn[k] if sell else p_up[k]), 4),
            "entry_time": BAR_TIME[t], "entry_bar": t, "entry_price": round(entry, 2),
            "target": round(goal, 2),
            "exit_time": BAR_TIME[exit_bar], "exit_bar": exit_bar,
            "exit_price": round(exit_px, 2), "exit_reason": reason,
            "ret": round(ret, 4), "mae": round(min(mae, 0.0), 4),
            "worst_price": round(worst, 2) if len(seg) else None,
        })
        busy_until = exit_bar
    return trades
