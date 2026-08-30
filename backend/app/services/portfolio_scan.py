"""组合信号扫描 — 把回测验证过的三源组合落地成每日可执行的买入清单.

配置来自全 universe / 2015-2026 / 复利 + 冲击成本的回测:
    chan-1buy-wide + tdx-dual-kdj + chan-2buy, 20 仓位, 低流动性优先
    -> 14.52x / 年化 25.8% / 最大回撤 30.0% / 最长回撤 463 天
    (同期基准 等权买入持有 2.25x / 年化 7.2% / 回撤 59.5%)

三条规则直接照搬回测, 不可随意改动 —— 改了就不是那个回测结果了:

1. universe: 剔除 ETF / 688 科创 / 92|8|4 北交所 / ST 退市。
2. 流动性下限 20 日均额 ≥ 500 万, 且**低流动性优先**排序。
   这是反直觉但回测反复验证的: 高流动性优先只有年化 15~16%, 低流动性优先
   才有 25.8%。收益主体是小盘因子暴露, 不是信号本身(随机入场对照显示信号
   只贡献约 +6.8 个百分点)。
3. 涨停日不出信号 —— 买不进。

⚠️ 容量上限约 500 万本金。超过后冲击成本吃掉超额: 1000 万时年化降到 23.8%,
3000 万时 20.7% 且最优仓位要从 20 提到 30。
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import DailyCandle, StockBasic
from app.services.strategy_templates import TEMPLATE_REGISTRY

log = logging.getLogger(__name__)

# 回测验证过的三个信号源。顺序无关 —— 同一票多源触发时按 STRATEGIES 顺序取第一个。
STRATEGIES = ["chan-1buy-wide", "chan-2buy", "tdx-dual-kdj"]
STRATEGY_LABEL = {
    "chan-1buy-wide": "缠论1买(宽)",
    "chan-2buy": "缠论2买",
    "tdx-dual-kdj": "日周KDJ共振",
}

MIN_AMOUNT_K = 5000      # 20 日均额下限, 千元 (= 500 万)
HISTORY_DAYS = 400       # 缠论需要足够的历史才能识别笔和背驰
MIN_BARS = 130


def _universe_filter(ts_code: str, name: str | None) -> bool:
    if ts_code.startswith(("688", "92", "8", "4")):
        return False
    if name and ("ST" in name or "退" in name):
        return False
    return True


async def load_panel(db: AsyncSession, end: date) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """一次性拉全市场近 HISTORY_DAYS 的日线, 按票分组.

    逐票查询在 5000+ 只票上要几分钟; 单次批量查询几秒就够, 代价是峰值内存 ——
    所以只取需要的列并降到 float32 (生产机只有 1.6G 内存)。
    """
    start = end - timedelta(days=HISTORY_DAYS)
    rows = (await db.execute(
        select(DailyCandle.ts_code, DailyCandle.trade_date, DailyCandle.open,
               DailyCandle.high, DailyCandle.low, DailyCandle.close,
               DailyCandle.vol, DailyCandle.amount, DailyCandle.adj_factor)
        .where(DailyCandle.trade_date >= start, DailyCandle.trade_date <= end)
    )).all()
    if not rows:
        return {}, {}

    df = pd.DataFrame(rows, columns=["ts_code", "trade_date", "open", "high", "low",
                                     "close", "vol", "amount", "adj_factor"])
    basics = (await db.execute(
        select(StockBasic.ts_code, StockBasic.name, StockBasic.is_active)
    )).all()
    names = {c: n for c, n, _ in basics}
    active = {c for c, _, a in basics if a is not False}
    keep = {c for c in df.ts_code.unique()
            if c in active and _universe_filter(c, names.get(c))}
    df = df[df.ts_code.isin(keep)]

    for c in ("open", "high", "low", "close", "amount", "adj_factor"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").astype("float64")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    panel: dict[str, pd.DataFrame] = {}
    for cd, g in df.groupby("ts_code", sort=False):
        if len(g) < MIN_BARS:
            continue
        g = g.sort_values("trade_date").reset_index(drop=True)
        # 前复权 —— 与 quote_service 同口径: 以该票区间内最后一个 adj_factor 为基准
        latest = g.adj_factor.iloc[-1]
        if latest and latest > 0:
            f = (g.adj_factor / latest).values
            for c in ("open", "high", "low", "close"):
                g[c] = (g[c].values * f).round(4)
        panel[cd] = g[["trade_date", "open", "high", "low", "close", "vol", "amount"]]
    return panel, names


def scan_panel(
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
    lookback_days: int,
    top_n: int,
    on_progress=None,
) -> list[dict]:
    """在已加载的面板上跑三个策略, 返回按组合规则排序的候选。"""
    end = max(g.trade_date.iloc[-1] for g in panel.values())
    cutoff = end - pd.Timedelta(days=lookback_days)

    insts = {}
    for tid in STRATEGIES:
        cls = TEMPLATE_REGISTRY[tid]
        try:
            params = cls.parameter_candidates()[0]
            insts[tid] = cls(**params) if params else cls()
        except Exception as exc:          # noqa: BLE001
            log.warning("组合扫描: 策略 %s 无法实例化 — %s", tid, exc)

    # 按信号源分桶 —— 三个源的信号密度差 15 倍 (近5日: chan-1buy-wide 48 只 /
    # chan-2buy 696 / tdx-dual-kdj 600), 若全局只按流动性排序, 稀疏的
    # chan-1buy-wide 会被完全挤掉。回测里三个源是持续竞争同一批仓位的, 每个
    # 都实际贡献了交易, 所以这里按源配额取, 再合并按流动性排。
    buckets: dict[str, list[dict]] = {tid: [] for tid in STRATEGIES}
    hits: dict[str, dict] = {}
    for k, (cd, g) in enumerate(panel.items()):
        if on_progress:
            on_progress(k, len(panel))
        amt20 = g.amount.rolling(20).mean()
        liq = float(amt20.iloc[-1]) if not np.isnan(amt20.iloc[-1]) else 0.0
        if liq < MIN_AMOUNT_K:
            continue
        close = g.close.values
        # 涨停日买不进
        if len(close) > 1 and close[-1] > close[-2] * 1.099:
            continue

        for tid, inst in insts.items():
            try:
                sigs = inst.generate_signals(g)
            except Exception:             # noqa: BLE001
                continue
            buys = [s for s in sigs if s.get("action") == "buy"]
            if not buys:
                continue
            d = buys[-1]["date"]
            d = pd.Timestamp(d)
            if d < cutoff:
                continue
            if cd in hits:                # 多源触发: 保留更早的信号并记录来源
                hits[cd]["strategies"].append(STRATEGY_LABEL.get(tid, tid))
                if d < pd.Timestamp(hits[cd]["signal_date"]):
                    hits[cd]["signal_date"] = str(d.date())
                continue
            hits[cd] = dict(
                ts_code=cd, name=names.get(cd),
                signal_date=str(d.date()),
                latest_date=str(g.trade_date.iloc[-1].date()),
                latest_close=round(float(close[-1]), 3),
                amount_20d_wan=round(liq / 10, 1),      # 千元 -> 万元
                strategies=[STRATEGY_LABEL.get(tid, tid)],
                _src=tid,
            )
            buckets[tid].append(hits[cd])

    # 每源内部按流动性升序 (低流动性优先 —— 回测里这条决定了年化 25.8% vs 15~16%),
    # 然后轮流取, 直到凑满 top_n。信号少的源取完就不再占额度。
    #
    # ⚠️ 不要把"多源触发"当排序键: 回测里三个源平等竞争仓位, 若让双源优先,
    # 出来的 20 只会被 chan-2buy+kdj 垄断 (实测 20/20)。多源命中仍记在
    # strategies 里作为参考强度, 但不影响入选。
    for tid in buckets:
        buckets[tid].sort(key=lambda h: h["amount_20d_wan"])
    out: list[dict] = []
    cursors = {tid: 0 for tid in STRATEGIES}
    while len(out) < top_n:
        progressed = False
        for tid in STRATEGIES:
            i = cursors[tid]
            if i >= len(buckets[tid]):
                continue
            cand = buckets[tid][i]
            cursors[tid] = i + 1
            progressed = True
            if cand not in out:
                out.append(cand)
                if len(out) >= top_n:
                    break
        if not progressed:
            break

    out.sort(key=lambda h: h["amount_20d_wan"])
    for i, h in enumerate(out, 1):
        h["rank"] = i
        h["strategies"] = "+".join(dict.fromkeys(h["strategies"]))
        h.pop("_src", None)
    return out


async def portfolio_scan(
    db: AsyncSession, lookback_days: int = 3, top_n: int = 20, on_progress=None
) -> dict:
    t0 = time.time()
    end = (await db.execute(
        select(DailyCandle.trade_date).order_by(DailyCandle.trade_date.desc()).limit(1)
    )).scalar_one_or_none()
    if not end:
        return {"as_of": None, "picks": [], "scanned": 0, "elapsed_sec": 0.0}
    panel, names = await load_panel(db, end)
    picks = scan_panel(panel, names, lookback_days, top_n, on_progress)
    return {
        "as_of": str(end),
        "picks": picks,
        "scanned": len(panel),
        "elapsed_sec": round(time.time() - t0, 1),
    }
