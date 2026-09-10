"""组合信号扫描 — 把回测验证过的三源组合落地成每日可执行的买入清单.

配置来自全 universe / 2015-2026 / 复利 + 冲击成本的回测:
    (原为 chan-1buy-wide + chan-2buy + tdx-dual-kdj, 均已因未来函数下架)
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

import asyncio
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
# 原来的三个策略全部下架: chan-* 系 ZigZag 重绘未来函数,
# tdx-dual-kdj 周内多周期穿越。等新策略选定后再填。
STRATEGIES: list[str] = []
STRATEGY_LABEL = {
    "chan-1buy-wide": "缠论1买(宽)",
    "chan-2buy": "缠论2买",
}

MIN_AMOUNT_K = 5000      # 20 日均额下限, 千元 (= 500 万)
HISTORY_DAYS = 400       # 缠论需要足够的历史才能识别笔和背驰
MIN_BARS = 130

# ⚠️ 分批扫描, 不要改回"一次性拉全市场"。
# 曾经的写法是单条 SQL 拉全部 ~5500 只 × 400 天 ≈ 140 万行再 groupby, 本地
# 24G 内存上 21 秒跑完, 但当时生产机只有 1.6G —— 直接把机器压到 SSH 都连不上。
# ⚠️ 生产机后来升到 8G(见 docker-compose.prod.yml 的 shared_buffers 注释),
#    但【分批仍然要保留】: 省下来的内存现在给 postgres 的 1G shared_buffers
#    和模型推理用(裸K CNN 单次全市场推理峰值约 743MB)。
# 现在按票分块: 取一批 -> 算完 -> 释放, 峰值内存只跟 CHUNK 有关, 与全市场
# 规模无关。代价是慢一些(生产约 1~2 分钟), 换来的是内存可预测。
CHUNK = 300              # 每批股票数
MAX_CONCURRENT = 1       # 同时只允许一个扫描任务


def _universe_filter(ts_code: str, name: str | None) -> bool:
    if ts_code.startswith(("688", "92", "8", "4")):
        return False
    if name and ("ST" in name or "退" in name):
        return False
    return True


async def _list_universe(db: AsyncSession) -> tuple[list[str], dict[str, str]]:
    """先只取代码和名称 —— 几千行, 内存可忽略。"""
    basics = (await db.execute(
        select(StockBasic.ts_code, StockBasic.name, StockBasic.is_active)
    )).all()
    names = {c: n for c, n, _ in basics}
    codes = [c for c, n, a in basics
             if a is not False and _universe_filter(c, n)]
    return sorted(codes), names


async def _load_chunk(db: AsyncSession, codes: list[str], end: date) -> dict[str, pd.DataFrame]:
    """取一批票的近 HISTORY_DAYS 日线并做前复权。"""
    start = end - timedelta(days=HISTORY_DAYS)
    rows = (await db.execute(
        select(DailyCandle.ts_code, DailyCandle.trade_date, DailyCandle.open,
               DailyCandle.high, DailyCandle.low, DailyCandle.close,
               DailyCandle.vol, DailyCandle.amount, DailyCandle.adj_factor)
        .where(DailyCandle.ts_code.in_(codes),
               DailyCandle.trade_date >= start,
               DailyCandle.trade_date <= end)
    )).all()
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["ts_code", "trade_date", "open", "high", "low",
                                     "close", "vol", "amount", "adj_factor"])
    for c in ("open", "high", "low", "close", "amount", "adj_factor", "vol"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    out: dict[str, pd.DataFrame] = {}
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
        out[cd] = g[["trade_date", "open", "high", "low", "close", "vol", "amount"]]
    return out


def build_instances() -> dict:
    insts = {}
    for tid in STRATEGIES:
        cls = TEMPLATE_REGISTRY[tid]
        try:
            params = cls.parameter_candidates()[0]
            insts[tid] = cls(**params) if params else cls()
        except Exception as exc:          # noqa: BLE001
            log.warning("组合扫描: 策略 %s 无法实例化 — %s", tid, exc)
    return insts


def scan_chunk(
    panel: dict[str, pd.DataFrame],
    names: dict[str, str],
    insts: dict,
    cutoff: pd.Timestamp,
    buckets: dict[str, list[dict]],
    hits: dict[str, dict],
) -> None:
    """扫一批票, 命中累积进 buckets / hits (跨批共享)。"""
    for cd, g in panel.items():
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



def rank_picks(buckets: dict[str, list[dict]], top_n: int) -> list[dict]:
    """把各源桶合成最终清单。

    每源内部按流动性升序 (低流动性优先 —— 回测里这条决定了年化 25.8% vs 15~16%),
    然后轮流取, 直到凑满 top_n。信号少的源取完就不再占额度。

    ⚠️ 不要把"多源触发"当排序键: 回测里三个源平等竞争仓位, 若让双源优先,
    出来的 20 只会被 chan-2buy+kdj 垄断 (实测 20/20)。多源命中仍记在
    strategies 里作为参考强度, 但不影响入选。

    ⚠️ 也不要纯按流动性全局排: 三个源信号密度差 15 倍 (近5日 48 / 696 / 600),
    稀疏的 chan-1buy-wide 会被完全挤掉。
    """
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




# 同时只允许一个扫描 —— 两个并发扫描会让峰值内存翻倍。机器已升到 8G, 但
# 现在要和 postgres(1G) + CNN 推理(743MB) 分这 8G, 并发扫描依然不划算
_scan_lock = asyncio.Semaphore(MAX_CONCURRENT)


async def portfolio_scan(
    db: AsyncSession, lookback_days: int = 3, top_n: int = 20, on_progress=None
) -> dict:
    """分批扫描全市场。峰值内存只跟 CHUNK 有关, 与市场规模无关。"""
    if _scan_lock.locked():
        raise RuntimeError("已有扫描在运行 —— 同时只允许一个 (内存保护)")

    async with _scan_lock:
        t0 = time.time()
        end = (await db.execute(
            select(DailyCandle.trade_date).order_by(DailyCandle.trade_date.desc()).limit(1)
        )).scalar_one_or_none()
        if not end:
            return {"as_of": None, "picks": [], "scanned": 0, "elapsed_sec": 0.0}

        codes, names = await _list_universe(db)
        cutoff = pd.Timestamp(end) - pd.Timedelta(days=lookback_days)
        insts = build_instances()
        buckets: dict[str, list[dict]] = {tid: [] for tid in STRATEGIES}
        hits: dict[str, dict] = {}
        scanned = 0

        # 流水线: 取数和计算各占约一半耗时(实测 300 只 4.3s / 4.2s), 且都随票数
        # 线性增长 —— 调大批次没用, 但可以重叠: 一边算当前批, 一边预取下一批。
        # scan_chunk 是 CPU 密集的同步代码, 丢进线程池才不会阻塞取数的 await。
        loop = asyncio.get_running_loop()
        chunks = [codes[i:i + CHUNK] for i in range(0, len(codes), CHUNK)]
        panel = await _load_chunk(db, chunks[0], end) if chunks else {}
        for idx in range(len(chunks)):
            nxt = (
                asyncio.create_task(_load_chunk(db, chunks[idx + 1], end))
                if idx + 1 < len(chunks) else None
            )
            await loop.run_in_executor(
                None, scan_chunk, panel, names, insts, cutoff, buckets, hits
            )
            scanned += len(panel)
            panel.clear()               # 尽早释放, 让 GC 回收这一批
            if on_progress:
                on_progress(min((idx + 1) * CHUNK, len(codes)), len(codes))
            panel = await nxt if nxt is not None else {}

        picks = rank_picks(buckets, top_n)
        return {
            "as_of": str(end),
            "picks": picks,
            "scanned": scanned,
            "elapsed_sec": round(time.time() - t0, 1),
        }
