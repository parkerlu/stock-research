"""腾讯行情数据源 — 实时快照 / 批量快照 / 分时数据.

只提供实时类数据; `fetch_daily` 与 `fetch_stock_basic` 返回空, 由
DataSourceManager 自动回落到 TuShare (历史数据仍以 TuShare 为准)。

两个上游接口 (都必须 HTTPS, HTTP 会 302):
  1. https://qt.gtimg.cn/q=sh603319,sz000001
     GBK 编码, 每行 v_<code>="字段1~字段2~...";  支持逗号批量, 一次几十只
  2. https://web.ifzq.gtimg.cn/appstock/app/minute/query?code=sh603319
     JSON, 一次返回 分时明细 + 实时报价 + 各市场开闭市状态

⚠️ 腾讯不是官方数据源, 字段位置靠约定。字段索引集中在 _QT 常量里, 若上游
调整格式只需改这一处。已用 603319.SH / 000001.SZ 实测校验 (2026-08-28)。
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, timedelta, datetime

import json

import httpx
import pandas as pd

from app.datasources.base import DataProvider

logger = logging.getLogger(__name__)

QT_URL = "https://qt.gtimg.cn/q={codes}"
MINUTE_URL = "https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={code}"
# 历史K线。fq 传 "qfq" 取前复权, 空串取不复权。
#
# ⚠️ count 实测上限约 800: 传更大的值接口反而退回默认 640 行, 所以要分段请求。
# ⚠️ 这个接口一次只能查一只票, 且有未公开的 IP 限流 —— 实测 8 路并发跑约
#    2400 次请求后被 TLS 层直接断连(不是错误码, 是 SSL EOF), 恢复用了约 3 小时。
#    单票按需取数用它很好(11.6 年 2833 行只要 1.6 秒, 前复权直出);
#    全市场批量回补请走 TuShare 的 pro.daily(trade_date=...) 按交易日取,
#    2833 次调用即可覆盖全市场, 配额明确可控。
#    若必须用本接口批量, 建议串行 + 1 秒间隔, 不要并发。
KLINE_URL = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
             "?param={code},day,{start},{end},{count},{fq}")
_KLINE_MAX = 700          # 单次请求的安全行数
_KLINE_CHUNK_DAYS = 1000  # 约 700 个交易日
_TIMEOUT = 8.0

# qt 数组字段索引 (0-based)。实测 603319.SH:
#   [3]=26.52 现价  [4]=29.45 昨收  [5]=29.00 今开  [6]=309315 成交量(手)
#   [31]=-2.93 涨跌  [32]=-9.95 涨跌%  [33]=29.07 最高  [34]=26.51 最低
#   [37]=83626 成交额(万)  [38]=9.12 换手%  [47]=32.40 涨停  [48]=26.51 跌停
_QT = {
    "name": 1, "code": 2, "price": 3, "prev_close": 4, "open": 5, "vol": 6,
    "time": 30, "change": 31, "change_pct": 32, "high": 33, "low": 34,
    "detail": 35, "amount_wan": 37, "turnover": 38, "pe": 39,
    "amplitude": 43, "float_cap": 44, "total_cap": 45, "pb": 46,
    "high_limit": 47, "low_limit": 48,
}


def to_tencent_code(ts_code: str) -> str:
    """603319.SH -> sh603319;  000001.SZ -> sz000001。已是腾讯格式则原样返回。"""
    ts_code = ts_code.strip()
    if re.fullmatch(r"(sh|sz|bj)\d{6}", ts_code, re.I):
        return ts_code.lower()
    if "." in ts_code:
        num, mkt = ts_code.split(".", 1)
        return f"{mkt.strip().lower()}{num.strip()}"
    # 裸 6 位数字 — 按代码段推断交易所
    if re.fullmatch(r"\d{6}", ts_code):
        if ts_code.startswith(("60", "68", "51", "58", "11")):
            return f"sh{ts_code}"
        if ts_code.startswith(("4", "8", "92")):
            return f"bj{ts_code}"
        return f"sz{ts_code}"
    return ts_code.lower()


def to_ts_code(tencent_code: str) -> str:
    """sh603319 -> 603319.SH"""
    m = re.fullmatch(r"(sh|sz|bj)(\d{6})", tencent_code, re.I)
    return f"{m.group(2)}.{m.group(1).upper()}" if m else tencent_code


def _f(parts: list[str], key: str, default: float = 0.0) -> float:
    try:
        v = parts[_QT[key]].strip()
        return float(v) if v else default
    except (IndexError, ValueError, KeyError):
        return default


def _parse_qt_line(parts: list[str], tencent_code: str) -> dict | None:
    """把 qt 字段数组转成 snapshot dict。停牌(现价=0)时返回昨收以免前端显示 0。

    `tencent_code` (如 sh603319) 必传 —— 数组里的 [2] 只有 6 位数字、没有交易所,
    单靠它拼不出 ts_code, 会导致调用方按 "603319.SH" 查不到而误回落到其他源。
    """
    if len(parts) < 49:
        return None
    price = _f(parts, "price")
    prev_close = _f(parts, "prev_close")
    if price <= 0:                      # 停牌 / 未开盘
        price = prev_close

    # 成交额: 优先用 detail 字段的精确值 "现价/成交量/成交额", 否则用 万元 换算
    amount = 0.0
    try:
        seg = parts[_QT["detail"]].split("/")
        if len(seg) >= 3 and seg[2]:
            amount = float(seg[2])
    except (IndexError, ValueError):
        pass
    if amount <= 0:
        amount = _f(parts, "amount_wan") * 10_000

    # 五档盘口。买一~买五在 [9..18], 卖一~卖五在 [19..28], 均为 价,量 交替。
    # 实测 000001.SZ: 买 11.61/11.60/11.59/11.58/11.57 (递减),
    #                卖 11.62/11.63/11.64/11.65/11.66 (递增) — 索引正确。
    def _levels(base: int) -> list[dict]:
        out = []
        for i in range(5):
            try:
                p = float(parts[base + i * 2] or 0)
                v = float(parts[base + i * 2 + 1] or 0)
            except (IndexError, ValueError):
                p = v = 0.0
            out.append({"price": p, "vol": int(v)})   # vol 单位: 手
        return out

    bids = _levels(9)    # 买一..买五
    asks = _levels(19)   # 卖一..卖五

    ts_raw = parts[_QT["time"]].strip()
    try:
        quote_time = datetime.strptime(ts_raw, "%Y%m%d%H%M%S").isoformat()
    except ValueError:
        quote_time = None

    return {
        # 与 akshare/tushare 版 snapshot 保持同名同单位, 前端无需改动
        "symbol": to_ts_code(tencent_code),
        "name": parts[_QT["name"]].strip(),
        "price": price,
        "change": _f(parts, "change"),
        "change_pct": _f(parts, "change_pct"),
        "open": _f(parts, "open"),
        "high": _f(parts, "high"),
        "low": _f(parts, "low"),
        "vol": int(_f(parts, "vol")),          # 手
        "amount": amount,                       # 元
        "turnover": _f(parts, "turnover"),      # %
        # 腾讯额外字段 (增量, 不影响既有调用方)
        "prev_close": prev_close,
        "high_limit": _f(parts, "high_limit"),
        "low_limit": _f(parts, "low_limit"),
        "amplitude": _f(parts, "amplitude"),
        "pe": _f(parts, "pe"),
        "pb": _f(parts, "pb"),
        "total_cap": _f(parts, "total_cap"),    # 亿元
        "float_cap": _f(parts, "float_cap"),    # 亿元
        "quote_time": quote_time,
        "bids": bids,   # 买一..买五 [{price, vol}]
        "asks": asks,   # 卖一..卖五 [{price, vol}]
        "source": "tencent",
    }


class TencentProvider(DataProvider):
    """实时行情提供者。历史数据不支持 — 由 manager 回落到 TuShare。"""

    async def _get_text(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as c:
            r = await c.get(url, headers={"Referer": "https://finance.qq.com/"})
            r.raise_for_status()
            # qt 接口是 GBK
            return r.content.decode("gbk", errors="replace")

    # ---------- 实时快照 ----------

    async def fetch_snapshots(self, ts_codes: list[str]) -> dict[str, dict]:
        """批量快照。腾讯单次可查几十只, 这里按 60 只分批并发。

        返回 {ts_code: snapshot}; 查不到的 code 直接缺席。
        """
        if not ts_codes:
            return {}
        chunks = [ts_codes[i:i + 60] for i in range(0, len(ts_codes), 60)]

        async def one(chunk: list[str]) -> dict[str, dict]:
            codes = ",".join(to_tencent_code(c) for c in chunk)
            try:
                text = await self._get_text(QT_URL.format(codes=codes))
            except Exception:
                logger.warning("tencent batch quote failed", exc_info=True)
                return {}
            out: dict[str, dict] = {}
            for line in text.splitlines():
                m = re.match(r'v_([a-z]{2}\d{6})="(.*)";?', line.strip())
                if not m:
                    continue
                snap = _parse_qt_line(m.group(2).split("~"), m.group(1))
                if snap:
                    out[snap["symbol"]] = snap
            return out

        merged: dict[str, dict] = {}
        for part in await asyncio.gather(*(one(c) for c in chunks)):
            merged.update(part)
        return merged

    async def fetch_snapshot(self, ts_code: str) -> dict | None:
        got = await self.fetch_snapshots([ts_code])
        return got.get(to_ts_code(to_tencent_code(ts_code)))

    # ---------- 分时 ----------

    async def fetch_minute(self, ts_code: str) -> dict | None:
        """当日分时。返回 {trade_date, prev_close, market_open, bars:[...]}。

        bars 每项: {time "0930", price, avg_price 均价, vol 当分钟手数, amount 当分钟成交额}
        上游给的是累计量, 这里差分成每分钟量 (分时图下方的量柱要的是分钟量)。
        """
        code = to_tencent_code(ts_code)
        url = MINUTE_URL.format(code=code)
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as c:
                r = await c.get(url, headers={"Referer": "https://finance.qq.com/"})
                r.raise_for_status()
                payload = r.json()
        except Exception:
            logger.warning("tencent minute fetch failed for %s", ts_code, exc_info=True)
            return None

        node = (payload.get("data") or {}).get(code)
        if not node:
            return None
        inner = node.get("data") or {}
        raw_bars = inner.get("data") or []

        qt_arr = (node.get("qt") or {}).get(code)
        snap = _parse_qt_line(qt_arr, code) if qt_arr else None
        prev_close = snap["prev_close"] if snap else 0.0

        # 上证是否在交易 — 直接用上游的市场状态, 不硬编码交易时段
        market_open = False
        try:
            market_line = (node.get("qt") or {}).get("market", [""])[0]
            m = re.search(r"SH_(open|close)_([^|]*)", market_line)
            if m:
                market_open = m.group(1) == "open"
        except Exception:
            pass

        bars = []
        prev_vol = prev_amt = 0.0
        for row in raw_bars:
            seg = row.split()
            if len(seg) < 3:
                continue
            hhmm, price = seg[0], float(seg[1])
            cum_vol = float(seg[2])
            cum_amt = float(seg[3]) if len(seg) > 3 else 0.0
            # 均价 = 累计成交额 / (累计手数 × 100)
            avg = (cum_amt / (cum_vol * 100)) if cum_vol > 0 and cum_amt > 0 else price
            bars.append({
                "time": hhmm,
                "price": price,
                "avg_price": round(avg, 3),
                "vol": max(cum_vol - prev_vol, 0),
                "amount": max(cum_amt - prev_amt, 0),
            })
            prev_vol, prev_amt = cum_vol, cum_amt

        return {
            "symbol": to_ts_code(code),
            "trade_date": inner.get("date"),
            "prev_close": prev_close,
            "market_open": market_open,
            "snapshot": snap,
            "bars": bars,
        }

    # ---------- 历史日线 ----------

    async def _kline_chunk(self, code: str, s: date, e: date, fq: str) -> list[list]:
        url = KLINE_URL.format(code=code, start=s.isoformat(), end=e.isoformat(),
                               count=_KLINE_MAX, fq=fq)
        try:
            txt = await self._get_text(url)
            node = json.loads(txt).get("data", {}).get(code)
        except Exception as exc:
            logger.warning("tencent kline %s %s~%s failed: %s", code, s, e, exc)
            return []
        if not isinstance(node, dict):
            return []
        # 前复权在 qfqday, 不复权在 day
        rows = node.get("qfqday") or node.get("day") or []
        return [r for r in rows if isinstance(r, list) and len(r) >= 6]

    async def fetch_daily(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        """前复权日线。腾讯直接给复权价, 不必再合并 adj_factor。

        腾讯 K 线只有成交量没有成交额, 而成交额是流动性排序的依据 —— 这里额外
        取一份不复权价, 用 vol(手) × 100 × 不复权收盘 / 1000 折算出千元成交额
        (与 TuShare 口径实测差 ~0.1%)。
        """
        code = to_tencent_code(ts_code)
        if not code:
            return pd.DataFrame()

        # 分段: 单次最多 ~700 行
        spans: list[tuple[date, date]] = []
        cur = start
        while cur <= end:
            nxt = min(cur + timedelta(days=_KLINE_CHUNK_DAYS), end)
            spans.append((cur, nxt))
            cur = nxt + timedelta(days=1)

        qfq_rows: list[list] = []
        raw_rows: list[list] = []
        for s_, e_ in spans:
            qfq_rows += await self._kline_chunk(code, s_, e_, "qfq")
            raw_rows += await self._kline_chunk(code, s_, e_, "")
        if not qfq_rows:
            return pd.DataFrame()

        # 行格式: [日期, 开, 收, 高, 低, 量(手), ...]
        def to_df(rows: list[list], cols: list[str]) -> pd.DataFrame:
            df = pd.DataFrame([r[:6] for r in rows], columns=cols)
            df = df.drop_duplicates(subset=["trade_date"])
            for c in cols[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            return df.dropna(subset=["open", "high", "low", "close"])

        cols = ["trade_date", "open", "close", "high", "low", "vol"]
        df = to_df(qfq_rows, cols)
        if raw_rows:
            raw = to_df(raw_rows, cols)[["trade_date", "close"]].rename(
                columns={"close": "raw_close"})
            df = df.merge(raw, on="trade_date", how="left")
        else:
            df["raw_close"] = df["close"]
        df["raw_close"] = df["raw_close"].fillna(df["close"])
        # 成交额(千元) = 手 × 100股 × 价 / 1000
        df["amount"] = (df["vol"] * 100 * df["raw_close"] / 1000).round(3)
        df["ts_code"] = ts_code
        df["adj_factor"] = 1.0        # 已是前复权价, 下游无需再调整
        df = df[(df.trade_date >= start) & (df.trade_date <= end)]
        return df.sort_values("trade_date")[
            ["ts_code", "trade_date", "open", "high", "low", "close",
             "vol", "amount", "adj_factor"]
        ].reset_index(drop=True)

    async def fetch_stock_basic(self) -> pd.DataFrame:
        return pd.DataFrame()
