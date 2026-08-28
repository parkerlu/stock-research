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
from datetime import date, datetime

import httpx
import pandas as pd

from app.datasources.base import DataProvider

logger = logging.getLogger(__name__)

QT_URL = "https://qt.gtimg.cn/q={codes}"
MINUTE_URL = "https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={code}"
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

    # ---------- 历史数据不支持 — 交给 manager 回落 ----------

    async def fetch_daily(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def fetch_stock_basic(self) -> pd.DataFrame:
        return pd.DataFrame()
