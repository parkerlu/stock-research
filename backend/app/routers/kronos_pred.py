"""Kronos K线预测 —— 点一根K线, 预测它之后的走势。

⚠️ 这是 Kronos 的【看家用法】(生成式预测), 与本项目把它当特征提取器的
   实验是两回事。那个实验的结论是: 拿它的表示做选股, 组合比值只有 0.08,
   打不过我们自己训的模型。但"预测未来K线并画出来"它是专门为此设计的。

⚠️ 必须跑在本地 CPU, 不能依赖临时 GPU 机器 —— 那是抢占式实例, 随时释放。
   实测 CPU 上预测 30 根仅 0.7 秒, 交互完全够用。

⚠️ 用【历史某一天】做起点时, 后面的真实K线我们是有的 ——
   前端把预测和真实画在一起, 才能一眼看出它准不准。
   这个功能的价值恰恰在于【可证伪】, 而不是拿它当水晶球。

⚠️ 模型是有随机性的(自回归采样)。sample_count>1 时会采样多条路径取均值,
   单条路径只能看个大概方向, 不要当成精确预测。
"""
from __future__ import annotations

import logging
import os
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from app.db import async_session

log = logging.getLogger("kronos_pred")
router = APIRouter(prefix="/api/kronos", tags=["kronos"])

# 模型只加载一次。⚠️ 每次请求都 from_pretrained 会重复读盘+建图, 单次 11 秒。
_MODEL: dict = {}


def _get(model_name: str):
    if model_name in _MODEL:
        return _MODEL[model_name]
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/app/data/research/hf")
    from app.services.kronos import Kronos, KronosPredictor, KronosTokenizer

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained(f"NeoQuasar/{model_name}")
    ctx = 2048 if model_name == "Kronos-mini" else 512
    _MODEL[model_name] = KronosPredictor(mdl, tok, device="cpu", max_context=ctx)
    log.info("Kronos %s 已加载 (上下文 %d)", model_name, ctx)
    return _MODEL[model_name]


@router.get("/predict/{ts_code}")
async def predict(
    ts_code: str,
    end: date = Query(..., description="以这一天为起点, 预测其之后的走势"),
    lookback: int = Query(400, ge=100, le=2000, description="喂给模型多少根历史"),
    pred_len: int = Query(30, ge=5, le=120, description="预测多少根"),
    samples: int = Query(1, ge=1, le=10, description="采样几条路径取均值"),
    model: str = Query("Kronos-mini", pattern="^Kronos-(mini|small)$"),
    temperature: float = Query(1.0, ge=0.1, le=2.0),
):
    """返回预测的K线, 以及(若有)同期的真实K线供对照。"""
    async with async_session() as db:
        rows = (await db.execute(text("""
            select trade_date, open, high, low, close, vol, amount, adj_factor
            from daily_candle
            where ts_code = :c and trade_date <= :e
            order by trade_date desc limit :n
        """), {"c": ts_code, "e": end, "n": lookback})).fetchall()
        if len(rows) < 100:
            raise HTTPException(404, f"{ts_code} 在 {end} 之前只有 {len(rows)} 根K线, 不足以预测")
        # 真实的后续走势 —— 历史起点时用来对照
        fut = (await db.execute(text("""
            select trade_date, open, high, low, close, adj_factor
            from daily_candle where ts_code = :c and trade_date > :e
            order by trade_date limit :n
        """), {"c": ts_code, "e": end, "n": pred_len})).fetchall()
        # ⚠️ 前复权基准取【该股全局最新】的 adj_factor, 与 quote_service 同口径,
        #    否则同一根K线在不同页面上价格不一样。
        latest_adj = (await db.execute(text(
            "select adj_factor from daily_candle where ts_code = :c "
            "order by trade_date desc limit 1"), {"c": ts_code})).scalar()

    la = float(latest_adj or 1.0) or 1.0
    hist = pd.DataFrame(rows[::-1], columns=["timestamps", "open", "high", "low",
                                             "close", "volume", "amount", "adj"])
    for c in ("open", "high", "low", "close", "volume", "amount", "adj"):
        hist[c] = pd.to_numeric(hist[c], errors="coerce").astype(float)
    hist["timestamps"] = pd.to_datetime(hist["timestamps"])
    # 喂给模型的是【后复权】(原价×factor), 保证序列连续无除权跳空
    for c in ("open", "high", "low", "close"):
        hist[c] = hist[c] * hist["adj"]

    predictor = _get(model)
    y_ts = pd.Series(pd.date_range(hist["timestamps"].iloc[-1],
                                   periods=pred_len + 1, freq="B")[1:])
    try:
        out = predictor.predict(
            df=hist[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=hist["timestamps"], y_timestamp=y_ts,
            pred_len=pred_len, T=temperature, top_p=0.9,
            sample_count=samples, verbose=False)
    except Exception as exc:  # noqa: BLE001
        log.exception("预测失败 %s", ts_code)
        raise HTTPException(500, f"预测失败: {exc}") from exc

    # 换回前复权口径, 与图表一致
    def _fw(v: float) -> float:
        return round(float(v) / la, 4)

    # 真实后续的日期优先用真实交易日; 没有(预测未来)就用生成的工作日
    real_dates = [r[0] for r in fut]
    dates = real_dates + [d.date() for d in y_ts[len(real_dates):]]

    pred_bars = [{
        "date": str(dates[i]) if i < len(dates) else str(y_ts.iloc[i].date()),
        "open": _fw(out["open"].iloc[i]), "high": _fw(out["high"].iloc[i]),
        "low": _fw(out["low"].iloc[i]), "close": _fw(out["close"].iloc[i]),
    } for i in range(len(out))]

    real_bars = [{
        "date": str(r[0]), "open": round(float(r[1]) * float(r[5]) / la, 4),
        "high": round(float(r[2]) * float(r[5]) / la, 4),
        "low": round(float(r[3]) * float(r[5]) / la, 4),
        "close": round(float(r[4]) * float(r[5]) / la, 4),
    } for r in fut]

    anchor = _fw(hist["close"].iloc[-1])
    return {
        "ts_code": ts_code, "model": model, "anchor_date": str(end),
        "anchor_close": anchor, "lookback": len(hist), "pred_len": pred_len,
        "samples": samples,
        "predicted": pred_bars,
        "actual": real_bars,          # 历史起点时非空, 用于对照
        "is_historical": len(real_bars) > 0,
    }
