"""重训做 T 模型 (t0_up / t0_dn)。

模型权重按仓库惯例不进 git(见 .gitignore), 这个脚本是它们唯一的还原路径。

用法:
    python -m scripts.train_t0 --fetch          # 拉 5min 数据(慢, 断点续传)
    python -m scripts.train_t0 --train          # 训练并写出模型
    python -m scripts.train_t0 --fetch --train  # 一条龙
    python -m scripts.train_t0 --train --smoke  # 只用 5 只票跑通流程

⚠️ 特征一律复用 app.services.t0_indicator.build_features, 不在这里重写。
   训练与推理的特征口径必须逐字一致 —— 不一致时模型不会报错, 只会静默给出
   错误概率。本项目已经因为"为性能重写已有实现"栽过一次(买卖很准 v3, 648 天里
   59 天对不上)。

数据落在 /app/data/research/ —— 那是挂载卷。不要写 /app/fable, 容器重建即丢失
(做T的 5min 原始数据就是这么没的, 这个脚本本身就是那次教训的产物)。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.t0_indicator import (
    BARS_PER_DAY, FEATURES, TARGET, build_features, is_supported, limit_of,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_t0")

RESEARCH = Path("/app/data/research")
BAR_PARQUET = RESEARCH / "t0_bars.parquet"
# ⚠️ /app/models 在容器里是只读挂载(docker-compose: ./backend/models:/app/models:ro),
# 所以默认写到可写的 research 目录, 再由使用者复制到宿主机 backend/models/。
MODEL_OUT = RESEARCH / "models_out"

TRAIN_END = "2023-12-31"      # 训练集截止; 之后全部留作样本外
START = "2020-01-01"          # baostock 5min 免费数据从 2020 起
UNIVERSE_N = 307              # 与原模型一致
SMOKE_N = 5


# ---------------------------------------------------------------- 选股票池
async def pick_universe(n: int) -> list[str]:
    """训练票池: 沪深主板/创业板/科创板, 剔除 ST, 按训练期日均成交额取前 n。

    按成交额而不是随机取 —— 做 T 需要流动性, 冷门票的 5min bar 大量缺失,
    特征里的量比、VWAP 全是噪声。
    """
    from datetime import date

    from sqlalchemy import text

    from app.db import engine

    async with engine.connect() as c:
        rows = (await c.execute(text("""
            select d.ts_code, avg(d.close * d.vol) amt
            from daily_candle d
            where d.trade_date between :s and :e
            group by d.ts_code
            having count(*) > 600
            order by amt desc
        """), {"s": date.fromisoformat(START), "e": date.fromisoformat(TRAIN_END)})).fetchall()
        # 现名带 ST 的一律排除(与训练时同口径; 用现名而非历史名, 从严)
        st = {r[0] for r in (await c.execute(text(
            "select ts_code from stock_basic where name like '%ST%'"))).fetchall()}

    out = [r[0] for r in rows if is_supported(r[0]) and r[0] not in st]
    log.info("票池: %d 只(候选 %d, 剔除 ST %d)", min(n, len(out)), len(rows), len(st))
    return out[:n]


# ---------------------------------------------------------------- 拉 5min
def fetch_bars(codes: list[str], end: str) -> None:
    """从 baostock 拉 5min 前复权数据, 逐票 append 落盘(断点续传)。"""
    import baostock as bs

    done: set[str] = set()
    if BAR_PARQUET.exists():
        done = set(pd.read_parquet(BAR_PARQUET, columns=["ts_code"])["ts_code"].unique())
        log.info("已有 %d 只, 续传", len(done))

    todo = [c for c in codes if c not in done]
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    try:
        buf: list[pd.DataFrame] = []
        for i, ts in enumerate(todo, 1):
            code, _, mkt = ts.partition(".")
            bs_code = f"{mkt.lower()}.{code}"
            rs = bs.query_history_k_data_plus(
                bs_code, "time,close,volume", start_date=START, end_date=end,
                frequency="5", adjustflag="2",      # 2 = 前复权
            )
            recs = []
            while rs.error_code == "0" and rs.next():
                recs.append(rs.get_row_data())
            if not recs:
                log.warning("  %s 无数据", ts)
                continue
            d = pd.DataFrame(recs, columns=["time", "close", "volume"])
            # baostock time: "20200102093500000" → 日期 + HHMM(bar 结束时刻)
            d["trade_date"] = d["time"].str[:8]
            d["hhmm"] = d["time"].str[8:12]
            d["close"] = pd.to_numeric(d["close"], errors="coerce")
            d["volume"] = pd.to_numeric(d["volume"], errors="coerce")   # 股
            d["ts_code"] = ts
            buf.append(d[["ts_code", "trade_date", "hhmm", "close", "volume"]])
            if i % 20 == 0 or i == len(todo):
                _append(pd.concat(buf, ignore_index=True))
                buf.clear()
                log.info("  %d/%d", i, len(todo))
    finally:
        bs.logout()


def _append(df: pd.DataFrame) -> None:
    if BAR_PARQUET.exists():
        df = pd.concat([pd.read_parquet(BAR_PARQUET), df], ignore_index=True)
    BAR_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BAR_PARQUET, index=False)


# ---------------------------------------------------------------- 造样本
def _day_matrix(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """当日 5min 收盘/成交量 → 定长 48 槽(缺的留 NaN, 与推理侧一致)。"""
    from app.services.t0_indicator import BAR_TIME

    slot = {t: i for i, t in enumerate(BAR_TIME)}
    close = np.full(BARS_PER_DAY, np.nan)
    vol = np.zeros(BARS_PER_DAY)
    for hhmm, c, v in zip(g["hhmm"], g["close"], g["volume"]):
        i = slot.get(hhmm)
        if i is not None:
            close[i], vol[i] = c, v
    return close, vol


def build_dataset(bars: pd.DataFrame) -> pd.DataFrame:
    """逐(票, 日)造特征与标签。

    标签 = 从当前时点到收盘, 收盘价序列是否触及 ±3%:
      up (买/低吸): max(close[t+1:]) / close[t] - 1 >=  0.03
      dn (卖/高抛): min(close[t+1:]) / close[t] - 1 <= -0.03
    与 compute_signals 里的成交判定(goal = entry × (1±TARGET))完全对齐。
    """
    out = []
    for ts, gs in bars.groupby("ts_code", sort=False):
        lim = limit_of(ts)
        gs = gs.sort_values(["trade_date", "hhmm"])
        prev = None    # (close_arr, vol_arr) of previous day
        for day, g in gs.groupby("trade_date", sort=True):
            close, vol = _day_matrix(g)
            if prev is not None and np.isfinite(close).sum() >= 20:
                pc, pv = prev
                real_p = np.isfinite(pc)
                if real_p.any():
                    prev_close = float(pc[real_p][-1])
                    prev_open = float(pc[real_p][0])
                    prev_amp = float((np.nanmax(pc) - np.nanmin(pc)) / prev_open) if prev_open > 0 else 0.0
                    prev_ret = float(prev_close / prev_open - 1) if prev_open > 0 else 0.0
                    prev_vol_shares = float(np.nansum(pv))
                    real = np.isfinite(close)
                    open_px = float(close[real][0])
                    X, idx = build_features(
                        close, vol, open_px, prev_close,
                        prev_amp, prev_ret, prev_vol_shares, lim,
                    )
                    if len(idx):
                        last_real = int(np.where(real)[0][-1])
                        y_up, y_dn = [], []
                        for t in idx:
                            fut = close[t + 1: last_real + 1]
                            fut = fut[np.isfinite(fut)]
                            c = close[t]
                            if fut.size == 0:
                                y_up.append(0); y_dn.append(0); continue
                            y_up.append(int(fut.max() / c - 1 >= TARGET))
                            y_dn.append(int(fut.min() / c - 1 <= -TARGET))
                        df = pd.DataFrame(X, columns=FEATURES)
                        df["y_up"], df["y_dn"] = y_up, y_dn
                        df["trade_date"], df["ts_code"] = day, ts
                        out.append(df)
            prev = (close, vol)
    if not out:
        raise SystemExit("没有可用样本 —— 先跑 --fetch")
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 训练
def train(ds: pd.DataFrame, out: Path) -> None:
    import json

    import xgboost as xgb

    cut = TRAIN_END.replace("-", "")
    tr, te = ds[ds["trade_date"] <= cut], ds[ds["trade_date"] > cut]
    log.info("训练 %d 行 / 样本外 %d 行", len(tr), len(te))
    if tr.empty:
        raise SystemExit("训练集为空")

    out.mkdir(parents=True, exist_ok=True)
    meta_common = {
        "features": FEATURES,
        "label": "从当前时点到收盘是否还有3%空间",
        "threshold_recommended": 0.7,
        "bar": "5min",
        "trained": f"{START[:4]}-{TRAIN_END[:4]}",
        "oos": f"{int(TRAIN_END[:4]) + 1}-",
    }
    for side, col in (("up", "y_up"), ("dn", "y_dn")):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=42,
        )
        m.fit(tr[FEATURES], tr[col])
        m.save_model(str(out / f"t0_{side}.json"))
        (out / f"t0_{side}_meta.json").write_text(
            json.dumps(meta_common, ensure_ascii=False), encoding="utf-8")

        line = f"  t0_{side}: 正样本率 {tr[col].mean():.1%}"
        if not te.empty:
            p = m.predict_proba(te[FEATURES])[:, 1]
            hit = p >= 0.7
            prec = te[col][hit].mean() if hit.any() else float("nan")
            line += f" | 样本外 阈值0.7 命中 {hit.sum()} 次, 精确率 {prec:.1%}"
        log.info(line)
    log.info("模型已写入 %s", out)
    log.info("装载到线上(宿主机执行): docker cp stock-backend-1:%s/. backend/models/ && "
             "docker compose restart backend", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="只用 5 只票, 验证流程能跑通")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--out", default=str(MODEL_OUT), help="模型输出目录(默认可写的 research 下)")
    a = ap.parse_args()
    if not (a.fetch or a.train):
        ap.error("至少给一个 --fetch / --train")

    n = SMOKE_N if a.smoke else UNIVERSE_N
    if a.fetch:
        fetch_bars(asyncio.run(pick_universe(n)), a.end)
    if a.train:
        if not BAR_PARQUET.exists():
            raise SystemExit(f"{BAR_PARQUET} 不存在 —— 先跑 --fetch")
        bars = pd.read_parquet(BAR_PARQUET)
        if a.smoke:
            bars = bars[bars["ts_code"].isin(bars["ts_code"].unique()[:SMOKE_N])]
        log.info("5min 行数 %s, 票 %d", f"{len(bars):,}", bars["ts_code"].nunique())
        train(build_dataset(bars), Path(a.out))


if __name__ == "__main__":
    main()
