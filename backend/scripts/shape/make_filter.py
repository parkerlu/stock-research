"""用 v5 模型产出【排除名单】—— 每日打分最差的 20%。

⚠️ 这个模型做不了选股策略(带成交约束后组合比值只有 0.24), 但它的空头端
   六年一致: Bot20% 超额 -0.66pp, 逐年全负。所以拿它当【过滤器】用 ——
   不是"买它选的", 而是"别买它嫌弃的"。

输出: bottom20_v5.parquet (ts_code, trade_date), 供各策略叠加排除。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("filter")
DIR = "/app/data/research/shape"
MODEL = f"{DIR}/shape_v5_y_0.15_0.08_h20.json"
META = ("ts_code", "trade_date", "c")


def main() -> None:
    # 还原 cid -> ts_code。train_v5 用 pd.Categorical, 类别是排序后的唯一值。
    codes = pq.read_table(f"{DIR}/features_v2.parquet",
                          columns=["ts_code"]).to_pandas()["ts_code"]
    code_index = pd.Index(pd.Categorical(codes).categories)
    del codes
    cols = [c for c in pq.ParquetFile(f"{DIR}/features_v2.parquet").schema_arrow.names
            if c not in META]

    m = xgb.XGBRegressor()
    m.load_model(MODEL)

    outs = []
    for y in range(2016, 2027):
        p = f"{DIR}/_yr/{y}.parquet"
        try:
            d = pd.read_parquet(p)
        except FileNotFoundError:
            continue
        sc = np.empty(len(d), np.float32)
        for i in range(0, len(d), 500_000):
            sc[i:i + 500_000] = m.predict(d[cols].iloc[i:i + 500_000]).astype(np.float32)
        d = d[["day", "cid"]].assign(score=sc)
        d["rk"] = d.groupby("day")["score"].rank(pct=True)
        bad = d[d.rk < 0.20][["day", "cid"]]
        outs.append(bad)
        log.info("  %d: %s 行中最差 %s 只", y, f"{len(d):,}", f"{len(bad):,}")
        del d, sc

    out = pd.concat(outs, ignore_index=True)
    out["ts_code"] = code_index[out["cid"].to_numpy()]
    out["trade_date"] = pd.to_datetime(out["day"], unit="D").dt.date
    out[["ts_code", "trade_date"]].to_parquet(f"{DIR}/bottom20_v5.parquet", index=False)
    log.info("排除名单 %s 行 -> bottom20_v5.parquet", f"{len(out):,}")


if __name__ == "__main__":
    main()
