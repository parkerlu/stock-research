"""用训好的裸K模型给【全部样本外行】打分, 产出 Top1% 信号供组合回测。

⚠️ 必须给全部行打分, 不能用评估时的抽样子集: rank 是【当日横截面】概念,
   在子集上排名和在全市场排名不是一回事。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from scripts.rawk.train import DEV, NBAR, Net, make_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rawk.sig")
DIR = "/app/data/research/rawk"


def main() -> None:
    d = np.load(f"{DIR}/panel.npz")
    ohlcv, idx, y = d["ohlcv"], d["idx"], d["y"]
    lday, lcid = d["lab_day"], d["lab_cid"]
    day_all, mkt, day_min = d["day"], d["mkt"], int(d["day_min"])
    n_ch = 6 + mkt.shape[0] * 4

    net = Net(ch=n_ch).to(DEV)
    net.load_state_dict(torch.load(f"{DIR}/cnn.pt", map_location=DEV))
    net.eval()

    yr = pd.to_datetime(pd.Series(lday), unit="D").dt.year.to_numpy()
    te = np.flatnonzero(yr >= 2021)
    log.info("给 %s 行样本外打分 ...", f"{len(te):,}")

    sc = np.empty(len(te), np.float32)
    bs = 4096
    with torch.no_grad():
        for i in range(0, len(te), bs):
            b = te[i:i + bs]
            x = make_batch(ohlcv, idx[b], day=day_all, mkt=mkt, day_min=day_min)
            sc[i:i + bs] = net(torch.as_tensor(x, device=DEV)).cpu().numpy()
            if (i // bs) % 200 == 0:
                log.info("  %s / %s", f"{i:,}", f"{len(te):,}")

    codes = pq.read_table("/app/data/research/shape/features_v2.parquet",
                          columns=["ts_code"]).to_pandas()["ts_code"]
    code_index = pd.Index(pd.Categorical(codes).categories)
    del codes

    df = pd.DataFrame({"day": lday[te], "cid": lcid[te], "score": sc})
    df["rk"] = df.groupby("day")["score"].rank(pct=True)
    top = df[df.rk >= 0.99].copy()
    top["ts_code"] = code_index[top["cid"].to_numpy()]
    top["trade_date"] = pd.to_datetime(top["day"], unit="D")
    top[["ts_code", "trade_date", "score", "rk"]].to_parquet(
        f"{DIR}/oos_rawk.parquet", index=False)
    log.info("Top1%% %s 条 -> oos_rawk.parquet", f"{len(top):,}")


if __name__ == "__main__":
    main()
