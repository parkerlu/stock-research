"""打一个给 Kaggle/Colab 用的数据包 —— 只带必需的, 压缩后上传。

⚠️ 不带任何我方的特征工程, 只有原始 OHLCV + 标签 + 真实指数。
   目的是让 Kronos 在【和我们裸K模型完全相同的信息】下比, 否则比较不公平。
"""
from __future__ import annotations

import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pack")
DIR = "/app/data/research/rawk"


def main() -> None:
    d = np.load(f"{DIR}/panel.npz")
    idx, y = d["idx"], d["y"]
    lday, lcid = d["lab_day"], d["lab_cid"]

    # 训练/测试都抽样 —— Kaggle 单次会话有时限, 全量 1040 万条跑不完。
    # 训练 120 万(够训一个头), 测试【全部 2021 年后】以便回测口径一致。
    import pandas as pd
    yr = pd.to_datetime(pd.Series(lday), unit="D").dt.year.to_numpy()
    rng = np.random.default_rng(42)
    tr = np.flatnonzero(yr <= 2020)
    tr = rng.choice(tr, min(1_200_000, len(tr)), replace=False)
    te = np.flatnonzero(yr >= 2021)
    # 测试也抽样: 每 3 天取 1 天, 保留完整横截面(同一天的票要么全取要么全不取)
    days_te = np.unique(lday[te])
    keep_days = set(days_te[::3].tolist())
    te = te[np.isin(lday[te], list(keep_days))]
    sel = np.concatenate([tr, te])
    log.info("训练 %s / 测试 %s (每3天取1天, %d 个交易日)",
             f"{len(tr):,}", f"{len(te):,}", len(keep_days))

    np.savez_compressed(
        f"{DIR}/kaggle_pack.npz",
        ohlcv=d["ohlcv"], day=d["day"], mkt=d["mkt"], day_min=d["day_min"],
        idx=idx[sel], y=y[sel], lab_day=lday[sel], lab_cid=lcid[sel],
        is_train=(np.arange(len(sel)) < len(tr)),
    )
    mb = os.path.getsize(f"{DIR}/kaggle_pack.npz") / 1e6
    log.info("kaggle_pack.npz  %.0f MB  (样本 %s)", mb, f"{len(sel):,}")


if __name__ == "__main__":
    main()
