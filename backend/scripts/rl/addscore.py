"""给 RL 面板补两列: 监督模型(v5)的当日分位, 以及是否 Top1%。

⚠️ 为什么要这个: PPO 从随机策略起步时, 随机换手 40 天要亏 3.86%(几乎全是
   手续费), 梯度被成本主导, 策略在发现 alpha 之前就先学会"永远空仓"
   (实测 dd=0 也一样收敛到仓位 0.00)。

   把监督模型的分数作为状态给它, 相当于给了一个现成的先验 ——
   "分数高时买"这条策略它一步就能表达出来, 不必从零摸索。
   之后再用行为克隆热启动, 让 PPO 在及格线之上改进而不是从负分爬起。

⚠️ 分数必须是【当日横截面分位】, 与 shape_score 生产口径一致。
"""
from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rl.score")
OUT = "/app/data/research/rl"
MODEL = "/app/data/research/shape/shape_v5_y_0.15_0.08_h20.json"


def main() -> None:
    m = xgb.XGBRegressor()
    m.load_model(MODEL)
    mcols = list(m.get_booster().feature_names)

    for tag in ("train", "test"):
        d = np.load(f"{OUT}/{tag}.npz", allow_pickle=True)
        feats, cols = d["feats"], [str(c) for c in d["cols"]]
        day = d["day"]
        assert cols == mcols, f"特征顺序与模型不一致: {cols[:3]} vs {mcols[:3]}"
        sc = np.empty(len(feats), np.float32)
        for i in range(0, len(feats), 500_000):
            sc[i:i + 500_000] = m.predict(feats[i:i + 500_000]).astype(np.float32)
        rk = (pd.Series(sc).groupby(pd.Series(day)).rank(pct=True)
              .to_numpy(np.float32))
        newf = np.concatenate([feats, rk[:, None],
                               (rk >= 0.99).astype(np.float32)[:, None]], axis=1)
        np.savez(f"{OUT}/{tag}.npz", feats=newf, ret=d["ret"], cid=d["cid"],
                 day=day, cols=np.array(cols + ["sup_rank", "sup_top1"]))
        log.info("%s: %s 行, 特征 %d -> %d (Top1%% 占 %.2f%%)",
                 tag, f"{len(newf):,}", len(cols), newf.shape[1],
                 float((rk >= 0.99).mean()) * 100)
        del d, feats, sc, rk, newf
        gc.collect()


if __name__ == "__main__":
    main()
