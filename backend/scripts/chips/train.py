"""筹码特征 + XGBoost 三分类 —— 与裸K 同题目、同评估、可直接比。

⚠️ 用 xgboost 不用 lightgbm: 项目里 build_maimai / build_pump / shape 全是
   xgboost, 引第二个梯度提升库只会多一套要维护的东西。

⚠️ 三段划分与裸K 完全一致(train ≤2019 / val 2020 / test ≥2021), 但筹码
   2018-01 才有数据, 所以【训练集实际只有 2018-2019 两年】, 而裸K 有 2016-2019
   四年。样本更紧, 模型必须做小 —— 这是这条线的先天劣势, 报结果时要说明。

⚠️ 净化采样: 标签是未来 10 根K线, 相邻两天的标签重叠 90%。不做净化的话
   名义样本量会虚高一个数量级, 交叉验证和早停都会被带偏(裸K 上实测名义
   241 万 -> 净化后 17 万, 只有 7.1%)。这里按【同股间隔 >= hold 根】抽。

⚠️ 输出格式与 rawk 的 oos_v2_*.npz 严格一致, 好让 pool.py / decide.py /
   volstrat.py 直接吃 —— 换了信息源但评估口径一个字不改, 才谈得上可比。

用法: python train.py --tag _chips_v1
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("chips.train")

DIR = "/app/data/research"


def purge(day: np.ndarray, cid: np.ndarray, gap: int, seed: int = 0) -> np.ndarray:
    """净化采样: 同一只股票的相邻入选样本至少隔 gap 根K线。

    ⚠️ 必须按股票分组做, 不能全局按日期抽 —— 全局抽会让同一天的几千只票
       全部入选(它们彼此不重叠, 没问题), 但同一只票的连续 10 天也全部入选,
       而那 10 天的标签重叠 90%。
    """
    order = np.lexsort((day, cid))
    keep = np.zeros(len(day), bool)
    last_cid, last_day = -1, -10 ** 9
    for i in order:
        c, d = cid[i], day[i]
        if c != last_cid or d - last_day >= gap:
            keep[i] = True
            last_cid, last_day = c, d
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_chips_v1")
    ap.add_argument("--label", default="label3_dn8.npz")
    ap.add_argument("--feat", default="chips_feat.npz")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--drop-control", action="store_true",
                    help="去掉两个对照特征(集中度/宽度), 验证它们确实没用")
    a = ap.parse_args()

    f = np.load(f"{DIR}/chips/{a.feat}", allow_pickle=True)
    X, names, ok_feat = f["X"], [str(x) for x in f["names"]], f["ok"]
    l3 = np.load(f"{DIR}/rawk/{a.label}")
    y3, ret3, hold = l3["y3"], l3["ret3"], int(l3["hold"])
    p = np.load(f"{DIR}/rawk/panel.npz")
    lab_day, lab_cid = p["lab_day"], p["lab_cid"]

    if a.drop_control:
        keep_col = [i for i, n in enumerate(names) if "对照" not in n]
        X, names = X[:, keep_col], [names[i] for i in keep_col]
        log.info("去掉对照特征, 剩 %d 个", len(names))

    valid = ok_feat & (y3 >= 0)
    yr = pd.to_datetime(lab_day, unit="D").year.to_numpy()
    log.info("有效样本 %s (筹码齐全 & 标签有效)", f"{valid.sum():,}")

    tr_m = valid & (yr <= 2019)
    va_m = valid & (yr == 2020)
    te_m = valid & (yr >= 2021)
    log.info("划分 训练 %s / 验证 %s / 测试 %s",
             f"{tr_m.sum():,}", f"{va_m.sum():,}", f"{te_m.sum():,}")
    log.info("⚠️ 训练集实际起点 %s —— 筹码 2018 才有, 比裸K 少两年",
             pd.to_datetime(lab_day[tr_m].min(), unit="D").date())

    tr_idx = np.where(tr_m)[0]
    keep = purge(lab_day[tr_idx], lab_cid[tr_idx], hold)
    tr_use = tr_idx[keep]
    log.info("⚠️ 净化后训练样本 %s -> %s (%.1f%%) —— 这才是【有效】样本量",
             f"{len(tr_idx):,}", f"{len(tr_use):,}", len(tr_use) / len(tr_idx) * 100)

    va_idx = np.where(va_m)[0]
    te_idx = np.where(te_m)[0]
    Xtr, ytr = X[tr_use], y3[tr_use].astype(int)
    Xva, yva = X[va_idx], y3[va_idx].astype(int)
    log.info("类别分布 训练 涨%.1f%% 跌%.1f%% | 验证 涨%.1f%% 跌%.1f%%",
             (ytr == 1).mean() * 100, (ytr == 2).mean() * 100,
             (yva == 1).mean() * 100, (yva == 2).mean() * 100)

    up, dn = float(l3["up"]), float(l3["dn"])
    va_scores, te_scores = [], []
    for si in range(a.seeds):
        clf = xgb.XGBClassifier(
            n_estimators=a.rounds, max_depth=a.depth, learning_rate=a.lr,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            early_stopping_rounds=40, random_state=si, n_jobs=8, tree_method="hist")
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pv = clf.predict_proba(Xva)
        pt = clf.predict_proba(X[te_idx])
        va_scores.append(pv)
        te_scores.append(pt)
        ev = up * pv[:, 1] - dn * pv[:, 2] - 0.003
        d = pd.DataFrame({"d": lab_day[va_idx], "ev": ev, "r": ret3[va_idx]})
        top = d[d.groupby("d")["ev"].rank(pct=True) >= 0.95]
        log.info("种子%d 最佳轮 %d  验证Top5%% 期望%+.2f%% 涨%.1f%%", si,
                 clf.best_iteration, float(top.r.mean() * 100),
                 float((y3[va_idx][d.groupby("d")["ev"].rank(pct=True) >= 0.95] == 1).mean() * 100))
        if si == 0:
            imp = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])
            log.info("特征重要性:")
            for n, v in imp:
                log.info("    %-18s %.4f", n, v)

    pv = np.mean(va_scores, axis=0)
    pt = np.mean(te_scores, axis=0)
    out = f"{DIR}/chips/oos_v2{a.tag}.npz"
    np.savez(out,
             va_day=lab_day[va_idx], va_p_up=pv[:, 1].astype(np.float32),
             va_p_dn=pv[:, 2].astype(np.float32), va_idx=va_idx,
             lab_day=lab_day[te_idx], p_up=pt[:, 1].astype(np.float32),
             p_dn=pt[:, 2].astype(np.float32), te_idx=te_idx)
    ev = up * pt[:, 1] - dn * pt[:, 2] - 0.003
    d = pd.DataFrame({"d": lab_day[te_idx], "ev": ev, "r": ret3[te_idx],
                      "y": y3[te_idx]})
    rk = d.groupby("d")["ev"].rank(pct=True)
    t5 = d[rk >= 0.95]
    log.info("★ %d种子集成 | 测试Top5%% 期望%+.2f%% 识别%.1f%% (基准 期望%+.2f%% 识别%.1f%%)",
             a.seeds, float(t5.r.mean() * 100), float((t5.y == 1).mean() * 100),
             float(d.r.mean() * 100), float((d.y == 1).mean() * 100))
    log.info("存 %s", out)


if __name__ == "__main__":
    main()
