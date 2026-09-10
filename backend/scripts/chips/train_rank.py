"""排序目标 vs 分类目标 —— 用户 2026-09-10 提出的想法的直接检验。

用户原话大意: 与其让模型记住"涨/跌/平"三类, 不如只让它记住会涨的,
对不涨的票直接说"预测不了"。

这个想法指向 train2.py 里记了很久的【缺陷三】: 训练目标与使用方式不一致。
    训练时: 交叉熵逼模型对【每一只】票都拟合出一个确定答案, 哪怕它没把握 ——
            模型把容量浪费在记忆不可预测的样本上。
    使用时: 只取当日横截面 Top1%, 其余 99% 全部丢弃, 绝对概率从来没用过。

所以正确的目标函数应该是【按天分组的排序】: 只要求同一天里会涨的排在前面,
不要求它说出"涨的概率是 31%"。模型可以对绝大多数票保持"不知道",
只要它把有把握的那几只顶上去就行 —— 这正是用户说的"预测不了"。

⚠️ 对照必须严格同口径: 同一批特征、同一个三段划分、同一套净化采样、同样 3 种子,
   只换 objective。否则分不清是目标函数的功劳还是别的。

⚠️ 排序标签用 涨=2 / 平=1 / 跌=0 —— 有序, 且刻意让"平"排在中间:
   我们宁可买到不涨不跌的, 也不要买到跌的。

用法: python train_rank.py --tag _chips_rank
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.chips.features import FEAT_NAMES
from scripts.chips.train import purge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("chips.rank")
DIR = "/app/data/research"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_chips_rank")
    ap.add_argument("--feat", default="chips_feat.npz")
    ap.add_argument("--label", default="label3_dn8.npz")
    ap.add_argument("--obj", default="rank:pairwise",
                    choices=["rank:pairwise", "rank:ndcg", "multi:softprob"])
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-early", action="store_true", help="关早停, 训满")
    a = ap.parse_args()

    f = np.load(f"{DIR}/chips/{a.feat}", allow_pickle=True)
    X, ok_feat = f["X"], f["ok"]
    names = [str(x) for x in f["names"]]
    l3 = np.load(f"{DIR}/rawk/{a.label}")
    y3, ret3, hold = l3["y3"], l3["ret3"], int(l3["hold"])
    up, dn = float(l3["up"]), float(l3["dn"])
    p = np.load(f"{DIR}/rawk/panel.npz")
    lab_day, lab_cid = p["lab_day"], p["lab_cid"]

    valid = ok_feat & (y3 >= 0)
    yr = pd.to_datetime(lab_day, unit="D").year.to_numpy()
    tr_idx = np.where(valid & (yr <= 2019))[0]
    va_idx = np.where(valid & (yr == 2020))[0]
    te_idx = np.where(valid & (yr >= 2021))[0]
    keep = purge(lab_day[tr_idx], lab_cid[tr_idx], hold)
    tr_use = tr_idx[keep]
    log.info("目标 %s | 训练 %s(净化后) / 验证 %s / 测试 %s", a.obj,
             f"{len(tr_use):,}", f"{len(va_idx):,}", f"{len(te_idx):,}")

    # ⚠️ 排序目标要求样本按 group 连续排列, group 是每天的样本数
    def by_day(ix: np.ndarray):
        order = np.argsort(lab_day[ix], kind="stable")
        ix2 = ix[order]
        _, cnt = np.unique(lab_day[ix2], return_counts=True)
        return ix2, cnt

    is_rank = a.obj.startswith("rank")
    if is_rank:
        tr_use, tr_grp = by_day(tr_use)
        va_ix, va_grp = by_day(va_idx)
    else:
        va_ix, tr_grp, va_grp = va_idx, None, None

    # 排序标签: 涨=2 / 平=1 / 跌=0。刻意让"平"在中间 —— 宁可买不涨不跌的, 不要买跌的
    def rank_label(y):
        return np.where(y == 1, 2, np.where(y == 2, 0, 1)).astype(int)

    ytr = rank_label(y3[tr_use]) if is_rank else y3[tr_use].astype(int)
    yva = rank_label(y3[va_ix]) if is_rank else y3[va_ix].astype(int)

    va_scores, te_scores = [], []
    for si in range(a.seeds):
        common = dict(n_estimators=a.rounds, max_depth=a.depth, learning_rate=a.lr,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                      random_state=si, n_jobs=8, tree_method="hist")
        if not a.no_early:
            common["early_stopping_rounds"] = 40
        if is_rank:
            # ⚠️ eval_metric 必须贴合实际用法。第一版用 ndcg@100, 结果三个种子的
            #    最佳轮是 0/9/1 —— 每天几千只票时 Top100 的 ndcg 太不敏感,
            #    第一轮就"最优", 模型等于没训。改成 ndcg@50: 我们每天只买
            #    Top1%(约50只), 评估口径要对上。
            clf = xgb.XGBRanker(objective=a.obj, eval_metric="ndcg@50", **common)
            clf.fit(X[tr_use], ytr, group=tr_grp,
                    eval_set=[(X[va_ix], yva)], eval_group=[va_grp], verbose=False)
            sv, st = clf.predict(X[va_ix]), clf.predict(X[te_idx])
        else:
            clf = xgb.XGBClassifier(objective=a.obj, num_class=3,
                                    eval_metric="mlogloss", **common)
            clf.fit(X[tr_use], ytr, eval_set=[(X[va_ix], yva)], verbose=False)
            pv, pt = clf.predict_proba(X[va_ix]), clf.predict_proba(X[te_idx])
            sv = up * pv[:, 1] - dn * pv[:, 2]
            st = up * pt[:, 1] - dn * pt[:, 2]
        va_scores.append(sv)
        te_scores.append(st)
        d = pd.DataFrame({"d": lab_day[va_ix], "s": sv, "r": ret3[va_ix],
                          "y": y3[va_ix]})
        top = d[d.groupby("d")["s"].rank(pct=True) >= 0.95]
        best = clf.best_iteration if not a.no_early else f"训满{a.rounds}"
        log.info("种子%d 最佳轮 %s  验证Top5%% 期望%+.2f%% 涨%.1f%%", si,
                 best, float(top.r.mean() * 100),
                 float((top.y == 1).mean() * 100))
        if si == 0:
            imp = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])[:5]
            log.info("重要性前5: %s", ", ".join(f"{n} {v:.3f}" for n, v in imp))

    sv = np.mean(va_scores, axis=0)
    st = np.mean(te_scores, axis=0)
    d = pd.DataFrame({"d": lab_day[te_idx], "s": st, "r": ret3[te_idx], "y": y3[te_idx]})
    rk = d.groupby("d")["s"].rank(pct=True)
    for lab, m in [("Top5%", rk >= 0.95), ("Top1%", rk >= 0.99)]:
        g = d[m]
        log.info("★ 测试 %-6s 期望%+.2f%% 识别%.1f%% 跌%.1f%% (基准 %+.2f%% / %.1f%%)",
                 lab, float(g.r.mean() * 100), float((g.y == 1).mean() * 100),
                 float((g.y == 2).mean() * 100), float(d.r.mean() * 100),
                 float((d.y == 1).mean() * 100))
    # 方向判别力: 只在真的动了的票里看
    mv = d[d.y.isin([1, 2])].copy()
    mv["rk"] = mv.groupby("d")["s"].rank(pct=True)
    base = (mv.y == 1).mean() * 100
    hi = (mv[mv.rk >= 0.90].y == 1).mean() * 100
    lo = (mv[mv.rk <= 0.10].y == 1).mean() * 100
    log.info("★ 方向判别力(只看真动了的): 基准%.1f%%  最看多Top10%% %.1f%%(%+.1fpt)  "
             "最看空Bot10%% %.1f%%(%+.1fpt)", base, hi, hi - base, lo, lo - base)

    out = f"{DIR}/chips/oos_v2{a.tag}.npz"
    np.savez(out, va_day=lab_day[va_ix], va_idx=va_ix,
             va_p_up=((sv - sv.min()) / max(sv.max() - sv.min(), 1e-9)).astype(np.float32),
             va_p_dn=np.zeros(len(va_ix), np.float32),
             lab_day=lab_day[te_idx], te_idx=te_idx,
             p_up=((st - st.min()) / max(st.max() - st.min(), 1e-9)).astype(np.float32),
             p_dn=np.zeros(len(te_idx), np.float32))
    log.info("存 %s  ⚠️ 排序分已线性映射到 [0,1] 冒充 p_up, 只为复用 pool.py; "
             "它不是概率, 别当概率解释", out)


if __name__ == "__main__":
    main()
