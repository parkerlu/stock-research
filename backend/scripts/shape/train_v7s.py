"""v7 补跑: 只训 all+S 一组(上一进程在第6组被 OOM kill)。
直接读 _yr7 落盘数据, 不再重建标签/分位 —— 轻量进程。"""
from __future__ import annotations
import gc, logging
import numpy as np, pandas as pd, xgboost as xgb

log = logging.getLogger("shape.v7s")
DIR = "/app/data/research/shape"; TMP = f"{DIR}/_yr7"; TAG = "y_0.15_0.08"
GROUP_L = ["ret120_z","ret250_z","pos120","pos250","dist_ma120","dist_ma250","dhi250","dlo250","vol_60_250"]
GROUP_P = ["er20","er60","up_rate20","up_rate60","skew60","spike20","clv_std20","amp_5_20","gapfreq20"]
GROUP_D = ["pv_div20","illiq20","vol_conc20","vol_accel"]; GROUP_M = ["corr_mkt60","beta60"]
GROUP_S = ["sig_pump","sig_maimai","sig_brk"]

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    import pyarrow.parquet as pq
    allc = pq.ParquetFile(f"{TMP}/2016.parquet").schema_arrow.names
    feat_cols = [c for c in allc if c not in ("day","cid",TAG,"fwd")]
    base = [c for c in feat_cols if c not in GROUP_L+GROUP_P+GROUP_D+GROUP_M+GROUP_S]
    cc = base + GROUP_L + GROUP_P + GROUP_D + GROUP_M + GROUP_S
    tr_years, te_years = list(range(2016,2021)), list(range(2021,2027))
    per = 3_000_000 // len(tr_years)
    trs = []
    for v in tr_years:
        d = pd.read_parquet(f"{TMP}/{v}.parquet", columns=cc+[TAG])
        trs.append(d.sample(min(per,len(d)), random_state=42)); del d; gc.collect()
    tr = pd.concat(trs, ignore_index=True); del trs; gc.collect()
    y = tr[TAG].to_numpy(np.float32)
    X = tr[cc]; log.info("训练 %s 行 × %d 列", f"{len(tr):,}", len(cc))
    m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                         reg_lambda=2.0, tree_method="hist", n_jobs=8, random_state=42)
    m.fit(X, y); m.save_model(f"{DIR}/shape_v7_all_S.json")
    del X, tr, y; gc.collect()
    imp = dict(zip(cc, m.feature_importances_))
    new_share = sum(v for k,v in imp.items() if k not in base)
    sig_share = sum(v for k,v in imp.items() if k in GROUP_S)
    top_new = sorted(((k,v) for k,v in imp.items() if k not in base), key=lambda x:-x[1])[:5]
    acc = {k:[0,0.0] for k in ("Top1%","Top5%","Bot20%")}; yearly, dl = [], []
    for v in te_years:
        d = pd.read_parquet(f"{TMP}/{v}.parquet", columns=cc+[TAG,"day","fwd"])
        sc = np.empty(len(d), np.float32)
        for i in range(0,len(d),500_000):
            sc[i:i+500_000] = m.predict(d[cc].iloc[i:i+500_000]).astype(np.float32)
        d = d[["day","fwd",TAG]].assign(score=sc)
        d["rk"] = d.groupby("day")["score"].rank(pct=True)
        for k,(lo,hi) in {"Top1%":(0.99,1.01),"Top5%":(0.95,1.01),"Bot20%":(0.0,0.20)}.items():
            g = d[(d.rk>=lo)&(d.rk<hi)]; acc[k][0]+=len(g); acc[k][1]+=float(g[TAG].sum())
        yearly.append(float(d[d.rk>=0.95][TAG].mean())*100)
        dl.append(d.groupby("day").agg(avg=("score","mean"), m=("fwd","mean")).reset_index())
        del d, sc; gc.collect()
    jd = pd.concat(dl, ignore_index=True); rcorr = jd["avg"].corr(jd["m"])
    t1=acc["Top1%"][1]/max(acc["Top1%"][0],1)*100; t5=acc["Top5%"][1]/max(acc["Top5%"][0],1)*100
    b20=acc["Bot20%"][1]/max(acc["Bot20%"][0],1)*100
    log.info("all+S     Top1%% %+5.2f  Top5%% %+5.2f  Bot20%% %+5.2f | 负年 %d 最小 %+5.2f | 审计 %+.3f | 新特征占比 %4.1f%% (信号 %4.1f%%) | top新: %s",
             t1,t5,b20,sum(1 for x in yearly if x<0),min(yearly),rcorr,new_share*100,sig_share*100,
             " ".join(f"{k}:{v:.3f}" for k,v in top_new))
    log.info("all+S       逐年: %s", "  ".join(f"{y_}:{v:+.2f}" for y_,v in zip(te_years,yearly)))

if __name__ == "__main__":
    main()
