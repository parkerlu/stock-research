"""实际持有天数 —— 标签是"先碰 +10% 或 -8% 就出场", 但 decide.py 的资金曲线
按固定 10 天摊, 等于假设提前出场的钱要在场外空等到第 10 天。

用户 2026-09-09 指出: "买入到10%就抛, 等下次" —— 资金周转应该跟着变快。
这里先量化: 实际平均持有几天。
"""
import numpy as np

D = "/root/data"
d = np.load(f"{D}/panel.npz")
ohlcv, cid, idx = d["ohlcv"], d["cid"], d["idx"].astype(np.int64)
o, h, l, c = ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
l3 = np.load(f"{D}/label3_dn8.npz")
y3, up, dn, H = l3["y3"], float(l3["up"]), float(l3["dn"]), int(l3["hold"])
n_bar, n = len(o), len(idx)

j = np.minimum(idx + 1, n_bar - 1)
entry = o[j]
up_lv, dn_lv = entry * (1 + up), entry * (1 - dn)
up_first = np.full(n, H + 1, np.int16)
dn_first = np.full(n, H + 1, np.int16)
for k in range(1, H + 1):
    jj = np.minimum(idx + k, n_bar - 1)
    hu = (h[jj] >= up_lv) & (up_first > H)
    hd = (l[jj] <= dn_lv) & (dn_first > H)
    up_first[hu] = k
    dn_first[hd] = k

days = np.full(n, H, np.int16)
m_up = (y3 == 1)
m_dn = (y3 == 2)
days[m_up] = up_first[m_up]
days[m_dn] = dn_first[m_dn]
np.savez(f"{D}/holddays_dn8.npz", days=days)

ok = y3 >= 0
print(f"有效样本 {ok.sum():,}")
for nm, m in [("涨(先碰+10%)", m_up & ok), ("跌(先碰-8%)", m_dn & ok), ("平(持满)", (y3 == 0) & ok)]:
    if m.sum():
        dd = days[m]
        print(f"{nm:14s} 占比 {m.mean()*100:5.1f}%  平均持有 {dd.mean():5.2f} 天  "
              f"中位 {np.median(dd):.0f}  第1天出场占 {(dd==1).mean()*100:4.1f}%")
print(f"\n全样本平均持有 {days[ok].mean():.2f} 天 (名义 {H} 天)")
print(f"资金周转加速 {H/days[ok].mean():.2f}x")
