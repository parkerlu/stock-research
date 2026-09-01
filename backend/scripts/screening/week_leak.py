"""专测多周期穿越: 周线值是否含当周未来几天的数据?

S1 那版检验切点随机, 信号落在残缺周内的概率低, 检验力不足 —— 这版专门
在每个"周"的第 1/2/3/4 天切, 直击问题。

判据: 若周线是因果的, 用 data[:cut] 算出的最后一天的 wk/wd, 应等于用全量
数据算出的同一天的 wk/wd。不等 = 那天的周线值依赖了当天之后的K线。
"""
import asyncio, numpy as np, pandas as pd
from sqlalchemy import select
from app.db import async_session
from app.models.schema import DailyCandle
from app.services.strategy_templates.tdx_classics import _compute_pack
import screen as S

async def main():
    codes = await S.load_codes()
    codes = list(np.random.default_rng(5).choice(codes, 40, replace=False))
    diff_by_pos = {0: [], 1: [], 2: [], 3: [], 4: []}
    n_sig_diff = 0
    for cd in codes:
        df = await S.load_bars(cd)
        if df is None or len(df) < 400: continue
        c = df.close.to_numpy('float64'); h = df.high.to_numpy('float64')
        l = df.low.to_numpy('float64'); v = df.vol.to_numpy('float64')
        full = _compute_pack(c, h, l, v)
        n = len(c)
        for cut in range(320, n - 1, 37):
            part = _compute_pack(c[:cut], h[:cut], l[:cut], v[:cut])
            i = cut - 1                      # 截断后的最后一天
            pos = i % 5                      # 它在本"周"的第几天 (0=周一)
            d_wk = abs(part["wk"][i] - full["wk"][i])
            diff_by_pos[pos].append(d_wk)
            if d_wk > 1e-9:
                n_sig_diff += 1
    print(f"{'当周第几天':<12}{'样本':>7}{'周线KDJ差异均值':>18}{'有差异占比':>12}")
    for p in range(5):
        a = np.array(diff_by_pos[p])
        if len(a) == 0: continue
        print(f"  第 {p+1} 天{'':<6}{len(a):>7}{a.mean():>18.4f}{100*(a>1e-9).mean():>11.1f}%")
    print(f"\n总计有差异的切点: {n_sig_diff}")
    print("解读: 若第 1~4 天差异显著而第 5 天为 0, 说明周线用了整周数据 = 穿越")

if __name__ == "__main__":
    asyncio.run(main())
