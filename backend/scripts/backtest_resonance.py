"""v4/v5 组合回测 —— 含成本、滑点、T+1 进场、大盘择时。

用法: docker exec stock-backend-1 python -m scripts.backtest_resonance

⚠️ 共振信号的时间窗只能向后看。原定义 mm.trade_date ± 5/± 7 是前视:
T 日信号要求知道未来会不会上龙虎榜。CAUSAL=False 可复现当时的错误数字。

结论(2026-09-05): v5 因果口径 7.7 年仅 58 个信号, 做不了策略;
v4 最好一档 年化 22.8% / 回撤 28.2% = 0.81, 仍不满足"年化 > 回撤"。
"""
import asyncio, numpy as np, pandas as pd
from sqlalchemy import text
from app.db import engine

# A股实际成本(2023-08 印花税减半后)
COMM, STAMP, TRANSFER = 0.00025, 0.0005, 0.00001   # 佣金双边/印花税卖出/过户费双边
SLIP = 0.001                                        # 滑点单边

CAUSAL = True
USE_LB = False
REGIME = True   # 大盘择时: 全市场等权指数在自身20日均线之上才建仓   # True: v5(三重); False: v4(v3强×吸筹强, 不要龙虎榜)
V5_SQL = """
with mm as (select ts_code, trade_date, rank_pct from maimai_signal
            where side='buy' and rank_pct >= 0.8),
pp as (select ts_code, trade_date from pump_signal where rank_pct >= 0.95),
lb as (select ts_code, trade_date, net_amount from top_list where net_amount > 0)
select mm.ts_code, mm.trade_date, max(lb.net_amount) net_amt
from mm join pp on pp.ts_code=mm.ts_code
          and pp.trade_date between mm.trade_date - 5 and mm.trade_date + {PF}
        join lb on lb.ts_code=mm.ts_code
          and lb.trade_date between mm.trade_date - 7 and mm.trade_date + {LF}
group by mm.ts_code, mm.trade_date
"""
V4_SQL = """
with mm as (select ts_code, trade_date, rank_pct from maimai_signal
            where side='buy' and rank_pct >= 0.8),
pp as (select ts_code, trade_date from pump_signal where rank_pct >= 0.95)
select mm.ts_code, mm.trade_date, mm.rank_pct::float net_amt
from mm join pp on pp.ts_code=mm.ts_code
          and pp.trade_date between mm.trade_date - 5 and mm.trade_date + {PF}
group by mm.ts_code, mm.trade_date, mm.rank_pct
"""

async def load():
    async with engine.connect() as c:
        sql = (V5_SQL if USE_LB else V4_SQL).format(PF=0 if CAUSAL else 5, LF=0 if CAUSAL else 7)
        sig = pd.DataFrame((await c.execute(text(sql))).fetchall(),
                           columns=['ts_code','trade_date','net_amt'])
        codes = tuple(sig['ts_code'].unique())
        d = pd.DataFrame((await c.execute(text(
            "select ts_code,trade_date,open,close,adj_factor from daily_candle "
            "where trade_date>='2019-01-01' and ts_code = any(:cs)"), {"cs": list(codes)})).fetchall(),
            columns=['ts_code','trade_date','open','close','adj'])
        cal = pd.DataFrame((await c.execute(text(
            "select distinct trade_date from daily_candle where trade_date>='2019-01-01' "
            "order by trade_date"))).fetchall(), columns=['trade_date'])
        mkt = pd.DataFrame((await c.execute(text("""
            with r as (select trade_date, close/nullif(lag(close) over
                       (partition by ts_code order by trade_date),0) - 1 ret
                       from daily_candle where trade_date >= '2018-06-01')
            select trade_date, avg(ret) m from r where ret is not null group by trade_date
        """))).fetchall(), columns=['trade_date','m'])
    return sig, d, cal, mkt

def run(sig, px, dates, HOLD=20, MAXPOS=8, CAP=1_000_000.0):
    """T日收盘后出信号 → T+1开盘买入 → 持有 HOLD 个交易日 → 收盘卖出。"""
    di = {d: i for i, d in enumerate(dates)}
    # 按信号日分组, 同日多个信号按龙虎榜净买入降序(净买入大的优先)
    sig = sig.sort_values(['trade_date','net_amt'], ascending=[True, False])
    queue = {}
    for d, g in sig.groupby('trade_date'):
        i = di.get(d)
        if i is not None and i + 1 < len(dates):
            queue.setdefault(dates[i+1], []).extend(g['ts_code'].tolist())

    cash, holds, trades, eq = CAP, [], [], []
    for i, d in enumerate(dates):
        # 到期卖出(收盘)
        still = []
        for h in holds:
            if i >= h['exit_i']:
                p = px.get((h['ts_code'], d))
                if p is None or not np.isfinite(p[1]):
                    still.append(h); continue
                sell = p[1] * (1 - SLIP)
                proceeds = h['sh'] * sell * (1 - COMM - STAMP - TRANSFER)
                cash += proceeds
                trades.append({'code': h['ts_code'], 'in': h['in_d'], 'out': d,
                               'ret': proceeds / h['cost'] - 1})
            else:
                still.append(h)
        holds = still
        # 建仓(开盘)
        if REGIME and d not in globals().get('OK_DAYS', set()):
            queue.pop(d, None)                       # 大盘在均线下, 当日信号直接放弃
        for code in queue.get(d, []):
            if len(holds) >= MAXPOS: break
            p = px.get((code, d))
            if p is None or not np.isfinite(p[0]) or p[0] <= 0: continue
            equity = cash + sum(hh['sh'] * hh['last'] for hh in holds)
            alloc = min(equity / MAXPOS, cash)
            if alloc < 1000: continue
            buy = p[0] * (1 + SLIP)
            sh = alloc / (buy * (1 + COMM + TRANSFER))
            cost = sh * buy * (1 + COMM + TRANSFER)
            cash -= cost
            holds.append({'ts_code': code, 'sh': sh, 'cost': cost, 'in_d': d,
                          'exit_i': i + HOLD, 'last': p[1]})
        # 估值
        mv = 0.0
        for h in holds:
            p = px.get((h['ts_code'], d))
            if p and np.isfinite(p[1]): h['last'] = p[1]
            mv += h['sh'] * h['last']
        eq.append({'d': d, 'eq': cash + mv, 'n': len(holds)})
    return pd.DataFrame(eq), pd.DataFrame(trades)

def stats(eq, tr, CAP):
    e = eq['eq'].values
    yrs = len(eq) / 252
    cagr = (e[-1]/CAP)**(1/yrs) - 1
    dd = float(np.max(1 - e/np.maximum.accumulate(e)))
    r = np.diff(e)/e[:-1]
    sharpe = r.mean()/r.std()*np.sqrt(252) if r.std() > 0 else 0
    return dict(总收益=f"{e[-1]/CAP-1:+.1%}", 年化=f"{cagr:+.2%}", 最大回撤=f"{dd:.2%}",
                年化除回撤=f"{cagr/dd:.2f}" if dd>0 else "—", 夏普=f"{sharpe:.2f}",
                交易数=len(tr), 胜率=f"{(tr['ret']>0).mean():.1%}" if len(tr) else "—",
                均收益=f"{tr['ret'].mean():+.2%}" if len(tr) else "—",
                平均持仓数=f"{eq['n'].mean():.1f}", 空仓天占比=f"{(eq['n']==0).mean():.1%}")

async def main():
    sig, d, cal, mkt = await load()
    d['trade_date']=pd.to_datetime(d['trade_date']); sig['trade_date']=pd.to_datetime(sig['trade_date'])
    d['o']=d['open'].astype(float)*d['adj'].astype(float)
    d['c']=d['close'].astype(float)*d['adj'].astype(float)
    px = {(r.ts_code, r.trade_date): (r.o, r.c) for r in d.itertuples()}
    cal['trade_date']=pd.to_datetime(cal['trade_date'])
    dates = list(cal['trade_date'])
    mkt['trade_date']=pd.to_datetime(mkt['trade_date'])
    mkt=mkt.sort_values('trade_date')
    mkt['idx']=(1+mkt['m'].astype(float)).cumprod()
    mkt['ma']=mkt['idx'].rolling(20).mean()
    ok_days = set(mkt.loc[mkt['idx']>mkt['ma'],'trade_date'])
    globals()['OK_DAYS']=ok_days
    print(f"{'v5' if USE_LB else 'v4'} {'因果' if CAUSAL else '前视'}: 信号 {len(sig)} 个, 交易日 {len(dates)}, 区间 {dates[0].date()} ~ {dates[-1].date()}\n")
    CAP=1_000_000.0
    for MAXPOS in (4, 6, 8, 12):
        eq, tr = run(sig, px, dates, MAXPOS=MAXPOS, CAP=CAP)
        s = stats(eq, tr, CAP)
        print(f"MAXPOS={MAXPOS:2d}  " + "  ".join(f"{k} {v}" for k,v in s.items()))
    # 基准: 全市场等权
    dd = d.groupby('trade_date')['c'].apply(lambda x: x.mean())
    base = d.pivot_table(index='trade_date', values='c', aggfunc='mean')
    print()
    eq, tr = run(sig, px, dates, MAXPOS=8, CAP=CAP)
    eq.to_parquet('/app/data/research/v5_equity.parquet')
    tr.to_parquet('/app/data/research/v5_trades.parquet')
    print("按年:")
    tr['y']=pd.to_datetime(tr['out']).dt.year
    for y,g in tr.groupby('y'):
        print(f"  {y}  交易 {len(g):3d}  胜率 {(g['ret']>0).mean():5.1%}  均收益 {g['ret'].mean():+6.2%}")

asyncio.run(main())
