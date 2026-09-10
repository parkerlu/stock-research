"""共振打分 —— 筹码模型 × 裸K CNN，写 combo_score 表。

这是目前【唯一】组合层面过线的东西:
    walk-forward + 资金池 + 真实周转, 仓位20 -> 比值 1.07
    (年化 +27.7% / 回撤 −26.0%), 六年零负年。
    单用: 筹码 0.89 / 裸K 0.78。

⚠️ 组合方式是【两边 EV 等权平均】, 而且这个选择过了诚实复核:
   在 2020 验证集上按同一判据比 avg / rank / min -> 5.74 / 2.62 / 1.66,
   选出 avg 后测试集只跑这一次 -> 1.07。不是在测试集上挑的。

⚠️ 为什么共振有效: 两个模型当日横截面分位的相关只有 +0.021, 几乎正交 ——
   一个看 400 根K线的形态, 一个看持仓成本结构。纯选股超额是超加性的:
   筹码 +0.33pp、裸K +0.37pp, 合起来 +0.45pp。

⚠️ 裸K 侧要跑 CNN 推理。实测 batch=16 时全市场一天 221s / 743MB ——
   batch 不能开大: 64 时内存到 1026MB, 而速度一模一样(CPU 推理是内存带宽
   受限, 不是并行度受限)。

⚠️ 模型权重在挂载卷 /app/data/research/rawk/ckpt/。放 /app 别处重建镜像就没了
   (本项目已因此丢过三次东西)。

⚠️ 必须算全市场: rank_pct 是当日横截面分位, 单只算没有意义。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models.schema import ComboScore

log = logging.getLogger("combo_score")
EPOCH = date(1970, 1, 1)       # 与 panel.npz 的 day 基准一致
CKPT = "/app/data/research/rawk/ckpt"
TAG = "_cls512_nbar400"
SEEDS = 3
NBAR, NCH = 400, 14
H = 512
DIL = (1, 2, 4, 8, 16, 32, 64)      # nbar>=256 那一档, 必须与训练时一致
BS = 16                              # ⚠️ 见模块 docstring: 大 batch 只涨内存不提速
UP, DN, COST = 0.10, 0.08, 0.003
CH = 3000


def load_nets() -> list:
    import sys
    sys.path.insert(0, "/app/scripts")
    from rawk.train import Net

    nets = []
    for si in range(SEEDS):
        net = Net(ch=NCH, h=H, dilations=DIL, n_out=3)
        net.load_state_dict(torch.load(f"{CKPT}/cnn2{TAG}_s{si}.pt", map_location="cpu"))
        net.eval()
        nets.append(net)
    return nets


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1,
                    help="回补最近几个交易日。⚠️ 每天都要跑一遍 CNN 推理, "
                         "回补 N 天就是 N 倍时间")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--date", default=None,
                    help="只打某一天(YYYY-MM-DD) —— 用来和研究阶段的 oos 对账")
    ap.add_argument("--dry", action="store_true", help="不写库, 只算并打印")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    eng = create_async_engine(settings.database_url)
    async with eng.connect() as c:
        end = (await c.execute(text("select max(trade_date) from chips_score"))).scalar()
        if not end:
            raise SystemExit("chips_score 为空 —— 先跑 build_chips_scores")
        if a.date:
            days = [date.fromisoformat(a.date)]
        else:
            days = [r[0] for r in (await c.execute(text(
                "select distinct trade_date from chips_score where trade_date <= :e "
                "order by trade_date desc limit :n"), {"e": end, "n": a.days})).fetchall()]
    days = sorted(days)
    log.info("待打分 %d 个交易日: %s ~ %s", len(days), days[0], days[-1])

    # 裸K 需要每只票 NBAR 根历史 —— 多取 60% 的日历日覆盖停牌
    start = days[0] - timedelta(days=int(NBAR * 1.6) + 30)
    async with eng.connect() as c:
        codes = [r[0] for r in (await c.execute(text(
            "select distinct ts_code from chips_score where trade_date = :d"),
            {"d": days[-1]})).fetchall()]
        if not codes:      # 对账用的历史日可能还没算过筹码分
            codes = [r[0] for r in (await c.execute(text(
                "select d.ts_code from daily_candle d join stock_basic b "
                "on b.ts_code=d.ts_code where d.trade_date = :d "
                "and b.name not like :st"), {"d": days[-1], "st": "%ST%"})).fetchall()]
    log.info("全市场 %d 只(已排除 ST, 跟随 chips_score 的口径)", len(codes))

    nets = load_nets()
    log.info("载入 %d 个种子的 CNN (h=%d, nbar=%d)", len(nets), H, NBAR)

    # 大盘通道 —— 与训练时一致: 中证1000 + 沪深300 的 OHLC
    async with eng.connect() as c:
        idx_rows = (await c.execute(text("""
            select ts_code, trade_date, open, high, low, close from index_daily
            where ts_code in ('000852.SH','000300.SH') and trade_date between :a and :b
            order by ts_code, trade_date
        """), {"a": start, "b": end})).fetchall()
    idx = pd.DataFrame(idx_rows, columns=["ts_code", "trade_date", "o", "h", "l", "c"])
    for col in ("o", "h", "l", "c"):
        idx[col] = pd.to_numeric(idx[col], errors="coerce")
    log.info("大盘 %s 行", f"{len(idx):,}")

    out_rows: list[dict] = []
    async with eng.connect() as c:
        for i in range(0, len(codes), 300):
            batch_codes = codes[i:i + 300]
            rows = (await c.execute(text("""
                select ts_code, trade_date, open*adj_factor, high*adj_factor,
                       low*adj_factor, close*adj_factor, vol
                from daily_candle
                where ts_code = any(:cs) and trade_date between :a and :b
                order by ts_code, trade_date
            """), {"cs": batch_codes, "a": start, "b": end})).fetchall()
            if not rows:
                continue
            px = pd.DataFrame(rows, columns=["ts_code", "trade_date", "o", "h",
                                             "l", "c", "v"])
            for col in ("o", "h", "l", "c", "v"):
                px[col] = pd.to_numeric(px[col], errors="coerce").astype("float64")
            out_rows.extend(_score_chunk(px, idx, days, nets))
            if (i // 300) % 5 == 0:
                log.info("  %d/%d 只已打分, 累计 %s 行",
                         min(i + 300, len(codes)), len(codes), f"{len(out_rows):,}")

    if not out_rows:
        raise SystemExit("没有算出任何分数 —— 检查历史长度是否够 400 根")
    raw = pd.DataFrame(out_rows)
    log.info("裸K 打分 %s 行", f"{len(raw):,}")
    if a.dry:
        raw.to_parquet("/tmp/combo_dry.parquet")
        log.info("dry: 存 /tmp/combo_dry.parquet, 不写库")
        await eng.dispose()
        return

    # ---- 与筹码分数合并, 做 avg 共振 ----
    async with eng.connect() as c:
        ch_rows = (await c.execute(text(
            "select ts_code, trade_date, ev from chips_score where trade_date = any(:ds)"),
            {"ds": days})).fetchall()
    ch = pd.DataFrame(ch_rows, columns=["ts_code", "trade_date", "ev_chips"])
    ch["ev_chips"] = pd.to_numeric(ch["ev_chips"], errors="coerce").astype(float)
    m = raw.merge(ch, on=["ts_code", "trade_date"], how="inner")
    log.info("两边都有分的 %s 行 (裸K %s / 筹码 %s)",
             f"{len(m):,}", f"{len(raw):,}", f"{len(ch):,}")
    if m.empty:
        raise SystemExit("裸K 与筹码没有交集 —— 检查日期口径")

    # ⚠️ avg = 两边 EV 等权平均。这个选择在验证集上复核过(见模块 docstring)
    m["ev"] = (m.ev_rawk + m.ev_chips) / 2
    m["rank_pct"] = m.groupby("trade_date")["ev"].rank(pct=True)
    # ⚠️ 判据用【逐日】的分布, 不能拿单日去比研究阶段那个 +0.021 ——
    #    那是把六年所有天混在一起算的。实测逐日相关本身波动很大:
    #    中位 +0.013, p10 −0.182, p90 +0.209。单日 +0.23 只是 p91, 属正常。
    #    真正该报警的是【持续】偏高(说明两个模型开始看同一件事, 共振失去意义)。
    rho = m.groupby("trade_date")[["ev_rawk", "ev_chips"]].rank(pct=True).corr().iloc[0, 1]
    flag = "  ⚠️ 超出逐日 p10~p90 区间, 连续几天这样就要查共振是否还独立" \
        if not (-0.182 <= rho <= 0.209) else ""
    log.info("两模型当日分位相关 %+.3f (逐日常态 p10 −0.18 ~ p90 +0.21)%s", rho, flag)

    payload = [{"ts_code": r.ts_code, "trade_date": r.trade_date,
                "ev_rawk": round(float(r.ev_rawk), 6),
                "ev_chips": round(float(r.ev_chips), 6),
                "ev": round(float(r.ev), 6),
                "rank_pct": round(float(r.rank_pct), 5)}
               for r in m.itertuples()]
    async with eng.begin() as c:
        for i in range(0, len(payload), CH):
            st = pg_insert(ComboScore).values(payload[i:i + CH])
            await c.execute(st.on_conflict_do_update(
                index_elements=["ts_code", "trade_date"],
                set_={"ev_rawk": st.excluded.ev_rawk, "ev_chips": st.excluded.ev_chips,
                      "ev": st.excluded.ev, "rank_pct": st.excluded.rank_pct}))
        n = (await c.execute(text(
            "select count(*), min(trade_date), max(trade_date) from combo_score"))).fetchone()
    log.info("写入 %s 条; 表内合计 %s, %s ~ %s",
             f"{len(payload):,}", f"{n[0]:,}", n[1], n[2])
    last = m[m.trade_date == days[-1]]
    top = last[last.rank_pct >= 0.99].nlargest(8, "ev")
    log.info("%s 的 Top: %s", days[-1], ", ".join(
        f"{r.ts_code}({r.ev * 100:+.2f}%)" for r in top.itertuples()))
    await eng.dispose()


def _build_mkt(idx: pd.DataFrame, day_min: int, span: int) -> np.ndarray:
    """(2, 4, 天数) 的大盘数组, 非交易日前值填 —— 与 rawk/prep.py 逐字同一套做法。

    ⚠️ 顺序必须是 [中证1000, 沪深300], 与训练时的 IDX 一致。换了顺序模型拿到的
       就是两个对调的通道, 而且不报错。
    """
    codes = ["000852.SH", "000300.SH"]
    mkt = np.zeros((len(codes), 4, span), np.float32)
    for k, code in enumerate(codes):
        sub = idx[idx.ts_code == code]
        d_ = np.array([(d - EPOCH).days for d in sub["trade_date"]]) - day_min
        v_ = sub[["o", "h", "l", "c"]].to_numpy(np.float32)
        ok = (d_ >= 0) & (d_ < span)
        tmp = np.full((span, 4), np.nan, np.float32)
        tmp[d_[ok]] = v_[ok]
        mkt[k] = pd.DataFrame(tmp).ffill().bfill().to_numpy(np.float32).T
    return mkt


def _score_chunk(px: pd.DataFrame, idx: pd.DataFrame, days: list, nets: list) -> list[dict]:
    """对一批股票的若干交易日做裸K 打分。

    ⚠️⚠️ 通道构造【必须】调 rawk.train.make_batch, 绝不能自己拼。
       第一版我自己拼了一套(price/close_last), 而 make_batch 用的是
       log(price/close_last) —— 成交量是 log1p(v/vm)、MA20 和大盘也都是对数。
       口径不同的话模型拿到的是另一个分布, 输出是垃圾【而且不报错】。
       这正是本项目反复栽的"同一套算法抄两遍"。
       为此这里把每批股票整理成与 panel.npz 相同的扁平结构(ohlcv/day/mkt/day_min),
       再原样喂给 make_batch。
    """
    import sys
    sys.path.insert(0, "/app/scripts")
    from rawk.train import make_batch

    out: list[dict] = []
    # ---- 扁平化: 与 panel.npz 相同的 (N,5) ohlcv + (N,) day, 按股票分段 ----
    px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    ohlcv = px[["o", "h", "l", "c", "v"]].to_numpy(np.float32)
    day_arr = np.array([(d - EPOCH).days for d in px["trade_date"]], np.int64)
    all_days = np.concatenate([day_arr,
                               np.array([(d - EPOCH).days for d in idx["trade_date"]],
                                        np.int64)])
    day_min = int(all_days.min())
    span = int(all_days.max()) - day_min + 2
    mkt = _build_mkt(idx, day_min, span)

    # 每只票各自的行区间, 保证窗口不跨股票
    pos_list, meta = [], []
    for code, g in px.groupby("ts_code", sort=False):
        lo = g.index[0]
        for d in days:
            hit = g.index[g["trade_date"] == d]
            if len(hit) == 0:
                continue
            p = int(hit[0])
            if p - lo + 1 < NBAR:          # 该票历史不足 400 根
                continue
            pos_list.append(p)
            meta.append((code, d))
    if not pos_list:
        return out

    pos_arr = np.array(pos_list, np.int64)
    with torch.no_grad():
        for i in range(0, len(pos_arr), BS):
            xb = make_batch(ohlcv, pos_arr[i:i + BS], nbar=NBAR,
                            day=day_arr, mkt=mkt, day_min=day_min)
            t = torch.from_numpy(xb)
            pr = np.mean([torch.softmax(n(t), -1).numpy() for n in nets], axis=0)
            for j in range(len(pr)):
                code, d = meta[i + j]
                out.append({"ts_code": code, "trade_date": d,
                            "ev_rawk": UP * float(pr[j][1]) - DN * float(pr[j][2]) - COST})
    return out


if __name__ == "__main__":
    asyncio.run(main())
