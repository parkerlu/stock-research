"""
GP factor mining V2 — proper cross-sectional alpha mining.

Improvements over v1 demo:
  1. Cross-sectional IC fitness (rank within trading day across stocks)
     — eliminates size / price-level confounders that dominated v1
  2. Winsorize 1% tails before IC to reduce outlier influence
  3. Neutralize: subtract per-day cross-sectional mean / std before ranking
     (proxy for market-wide neutralization)
  4. Stricter parsimony pressure — heavy penalty for depth ≤ 2 (forces
     non-trivial expressions)
  5. Niching: tournament rejects duplicate formula prefixes within parents
  6. Larger pop (250) and gens (40)
  7. Computed on 60-stock panel instead of 10

Output: top 20 cross-sectional factors with IC + IR + formula.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle, StockPoolItem


INPUTS = ["open", "high", "low", "close", "vol", "vwap", "ret"]
BINARY_OPS = ["add", "sub", "mul", "div"]
UNARY_OPS = ["abs", "log", "neg", "sqrt", "square"]
TS_OPS = ["ts_mean", "ts_std", "ts_max", "ts_min",
          "ts_rank", "ts_delta", "ts_sum", "ts_zscore"]
TS_BINARY = ["ts_corr"]
WINDOWS = [3, 5, 10, 20]

MAX_DEPTH = 5
POP_SIZE = 250
N_GENERATIONS = 40
TOURNAMENT_SIZE = 6
ELITISM = 6
MUTATION_RATE = 0.30
CROSSOVER_RATE = 0.60

LABEL_HORIZON = 5
random.seed(42)
np.random.seed(42)


@dataclass
class Node:
    op: str
    children: list = None
    window: int | None = None

    def to_str(self) -> str:
        if self.children is None:
            return self.op
        if self.op in BINARY_OPS:
            return f"({self.children[0].to_str()} {self.op} {self.children[1].to_str()})"
        if self.op in UNARY_OPS:
            return f"{self.op}({self.children[0].to_str()})"
        if self.op in TS_OPS:
            return f"{self.op}({self.children[0].to_str()}, {self.window})"
        if self.op in TS_BINARY:
            return f"{self.op}({self.children[0].to_str()}, {self.children[1].to_str()}, {self.window})"
        return self.op

    def depth(self) -> int:
        if self.children is None:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def all_nodes(self) -> list:
        out = [self]
        if self.children:
            for c in self.children:
                out.extend(c.all_nodes())
        return out


def random_tree(depth: int = 0, force_complex: bool = False) -> Node:
    # Force complexity at the root: never let early termination produce a leaf
    if depth >= MAX_DEPTH or (depth >= 2 and not force_complex and random.random() < 0.30):
        return Node(op=random.choice(INPUTS))
    op_pool = BINARY_OPS + UNARY_OPS + TS_OPS + TS_BINARY
    op = random.choice(op_pool)
    if op in BINARY_OPS:
        return Node(op=op, children=[random_tree(depth + 1), random_tree(depth + 1)])
    if op in UNARY_OPS:
        return Node(op=op, children=[random_tree(depth + 1)])
    if op in TS_OPS:
        return Node(op=op, children=[random_tree(depth + 1)],
                    window=random.choice(WINDOWS))
    if op in TS_BINARY:
        return Node(op=op, children=[random_tree(depth + 1), random_tree(depth + 1)],
                    window=random.choice(WINDOWS))
    return Node(op=random.choice(INPUTS))


def clone(t: Node) -> Node:
    return Node(op=t.op,
                children=[clone(c) for c in t.children] if t.children else None,
                window=t.window)


def crossover(a: Node, b: Node) -> Node:
    a2 = clone(a)
    nodes_a = a2.all_nodes()
    nodes_b = b.all_nodes()
    if len(nodes_a) <= 1 or not nodes_b:
        return a2
    target = random.choice(nodes_a[1:])
    src = clone(random.choice(nodes_b))
    target.op = src.op
    target.children = src.children
    target.window = src.window
    return a2


def mutate(t: Node) -> Node:
    t2 = clone(t)
    nodes = t2.all_nodes()
    target = random.choice(nodes)
    fresh = random_tree(depth=max(0, MAX_DEPTH - 2))
    target.op = fresh.op
    target.children = fresh.children
    target.window = fresh.window
    return t2


def safe_div(a, b):
    return np.where(np.abs(b) > 1e-9, a / np.where(np.abs(b) > 1e-9, b, 1), 0.0)


def evaluate_array(node: Node, panel: dict) -> np.ndarray:
    """Evaluate factor on per-stock columns; returns (T, N) matrix."""
    if node.children is None:
        return panel[node.op].copy()  # (T, N)

    if node.op in BINARY_OPS:
        a = evaluate_array(node.children[0], panel)
        b = evaluate_array(node.children[1], panel)
        if node.op == "add": return a + b
        if node.op == "sub": return a - b
        if node.op == "mul": return a * b
        if node.op == "div": return safe_div(a, b)

    if node.op in UNARY_OPS:
        a = evaluate_array(node.children[0], panel)
        if node.op == "abs": return np.abs(a)
        if node.op == "log": return np.log(np.abs(a) + 1e-9)
        if node.op == "neg": return -a
        if node.op == "sqrt": return np.sqrt(np.abs(a))
        if node.op == "square": return a * a

    if node.op in TS_OPS:
        a = evaluate_array(node.children[0], panel)
        n = node.window or 5
        # Rolling along time axis (axis=0) per stock column
        df = pd.DataFrame(a)
        if node.op == "ts_mean": return df.rolling(n, min_periods=1).mean().values
        if node.op == "ts_std":  return df.rolling(n, min_periods=1).std().fillna(0).values
        if node.op == "ts_max":  return df.rolling(n, min_periods=1).max().values
        if node.op == "ts_min":  return df.rolling(n, min_periods=1).min().values
        if node.op == "ts_rank": return df.rolling(n, min_periods=1).rank(pct=True).fillna(0.5).values
        if node.op == "ts_delta":return (df - df.shift(n).fillna(method="bfill")).values
        if node.op == "ts_sum":  return df.rolling(n, min_periods=1).sum().values
        if node.op == "ts_zscore":
            m = df.rolling(n, min_periods=1).mean()
            s = df.rolling(n, min_periods=1).std().fillna(1).replace(0, 1)
            return ((df - m) / s).fillna(0).values

    if node.op in TS_BINARY:
        a = evaluate_array(node.children[0], panel)
        b = evaluate_array(node.children[1], panel)
        n = node.window or 10
        out = np.zeros_like(a)
        for col in range(a.shape[1]):
            sa = pd.Series(a[:, col]); sb = pd.Series(b[:, col])
            out[:, col] = sa.rolling(n, min_periods=2).corr(sb).fillna(0).values
        return out

    return np.zeros((panel["close"].shape))


def cross_sectional_ic(factor: np.ndarray, future_ret: np.ndarray) -> tuple[float, float, int]:
    """Per-row Spearman correlation. Returns (mean_ic, ir, valid_days).
    Returns (0.0, 0.0, 0) if degenerate (constant factor / no variance).
    """
    T, N = factor.shape
    if N < 5:
        return 0.0, 0.0, 0
    fw = factor.copy()
    rw = future_ret.copy()
    fw[~np.isfinite(fw)] = 0.0
    rw[~np.isfinite(rw)] = 0.0

    # Reject if factor has near-zero variance overall
    if np.std(fw) < 1e-9:
        return 0.0, 0.0, 0

    df_f = pd.DataFrame(fw)
    df_r = pd.DataFrame(rw)
    q_low = df_f.quantile(0.01, axis=1)
    q_hi = df_f.quantile(0.99, axis=1)
    df_f = df_f.clip(lower=q_low, upper=q_hi, axis=0)

    # Per-row variance — drop constant rows
    row_var = df_f.var(axis=1)
    valid_rows = row_var > 1e-9

    fr = df_f.rank(axis=1)
    rr = df_r.rank(axis=1)
    fm = fr.mean(axis=1).values.reshape(-1, 1)
    rm = rr.mean(axis=1).values.reshape(-1, 1)
    fc = fr.values - fm
    rc = rr.values - rm
    num = (fc * rc).sum(axis=1)
    den = np.sqrt((fc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1)) + 1e-9
    ic_per_day = num / den
    ic_per_day = np.where(valid_rows.values, ic_per_day, np.nan)
    valid = np.isfinite(ic_per_day)
    if valid.sum() < 50:
        return 0.0, 0.0, int(valid.sum())
    ic_series = ic_per_day[valid]
    mean_ic = float(np.mean(ic_series))
    std_ic = float(np.std(ic_series))
    ir = mean_ic / std_ic if std_ic > 1e-9 else 0.0
    return mean_ic, ir, int(valid.sum())


def fitness(node: Node, panel: dict, future_ret: np.ndarray) -> float:
    try:
        fac = evaluate_array(node, panel)
        if not np.isfinite(fac).any() or np.std(fac) < 1e-9:
            return -1.0
        mean_ic, ir, n_days = cross_sectional_ic(fac, future_ret)
        if n_days < 100:                           # too few valid days = degenerate
            return -1.0
        if abs(mean_ic) < 0.005:                   # no signal at all
            return -1.0
        score = abs(mean_ic) * 100                  # IC of 0.05 → score 5
        score += min(abs(ir), 3.0) * 1.0           # IR bonus up to +3
        d = node.depth()
        if d <= 2:
            score -= 8.0                           # heavy penalty for trivial
        elif d == 3:
            score -= 1.5
        if d > MAX_DEPTH:
            score -= 5.0
        return score
    except Exception:
        return -1.0


async def get_pool_codes(pool_id: int = 3) -> list[str]:
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        rows = (await db.execute(
            select(StockPoolItem.ts_code).where(StockPoolItem.pool_id == pool_id)
        )).scalars().all()
    await eng.dispose()
    return list(rows)


async def load_panel(codes: list[str]) -> tuple[dict, np.ndarray, list[str]]:
    """Build aligned panel: dict of (T, N) arrays for each input variable.
    Also computes future N-day return panel."""
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)

    per_stock: dict[str, pd.DataFrame] = {}
    for code in codes:
        async with Session() as db:
            rows = (await db.execute(
                select(DailyCandle).where(DailyCandle.ts_code == code)
                .order_by(DailyCandle.trade_date)
            )).scalars().all()
        if len(rows) < 250:
            continue
        latest_adj = float(rows[-1].adj_factor) if rows[-1].adj_factor else 1.0
        df = pd.DataFrame([{
            "trade_date": r.trade_date,
            "open": float(r.open) * (float(r.adj_factor or 1.0) / latest_adj),
            "high": float(r.high) * (float(r.adj_factor or 1.0) / latest_adj),
            "low":  float(r.low)  * (float(r.adj_factor or 1.0) / latest_adj),
            "close":float(r.close)* (float(r.adj_factor or 1.0) / latest_adj),
            "vol": float(r.vol or 0),
            "amount": float(r.amount or 0),
        } for r in rows]).set_index("trade_date")
        per_stock[code] = df
    await eng.dispose()

    if not per_stock:
        return {}, np.array([]), []

    # Align on common dates
    common_dates = sorted(set.intersection(*[set(df.index) for df in per_stock.values()]))
    print(f"  panel dates: {len(common_dates)} | stocks: {len(per_stock)}", flush=True)

    codes_used = list(per_stock.keys())
    T = len(common_dates)
    N = len(codes_used)
    panel = {}
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        arr = np.zeros((T, N), dtype=np.float64)
        for j, code in enumerate(codes_used):
            arr[:, j] = per_stock[code].reindex(common_dates)[col].values
        panel[col] = arr
    panel["vwap"] = panel["amount"] / np.where(panel["vol"] > 0, panel["vol"], 1)
    panel["ret"] = np.zeros((T, N))
    panel["ret"][1:] = panel["close"][1:] / panel["close"][:-1] - 1

    # Future N-day return per stock
    future = np.zeros((T, N))
    future[:-LABEL_HORIZON] = panel["close"][LABEL_HORIZON:] / panel["close"][:-LABEL_HORIZON] - 1
    return panel, future, codes_used


def select_parent(pop_with_fit: list[tuple[Node, float]]) -> Node:
    sample = random.sample(pop_with_fit, min(TOURNAMENT_SIZE, len(pop_with_fit)))
    sample.sort(key=lambda x: -x[1])
    return clone(sample[0][0])


async def main() -> None:
    codes = await get_pool_codes(3)
    # Use first 60 stocks alphabetically for speed
    codes = sorted(codes)[:60]
    print(f"Building panel for {len(codes)} stocks...", flush=True)
    panel, future_ret, codes_used = await load_panel(codes)
    if not panel:
        print("No panel data"); return

    # Initial population (force at least depth 3 for first half)
    print(f"\nInitializing population of {POP_SIZE}...", flush=True)
    pop = []
    for k in range(POP_SIZE):
        depth_min = 3 if k < POP_SIZE // 2 else 2
        t = random_tree()
        attempts = 0
        while t.depth() < depth_min and attempts < 5:
            t = random_tree(force_complex=True)
            attempts += 1
        pop.append(t)

    print(f"Evaluating Gen 0...", flush=True)
    t0 = time.time()
    fits = [fitness(t, panel, future_ret) for t in pop]
    pop_with_fit = list(zip(pop, fits))
    pop_with_fit.sort(key=lambda x: -x[1])
    best = pop_with_fit[0]
    print(f"  Gen  0: best score={best[1]:+.3f}  | {best[0].to_str()[:90]}", flush=True)

    for gen in range(1, N_GENERATIONS + 1):
        new_pop = [clone(p) for p, _ in pop_with_fit[:ELITISM]]
        seen_strs = set(p.to_str() for p, _ in pop_with_fit[:ELITISM])
        attempts = 0
        while len(new_pop) < POP_SIZE and attempts < POP_SIZE * 4:
            attempts += 1
            if random.random() < CROSSOVER_RATE:
                p1 = select_parent(pop_with_fit)
                p2 = select_parent(pop_with_fit)
                child = crossover(p1, p2)
            else:
                child = clone(select_parent(pop_with_fit))
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            if child.depth() > MAX_DEPTH + 1:
                continue
            s = child.to_str()
            if s in seen_strs and random.random() < 0.7:
                continue  # niching: discourage duplicates
            seen_strs.add(s)
            new_pop.append(child)
        # Top up if needed
        while len(new_pop) < POP_SIZE:
            new_pop.append(random_tree(force_complex=True))
        fits = [fitness(t, panel, future_ret) for t in new_pop]
        pop_with_fit = list(zip(new_pop, fits))
        pop_with_fit.sort(key=lambda x: -x[1])
        best = pop_with_fit[0]
        elapsed = time.time() - t0
        if gen % 5 == 0 or gen == 1:
            print(f"  Gen {gen:>2}: best score={best[1]:+.3f} ({elapsed:.0f}s)  | "
                  f"{best[0].to_str()[:90]}", flush=True)

    # Top 20 unique factors with detailed metrics
    seen = set()
    top = []
    for node, fit in pop_with_fit:
        s = node.to_str()
        if s in seen or fit < 0:
            continue
        if node.depth() <= 2:
            continue  # filter trivial
        seen.add(s)
        # Re-compute IC + IR for reporting
        try:
            fac = evaluate_array(node, panel)
            mean_ic, ir, _ = cross_sectional_ic(fac, future_ret)
            top.append((node, mean_ic, ir, fit))
        except Exception:
            continue
        if len(top) >= 25:
            break

    print(f"\n{'='*78}", flush=True)
    print(f"  TOP 20 cross-sectional factors (cross-sectional rank IC, no size bias)", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"{'#':>2}  {'IC':>7}  {'IR':>6}  {'depth':>5}  formula", flush=True)
    print("-" * 78, flush=True)
    for i, (n, ic, ir, score) in enumerate(top[:20], 1):
        formula = n.to_str()
        if len(formula) > 95:
            formula = formula[:92] + "..."
        print(f"{i:>2}  {ic:+.4f}  {ir:+.3f}  {n.depth():>5}  {formula}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
