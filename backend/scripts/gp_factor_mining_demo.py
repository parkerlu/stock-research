"""
GP factor mining demo — small-scale proof of concept.

Evolves mathematical expressions on OHLCV that maximize information
coefficient (IC) with future 5-day return. Outputs top-20 expressions
with their formulas + IC stats.

Run on a 10-stock subset, 30 generations, ~10-15 minutes on M3 Pro.
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
from app.models.schema import DailyCandle


# ============== Genetic Programming ==============

# Inputs (terminals)
INPUTS = ["open", "high", "low", "close", "vol", "vwap", "ret"]

# Element-wise binary ops
BINARY_OPS = ["add", "sub", "mul", "div"]
# Element-wise unary
UNARY_OPS = ["abs", "log", "neg", "sqrt", "square"]
# Time-series ops with window
TS_OPS = ["ts_mean", "ts_std", "ts_max", "ts_min",
          "ts_rank", "ts_delta", "ts_sum", "ts_zscore"]
# Time-series binary
TS_BINARY = ["ts_corr"]

WINDOWS = [3, 5, 10, 20]

MAX_DEPTH = 4
POP_SIZE = 100
N_GENERATIONS = 25
TOURNAMENT_SIZE = 5
ELITISM = 5
MUTATION_RATE = 0.25
CROSSOVER_RATE = 0.65

LABEL_HORIZON = 5  # predict next 5-day max return
random.seed(7)
np.random.seed(7)


@dataclass
class Node:
    op: str                       # input name OR op name
    children: list = None          # child nodes (or None for leaves)
    window: int | None = None      # for ts_ ops

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
        return f"{self.op}(...)"

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


# ============== Tree generation / crossover / mutation ==============

def random_tree(depth: int = 0) -> Node:
    """Recursively grow a random expression tree."""
    if depth >= MAX_DEPTH or (depth >= 1 and random.random() < 0.25):
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


def crossover(a: Node, b: Node) -> Node:
    """Return a copy of a with one of its sub-trees replaced by a sub-tree of b."""
    a2 = clone(a)
    nodes_a = a2.all_nodes()
    nodes_b = b.all_nodes()
    if len(nodes_a) <= 1 or len(nodes_b) == 0:
        return a2
    target = random.choice(nodes_a[1:])  # not root
    src = clone(random.choice(nodes_b))
    target.op = src.op
    target.children = src.children
    target.window = src.window
    return a2


def mutate(t: Node) -> Node:
    """Replace one node with a fresh random subtree."""
    t2 = clone(t)
    nodes = t2.all_nodes()
    target = random.choice(nodes)
    fresh = random_tree(depth=max(0, MAX_DEPTH - 2))
    target.op = fresh.op
    target.children = fresh.children
    target.window = fresh.window
    return t2


def clone(t: Node) -> Node:
    return Node(
        op=t.op,
        children=[clone(c) for c in t.children] if t.children else None,
        window=t.window,
    )


# ============== Expression evaluation ==============

def safe_div(a, b):
    return np.where(np.abs(b) > 1e-9, a / np.where(np.abs(b) > 1e-9, b, 1), 0.0)


def evaluate(node: Node, df: pd.DataFrame) -> np.ndarray:
    if node.children is None:
        if node.op == "vwap":
            return (df["amount"] / df["vol"].replace(0, 1)).values
        if node.op == "ret":
            return df["close"].pct_change().fillna(0).values
        return df[node.op].values.astype(float)

    if node.op in BINARY_OPS:
        a = evaluate(node.children[0], df)
        b = evaluate(node.children[1], df)
        if node.op == "add": return a + b
        if node.op == "sub": return a - b
        if node.op == "mul": return a * b
        if node.op == "div": return safe_div(a, b)

    if node.op in UNARY_OPS:
        a = evaluate(node.children[0], df)
        if node.op == "abs": return np.abs(a)
        if node.op == "log": return np.log(np.abs(a) + 1e-9)
        if node.op == "neg": return -a
        if node.op == "sqrt": return np.sqrt(np.abs(a))
        if node.op == "square": return a * a

    if node.op in TS_OPS:
        a = evaluate(node.children[0], df)
        s = pd.Series(a)
        n = node.window or 5
        if node.op == "ts_mean": return s.rolling(n, min_periods=1).mean().values
        if node.op == "ts_std":  return s.rolling(n, min_periods=1).std().fillna(0).values
        if node.op == "ts_max":  return s.rolling(n, min_periods=1).max().values
        if node.op == "ts_min":  return s.rolling(n, min_periods=1).min().values
        if node.op == "ts_rank": return s.rolling(n, min_periods=1).rank(pct=True).fillna(0.5).values
        if node.op == "ts_delta":return (s - s.shift(n).fillna(method="bfill")).values
        if node.op == "ts_sum":  return s.rolling(n, min_periods=1).sum().values
        if node.op == "ts_zscore":
            m = s.rolling(n, min_periods=1).mean()
            sd = s.rolling(n, min_periods=1).std().fillna(1).replace(0, 1)
            return ((s - m) / sd).fillna(0).values

    if node.op in TS_BINARY:
        a = evaluate(node.children[0], df)
        b = evaluate(node.children[1], df)
        n = node.window or 10
        sa = pd.Series(a); sb = pd.Series(b)
        if node.op == "ts_corr":
            return sa.rolling(n, min_periods=2).corr(sb).fillna(0).values

    return np.zeros(len(df))


# ============== Fitness ==============

def label_future_return(df: pd.DataFrame, horizon: int = LABEL_HORIZON) -> np.ndarray:
    """Future N-day max return (signed)."""
    fut = df["close"].shift(-horizon) / df["close"] - 1
    return fut.fillna(0).values


def fitness_ic(node: Node, panels: list[pd.DataFrame]) -> float:
    """Average rank correlation between factor and future return across stocks."""
    ics = []
    try:
        for df in panels:
            f = evaluate(node, df)
            f = pd.Series(f)
            y = pd.Series(label_future_return(df))
            mask = np.isfinite(f) & np.isfinite(y) & (y != 0)
            if mask.sum() < 50:
                continue
            ic = f[mask].rank().corr(y[mask].rank())
            if not np.isnan(ic):
                ics.append(ic)
    except Exception:
        return -1.0
    if not ics:
        return -1.0
    # Penalize complexity slightly
    return float(np.mean(ics)) - 0.005 * (node.depth() - 1)


# ============== Loop ==============

async def load_data(codes: list[str]) -> list[pd.DataFrame]:
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    out = []
    for code in codes:
        async with Session() as db:
            rows = (await db.execute(
                select(DailyCandle).where(DailyCandle.ts_code == code)
                .order_by(DailyCandle.trade_date)
            )).scalars().all()
        if len(rows) < 200:
            continue
        latest_adj = float(rows[-1].adj_factor) if rows[-1].adj_factor else 1.0
        df = pd.DataFrame([{
            "open": float(r.open) * (float(r.adj_factor or 1.0) / latest_adj),
            "high": float(r.high) * (float(r.adj_factor or 1.0) / latest_adj),
            "low":  float(r.low)  * (float(r.adj_factor or 1.0) / latest_adj),
            "close":float(r.close)* (float(r.adj_factor or 1.0) / latest_adj),
            "vol": float(r.vol or 0),
            "amount": float(r.amount or 0),
        } for r in rows])
        out.append(df)
    await eng.dispose()
    return out


def select_parent(pop_with_fit: list[tuple[Node, float]]) -> Node:
    sample = random.sample(pop_with_fit, min(TOURNAMENT_SIZE, len(pop_with_fit)))
    sample.sort(key=lambda x: -x[1])
    return clone(sample[0][0])


async def main() -> None:
    # Pick 10 stocks for the demo
    codes = ["688111.SH", "600519.SH", "603319.SH", "600104.SH", "000001.SZ",
             "000004.SZ", "000020.SZ", "000034.SZ", "000059.SZ", "000088.SZ"]
    print(f"Loading {len(codes)} stocks...", flush=True)
    panels = await load_data(codes)
    print(f"Loaded {len(panels)} stocks (each ~{len(panels[0]) if panels else 0} bars)\n", flush=True)

    # Initialize population
    pop = [random_tree() for _ in range(POP_SIZE)]
    print(f"Generation  0: evaluating {POP_SIZE} random trees...", flush=True)
    t0 = time.time()
    fits = [fitness_ic(t, panels) for t in pop]
    pop_with_fit = list(zip(pop, fits))
    pop_with_fit.sort(key=lambda x: -x[1])
    best = pop_with_fit[0]
    print(f"             best IC = {best[1]:+.4f}  | {best[0].to_str()[:90]}", flush=True)

    for gen in range(1, N_GENERATIONS + 1):
        new_pop = [clone(p) for p, _ in pop_with_fit[:ELITISM]]
        while len(new_pop) < POP_SIZE:
            if random.random() < CROSSOVER_RATE:
                p1 = select_parent(pop_with_fit)
                p2 = select_parent(pop_with_fit)
                child = crossover(p1, p2)
            else:
                p = select_parent(pop_with_fit)
                child = clone(p)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            if child.depth() > MAX_DEPTH + 1:
                continue
            new_pop.append(child)
        fits = [fitness_ic(t, panels) for t in new_pop]
        pop_with_fit = list(zip(new_pop, fits))
        pop_with_fit.sort(key=lambda x: -x[1])
        best = pop_with_fit[0]
        elapsed = time.time() - t0
        print(f"Generation {gen:>2}: best IC = {best[1]:+.4f}  ({elapsed:.0f}s)  | "
              f"{best[0].to_str()[:90]}", flush=True)

    # Show top 20 unique factors
    seen = set()
    top = []
    for node, fit in pop_with_fit:
        s = node.to_str()
        if s in seen:
            continue
        seen.add(s)
        top.append((node, fit))
        if len(top) >= 20:
            break

    print(f"\n{'='*78}\n  TOP 20 mined factors (by IC)\n{'='*78}", flush=True)
    print(f"{'#':>2}  {'IC':>7}  formula", flush=True)
    print('-' * 78, flush=True)
    for i, (n, f) in enumerate(top, 1):
        formula = n.to_str()
        if len(formula) > 100:
            formula = formula[:97] + "..."
        print(f"{i:>2}  {f:+.4f}  {formula}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
