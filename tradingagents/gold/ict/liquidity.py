"""Where the stops are, and whether they were just taken.

ICT's premise is mechanical, not mystical: retail stops cluster in predictable
places (above equal highs, below equal lows, beyond session extremes), and the
market is drawn to them because that is where size can be filled. So the model
has two jobs here:

* **enumerate the pools** — every price where a cluster of stops plausibly
  rests, ranked by how many touches built it;
* **decide whether one was swept** — a wick through the pool followed by a close
  back inside, within a few bars. Wick-only is not a sweep, it is a breakout
  that has not confirmed yet; the *close* back inside is what says the move was
  rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from tradingagents.gold.intraday.levels import swings

# Equal highs/lows are the same pool if they are within this many 15m-ATRs.
POOL_TOL_ATR = 0.15
# A pierce smaller than this is noise on the level, not a stop run.
MIN_PIERCE_ATR = 0.08
# A bar must have traded back to the level to sweep it: within this many ATRs of
# the pool on the approach side. Without it, a pool $100 away "sweeps" on every
# bar simply because price is above it.
APPROACH_ATR = 0.25
# Pools further than this are context, not triggers.
MAX_POOL_ATR = 12.0

BUY_SIDE = "buy_side"   # stops above the market: swept → bearish
SELL_SIDE = "sell_side"  # stops below the market: swept → bullish

# Which level kinds are real resting liquidity. VWAP and volume profile are
# magnets, not stop clusters, so they are deliberately excluded.
_POOL_KINDS = {
    "prior": "昨日极值",
    "range": "时段极值",
    "swing": "波段点",
    "round": "整数关口",
}


@dataclass(frozen=True)
class Pool:
    price: float
    kind: str
    label: str
    side: str
    touches: int = 1
    note: str = ""

    @property
    def trade_side(self) -> str:
        """The direction a sweep of this pool argues for."""
        return "short" if self.side == BUY_SIDE else "long"

    def distance(self, price: float) -> float:
        return self.price - price


@dataclass
class Sweep:
    pool: Pool
    bar_time: pd.Timestamp
    extreme: float
    pierce: float
    reclaimed: bool
    reclaim_time: pd.Timestamp | None = None
    reclaim_close: float | None = None
    displacement_atr: float = 0.0
    bars_since: int = 0

    @property
    def side(self) -> str:
        return self.pool.trade_side

    @property
    def status(self) -> str:
        if not self.reclaimed:
            return "已扫待回收"
        return "已回收" if self.bars_since <= 3 else "回收已久"


def _cluster(prices: Iterable[float], tol: float, pick: str) -> list[tuple[float, int]]:
    """Group near-identical extremes; ``pick`` decides which price represents
    the cluster (the highest high / lowest low, since that is where stops sit)."""
    ordered = sorted(prices)
    clusters: list[list[float]] = []
    for value in ordered:
        if clusters and value - clusters[-1][-1] <= tol:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    out = []
    for cluster in clusters:
        if len(cluster) >= 2:
            out.append((max(cluster) if pick == "max" else min(cluster), len(cluster)))
    return out


def find_pools(ctx) -> list[Pool]:
    """Every pool the trigger frame can act on, above and below the price."""
    tol = max(POOL_TOL_ATR * ctx.atr, 0.3)
    pools: list[Pool] = []

    def side_of(price: float) -> str:
        # Side is a fact about where price is now, not about the level's origin:
        # an equal high that price has already traded through is support, not
        # resting buy-side liquidity.
        return BUY_SIDE if price > ctx.price else SELL_SIDE

    # 1. Structural levels from the shared level map.
    for level in ctx.levels.levels:
        kind = _POOL_KINDS.get(level.kind)
        if kind is None or level.kind == "round":
            continue
        if level.kind == "prior" and "昨收" in level.name:
            continue  # a close is a reference, not a stop cluster
        pools.append(
            Pool(
                price=level.price,
                kind=level.kind,
                label=level.name,
                side=side_of(level.price),
                touches=1,
                note=level.note,
            )
        )

    # 2. Equal highs / equal lows — the pools ICT traders actually wait for.
    points = swings(ctx.bars15, k=2, lookback=160)
    highs = [p for side, p, _ in points if side == "high"]
    lows = [p for side, p, _ in points if side == "low"]
    for price, touches in _cluster(highs, tol, "max"):
        price = round(price, 2)
        above = price > ctx.price
        label = f"等高·买方流动性（{touches} 触）" if above else f"等高·已破转支撑（{touches} 触）"
        pools.append(Pool(price=price, kind="equal_highs", label=label, side=side_of(price), touches=touches))
    for price, touches in _cluster(lows, tol, "min"):
        price = round(price, 2)
        below = price < ctx.price
        label = f"等低·卖方流动性（{touches} 触）" if below else f"等低·已破转阻力（{touches} 触）"
        pools.append(Pool(price=price, kind="equal_lows", label=label, side=side_of(price), touches=touches))

    # 3. Collapse duplicates: one wall, one entry. Only same-side pools merge —
    # a level above and below price are different trades even at the same price.
    pools = [p for p in pools if abs(p.price - ctx.price) <= MAX_POOL_ATR * ctx.atr]
    pools.sort(key=lambda p: (p.price, p.side))
    merged: list[Pool] = []
    for pool in pools:
        if merged and merged[-1].side == pool.side and abs(pool.price - merged[-1].price) <= max(0.5 * ctx.atr, 0.5):
            prev = merged[-1]
            # Keep the stronger pool, but accumulate the touch count.
            keep = pool if pool.touches > prev.touches else prev
            merged[-1] = Pool(
                price=keep.price,
                kind=keep.kind,
                label=keep.label,
                side=keep.side,
                touches=prev.touches + pool.touches,
                note=keep.note,
            )
        else:
            merged.append(pool)
    return merged


def find_sweeps(
    ctx,
    pools: list[Pool] | None = None,
    *,
    lookback: int = 16,
    reclaim_window: int = 3,
) -> list[Sweep]:
    """Sweeps in the last ``lookback`` 15m bars, newest first.

    A sweep is only useful while it is fresh: the retest it offers decays with
    every bar, which is why ``bars_since`` is part of the record rather than a
    detail left to the caller.
    """
    pools = find_pools(ctx) if pools is None else pools
    bars = ctx.bars15
    if bars.empty:
        return []
    atr15 = ctx.atr
    window = bars.iloc[-lookback:]
    found: list[Sweep] = []

    for pool in pools:
        best: Sweep | None = None
        for offset, (stamp, bar) in enumerate(window.iterrows()):
            # A sweep is a sequence, not a single bar: price approaches the
            # level, pierces it, then closes back inside. The approach may be a
            # bar or two earlier — a displacement candle often gaps away from
            # the level it just took, and testing only that bar's low would
            # miss every fast sweep.
            approach_tol = APPROACH_ATR * atr15
            recent = window.iloc[max(0, offset - 2) : offset + 1]
            if pool.side == BUY_SIDE:
                approached = bool((recent["low"] <= pool.price + approach_tol).any())
                pierced = bar["high"] - pool.price >= MIN_PIERCE_ATR * atr15
            else:
                approached = bool((recent["high"] >= pool.price - approach_tol).any())
                pierced = pool.price - bar["low"] >= MIN_PIERCE_ATR * atr15
            if not (approached and pierced):
                continue

            reclaim = None
            for step in range(0, reclaim_window + 1):
                idx = offset + step
                if idx >= len(window):
                    break
                candidate = window.iloc[idx]
                if pool.side == BUY_SIDE and candidate["close"] < pool.price:
                    reclaim = (idx, candidate)
                    break
                if pool.side == SELL_SIDE and candidate["close"] > pool.price:
                    reclaim = (idx, candidate)
                    break

            if pool.side == BUY_SIDE:
                extreme = float(window.iloc[offset : (reclaim[0] + 1 if reclaim else offset + 1)]["high"].max())
            else:
                extreme = float(window.iloc[offset : (reclaim[0] + 1 if reclaim else offset + 1)]["low"].min())

            sweep = Sweep(
                pool=pool,
                bar_time=stamp,
                extreme=extreme,
                pierce=abs(extreme - pool.price),
                reclaimed=reclaim is not None,
                reclaim_time=window.index[reclaim[0]] if reclaim else None,
                reclaim_close=float(reclaim[1]["close"]) if reclaim else None,
                displacement_atr=(
                    abs(float(reclaim[1]["close"]) - float(reclaim[1]["open"])) / atr15 if reclaim and atr15 else 0.0
                ),
                bars_since=len(window) - 1 - (reclaim[0] if reclaim else offset),
            )
            best = sweep  # later bar wins: the most recent sweep is the live one

        if best is not None:
            found.append(best)

    found.sort(key=lambda s: s.bar_time, reverse=True)
    return found


def nearest(pools: list[Pool], price: float, side: str) -> Pool | None:
    candidates = [p for p in pools if p.side == side]
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs(p.price - price))
