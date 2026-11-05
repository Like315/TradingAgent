"""The 15m trigger: sweep, reclaim, retest, fixed 1:2.

Target selection is deliberately removed from the model. Everything is quoted
against the stop the sweep itself defines, and the target is exactly twice that
distance — so the trade either reaches 2R or it does not, and there is nothing
to argue about afterwards.

The gates that remain are the ones that keep a 2R model honest:

* the stop must sit beyond the sweep extreme (otherwise the sweep is not the
  invalidation, and the trade has no thesis);
* risk must be 0.3–1.6 ATR wide (too tight gets wicked, too wide is not a sweep
  entry any more);
* the entry band must be within 2.5 ATR of price (a retest that is already gone
  is a fantasy fill);
* an opposing pool inside the first 1.2R is flagged, because that is where the
  move usually stops — the target is not moved, the trader is warned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from tradingagents.gold.intraday.tape import SizingRow, size_position

from tradingagents.gold.ict.bias import Bias
from tradingagents.gold.ict.liquidity import BUY_SIDE, SELL_SIDE, Pool, Sweep

# Retest band around the swept level, in 15m-ATRs.
ENTRY_BAND_ATR = 0.10
# Stop sits this far beyond the sweep extreme.
STOP_BUFFER_ATR = 0.20
MIN_STOP_ATR = 0.30
MAX_STOP_ATR = 1.60
MAX_DRIFT_ATR = 2.50
# How old a reclaim may be before the retest is stale.
MAX_BARS_SINCE_RECLAIM = 6
# Fixed payoff. Not a preference — the whole model is validated against 2R.
FIXED_RR = 2.0


@dataclass
class IctSetup:
    side: str
    pool: Pool
    sweep: Sweep
    entry_low: float
    entry_high: float
    stop: float
    targets: list[float]
    rr: list[float]
    hold_minutes: int
    invalidation: str
    why: str
    score: float
    sizing: list[SizingRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def entry_mid(self) -> float:
        return (self.entry_low + self.entry_high) / 2.0

    @property
    def risk(self) -> float:
        return abs(self.entry_mid - self.stop)

    @property
    def side_cn(self) -> str:
        return "做多" if self.side == "long" else "做空"


def _snap(price: float) -> float:
    """Gold trades in 0.1 ticks; a level quoted to 4 decimals is not an order."""
    return round(price * 10.0) / 10.0


def _hold_minutes(distance: float, atr15: float) -> int:
    # 15m bars, a healthy impulse covers roughly a third of an ATR per bar.
    bars = distance / max(0.35 * atr15, 0.1)
    return int(max(15, min(240, round(bars * 15 / 5) * 5)))


def _opposing_pool(pools: list[Pool], setup_side: str, start: float, target: float) -> Pool | None:
    """The first pool standing between the entry and the target."""
    if setup_side == "long":
        blocking = [p for p in pools if p.side == BUY_SIDE and start < p.price < target]
        return min(blocking, key=lambda p: p.price) if blocking else None
    blocking = [p for p in pools if p.side == SELL_SIDE and target < p.price < start]
    return max(blocking, key=lambda p: p.price) if blocking else None


# Session pools (Asian/London/NY extremes) carry more stops than a lone swing.
SESSION_KINDS = {"session_asian", "session_extreme"}


def build_setups(
    ctx,
    bias: Bias,
    sweeps: list[Sweep],
    pools: list[Pool],
    *,
    equity: float,
    risk_pct: float,
    rejects: list[str] | None = None,
    sessions=None,
) -> list[IctSetup]:
    """Setups that survive the 1H bias and the sweep arithmetic."""
    out: list[IctSetup] = []
    rejects = rejects if rejects is not None else []
    atr = ctx.atr

    if not bias.allowed:
        rejects.append(f"1H 偏置不允许开仓（{bias.label}）")
        return []

    candidates = [
        s
        for s in sweeps
        if s.reclaimed and s.side == bias.direction and s.bars_since <= MAX_BARS_SINCE_RECLAIM
    ]
    if not candidates:
        armed = [s for s in sweeps if not s.reclaimed and s.side == bias.direction]
        if armed:
            rejects.append(f"有扫单但尚未回收（{armed[0].pool.label} {armed[0].pool.price:,.2f}），等 15m 收盘确认")
        else:
            rejects.append(f"最近 {MAX_BARS_SINCE_RECLAIM} 根 15m 内没有与 1H 同向的扫单回收")
        return []

    # Strongest pool first, then freshest, then the cleanest displacement.
    candidates.sort(key=lambda s: (-s.pool.touches, s.bars_since, -s.displacement_atr))

    for sweep in candidates[:2]:
        pool = sweep.pool
        band = ENTRY_BAND_ATR * atr
        if bias.direction == "long":
            entry_low, entry_high = pool.price - band, pool.price + band
            stop = sweep.extreme - STOP_BUFFER_ATR * atr
        else:
            entry_low, entry_high = pool.price - band, pool.price + band
            stop = sweep.extreme + STOP_BUFFER_ATR * atr

        entry_mid = (entry_low + entry_high) / 2.0
        risk = abs(entry_mid - stop)
        if risk <= 0:
            rejects.append(f"{pool.label}：止损与入场重合，跳过")
            continue
        if risk < MIN_STOP_ATR * atr:
            rejects.append(f"{pool.label}：止损只有 {risk / atr:.2f}×ATR（<{MIN_STOP_ATR}），会被噪音扫掉")
            continue
        if risk > MAX_STOP_ATR * atr:
            rejects.append(f"{pool.label}：止损宽达 {risk / atr:.2f}×ATR（>{MAX_STOP_ATR}），已不是扫单入场")
            continue
        drift = abs(entry_mid - ctx.price) / atr
        if drift > MAX_DRIFT_ATR:
            rejects.append(f"{pool.label}：入场区距现价 {drift:.2f}×ATR，回踩已不可及")
            continue

        target = entry_mid + FIXED_RR * risk if bias.direction == "long" else entry_mid - FIXED_RR * risk
        warnings: list[str] = []
        blocker = _opposing_pool(pools, bias.direction, entry_mid, target)
        if blocker is not None:
            room = abs(blocker.price - entry_mid) / risk
            warnings.append(f"前方 {blocker.label} {blocker.price:,.2f} 距入场 {room:.2f}R，2R 目标可能被它挡住")
            if room < 1.2:
                rejects.append(f"{pool.label}：前方流动性只有 {room:.2f}R，风险回报不成立")
                continue

        if sweep.displacement_atr < 0.5:
            warnings.append(f"回收 K 线实体只有 {sweep.displacement_atr:.2f}×ATR，位移偏弱")

        position_bonus = (
            (1.0 - (bias.position_pct or 0.5)) if bias.direction == "long" else (bias.position_pct or 0.5)
        )
        session_bonus = 0.1 if pool.kind in SESSION_KINDS else 0.0
        kz = sessions.killzone if sessions is not None else None
        kz_bonus = 0.1 if kz else 0.0
        score = (
            0.35 * min(sweep.pool.touches, 3) / 3.0
            + 0.25 * min(sweep.displacement_atr / 1.0, 1.0)
            + 0.25 * max(0.0, min(1.0, position_bonus))
            + session_bonus
            + kz_bonus
        )
        if kz:
            warnings.append(f"当前处于 {kz[1]}（SMC 入场窗口）")
        elif sessions is not None and sessions.asian.complete:
            warnings.append("非 killzone 时段：SMC 信号质量打折，SMC 传统上只在这三个窗口入场")
        if pool.kind in SESSION_KINDS:
            warnings.append("扫的是时段极值（亚洲/伦敦/纽约），止损密度高于普通波段点")
        if bias.origin == "session":
            warnings.append("方向来自 SMC session override（PO3），有效期到下一根 1H 收盘")

        why = (
            f"1H {bias.label}；{pool.label} {pool.price:,.2f} 被扫 "
            f"{sweep.pierce:.2f} 美元后收回，回收 K 线实体 {sweep.displacement_atr:.2f}×ATR"
        )
        invalidation = (
            f"15m 收盘重新站回 {sweep.extreme:,.2f} 之外（扫单被确认成突破），或入场后 {max(30, _hold_minutes(abs(target - entry_mid), atr))} 分钟未到 1R"
        )
        setup = IctSetup(
            side=bias.direction,
            pool=pool,
            sweep=sweep,
            entry_low=_snap(entry_low),
            entry_high=_snap(entry_high),
            stop=_snap(stop),
            targets=[_snap(target)],
            rr=[FIXED_RR],
            hold_minutes=_hold_minutes(abs(target - entry_mid), atr),
            invalidation=invalidation,
            why=why,
            score=round(score, 2),
            warnings=warnings,
        )
        setup.sizing = size_position(setup.risk, equity, (risk_pct,))
        out.append(setup)

    if not out and not rejects:
        rejects.append("有扫单回收，但没有任何一条通过止损宽度/回踩距离闸门")
    return out
