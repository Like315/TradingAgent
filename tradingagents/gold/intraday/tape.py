"""Regime read, scalp setups and position sizing.

Three things a scalper needs that a daily report cannot give:

* **a regime gate.** Trend continuation and mean reversion are opposite trades on
  the same level; which one you are allowed to take is decided by the efficiency
  ratio and the volatility percentile, not by opinion;
* **numbers, not adjectives.** Every setup carries an entry *zone* (not a price —
  you get filled somewhere in a range), a stop in dollars and ticks, two targets
  with their R multiples, an expected holding time and a written invalidation;
* **a size that makes the stop survivable.** Risk per trade is fixed first, then
  the stop distance sets the size. Never the other way round.

The cost model is explicit because it is the difference between a scalping
strategy and a donation: at $4,000 gold, a two-tick spread plus commission plus
one tick of slippage is about $0.35/oz round trip, which is 6% of a $6 target.
A setup whose T1 is under 1.2R after costs is discarded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.gold.intraday import feed
from tradingagents.gold.intraday.clock import SessionClock
from tradingagents.gold.intraday.levels import LevelMap, atr, efficiency_ratio, ema, volume_z

# Round-trip friction per ounce, decomposed so it can be argued with:
#   2 ticks spread (0.20) + 1 tick slippage (0.10) + $5/contract commission
#   on 100 oz (0.05) = 0.35 USD/oz.
ROUND_TRIP_COST = 0.35
TICK = 0.10
# A scalp that cannot clear 1.2R after costs is not worth the attention.
MIN_RR_AFTER_COST = 1.2
# Gold futures per contract, and the micro, in ounces.
OZ_PER_CONTRACT = 100.0
OZ_PER_MICRO = 10.0

# 5-minute bars in a year: 252 sessions x 23 hours x 12 bars.
_BARS_PER_YEAR_5M = 252 * 23 * 12


@dataclass
class Regime:
    code: str  # trend_up | trend_down | range | chaos | dead | mixed
    label: str
    er5: float
    er15: float
    ema_align: str
    slope_atr: float
    rv_1h: float
    vol_pct: float
    vol_basis: str
    atr_expansion: float
    dist_sigma: float
    volume_state: str
    bias: float  # -1 (short) .. +1 (long), a summary of the above
    notes: list[str] = field(default_factory=list)

    @property
    def trending(self) -> bool:
        return self.code in ("trend_up", "trend_down")


@dataclass
class SizingRow:
    risk_pct: float
    risk_usd: float
    ounces: float
    contracts: float
    micros: float
    lots: float


@dataclass
class Setup:
    name: str
    side: str  # long | short
    kind: str
    trigger: str
    entry_low: float
    entry_high: float
    stop: float
    targets: list[float]
    rr: list[float]
    hold_minutes: int
    invalidation: str
    why: str
    refs: list[str] = field(default_factory=list)
    gates: list[str] = field(default_factory=list)
    score: float = 0.0
    sizing: list[SizingRow] = field(default_factory=list)
    cost_pct: float = 0.0

    @property
    def entry_mid(self) -> float:
        return (self.entry_low + self.entry_high) / 2.0

    @property
    def stop_dollars(self) -> float:
        return abs(self.entry_mid - self.stop)

    @property
    def stop_ticks(self) -> int:
        return int(round(self.stop_dollars / TICK))


# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #


def _realized_vol_series(five_min: pd.DataFrame, window: int = 12) -> pd.Series:
    returns = np.log(five_min["close"]).diff()
    return returns.rolling(window).std(ddof=0) * math.sqrt(_BARS_PER_YEAR_5M)


def read_regime(five_min: pd.DataFrame, fifteen_min: pd.DataFrame, level_map: LevelMap) -> Regime:
    notes: list[str] = []
    close5 = five_min["close"]
    er5 = efficiency_ratio(close5, 20)
    er15 = efficiency_ratio(fifteen_min["close"], 20) if len(fifteen_min) > 21 else er5

    # EMA alignment on 15m: the higher-timeframe veto on a 5m signal.
    if len(fifteen_min) > 55:
        fast = float(ema(fifteen_min["close"], 20).iloc[-1])
        slow = float(ema(fifteen_min["close"], 50).iloc[-1])
        last = float(fifteen_min["close"].iloc[-1])
        if last > fast > slow:
            align = "多头排列"
        elif last < fast < slow:
            align = "空头排列"
        else:
            align = "均线纠缠"
    else:
        align = "样本不足"
        notes.append("15 分钟线不足 55 根，均线排列不可用")

    # Where price sits in ATR units over the last 20 bars — the size of the move.
    base = float(close5.iloc[-21]) if len(close5) > 21 else float(close5.iloc[0])
    slope_atr = (float(close5.iloc[-1]) - base) / level_map.atr5 if level_map.atr5 else 0.0

    rv_series = _realized_vol_series(five_min).dropna()
    rv_1h = float(rv_series.iloc[-1]) * 100.0 if not rv_series.empty else 0.0
    vol_basis = "5 日"
    if len(rv_series) > 60:
        vol_pct = float((rv_series.iloc[-1] >= rv_series).mean() * 100.0)
    else:
        vol_pct = 50.0
        vol_basis = "样本不足"
        notes.append("波动率分位样本不足，按 50% 中性处理")
    # A percentile over a single session is still useful (it says "quiet or wild
    # relative to today"), but it must not be sold as a five-day percentile.
    if len(rv_series) and len(feed.session_frames(five_min)) < 3:
        vol_basis = "当日"

    # Is the range expanding or contracting? atr(now) vs atr(the previous hour).
    expansion = 1.0
    if len(five_min) > 90:
        recent = atr(five_min.iloc[-30:])
        prior = atr(five_min.iloc[-90:-30])
        if prior:
            expansion = recent / prior

    sigma_move = 0.0
    if level_map.vwap and level_map.vwap_sigma:
        sigma_move = (level_map.price - level_map.vwap) / level_map.vwap_sigma

    v_z = volume_z(five_min)
    if v_z >= 1.2:
        volume_state = "放量"
    elif v_z <= -0.8:
        volume_state = "缩量"
    else:
        volume_state = "常量"

    # Classification order matters: chaos and dead trump trend, because both
    # make a clean trend signal unfillable.
    if vol_pct >= 97.0 and er5 < 0.28:
        code, label = "chaos", "无序高波动（数据行情/突发）"
        notes.append("当前小时波动处于 5 日极值，止损会被随机扫掉")
    elif vol_pct <= 8.0 and er5 < 0.25:
        code, label = "dead", "缩量停摆"
        notes.append("波动率极低，目标走不到，进场只是付点差")
    elif er5 >= 0.30 and abs(slope_atr) >= 0.8 and align in ("多头排列", "空头排列") and expansion >= 0.9:
        if slope_atr > 0 and align == "多头排列":
            code, label = "trend_up", "趋势上行（回踩做多）"
        elif slope_atr < 0 and align == "空头排列":
            code, label = "trend_down", "趋势下行（反抽做空）"
        else:
            code, label = "mixed", "方向不一致（观望）"
            notes.append("5 分钟趋势与 15 分钟均线排列相反，等两者一致")
    elif er5 <= 0.22 and abs(slope_atr) <= 1.2:
        code, label = "range", "区间震荡（高抛低吸）"
    else:
        code, label = "mixed", "过渡状态（等结构）"

    if expansion >= 1.35:
        notes.append("ATR 相对前一小时放大 35% 以上，止损要放宽或等收敛")
    if abs(sigma_move) >= 2.0:
        notes.append(f"价格离 VWAP {sigma_move:+.1f}σ，已在均值回归区，追单风险高")

    # A readable -1..1 bias built from four independent votes.
    votes = [
        (level_map.day_open is not None) and (1.0 if level_map.price > level_map.day_open else -1.0),
        (level_map.vwap is not None) and (1.0 if level_map.price > level_map.vwap else -1.0),
        1.0 if align == "多头排列" else (-1.0 if align == "空头排列" else 0.0),
        max(-1.0, min(1.0, slope_atr / 2.0)),
    ]
    bias = sum(v for v in votes if v is not False) / 4.0

    return Regime(
        code=code,
        label=label,
        er5=er5,
        er15=er15,
        ema_align=align,
        slope_atr=slope_atr,
        rv_1h=rv_1h,
        vol_pct=vol_pct,
        vol_basis=vol_basis,
        atr_expansion=expansion,
        dist_sigma=sigma_move,
        volume_state=volume_state,
        bias=round(bias, 3),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #


def size_position(stop_dollars: float, equity: float, risk_levels: tuple[float, ...] = (0.25, 0.5, 1.0)) -> list[SizingRow]:
    """Ounces and contracts for a fixed dollar risk and a given stop width.

    ``stop_dollars`` is per ounce (a $5 gold move is $5/oz, $500/contract). The
    arithmetic is deliberately dull: risk / stop = ounces, and everything else
    follows from the contract size.
    """
    if stop_dollars <= 0 or equity <= 0:
        return []
    rows = []
    for pct in risk_levels:
        risk_usd = equity * pct / 100.0
        ounces = risk_usd / stop_dollars
        rows.append(
            SizingRow(
                risk_pct=pct,
                risk_usd=round(risk_usd, 2),
                ounces=round(ounces, 2),
                contracts=round(ounces / OZ_PER_CONTRACT, 2),
                micros=round(ounces / OZ_PER_MICRO, 2),
                lots=round(ounces / OZ_PER_CONTRACT, 2),
            )
        )
    return rows


def _estimate_hold(distance: float, atr5: float) -> int:
    """Minutes to reach a target, from how far a 5m bar typically travels.

    One ATR per bar is optimistic for a stretch of consecutive bars, so the
    estimate assumes a bar covers 0.7 ATR and rounds to 5 minutes.
    """
    if atr5 <= 0:
        return 30
    bars = distance / (atr5 * 0.7)
    return max(5, int(round(bars * 5 / 5.0) * 5))


def _rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    return 0.0 if risk <= 0 else abs(target - entry) / risk


def _snap(price: float) -> float:
    return round(round(price / TICK) * TICK, 2)


# --------------------------------------------------------------------------- #
# Setups
# --------------------------------------------------------------------------- #


def _support_below(lm: LevelMap, kinds: tuple[str, ...] = ("vwap", "swing", "prior", "profile", "range")) -> list:
    return sorted(
        [lv for lv in lm.levels if lv.price < lm.price - 0.2 and lv.kind in kinds],
        key=lambda lv: lm.price - lv.price,
    )


def _resistance_above(lm: LevelMap, kinds: tuple[str, ...] = ("vwap", "swing", "prior", "profile", "range", "round")) -> list:
    return sorted(
        [lv for lv in lm.levels if lv.price > lm.price + 0.2 and lv.kind in kinds],
        key=lambda lv: lv.price - lm.price,
    )


def _pick_targets(entry: float, stop: float, candidates: list[float], minimum_rr: float, min_gap: float = 0.8) -> list[float]:
    """First two levels beyond ``minimum_rr``, padded with R multiples if absent.

    Using real levels as targets rather than fixed R multiples matters: gold
    stops exactly where yesterday's low sits, so a 2R target that is 30 cents
    short of it gets missed by a tick.

    ``min_gap`` keeps T1 and T2 from both landing on the same cluster of levels —
    two targets a dollar apart is one target with extra paperwork.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return []
    direction = 1.0 if entry > stop else -1.0
    out: list[float] = []
    for price in sorted(candidates, key=lambda p: abs(p - entry)):
        if direction * (price - entry) < risk * minimum_rr:
            continue
        if any(abs(price - existing) < min_gap for existing in out):
            continue
        out.append(_snap(price))
        if len(out) == 2:
            break
    if len(out) < 2:
        base = out[0] if out else entry + direction * risk * minimum_rr
        extra = base + direction * max(risk, min_gap)
        if len(out) < 1:
            out.append(_snap(base))
        out.append(_snap(extra))
    return out[:2]


def build_setups(
    lm: LevelMap,
    regime: Regime,
    clock: SessionClock,
    *,
    equity: float,
    five_min: pd.DataFrame,
    has_volume: bool = True,
    rejects: list[str] | None = None,
) -> list[Setup]:
    """Every setup whose gates pass right now, best first.

    Returns an empty list when the clock or the regime forbids trading; the
    caller shows :attr:`SessionClock.blockers` and :attr:`Regime.notes` so "no
    trade" always comes with a reason.
    """
    rejects = rejects if rejects is not None else []

    def reject(message: str) -> None:
        if message not in rejects:
            rejects.append(message)

    if not clock.tradable:
        reject("时段不允许开仓：" + "；".join(clock.blockers))
        return []
    if regime.code in ("chaos", "dead"):
        reject(f"状态不允许开仓：{regime.label}")
        return []
    if lm.atr5 <= 0:
        reject("ATR 为 0，止损无法定位")
        return []

    price = lm.price
    atr5 = lm.atr5
    setups: list[Setup] = []
    supports = _support_below(lm)
    resistances = _resistance_above(lm)

    # ---- 1. trend pullback: the highest-expectancy scalp in a trending tape ----
    if regime.code in ("trend_up", "trend_down"):
        long_side = regime.code == "trend_up"
        shelf = supports[0] if long_side else (resistances[0] if resistances else None)
        if shelf is not None:
            # Buy the shelf, not the breakout: the entry zone starts at the level
            # and extends a third of an ATR into it, which is where a retest
            # actually gets filled.
            if long_side:
                entry_low, entry_high = shelf.price, shelf.price + 0.3 * atr5
                stop = shelf.price - 0.7 * atr5
                candidates = [lv.price for lv in resistances]
                minimum = 1.0
            else:
                entry_low, entry_high = shelf.price - 0.3 * atr5, shelf.price
                stop = shelf.price + 0.7 * atr5
                candidates = [lv.price for lv in supports]
                minimum = 1.0
            entry = (entry_low + entry_high) / 2.0
            targets = _pick_targets(entry, stop, candidates, minimum, min_gap=max(0.8, 0.6 * atr5))
            if targets and _rr(entry, stop, targets[0]) < MIN_RR_AFTER_COST:
                reject(f"趋势单：{shelf.name} 到目标 {targets[0]:,.2f} 只有 {_rr(entry, stop, targets[0]):.2f}R，达不到 {MIN_RR_AFTER_COST}R")
            if targets:
                rr = [_rr(entry, stop, t) for t in targets]
                if rr[0] >= MIN_RR_AFTER_COST:
                    setups.append(
                        Setup(
                            name=f"{'回踩做多' if long_side else '反抽做空'}：{'支撑' if long_side else '阻力'} {shelf.name}",
                            side="long" if long_side else "short",
                            kind="trend_pullback",
                            trigger=(
                                f"{'5 分钟线触及' if long_side else '5 分钟线反抽至'} {shelf.price:,.1f} 后"
                                f"{'出现下影线/吞没' if long_side else '出现上影线/吞没'}，且 1 分钟不再创新"
                                f"{'低' if long_side else '高'}"
                            ),
                            entry_low=_snap(entry_low),
                            entry_high=_snap(entry_high),
                            stop=_snap(stop),
                            targets=targets,
                            rr=[round(r, 2) for r in rr],
                            hold_minutes=_estimate_hold(abs(targets[0] - entry), atr5),
                            invalidation=(
                                f"5 分钟收盘{'跌破' if long_side else '站上'} {_snap(stop):,.1f}"
                                f"（= 结构失效），或 {max(10, _estimate_hold(abs(targets[0] - entry), atr5) // 2)} 分钟内无跟随"
                            ),
                            why=(
                                f"ER(5m) {regime.er5:.2f}、15m {regime.ema_align}、"
                                f"20 根净移动 {regime.slope_atr:+.1f} ATR，趋势成立；"
                                f"{shelf.name} 是回踩位，止损 {abs(entry - stop):.1f} 美元只占 {abs(entry - stop) / atr5:.2f} ATR"
                            ),
                            refs=[shelf.name],
                            gates=["趋势状态", "回踩不破", "RR≥1.2"],
                            score=0.85,
                        )
                    )

    # ---- 2. range fade: sell the wall, buy the floor ----
    if regime.code in ("range", "mixed"):
        for side, pool in (("short", resistances), ("long", supports)):
            if not pool:
                continue
            wall = pool[0]
            side_name = "阻力" if side == "short" else "支撑"
            if abs(wall.distance) > 0.55 * atr5:
                # not at the wall yet, no edge in anticipating it
                reject(f"区间单：最近{side_name} {wall.name} {wall.price:,.2f} 距现价 {wall.atr:+.2f} ATR，超过贴墙阈值 0.55")
                continue
            structural = wall.kind in ("prior", "profile", "range")
            if wall.confluence < 2 and not structural:
                reject(f"区间单：{side_name} {wall.name} 只有 {wall.confluence} 层价位，不构成墙")
                continue
            # Fading a level while the 15-minute trend points the other way is
            # the classic "picking up pennies" trade: allowed only at a real
            # structural extreme, never on a mere touch of a mean-reversion band.
            against_trend = (side == "short" and regime.ema_align == "多头排列") or (
                side == "long" and regime.ema_align == "空头排列"
            )
            if against_trend:
                # Sitting right under a level while the higher timeframe points
                # down is how a fade becomes a knife catch: the level has to be
                # genuinely extended, and it has to be a level worth defending.
                if abs(wall.atr) < 0.6:
                    reject(f"区间单：{side_name} {wall.name} 距现价仅 {wall.atr:+.2f} ATR，且 15m {regime.ema_align}，逆势不接")
                    continue
                if wall.confluence < 2 and wall.kind not in ("prior", "profile"):
                    reject(f"区间单：逆势且 {side_name} 只有 {wall.confluence} 层价位叠加，不够强")
                    continue
            if side == "short":
                entry_low, entry_high = wall.price - 0.2 * atr5, wall.price + 0.05 * atr5
                stop = wall.price + 0.65 * atr5
                candidates = [lm.vwap] if lm.vwap else []
                candidates += [lv.price for lv in supports]
                candidates += [lm.prior.poc] if lm.prior else []
            else:
                entry_low, entry_high = wall.price - 0.05 * atr5, wall.price + 0.2 * atr5
                stop = wall.price - 0.65 * atr5
                candidates = [lm.vwap] if lm.vwap else []
                candidates += [lv.price for lv in resistances]
                candidates += [lm.prior.poc] if lm.prior else []
            entry = (entry_low + entry_high) / 2.0
            targets = _pick_targets(entry, stop, [c for c in candidates if c], 0.9, min_gap=max(0.8, 0.6 * atr5))
            if not targets:
                continue
            rr = [_rr(entry, stop, t) for t in targets]
            if rr[0] < MIN_RR_AFTER_COST:
                continue
            setups.append(
                Setup(
                    name=f"区间{'高抛' if side == 'short' else '低吸'}：{wall.name}",
                    side=side,
                    kind="range_fade",
                    trigger=(
                        f"价格触及 {wall.price:,.1f} 时出现{'上影线/滞涨' if side == 'short' else '下影线/止跌'}"
                        f"，且 1 分钟连续 2 根不创新{'高' if side == 'short' else '低'}"
                    ),
                    entry_low=_snap(entry_low),
                    entry_high=_snap(entry_high),
                    stop=_snap(stop),
                    targets=targets,
                    rr=[round(r, 2) for r in rr],
                    hold_minutes=_estimate_hold(abs(targets[0] - entry), atr5),
                    invalidation=(
                        f"5 分钟收盘{'站上' if side == 'short' else '跌破'} {_snap(stop):,.1f}"
                        f"，或放量（量能 z>1.5）突破 {wall.price:,.1f}"
                    ),
                    why=(
                        f"ER(5m) {regime.er5:.2f}、净移动 {regime.slope_atr:+.1f} ATR，是区间不是趋势；"
                        f"{wall.name} 距现价 {wall.distance:+.1f}（{wall.atr:+.2f} ATR），"
                        f"{'叠加 ' + str(wall.confluence) + ' 层价位' if wall.confluence > 1 else '单一结构位'}，"
                        f"回归目标 {'VWAP/POC' if lm.vwap or lm.prior else '对侧区间'} "
                    ),
                    refs=[wall.name],
                    gates=["区间状态", "贴墙", "RR≥1.2"],
                    score=0.7 + (0.05 if wall.confluence > 1 else 0.0),
                )
            )
            break  # one fade per side, best wall only

    # ---- 3. breakout retest: only with volume, only after a clean sweep ----
    reference = None
    if lm.session_high and abs(price - lm.session_high) < 1.2 * atr5:
        reference = ("high", lm.session_high, "当日高")
    if lm.prior and abs(price - lm.prior.high) < 1.2 * atr5:
        candidate = ("high", lm.prior.high, "昨日高")
        if reference is None or abs(price - lm.prior.high) < abs(price - reference[1]):
            reference = candidate
    if lm.prior and abs(price - lm.prior.low) < 1.2 * atr5 and (reference is None or price < lm.prior.low + atr5):
        if price < (lm.session_low or price):
            reference = ("low", lm.prior.low, "昨日低")
    if reference is not None:
        direction, level_price, level_name = reference
        vz = volume_z(five_min) if has_volume else 0.0
        broken = (price > level_price) if direction == "high" else (price < level_price)
        if not has_volume:
            reject("突破单：当前数据源无成交量，量能确认不可用，此类形态自动弃用")
        elif broken and vz < 1.2:
            reject(f"突破单：已破 {level_name} 但量能 z={vz:.1f} < 1.2，视为假突破")
        if broken and has_volume and vz >= 1.2:
            measure = 0.0
            if lm.asian_high and lm.asian_low:
                measure = abs(lm.asian_high - lm.asian_low)
            projection = max(measure, 1.5 * atr5)
            if direction == "high":
                entry_low, entry_high = level_price, level_price + 0.25 * atr5
                stop = level_price - 0.7 * atr5
                targets = [_snap(level_price + projection), _snap(level_price + projection + 1.0 * atr5)]
            else:
                entry_low, entry_high = level_price - 0.25 * atr5, level_price
                stop = level_price + 0.7 * atr5
                targets = [_snap(level_price - projection), _snap(level_price - projection - 1.0 * atr5)]
            entry = (entry_low + entry_high) / 2.0
            rr = [_rr(entry, stop, t) for t in targets]
            if rr[0] >= MIN_RR_AFTER_COST:
                setups.append(
                    Setup(
                        name=f"突破回踩做{'多' if direction == 'high' else '空'}：{level_name} {level_price:,.1f}",
                        side="long" if direction == "high" else "short",
                        kind="breakout_retest",
                        trigger=(
                            f"突破后回踩 {level_price:,.1f} 不破（1 分钟收盘回到该位"
                            f"{'上方' if direction == 'high' else '下方'}），量能保持 z>1.0"
                        ),
                        entry_low=_snap(entry_low),
                        entry_high=_snap(entry_high),
                        stop=_snap(stop),
                        targets=targets,
                        rr=[round(r, 2) for r in rr],
                        hold_minutes=_estimate_hold(abs(targets[0] - entry), atr5),
                        invalidation=f"1 分钟收盘回到 {level_price:,.1f} {'下方' if direction == 'high' else '上方'}（假突破）",
                        why=(
                            f"已破 {level_name} 且量能 z={vz:.1f}；等回踩确认而不是追第一根，"
                            f"量度目标取亚洲区间高度 {measure:.1f} 与 1.5 ATR 的较大者"
                        ),
                        refs=[level_name],
                        gates=["突破+放量", "回踩确认", "RR≥1.2"],
                        score=0.75,
                    )
                )

    # ---- 4. stop-run reversal: the sweep that fails ----
    if lm.prior and len(five_min) > 8:
        recent = five_min.iloc[-6:]
        for direction, level_price, level_name in (
            ("high", lm.prior.high, "昨日高"),
            ("high", lm.session_high or 0.0, "当日高"),
            ("low", lm.prior.low, "昨日低"),
            ("low", lm.session_low or 0.0, "当日低"),
        ):
            if not level_price:
                continue
            if direction == "high":
                swept = bool((recent["high"] > level_price).any()) and float(five_min["close"].iloc[-1]) < level_price
                if not swept:
                    continue
                sweep_extreme = float(recent["high"].max())
                entry_low, entry_high = price - 0.1 * atr5, price
                stop = sweep_extreme + 0.3 * atr5
                candidates = [lm.vwap] if lm.vwap else []
                candidates += [lv.price for lv in supports]
                side = "short"
            else:
                swept = bool((recent["low"] < level_price).any()) and float(five_min["close"].iloc[-1]) > level_price
                if not swept:
                    continue
                sweep_extreme = float(recent["low"].min())
                entry_low, entry_high = price, price + 0.1 * atr5
                stop = sweep_extreme - 0.3 * atr5
                candidates = [lm.vwap] if lm.vwap else []
                candidates += [lv.price for lv in resistances]
                side = "long"
            entry = (entry_low + entry_high) / 2.0
            targets = _pick_targets(entry, stop, [c for c in candidates if c], 0.9, min_gap=max(0.8, 0.6 * atr5))
            if not targets:
                continue
            rr = [_rr(entry, stop, t) for t in targets]
            if rr[0] < MIN_RR_AFTER_COST:
                continue
            setups.append(
                Setup(
                    name=f"假突破反转（扫损）{'做空' if side == 'short' else '做多'}：{level_name}",
                    side=side,
                    kind="sweep_reversal",
                    trigger=(
                        f"{level_name} {level_price:,.1f} 被扫到 {sweep_extreme:,.1f} 后 1 分钟收盘回到"
                        f"{'下方' if side == 'short' else '上方'}，随后不再创新{'高' if side == 'short' else '低'}"
                    ),
                    entry_low=_snap(min(entry_low, entry_high)),
                    entry_high=_snap(max(entry_low, entry_high)),
                    stop=_snap(stop),
                    targets=targets,
                    rr=[round(r, 2) for r in rr],
                    hold_minutes=_estimate_hold(abs(targets[0] - entry), atr5),
                    invalidation=f"价格再次{'站上' if side == 'short' else '跌破'} {sweep_extreme:,.1f}（扫损变成真突破）",
                    why=(
                        f"最近 30 分钟 {level_name} 被扫到 {sweep_extreme:,.1f} 但收盘收回，"
                        f"止损单已被吃、追单被套，反向回归 VWAP 的赔率最好；止损放在扫损极值外 0.3 ATR"
                    ),
                    refs=[level_name],
                    gates=["扫损收回", "RR≥1.2"],
                    score=0.8,
                )
            )
            break

    # ---- 5. VWAP exhaustion fade: only at 2σ with decelerating bars ----
    if lm.vwap and lm.vwap_sigma and abs(regime.dist_sigma) < 2.0:
        reject(f"VWAP 回归：偏离只有 {regime.dist_sigma:+.2f}σ，未达 2σ")
    if lm.vwap and lm.vwap_sigma and abs(regime.dist_sigma) >= 2.0:
        side = "short" if regime.dist_sigma > 0 else "long"
        ranges = (five_min["high"] - five_min["low"]).iloc[-3:]
        decelerating = float(ranges.mean()) < 0.8 * atr5
        if decelerating:
            if side == "short":
                entry_low, entry_high = price - 0.15 * atr5, price
                stop = float(five_min["high"].iloc[-4:].max()) + 0.4 * atr5
            else:
                entry_low, entry_high = price, price + 0.15 * atr5
                stop = float(five_min["low"].iloc[-4:].min()) - 0.4 * atr5
            entry = (entry_low + entry_high) / 2.0
            targets = [_snap(lm.vwap), _snap(lm.vwap + (0.5 if side == "short" else -0.5) * (lm.vwap_sigma or 0))]
            rr = [_rr(entry, stop, t) for t in targets]
            if all(abs(t - entry) > 0.5 for t in targets) and rr[0] >= MIN_RR_AFTER_COST:
                setups.append(
                    Setup(
                        name=f"VWAP 均值回归（{'做空' if side == 'short' else '做多'}）",
                        side=side,
                        kind="vwap_fade",
                        trigger=f"{abs(regime.dist_sigma):.1f}σ 外连续 3 根 5 分钟 K 收缩，且出现反向吞没",
                        entry_low=_snap(entry_low),
                        entry_high=_snap(entry_high),
                        stop=_snap(stop),
                        targets=targets,
                        rr=[round(r, 2) for r in rr],
                        hold_minutes=_estimate_hold(abs(targets[0] - entry), atr5),
                        invalidation=f"再创新{'高' if side == 'short' else '低'}且 5 分钟量能 z>1.5（趋势加速）",
                        why=(
                            f"价格偏离 VWAP {regime.dist_sigma:+.1f}σ，同时 5 分钟 K 线实体收缩，"
                            f"追单动能衰竭；目标就是 VWAP 本身，赔率靠 σ 的宽度"
                        ),
                        refs=["当日 VWAP"],
                        gates=["偏离≥2σ", "动能衰竭"],
                        score=0.6,
                    )
                )

    # Cost filter and sizing, then rank.
    for setup in setups:
        gap = abs(setup.targets[0] - setup.entry_mid)
        setup.cost_pct = round(ROUND_TRIP_COST / gap * 100.0, 2) if gap else 100.0
        setup.sizing = size_position(setup.stop_dollars, equity)
        if setup.cost_pct > 8.0:
            setup.score -= 0.1
    setups.sort(key=lambda s: (s.score, s.rr[0]), reverse=True)
    if not setups and not rejects:
        reject("没有任何形态同时通过时段、状态与赔率三道闸门")
    return setups[:3]
