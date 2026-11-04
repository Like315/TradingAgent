"""Indicators and the level map a scalper actually trades against.

The daily pipeline asks "where should gold be in two weeks". This module asks
the only question that matters at 5-minute resolution: **which prices have
orders sitting on them, and how far away is each one**.

The levels come from five places, in order of how much a gold desk respects them:

1. the session open (18:00 ET) and the Asian range it builds before London —
   the day's first reference and the box that breakout traders trade against;
2. the prior session's high / low / close / settle — where stops and resting
   orders were left behind;
3. the prior session's volume profile — POC and value area, estimated from
   5-minute bars (the only multi-session history the minute feed returns);
4. the current session VWAP and its ±1σ/±2σ bands — the mean every execution
   algo is measured against;
5. round numbers in $25 and $50 steps, where option strikes and retail stops
   cluster on a $4,000 handle.

Every level is returned with its distance in dollars and in ATR units, because
"$6 above" means nothing on its own while "1.4 ATR above" tells you whether a
scalp can reach it inside the trade's life.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from tradingagents.gold.intraday import feed
from tradingagents.gold.intraday.clock import LONDON

# $25 and $50: the steps where COMEX option strikes and round-number stops sit.
ROUND_STEPS = (25.0, 50.0)
# Anything closer than this is the same level for trading purposes.
MERGE_DOLLARS = 1.5
# Bin width for the estimated volume profile. $1 ≈ 10 ticks on GC.
PROFILE_BIN = 1.0


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR in dollars. The stop-distance unit for every setup."""
    if len(frame) < 2:
        return 0.0
    tr = true_range(frame).dropna()
    if tr.empty:
        return 0.0
    if len(tr) < period:
        return float(tr.mean())
    return float(tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])


def efficiency_ratio(close: pd.Series, period: int = 20) -> float:
    """Kaufman efficiency ratio in [0, 1].

    Net move divided by the distance actually travelled. A clean trend scores
    near 1 (it went straight there), a range scores near 0 (it wandered). This is
    the gate that decides trend-following versus mean-reversion, and it is far
    more stable at 5-minute resolution than ADX.
    """
    if len(close) <= period:
        return 0.0
    window = close.iloc[-period - 1 :]
    net = abs(float(window.iloc[-1]) - float(window.iloc[0]))
    path = float(window.diff().abs().sum())
    return 0.0 if path <= 0 else net / path


def session_vwap(frame: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """Volume-weighted average price from the day's open, plus its σ band.

    Returns ``(vwap, sigma, running_vwap)``. Sigma is the volume-weighted
    standard deviation of the typical price around VWAP, so ``vwap + 2σ`` is a
    genuinely stretched price rather than a fixed number of dollars.
    """
    if frame.empty or frame["volume"].sum() <= 0:
        typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
        vwap = float(typical.iloc[-1]) if len(typical) else float("nan")
        return vwap, 0.0, pd.Series(dtype=float)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    vol = frame["volume"].clip(lower=0)
    cum_vol = vol.cumsum().replace(0, np.nan)
    running = (typical * vol).cumsum() / cum_vol
    vwap = float(running.iloc[-1])
    # σ is measured against the final VWAP, not against each bar's contemporaneous
    # running VWAP: the latter makes every early bar contribute ~0 and halves the
    # band width, which silently turns "2σ" into "1.4σ".
    total_vol = float(vol.sum())
    sigma = float(np.sqrt(max(float(((typical - vwap) ** 2 * vol).sum() / total_vol), 0.0))) if total_vol else 0.0
    return vwap, sigma, running.dropna()


def swings(frame: pd.DataFrame, k: int = 2, lookback: int = 60) -> list[tuple[str, float, pd.Timestamp]]:
    """Fractal swing highs/lows: a bar that is the extreme of ``2k+1`` bars.

    Returns ``[("high"|"low", price, timestamp), ...]`` from oldest to newest,
    which is what a scalper draws as the micro demand/supply shelves.
    """
    window = frame.iloc[-lookback:]
    out: list[tuple[str, float, pd.Timestamp]] = []
    highs, lows = window["high"].to_numpy(), window["low"].to_numpy()
    for i in range(k, len(window) - k):
        span_high = highs[i - k : i + k + 1]
        span_low = lows[i - k : i + k + 1]
        if highs[i] == span_high.max():
            out.append(("high", float(highs[i]), window.index[i]))
        elif lows[i] == span_low.min():
            out.append(("low", float(lows[i]), window.index[i]))
    return out


def volume_z(frame: pd.DataFrame, period: int = 20) -> float:
    """How unusual the latest bar's volume is, in standard deviations."""
    vol = frame["volume"]
    if len(vol) < 5:
        return 0.0
    window = vol.iloc[-period - 1 : -1]
    std = float(window.std(ddof=0))
    if not std:
        return 0.0
    return (float(vol.iloc[-1]) - float(window.mean())) / std


# --------------------------------------------------------------------------- #
# Volume profile from minute bars
# --------------------------------------------------------------------------- #


@dataclass
class Profile:
    poc: float
    vah: float
    val: float
    vwap: float
    high: float
    low: float
    close: float
    settle: float | None
    bins: int


def volume_profile(frame: pd.DataFrame) -> Profile | None:
    """Estimate POC / value area / VWAP for a finished session.

    Bins the typical price of each bar weighted by its volume, takes the modal
    bin as the point of control, then grows a band around it until it holds 70%
    of the session's volume — the standard value-area construction, at $1
    resolution.

    This is an estimate and it is labelled as one: real volume-at-price needs
    tick data. It is still the best answer available for "where did yesterday
    actually trade", which is what the fade setups key off.
    """
    if frame.empty:
        return None
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    vol = frame["volume"].clip(lower=0)
    if float(vol.sum()) <= 0:
        return None
    low_edge = math.floor(float(frame["low"].min()) / PROFILE_BIN) * PROFILE_BIN
    high_edge = math.ceil(float(frame["high"].max()) / PROFILE_BIN) * PROFILE_BIN
    edges = np.arange(low_edge, high_edge + PROFILE_BIN, PROFILE_BIN)
    if len(edges) < 2:
        return None
    bucket = np.clip(np.digitize(typical.to_numpy(), edges) - 1, 0, len(edges) - 2)
    weights = np.zeros(len(edges) - 1)
    np.add.at(weights, bucket, vol.to_numpy())
    centres = (edges[:-1] + edges[1:]) / 2.0
    if weights.sum() <= 0:
        return None
    poc_index = int(np.argmax(weights))
    # Grow the value area outward from the POC, always taking the heavier neighbour.
    lo = hi = poc_index
    target = weights.sum() * 0.70
    held = weights[poc_index]
    while held < target and (lo > 0 or hi < len(weights) - 1):
        left = weights[lo - 1] if lo > 0 else -1.0
        right = weights[hi + 1] if hi < len(weights) - 1 else -1.0
        if right >= left:
            hi += 1
            held += weights[hi]
        else:
            lo -= 1
            held += weights[lo]
    settle = None
    # COMEX settles at 13:30 ET; keep the bar nearest to it as the settle proxy.
    try:
        ny = feed.to_new_york(frame.index)
        positions = [i for i, ts in enumerate(ny) if ts.hour == 13 and 25 <= ts.minute <= 35]
        if positions:
            settle = float(frame["close"].iloc[positions[-1]])
    except Exception:  # noqa: BLE001 — the settle proxy is a nicety
        settle = None
    vwap, _, _ = session_vwap(frame)
    return Profile(
        poc=float(centres[poc_index]),
        vah=float(centres[hi]),
        val=float(centres[lo]),
        vwap=vwap,
        high=float(frame["high"].max()),
        low=float(frame["low"].min()),
        close=float(frame["close"].iloc[-1]),
        settle=settle,
        bins=len(weights),
    )


# --------------------------------------------------------------------------- #
# The level map
# --------------------------------------------------------------------------- #


@dataclass
class Level:
    name: str
    price: float
    kind: str  # "open" | "range" | "prior" | "profile" | "vwap" | "round" | "swing"
    note: str = ""
    distance: float = 0.0  # signed dollars from last price
    atr: float = 0.0  # signed distance in ATR(5m) units
    side: str = ""  # "above" | "below" | "at"
    confluence: int = 1


@dataclass
class LevelMap:
    price: float
    atr5: float
    atr15: float
    atr_pct: float
    day_open: float | None
    session_high: float | None
    session_low: float | None
    asian_high: float | None
    asian_low: float | None
    asian_complete: bool
    vwap: float | None
    vwap_sigma: float | None
    prior: Profile | None
    levels: list[Level] = field(default_factory=list)
    swing_points: list[tuple[str, float, pd.Timestamp]] = field(default_factory=list)
    as_of: pd.Timestamp | None = None
    notes: list[str] = field(default_factory=list)
    # From the daily series, which is authoritative for yesterday's extremes:
    # the minute feed may not reach back a full session on every data source.
    prior_high: float | None = None
    prior_low: float | None = None
    prior_close: float | None = None
    daily_atr: float | None = None
    atr_estimated: bool = False

    @property
    def day_range(self) -> float | None:
        if self.session_high is None or self.session_low is None:
            return None
        return self.session_high - self.session_low

    @property
    def range_used(self) -> float | None:
        """How much of a normal day's range today has already travelled.

        Above ~1.2 the day is extended and mean reversion gets the benefit of
        the doubt; below ~0.5 there is usually still a directional move left.
        """
        if not self.daily_atr or not self.day_range:
            return None
        return self.day_range / self.daily_atr

    def nearest(self, side: str = "above", kinds: tuple[str, ...] | None = None) -> Level | None:
        pool = [lv for lv in self.levels if lv.side == side]
        if kinds:
            pool = [lv for lv in pool if lv.kind in kinds]
        if not pool:
            return None
        return min(pool, key=lambda lv: abs(lv.distance))


def _round_levels(price: float, width: float = 40.0) -> list[Level]:
    out: list[Level] = []
    seen: set[float] = set()
    # $50 first so that 4,050 keeps the stronger label when it is also a $25 step.
    for step in (50.0, 25.0):
        base = math.floor(price / step) * step
        for mult in (-1, 0, 1, 2):
            level = base + mult * step
            if abs(level - price) > width or level in seen:
                continue
            seen.add(level)
            tag = "整数关口" if step == 50.0 else "25 美元步长"
            out.append(Level(name=f"{level:,.0f}", price=float(level), kind="round", note=tag))
    return out


def _prior_daily_row(daily: pd.DataFrame, today: date):
    """Yesterday's OHLC from the daily series, or None.

    The daily feed carries the live session as its last row, so "previous" means
    the last row strictly before today's trading date.
    """
    if daily is None or daily.empty:
        return None
    days = [ts.date() for ts in daily.index]
    earlier = [i for i, day in enumerate(days) if day < today]
    return None if not earlier else daily.iloc[earlier[-1]]


def _daily_atr(daily: pd.DataFrame) -> float | None:
    if daily is None or len(daily) < 21:
        return None
    ranges = (daily["high"] - daily["low"]).dropna()
    if len(ranges) < 21:
        return None
    return float(ranges.iloc[-21:-1].mean())


def _london_open_beijing(session_start: pd.Timestamp) -> pd.Timestamp:
    """The first 08:00 London instant after the session opened, in Beijing time.

    Deriving it from the London zone rather than "03:00 New York" keeps the
    Asian-range boundary correct during the two weeks a year when the US and the
    EU change clocks on different dates.
    """
    start = session_start.tz_localize(feed.SHANGHAI)
    in_london = start.tz_convert(LONDON)
    candidate = in_london.normalize() + pd.Timedelta(hours=8)
    while candidate <= in_london:
        candidate += pd.Timedelta(days=1)
    return candidate.tz_convert(feed.SHANGHAI).tz_localize(None)


def build_level_map(
    one_min: pd.DataFrame,
    five_min: pd.DataFrame,
    fifteen_min: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    *,
    now: pd.Timestamp | None = None,
) -> LevelMap:
    """Assemble every level a scalp decision needs, in one pass."""
    if five_min.empty:
        raise ValueError("no 5m bars: cannot build a level map")
    notes: list[str] = []
    has_volume = bool(five_min.attrs.get("has_volume", True))
    price = float(five_min["close"].iloc[-1])
    if not one_min.empty:
        price = float(one_min["close"].iloc[-1])

    atr5 = atr(five_min)
    atr15 = atr(fifteen_min) if not fifteen_min.empty else atr5 * 1.7
    if not has_volume:
        # Frames built from minute closes lose the intrabar wick; ATR is the one
        # derived number that must be corrected for it, because it sets stops.
        atr5 *= feed.WICK_ADJUST
        atr15 *= feed.WICK_ADJUST
        notes.append(f"数据源只有分钟收盘价，ATR 按 {feed.WICK_ADJUST}x 修正（止损宁宽不窄）")
    atr_pct = (atr5 / price * 100.0) if price else 0.0

    today = feed.session_day(five_min.index[-1])
    sessions = feed.session_frames(five_min)
    today_bars = sessions.get(today, five_min.iloc[0:0])
    prior_days = [d for d in sessions if d < today]
    prior_bars = sessions[prior_days[-1]] if prior_days else five_min.iloc[0:0]
    prior = volume_profile(prior_bars) if len(prior_bars) > 20 else None
    if prior is None:
        notes.append("没有上一交易日的分钟数据，POC/价值区不可用（其余价位不受影响）")

    # Yesterday's extremes come from the daily series when it is there: it is
    # the authoritative record and survives a minute feed that only covers today.
    prior_row = _prior_daily_row(daily, today) if daily is not None else None
    prior_high = float(prior_row["high"]) if prior_row is not None else (prior.high if prior else None)
    prior_low = float(prior_row["low"]) if prior_row is not None else (prior.low if prior else None)
    prior_close = float(prior_row["close"]) if prior_row is not None else (prior.close if prior else None)
    daily_atr = _daily_atr(daily) if daily is not None else None

    day_open = float(today_bars["open"].iloc[0]) if not today_bars.empty else None
    session_high = float(today_bars["high"].max()) if not today_bars.empty else None
    session_low = float(today_bars["low"].min()) if not today_bars.empty else None

    asian_high = asian_low = None
    asian_complete = False
    if not today_bars.empty:
        london_open = _london_open_beijing(today_bars.index[0])
        asian_bars = today_bars[today_bars.index < london_open]
        asian_complete = bool(today_bars.index[-1] >= london_open)
        if asian_bars.empty:  # session began after London opened
            asian_bars = today_bars
            asian_complete = False
        asian_high = float(asian_bars["high"].max())
        asian_low = float(asian_bars["low"].min())

    vwap = sigma = None
    if not today_bars.empty:
        vwap, sigma, _ = session_vwap(today_bars)

    levels: list[Level] = []

    def add(name: str, level_price: float | None, kind: str, note: str = "") -> None:
        if level_price is None or not math.isfinite(level_price):
            return
        dist = level_price - price
        if abs(dist) > 60.0:  # further than a scalp can travel inside the session
            return
        side = "at" if abs(dist) < MERGE_DOLLARS else ("above" if dist > 0 else "below")
        levels.append(
            Level(
                name=name,
                price=round(float(level_price), 2),
                kind=kind,
                note=note,
                distance=round(dist, 2),
                atr=round(dist / atr5, 2) if atr5 else 0.0,
                side=side,
            )
        )

    add("当日开盘（18:00 ET 重开）", day_open, "open", "多空分界，站上偏多")
    add("当日高", session_high, "range", "日内高点，破位要看量")
    add("当日低", session_low, "range", "日内低点，破位要看量")
    if asian_high and asian_low:
        add("亚洲区间高" + ("" if asian_complete else "（进行中）"), asian_high, "range", "伦敦开盘前的高点，最常被扫")
        add("亚洲区间低" + ("" if asian_complete else "（进行中）"), asian_low, "range", "伦敦开盘前的低点，最常被扫")
    add("昨日高", prior_high, "prior", "隔夜挂单与止损聚集")
    add("昨日低", prior_low, "prior", "隔夜挂单与止损聚集")
    add("昨收", prior_close, "prior", "跳空缺口参照")
    if prior:
        add("昨日 POC", prior.poc, "profile", "成交量最大价位，回归磁石")
        add("昨日价值区上沿", prior.vah, "profile", "70% 成交量上沿")
        add("昨日价值区下沿", prior.val, "profile", "70% 成交量下沿")
    if vwap is not None and sigma:
        add("当日 VWAP", vwap, "vwap", "执行算法的基准线")
        for mult in (1.0, 2.0):
            add(f"VWAP +{mult:.0f}σ", vwap + mult * sigma, "vwap", "偏离均值，易回抽")
            add(f"VWAP -{mult:.0f}σ", vwap - mult * sigma, "vwap", "偏离均值，易回抽")
    for level in _round_levels(price):
        add(level.name, level.price, level.kind, level.note)

    swing_points = swings(five_min, k=2, lookback=80)
    for kind, level_price, stamp in swing_points[-6:]:
        add(
            f"{'波段高' if kind == 'high' else '波段低'} {stamp.strftime('%H:%M')}",
            level_price,
            "swing",
            "近 6 小时微观结构",
        )

    # Merge levels that sit on top of each other: a $1.5 cluster is one wall.
    levels.sort(key=lambda lv: lv.price)
    merged: list[Level] = []
    for level in levels:
        if merged and abs(level.price - merged[-1].price) <= MERGE_DOLLARS:
            keeper = merged[-1]
            keeper.confluence += 1
            keeper.note = f"{keeper.note} + {level.name}" if keeper.note else level.name
            if level.kind in ("prior", "profile") and keeper.kind in ("round", "swing"):
                keeper.name, keeper.kind = level.name, level.kind
            continue
        merged.append(level)
    merged.sort(key=lambda lv: lv.price, reverse=True)

    if daily_atr and session_high and session_low:
        used = (session_high - session_low) / daily_atr
        if used >= 1.2:
            notes.append(f"今日振幅已达 20 日均幅的 {used:.0%}，空间被消耗，优先做回归而不是追趋势")
        elif used <= 0.5:
            notes.append(f"今日振幅只有 20 日均幅的 {used:.0%}，通常还有一段扩张没走")

    return LevelMap(
        price=price,
        atr5=atr5,
        atr15=atr15,
        atr_pct=atr_pct,
        day_open=day_open,
        session_high=session_high,
        session_low=session_low,
        asian_high=asian_high,
        asian_low=asian_low,
        asian_complete=asian_complete,
        vwap=vwap,
        vwap_sigma=sigma,
        prior=prior,
        levels=merged,
        swing_points=swing_points,
        as_of=one_min.index[-1] if not one_min.empty else five_min.index[-1],
        notes=notes,
        prior_high=prior_high,
        prior_low=prior_low,
        prior_close=prior_close,
        daily_atr=daily_atr,
        atr_estimated=not has_volume,
    )
