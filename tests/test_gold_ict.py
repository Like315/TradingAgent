"""Tests for the ICT model: pools, sweeps, 1H bias, fixed-2R entries.

Offline only. The parts worth pinning are the ones that quietly ruin an ICT
model when they drift:

* a pool's side is decided by where price is **now** — an equal high that price
  already traded through is support, not resting buy-side liquidity;
* a sweep needs the bar to *straddle* the level. Without that, every bar above
  a distant pool "sweeps" it and the board fills with fake signals;
* the 1H bias must refuse the premium half in a bull market — that single rule
  is what stops the model buying the top of a range;
* the target is exactly 2R from the entry, never a level the model liked.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.gold.ict.bias import Bias, _structure
from tradingagents.gold.ict.entry import build_setups
from tradingagents.gold.ict.liquidity import (
    BUY_SIDE,
    SELL_SIDE,
    Pool,
    Sweep,
    find_pools,
    find_sweeps,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def frame(closes: list[float], high_pad: float = 0.3, low_pad: float = 0.3, start: str = "2026-11-04 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="15min")
    rows = [
        {"open": c, "high": c + high_pad, "low": c - low_pad, "close": c, "volume": 10.0}
        for c in closes
    ]
    return pd.DataFrame(rows, index=idx)


def ctx_for(bars: pd.DataFrame, *, atr: float = 10.0, price: float | None = None, levels: list | None = None):
    return SimpleNamespace(
        atr=atr,
        atr15=atr,
        atr1h=atr * 2,
        price=price if price is not None else float(bars["close"].iloc[-1]),
        bars15=bars,
        bars1h=bars,
        now=datetime(2026, 11, 4, 22, 10),
        levels=SimpleNamespace(levels=levels or []),
    )


def level(price: float, name: str, kind: str = "range"):
    return SimpleNamespace(price=price, name=name, kind=kind, note="")


def zigzag_1h() -> pd.DataFrame:
    """A clean bullish 1H structure: low 99 → high 108 → higher low 102 → higher high 110."""
    closes = [
        100, 99.5, 99, 99.5, 100, 101, 102, 103,
        108, 107, 106, 105, 104, 103, 102, 103,
        104, 105, 106, 107, 108, 109, 110, 109,
        108, 107, 106, 105, 104, 105, 106,
    ]
    idx = pd.date_range("2026-11-01 00:00", periods=len(closes), freq="1h")
    return pd.DataFrame(
        [{"open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 10.0} for c in closes],
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Pools
# --------------------------------------------------------------------------- #


def test_equal_highs_cluster_into_one_pool_and_take_side_from_price():
    # Two touches near 120, price has since fallen to 100: it is now a
    # buy-side pool above the market.
    bars = frame([118, 119, 120, 119, 118, 117, 118, 119.9, 118, 117, 110, 105, 100])
    ctx = ctx_for(bars, atr=5.0, price=100.0)
    pools = find_pools(ctx)
    highs = [p for p in pools if p.kind == "equal_highs"]
    assert highs and highs[0].touches >= 2
    assert all(p.side == BUY_SIDE for p in highs if p.price > 100)


def test_pool_below_price_is_sell_side_even_if_it_was_a_high():
    # The same equal-high cluster, but price is now above it: it is support.
    bars = frame([110, 110.1, 110, 110.2, 110, 115, 120, 125, 130])
    ctx = ctx_for(bars, atr=5.0, price=130.0)
    pools = find_pools(ctx)
    for pool in pools:
        expected = BUY_SIDE if pool.price > 130 else SELL_SIDE
        assert pool.side == expected


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #


def test_sweep_requires_the_bar_to_straddle_the_pool():
    pool = Pool(price=100.0, kind="equal_highs", label="等高等低·高", side=BUY_SIDE, touches=2)
    # Price runs from 95 to 103 and closes back at 99: a real sweep.
    bars = frame([95, 96, 97, 98, 99, 103, 99, 98], high_pad=0.2, low_pad=0.2)
    ctx = ctx_for(bars, atr=10.0, price=98.0)
    sweeps = find_sweeps(ctx, [pool])
    assert sweeps and sweeps[0].reclaimed
    assert sweeps[0].pool.price == 100.0
    assert sweeps[0].extreme >= 103.0


def test_distant_pool_is_not_swept_by_price_far_above_it():
    pool = Pool(price=100.0, kind="equal_highs", label="远端池", side=BUY_SIDE, touches=2)
    bars = frame([130, 131, 132, 133, 134, 135])  # never comes near 100
    ctx = ctx_for(bars, atr=10.0, price=135.0)
    assert find_sweeps(ctx, [pool]) == []


def test_pierced_but_not_reclaimed_is_reported_as_armed():
    pool = Pool(price=100.0, kind="equal_highs", label="等高等低·高", side=BUY_SIDE, touches=2)
    bars = frame([98, 99, 101, 102], high_pad=0.2, low_pad=0.2)  # closes above: not reclaimed
    ctx = ctx_for(bars, atr=10.0, price=102.0)
    sweeps = find_sweeps(ctx, [pool])
    assert sweeps and not sweeps[0].reclaimed
    assert sweeps[0].status == "已扫待回收"


# --------------------------------------------------------------------------- #
# Bias
# --------------------------------------------------------------------------- #


def test_structure_reads_higher_highs_and_higher_lows():
    points = [("high", 100.0, 0), ("low", 90.0, 1), ("high", 110.0, 2), ("low", 95.0, 3)]
    structure, _ = _structure(points)
    assert structure == "bullish"
    points = [("high", 110.0, 0), ("low", 95.0, 1), ("high", 100.0, 2), ("low", 90.0, 3)]
    assert _structure(points)[0] == "bearish"


def test_bullish_bias_in_discount_allows_longs():
    from tradingagents.gold.ict.bias import read_bias

    bars = zigzag_1h()
    ctx = ctx_for(bars, atr=3.0, price=104.0)  # 102-110 range → 25% = discount
    bias = read_bias(ctx, pools=[])
    assert bias.structure == "bullish"
    assert bias.direction == "long"
    assert bias.position == "discount"
    assert bias.allowed


def test_bullish_bias_in_premium_refuses_to_buy_the_top():
    from tradingagents.gold.ict.bias import read_bias

    bars = zigzag_1h()
    ctx = ctx_for(bars, atr=3.0, price=109.5)  # 94% of the range
    bias = read_bias(ctx, pools=[])
    assert bias.structure == "bullish"
    assert bias.direction == "none"
    assert bias.position == "premium"
    assert any("premium" in reason for reason in bias.reasons)


# --------------------------------------------------------------------------- #
# Entry: fixed 2R
# --------------------------------------------------------------------------- #


def _long_sweep(extreme: float = 3990.0, pool_price: float = 4000.0, bars_since: int = 1) -> Sweep:
    pool = Pool(price=pool_price, kind="equal_lows", label="等高等低·低（3 触）", side=SELL_SIDE, touches=3)
    return Sweep(
        pool=pool,
        bar_time=pd.Timestamp("2026-11-04 21:45"),
        extreme=extreme,
        pierce=pool_price - extreme,
        reclaimed=True,
        reclaim_time=pd.Timestamp("2026-11-04 22:00"),
        reclaim_close=4002.0,
        displacement_atr=0.8,
        bars_since=bars_since,
    )


def test_target_is_exactly_two_r_from_the_entry():
    bars = frame([4003.0] * 10)
    ctx = ctx_for(bars, atr=10.0, price=4003.0)
    bias = Bias(direction="long", structure="bullish", label="1H 偏多（折价区）", position_pct=0.2, position="discount")
    setups = build_setups(ctx, bias, [_long_sweep()], [], equity=50_000, risk_pct=0.3)
    assert setups, "a clean sweep in the bias direction must produce a setup"
    setup = setups[0]
    entry_mid = (setup.entry_low + setup.entry_high) / 2
    assert setup.stop == pytest.approx(3988.0)  # 3990 extreme - 0.2*ATR
    assert setup.targets[0] == pytest.approx(entry_mid + 2 * abs(entry_mid - setup.stop), abs=0.05)
    assert setup.rr == [2.0]
    assert setup.sizing and setup.sizing[0].ounces > 0


def test_counter_bias_sweep_is_refused():
    bars = frame([4003.0] * 10)
    ctx = ctx_for(bars, atr=10.0, price=4003.0)
    bias = Bias(direction="short", structure="bearish", label="1H 偏空（溢价区）", position_pct=0.8, position="premium")
    rejects: list[str] = []
    setups = build_setups(ctx, bias, [_long_sweep()], [], equity=50_000, risk_pct=0.3, rejects=rejects)
    assert setups == []
    assert any("没有与 1H 同向" in reason for reason in rejects)


def test_opposing_pool_inside_one_point_two_r_blocks_the_trade():
    bars = frame([4003.0] * 10)
    ctx = ctx_for(bars, atr=10.0, price=4003.0)
    bias = Bias(direction="long", structure="bullish", label="1H 偏多（折价区）", position_pct=0.2, position="discount")
    blocker = Pool(price=4004.0, kind="equal_highs", label="近端买方池", side=BUY_SIDE, touches=2)
    rejects: list[str] = []
    setups = build_setups(ctx, bias, [_long_sweep()], [blocker], equity=50_000, risk_pct=0.3, rejects=rejects)
    assert setups == []
    assert any("前方流动性" in reason for reason in rejects)


# --------------------------------------------------------------------------- #
# SMC session layer: Asian → London → NY, PO3, killzones
# --------------------------------------------------------------------------- #

from tradingagents.gold.ict.sessions import Manipulation, SessionRange, Sessions, compute_sessions, killzone
from tradingagents.gold.ict.bias import read_bias

from tradingagents.gold.intraday import feed

SHANGHAI_TZ = feed.SHANGHAI


def sessions_ctx(bars, *, now=None, atr=10.0, levels=None, price=None):
    c = ctx_for(bars, atr=atr, levels=levels, price=price)
    if now is not None:
        c.now = now
    return c


def test_killzone_windows_follow_their_own_timezone():
    # 10:30 Tokyo = Asian killzone
    assert killzone(datetime(2026, 11, 4, 9, 30, tzinfo=SHANGHAI_TZ))[0] == "asian"
    # 16:30 Beijing = 08:30/09:30 London (DST) → London killzone
    assert killzone(datetime(2026, 11, 4, 16, 30, tzinfo=SHANGHAI_TZ))[0] == "london"
    # 22:00 Beijing = 09:00/10:00 New York → NY killzone
    assert killzone(datetime(2026, 11, 4, 22, 0, tzinfo=SHANGHAI_TZ))[0] == "new_york"
    # 14:00 Beijing = Tokyo lunch, London pre-open, NY asleep → nothing
    assert killzone(datetime(2026, 11, 4, 14, 0, tzinfo=SHANGHAI_TZ)) is None


def _session_bars():
    """One trading day: flat Asia, London dips below the Asian low then reclaims.

    The Globex day opens 18:00 ET on Nov 3 (= 06:00/07:00 Beijing Nov 4) and
    runs to 17:00 ET Nov 4. Bars are hourly-ish 15m rows built per phase.
    """
    idx = pd.date_range("2026-11-04 06:00", periods=64, freq="15min")  # through 22:00 Beijing
    rows = []
    for ts in idx:
        hour = ts.hour + ts.minute / 60
        if hour < 10.0:  # Asia: flat 4000
            px = 4000.0
        elif hour < 16.0:  # between Asia end (london open 16:00) boundary slop: keep flat
            px = 4000.0
        elif hour < 18.0:  # London: sweep the Asian low (3996) then reclaim
            px = 3994.0 if ts.minute == 0 else 4000.0
        else:  # NY: distribution up
            px = 4004.0
        rows.append({"open": px, "high": px + 1.0, "low": px - 1.0, "close": px, "volume": 10.0})
    return pd.DataFrame(rows, index=idx)


def test_sessions_compute_asian_london_ny_ranges():
    bars = _session_bars()
    ctx = sessions_ctx(bars, now=datetime(2026, 11, 4, 22, 0, tzinfo=SHANGHAI_TZ))
    s = compute_sessions(ctx)
    assert s.asian.high is not None and s.asian.complete
    assert s.london.high is not None
    assert s.new_york.high is not None and not s.new_york.complete
    # Asian range is the flat 4000 block (±1 pad)
    assert 3999.0 <= s.asian.low <= 4001.0
    assert 3999.0 <= s.asian.high <= 4001.5


def test_london_sweep_of_asian_low_is_detected_as_manipulation():
    bars = _session_bars()
    ctx = sessions_ctx(bars, now=datetime(2026, 11, 4, 22, 0, tzinfo=SHANGHAI_TZ))
    s = compute_sessions(ctx)
    m = s.manipulation
    assert m.swept_side == "low"
    assert m.reclaimed
    assert m.by_session == "london"
    assert m.expected_direction == "long"
    assert "向上" in s.po3_phase or "派发" in s.po3_phase


def test_no_london_sweep_means_no_manipulation():
    idx = pd.date_range("2026-11-04 06:00", periods=40, freq="15min")
    rows = [{"open": 4000.0, "high": 4000.8, "low": 3999.2, "close": 4000.0, "volume": 10.0}] * len(idx)
    ctx = sessions_ctx(pd.DataFrame(rows, index=idx), now=datetime(2026, 11, 4, 16, 30, tzinfo=SHANGHAI_TZ))
    s = compute_sessions(ctx)
    assert s.manipulation.swept_side == ""
    assert "等操纵" in s.po3_phase or "积累" in s.po3_phase


def test_session_override_grants_long_in_premium_after_asian_low_sweep():
    bars = _session_bars().iloc[:46]  # cut at 17:30: the reclaim is still fresh
    ctx = sessions_ctx(bars, now=datetime(2026, 11, 4, 17, 30, tzinfo=SHANGHAI_TZ))
    s = compute_sessions(ctx)
    # 1H structure bullish (rising closes) but price at the top of its dealing
    # range → the pure gate says "wait in premium". The reclaimed Asian-low
    # sweep must override that.
    points = [("high", 3990.0, "a"), ("low", 3980.0, "b"), ("high", 3995.0, "c"), ("low", 3985.0, "d")]
    ctx.points = points
    import tradingagents.gold.ict.bias as bias_mod

    original = bias_mod.swings
    bias_mod.swings = lambda *a, **k: points
    try:
        bias = read_bias(ctx, pools=[], sessions=s)
    finally:
        bias_mod.swings = original
    assert bias.direction == "long"
    assert bias.origin == "session"
    assert any("Session override" in r for r in bias.reasons)
    # The draw is the untouched Asian high, not a generic pool
    assert bias.draw is not None and bias.draw.kind == "session_asian"


def test_session_override_does_not_fight_a_conflicting_structure():
    bars = _session_bars().iloc[:46]
    # price parked in discount so the bearish structure alone cannot authorise
    # the short — only the refused override is under test here
    ctx = sessions_ctx(bars, now=datetime(2026, 11, 4, 17, 30, tzinfo=SHANGHAI_TZ), price=3992.0)
    s = compute_sessions(ctx)
    points = [("high", 4020.0, "a"), ("low", 4000.0, "b"), ("high", 4010.0, "c"), ("low", 3990.0, "d")]
    import tradingagents.gold.ict.bias as bias_mod

    original = bias_mod.swings
    bias_mod.swings = lambda *a, **k: points
    try:
        bias = read_bias(ctx, pools=[], sessions=s)
    finally:
        bias_mod.swings = original
    # Bearish structure vs an upward PO3: the override must refuse.
    assert bias.direction == "none"


def test_stale_manipulation_is_not_an_override():
    bars = _session_bars()
    ctx = sessions_ctx(bars, now=datetime(2026, 11, 4, 22, 0, tzinfo=SHANGHAI_TZ))
    s = compute_sessions(ctx)
    s.manipulation.bars_since = 50  # ~12 hours ago
    import tradingagents.gold.ict.bias as bias_mod

    points = [("high", 3990.0, "a"), ("low", 3980.0, "b"), ("high", 3995.0, "c"), ("low", 3985.0, "d")]
    original = bias_mod.swings
    bias_mod.swings = lambda *a, **k: points
    try:
        bias = read_bias(ctx, pools=[], sessions=s)
    finally:
        bias_mod.swings = original
    assert bias.direction == "none"
    assert any("太旧" in r for r in bias.reasons)


def test_session_extremes_enter_the_pool_inventory():
    bars = _session_bars()
    ctx = sessions_ctx(bars, now=datetime(2026, 11, 4, 22, 0, tzinfo=SHANGHAI_TZ))
    s = compute_sessions(ctx)
    pools = find_pools(ctx, s)
    labels = " ".join(p.label for p in pools)
    assert "伦敦" in labels and "纽约" in labels
    kinds = {p.kind for p in pools}
    assert "session_extreme" in kinds
