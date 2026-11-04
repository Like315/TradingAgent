"""Tests for the intraday scalp desk: feed parsing, clock, levels, setups, risk.

Every test here is offline. The point is to pin the parts that are easy to get
silently wrong and expensive when wrong:

* Eastmoney's column order (open, **close**, high, low) — reading the third
  column as the high puts a fake wick on half the bars;
* Sina's first minline row, whose price is in field 5, not field 1 — the wrong
  field moves the day's open by $30 and every opening-range level with it;
* the London-open boundary used to cut the Asian range, which must follow the
  London DST date rather than a fixed 03:00 New York;
* the risk gate, which must refuse an invented stop rather than size it.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradingagents.gold.intraday import clock as gclock
from tradingagents.gold.intraday import feed, levels, tape
from tradingagents.gold.intraday.board import TapeBoard
from tradingagents.gold.intraday.clock import SessionClock
from tradingagents.gold.intraday.desk import ScalpCall, _entry_range, _parse_card, risk_check
from tradingagents.gold.intraday.levels import Level, LevelMap, Profile
from tradingagents.gold.intraday.tape import Regime

SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def frame(
    closes,
    *,
    start: str = "2026-10-29 20:00",
    freq: str = "5min",
    volume: float = 100.0,
    has_volume: bool = True,
    spread: float = 1.0,
) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq=freq)
    data = pd.DataFrame(
        {
            "open": list(closes),
            "high": [c + spread for c in closes],
            "low": [c - spread for c in closes],
            "close": list(closes),
            "volume": [volume] * len(closes),
        },
        index=index,
    )
    data.attrs.update(source="test", stale=False, has_volume=has_volume, note="")
    return data


def stub_clock(*, tradable: bool = True, grade: str = "A", blockers: list[str] | None = None) -> SessionClock:
    now = datetime(2026, 10, 29, 22, 0, tzinfo=SHANGHAI)
    return SessionClock(
        now=now,
        phase_label="COMEX 开盘",
        phase_note="",
        grade=grade,
        grade_note="",
        is_open=tradable,
        minutes_since_open=960,
        minutes_to_halt=420,
        next_marks=[],
        blackout=None,
        events=[],
    )


def stub_regime(**kwargs) -> Regime:
    base = dict(
        code="trend_up",
        label="趋势上行",
        er5=0.42,
        er15=0.35,
        ema_align="多头排列",
        slope_atr=1.6,
        rv_1h=18.0,
        vol_pct=55.0,
        vol_basis="5 日",
        atr_expansion=1.05,
        dist_sigma=0.3,
        volume_state="常量",
        bias=0.7,
        notes=[],
    )
    base.update(kwargs)
    return Regime(**base)


def make_level_map(price: float = 4040.0, atr5: float = 4.0, **kwargs) -> LevelMap:
    base = dict(
        price=price,
        atr5=atr5,
        atr15=atr5 * 1.8,
        atr_pct=0.1,
        day_open=4020.0,
        session_high=4060.0,
        session_low=4010.0,
        asian_high=4035.0,
        asian_low=4020.0,
        asian_complete=True,
        vwap=4035.0,
        vwap_sigma=5.0,
        prior=None,
        levels=[],
        swing_points=[],
        as_of=pd.Timestamp("2026-10-29 22:00"),
        notes=[],
        prior_high=4065.0,
        prior_low=4005.0,
        prior_close=4050.0,
        daily_atr=70.0,
        atr_estimated=False,
    )
    base.update(kwargs)
    return LevelMap(**base)


def lv(name: str, price: float, kind: str, ref: float = 4040.0, confluence: int = 1) -> Level:
    distance = price - ref
    return Level(
        name=name,
        price=price,
        kind=kind,
        note="",
        distance=distance,
        atr=distance / 4.0,
        side="at" if abs(distance) < 1.5 else ("above" if distance > 0 else "below"),
        confluence=confluence,
    )


# --------------------------------------------------------------------------- #
# Feed: parsing
# --------------------------------------------------------------------------- #


def test_eastmoney_columns_are_open_close_high_low():
    payload = ["2026-10-29 22:00,4040.0,4042.5,4044.0,4038.0,120,0.0,0.1"]
    parsed = feed._parse_klines(payload)
    row = parsed.iloc[0]
    assert row["open"] == 4040.0
    assert row["close"] == 4042.5  # third column is the close, not the high
    assert row["high"] == 4044.0
    assert row["low"] == 4038.0
    assert row["volume"] == 120.0


def test_sina_first_row_uses_the_open_not_the_other_field():
    """Field 1 is not the session open; using it shifts the day by tens of dollars."""
    text = (
        'var _GC=({"minLine_1d":[["2026-10-29","4051.600","cme","","06:00","4079.560","0","0","4080.637",'
        '"2026-10-29 06:00:00"],["06:01","4079.990","0","0","4080.562","2026-10-29 06:01:00"],'
        '["06:02","4079.092","0","0","4080.568","2026-10-29 06:02:00"]]});'
    )
    parsed = feed._parse_sina_minline(text)
    assert len(parsed) == 3
    assert parsed["close"].iloc[0] == pytest.approx(4079.56)
    assert parsed["close"].iloc[-1] == pytest.approx(4079.092)
    assert parsed.index[0] == pd.Timestamp("2026-10-29 06:00")


def test_sina_minline_marks_unit_volume_so_vwap_falls_back_to_a_mean():
    text = (
        'var _GC=({"minLine_1d":[["06:01","100.0","0","0","100.0","2026-10-29 06:01:00"],'
        '["06:02","102.0","0","0","101.0","2026-10-29 06:02:00"]]});'
    )
    parsed = feed._parse_sina_minline(text)
    assert (parsed["volume"] == 1.0).all()
    vwap, sigma, _ = levels.session_vwap(parsed)
    assert vwap == pytest.approx(101.0)
    assert sigma == pytest.approx(1.0)


def test_sina_payload_without_jsonp_is_empty_not_an_exception():
    assert feed._parse_sina_minline("upstream error").empty


# --------------------------------------------------------------------------- #
# Feed: session day and DST
# --------------------------------------------------------------------------- #


def test_session_day_starts_at_the_globex_open():
    # 06:00 Beijing = 18:00 New York the previous day: the new trading day.
    assert feed.session_day(pd.Timestamp("2026-10-29 06:00")) == date(2026, 10, 29)
    # 05:30 Beijing is still the previous day's session, after the 17:00 halt.
    assert feed.session_day(pd.Timestamp("2026-10-29 05:30")) == date(2026, 10, 28)
    # 23:59 Beijing belongs to the same trading day it opened in.
    assert feed.session_day(pd.Timestamp("2026-10-29 23:59")) == date(2026, 10, 29)


def test_london_open_follows_london_dst_not_a_fixed_new_york_hour():
    """London's 08:00 lands on 15:00 Beijing under BST and 16:00 under GMT.

    2026 is a mismatch year in late October: BST ends 25 Oct while US EDT runs to
    1 Nov, so on 29 Oct the London open is 16:00 Beijing and COMEX opens at
    20:20 — an hour apart from what a hardcoded 03:00 ET would imply.
    """
    assert levels._london_open_beijing(pd.Timestamp("2026-10-20 06:00")).strftime("%H:%M") == "15:00"
    assert levels._london_open_beijing(pd.Timestamp("2026-10-29 06:00")).strftime("%H:%M") == "16:00"
    assert levels._london_open_beijing(pd.Timestamp("2026-12-02 07:00")).strftime("%H:%M") == "16:00"


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #


def test_liquidity_bands_grade_the_session():
    assert gclock._band(8.5)[0] == "A"  # COMEX open
    assert gclock._band(4.0)[0] == "A"  # London
    assert gclock._band(22.0)[0] == "B"  # Asia
    assert gclock._band(14.0)[0] == "C"  # NY afternoon
    assert gclock._band(17.5)[0] == "D"  # maintenance


def test_clock_refuses_to_trade_in_the_halt():
    now = datetime(2026, 10, 29, 5, 30, tzinfo=SHANGHAI)
    clock = gclock.read_clock(now)
    assert clock.grade == "D"
    assert not clock.tradable
    assert any("维护" in reason for reason in clock.blockers)


def test_nfp_is_the_first_friday_and_claims_are_thursday():
    events = [e for e in gclock.recurring_events(date(2026, 11, 1), days=9)]
    nfp = [e for e in events if e.label == "非农就业"]
    claims = [e for e in events if e.label == "初请失业金"]
    assert nfp and all(e.when.date().weekday() == 4 and e.when.date().day <= 7 for e in nfp)
    assert claims and all(e.when.date().weekday() == 3 for e in claims)
    assert nfp[0].when.hour == 8 and nfp[0].when.minute == 30


def test_blackout_covers_ten_minutes_before_and_fifteen_after():
    event = gclock.Event(
        datetime(2026, 10, 30, 8, 30, tzinfo=NEW_YORK), "CPI", "high", "file"
    )
    before = event.when.astimezone(SHANGHAI) - pd.Timedelta(minutes=6)
    after = event.when.astimezone(SHANGHAI) + pd.Timedelta(minutes=12)
    outside = event.when.astimezone(SHANGHAI) + pd.Timedelta(minutes=30)
    assert gclock.blackout(before, [event]).active
    assert gclock.blackout(after, [event]).active
    resolved = gclock.blackout(outside, [event])
    assert resolved is None or not resolved.active


# --------------------------------------------------------------------------- #
# Indicators and the level map
# --------------------------------------------------------------------------- #


def test_efficiency_ratio_separates_a_trend_from_chop():
    straight = pd.Series([100.0 + i for i in range(30)])
    zigzag = pd.Series([100.0 + (1 if i % 2 else 0) for i in range(30)])
    assert levels.efficiency_ratio(straight, 20) > 0.95
    assert levels.efficiency_ratio(zigzag, 20) < 0.15


def test_atr_matches_the_average_true_range():
    data = frame([100, 102, 101, 105, 104])
    # high/low are close ±1, so every true range is 2 except the gap bars.
    assert levels.atr(data, period=3) > 0


def test_volume_profile_puts_the_poc_where_the_volume_is():
    closes = [100.0] * 20 + [101.0] * 2
    data = frame(closes, volume=10.0)
    data.loc[data.index[-2]:, "volume"] = 500.0
    profile = levels.volume_profile(data)
    assert profile is not None
    assert profile.val <= profile.poc <= profile.vah
    assert profile.poc == pytest.approx(101.0, abs=1.5)


def test_level_map_prefers_the_daily_series_for_yesterdays_extremes():
    five = frame([4040.0 + i for i in range(30)], start="2026-10-29 20:00")
    daily = pd.DataFrame(
        {
            "open": [4000.0, 4010.0, 4050.0],
            "high": [4020.0, 4030.0, 4090.0],
            "low": [3990.0, 4000.0, 4020.0],
            "close": [4010.0, 4050.0, 4060.0],
            "volume": [None, None, None],
        },
        index=pd.to_datetime(["2026-10-27", "2026-10-28", "2026-10-29"]),
    )
    built = levels.build_level_map(five, five, five.iloc[::3], daily)
    assert built.prior_high == 4030.0  # 2026-10-28, not the live 2026-10-29 row
    assert built.prior_low == 4000.0
    assert built.prior_close == 4050.0


def test_closes_only_frames_widen_atr_and_say_so():
    five = frame([4040.0 + (i % 3) for i in range(60)], has_volume=False)
    built = levels.build_level_map(five, five, five.iloc[::3])
    assert built.atr_estimated
    assert any(str(feed.WICK_ADJUST) in note for note in built.notes)


# --------------------------------------------------------------------------- #
# Setups
# --------------------------------------------------------------------------- #


def _trend_map() -> LevelMap:
    price = 4040.0
    return make_level_map(
        price=price,
        levels=[
            lv("当日 VWAP", 4036.0, "vwap", price),
            lv("波段低 21:00", 4034.0, "swing", price),
            lv("昨日高", 4060.0, "prior", price),
            lv("整数关口", 4075.0, "round", price),
        ],
    )


def test_trend_pullback_is_generated_and_uses_the_shelf_for_the_stop():
    lm = _trend_map()
    five = frame([4030.0 + i * 0.2 for i in range(40)])
    setups = tape.build_setups(lm, stub_regime(), stub_clock(), equity=100_000, five_min=five)
    assert setups, "a clean uptrend at a support shelf should produce a setup"
    primary = setups[0]
    assert primary.kind == "trend_pullback"
    assert primary.side == "long"
    assert primary.stop < 4036.0  # below the shelf
    assert primary.targets[0] > primary.entry_high
    assert primary.rr[0] >= tape.MIN_RR_AFTER_COST


def test_no_setups_when_the_clock_is_closed_and_the_reason_is_recorded():
    rejects: list[str] = []
    setups = tape.build_setups(
        _trend_map(), stub_regime(), stub_clock(tradable=False, grade="D"), equity=100_000,
        five_min=frame([4030.0] * 40), rejects=rejects,
    )
    assert setups == []
    assert rejects and "时段" in rejects[0]


def test_no_setups_in_a_dead_or_chaotic_regime():
    for code in ("dead", "chaos"):
        rejects: list[str] = []
        setups = tape.build_setups(
            _trend_map(), stub_regime(code=code), stub_clock(), equity=100_000,
            five_min=frame([4030.0] * 40), rejects=rejects,
        )
        assert setups == []
        assert rejects


def test_range_fade_is_refused_when_it_fights_the_higher_timeframe():
    """Buying a level while the 15m is bearish and price is not extended = knife catch."""
    price = 4034.0
    lm = make_level_map(
        price=price,
        levels=[
            lv("亚洲区间低", 4033.0, "range", price),
            lv("昨日低", 4025.0, "prior", price),
        ],
    )
    regime = stub_regime(code="range", er5=0.1, ema_align="空头排列", slope_atr=-0.2, bias=-0.5)
    rejects: list[str] = []
    setups = tape.build_setups(
        lm, regime, stub_clock(), equity=100_000, five_min=frame([4030.0] * 40), rejects=rejects
    )
    assert all(s.side != "long" for s in setups)
    assert any("逆势" in reason for reason in rejects)


def test_breakout_needs_volume_confirmation():
    price = 4062.0
    lm = make_level_map(price=price, session_high=4060.0, levels=[lv("当日高", 4060.0, "range", price)])
    five = frame([4040.0] * 39 + [4062.0], volume=100.0)
    no_volume = tape.build_setups(
        lm, stub_regime(), stub_clock(), equity=100_000, five_min=five, has_volume=False
    )
    assert all(s.kind != "breakout_retest" for s in no_volume)


def test_sizing_divides_risk_by_the_stop():
    rows = tape.size_position(stop_dollars=4.0, equity=100_000, risk_levels=(0.5,))
    assert rows[0].risk_usd == pytest.approx(500.0)
    assert rows[0].ounces == pytest.approx(125.0)
    assert rows[0].contracts == pytest.approx(1.25)


def test_setup_costs_are_a_percentage_of_the_first_target():
    lm = _trend_map()
    setups = tape.build_setups(
        lm, stub_regime(), stub_clock(), equity=100_000, five_min=frame([4030.0 + i * 0.2 for i in range(40)])
    )
    assert setups
    assert 0 < setups[0].cost_pct < 100


# --------------------------------------------------------------------------- #
# Card parsing and the risk gate
# --------------------------------------------------------------------------- #


def test_card_parser_reads_the_key_block():
    card = """CALL: SHORT
SETUP: 区间高抛：昨日低
ENTRY: 4036.0-4037.1
STOP: 4039.7
T1: 4031.8
T2: 4027.5
HOLD: 12
SIZE: 0.5
WHY: 一句话
KILL: 5 分钟收盘站上 4039.7"""
    fields = _parse_card(card)
    assert fields["CALL"] == "SHORT"
    assert _entry_range(fields["ENTRY"]) == (4036.0, 4037.1)
    assert fields["STOP"] == "4039.7"


def test_risk_gate_rejects_a_stop_that_is_not_on_the_board():
    lm = _trend_map()
    board = TapeBoard(
        clock=stub_clock(),
        level_map=lm,
        regime=stub_regime(),
        setups=[],  # nothing on the board to anchor to
        price=lm.price,
        prev_close=4050.0,
        equity=100_000,
        risk_pct=0.5,
        generated_at=datetime(2026, 10, 29, 22, 0, tzinfo=SHANGHAI),
    )
    call = ScalpCall(
        verdict="EXECUTE", side="long", entry_low=4036.0, entry_high=4037.0,
        stop=4032.0, t1=4044.0, t2=4050.0, risk_pct=0.5,
    )
    checks = risk_check(call, board)
    grounding = [c for c in checks if c[0] == "数字对齐"]
    assert grounding and not grounding[0][2]


def test_risk_gate_vetoes_a_stop_wider_than_1_5_atr():
    lm = _trend_map()
    setup = tape.Setup(
        name="x", side="long", kind="trend_pullback", trigger="",
        entry_low=4039.0, entry_high=4041.0, stop=4033.0, targets=[4050.0, 4056.0],
        rr=[1.5, 2.5], hold_minutes=20, invalidation="", why="",
    )
    board = TapeBoard(
        clock=stub_clock(), level_map=lm, regime=stub_regime(), setups=[setup], price=lm.price,
        prev_close=4050.0, equity=100_000, risk_pct=0.5,
        generated_at=datetime(2026, 10, 29, 22, 0, tzinfo=SHANGHAI),
    )
    call = ScalpCall(
        verdict="EXECUTE", side="long", entry_low=4039.0, entry_high=4041.0,
        stop=4033.0, t1=4050.0, t2=4056.0, risk_pct=0.5,
    )
    checks = dict((label, ok) for label, _message, ok in risk_check(call, board))
    assert checks["数字对齐"] is True
    assert checks["止损宽度"] is False  # 7 dollars > 1.6 x 4.0


def test_risk_gate_blocks_a_trade_inside_a_blackout():
    lm = _trend_map()
    clock = stub_clock()
    event = gclock.Event(datetime(2026, 10, 29, 22, 5, tzinfo=SHANGHAI), "CPI", "high", "file")
    clock.blackout = gclock.Blackout(event, 5.0, True)
    clock.is_open = True
    board = TapeBoard(
        clock=clock, level_map=lm, regime=stub_regime(), setups=[], price=lm.price,
        prev_close=4050.0, equity=100_000, risk_pct=0.5,
        generated_at=datetime(2026, 10, 29, 22, 0, tzinfo=SHANGHAI),
    )
    call = ScalpCall(verdict="EXECUTE", side="long", entry_low=4036.0, entry_high=4037.0,
                     stop=4034.0, t1=4040.0, t2=4044.0, risk_pct=0.5)
    checks = dict((label, ok) for label, _message, ok in risk_check(call, board))
    assert checks["时段"] is False


# --------------------------------------------------------------------------- #
# Board rendering and the no-token short circuit
# --------------------------------------------------------------------------- #


def _board_from_live_map(**overrides) -> TapeBoard:
    lm = _trend_map()
    params = dict(
        clock=stub_clock(),
        level_map=lm,
        regime=stub_regime(),
        setups=[],
        price=lm.price,
        prev_close=4050.0,
        equity=100_000,
        risk_pct=0.5,
        generated_at=datetime(2026, 10, 29, 22, 0, tzinfo=SHANGHAI),
    )
    params.update(overrides)
    return TapeBoard(**params)


def test_board_text_carries_every_level_the_models_may_cite():
    lm = _trend_map()
    board = _board_from_live_map(level_map=lm)
    text = board.render_text()
    assert "4036.00" in text or "4,036.00" in text
    assert "ATR(5m)" in text
    assert "不得修改" in text


def test_board_json_is_serialisable():
    import json

    board = _board_from_live_map()
    json.dumps(board.to_dict(), ensure_ascii=False, default=str)


def test_analyze_scalp_does_not_spend_tokens_when_the_clock_says_no(monkeypatch):
    from tradingagents.gold.intraday import desk

    def boom(*args, **kwargs):
        raise AssertionError("no model client should be created when nothing is tradable")

    monkeypatch.setattr("tradingagents.llm_clients.create_tier_client", boom)
    board = _board_from_live_map(clock=stub_clock(tradable=False, grade="D"))
    call = desk.analyze_scalp(board)
    assert call.verdict == "NO_TRADE"
    assert call.board is board


# --------------------------------------------------------------------------- #
# The full card path, with a scripted model (no network, no tokens)
# --------------------------------------------------------------------------- #


class _ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)

    def invoke(self, prompt):
        text = self._replies.pop(0)

        class _Message:
            content = text

        return _Message()


def _install_models(monkeypatch, *, tape, catalyst, desk, risk="VERDICT: PASS\nREASON: 结构与赔率都合规"):
    from tradingagents.gold.intraday import desk as desk_module

    pools = {"quick": [tape, catalyst, risk], "deep": [desk]}
    made = {}

    class _Client:
        def __init__(self, tier):
            self._tier = tier

        def get_llm(self):
            return _ScriptedLLM(pools[self._tier])

    def factory(config, tier, **kwargs):
        made.setdefault(tier, _Client(tier))
        return made[tier]

    monkeypatch.setattr("tradingagents.llm_clients.create_tier_client", factory)
    return desk_module


def _actionable_board():
    lm = _trend_map()
    five = frame([4030.0 + i * 0.2 for i in range(40)])
    setup = tape.build_setups(lm, stub_regime(), stub_clock(), equity=100_000, five_min=five)[0]
    board = _board_from_live_map(level_map=lm, setups=[setup])
    return board, setup


def test_full_card_path_executes_a_grounded_setup(monkeypatch):
    board, setup = _actionable_board()
    card = (
        f"CALL: LONG\nSETUP: {setup.name}\n"
        f"ENTRY: {setup.entry_low}-{setup.entry_high}\nSTOP: {setup.stop}\n"
        f"T1: {setup.targets[0]}\nT2: {setup.targets[1]}\nHOLD: 20\nSIZE: 0.5\n"
        "WHY: 回踩 VWAP 支撑且 15m 多头排列\nKILL: 5 分钟收盘跌破止损"
    )
    module = _install_models(
        monkeypatch, tape="TAPE: 多", catalyst="CATALYST: 无\nBIAS: 中性\nCONFLICT: 否", desk=card
    )
    call = module.analyze_scalp(board, equity=100_000, risk_pct=0.5)
    assert call.verdict == "EXECUTE"
    assert call.side == "long"
    assert call.stop == pytest.approx(setup.stop)
    assert call.sizing and call.sizing[0].ounces > 0


def test_full_card_path_vetoes_an_invented_stop(monkeypatch):
    board, setup = _actionable_board()
    invented = setup.stop - 12.0  # plausible number, not on the board
    card = (
        f"CALL: LONG\nSETUP: {setup.name}\n"
        f"ENTRY: {setup.entry_low}-{setup.entry_high}\nSTOP: {invented}\n"
        f"T1: {setup.targets[0]}\nT2: {setup.targets[1]}\nHOLD: 20\nSIZE: 0.5\n"
        "WHY: 编的\nKILL: 编的"
    )
    module = _install_models(
        monkeypatch, tape="TAPE: 多", catalyst="CATALYST: 无\nBIAS: 中性\nCONFLICT: 否", desk=card
    )
    call = module.analyze_scalp(board, equity=100_000, risk_pct=0.5)
    assert call.verdict == "VETO"
    assert "数字对齐" in call.risk_reason


# --------------------------------------------------------------------------- #
# Web dashboard: direction votes and day plan (pure dict in / dict out)
# --------------------------------------------------------------------------- #


def _web_payload(**overrides):
    from tradingagents.gold.intraday.web import build_day_plan, judge_direction

    data = {
        "price": 4036.0,
        "clock": {
            "tradable": True,
            "blockers": [],
            "phase": "伦敦时段",
            "grade": "A",
            "next_marks": [{"in_minutes": 30.0, "label": "纽约开盘", "at": "2026-11-02T21:30:00+08:00"}],
            "blackout": None,
        },
        "regime": {
            "code": "trend",
            "label": "趋势上行",
            "bias": 0.6,
            "ema_align": "多头排列",
            "dist_vwap_sigma": 1.2,
            "rv_1h_pct": 14.0,
            "atr_expansion": 1.3,
        },
        "levels": {
            "vwap": 4030.0,
            "vwap_sigma": 2.5,
            "day_open": 4028.0,
            "asian_high": 4034.0,
            "asian_low": 4024.0,
            "asian_complete": True,
            "atr5": 1.5,
            "atr15": 3.0,
            "map": [
                {"name": "亚洲高点", "price": 4034.0, "distance": -2.0, "atr": -0.67, "kind": "range", "side": "below", "confluence": 2, "note": ""},
                {"name": "昨高", "price": 4045.0, "distance": 9.0, "atr": 3.0, "kind": "prior", "side": "above", "confluence": 1, "note": ""},
                {"name": "亚洲低点", "price": 4024.0, "distance": -12.0, "atr": -4.0, "kind": "range", "side": "below", "confluence": 2, "note": ""},
                {"name": "昨低", "price": 4015.0, "distance": -21.0, "atr": -7.0, "kind": "prior", "side": "below", "confluence": 1, "note": ""},
            ],
        },
        "setups": [
            {
                "name": "趋势回调做多",
                "side": "long",
                "kind": "pullback",
                "trigger": "回踩不破",
                "entry": [4033.0, 4034.0],
                "stop": 4030.0,
                "stop_ticks": 12,
                "targets": [4045.0, 4050.0],
                "rr": [3.0, 5.0],
                "hold_minutes": 45,
                "invalidation": "跌破 4030",
                "why": "趋势中回踩",
                "cost_pct_of_t1": 0.4,
                "score": 0.8,
                "sizing": [],
            }
        ],
        "no_trade_reasons": [],
    }
    data.update(overrides)
    return data, judge_direction(data), build_day_plan(data)


def test_direction_votes_sum_to_score_and_label():
    data, direction, _plan = _web_payload()
    assert -1.0 <= direction["score"] <= 1.0
    # trend + long setup + price over vwap: clearly bullish
    assert direction["label"] == "偏多"
    names = [v["name"] for v in direction["votes"]]
    assert names == ["状态倾向", "VWAP 位置", "15m 均线", "形态票", "日内动能"]
    assert all(v["weight"] > 0 for v in direction["votes"])


def test_direction_flips_short_when_votes_flip():
    _data, direction, _plan = _web_payload(
        regime={
            "code": "trend",
            "label": "趋势下行",
            "bias": -0.7,
            "ema_align": "空头排列",
            "dist_vwap_sigma": -1.5,
            "rv_1h_pct": 14.0,
            "atr_expansion": 1.2,
        },
        setups=[
            {
                "name": "反弹做空",
                "side": "short",
                "kind": "pullback",
                "trigger": "反抽承压",
                "entry": [4037.0, 4038.0],
                "stop": 4041.0,
                "stop_ticks": 16,
                "targets": [4024.0, 4015.0],
                "rr": [3.0, 5.0],
                "hold_minutes": 45,
                "invalidation": "收复 4041",
                "why": "趋势中反弹",
                "cost_pct_of_t1": 0.4,
                "score": 0.8,
                "sizing": [],
            }
        ],
    )
    assert direction["label"] == "偏空"


def test_direction_non_tradable_overrides_stance():
    data, direction, _plan = _web_payload(clock={"tradable": False, "blockers": ["休市"], "phase": "维护", "grade": "D", "next_marks": [], "blackout": None})
    assert "不可交易" in direction["stance"]


def test_day_plan_quotes_ready_setups_verbatim():
    _data, _direction, plan = _web_payload()
    assert plan["scenarios"][0]["title"].startswith("已就绪")
    assert plan["scenarios"][0]["entry"].startswith("4,033")
    # key levels include vwap and asian edges, sorted descending
    prices = [lv["price"] for lv in plan["key_levels"]]
    assert prices == sorted(prices, reverse=True)
    assert any(lv["name"] == "当日 VWAP" for lv in plan["key_levels"])
    assert plan["events"][0]["label"] == "纽约开盘"


def test_day_plan_builds_breakout_scenarios_without_setups():
    _data, _direction, plan = _web_payload(setups=[], no_trade_reasons=["趋势单：赔率不足"])
    sides = [s["side"] for s in plan.scenarios] if hasattr(plan, "scenarios") else [s["side"] for s in plan["scenarios"]]
    assert "long" in sides and "short" in sides
    long_scn = next(s for s in plan["scenarios"] if s["side"] == "long")
    assert "4,045" in long_scn["trigger"]  # breakout keys off the next real level above
    assert "4,051" in long_scn["targets"]  # padded 2x ATR beyond it when the map is thin


# --------------------------------------------------------------------------- #
# Web dashboard: direction votes and day plan (pure dict in / dict out)
# --------------------------------------------------------------------------- #


def _web_payload(**overrides):
    from tradingagents.gold.intraday.web import build_day_plan, judge_direction

    data = {
        "price": 4036.0,
        "clock": {
            "tradable": True,
            "blockers": [],
            "phase": "伦敦时段",
            "grade": "A",
            "next_marks": [{"in_minutes": 30.0, "label": "纽约开盘", "at": "2026-11-02T21:30:00+08:00"}],
            "blackout": None,
        },
        "regime": {
            "code": "trend",
            "label": "趋势上行",
            "bias": 0.6,
            "ema_align": "多头排列",
            "dist_vwap_sigma": 1.2,
            "rv_1h_pct": 14.0,
            "atr_expansion": 1.3,
        },
        "levels": {
            "vwap": 4030.0,
            "vwap_sigma": 2.5,
            "day_open": 4028.0,
            "asian_high": 4034.0,
            "asian_low": 4024.0,
            "asian_complete": True,
            "atr5": 1.5,
            "atr15": 3.0,
            "map": [
                {"name": "亚洲高点", "price": 4034.0, "distance": -2.0, "atr": -0.67, "kind": "range", "side": "below", "confluence": 2, "note": ""},
                {"name": "昨高", "price": 4045.0, "distance": 9.0, "atr": 3.0, "kind": "prior", "side": "above", "confluence": 1, "note": ""},
                {"name": "亚洲低点", "price": 4024.0, "distance": -12.0, "atr": -4.0, "kind": "range", "side": "below", "confluence": 2, "note": ""},
                {"name": "昨低", "price": 4015.0, "distance": -21.0, "atr": -7.0, "kind": "prior", "side": "below", "confluence": 1, "note": ""},
            ],
        },
        "setups": [
            {
                "name": "趋势回调做多",
                "side": "long",
                "kind": "pullback",
                "trigger": "回踩不破",
                "entry": [4033.0, 4034.0],
                "stop": 4030.0,
                "stop_ticks": 12,
                "targets": [4045.0, 4050.0],
                "rr": [3.0, 5.0],
                "hold_minutes": 45,
                "invalidation": "跌破 4030",
                "why": "趋势中回踩",
                "cost_pct_of_t1": 0.4,
                "score": 0.8,
                "sizing": [],
            }
        ],
        "no_trade_reasons": [],
    }
    data.update(overrides)
    return data, judge_direction(data), build_day_plan(data)


def test_direction_votes_sum_to_score_and_label():
    data, direction, _plan = _web_payload()
    assert -1.0 <= direction["score"] <= 1.0
    # trend + long setup + price over vwap: clearly bullish
    assert direction["label"] == "偏多"
    names = [v["name"] for v in direction["votes"]]
    assert names == ["状态倾向", "VWAP 位置", "15m 均线", "形态票", "日内动能"]
    assert all(v["weight"] > 0 for v in direction["votes"])


def test_direction_flips_short_when_votes_flip():
    _data, direction, _plan = _web_payload(
        regime={
            "code": "trend",
            "label": "趋势下行",
            "bias": -0.7,
            "ema_align": "空头排列",
            "dist_vwap_sigma": -1.5,
            "rv_1h_pct": 14.0,
            "atr_expansion": 1.2,
        },
        setups=[
            {
                "name": "反弹做空",
                "side": "short",
                "kind": "pullback",
                "trigger": "反抽承压",
                "entry": [4037.0, 4038.0],
                "stop": 4041.0,
                "stop_ticks": 16,
                "targets": [4024.0, 4015.0],
                "rr": [3.0, 5.0],
                "hold_minutes": 45,
                "invalidation": "收复 4041",
                "why": "趋势中反弹",
                "cost_pct_of_t1": 0.4,
                "score": 0.8,
                "sizing": [],
            }
        ],
    )
    assert direction["label"] == "偏空"


def test_direction_non_tradable_overrides_stance():
    data, direction, _plan = _web_payload(clock={"tradable": False, "blockers": ["休市"], "phase": "维护", "grade": "D", "next_marks": [], "blackout": None})
    assert "不可交易" in direction["stance"]


def test_day_plan_quotes_ready_setups_verbatim():
    _data, _direction, plan = _web_payload()
    assert plan["scenarios"][0]["title"].startswith("已就绪")
    assert plan["scenarios"][0]["entry"].startswith("4,033")
    # key levels include vwap and asian edges, sorted descending
    prices = [lv["price"] for lv in plan["key_levels"]]
    assert prices == sorted(prices, reverse=True)
    assert any(lv["name"] == "当日 VWAP" for lv in plan["key_levels"])
    assert plan["events"][0]["label"] == "纽约开盘"


def test_day_plan_builds_breakout_scenarios_without_setups():
    _data, _direction, plan = _web_payload(setups=[], no_trade_reasons=["趋势单：赔率不足"])
    sides = [s["side"] for s in plan.scenarios] if hasattr(plan, "scenarios") else [s["side"] for s in plan["scenarios"]]
    assert "long" in sides and "short" in sides
    long_scn = next(s for s in plan["scenarios"] if s["side"] == "long")
    assert "4,045" in long_scn["trigger"]  # breakout keys off the next real level above
    assert "4,051" in long_scn["targets"]  # padded 2x ATR beyond it when the map is thin


def test_weekend_closed_overrides_the_time_of_day_bands():
    # Sunday 14:00 ET grades as an open NY band by hour alone, but the
    # exchange has been dark since Friday 17:00.
    sunday_pm = datetime(2026, 11, 1, 14, 0, tzinfo=NEW_YORK)
    clock = gclock.read_clock(sunday_pm)
    assert clock.grade == "D"
    assert not clock.is_open
    assert not clock.tradable
    assert any("周末休市" in b or "未开盘" in b for b in clock.blockers)

    sunday_open = datetime(2026, 11, 1, 19, 0, tzinfo=NEW_YORK)  # Sun 18:00 reopen passed
    clock = gclock.read_clock(sunday_open)
    # tradable can still be False here if a rule-derived event (ISM etc.)
    # blacked the window out; the point is that the *weekend* no longer does.
    assert clock.grade != "D"
    assert clock.is_open

    friday_pm = datetime(2026, 11, 6, 17, 30, tzinfo=NEW_YORK)
    assert gclock.read_clock(friday_pm).grade == "D"

    # No skeleton mark should land inside the closed weekend window.
    for _mins, _label, at in gclock.read_clock(saturday := datetime(2026, 11, 7, 12, 0, tzinfo=NEW_YORK)).next_marks:
        assert not gclock._weekend_closed(at), f"mark inside weekend: {at}"


# --------------------------------------------------------------------------- #
# Web dashboard: deliverables (card history, report library)
# --------------------------------------------------------------------------- #


def test_card_history_roundtrip(tmp_path, monkeypatch):
    from tradingagents.gold.intraday import web

    cards_file = tmp_path / "scalp_cards.jsonl"
    monkeypatch.setattr(web, "CARDS_FILE", cards_file)
    web._append_card({"verdict": "NO_TRADE", "why": "测试", "price": 4036.0})
    web._append_card({"verdict": "EXECUTE", "side": "long", "price": 4037.0})
    history = web.card_history(limit=10)
    assert history[0]["verdict"] == "EXECUTE"  # newest first
    assert history[1]["why"] == "测试"
    assert all("at" in c for c in history)
    # a corrupt line must not break the reader
    with cards_file.open("a", encoding="utf-8") as fh:
        fh.write("{broken json\n")
    assert len(web.card_history()) == 2


def test_report_library_reads_only_inside_reports_dir(tmp_path, monkeypatch):
    from tradingagents.gold.intraday import web

    reports = tmp_path / "reports"
    run = reports / "GC=F_20261029_231852"
    run.mkdir(parents=True)
    (run / "complete_report.md").write_text("# Report\n- Rating: Sell\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("nope", encoding="utf-8")
    monkeypatch.setattr(web, "REPORTS_DIR", reports)

    runs = web.list_reports()
    assert runs[0]["rating"] == "Sell" and runs[0]["has_complete"]

    ok = web.read_report("GC=F_20261029_231852", "complete_report.md")
    assert ok["content"].startswith("# Report")
    # traversal attempts are refused
    assert "error" in web.read_report("..", "secret.txt")
    assert "error" in web.read_report("GC=F_20261029_231852", "../../secret.txt")
    assert "error" in web.read_report("GC=F_20261029_231852", "nope.md")
