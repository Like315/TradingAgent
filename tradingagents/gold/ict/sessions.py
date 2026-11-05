"""SMC session layer: Asian → London → New York, in that order.

ICT's textbook session model is America-centric; gold's real flow is not. The
trading day actually builds in three overlapping stages:

* **Asia** (Globex open → London open) — accumulation. The range formed here is
  the day's first liquidity map: its high and low are the two walls everyone
  watches.
* **London** (London open → COMEX open) — manipulation. The classic SMC read is
  the *Judas swing*: London's first hour tends to sweep one side of the Asian
  range, take the stops, and only then start the real move to the other side.
  This is the Power-of-Three (accumulation–manipulation–distribution).
* **New York** (COMEX open → halt) — distribution. The Asian sweep direction is
  usually the fake one; the real direction runs toward the untouched opposite
  wall.

The bias module consumes this to do two things the pure 1H model cannot:

1. pick the **draw on liquidity** from session walls (Asian high after an Asian
   low sweep, not some arbitrary pool);
2. grant a **session override**: bullish 1H structure stuck in premium is
   normally "wait", but if London just swept the Asian low and reclaimed it,
   that IS the discount entry — the manipulation revealed the direction.

All boundaries are derived from real exchange timezones (Europe/London for the
London open, America/New_York for COMEX marks, Asia/Tokyo for the Asian
killzone) so DST mismatches — the two weeks a year the US and EU change clocks
on different dates — stay correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.gold.intraday import feed

TOKYO = ZoneInfo("Asia/Tokyo")
LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")

# A pierce smaller than this on a session wall is noise, not manipulation.
# Same constant as the pool sweep definition — one threshold, one idea.
MIN_PIERCE_ATR = 0.08


@dataclass
class SessionRange:
    """One session's extremes on the current trading day."""

    name: str  # asian | london | new_york
    label: str
    high: float | None = None
    low: float | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None  # None while still running
    complete: bool = False

    @property
    def mid(self) -> float | None:
        return None if self.high is None else (self.high + self.low) / 2.0

    @property
    def status(self) -> str:
        if self.high is None:
            return "未开始"
        return "已结束" if self.complete else "进行中"


@dataclass
class Manipulation:
    """The Power-of-Three state: which Asian wall London/NY took out."""

    swept_side: str = ""  # "low" | "high" | ""
    extreme: float | None = None  # the wick extreme of the sweep
    pierce_atr: float = 0.0
    reclaimed: bool = False
    reclaim_time: pd.Timestamp | None = None
    by_session: str = ""  # london | new_york
    bars_since: int = 0

    @property
    def label(self) -> str:
        if not self.swept_side:
            return "未扫"
        wall = "亚洲低点" if self.swept_side == "low" else "亚洲高点"
        state = "已收回" if self.reclaimed else "未收回"
        return f"{self.by_session} 扫{wall}（{state}）"

    @property
    def expected_direction(self) -> str:
        """The side of the PO3: swept the low → the real move is up."""
        if self.swept_side == "low" and self.reclaimed:
            return "long"
        if self.swept_side == "high" and self.reclaimed:
            return "short"
        return ""


KILLZONES = (
    ("asian", "亚洲 killzone（东京上午）", TOKYO, time(9, 0), time(11, 0)),
    ("london", "伦敦 killzone", LONDON, time(8, 0), time(11, 0)),
    ("new_york", "纽约 killzone（COMEX 开盘）", NEW_YORK, time(8, 20), time(11, 0)),
)


def killzone(now: datetime) -> tuple[str, str] | None:
    """The killzone ``now`` falls in, if any — SMC entries want these windows."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=feed.SHANGHAI)
    for key, label, zone, start, end in KILLZONES:
        local = now.astimezone(zone)
        if start <= local.time() < end:
            return key, label
    return None


def _localize(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Bars carry naive Beijing timestamps; make them Shanghai-aware."""
    if index.tz is None:
        return index.tz_localize(feed.SHANGHAI)
    return index.tz_convert(feed.SHANGHAI)


def _trading_day(now: datetime) -> pd.Timestamp:
    """The Globex day ``now`` belongs to: starts 18:00 ET, ends 17:00 ET."""
    et = now.astimezone(NEW_YORK)
    day = et.date()
    if et.time() < time(17, 0):
        day -= timedelta(days=1)
    return pd.Timestamp(day)


def _bounds(day: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Session boundaries for one trading day, as Beijing timestamps."""
    et_day = day.to_pydatetime().date()
    globex = datetime.combine(et_day, time(18, 0), tzinfo=NEW_YORK)
    comex = datetime.combine(et_day + timedelta(days=1), time(8, 20), tzinfo=NEW_YORK)
    halt = datetime.combine(et_day + timedelta(days=1), time(17, 0), tzinfo=NEW_YORK)
    # London open: the 08:00 London instant on the calendar day after the
    # Globex open (the Globex day spans two London mornings; the second one is
    # the real London session).
    ldn_day = (et_day + timedelta(days=1)) if globex.hour >= 12 else et_day
    ldn_open = datetime.combine(et_day + timedelta(days=1), time(8, 0), tzinfo=LONDON)
    del ldn_day
    return {
        "globex": pd.Timestamp(globex).tz_convert(feed.SHANGHAI),
        "london": pd.Timestamp(ldn_open).tz_convert(feed.SHANGHAI),
        "comex": pd.Timestamp(comex).tz_convert(feed.SHANGHAI),
        "halt": pd.Timestamp(halt).tz_convert(feed.SHANGHAI),
    }


@dataclass
class Sessions:
    """Everything the SMC layer knows about today's session structure."""

    asian: SessionRange
    london: SessionRange
    new_york: SessionRange
    manipulation: Manipulation = field(default_factory=Manipulation)
    killzone: tuple[str, str] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def po3_phase(self) -> str:
        if not self.asian.complete:
            return "积累（亚洲区间进行中）"
        if not self.manipulation.swept_side:
            return "等操纵（伦敦还没扫边）"
        if not self.manipulation.reclaimed:
            return "操纵中（扫了还没收回）"
        return "派发（扫完收回，真方向启动）"

    def to_dict(self) -> dict:
        return {
            "killzone": None if not self.killzone else {"key": self.killzone[0], "label": self.killzone[1]},
            "po3_phase": self.po3_phase,
            "asian": self._range_dict(self.asian),
            "london": self._range_dict(self.london),
            "new_york": self._range_dict(self.new_york),
            "manipulation": {
                "label": self.manipulation.label,
                "swept_side": self.manipulation.swept_side,
                "reclaimed": self.manipulation.reclaimed,
                "by_session": self.manipulation.by_session,
                "extreme": self.manipulation.extreme,
                "pierce_atr": round(self.manipulation.pierce_atr, 2),
                "bars_since": self.manipulation.bars_since,
                "expected_direction": self.manipulation.expected_direction,
            },
            "notes": self.notes,
        }

    @staticmethod
    def _range_dict(r: SessionRange) -> dict:
        return {
            "label": r.label,
            "high": r.high,
            "low": r.low,
            "mid": r.mid,
            "status": r.status,
            "start": r.start.isoformat() if r.start is not None else None,
            "end": r.end.isoformat() if r.end is not None else None,
        }


def _empty_sessions(now: datetime) -> Sessions:
    return Sessions(
        asian=SessionRange("asian", "亚洲时段"),
        london=SessionRange("london", "伦敦/欧洲时段"),
        new_york=SessionRange("new_york", "纽约时段"),
        killzone=killzone(now),
    )


def compute_sessions(ctx) -> Sessions:
    """Asian / London / NY ranges for the current trading day, from 15m bars."""
    bars = ctx.bars15
    now = ctx.now
    if bars.empty:
        return _empty_sessions(now)

    index = _localize(bars.index)
    b = bars.set_axis(index)

    day = _trading_day(now)
    bounds = _bounds(day)
    today = b[(b.index >= bounds["globex"]) & (b.index < bounds["halt"])]
    if today.empty:
        # Session just rolled over; the previous day's data is the context.
        prev = _bounds(day - pd.Timedelta(days=1))
        today = b[(b.index >= prev["globex"]) & (b.index < prev["halt"])]
        bounds = prev
        if today.empty:
            return _empty_sessions(now)

    sessions = _empty_sessions(now)
    atr = ctx.atr

    def _fill(rng: SessionRange, lo: pd.Timestamp, hi: pd.Timestamp | None) -> None:
        part = today[today.index >= lo] if hi is None else today[(today.index >= lo) & (today.index < hi)]
        if part.empty:
            return
        rng.start = part.index[0]
        rng.high, rng.low = float(part["high"].max()), float(part["low"].min())
        rng.complete = hi is not None and part.index[-1] >= hi - pd.Timedelta(minutes=15) or (
            hi is not None and now >= hi.to_pydatetime()
        )
        rng.end = hi if rng.complete else None

    _fill(sessions.asian, bounds["globex"], bounds["london"])
    _fill(sessions.london, bounds["london"], bounds["comex"])
    _fill(sessions.new_york, bounds["comex"], None)
    if sessions.asian.complete:
        sessions.notes.append(f"亚洲区间 {sessions.asian.low:,.2f}–{sessions.asian.high:,.2f}，是今天第一张流动性地图")

    sessions.manipulation = _detect_manipulation(sessions, today, bounds, atr, now)
    return sessions


def _detect_manipulation(
    sessions: Sessions,
    today: pd.DataFrame,
    bounds: dict[str, pd.Timestamp],
    atr: float,
    now: datetime,
) -> Manipulation:
    """Which Asian wall did London (or NY) pierce and reclaim after the open?

    The sweep must happen *after* the London open — the Asian session merely
    building its range is not manipulation. The reclaim is the tell: a wick
    through the wall that closes back inside says the stops were the target.
    """
    result = Manipulation()
    asian = sessions.asian
    if asian.high is None or not sessions.asian.complete:
        return result

    post = today[today.index >= bounds["london"]]
    if post.empty:
        return result

    best: Manipulation | None = None
    for side, wall in (("low", asian.low), ("high", asian.high)):
        if wall is None:
            continue
        for offset, (stamp, bar) in enumerate(post.iterrows()):
            if side == "low":
                pierce = wall - float(bar["low"])
                back_inside = float(bar["close"]) > wall
            else:
                pierce = float(bar["high"]) - wall
                back_inside = float(bar["close"]) < wall
            if pierce < MIN_PIERCE_ATR * atr:
                continue

            reclaim_time = None
            if back_inside:
                reclaim_time = stamp
            else:
                for later in post.index[offset + 1 : offset + 4]:
                    later_bar = post.loc[later]
                    ok = float(later_bar["close"]) > wall if side == "low" else float(later_bar["close"]) < wall
                    if ok:
                        reclaim_time = later
                        break

            extreme = (
                float(post.iloc[offset : (offset + 4)]["low"].min())
                if side == "low"
                else float(post.iloc[offset : (offset + 4)]["high"].max())
            )
            m = Manipulation(
                swept_side=side,
                extreme=extreme,
                pierce_atr=pierce / atr,
                reclaimed=reclaim_time is not None,
                reclaim_time=reclaim_time,
                by_session="london" if stamp < bounds["comex"] else "new_york",
                bars_since=len(post) - 1 - (post.index.get_loc(reclaim_time) if reclaim_time is not None else offset),
            )
            # Prefer the freshest reclaimed sweep; among unreclaimed, the deepest.
            if best is None or (m.reclaimed and not best.reclaimed) or (m.reclaimed == best.reclaimed and m.pierce_atr > best.pierce_atr):
                best = m

    return best or result
