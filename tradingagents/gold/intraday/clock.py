"""The intraday clock: which session is live, and what is about to hit the tape.

A scalper's first question is never "what is gold worth", it is "is this a good
minute to be in the market". That depends on two clocks:

* the **liquidity clock** — gold trades 23 hours a day but only a few of them
  have enough depth to enter and exit without paying the spread twice;
* the **event clock** — a CPI print at 08:30 ET turns a clean level into a
  random number for ninety seconds, and being filled two minutes before it is
  the single most common way a scalping day goes wrong.

Everything is computed from the IANA tz database rather than hardcoded offsets,
because the US and EU switch DST on different dates: for about three weeks a
year London opens at 04:00 New York time instead of 03:00. Hardcoding 03:00
would put the "London open" marker an hour wrong exactly when it matters.

Beijing time is the display currency of this module (the desk's audience is in
China), so every mark carries both.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

from zoneinfo import ZoneInfo

from tradingagents.gold.intraday import feed

logger = logging.getLogger(__name__)

SHANGHAI = feed.SHANGHAI
NEW_YORK = feed.NEW_YORK
LONDON = ZoneInfo("Europe/London")

# Liquidity bands, in New York local time. Graded from how much depth the book
# carries, not from a guess about direction: A is where a 1-lot market order
# costs one tick, D is where it costs ten.
#   (start_hour, end_hour, grade, label, note)
_BANDS = (
    (17.0, 18.0, "D", "日终维护", "COMEX 休市，无法交易"),
    (18.0, 20.0, "C", "亚洲早盘", "澳洲/日本未开，盘口薄，容易假突破"),
    (20.0, 2.0, "B", "亚洲主盘", "东京+上海时段，区间通常在这里成型"),
    (2.0, 3.0, "C", "欧洲盘前", "流动性低谷，等伦敦方向"),
    (3.0, 5.0, "A", "伦敦盘", "伦敦开盘定方向，趋势单胜率最高的一段"),
    (5.0, 8.2, "B", "伦敦午盘", "欧盘午休，常出现回抽"),
    (8.2, 11.5, "A", "COMEX 开盘 + 数据窗口", "全天最厚，突破/回踩都成立"),
    (11.5, 13.5, "B", "美盘午前", "趋势延续段，注意动能衰减"),
    (13.5, 16.0, "C", "美盘午后", "结算后流动性下降，区间为主"),
    (16.0, 17.0, "C", "NY 收盘调仓", "机构调仓，15:00-16:00 常见反向脉冲"),
)

_GRADE_NOTE = {
    "A": "可以下手：点差成本最低，止损最可靠",
    "B": "可做但降规模：等触发确认，别挂提前单",
    "C": "只做最好的形态：宁可错过",
    "D": "不交易：休市或薄到无法止损",
}


@dataclass(frozen=True)
class Mark:
    """A fixed time-of-day event in the trading day (open, fix, settle...)."""

    key: str
    label: str
    hour: int
    minute: int
    zone: str  # "NY" or "LDN"
    note: str = ""


# The skeleton of every COMEX gold session, in the exchange's own local times.
_MARKS = (
    Mark("globex_open", "COMEX 电子盘开盘（新交易日起点）", 18, 0, "NY", "日开盘价是当天多空的第一条分界"),
    Mark("london_open", "伦敦开盘", 8, 0, "LDN", "欧洲资金进场，方向确认窗口"),
    Mark("comex_open", "COMEX 公开喊价时段开盘", 8, 20, "NY", "全天成交量峰值起点"),
    Mark("us_data", "美国数据窗口 08:30 ET", 8, 30, "NY", "CPI/非农/初请都落在这里，前后 15 分钟不进新单"),
    Mark("london_fix", "伦敦金定盘价（AM）", 10, 30, "LDN", "定盘前后常有 3-5 美元的瞬时抽动"),
    Mark("comex_settle", "COMEX 结算", 13, 30, "NY", "结算价决定当日收盘基准"),
    Mark("ny_close", "NY 收盘 / 机构调仓", 16, 0, "NY", "调仓脉冲常反向，别追"),
    Mark("globex_halt", "日终维护开始", 17, 0, "NY", "05:00-06:00 北京不可交易"),
)

_ZONE = {"NY": NEW_YORK, "LDN": LONDON}


def _zone(name: str) -> ZoneInfo:
    return _ZONE[name]


def _mark_datetimes(mark: Mark, around: datetime) -> list[datetime]:
    """The mark's occurrences on the previous, current and next local day."""
    zone = _zone(mark.zone)
    local_now = around.astimezone(zone)
    out = []
    for offset in (-1, 0, 1):
        day = (local_now + timedelta(days=offset)).date()
        out.append(datetime.combine(day, time(mark.hour, mark.minute), tzinfo=zone))
    return out


def _band(et_hour: float) -> tuple[str, str, str]:
    for start, end, grade, label, note in _BANDS:
        if start < end:
            if start <= et_hour < end:
                return grade, label, note
        else:  # crosses midnight
            if et_hour >= start or et_hour < end:
                return grade, label, note
    return "C", "过渡时段", ""


def _weekend_closed(ny: datetime) -> bool:
    """True from Friday 17:00 ET until Sunday 18:00 ET — the one stretch the
    time-of-day bands cannot see: Sunday 14:00 ET grades as "美盘" by hour but
    the exchange has been dark since Friday."""
    hour = ny.hour + ny.minute / 60.0
    weekday = ny.weekday()  # Mon=0 .. Sun=6
    if weekday == 4 and hour >= 17.0:  # Friday, after the halt
        return True
    if weekday == 5:  # Saturday
        return True
    if weekday == 6 and hour < 18.0:  # Sunday, before the Globex reopen
        return True
    return False


# --------------------------------------------------------------------------- #
# Event calendar
# --------------------------------------------------------------------------- #

EVENTS_FILE = Path(os.getenv("TRADINGAGENTS_GOLD_HOME") or Path.home() / ".tradingagents" / "gold") / "events.json"

_EVENTS_TEMPLATE = {
    "_howto": (
        "把已知的高影响事件按下面格式加进 events 数组（当地时间用 timezone 指定的时区）。"
        "CPI 和 FOMC 的日期无法由规则推算，必须手工填；填了之后模拟盘会在事件前 10 分钟到后 15 分钟自动禁止开新仓。"
    ),
    "timezone": "America/New_York",
    "events": [
        # {"date": "2026-11-12", "time": "08:30", "label": "CPI 同比", "impact": "high"}
    ],
}


@dataclass(frozen=True)
class Event:
    when: datetime  # tz-aware
    label: str
    impact: str  # "high" | "medium" | "low"
    source: str  # "rule" | "file"

    @property
    def blackout_before(self) -> timedelta:
        return timedelta(minutes=10 if self.impact == "high" else 5)

    @property
    def blackout_after(self) -> timedelta:
        return timedelta(minutes=15 if self.impact == "high" else 10)


def _first_business_day(day: date) -> date:
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _last_business_day(day: date) -> date:
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th given weekday of a month (n=1 -> first)."""
    day = date(year, month, 1)
    shift = (weekday - day.weekday()) % 7
    return day + timedelta(days=shift + 7 * (n - 1))


def _dates_in_window(start: date, days: int):
    for offset in range(days):
        yield start + timedelta(days=offset)


def recurring_events(start: date, days: int = 3) -> list[Event]:
    """Events that follow a calendar rule, so they need no data feed.

    Covers the weekly jobless claims, the first-Friday payrolls report, ISM on
    the first business day and the month-end rebalance. CPI and FOMC are *not*
    here: their dates move with the release calendar and guessing them would be
    worse than saying nothing, so they come from ``events.json`` instead.
    """
    out: list[Event] = []
    for day in _dates_in_window(start - timedelta(days=1), days + 2):
        # initial claims: every Thursday 08:30 ET
        if day.weekday() == 3:
            out.append(Event(datetime.combine(day, time(8, 30), NEW_YORK), "初请失业金", "medium", "rule"))
        # non-farm payrolls: first Friday 08:30 ET
        if day.weekday() == 4 and day.day <= 7:
            out.append(Event(datetime.combine(day, time(8, 30), NEW_YORK), "非农就业", "high", "rule"))
        # ISM manufacturing: first business day 10:00 ET
        if day == _first_business_day(date(day.year, day.month, 1)):
            out.append(Event(datetime.combine(day, time(10, 0), NEW_YORK), "ISM 制造业", "medium", "rule"))
        # month-end rebalance
        nxt = date(day.year + (day.month == 12), day.month % 12 + 1, 1)
        if day == _last_business_day(nxt - timedelta(days=1)):
            out.append(Event(datetime.combine(day, time(15, 0), NEW_YORK), "月末调仓流", "medium", "rule"))
    return out


def load_custom_events() -> list[Event]:
    """Hand-maintained events (CPI, FOMC, ...) from ``events.json``.

    A missing or broken file returns an empty list: the calendar degrades to the
    rule-based events and the board says so, rather than the whole run failing
    because one JSON comma was wrong.
    """
    if not EVENTS_FILE.exists():
        try:
            EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            EVENTS_FILE.write_text(json.dumps(_EVENTS_TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.debug("could not create %s: %s", EVENTS_FILE, exc)
        return []
    try:
        payload = json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s is not valid JSON (%s); ignoring custom events", EVENTS_FILE, exc)
        return []
    zone = ZoneInfo(payload.get("timezone") or "America/New_York")
    out = []
    for row in payload.get("events") or []:
        try:
            when = datetime.combine(
                date.fromisoformat(row["date"]),
                time.fromisoformat(row.get("time") or "08:30"),
                tzinfo=zone,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping malformed event %r: %s", row, exc)
            continue
        out.append(Event(when, str(row.get("label") or "事件"), str(row.get("impact") or "high"), "file"))
    return out


def upcoming_events(now: datetime, horizon_hours: int = 30) -> list[Event]:
    start = now.astimezone(NEW_YORK).date()
    events = recurring_events(start, days=max(3, horizon_hours // 24 + 2)) + load_custom_events()
    limit = now + timedelta(hours=horizon_hours)
    alive = [e for e in events if now - e.blackout_after <= e.when <= limit]
    return sorted(alive, key=lambda e: e.when)


@dataclass(frozen=True)
class Blackout:
    event: Event
    minutes_to: float  # negative when the event already passed
    active: bool

    @property
    def text(self) -> str:
        if self.active:
            if self.minutes_to >= 0:
                return f"距 {self.event.label} 还有 {self.minutes_to:.0f} 分钟：禁止开新仓"
            return f"{self.event.label} 已过去 {abs(self.minutes_to):.0f} 分钟：等波动收敛再进"
        return f"{self.event.label}，还有 {self.minutes_to:.0f} 分钟"


def blackout(now: datetime, events: list[Event] | None = None) -> Blackout | None:
    """The event whose no-trade window contains ``now``, else the next one."""
    events = events if events is not None else upcoming_events(now)
    for event in events:
        if event.when - event.blackout_before <= now <= event.when + event.blackout_after:
            return Blackout(event, (event.when - now).total_seconds() / 60.0, True)
    ahead = [e for e in events if e.when > now]
    if ahead:
        event = ahead[0]
        return Blackout(event, (event.when - now).total_seconds() / 60.0, False)
    return None


# --------------------------------------------------------------------------- #
# The clock
# --------------------------------------------------------------------------- #


@dataclass
class SessionClock:
    now: datetime
    phase_label: str
    phase_note: str
    grade: str
    grade_note: str
    is_open: bool
    minutes_since_open: int
    minutes_to_halt: int
    next_marks: list[tuple[float, str, datetime]] = field(default_factory=list)
    blackout: Blackout | None = None
    events: list[Event] = field(default_factory=list)

    @property
    def tradable(self) -> bool:
        if not self.is_open:
            return False
        if self.blackout is not None and self.blackout.active:
            return False
        return self.grade in ("A", "B")

    @property
    def blockers(self) -> list[str]:
        out = []
        if not self.is_open:
            out.append(f"COMEX 未开盘：{self.phase_label}（{self.phase_note}）")
        if self.grade == "D":
            out.append("流动性 D 级")
        elif self.grade == "C":
            out.append("流动性 C 级：只做最好的形态")
        if self.blackout is not None and self.blackout.active:
            out.append(self.blackout.text)
        return out

    def bj(self, stamp: datetime) -> str:
        return stamp.astimezone(SHANGHAI).strftime("%H:%M")


def read_clock(now: datetime | None = None) -> SessionClock:
    now = (now or datetime.now(SHANGHAI)).astimezone(SHANGHAI)
    ny = now.astimezone(NEW_YORK)
    et_hour = ny.hour + ny.minute / 60.0
    grade, label, note = _band(et_hour)
    if _weekend_closed(ny):
        grade, label, note = "D", "周末休市", "周五 17:00 → 周日 18:00（纽约时间）COMEX 全程休市"
    is_open = grade != "D"

    # Next marks: the next occurrences of every skeleton time, soonest first.
    marks: list[tuple[float, str, datetime]] = []
    for mark in _MARKS:
        for candidate in _mark_datetimes(mark, now):
            if candidate > now:
                marks.append(((candidate - now).total_seconds() / 60.0, mark.label, candidate))
                break
    # A mark that lands inside the weekend (e.g. "Saturday 18:00 reopen") is a
    # skeleton artefact, not a session: drop it.
    marks = [m for m in marks if not _weekend_closed(m[2])]
    marks.sort()

    # Minutes since the 18:00 ET reopen, and to the 17:00 halt.
    reopen_candidates = [c for c in _mark_datetimes(_MARKS[0], now) if c <= now]
    halt_candidates = [c for c in _mark_datetimes(_MARKS[-1], now) if c > now]
    if is_open:
        since_open = int((now - reopen_candidates[-1]).total_seconds() // 60) if reopen_candidates else 0
        to_halt = int((halt_candidates[0] - now).total_seconds() // 60) if halt_candidates else 0
    else:
        # Closed: both counters are meaningless (the last "reopen" may be two
        # days away) and would read as stale/odd numbers on the board.
        since_open = 0
        to_halt = 0

    return SessionClock(
        now=now,
        phase_label=label,
        phase_note=note,
        grade=grade,
        grade_note=_GRADE_NOTE[grade],
        is_open=is_open,
        minutes_since_open=since_open,
        minutes_to_halt=to_halt,
        next_marks=marks[:5],
        blackout=blackout(now),
        events=upcoming_events(now, horizon_hours=36),
    )
