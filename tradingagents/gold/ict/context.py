"""One fetch, both timeframes: the numbers the ICT model is allowed to use.

1H drives the bias, 15m drives the trigger, and the daily series supplies the
levels that survive a single session (prior day high/low). Everything is
computed once per refresh so the bias and the trigger cannot disagree about
what ATR is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from tradingagents.gold.intraday import feed
from tradingagents.gold.intraday.clock import SessionClock, read_clock
from tradingagents.gold.intraday.levels import LevelMap, atr, build_level_map


@dataclass
class MarketContext:
    now: datetime
    clock: SessionClock
    price: float
    atr15: float
    atr1h: float
    levels: LevelMap
    bars15: pd.DataFrame
    bars1h: pd.DataFrame
    source: str = ""
    stale: bool = False
    has_volume: bool = True
    bar_age: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def atr(self) -> float:
        """The unit every distance is quoted in: 15m ATR, the trigger frame."""
        return self.atr15 or 1.0

    def dollars(self, price: float | None, reference: float | None = None) -> float:
        ref = self.price if reference is None else reference
        return 0.0 if price is None else price - ref

    def in_r(self, price: float | None, reference: float | None = None) -> float:
        """Distance in 15m-ATR units — the only scale that compares across days."""
        return 0.0 if price is None else self.dollars(price, reference) / self.atr


def build_context(now: datetime | None = None, *, fresh: bool = False) -> MarketContext:
    """Fetch 1H, 15m and daily bars and derive the shared units."""
    bars15 = feed.bars("15m", 800, fresh=fresh)
    bars1h = feed.bars("60m", 600, fresh=fresh)
    daily = feed.daily_bars(120, fresh=fresh)
    if bars15.empty:
        raise feed.FeedUnavailable("15 分钟线为空，ICT 模型无法建立触发")
    if bars1h.empty:
        raise feed.FeedUnavailable("1 小时线为空，ICT 模型无法建立偏置")

    # The level map is reused with 15m bars standing in for every intraday
    # frame: it gives the session/Asian range, prior day extremes and swing
    # points with the London-DST logic already correct. Its ATR fields are
    # ignored here — this module computes its own 15m and 1H ATR below.
    levels = build_level_map(bars15, bars15, bars15, daily)
    price = float(bars15["close"].iloc[-1])
    notes: list[str] = []
    if not bool(bars15.attrs.get("has_volume", True)):
        notes.append("数据源无成交量：扫单只能靠影线+收回定义，无法用量能确认")
    if bars15.attrs.get("stale"):
        notes.append("行情延迟，偏置仅供复盘")

    return MarketContext(
        now=now or datetime.now(feed.SHANGHAI),
        clock=read_clock(now),
        price=price,
        atr15=atr(bars15),
        atr1h=atr(bars1h),
        levels=levels,
        bars15=bars15,
        bars1h=bars1h,
        source=str(bars15.attrs.get("source") or ""),
        stale=bool(bars15.attrs.get("stale", False)),
        has_volume=bool(bars15.attrs.get("has_volume", True)),
        bar_age=feed.frame_age_minutes(bars15),
        notes=notes,
    )
