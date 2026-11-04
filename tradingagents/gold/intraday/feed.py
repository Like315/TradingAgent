"""Intraday bars for COMEX gold, with a three-tier source chain.

The daily pipeline in :mod:`tradingagents.gold.data` is built around one row per
session, which is useless to a scalper. This module adds the minute-level feed
the scalp desk runs on.

**Why three tiers.** Eastmoney's K-line endpoint is the only reachable source
that returns true 1/5/15/60-minute OHLC *with volume*, but it drops the
connection for every host once a client asks too often — probing it with a
handful of rapid requests is enough to earn a few minutes of silence. So the
feed degrades instead of failing:

1. **Eastmoney** — full OHLCV, 5 days of 5-minute history;
2. **Sina minute line** — one request returns the whole session as a price
   series plus the exchange's running average (a true session VWAP). No volume,
   and no intrabar high/low, so frames from here are marked ``has_volume=False``
   and every volume-gated setup switches itself off;
3. **disk cache** — the last good frame, carrying its age so the board can say
   "this is 7 minutes old" instead of quietly trading a stale price.

Callers never need to know which tier answered: every frame carries
``df.attrs = {source, stale, has_volume, age_minutes, note}`` and the board
prints it.

Two details are not obvious and are pinned by tests, because getting either
wrong corrupts everything downstream:

* Eastmoney returns ``time, open, close, high, low, volume, ...`` — **close
  before high/low**. Reading column 3 as the high would put a fake wick on half
  the bars.
* Bar timestamps are Beijing wall clock. A COMEX trading day opens 18:00 New
  York and runs 23 hours, so "today" starts at 06:00 Beijing in summer and
  07:00 in winter; :func:`session_day` derives that from the tz database
  instead of hardcoding the offset.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")

# COMEX front-month continuous contract.
SECID = "101.GC00Y"

# First host that answers wins. The bare host is unreliable; the numbered ones
# rotate, so a longer list means a better chance one is outside its cooldown.
HOSTS = (
    "1.push2his.eastmoney.com",
    "7.push2his.eastmoney.com",
    "13.push2his.eastmoney.com",
    "29.push2his.eastmoney.com",
    "41.push2his.eastmoney.com",
    "61.push2his.eastmoney.com",
    "82.push2his.eastmoney.com",
    "push2his.eastmoney.com",
)
KLINE_URL = "https://{host}/api/qt/stock/kline/get"

SINA_MINLINE = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
    "var%20_GC=/GlobalFuturesService.getGlobalFuturesMinLine?symbol=GC"
)
SINA_DAILY = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
    "var%20_GC=/GlobalFuturesService.getGlobalFuturesDailyKLine?symbol=GC"
)

INTERVALS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60", "1d": "101"}
DEFAULT_LIMITS = {"1m": 1200, "5m": 3000, "15m": 1200, "30m": 800, "60m": 500, "1d": 400}

# A scalper refreshing every 30 seconds should not turn into a request per second.
CACHE_TTL = {"1m": 20.0, "5m": 45.0, "15m": 90.0, "30m": 120.0, "60m": 120.0, "1d": 300.0}
TIMEOUT = 15
# A cached frame older than this is not worth showing for a 20-minute trade.
MAX_CACHE_AGE_MINUTES = 180

# Eastmoney drops the connection for the whole IP after a burst, so the feed
# polices itself: at least this long between requests, and a full cooling-off
# period after a tier-wide failure. Without this, a 30-second refresh loop walks
# straight into the ban and stays there.
MIN_REQUEST_SPACING = 0.5
COOLDOWN_SECONDS = 150.0
_last_request_at = 0.0
_cooldown_until = 0.0

CACHE_DIR = Path.home() / ".tradingagents" / "gold" / "cache" / "intraday"

# Closes-only frames lose the intrabar wick, which understates ATR. 1.3 is a
# deliberately conservative placeholder: gold's one-minute wick is small but not
# zero, and a wider ATR only makes stops wider and size smaller, which is the
# safe direction. Honest caveat: it has **not** been calibrated against the true
# 5-minute OHLC on this machine — the primary feed was rate-limited while this
# was written — so when the board says "ATR 估算", treat it as an estimate and
# prefer running while the Eastmoney tier is healthy.
WICK_ADJUST = 1.3

_CACHE: dict[tuple[str, int], tuple[float, pd.DataFrame]] = {}

_session = requests.Session()
_session.trust_env = False  # the machine's proxy 502s these hosts intermittently
_session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://quote.eastmoney.com/",
    }
)
_sina = requests.Session()
_sina.trust_env = False
_sina.headers.update(
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://finance.sina.com.cn"}
)


class FeedUnavailable(RuntimeError):
    """No tier answered. Callers must degrade, never invent prices."""


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _parse_klines(klines: list[str]) -> pd.DataFrame:
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        rows.append(
            {
                "time": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),  # eastmoney order: open, close, high, low
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5] or 0.0),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%d %H:%M", errors="coerce")
    frame = frame.dropna(subset=["time"]).set_index("time").sort_index()
    frame = frame[["open", "high", "low", "close", "volume"]]
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.astype(float)


def _parse_sina_minline(text: str) -> pd.DataFrame:
    """Sina's intraday line -> closes-only frame with unit volume.

    Payload is a JSONP object whose ``minLine_1d`` list holds one row per minute:
    ``[HH:MM, price, _, _, running_average, "YYYY-MM-DD HH:MM:00"]``. The first
    row carries the previous close and the session open instead of a time, so it
    is detected by length rather than by position.

    Volume is set to 1.0 for every bar on purpose. It is not a lie about volume:
    it is the signal to the rest of the package that this frame is price-only,
    and it makes VWAP and σ fall back to the equal-weighted mean and standard
    deviation, which is the correct statistic for a sample of prices.
    """
    start, end = text.find("("), text.rfind(")")
    if start < 0 or end <= start:
        return pd.DataFrame()
    try:
        payload = json.loads(text[start + 1 : end])
    except json.JSONDecodeError:
        return pd.DataFrame()
    rows = payload.get("minLine_1d") or []
    records = []
    for row in rows:
        if len(row) < 6:
            continue
        stamp = pd.to_datetime(row[-1], errors="coerce")
        if pd.isna(stamp):
            continue
        # The first row is a summary, not a print: [date, <other contract's
        # close>, "cme", "", "06:00", open, 0, 0, running_avg, timestamp]. The
        # real first print is field 5; field 1 (4051.6 on the session checked)
        # does not match the daily open of 4081.1 and must not become the day's
        # open, or every opening-range level is wrong by $30.
        field = 5 if len(row) >= 10 else 1
        try:
            price = float(row[field])
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue
        records.append({"time": stamp, "price": price})
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).set_index("time").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    price = frame["price"]
    out = pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": 1.0}
    )
    out.index.name = "time"
    return out


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def _cooldown_file() -> Path:
    return CACHE_DIR / ".cooldown"


def _cooling() -> bool:
    """True while the tier is in its post-ban cooling-off period.

    The state is kept on disk as well as in memory: every CLI invocation is a new
    process, and without this a banned IP pays a full host sweep before falling
    back on each and every run.
    """
    global _cooldown_until
    if _cooldown_until > time.monotonic():
        return True
    try:
        until = float(_cooldown_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if until > time.time():
        _cooldown_until = time.monotonic() + (until - time.time())
        return True
    return False


def _start_cooldown() -> None:
    global _cooldown_until
    _cooldown_until = time.monotonic() + COOLDOWN_SECONDS
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cooldown_file().write_text(str(time.time() + COOLDOWN_SECONDS), encoding="utf-8")
    except OSError:
        pass


def _end_cooldown() -> None:
    global _cooldown_until
    _cooldown_until = 0.0
    try:
        _cooldown_file().unlink(missing_ok=True)
    except OSError:
        pass


def _eastmoney(interval: str, limit: int) -> pd.DataFrame:
    global _last_request_at
    if _cooling():
        raise FeedUnavailable("eastmoney 处于冷却期（上一轮被拒后自动降级到新浪分时）")
    params = {
        "secid": SECID,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "klt": INTERVALS[interval],
        "fqt": "1",
        "lmt": str(limit),
        "end": "20500101",
    }
    last_error: Exception | None = None
    # One pass only. A second sweep over every host has never recovered a ban —
    # the endpoint refuses the IP, not the request — and it tripled the wait.
    for host in HOSTS:
        gap = time.monotonic() - _last_request_at
        if gap < MIN_REQUEST_SPACING:
            time.sleep(MIN_REQUEST_SPACING - gap)
        _last_request_at = time.monotonic()
        try:
            response = _session.get(KLINE_URL.format(host=host), params=params, timeout=TIMEOUT)
            response.raise_for_status()
            payload = response.json() or {}
            klines = ((payload.get("data") or {}).get("klines")) or []
            frame = _parse_klines(klines)
            if not frame.empty:
                _end_cooldown()
                frame.attrs.update(source="eastmoney", stale=False, has_volume=True, note="")
                return frame
        except Exception as exc:  # noqa: BLE001 — try the next host
            last_error = exc
    _start_cooldown()
    raise FeedUnavailable(
        f"eastmoney: no host returned {interval} bars"
        + (f" ({type(last_error).__name__})" if last_error else "")
        + f"; 冷却 {COOLDOWN_SECONDS:.0f}s"
    )


def _sina_1m() -> pd.DataFrame:
    response = _sina.get(SINA_MINLINE, timeout=TIMEOUT)
    response.raise_for_status()
    frame = _parse_sina_minline(response.text)
    if frame.empty:
        raise FeedUnavailable("sina: minute line was empty")
    frame.attrs.update(
        source="sina",
        stale=False,
        has_volume=False,
        note="新浪分时：只有每分钟收盘价，缺盘中高低点与成交量",
    )
    return frame


def _sina_daily() -> pd.DataFrame:
    """Daily OHLC for the same contract, ~10 years, no volume.

    Used for two things the minute feed cannot supply on its own: the prior
    session's true high/low/close (a level map without yesterday's extremes is
    half a level map) and a 20-day average range, which tells a scalper whether
    today has already spent its move.
    """
    response = _sina.get(SINA_DAILY, timeout=TIMEOUT)
    response.raise_for_status()
    text = response.text
    start, end = text.find("("), text.rfind(")")
    if start < 0 or end <= start:
        raise FeedUnavailable("sina: daily payload was not JSONP")
    try:
        payload = json.loads(text[start + 1 : end])
    except json.JSONDecodeError as exc:
        raise FeedUnavailable(f"sina: daily payload not JSON ({exc})") from exc
    rows = payload if isinstance(payload, list) else (payload.get("data") or [])
    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            records.append(
                {
                    "time": pd.to_datetime(row["date"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not records:
        raise FeedUnavailable("sina: daily payload had no usable rows")
    frame = pd.DataFrame(records).set_index("time").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame["volume"] = np.nan  # the endpoint reports 0 for every row
    frame.attrs.update(source="sina-daily", stale=False, has_volume=False, note="")
    return frame


def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    out = out.dropna(subset=["open"])
    out.attrs.update(frame.attrs)
    return out


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #


def _cache_path(interval: str) -> Path:
    return CACHE_DIR / f"{interval}.csv"


def _save_disk(interval: str, frame: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        meta = {
            "source": frame.attrs.get("source", "eastmoney"),
            "has_volume": bool(frame.attrs.get("has_volume", True)),
            "saved_at": datetime.now(SHANGHAI).isoformat(),
        }
        path = _cache_path(interval)
        frame.to_csv(path)
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — the cache is an optimisation
        logger.debug("could not write intraday cache for %s: %s", interval, exc)


def _load_disk(interval: str) -> pd.DataFrame | None:
    path = _cache_path(interval)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not read intraday cache for %s: %s", interval, exc)
        return None
    if frame.empty:
        return None
    meta = {}
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
    age = (datetime.now(SHANGHAI) - frame.index[-1].tz_localize(SHANGHAI)).total_seconds() / 60.0
    if age > MAX_CACHE_AGE_MINUTES:
        return None
    frame.attrs.update(
        source="cache",
        stale=True,
        has_volume=bool(meta.get("has_volume", True)),
        age_minutes=round(age, 1),
        note=f"磁盘缓存，最后一根 K 线是 {age:.0f} 分钟前",
    )
    return frame


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def bars(interval: str = "5m", limit: int | None = None, *, fresh: bool = False) -> pd.DataFrame:
    """Minute bars, oldest first, indexed by Beijing wall clock.

    Raises :class:`FeedUnavailable` only when all three tiers fail. Check
    ``frame.attrs["source"]`` before trusting volume-dependent fields.
    """
    if interval not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}; pick one of {sorted(INTERVALS)}")
    limit = limit or DEFAULT_LIMITS[interval]
    key = (interval, limit)

    now = time.monotonic()
    cached = _CACHE.get(key)
    if not fresh and cached and now - cached[0] < CACHE_TTL.get(interval, 30.0):
        return cached[1].copy()

    errors: list[str] = []
    frame: pd.DataFrame | None = None

    # Tier 1: the vendor's own interval.
    try:
        frame = _eastmoney(interval, limit)
    except FeedUnavailable as exc:
        errors.append(str(exc))

    # Tier 2: a finer interval resampled locally. 5m -> 15m/30m/60m is exact
    # arithmetic on the vendor's own bars, and 1m -> 5m is the same aggregation
    # the vendor performs, so this costs nothing but a smaller history.
    if frame is None:
        fallback_plan = {
            "5m": [("1m", "5min")],
            "15m": [("5m", "15min"), ("1m", "15min")],
            "30m": [("5m", "30min"), ("1m", "30min")],
            "60m": [("15m", "60min"), ("5m", "60min")],
            "1m": [],
        }.get(interval, [])
        for source_interval, rule in fallback_plan:
            try:
                finer = _eastmoney(source_interval, DEFAULT_LIMITS[source_interval])
            except FeedUnavailable:
                continue
            aggregated = _resample(finer, rule)
            if not aggregated.empty:
                aggregated.attrs["source"] = f"eastmoney/{source_interval}"
                frame = aggregated
                break

    # Tier 3: Sina's session minute line.
    if frame is None and interval != "1d":
        try:
            line = _sina_1m()
            frame = line if interval == "1m" else _resample(line, {"5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min"}[interval])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sina: {type(exc).__name__}: {exc}")

    # Tier 4: whatever we last saw.
    if frame is None:
        frame = _load_disk(interval)
        if frame is None:
            raise FeedUnavailable("; ".join(errors) or f"no source returned {interval} bars")

    frame.attrs.setdefault("source", "unknown")
    frame.attrs.setdefault("stale", False)
    frame.attrs.setdefault("has_volume", True)
    frame.attrs.setdefault("note", "")
    frame.attrs.setdefault("age_minutes", 0.0)
    if frame.attrs.get("source") == "eastmoney":
        _save_disk(interval, frame)
    _CACHE[key] = (now, frame)
    return frame.copy()


def clear_cache() -> None:
    _CACHE.clear()
    _DAILY.clear()


_DAILY: dict[str, tuple[float, pd.DataFrame]] = {}


def daily_bars(limit: int = 120, *, fresh: bool = False) -> pd.DataFrame:
    """Daily OHLC, oldest first, indexed by trading date.

    Never raises: an empty frame means "no daily context", and the level map
    then says so instead of dropping the run. It is a convenience layer, not a
    dependency.
    """
    now = time.monotonic()
    cached = _DAILY.get("daily")
    if not fresh and cached and now - cached[0] < 600:
        return cached[1].tail(limit).copy()
    try:
        frame = _sina_daily()
    except Exception as exc:  # noqa: BLE001 — daily context is optional
        logger.debug("daily bars unavailable: %s", exc)
        if cached:
            return cached[1].tail(limit).copy()
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    _DAILY["daily"] = (now, frame)
    return frame.tail(limit).copy()


# --------------------------------------------------------------------------- #
# Session day
# --------------------------------------------------------------------------- #


def to_new_york(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize(SHANGHAI).tz_convert(NEW_YORK)


def session_day(timestamp: pd.Timestamp | datetime) -> date:
    """The COMEX trading day a Beijing timestamp belongs to.

    The Globex day opens 18:00 New York and closes 17:00 the next afternoon, so
    the label rolls at 18:00 ET, not at midnight. Shifting six hours puts 18:00
    ET at midnight of the *new* day and leaves every session hour on the right
    side of the boundary in both DST regimes.
    """
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(SHANGHAI)
    return (stamp.tz_convert(NEW_YORK) + timedelta(hours=6)).date()


def session_frames(frame: pd.DataFrame) -> dict[date, pd.DataFrame]:
    if frame.empty:
        return {}
    days = pd.Series([session_day(ts) for ts in frame.index], index=frame.index)
    return {day: frame.loc[days == day] for day in sorted(days.unique())}


def last_price() -> float | None:
    try:
        frame = bars("1m", 5)
    except FeedUnavailable:
        return None
    return None if frame.empty else float(frame["close"].iloc[-1])


def frame_age_minutes(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("inf")
    last = frame.index[-1]
    last = last.tz_localize(SHANGHAI) if last.tzinfo is None else last
    return (datetime.now(SHANGHAI) - last).total_seconds() / 60.0
