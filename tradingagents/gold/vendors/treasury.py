"""US Treasury par-yield curves — the keyless source for gold's discount rate.

Gold's single most important driver is the 10-year *real* yield, which FRED
serves as ``DFII10`` behind a free-but-required API key. The Treasury publishes
the same daily real-yield curve itself, keyless, and the nominal curve beside it
— so the breakeven (nominal minus real) is derived rather than fetched. That
makes the macro panel work out of the box, with no registration and no key.

Two calls per year of history: the endpoint returns one calendar year of daily
rows, so a 365-day window needs this year and last year.
"""

from __future__ import annotations

import logging
from datetime import date as _date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)
TIMEOUT = 25
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; tradingagents-gold/1.0)"}

# (curve, year) -> (fetched-on date, frame): a run asks for the same year from
# both curves and from several tools, and the endpoint should be hit once.
_YEAR_CACHE: dict[tuple[str, int], tuple[str, pd.DataFrame | None]] = {}

CURVES = {
    "real": "daily_treasury_real_yield_curve",
    "nominal": "daily_treasury_yield_curve",
}


def _get(url: str, params: dict):
    """GET with one retry that bypasses a system/env proxy.

    `requests` inherits a machine-wide proxy (registry on Windows, env vars
    elsewhere); a stale or broken one fails every request with ProxyError or a
    reset connection. The retry connects directly.
    """
    try:
        return requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    except Exception:  # noqa: BLE001 — retry once, directly
        session = requests.Session()
        session.trust_env = False
        session.headers.update(HEADERS)
        return session.get(url, params=params, timeout=TIMEOUT)


def _fetch_year(curve: str, year: int) -> pd.DataFrame | None:
    key = (curve, year)
    cached = _YEAR_CACHE.get(key)
    today = _date.today().isoformat()
    if cached and cached[0] == today:
        return cached[1]
    params = {
        "type": CURVES[curve],
        "field_tdr_date_value": str(year),
        "page": "",
        "_format": "csv",
    }
    try:
        response = _get(BASE.format(year=year), params)
        response.raise_for_status()
        from io import StringIO

        frame = pd.read_csv(StringIO(response.text))
    except Exception as exc:  # noqa: BLE001 — a fallback must never raise
        logger.debug("Treasury %s curve %s failed: %s", curve, year, exc)
        return None
    if frame.empty or "Date" not in frame.columns:
        return None
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y", errors="coerce")
    frame = frame.dropna(subset=["Date"])
    _YEAR_CACHE[key] = (today, frame)
    return frame


def curve(curve: str, tenor: str, as_of: str, look_back_days: int = 3 * 365) -> pd.Series:
    """One tenor of a Treasury curve as a date-indexed float series, oldest first.

    ``curve`` is "real" or "nominal"; ``tenor`` is a column label such as
    "10 YR" (the real curve carries 5/7/10/20/30 YR, the nominal one more).
    """
    end = pd.to_datetime(as_of)
    start = end - pd.Timedelta(days=look_back_days)
    frames = []
    # One request per calendar year in the window: a 3-year lookback needs four,
    # and fetching only the endpoints' years would leave a hole in the middle
    # (which silently broke the 1-year change and thinned the regression).
    for year in range(start.year, end.year + 1):
        frame = _fetch_year(curve, year)
        if frame is not None:
            frames.append(frame)
    if not frames:
        return pd.Series(dtype=float)

    frame = pd.concat(frames).drop_duplicates(subset=["Date"]).sort_values("Date")
    # The two curves label the same tenor differently ("10 YR" on the real curve,
    # "10 Yr" on the nominal one), so match case- and space-insensitively.
    wanted = " ".join(tenor.lower().split())
    column = next((c for c in frame.columns if " ".join(str(c).lower().split()) == wanted), None)
    if column is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=frame["Date"].to_numpy())
    series = series.dropna()
    series = series[(series.index >= start) & (series.index <= end)]
    return series


def ten_year(as_of: str, look_back_days: int = 3 * 365) -> dict[str, pd.Series]:
    """The 10y nominal, 10y real and the derived breakeven, as one dict."""
    out: dict[str, pd.Series] = {}
    real = curve("real", "10 YR", as_of, look_back_days)
    nominal = curve("nominal", "10 YR", as_of, look_back_days)
    if not real.empty:
        out["real"] = real
    if not nominal.empty:
        out["nominal"] = nominal
    if not real.empty and not nominal.empty:
        joined = pd.concat([nominal.rename("n"), real.rename("r")], axis=1).dropna()
        if not joined.empty:
            out["breakeven"] = (joined["n"] - joined["r"]).rename("breakeven")
    return out


def is_available(as_of: str | None = None) -> bool:
    return bool(curve("real", "10 YR", as_of or _date.today().isoformat(), look_back_days=30).size)
