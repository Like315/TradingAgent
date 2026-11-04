"""Resolution of gold's macro drivers, with a keyed source first and a keyless one behind it.

The desk must work whether or not the user has registered for a FRED key, and
whether or not they can reach FRED from their network. Each driver is therefore
resolved through a chain:

    real yield      FRED DFII10        -> US Treasury real-yield curve
    nominal 10y     FRED DGS10         -> US Treasury nominal curve
    breakeven       FRED T10YIE        -> nominal - real (derived)
    Fed funds       FRED FEDFUNDS      -> (no keyless source)
    CPI YoY         FRED CPIAUCSL      -> (no keyless source)

What was used is reported per driver, so a report never implies FRED data it
did not have. The CPI series is converted to year-over-year inside here: a raw
index level is not a rate and models routinely misread it.
"""

from __future__ import annotations

import logging

import pandas as pd

from tradingagents.gold import data as gdata

logger = logging.getLogger(__name__)

# id -> (label, FRED series ids, treasury fallback key, transform)
DRIVERS = {
    "real": ("10y real yield (TIPS)", ("DFII10",), "real", "pct"),
    "nominal": ("10y nominal Treasury", ("DGS10",), "nominal", "pct"),
    "breakeven": ("10y breakeven inflation", ("T10YIE",), "breakeven", "pct"),
    "policy": ("Fed funds (effective)", ("FEDFUNDS",), None, "pct"),
    "cpi": ("US CPI (YoY)", ("CPIAUCSL",), None, "yoy"),
}


def load_drivers(trade_date: str, look_back_days: int = 3 * 365) -> tuple[dict, dict, list[str]]:
    """Resolve every driver; return (series by id, source by id, notes)."""
    series: dict[str, pd.Series] = {}
    sources: dict[str, str] = {}
    notes: list[str] = []
    # A 1-year change needs more than 365 days of history to compare against.
    look_back_days = max(int(look_back_days), 730)

    treasury: dict[str, pd.Series] = {}
    need_treasury = False

    failures: dict[str, str] = {}
    for key, (label, fred_ids, _fallback, transform) in DRIVERS.items():
        got = None
        for fred_id in fred_ids:
            try:
                candidate = gdata.fred_series(fred_id, trade_date, look_back_days=look_back_days)
            except Exception as exc:  # noqa: BLE001 — one driver failing is not fatal
                logger.debug("FRED %s unavailable: %s", fred_id, exc)
                failures[key] = str(exc).split(". Get a free key")[0]
                continue
            if not candidate.empty:
                got = candidate
                sources[key] = f"FRED {fred_id}"
                break
        if got is None and _fallback:
            need_treasury = True
        if got is not None:
            series[key] = _transform(got, transform)

    if need_treasury:
        try:
            from tradingagents.gold.vendors import treasury as treasury_vendor

            treasury = treasury_vendor.ten_year(trade_date, look_back_days)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Treasury curves unavailable: %s", exc)
            treasury = {}

    for key, (label, fred_ids, fallback, transform) in DRIVERS.items():
        if key in series or not fallback:
            continue
        if fallback in treasury and not treasury[fallback].empty:
            series[key] = _transform(treasury[fallback], transform)
            sources[key] = (
                "10y nominal - 10y real (US Treasury curves)"
                if fallback == "breakeven"
                else "US Treasury daily yield curve"
            )

    for key, (label, _fred, fallback, _t) in DRIVERS.items():
        if key in series:
            continue
        reason = failures.get(key) or "no data"
        if fallback:
            notes.append(
                f"{label}: unavailable — FRED said '{reason}' and the keyless "
                f"Treasury curves could not replace it; treat this driver as unknown"
            )
        else:
            notes.append(f"{label}: unavailable — '{reason}'; no keyless source for this series")
    return series, sources, notes


def _transform(series: pd.Series, kind: str) -> pd.Series:
    if kind == "yoy":
        # Monthly index -> year-over-year percent. shift(12) assumes the series is
        # monthly, which CPIAUCSL is; a shorter series yields NaN and drops out.
        return ((series / series.shift(12) - 1.0) * 100.0).dropna()
    return series
