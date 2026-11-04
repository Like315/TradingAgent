"""Data access for the gold desk: price frames and macro series as numbers.

The analyst tools in the upstream project return formatted text, which is fine
for an LLM to read but useless for computing a beta, a percentile or a
correlation. Gold analysis is mostly comparisons — gold against real yields,
gold against the dollar, this month against the same month in past years — so
this module returns pandas objects and the tools in ``gold.tools`` do the maths
and render the result once.

Everything here is point-in-time: price rows are cut at the analysis date by
``load_ohlcv``, and FRED observations are requested at that date's vintage, so a
backtest cannot read a later revision.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def price_frame(symbol: str, as_of: str, fill_gaps: bool = True) -> pd.DataFrame:
    """Daily OHLCV on or before ``as_of``, date-sorted, with a ``Date`` column."""
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

    data = load_ohlcv(symbol, as_of, fill_gaps=fill_gaps)
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


def closes(symbol: str, as_of: str, rows: int = 750) -> pd.Series:
    """The last ``rows`` closes as a float series indexed by date."""
    df = price_frame(symbol, as_of, fill_gaps=False)
    series = pd.Series(df["Close"].to_numpy(dtype=float), index=df["Date"].to_numpy())
    series = series.dropna()
    return series.tail(rows)


def spot(symbol: str) -> tuple[float | None, str]:
    """Latest realtime level for a symbol, plus a label saying where it came from.

    A single point, so no change, no percentile, no regression: callers may quote
    the level and must label it realtime. It exists because the historical
    K-line endpoint refuses in bursts while the realtime quotes keep answering,
    and a labelled dollar level beats "unavailable" on a macro dashboard.

    Eastmoney first — it scales by the endpoint's own decimal count and names the
    instrument — then Sina, a different host, which is what keeps the panel alive
    when Eastmoney drops a burst. The labels differ because the instruments do:
    Eastmoney quotes the *offshore* CNY (USD/CNH) and Sina the onshore rate.
    """
    try:
        from tradingagents.gold.vendors import eastmoney

        snapshot = eastmoney.quote(symbol)
    except Exception:  # noqa: BLE001
        snapshot = None
    if snapshot:
        name = snapshot.get("name") or symbol
        return float(snapshot["price"]), f"{name} 实时（东财，仅最新价）"

    try:
        from tradingagents.gold.vendors import sina

        code = sina.SPOT_CODES.get(symbol)
        if code:
            prices = sina.spot_batch([code])
            if prices:
                return float(prices[code]), f"{sina.SPOT_NAMES.get(code, symbol)} 实时（新浪，仅最新价）"
    except Exception:  # noqa: BLE001
        pass
    return None, ""


def fx_closes(as_of: str, rows: int = 30, symbol: str = "CNY=X") -> pd.Series:
    """USD/CNY closes, from whichever source answers: Yahoo first, else Eastmoney.

    Without this the domestic parity panel disappears entirely on a network that
    cannot reach Yahoo, which is exactly the network this deck is built for.
    Eastmoney quotes the offshore rate (USD/CNH); it tracks the onshore fix
    within a few dozen pips — far tighter than the physical premium the parity
    table exists to expose — so it is used unchanged, never adjusted to look
    like an onshore quote.
    """
    try:
        series = closes(symbol, as_of, rows=rows)
    except Exception as exc:  # noqa: BLE001 — fall through to the domestic source
        logger.debug("Yahoo %s unavailable: %s", symbol, exc)
        series = pd.Series(dtype=float)
    if not series.empty:
        return series
    try:
        from tradingagents.gold.vendors import eastmoney

        frame = eastmoney.ohlcv(symbol, as_of, rows=rows)
    except Exception:  # noqa: BLE001
        frame = None
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    return pd.Series(frame["Close"].to_numpy(dtype=float), index=frame["Date"].to_numpy())


def fred_series(indicator: str, as_of: str, look_back_days: int = 3 * 365) -> pd.Series:
    """One FRED observation series as floats indexed by observation date.

    ``indicator`` is a friendly alias ("10y_treasury") or a raw FRED ID
    ("DFII10"). Raises when FRED has no key, no such series, or no network — the
    caller decides whether that is fatal for its report.
    """
    from tradingagents.dataflows.vendors.fred import _fred_today, _request, _resolve_series_id

    series_id = _resolve_series_id(indicator)
    start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    # Pin the vintage so a historical run sees what was published then (#1275).
    pit = min(as_of, _fred_today())
    payload = _request(
        "series/observations",
        {
            "series_id": series_id,
            "observation_start": start,
            "observation_end": as_of,
            "sort_order": "asc",
            "realtime_start": pit,
            "realtime_end": pit,
        },
    )
    dates, values = [], []
    for obs in payload.get("observations", []):
        raw = obs.get("value", ".")
        if raw in (".", "", None):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
        dates.append(pd.Timestamp(obs["date"]))
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


def aligned(*series: pd.Series) -> pd.DataFrame:
    """Outer-join named series onto one daily index, then forward-fill.

    Macro series arrive at different frequencies (daily yields, weekly dollar
    index, monthly CPI). Forward-filling is the honest alignment for a daily
    regression — it carries the last *published* value forward instead of
    interpolating a value nobody had yet.
    """
    frame = pd.concat(series, axis=1).sort_index()
    return frame.ffill()


def last_change(series: pd.Series, days: int) -> float | None:
    """Change over the last ``days`` calendar days; None when the window is empty."""
    if series is None or series.empty:
        return None
    end = series.index[-1]
    window = series[series.index <= end - pd.Timedelta(days=days)]
    if window.empty:
        return None
    return float(series.iloc[-1] - window.iloc[-1])


def percentile_of_last(series: pd.Series, window: int | None = None) -> float | None:
    """Where the latest value sits in its own history, 0-100."""
    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    if window:
        values = values.tail(window)
    if len(values) < 20:
        return None
    return float((values <= values.iloc[-1]).mean() * 100.0)


def realized_vol(returns: pd.Series, window: int = 20) -> float | None:
    """Annualized realized volatility of the last ``window`` daily returns, in %."""
    clean = returns.dropna()
    if len(clean) < window:
        return None
    return float(clean.tail(window).std() * (TRADING_DAYS ** 0.5) * 100.0)


def beta_regression(y: pd.Series, x: pd.DataFrame) -> dict:
    """OLS of ``y`` on ``x`` (no intercept column needed).

    Returns coefficients, R-squared, residual standard error and the latest
    residual, which is the part of gold's move its macro drivers do not explain.
    Implemented with numpy through pandas to avoid pulling in statsmodels for
    one regression.
    """
    frame = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(frame) < 60:
        raise ValueError("not enough overlapping observations for a regression")
    yv = frame["y"].to_numpy(dtype=float)
    xv = frame.drop(columns=["y"])
    matrix = np.column_stack([np.ones(len(xv)), xv.to_numpy(dtype=float)])
    coefs, *_ = np.linalg.lstsq(matrix, yv, rcond=None)
    fitted = matrix @ coefs
    resid = yv - fitted
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    names = ["const", *xv.columns]
    return {
        "coef": dict(zip(names, coefs.tolist())),
        "r2": r2,
        "resid_std": float(resid.std(ddof=len(names))),
        "latest_residual": float(resid[-1]),
        "n": int(len(frame)),
        "as_of": frame.index[-1],
    }
