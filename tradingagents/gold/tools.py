"""Gold-specific data tools for the analysts.

Three questions an equity tool set cannot answer for gold:

1. *What is the macro discount rate doing?* — real yields, breakevens, the dollar,
   the policy path, and how much of gold's move those explain
   (:func:`get_gold_macro_dashboard`).
2. *Who is buying?* — ETF flow proxies, cross-metal and cross-asset ratios,
   measured seasonality (:func:`get_gold_flows_and_positioning`).
3. *What does it cost to be wrong right now?* — realized-volatility regime,
   ATR-based stop distance and position size (:func:`get_gold_volatility_regime`).

Every tool degrades to a ``DATA_UNAVAILABLE`` sentinel rather than raising: an
exception inside a LangGraph tool ends the whole run, and a missing FRED key
should cost the report one section, not the analysis.
"""

from __future__ import annotations

import math
from typing import Annotated

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from tradingagents.dataflows.date_window import as_of
from tradingagents.gold import data as gdata
from tradingagents.gold.spec import GOLD_SYMBOL, cny_per_gram

DOLLAR_INDEX = "DX-Y.NYB"      # ICE US Dollar Index, daily, Yahoo-native
SILVER = "SI=F"
EQUITY_BENCH = "^GSPC"
GOLD_ETFS = ("GLD", "IAU")
USDCNY = "CNY=X"

# Which transform each driver needs when rendered ("pct" = a rate, "yoy" = a
# year-over-year rate, "level" = an index level quoted as a percent change).
_DRIVER_KIND = {"real": "pct", "nominal": "pct", "breakeven": "pct", "policy": "pct", "cpi": "yoy"}


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _num(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(value) or math.isinf(value):
        return "n/a"
    return f"{value:,.{digits}f}{suffix}"


def _pct(value, digits: int = 2) -> str:
    return "n/a" if value is None else _num(value, digits, "%")


def _bp(value) -> str:
    """A yield change in basis points, signed."""
    if value is None:
        return "n/a"
    return f"{value * 100:+,.0f} bp"


def _chg_pct(series: pd.Series, days: int) -> float | None:
    """Percent change over the last ``days`` calendar days."""
    if series is None or len(series) < 2:
        return None
    end = series.index[-1]
    window = series[series.index <= end - pd.Timedelta(days=days)]
    if window.empty:
        return None
    base = float(window.iloc[-1])
    if base == 0:
        return None
    return (float(series.iloc[-1]) / base - 1.0) * 100.0


def _corr(a: pd.Series, b: pd.Series, window: int) -> float | None:
    frame = pd.concat([a, b], axis=1).dropna().tail(window)
    if len(frame) < 20:
        return None
    value = frame.iloc[:, 0].corr(frame.iloc[:, 1])
    return None if value is None or math.isnan(value) else float(value)


def _atr_pct(df: pd.DataFrame, window: int = 14) -> float | None:
    """ATR as a percentage of the last close — the unit stops should be sized in."""
    if df is None or len(df) < window + 1:
        return None
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    last = float(close.iloc[-1])
    if not last or math.isnan(atr):
        return None
    return float(atr / last * 100.0)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 1. macro dashboard
# --------------------------------------------------------------------------- #
def _macro_dashboard(trade_date: str, look_back_days: int) -> str:
    gold = gdata.closes(GOLD_SYMBOL, trade_date, rows=max(look_back_days, 760))
    if gold.empty:
        return (
            f"DATA_UNAVAILABLE: no gold price series on or before {trade_date}. "
            "Report the macro section as unavailable; do not estimate levels."
        )

    blocks: list[str] = [
        f"## Gold macro dashboard — {trade_date} ({GOLD_SYMBOL}, USD/oz)",
        "",
        f"Latest close: **{_num(gold.iloc[-1])}** on {gold.index[-1].date()} "
        f"(1w {_pct(_chg_pct(gold, 7), 2)}, 1m {_pct(_chg_pct(gold, 30), 2)}, "
        f"3m {_pct(_chg_pct(gold, 91), 2)}, 1y {_pct(_chg_pct(gold, 365), 2)}).",
        "",
    ]

    notes: list[str] = []
    from tradingagents.gold.macro import load_drivers

    drivers, sources, driver_notes = load_drivers(trade_date, look_back_days)
    notes.extend(driver_notes)

    dxy_realtime: float | None = None
    try:
        dxy = gdata.closes(DOLLAR_INDEX, trade_date, rows=max(look_back_days, 760))
    except Exception as exc:  # noqa: BLE001
        dxy = pd.Series(dtype=float)
        # The historical endpoint flaps; one realtime level still lets the
        # dashboard quote the dollar, which is the driver the model asks about.
        dxy_realtime, _dxy_label = gdata.spot(DOLLAR_INDEX)
        if dxy_realtime is None:
            notes.append(f"Dollar index: unavailable ({exc})")
        else:
            notes.append(
                f"Dollar index: no daily history right now ({exc}); the level in the table is a "
                "single realtime quote, so its changes read n/a — do not infer a trend from it"
            )

    def _row(label: str, values: pd.Series, direction: str, kind: str, source: str = "") -> list[str]:
        latest = float(values.iloc[-1])
        one_month = gdata.last_change(values, 30)
        one_year = gdata.last_change(values, 365)
        if kind in ("pct", "yoy"):
            latest_s, m_s, y_s = _num(latest, 2, "%"), _bp(one_month), _bp(one_year)
        else:  # index level
            latest_s = _num(latest, 2)
            m_s = _pct((latest / (latest - one_month) - 1) * 100 if one_month not in (None, 0) else None)
            y_s = _pct((latest / (latest - one_year) - 1) * 100 if one_year not in (None, 0) else None)
        return [label, latest_s, m_s, y_s, direction, source]

    rows: list[list[str]] = []
    for key, label, direction in (
        ("real", "10y real yield (TIPS)", "inverse — the main driver"),
        ("nominal", "10y nominal yield", "inverse, weaker than real"),
        ("breakeven", "10y breakeven", "mildly positive (inflation hedge)"),
        ("policy", "Fed funds", "inverse via the expected path"),
        ("cpi", "US CPI YoY", "positive if it lifts breakevens"),
    ):
        if key in drivers:
            rows.append(_row(label, drivers[key], direction, _DRIVER_KIND.get(key, "pct"),
                             sources.get(key, "")))
    if not dxy.empty:
        rows.append(_row("Broad dollar (DXY)", dxy, "inverse", "level", "market data"))
    elif dxy_realtime is not None:
        rows.append(["Broad dollar (DXY)", _num(dxy_realtime, 2), "n/a", "n/a",
                     "inverse", "realtime quote, level only"])

    if rows:
        blocks += [
            _table(["Driver", "Latest", "1-month change", "1-year change", "Sign for gold", "Source"], rows),
            "",
            "Yield changes are in basis points; the dollar is in percent. A move in the "
            "*change* of real yields matters more than the level: gold rallies when real "
            "yields stop rising even at a high level.",
            "",
        ]

    # Gold's measured sensitivity to those drivers.
    if "real" in drivers or not dxy.empty:
        try:
            gold_ret = gold.pct_change() * 100.0
            frame_drivers = pd.DataFrame(index=gold.index)
            if "real" in drivers:
                frame_drivers["real_yield_chg_pp"] = drivers["real"].diff()
            if not dxy.empty:
                frame_drivers["dollar_ret_pct"] = dxy.pct_change() * 100.0
            frame_drivers = frame_drivers.reindex(gold.index).ffill()
            frame = pd.concat([gold_ret.rename("gold_ret_pct"), frame_drivers], axis=1).dropna()

            corr_lines = []
            for column, label in (
                ("real_yield_chg_pp", "10y real yield change"),
                ("dollar_ret_pct", "dollar return"),
            ):
                if column not in frame:
                    continue
                c60 = _corr(frame["gold_ret_pct"], frame[column], 60)
                c250 = _corr(frame["gold_ret_pct"], frame[column], 250)
                corr_lines.append(f"- gold vs {label}: 60d {_num(c60, 2)}, 250d {_num(c250, 2)}")
            if corr_lines:
                blocks += ["**Measured co-movement (daily)**", *corr_lines, ""]

            window = min(len(frame), 250)
            model = gdata.beta_regression(
                frame["gold_ret_pct"].tail(window), frame[frame_drivers.columns].tail(window)
            )
            coef = model["coef"]
            parts = []
            if "real_yield_chg_pp" in coef:
                parts.append(f"{coef['real_yield_chg_pp']:+.2f}% per +1pp real yield")
            if "dollar_ret_pct" in coef:
                parts.append(f"{coef['dollar_ret_pct']:+.2f}% per +1% dollar")
            resid = model["latest_residual"]
            resid_sigma = (resid / model["resid_std"]) if model["resid_std"] else 0.0
            # Name only the regressors that actually entered, so the verdict never
            # claims the dollar explained something it was not in the model for.
            regressors = " and ".join(
                name for column, name in (("real_yield_chg_pp", "real yields"),
                                          ("dollar_ret_pct", "the dollar"))
                if column in coef
            ) or "its macro drivers"
            verdict = (
                "rich versus its macro drivers — the move has a non-macro (flow / risk / "
                "official-sector) bid behind it, which can persist but mean-reverts when that bid fades"
                if resid_sigma > 1.0
                else "cheap versus its macro drivers — macro explains more of the move down than the "
                "price shows, a setup that often mean-reverts upward"
                if resid_sigma < -1.0
                else f"in line with what {regressors} explain"
            )
            blocks += [
                f"**Macro beta ({model['n']} sessions, R² {_num(model['r2'], 2)})**: "
                + "; ".join(parts)
                + f". Latest daily residual {resid:+.2f}% ({resid_sigma:+.2f}σ) — gold is {verdict}.",
                "",
            ]
        except Exception as exc:  # noqa: BLE001 — the table above is still useful
            blocks += [f"Macro-beta regression unavailable ({exc}).", ""]

    if notes:
        blocks += ["**Data gaps**", *[f"- {note}" for note in notes], ""]
    blocks.append(
        "FRED series are read at this date's vintage, so a backtest sees the values "
        "published then, not later revisions; the keyless Treasury fallback serves the "
        "published curve and carries no revision history."
    )
    return "\n".join(blocks)


@tool
def get_gold_macro_dashboard(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Trailing window in days for the driver series"] = 365,
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """
    The macro state that sets gold's discount rate, as of the current date.

    Returns the 10y nominal, 10y real (TIPS) and breakeven yields, the Fed funds
    rate, US CPI YoY and the dollar index with their 1-month and 1-year changes;
    gold's measured daily correlation with real-yield changes and the dollar; and
    an OLS beta plus the latest residual, which says whether gold is trading rich
    or cheap versus those drivers.

    Call this before making any claim about real yields, the dollar, the policy
    path or "gold looks expensive" — it is the source of truth for those numbers.

    Args:
        curr_date (str): Current date, yyyy-mm-dd
        look_back_days (int): Trailing window in days
    """
    try:
        return _macro_dashboard(as_of(curr_date, trade_date), look_back_days)
    except Exception as exc:  # noqa: BLE001 — never end the run from a tool
        return (
            f"DATA_UNAVAILABLE: the gold macro dashboard could not be built ({exc}). "
            "Report the macro drivers as unavailable rather than estimating them."
        )


# --------------------------------------------------------------------------- #
# 2. flows and positioning
# --------------------------------------------------------------------------- #
def _flows_and_positioning(trade_date: str) -> str:
    gold = gdata.closes(GOLD_SYMBOL, trade_date, rows=800)
    if gold.empty:
        return f"DATA_UNAVAILABLE: no gold price series on or before {trade_date}."

    blocks = [f"## Gold flows and positioning — {trade_date}", ""]
    notes: list[str] = []

    # Participation is the one flow measure always available: COMEX volume on the
    # gold series itself. ETF turnover follows when that data is reachable.
    etf_rows: list[list[str]] = []
    try:
        series_frame = gdata.price_frame(GOLD_SYMBOL, trade_date, fill_gaps=False).tail(300)
        volume = series_frame["Volume"].astype(float)
        turnover = (series_frame["Close"] * volume).astype(float)
        if len(volume) >= 60:
            v_mean, v_std = float(volume.tail(250).mean()), float(volume.tail(250).std())
            t_mean, t_std = float(turnover.tail(250).mean()), float(turnover.tail(250).std())
            vol_z = (volume.tail(20).mean() - v_mean) / v_std if v_std else None
            turn_z = (turnover.tail(20).mean() - t_mean) / t_std if t_std else None
            etf_rows.append([
                "GC=F (COMEX volume)",
                _num(series_frame["Close"].iloc[-1]),
                _pct(_chg_pct(series_frame.set_index("Date")["Close"], 20)),
                _num(turn_z, 2),
                _num(vol_z, 2),
            ])
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Gold volume: unavailable ({exc})")

    for etf in GOLD_ETFS:
        try:
            frame = gdata.price_frame(etf, trade_date, fill_gaps=False).tail(300)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{etf}: unavailable ({exc})")
            continue
        if len(frame) < 60:
            notes.append(f"{etf}: too few rows to score")
            continue
        turnover = (frame["Close"] * frame["Volume"]).astype(float)
        avg20 = turnover.rolling(20).mean().iloc[-1]
        base = turnover.tail(250)
        mean, std = float(base.mean()), float(base.std())
        z = (avg20 - mean) / std if std else None
        vol_z = None
        v = frame["Volume"].astype(float)
        v_mean, v_std = float(v.tail(250).mean()), float(v.tail(250).std())
        if v_std:
            vol_z = float((v.tail(20).mean() - v_mean) / v_std)
        etf_rows.append([
            etf,
            _num(frame["Close"].iloc[-1]),
            _pct(_chg_pct(frame.set_index("Date")["Close"], 20)),
            _num(z, 2),
            _num(vol_z, 2),
        ])
    if etf_rows:
        blocks += [
            _table(
                ["Series", "Close", "20d price chg", "20d turnover z-score", "20d volume z-score"],
                etf_rows,
            ),
            "",
            "Turnover and volume z-scores are a **flow proxy**, not reported creations/"
            "redemptions: a z-score above +1 with rising price is a conviction bid, below -1 "
            "with rising price is a thinning rally. Reported share counts are not in this build.",
            "",
        ]

    # Cross-asset ratios, as percentiles rather than bare numbers. Each leg is
    # fetched on its own: one unreachable series must not cost the other ratio.
    ratio_rows = []
    for label, peer, high_read, low_read in (
        ("Gold / silver", SILVER,
         "high = risk-off / industrial-demand scare; silver is cheap relative to gold",
         "low = reflation / industrial demand leading; silver is rich relative to gold"),
        ("Gold / S&P 500", EQUITY_BENCH,
         "high = gold outperforming equities, defensive regime",
         "low = risk appetite strong, gold lagging equities"),
    ):
        try:
            other = gdata.closes(peer, trade_date, rows=max(900, 760))
            if other.empty:
                notes.append(f"{label}: {peer} unavailable")
                continue
            joined = pd.concat([gold.rename("g"), other.rename("o")], axis=1).dropna()
            if len(joined) < 60:
                continue
            ratio = joined["g"] / joined["o"]
            pctile = gdata.percentile_of_last(ratio, window=750)
            if pctile is None:
                continue
            read = high_read if pctile >= 80 else low_read if pctile <= 20 else "mid-range"
            ratio_rows.append([label, _num(ratio.iloc[-1], 2), _pct(pctile, 0), read])
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{label}: unavailable ({exc})")
    if ratio_rows:
        blocks += [
            _table(["Ratio", "Latest", "Percentile (3y)", "Read"], ratio_rows),
            "",
        ]

    # Measured seasonality — shown only if the sample supports it.
    try:
        history = gdata.closes(GOLD_SYMBOL, trade_date, rows=2600)
        monthly = history.resample("ME").last().pct_change().dropna() * 100.0
        frame = pd.DataFrame({"ret": monthly})
        frame["month"] = frame.index.month
        stats = frame.groupby("month")["ret"].agg(
            mean="mean", median="median", hit=lambda s: (s > 0).mean() * 100.0, n="count"
        )
        current_month = int(gold.index[-1].month)
        rows = []
        for month, row in stats.iterrows():
            rows.append([
                f"{month:02d}" + ("  ← current" if month == current_month else ""),
                _pct(row["mean"]), _pct(row["median"]), _pct(row["hit"], 0), str(int(row["n"])),
            ])
        blocks += [
            "**Seasonality of monthly gold returns (full price history in the series)**",
            "",
            _table(["Month", "Mean return", "Median", "Hit rate", "Years"], rows),
            "",
            f"Current month is {current_month:02d}. Treat seasonality as a tie-breaker: with "
            f"{int(stats['n'].max())} observations per month it is not a thesis on its own.",
            "",
        ]
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Seasonality: unavailable ({exc})")

    blocks += [
        "**Not available in this build**: CFTC Commitments of Traders (managed-money "
        "net length), COMEX open interest, official ETF share counts, central-bank "
        "tonnage. Say so rather than inferring positioning from price alone.",
    ]
    if notes:
        blocks += ["", "**Data gaps**", *[f"- {note}" for note in notes]]
    return "\n".join(blocks)


@tool
def get_gold_flows_and_positioning(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """
    Who is buying gold, and how crowded the trade is.

    Returns ETF turnover/volume z-scores for GLD and IAU (a flow proxy), the
    gold/silver and gold/equity ratios with their 3-year percentiles, and the
    measured monthly seasonality of gold returns. Also lists what positioning
    data this build does *not* have (CFTC COT, open interest, ETF share counts).

    Args:
        curr_date (str): Current date, yyyy-mm-dd
    """
    try:
        return _flows_and_positioning(as_of(curr_date, trade_date))
    except Exception as exc:  # noqa: BLE001
        return (
            f"DATA_UNAVAILABLE: flow and positioning data could not be built ({exc}). "
            "Report it as unavailable; do not infer positioning from price alone."
        )


# --------------------------------------------------------------------------- #
# 3. volatility regime and sizing
# --------------------------------------------------------------------------- #
def _volatility_regime(trade_date: str) -> str:
    frame = gdata.price_frame(GOLD_SYMBOL, trade_date, fill_gaps=False)
    if frame.empty or len(frame) < 60:
        return f"DATA_UNAVAILABLE: not enough gold history on or before {trade_date}."

    frame = frame.tail(760).reset_index(drop=True)
    close = frame["Close"].astype(float)
    ret = close.pct_change()

    rv20 = gdata.realized_vol(ret, 20)
    rv60 = gdata.realized_vol(ret, 60)
    rv_series = ret.rolling(20).std() * np.sqrt(gdata.TRADING_DAYS) * 100.0
    rv_pctile = gdata.percentile_of_last(rv_series.dropna(), window=750)
    atr = _atr_pct(frame)
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    last = float(close.iloc[-1])
    year = close.tail(252)
    hi, lo = float(year.max()), float(year.min())
    pos = (last - lo) / (hi - lo) * 100.0 if hi > lo else None
    biggest = float(ret.tail(60).abs().max() * 100.0) if ret.tail(60).notna().any() else None

    regime = (
        "high-volatility / event-driven"
        if (rv_pctile or 0) >= 75
        else "low-volatility / trend-friendly"
        if (rv_pctile or 100) <= 25
        else "normal"
    )

    rows = [
        ["Last close", _num(last), f"as of {frame['Date'].iloc[-1].date()}"],
        ["20d realized vol (ann.)", _pct(rv20), f"{_pct(rv_pctile, 0)} percentile of 3y"],
        ["60d realized vol (ann.)", _pct(rv60), "regime context"],
        ["ATR-14 (% of price)", _pct(atr), "stop-distance unit"],
        ["Distance to 50d SMA", _pct((last / sma50 - 1) * 100 if sma50 else None), _num(sma50)],
        ["Distance to 200d SMA", _pct((last / sma200 - 1) * 100 if sma200 else None), _num(sma200)],
        ["Position in 52w range", _pct(pos, 0), f"{_num(lo)} – {_num(hi)}"],
        ["Largest 1-day move (60d)", _pct(biggest), "gap-risk reference"],
        ["Regime", regime, "from 20d vol percentile"],
    ]

    blocks = [
        f"## Gold volatility regime — {trade_date}",
        "",
        _table(["Metric", "Value", "Note"], rows),
        "",
    ]

    if atr:
        sizing = []
        for risk in (1.0, 0.5):
            for mult in (1.5, 2.0):
                stop_pct = atr * mult
                size = risk / stop_pct * 100.0  # % of account notional
                sizing.append([
                    f"{risk:.1f}% of account",
                    f"{mult:.1f}×ATR = {_num(stop_pct)}%",
                    f"{_num(size, 0)}% of account notional",
                ])
        blocks += [
            "**Position sizing from the ATR stop** (size = account risk / stop distance; "
            "applies to spot-equivalent exposure, not to a leveraged contract count):",
            "",
            _table(["Account risk per trade", "Stop distance", "Implied exposure"], sizing),
            "",
            f"At {_num(last)} USD/oz, a 1.5×ATR stop is about {_num(last * atr * 0.015)} USD/oz away. "
            "Size the trade from this table rather than from a fixed lot count, and halve the "
            "size when the regime is high-volatility or a scheduled event (FOMC, CPI, NFP) "
            "falls inside the holding period.",
        ]
    return "\n".join(blocks)


@tool
def get_gold_volatility_regime(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """
    Gold's current volatility regime and the stop/size arithmetic that follows.

    Returns 20d and 60d annualized realized volatility with the 20d reading's
    3-year percentile, ATR-14 as a percent of price, distance to the 50d and
    200d SMAs, the position inside the 52-week range, the largest recent
    one-day move, and a position-sizing table derived from the ATR stop.

    Call this before quoting any stop, target or position size.

    Args:
        curr_date (str): Current date, yyyy-mm-dd
    """
    try:
        return _volatility_regime(as_of(curr_date, trade_date))
    except Exception as exc:  # noqa: BLE001
        return (
            f"DATA_UNAVAILABLE: the volatility regime could not be computed ({exc}). "
            "Do not quote a stop or a size without it."
        )


# --------------------------------------------------------------------------- #
# 4. domestic (CNY) reference
# --------------------------------------------------------------------------- #
def _domestic_reference(trade_date: str) -> str:
    gold = gdata.closes(GOLD_SYMBOL, trade_date, rows=30)
    if gold.empty:
        return f"DATA_UNAVAILABLE: no gold price on or before {trade_date}."
    try:
        fx = gdata.fx_closes(trade_date, rows=30, symbol=USDCNY)
    except Exception as exc:  # noqa: BLE001
        fx = None
        fx_error = exc
    else:
        fx_error = None
    if fx is not None and not fx.empty:
        rate, rate_label = float(fx.iloc[-1]), f"USD/CNY 收于 {fx.index[-1]}"
    else:
        # No history: one realtime quote is enough for a parity conversion, and
        # far better than dropping the panel on a network that cannot reach Yahoo.
        rate, rate_label = gdata.spot(USDCNY)
        if rate is None:
            reason = f" ({fx_error})" if fx_error else ""
            return f"DATA_UNAVAILABLE: USD/CNY unavailable{reason}."

    usd_oz = float(gold.iloc[-1])
    parity = cny_per_gram(usd_oz, rate)
    rows = []
    for move in (-2.0, -1.0, 0.0, 1.0, 2.0):
        level = usd_oz * (1 + move / 100.0)
        rows.append([
            f"{move:+.0f}%",
            _num(level, 2),
            _num(cny_per_gram(level, rate), 2),
            _num(cny_per_gram(level, rate) * 1000, 0),
        ])

    # The real domestic quote, when a China-reachable source has it: the premium
    # over parity is the physical/financing tightness a domestic trader actually
    # watches, and it cannot be derived from the USD price.
    domestic_line = ""
    try:
        from tradingagents.gold.vendors import eastmoney

        domestic = eastmoney.domestic_gold_close(trade_date, rows=10)
        domestic_cny_g = float(domestic.iloc[-1]) if not domestic.empty else None
        kind = "沪金主连（上期所黄金主力，元/克）"
        if domestic_cny_g is None:
            # No daily history right now: the realtime quote still carries the
            # number the premium is computed from.
            snapshot = eastmoney.quote("AU.SHF")
            if snapshot:
                domestic_cny_g = float(snapshot["price"])
                kind = "沪金主力实时（东财，元/克）"
        if domestic_cny_g:
            premium = (domestic_cny_g / parity - 1.0) * 100.0
            domestic_line = (
                f"- {kind}：**{_num(domestic_cny_g)}**，"
                f"相对平水 **{premium:+.2f}%**（升水为正）\n"
            )
    except Exception:  # noqa: BLE001 — the parity table stands on its own
        domestic_line = ""

    return "\n".join([
        f"## Gold domestic reference — {trade_date}",
        "",
        f"- COMEX front month: **{_num(usd_oz)} USD/oz**",
        f"- USD/CNY: **{_num(rate, 4)}** ({rate_label})",
        f"- CNY/g parity: **{_num(parity)} CNY/g** (= USD/oz × USDCNY ÷ "
        f"{31.1034768:.4f}), i.e. **{_num(parity * 1000, 0)} CNY/kg**",
        *([domestic_line.rstrip()] if domestic_line else []),
        "",
        _table(["Gold move", "USD/oz", "CNY/g parity", "CNY/kg parity"], rows),
        "",
        "This is a **parity reference**, not the Shanghai price: SGE Au99.99 / Au(T+D) and "
        "SHFE 沪金 also carry a physical and financing premium (usually a fraction of a "
        "percent, wider when domestic demand is tight or imports are constrained), and the "
        "CNY contract has different hours, margin and lot size. Quote it as a conversion, "
        "never as the domestic quote.",
    ])


@tool
def get_gold_domestic_reference(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """
    Convert the USD/oz gold level into a CNY/gram and CNY/kg reference.

    Returns the latest COMEX close, USD/CNY, the CNY/g parity and a table of
    parity levels for gold moves of -2% to +2%. This is a conversion reference,
    not the Shanghai Gold Exchange price — the SGE contract carries a physical
    premium that this build does not fetch.

    Args:
        curr_date (str): Current date, yyyy-mm-dd
    """
    try:
        return _domestic_reference(as_of(curr_date, trade_date))
    except Exception as exc:  # noqa: BLE001
        return f"DATA_UNAVAILABLE: the domestic reference could not be computed ({exc})."


# --------------------------------------------------------------------------- #
# 5. gold headlines from China-reachable feeds
# --------------------------------------------------------------------------- #
def _headlines(trade_date: str, limit: int = 20) -> str:
    from datetime import date as _date

    from tradingagents.gold.vendors import news_cn

    if trade_date != _date.today().isoformat():
        # These are live wire feeds with no archive. Presenting today's flashes
        # as news for a past date would be worse than having none.
        return (
            f"DATA_UNAVAILABLE: the Sina wire feeds only carry the current session, so "
            f"there is no headline archive for {trade_date}. Treat that day's event "
            f"calendar as unknown and work from the macro, flows and volatility panels."
        )

    items = news_cn.headlines(limit=limit)
    if not items:
        return (
            "DATA_UNAVAILABLE: no gold-relevant headline came back from the Sina feeds "
            f"for {trade_date}; the feeds may be blocked on this network. Say the event "
            "calendar is unknown rather than inventing one."
        )
    return "\n".join([
        f"## Gold headlines — {trade_date}",
        "",
        *[f"- {item['ts']} | {item['text']} — *{item['source']}*" for item in items],
        "",
        "Sources: Sina 7x24 flashes (event speed) and the Sina finance roll (longer "
        "articles), filtered here to gold drivers — the metal, the Fed, the dollar and "
        "yields, inflation, geopolitics, official-sector demand. These are **headlines, "
        "not filings or central-bank statements**: report what they say, and never "
        "upgrade a headline into a confirmed fact. A story missing from this list is not "
        "evidence that it did not happen.",
    ])


@tool
def get_gold_headlines(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    limit: Annotated[int, "How many headlines to return (default 20)"] = 20,
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """
    Today's gold-relevant headlines, filtered from China-reachable wire feeds.

    Returns timestamped one-line flashes and article titles matching gold's
    drivers: the metal itself, the Fed and the policy path, the dollar, yields,
    inflation prints, geopolitics, and central-bank or ETF demand. Use it to name
    the specific catalyst behind a move instead of describing the move.

    It serves the current session only — a run dated in the past gets an explicit
    "no archive" answer, which you must respect rather than filling the gap with
    today's news.

    Args:
        curr_date (str): Current date, yyyy-mm-dd
        limit (int): How many headlines to return (default 20)
    """
    try:
        return _headlines(as_of(curr_date, trade_date), limit=min(max(int(limit), 1), 40))
    except Exception as exc:  # noqa: BLE001
        return f"DATA_UNAVAILABLE: headlines could not be fetched ({exc})."


GOLD_MACRO_TOOLS = (get_gold_macro_dashboard,)
GOLD_POSITIONING_TOOLS = (get_gold_flows_and_positioning, get_gold_volatility_regime)
GOLD_TECHNICAL_EXTRA_TOOLS = (get_gold_volatility_regime, get_gold_domestic_reference)
GOLD_HEADLINE_TOOLS = (get_gold_headlines,)
