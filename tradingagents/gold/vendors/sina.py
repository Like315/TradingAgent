"""Sina's global-futures daily series — a second China-reachable gold source.

Two independent domestic hosts are worth having: Eastmoney rate-limits (and
briefly drops connections from) an IP that requests too often, and a desk that
runs several tools per analysis would trip that on its own. Sina serves the same
COMEX contracts from a different domain, so when one refuses the other usually
answers.

Sina returns JSONP with prices as strings and no volume for these contracts;
volume is left at 0 rather than invented, and callers that need volume (the ETF
flow proxy) read it from elsewhere.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date as _date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KLINE_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
    "var%20_{symbol}=/GlobalFuturesService.getGlobalFuturesDailyKLine"
)
# Canonical symbol -> Sina realtime code. Separate from SYMBOLS (the futures
# K-line codes) because the dollar index and USD/CNY only exist as quotes.
#
# Only the quote codes are listed, deliberately. Their layout puts the latest
# print in field 8, but the foreign-futures quotes (``hf_GC``, ``hf_SI``) put it
# in field 0 with the name at field 13 — the same parser would read field 8 as
# gold's *high*. Those symbols already have a daily-series fallback, so they are
# left out rather than parsed with a different rule.
SPOT_URL = "https://hq.sinajs.cn/list={codes}"
SPOT_CODES = {"DX-Y.NYB": "DINIW", "CNY=X": "fx_susdcny"}
SPOT_NAMES = {"DINIW": "美元指数", "fx_susdcny": "USD/CNY 在岸"}
# A quote outside its band means the payload is not what this parser expects.
SPOT_BANDS = {"DINIW": (50.0, 200.0), "fx_susdcny": (5.0, 9.0)}
TIMEOUT = 20
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://finance.sina.com.cn/",
}
# Sina's own contract codes for the instruments this desk needs.
SYMBOLS = {"GC=F": "GC", "SI=F": "SI"}

_MEMORY_CACHE: dict[str, tuple[str, pd.DataFrame]] = {}


def _get(url: str, params: dict):
    try:
        return requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError):
        session = requests.Session()
        session.trust_env = False
        session.headers.update(HEADERS)
        return session.get(url, params=params, timeout=TIMEOUT)


def fetch_daily(code: str) -> pd.DataFrame | None:
    """Daily bars for a Sina contract code (``GC``, ``SI``); None on any failure."""
    today = _date.today().isoformat()
    cached = _MEMORY_CACHE.get(code)
    if cached and cached[0] == today:
        return cached[1].copy()

    url = KLINE_URL.format(symbol=code)
    payload = None
    for attempt in range(2):
        if attempt:
            time.sleep(1.0)
        try:
            response = _get(url, {"symbol": code, "_": str(int(time.time() * 1000))})
            response.raise_for_status()
            text = response.text
            start = text.find("=(")
            end = text.rfind(")")
            if start < 0 or end < 0:
                continue
            payload = json.loads(text[start + 2:end])
            if payload:
                break
        except Exception as exc:  # noqa: BLE001 — a fallback must never raise
            logger.debug("Sina kline for %s failed (attempt %d): %s", code, attempt + 1, exc)
            payload = None
    if not payload:
        return None

    rows = []
    for item in payload:
        try:
            rows.append({
                "Date": pd.Timestamp(item["date"]),
                "Open": float(item["open"]),
                "High": float(item["high"]),
                "Low": float(item["low"]),
                "Close": float(item["close"]),
                "Volume": float(item.get("volume") or 0.0),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return None
    frame = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    _MEMORY_CACHE[code] = (today, frame)
    return frame


def ohlcv(symbol: str, as_of: str | None = None) -> pd.DataFrame | None:
    code = SYMBOLS.get(symbol)
    if code is None:
        return None
    frame = fetch_daily(code)
    if frame is None or frame.empty:
        return None
    if as_of:
        cutoff = pd.to_datetime(as_of).normalize()
        frame = frame[frame["Date"] <= cutoff]
        if frame.empty:
            return None
    return frame.reset_index(drop=True)


def spot_batch(codes: list[str] | tuple[str, ...]) -> dict[str, float]:
    """Latest print for each Sina code, fetched in one request; unusable codes drop out.

    Each line is ``var hq_str_<code>="time,bid,ask,...,last,name,..."`` with
    prices as decimal strings: field 8 is the latest print and field 1 the bid.
    Both agree in practice, and a pip either way cannot move a CNY/gram parity,
    so field 8 is read. Every value must fall inside its band — if Sina ever
    reorders these fields, the band is what turns a wrong number into an honest
    "unavailable" instead of converting gold at a nonsense rate.

    One request for several codes matters: the desk asks for the dollar index and
    the CNY rate in the same run, and one call keeps the request count down.
    """
    wanted = [code for code in codes if code]
    if not wanted:
        return {}
    try:
        # No query string: Sina parses everything after "list=" as a symbol, so
        # a cache-buster would turn into "fx_susdcny?_=..." and return empty.
        response = _get(SPOT_URL.format(codes=",".join(wanted)), {})
        response.raise_for_status()
        text = response.text
    except Exception as exc:  # noqa: BLE001 — a fallback must never raise
        logger.debug("Sina spot batch failed: %s", exc)
        return {}

    out: dict[str, float] = {}
    for code in wanted:
        marker = f'var hq_str_{code}="'
        start = text.find(marker)
        if start < 0:
            continue
        start += len(marker)
        end = text.find('"', start)
        if end < 0:
            continue
        fields = text[start:end].split(",")
        if len(fields) <= 8:
            continue
        try:
            price = float(fields[8])
        except ValueError:
            continue
        low, high = SPOT_BANDS.get(code, (0.0, float("inf")))
        if low <= price <= high:
            out[code] = price
    return out


def spot(symbol: str) -> float | None:
    """Latest print for a canonical symbol from Sina's realtime quote, or None."""
    code = SPOT_CODES.get(symbol)
    if code is None:
        return None
    return spot_batch([code]).get(code)


def spot_usdcny() -> float | None:
    """Onshore USD/CNY from Sina's realtime quote; None when unusable."""
    return spot("CNY=X")


def is_available(symbol: str = "GC=F", as_of: str | None = None) -> bool:
    return ohlcv(symbol, as_of or _date.today().isoformat()) is not None
