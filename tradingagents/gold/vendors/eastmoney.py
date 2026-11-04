"""A China-reachable price source for gold, used when Yahoo Finance is not.

Why this exists: the whole framework reads prices through Yahoo Finance, and in
mainland China (corporate proxies, GFW-adjacent filtering, or a Yahoo outage)
``query1.finance.yahoo.com`` frequently answers 502/timeout. A gold desk that
cannot fetch gold is useless, so this module reads the same instruments from
Eastmoney's public K-line endpoint, which is reachable from mainland networks and
carries COMEX gold with decades of daily history.

Coverage is deliberately narrow — only the series gold analysis actually needs:

    GC=F        COMEX 黄金主力连续   101.GC00Y   (USD/oz, the traded series)
    DX-Y.NYB    美元指数             100.UDI
    CNY=X       USD/CNH              133.USDCNH
    沪金主连    SHFE 黄金主力连续     113.aum     (CNY/g, for the domestic basis)

Anything else returns None and the caller degrades as before. The endpoint is
unofficial and can change; every failure here is swallowed into ``None`` so a
Yahoo-capable environment never notices this module exists.
"""

from __future__ import annotations

import logging
import time
from datetime import date as _date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
# The realtime endpoint is served by two host names that fail independently:
# push2 refuses while push2his answers, and the other way round. Trying both
# turns "the panel vanished" into one extra request.
QUOTE_HOSTS = ("https://push2.eastmoney.com", "https://push2his.eastmoney.com")
TIMEOUT = 20
# ~5 years of trading days, matching the window load_ohlcv downloads from Yahoo.
DEFAULT_ROWS = 1250
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

SECIDS = {
    "GC=F": "101.GC00Y",      # COMEX gold, front/continuous
    "DX-Y.NYB": "100.UDI",    # ICE dollar index
    "CNY=X": "133.USDCNH",    # offshore CNY, close enough for a parity reference
    "AU.SHF": "113.aum",      # SHFE gold continuous, CNY per gram
}

# Order of the fields2 columns requested from the endpoint.
_FIELDS = "f51,f52,f53,f54,f55,f56"


_SESSION = None
# secid -> (fetched-on date, frame). A run asks for the same series from several
# tools; the endpoint dislikes bursts, so each series is fetched once per day.
_MEMORY_CACHE: dict[str, tuple[str, pd.DataFrame]] = {}


def _session():
    """One keep-alive session that ignores proxies unless one is explicitly set."""
    global _SESSION
    if _SESSION is None:
        session = requests.Session()
        session.trust_env = False          # don't inherit a broken system proxy
        session.headers.update(HEADERS)
        _SESSION = session
    return _SESSION


_LAST_CALL = 0.0
# Minimum spacing between requests: a run touches ~6 secids in a row, and the
# endpoint starts dropping connections when they arrive back to back.
_MIN_INTERVAL = 0.8


def _get(url: str, params: dict):
    """GET, retrying once with proxies bypassed.

    A machine-wide proxy (corporate gateway, sandbox, or a stale WinINET setting
    that `requests` picks up from the registry) is a common way for a working
    endpoint to fail with ProxyError while curl succeeds. The retry connects
    directly, ignoring both environment and system proxy settings.
    """
    global _LAST_CALL
    wait = _MIN_INTERVAL - (time.monotonic() - _LAST_CALL)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL = time.monotonic()
    try:
        return _session().get(url, params=params, timeout=TIMEOUT)
    except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError):
        session = requests.Session()
        session.trust_env = False
        session.headers.update(HEADERS)
        return session.get(url, params=params, timeout=TIMEOUT)


def secid_for(symbol: str) -> str | None:
    return SECIDS.get(symbol)


def fetch_daily(secid: str, rows: int = DEFAULT_ROWS) -> pd.DataFrame | None:
    """Daily OHLCV for an Eastmoney ``secid``, oldest first; None on any failure."""
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5",
        "fields2": _FIELDS,
        "klt": "101",        # daily
        "fqt": "1",          # forward-adjusted (no-op for futures, correct for ETFs)
        "lmt": str(rows),
        "end": "20500101",
    }
    key = f"{secid}:{rows}"
    today = _date.today().isoformat()
    cached = _MEMORY_CACHE.get(key)
    if cached and cached[0] == today:
        return cached[1].copy()

    payload = None
    # The endpoint drops bursts of connections (RemoteDisconnected). Three spaced
    # attempts turn a transient reset into a normal answer.
    for attempt in range(3):
        if attempt:
            time.sleep(1.0 * attempt)
        try:
            response = _get(KLINE_URL, params)
            response.raise_for_status()
            payload = response.json()
            if (payload or {}).get("data"):
                break
        except Exception as exc:  # noqa: BLE001 — a fallback must never raise
            logger.debug("Eastmoney kline for %s failed (attempt %d): %s", secid, attempt + 1, exc)
            payload = None
    if payload is None:
        return None

    data = (payload or {}).get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        logger.debug("Eastmoney returned no klines for %s", secid)
        return None

    rows_out = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            rows_out.append(
                {
                    "Date": pd.Timestamp(parts[0]),
                    "Open": float(parts[1]),
                    "Close": float(parts[2]),
                    "High": float(parts[3]),
                    "Low": float(parts[4]),
                    "Volume": float(parts[5]) if len(parts) > 5 and parts[5] else 0.0,
                }
            )
        except (TypeError, ValueError):
            continue
    if not rows_out:
        return None
    frame = pd.DataFrame(rows_out).sort_values("Date").reset_index(drop=True)
    _MEMORY_CACHE[key] = (today, frame)
    return frame


def ohlcv(symbol: str, as_of: str | None = None, rows: int = DEFAULT_ROWS) -> pd.DataFrame | None:
    """Daily bars for ``symbol`` cut at ``as_of``, in the shape ``load_ohlcv`` returns."""
    secid = secid_for(symbol)
    if secid is None:
        return None
    frame = fetch_daily(secid, rows=rows)
    if frame is None or frame.empty:
        return None
    if as_of:
        cutoff = pd.to_datetime(as_of).normalize()
        frame = frame[frame["Date"] <= cutoff]
        if frame.empty:
            return None
    return frame.reset_index(drop=True)


def quote(symbol: str) -> dict | None:
    """Latest realtime quote for ``symbol``; None when the source refuses.

    The K-line endpoint above drops bursts of connections while this one keeps
    answering, so it is the last-resort source for a single number — the
    USD/CNY rate a parity conversion needs. The price is divided by the
    endpoint's own ``f59`` decimal count rather than a hardcoded factor, because
    that count differs per instrument (4 for USD/CNH, 2 for the dollar index,
    1 for COMEX gold).
    """
    secid = secid_for(symbol)
    if secid is None:
        return None
    data = None
    for attempt, host in enumerate(QUOTE_HOSTS * 2):
        if attempt:
            time.sleep(1.0)
        try:
            response = _get(f"{host}/api/qt/stock/get",
                            {"secid": secid, "fields": "f43,f57,f58,f59,f60,f86"})
            response.raise_for_status()
            data = (response.json() or {}).get("data") or {}
            if data.get("f43"):
                break
        except Exception as exc:  # noqa: BLE001 — a fallback must never raise
            logger.debug("Eastmoney quote for %s on %s failed: %s", symbol, host, exc)
            data = None
    if not data:
        return None

    def _scaled(raw):
        if not isinstance(raw, (int, float)) or raw == 0:
            return None
        return float(raw) / (10 ** int(data.get("f59") or 2))

    price = _scaled(data.get("f43"))
    if price is None:
        return None
    return {
        "symbol": symbol,
        "name": data.get("f58") or "",
        "price": price,
        "prev_close": _scaled(data.get("f60")),
        "timestamp": data.get("f86"),
    }


def domestic_gold_close(as_of: str | None = None, rows: int = 60) -> pd.Series:
    """SHFE gold continuous (CNY/gram) closes, oldest last; empty when unavailable."""
    frame = ohlcv("AU.SHF", as_of, rows=rows)
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    return pd.Series(frame["Close"].to_numpy(dtype=float), index=frame["Date"].to_numpy())


def is_available(symbol: str = "GC=F", as_of: str | None = None) -> bool:
    """Whether this source can serve ``symbol`` right now (used by ``doctor``)."""
    return ohlcv(symbol, as_of or _date.today().isoformat(), rows=5) is not None
