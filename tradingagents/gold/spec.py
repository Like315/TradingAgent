"""The instrument this fork trades: gold, quoted one way only.

Data source is Yahoo Finance, which has no spot XAU/USD series. The closest
liquid, continuous, history-backed series is the COMEX front-month future
``GC=F`` (USD per troy ounce), so that is the canonical symbol everything
resolves to. Broker/retail spellings ("XAUUSD+", "伦敦金", "GOLD#") are aliases
of the same market, not separate instruments.

CNY-quoted domestic contracts (SGE Au(T+D), SHFE 沪金, gold ETFs) are a
*different* instrument: same metal, different currency, unit, contract size,
trading hours and premium. They are rejected with an explanation instead of
being silently mapped onto a USD/oz series, because a report that mixes
"4,000" (USD/oz) and "930" (CNY/g) is worse than no report. ``cny_per_gram``
converts a USD/oz quote into a CNY/gram *parity* reference for readers who
think in the domestic price — parity, not the SGE price, which also carries a
physical premium.
"""

from __future__ import annotations

import re

GOLD_SYMBOL = "GC=F"
GOLD_LABEL = "COMEX gold front-month future (GC=F), USD per troy ounce"

# Contract facts the agents are told so they never size or price gold like a stock.
CONTRACT_FACTS = (
    "Contract: COMEX gold future, 100 troy ounces, quoted USD/oz, tick 0.10 "
    "(= USD 10 per contract), nearly 24h Globex session with a 60-minute break. "
    "Front month rolls roughly every two months (Feb/Apr/Jun/Aug/Dec), so a "
    "continuous GC=F series contains small calendar-spread jumps at each roll — "
    "do not read a roll gap as a breakout, and flag contract expiry when a "
    "holding period crosses a roll."
)
OUNCES_PER_GRAM = 31.1034768

# Retail / broker spellings of spot gold, all of which are GC=F here.
GOLD_ALIASES = frozenset(
    {
        "XAU", "XAUUSD", "XAUUSD+", "XAUUSD=X", "XAU-USD", "GOLD", "GOLD#",
        "GOLDUSD", "GOLDSPOT", "GC", "GC=F", "GC1!", "GCM", "GOLD.FOREX",
        "黄金", "伦敦金", "现货黄金", "国际金", "美黄金", "黄金现货",
    }
)

# CNY-quoted gold venues: real gold, wrong series for this pipeline.
DOMESTIC_SYMBOLS = frozenset(
    {"AU0", "AU9999", "AU(T+D)", "AUTD", "AU2612", "AU2610", "沪金", "上海金", "SGE", "518880", "159934"}
)
_DOMESTIC_HINT = ("AU", "TD", "沪", "上海金", "518", "1599")

# GC=F, GC1!, GCZ26.CMX, GCZ26 — the COMEX gold contract in Yahoo's spellings.
_FUTURES_CODE = re.compile(r"^GC(?:[FGHJKMNQUVXZ]\d{2})?(?:=(?:F|\d+)|!|\.CMX)?$")


class NotGoldError(ValueError):
    """Raised when a symbol is not a USD-quoted gold series."""


def _clean(raw: str) -> str:
    return raw.strip().upper().replace(" ", "")


def resolve_gold_symbol(raw: str | None) -> str:
    """Map any accepted gold spelling to ``GC=F``; reject anything else.

    Accepting an arbitrary ticker here would silently turn the gold pipeline
    into "analyze a stock with gold prompts", which is the one failure this
    fork exists to prevent. So an unknown symbol raises instead of passing
    through.
    """
    if raw is None or not str(raw).strip():
        return GOLD_SYMBOL
    cleaned = _clean(str(raw))
    # Aliases first: "XAUUSD" contains "AU", so a domestic-venue check running
    # before the alias table would reject the most common spot spelling.
    if cleaned in GOLD_ALIASES:
        return GOLD_SYMBOL
    # Yahoo-native contract spellings: GC=F, GC1!, GCZ26.CMX, GCZ26
    if _FUTURES_CODE.match(cleaned):
        return GOLD_SYMBOL
    if cleaned in DOMESTIC_SYMBOLS or any(h in cleaned for h in _DOMESTIC_HINT):
        raise NotGoldError(
            f"'{raw}' is a CNY-quoted domestic gold contract, not the USD/oz series this "
            f"pipeline trades. Use the default GC=F (COMEX front month) instead; convert "
            f"levels with the domestic reference tool (CNY/g parity = USD/oz x USDCNY / "
            f"{OUNCES_PER_GRAM:.4f}), and remember the SGE price also carries a physical "
            f"premium this parity does not."
        )
    raise NotGoldError(
        f"'{raw}' is not a gold symbol. This build analyzes gold only "
        f"(GC=F / XAUUSD / 伦敦金). Unsupported instruments are rejected on purpose: "
        f"the analyst prompts and tools here are gold-specific."
    )


def is_gold(raw: str | None) -> bool:
    try:
        resolve_gold_symbol(raw)
    except NotGoldError:
        return False
    return True


def cny_per_gram(usd_per_oz: float, usdcny: float) -> float:
    """USD/oz -> CNY/g parity reference (no SGE premium, no VAT, no fees)."""
    return usd_per_oz * usdcny / OUNCES_PER_GRAM
