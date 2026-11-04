"""The gold desk's configuration: one instrument, gold-only defaults.

Everything here is an override of ``default_config.build_default_config()``, so
``TRADINGAGENTS_*`` environment variables still apply and the LLM provider /
model selection keeps working exactly as upstream. Only what is gold-specific
changes: the instrument, the analyst set, the news queries, the benchmark, the
data directories and the display labels.

The benchmark deserves a note. Upstream measures alpha against a regional equity
index, which is meaningless here — a gold call should be judged against *holding
gold*. So ``benchmark_ticker`` is GC=F itself, and the memory log's reflection
asks whether the call beat buy-and-hold gold.
"""

from __future__ import annotations

import os

from tradingagents.default_config import build_default_config
from tradingagents.gold.prompts import GOLD_DOMAIN_BRIEF
from tradingagents.gold.spec import GOLD_SYMBOL

# The analysts a gold run uses, in display order. All four are gold-specific:
# the equity fundamentals analyst and the social-sentiment analyst are replaced,
# because gold has no financial statements and retail chatter is not what moves
# it.
GOLD_ANALYSTS = ("gold_market", "gold_macro", "gold_positioning", "gold_news")

GOLD_ANALYST_TITLES = {
    "gold_market": "Gold Technical Analyst",
    "gold_macro": "Gold Macro Analyst",
    "gold_positioning": "Gold Flows Analyst",
    "gold_news": "Gold News Analyst",
    # kept so a saved config that still names the upstream keys renders sensibly
    "market": "Gold Technical Analyst",
    "fundamentals": "Gold Macro Analyst",
    "social": "Gold Flows Analyst",
    "news": "Gold News Analyst",
}

GOLD_REPORT_LABELS = {
    "market_report": "Gold technical report",
    "fundamentals_report": "Gold macro report (real yields, dollar, policy, inflation)",
    "sentiment_report": "Gold flows and positioning report (ETF flows, ratios, seasonality)",
    "news_report": "Gold news and event-calendar report",
}

GOLD_NEWS_QUERIES = [
    "gold price Federal Reserve real yields Treasury",
    "central bank gold buying reserves official sector demand",
    "gold ETF flows holdings SPDR GLD",
    "dollar index DXY treasury yields inflation outlook",
    "safe haven demand geopolitical risk escalation gold",
    "China India gold demand jewellery premium imports",
    "COMEX gold futures open interest positioning",
]

GOLD_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents", "gold")


def gold_home() -> str:
    """Where a gold run keeps results, cache and memory (env still wins)."""
    return os.getenv("TRADINGAGENTS_GOLD_HOME") or GOLD_HOME


def gold_config(
    *,
    language: str = "Chinese",
    symbol: str = GOLD_SYMBOL,
    thorough: bool = True,
    **overrides,
) -> dict:
    """Default config with the gold profile applied.

    Args:
        language: Language of the user-visible reports. Internal debate stays in
            English for reasoning quality (upstream behaviour).
        symbol: Kept for record-keeping; the pipeline always trades GC=F.
        thorough: Two debate rounds on each side instead of one. Gold theses are
            usually "macro says yes, positioning says no", which one round tends
            to flatten into a consensus instead of resolving.
        **overrides: Anything else, applied last (e.g. ``llm_provider``).
    """
    config = build_default_config()
    home = gold_home()

    config.update(
        {
            "gold_mode": True,
            "gold_symbol": symbol,
            "analysts": list(GOLD_ANALYSTS),
            "output_language": language,
            "instrument_context_note": GOLD_DOMAIN_BRIEF,
            "analyst_titles": dict(GOLD_ANALYST_TITLES),
            "report_slot_labels": dict(GOLD_REPORT_LABELS),
            # Alpha is measured against holding gold, not against an equity index.
            "benchmark_ticker": GOLD_SYMBOL,
            "holding_period_days": 5,
            "max_debate_rounds": 2 if thorough else 1,
            "max_risk_discuss_rounds": 2 if thorough else 1,
            "max_tool_rounds": 12,
            # Where gold prices come from:
            #   "domestic" (default) Eastmoney/Sina first — reachable from
            #             mainland China, no key, no Yahoo timeout
            #   "auto"     Yahoo first, then the domestic sources
            #   "yahoo"    Yahoo only
            "gold_price_source": "domestic",
            "news_article_limit": 25,
            "global_news_article_limit": 15,
            "global_news_lookback_days": 10,
            "global_news_queries": list(GOLD_NEWS_QUERIES),
            "data_vendors": {
                "core_stock_apis": "yfinance",
                "technical_indicators": "yfinance",
                "fundamental_data": "yfinance",
                "news_data": "yfinance",
                "macro_data": "fred",
                "prediction_markets": "polymarket",
            },
        }
    )

    # Separate state for a gold desk: its memory log is gold-only, so lessons from
    # a gold call never mix with (or get pruned by) equity runs. An explicit
    # TRADINGAGENTS_* path still wins, which build_default_config already applied.
    if not os.getenv("TRADINGAGENTS_RESULTS_DIR"):
        config["results_dir"] = os.path.join(home, "logs")
    if not os.getenv("TRADINGAGENTS_CACHE_DIR"):
        config["data_cache_dir"] = os.path.join(home, "cache")
    if not os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH"):
        config["memory_log_path"] = os.path.join(home, "memory", "trading_memory.md")

    config.update(overrides)
    return config
