"""Gold-only specialization of the TradingAgents framework.

The upstream project researches any Yahoo-covered instrument (equities, ETFs,
indices, crypto) with a company-fundamentals analyst at its core. Gold has no
cash flow and no management: its price is driven by real yields, the dollar,
ETF/central-bank demand and geopolitics. This package swaps the analysts,
tools, prompts and defaults that assume a company for ones built around gold,
and locks the run to a single instrument so the whole pipeline is pointed at
one market.

Entry points:
    tradingagents-gold                  CLI (interactive) — see ``gold.cli``
    from tradingagents.gold import analyze_gold   programmatic API
"""

__all__ = ["analyze_gold", "gold_config", "resolve_gold_symbol", "GOLD_SYMBOL"]


def __getattr__(name):  # lazy: keep importing ``tradingagents.gold`` cheap
    if name == "analyze_gold":
        from tradingagents.gold.runner import analyze_gold

        return analyze_gold
    if name == "gold_config":
        from tradingagents.gold.config import gold_config

        return gold_config
    if name == "GOLD_SYMBOL":
        from tradingagents.gold.spec import GOLD_SYMBOL

        return GOLD_SYMBOL
    if name == "resolve_gold_symbol":
        from tradingagents.gold.spec import resolve_gold_symbol

        return resolve_gold_symbol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "1.0.0"
