"""Programmatic entry point for a gold run.

    from tradingagents.gold import analyze_gold

    run = analyze_gold()                       # today, all four gold analysts
    print(run.signal, run.report_path)
    print(run.state["final_trade_decision"])

The pipeline is locked to gold: ``symbol`` only accepts gold spellings (and
anything else raises), and the analyst set defaults to the four gold analysts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any

from tradingagents.gold.config import GOLD_ANALYSTS, gold_config
from tradingagents.gold.spec import GOLD_SYMBOL, resolve_gold_symbol


@dataclass
class GoldRun:
    """What a gold run produced."""

    state: dict = field(default_factory=dict)
    signal: str = ""
    report_path: Path | None = None
    config: dict = field(default_factory=dict)
    trade_date: str = ""

    @property
    def decision(self) -> str:
        return self.state.get("final_trade_decision", "")

    @property
    def rating(self) -> str:
        return self.state.get("final_rating", "") or self.signal

    def ratings_by_section(self) -> dict:
        """The analyst report lengths, as a quick "what came back" summary."""
        return {
            "technical": len(self.state.get("market_report") or ""),
            "macro": len(self.state.get("fundamentals_report") or ""),
            "flows": len(self.state.get("sentiment_report") or ""),
            "news": len(self.state.get("news_report") or ""),
        }


def analyze_gold(
    trade_date: str | None = None,
    *,
    symbol: str = GOLD_SYMBOL,
    portfolio: Any = None,
    config: dict | None = None,
    analysts: tuple[str, ...] | list[str] | None = None,
    debug: bool = False,
    save: bool = True,
    html: bool = True,
    save_path: str | Path | None = None,
    callbacks: list | None = None,
) -> GoldRun:
    """Run the gold desk once and return its decision.

    Args:
        trade_date: ``YYYY-MM-DD``; defaults to today. Data is served as of it.
        symbol: A gold spelling (GC=F, XAUUSD, 伦敦金, …). Non-gold raises.
        portfolio: Optional ``PortfolioContext`` so the trader sizes against a
            real book instead of a notional one.
        config: A full config dict; defaults to :func:`gold_config`.
        analysts: Subset of ``("gold_market", "gold_macro", "gold_positioning",
            "gold_news")``.
        save: Write the report tree (markdown + one HTML page).
    """
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    resolve_gold_symbol(symbol)  # reject non-gold instruments before any spend
    trade_date = trade_date or _date.today().isoformat()

    cfg = dict(config) if config is not None else gold_config()
    selected = tuple(analysts) if analysts else tuple(cfg.get("analysts") or GOLD_ANALYSTS)
    unknown = [key for key in selected if key not in GOLD_ANALYSTS]
    if unknown:
        raise ValueError(
            f"unknown gold analyst(s): {', '.join(unknown)}. "
            f"Available: {', '.join(GOLD_ANALYSTS)}"
        )
    cfg["analysts"] = list(selected)
    cfg.setdefault("gold_mode", True)
    cfg.setdefault("gold_symbol", GOLD_SYMBOL)

    graph = TradingAgentsGraph(
        selected_analysts=selected, debug=debug, config=cfg, callbacks=callbacks
    )
    state, signal = graph.propagate(
        GOLD_SYMBOL, trade_date, asset_type="commodity", portfolio=portfolio
    )

    report_path = None
    if save:
        report_path = graph.save_reports(
            state, GOLD_SYMBOL, save_path=Path(save_path) if save_path else None, html=html
        )
    return GoldRun(state=state, signal=signal, report_path=report_path, config=cfg,
                   trade_date=trade_date)
