"""Intraday (scalping) desk for gold.

The daily pipeline answers "where should gold be in two weeks". This package
answers "should I be long, short or flat for the next thirty minutes, and at
what price does that idea stop being right".

Entry points:

* :func:`build_board` — the deterministic tape board: session clock, level map,
  regime read, and the setups that pass their gates. No model tokens.
* :func:`analyze_scalp` — the same board handed to three short-prompt agents
  (tape, catalyst, desk) plus a risk veto, returning a single actionable card.

Nothing here reads the daily reports: a scalper's edge lives in minutes, and
mixing a two-week macro view into a 20-minute trade is how a scalp becomes an
accidental swing position.
"""

from __future__ import annotations

__version__ = "1.0.0"

from tradingagents.gold.intraday.board import TapeBoard, build_board  # noqa: E402
from tradingagents.gold.intraday.desk import ScalpCall, analyze_scalp  # noqa: E402

__all__ = ["TapeBoard", "build_board", "ScalpCall", "analyze_scalp", "__version__"]
