"""``tradingagents-gold ict`` — the 1H bias / 15m sweep board.

    tradingagents-gold ict                  # one board
    tradingagents-gold ict --watch 60       # refresh every minute
    tradingagents-gold ict --json           # machine-readable
    tradingagents-gold ict --equity 20000 --risk 0.3

Arithmetic only. The bias is recomputed from 1H bars and the trigger from 15m
bars on every refresh, so a one-minute loop is cheap: no model calls, just two
rate-limited feeds behind a cache.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import typer
from rich.console import Console

from tradingagents.gold.intraday import feed
from tradingagents.gold.ict.board import build_ict_board

console = Console()

ICT_HELP = "ICT model: 1H bias (structure/premium-discount) + 15m liquidity-sweep entries at fixed 1:2."


def ict_command(
    equity: float = typer.Option(100_000.0, "--equity", "-e", help="账户净值（美元），用于仓位换算"),
    risk: float = typer.Option(0.5, "--risk", "-r", help="单笔风险%，建议 0.25-0.5"),
    watch: int = typer.Option(0, "--watch", "-w", help="每 N 秒刷新一次（0 = 只跑一次）"),
    as_json: bool = typer.Option(False, "--json", help="输出机器可读 JSON"),
) -> None:
    """Print the ICT board; optionally refresh on a loop."""
    if watch <= 0:
        _one_pass(equity=equity, risk=risk, as_json=as_json)
        return
    try:
        while True:
            if not as_json:
                console.clear()
            _one_pass(equity=equity, risk=risk, as_json=as_json)
            if not as_json:
                console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')} 每 {watch} 秒刷新，Ctrl-C 退出[/dim]")
            time.sleep(max(10, watch))
    except KeyboardInterrupt:
        console.print("\n[dim]已退出[/dim]")


def _one_pass(*, equity: float, risk: float, as_json: bool) -> None:
    try:
        board = build_ict_board(equity=equity, risk_pct=risk)
    except feed.FeedUnavailable as exc:
        console.print(f"[bold red]行情源不可用[/bold red]\n[dim]{exc}[/dim]")
        return
    if as_json:
        console.print_json(json.dumps(board.to_dict(), ensure_ascii=False, default=str))
        return
    board.render(console)
