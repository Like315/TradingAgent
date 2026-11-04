"""``tradingagents-gold scalp`` — the intraday board and the scalp desk.

    tradingagents-gold scalp                  # one board, no model calls
    tradingagents-gold scalp --watch 60       # refresh every minute
    tradingagents-gold scalp --llm            # board + a written card
    tradingagents-gold scalp --llm --equity 20000 --risk 0.3

Default is arithmetic only: a scalper refreshing a screen every minute should
not pay for 50 model calls an hour, and the numbers are the part that matters.
``--llm`` adds the three-agent read plus the risk gate on top of the same board.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from tradingagents.gold.config import gold_config
from tradingagents.gold.intraday import feed
from tradingagents.gold.intraday.board import build_board
from tradingagents.gold.intraday.desk import analyze_scalp

console = Console()

SCALP_HELP = "Gold intraday / scalping board: minute bars, session clock, level map, setups."

DASHBOARD_HELP = "Local web dashboard: live board, direction votes and a day plan in the browser."


def dashboard_command(
    equity: float = typer.Option(100_000.0, "--equity", "-e", help="账户净值（美元），用于仓位换算"),
    risk: float = typer.Option(0.5, "--risk", "-r", help="单笔风险%，剥头皮建议 0.25-0.5"),
    port: int = typer.Option(8686, "--port", "-p", help="本地端口"),
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址；仅本机使用保持默认"),
    no_browser: bool = typer.Option(False, "--no-browser", help="启动后不自动打开浏览器"),
) -> None:
    """Serve the live intraday dashboard at http://127.0.0.1:<port>."""
    from tradingagents.gold.intraday.web import serve

    serve(equity=equity, risk_pct=risk, host=host, port=port, open_browser=not no_browser)


def scalp_command(
    equity: float = typer.Option(100_000.0, "--equity", "-e", help="账户净值（美元），用于仓位换算"),
    risk: float = typer.Option(0.5, "--risk", "-r", help="单笔风险%，剥头皮建议 0.25-0.5"),
    watch: int = typer.Option(0, "--watch", "-w", help="每 N 秒刷新一次（0 = 只跑一次）"),
    llm: bool = typer.Option(False, "--llm", help="调用模型给出单子卡（约 4 次模型调用）"),
    llm_every: int = typer.Option(10, "--llm-every", help="循环模式下每隔多少次刷新重跑一次模型"),
    as_json: bool = typer.Option(False, "--json", help="输出机器可读的 board JSON"),
    clear: bool = typer.Option(True, "--clear/--no-clear", help="循环模式下每次刷新清屏"),
    provider: str | None = typer.Option(None, "--provider", help="临时覆盖 LLM 厂商（如 deepseek）"),
    force: bool = typer.Option(False, "--force", help="即使闸门说不做也跑模型（复盘/研究用）"),
) -> None:
    """Print the intraday board; optionally run the scalp desk on it."""
    cfg = gold_config(**({"llm_provider": provider} if provider else {}))

    if watch <= 0:
        _one_pass(equity=equity, risk=risk, llm=llm, as_json=as_json, cfg=cfg, force=force)
        return

    iteration = 0
    try:
        while True:
            if clear and not as_json:
                console.clear()
            run_llm = llm and (iteration == 0 or (llm_every > 0 and iteration % llm_every == 0))
            _one_pass(equity=equity, risk=risk, llm=run_llm, as_json=as_json, cfg=cfg, force=force)
            if not as_json:
                console.print(
                    f"[dim]{datetime.now().strftime('%H:%M:%S')} 刷新中，每 {watch} 秒一次，Ctrl-C 退出"
                    + ("（下次模型复核在第 %d 次刷新）" % (llm_every - iteration % llm_every) if llm and llm_every else "")
                    + "[/dim]"
                )
            iteration += 1
            time.sleep(max(5, watch))
    except KeyboardInterrupt:
        console.print("\n[dim]已退出[/dim]")


def _one_pass(*, equity: float, risk: float, llm: bool, as_json: bool, cfg: dict, force: bool = False) -> None:
    try:
        # Not fresh=True: the feed's own TTLs (20s for 1m) already keep a
        # refresh loop current, and forcing a refetch every iteration is what
        # earns the vendor's rate limit.
        board = build_board(equity=equity, risk_pct=risk)
    except feed.FeedUnavailable as exc:
        message = Text()
        message.append("行情源不可用\n", style="bold red")
        message.append(f"{exc}\n\n", style="dim")
        message.append("三级兜底（东财 K 线 → 新浪分时 → 本地缓存）全部失败时，任何价位都是过期的。", style="dim")
        console.print(Panel(message, border_style="red"))
        return

    if as_json:
        console.print_json(json.dumps(board.to_dict(), ensure_ascii=False, default=str))
        return

    board.render(console)
    if llm:
        call = analyze_scalp(board, config=cfg, force=force)
        call.render(console)
