"""The intraday agent loop: two readers, one decision, one risk check.

Four model calls, no tool rounds. The board already contains every number the
models are allowed to use, so there is nothing to look up — which is what keeps
a scalp desk run to tens of seconds instead of the twenty-odd minutes the daily
graph takes. A scalper who waits twenty minutes for an opinion has already
missed the trade.

The division of labour is deliberate:

* the **models** write judgement (which level matters, what the news flow
  implies, where the trap is);
* **arithmetic** decides the hard limits (`risk_check`). A model cannot approve a
  stop that is 3 ATR wide, a trade inside an event blackout, or a card whose
  prices do not exist on the board. Every veto names the rule and the number, so
  it can be argued with instead of quietly obeyed.

A parsed card is also *grounded*: its stop must sit within $2 of a setup the
board actually produced. A model that invents a level gets refused rather than
executed, because a plausible-looking invented stop is the most expensive
failure mode this system has.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tradingagents.gold.config import gold_config
from tradingagents.gold.intraday import prompts
from tradingagents.gold.intraday.board import TapeBoard, build_board
from tradingagents.gold.intraday.clock import SHANGHAI
from tradingagents.gold.intraday.tape import MIN_RR_AFTER_COST, size_position

logger = logging.getLogger(__name__)

_UP = "bold red"
_DOWN = "bold green"

# A card whose stop is further than this from the market is not a scalp.
MAX_STOP_ATR = 1.6
# Slack allowed between the model's stop and the board's setup stop, in dollars.
GROUNDING_TOLERANCE = 2.0


@dataclass
class ScalpCall:
    verdict: str  # EXECUTE | NO_TRADE | VETO
    side: str = "none"  # long | short | none
    setup: str = ""
    entry_low: float | None = None
    entry_high: float | None = None
    stop: float | None = None
    t1: float | None = None
    t2: float | None = None
    hold_minutes: int | None = None
    risk_pct: float = 0.5
    why: str = ""
    kill: str = ""
    sizing: list = field(default_factory=list)
    tape_read: str = ""
    catalyst_read: str = ""
    card_raw: str = ""
    risk_verdict: str = ""
    risk_reason: str = ""
    checks: list[tuple[str, str, bool]] = field(default_factory=list)
    board: TapeBoard | None = None
    elapsed: float = 0.0

    @property
    def rr(self) -> float | None:
        if self.entry_low is None or self.entry_high is None or self.stop is None or self.t1 is None:
            return None
        entry = (self.entry_low + self.entry_high) / 2.0
        risk = abs(entry - self.stop)
        return None if risk <= 0 else abs(self.t1 - entry) / risk

    @property
    def stop_dollars(self) -> float | None:
        if self.entry_low is None or self.entry_high is None or self.stop is None:
            return None
        return abs((self.entry_low + self.entry_high) / 2.0 - self.stop)

    def render(self, console: Console | None = None) -> None:
        console = console or Console()
        if self.verdict != "EXECUTE":
            body = Text()
            body.append(f"{self.verdict}\n", style="bold yellow")
            if self.setup:
                body.append(f"{self.setup}\n", style="bold")
            for label, message, ok in self.checks:
                if not ok:
                    body.append(f"· {label}：{message}\n", style="yellow")
            if self.why and not any(self.why in message for _label, message, _ok in self.checks):
                body.append(f"\n{self.why}\n", style="dim")
            if self.risk_reason:
                body.append(f"风控：{self.risk_reason}\n", style="dim")
            console.print(Panel(body, title="[bold yellow]不下单[/bold yellow]", border_style="yellow"))
            self._render_reader(console)
            return

        style = _UP if self.side == "long" else _DOWN
        side_text = "做多" if self.side == "long" else "做空"
        head = Text.assemble((f"{side_text} ", style), (self.setup or "", "bold"))
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("入场区", f"[bold]{self.entry_low:,.2f} - {self.entry_high:,.2f}[/bold]")
        table.add_row("止损", f"{self.stop:,.2f}（{self.stop_dollars:.2f} 美元）")
        table.add_row("目标", f"T1 {self.t1:,.2f}　T2 {self.t2:,.2f}" + (f"　{self.rr:.1f}R" if self.rr else ""))
        table.add_row("持有", f"{self.hold_minutes or '-'} 分钟内")
        table.add_row("风险", f"{self.risk_pct:.2f}% ≈ {self.risk_pct / 100 * (self.board.equity if self.board else 0):,.0f} 美元")
        table.add_row("依据", f"[dim]{self.why}[/dim]")
        table.add_row("失效", self.kill)
        if self.risk_verdict:
            table.add_row("风控", f"{self.risk_verdict}　{self.risk_reason}")
        console.print(Panel(Group(head, table), border_style="red" if self.side == "long" else "green", padding=(0, 1)))
        self._render_reader(console)

    def _render_reader(self, console: Console) -> None:
        if not (self.tape_read or self.catalyst_read):
            return
        table = Table(show_header=False, box=None, padding=(0, 1))
        if self.tape_read:
            table.add_row("读盘", self.tape_read.strip())
        if self.catalyst_read:
            table.add_row("事件", self.catalyst_read.strip())
        console.print(Panel(table, title="[bold]两位下属[/bold]", border_style="bright_black"))


# --------------------------------------------------------------------------- #
# Inputs for the catalyst reader
# --------------------------------------------------------------------------- #


def _headlines_text(limit: int = 14) -> str:
    try:
        from tradingagents.gold.vendors import news_cn

        items = news_cn.headlines(limit=limit)
    except Exception as exc:  # noqa: BLE001 — no news is not a reason to stop
        return f"新闻源不可用（{type(exc).__name__}）：不要编造新闻"
    if not items:
        return "无相关新闻"
    lines = []
    for item in items:
        stamp = str(item.get("ts") or "")[5:16]
        text = str(item.get("text") or "")[:120]
        lines.append(f"{stamp} {text}")
    return "\n".join(lines)


def _events_text(board: TapeBoard) -> str:
    clock = board.clock
    lines = []
    if clock.blackout is not None:
        lines.append(clock.blackout.text)
    for event in clock.events[:4]:
        minutes = (event.when - clock.now).total_seconds() / 60.0
        if minutes < -30:
            continue
        when = event.when.astimezone(SHANGHAI).strftime("%m-%d %H:%M")
        lines.append(f"{when}（北京）{event.label}　还有 {minutes:.0f} 分钟　影响 {event.impact}")
    return "\n".join(lines) or "未来 2 小时无已知事件"


# --------------------------------------------------------------------------- #
# Model plumbing
# --------------------------------------------------------------------------- #


def _content(response: Any) -> str:
    text = getattr(response, "content", response)
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(text).strip()


def _ask(llm: Any, prompt: str) -> str:
    return _content(llm.invoke(prompt))


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _num(text: str | None) -> float | None:
    if not text:
        return None
    match = _NUMBER.search(text)
    return None if match is None else float(match.group(0))


def _parse_card(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        # T1/T2 contain a digit, so the key pattern has to allow them; a
        # letters-only pattern silently drops both targets and every card then
        # fails the "missing fields" check.
        match = re.match(r"^([A-Z][A-Z0-9_]{1,11})\s*[:：]\s*(.+)$", line)
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def _entry_range(value: str | None) -> tuple[float | None, float | None]:
    """Read "4036.0-4037.1" as a range, not as 4036.0 and minus 4037.1."""
    if not value:
        return None, None
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", value)]
    if not numbers:
        return None, None
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


# --------------------------------------------------------------------------- #
# Deterministic risk
# --------------------------------------------------------------------------- #


def risk_check(call: ScalpCall, board: TapeBoard) -> list[tuple[str, str, bool]]:
    """Every hard limit, as ``(label, message, ok)``.

    Arithmetic, not opinion: this is the part that must not be negotiable, so it
    is not delegated to a model.
    """
    checks: list[tuple[str, str, bool]] = []
    clock = board.clock
    atr5 = board.level_map.atr5 or 1.0

    if board.freshness_blocker:
        checks.append(("数据新鲜度", board.freshness_blocker, False))
    if not clock.tradable:
        checks.append(("时段", "；".join(clock.blockers) or "当前时段不可交易", False))
    if board.regime.code in ("chaos", "dead"):
        checks.append(("状态", board.regime.label, False))

    if call.side in ("long", "short"):
        entry = ((call.entry_low or 0) + (call.entry_high or 0)) / 2.0
        # Grounding: the stop must belong to a setup the board produced.
        matched = [
            s for s in board.setups if s.side == call.side and call.stop is not None and abs(s.stop - call.stop) <= GROUNDING_TOLERANCE
        ]
        checks.append(
            (
                "数字对齐",
                f"止损 {call.stop:,.2f} 与板面形态匹配" if matched else "止损不在板面任何形态上，视为臆造，拒绝执行",
                bool(matched),
            )
        )
        if call.stop is not None:
            distance = abs(entry - call.stop)
            checks.append(
                (
                    "止损宽度",
                    f"{distance:.2f} 美元 = {distance / atr5:.2f} ATR（上限 {MAX_STOP_ATR} ATR）",
                    distance <= MAX_STOP_ATR * atr5,
                )
            )
            checks.append(
                (
                    "止损不要过窄",
                    f"{distance:.2f} 美元 < 0.3 ATR，会被噪音扫掉",
                    distance >= 0.3 * atr5,
                )
            )
        rr = call.rr
        checks.append(("赔率", f"T1 为 {rr:.2f}R（下限 {MIN_RR_AFTER_COST}）" if rr else "未给出目标", bool(rr and rr >= MIN_RR_AFTER_COST)))
        checks.append(
            (
                "方向一致",
                "T1 与入场方向一致"
                if (call.t1 is not None and ((call.side == "long" and call.t1 > entry) or (call.side == "short" and call.t1 < entry)))
                else "T1 在入场的反方向，单子卡自相矛盾",
                bool(call.t1 is not None and ((call.side == "long" and call.t1 > entry) or (call.side == "short" and call.t1 < entry))),
            )
        )
        checks.append(("单笔风险", f"{call.risk_pct:.2f}%（上限 1.00%）", call.risk_pct <= 1.0))
        drift = abs(entry - board.price) / atr5
        checks.append(("离现价距离", f"入场区距现价 {drift:.2f} ATR（超过 2.5 就是幻想成交）", drift <= 2.5))
    return checks


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def analyze_scalp(
    board: TapeBoard | None = None,
    *,
    equity: float = 100_000.0,
    risk_pct: float = 0.5,
    equity_per_trade: float | None = None,
    config: dict | None = None,
    force: bool = False,
    callbacks: Any = None,
) -> ScalpCall:
    """One scalp decision. Returns a card even when the answer is "no trade"."""
    started = time.time()
    board = board or build_board(equity=equity, risk_pct=risk_pct)

    # Arithmetic already says no: do not spend tokens to be told so.
    if not force and (board.freshness_blocker or not board.clock.tradable or not board.setups):
        reasons = []
        if board.freshness_blocker:
            reasons.append(board.freshness_blocker)
        if not board.clock.tradable:
            reasons.extend(board.clock.blockers or ["当前时段不可交易"])
        if not board.setups and board.clock.tradable:
            reasons.extend(board.no_trade_reasons[:3] or [f"没有形态通过闸门（{board.regime.label}）"])
        return ScalpCall(
            verdict="NO_TRADE",
            why="；".join(reasons),
            checks=[("时钟/闸门", "；".join(reasons), False)],
            board=board,
            elapsed=time.time() - started,
        )

    cfg = config or gold_config()
    from tradingagents.llm_clients import create_tier_client

    extra = {"callbacks": callbacks} if callbacks else {}
    quick = create_tier_client(cfg, "quick", **extra).get_llm()
    deep = create_tier_client(cfg, "deep", **extra).get_llm()

    board_text = board.render_text()
    tape_read = _ask(quick, prompts.tape_prompt(board_text))
    catalyst_read = _ask(quick, prompts.catalyst_prompt(board_text, _headlines_text(), _events_text(board)))
    card_text = _ask(deep, prompts.desk_prompt(board_text, tape_read, catalyst_read))

    fields = _parse_card(card_text)
    call = ScalpCall(
        verdict="NO_TRADE",
        tape_read=tape_read,
        catalyst_read=catalyst_read,
        card_raw=card_text,
        board=board,
        risk_pct=risk_pct,
    )
    verdict_word = (fields.get("CALL") or "").upper()
    if "NO_TRADE" in verdict_word or "NO TRADE" in verdict_word or "不做" in card_text[:80]:
        call.why = fields.get("WHY") or "模型判断不做"
        call.checks = [("模型", "主管判断当前不进场", False)]
        call.setup = fields.get("SETUP", "NO_TRADE")
        call.elapsed = time.time() - started
        return call

    if "LONG" in verdict_word or "多" in verdict_word:
        call.side = "long"
    elif "SHORT" in verdict_word or "空" in verdict_word:
        call.side = "short"
    else:
        call.why = "CALL 字段无法识别，按不做处理"
        call.elapsed = time.time() - started
        return call

    call.setup = fields.get("SETUP", "")
    call.entry_low, call.entry_high = _entry_range(fields.get("ENTRY"))
    call.stop = _num(fields.get("STOP"))
    call.t1 = _num(fields.get("T1"))
    call.t2 = _num(fields.get("T2"))
    hold = _num(fields.get("HOLD"))
    call.hold_minutes = int(hold) if hold else None
    risk = _num(fields.get("SIZE"))
    if risk is not None:
        call.risk_pct = min(max(risk, 0.1), 1.0)
    call.why = fields.get("WHY", "")
    call.kill = fields.get("KILL", "")

    if None in (call.entry_low, call.entry_high, call.stop, call.t1):
        call.verdict = "NO_TRADE"
        call.why = "单子卡缺字段（入场/止损/目标），按不做处理"
        call.elapsed = time.time() - started
        return call

    call.checks = risk_check(call, board)
    hard_fail = [c for c in call.checks if not c[2]]

    if hard_fail:
        call.verdict = "VETO"
        call.why = call.why or ""
        call.risk_reason = "；".join(f"{label}：{message}" for label, message, _ in hard_fail)
        call.elapsed = time.time() - started
        return call

    # Soft review: the model may still object, but it cannot undo a pass either.
    try:
        risk_raw = _ask(quick, prompts.risk_prompt(board_text, card_text))
        call.risk_verdict = (re.search(r"VERDICT\s*[:：]\s*(\w+)", risk_raw) or [None, ""])[1].upper()
        match = re.search(r"REASON\s*[:：]\s*(.+)", risk_raw)
        call.risk_reason = match.group(1).strip() if match else risk_raw[:160]
    except Exception as exc:  # noqa: BLE001 — a failed review must not block a verified card
        call.risk_verdict = "SKIP"
        call.risk_reason = f"风控复核未完成（{type(exc).__name__}）"

    call.verdict = "VETO" if call.risk_verdict == "VETO" else "EXECUTE"
    if call.verdict == "EXECUTE":
        call.sizing = size_position(call.stop_dollars or 0, board.equity, (call.risk_pct,))
    call.elapsed = time.time() - started
    return call
