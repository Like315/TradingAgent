"""The ICT board: 1H bias on top, 15m trigger underneath.

One screen, one decision: *is the higher timeframe pointing somewhere, and has
the trigger frame just taken liquidity in that direction*. Everything else —
pool inventory, sweep age, the fixed 2R card — is supporting detail that exists
so the trader can disagree with a specific number rather than with a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tradingagents.gold.intraday import feed
from tradingagents.gold.intraday.clock import SHANGHAI
from tradingagents.gold.ict.bias import Bias, read_bias
from tradingagents.gold.ict.context import MarketContext, build_context
from tradingagents.gold.ict.entry import FIXED_RR, IctSetup, build_setups
from tradingagents.gold.ict.liquidity import BUY_SIDE, SELL_SIDE, Pool, Sweep, find_pools, find_sweeps

_UP = "bold red"
_DOWN = "bold green"


@dataclass
class IctBoard:
    ctx: MarketContext
    bias: Bias
    pools: list[Pool]
    sweeps: list[Sweep]
    setups: list[IctSetup]
    rejects: list[str] = field(default_factory=list)
    freshness_blocker: str = ""

    @property
    def price(self) -> float:
        return self.ctx.price

    @property
    def primary(self) -> IctSetup | None:
        return self.setups[0] if self.setups else None

    @property
    def stance(self) -> str:
        if not self.ctx.clock.tradable:
            return "不可交易"
        if self.freshness_blocker:
            return "数据过期，不给信号"
        if self.setups:
            return f"可执行 · {self.setups[0].pool.label}"
        return f"{self.bias.label} · 等扫单"

    # -- machine ---------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        ctx, bias = self.ctx, self.bias
        return {
            "generated_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
            "price": round(self.price, 2),
            "stance": self.stance,
            "source": ctx.source,
            "stale": ctx.stale,
            "bar_age_minutes": round(ctx.bar_age, 1),
            "freshness_blocker": self.freshness_blocker,
            "atr": {"m15": round(ctx.atr15, 2), "h1": round(ctx.atr1h, 2)},
            "clock": {
                "phase": ctx.clock.phase_label,
                "grade": ctx.clock.grade,
                "tradable": ctx.clock.tradable,
                "blockers": ctx.clock.blockers,
                "beijing": ctx.clock.now.isoformat(),
            },
            "bias": {
                "direction": bias.direction,
                "structure": bias.structure,
                "label": bias.label,
                "allowed": bias.allowed,
                "dealing_low": bias.dealing_low,
                "dealing_high": bias.dealing_high,
                "position_pct": None if bias.position_pct is None else round(bias.position_pct, 3),
                "position": bias.position,
                "score": bias.score,
                "draw": None
                if bias.draw is None
                else {"label": bias.draw.label, "price": bias.draw.price, "r": round(bias.draw_r or 0.0, 2)},
                "reasons": bias.reasons,
                "valid_until": bias.valid_until.isoformat() if bias.valid_until else None,
            },
            "pools": [
                {
                    "label": p.label,
                    "price": p.price,
                    "side": p.side,
                    "kind": p.kind,
                    "touches": p.touches,
                    "distance": round(p.price - self.price, 2),
                    "r": round((p.price - self.price) / ctx.atr, 2),
                }
                for p in sorted(self.pools, key=lambda p: abs(p.price - self.price))
            ],
            "sweeps": [
                {
                    "label": s.pool.label,
                    "price": s.pool.price,
                    "side": s.side,
                    "status": s.status,
                    "reclaimed": s.reclaimed,
                    "extreme": round(s.extreme, 2),
                    "pierce": round(s.pierce, 2),
                    "displacement": round(s.displacement_atr, 2),
                    "bars_since": s.bars_since,
                    "bar_time": s.bar_time.isoformat(),
                }
                for s in self.sweeps[:8]
            ],
            "setups": [
                {
                    "side": s.side,
                    "pool": s.pool.label,
                    "pool_price": s.pool.price,
                    "entry": [s.entry_low, s.entry_high],
                    "stop": s.stop,
                    "targets": s.targets,
                    "rr": s.rr,
                    "risk": round(s.risk, 2),
                    "risk_r": round(s.risk / ctx.atr, 2),
                    "hold_minutes": s.hold_minutes,
                    "invalidation": s.invalidation,
                    "why": s.why,
                    "score": s.score,
                    "warnings": s.warnings,
                    "sizing": [
                        {
                            "risk_pct": row.risk_pct,
                            "risk_usd": row.risk_usd,
                            "ounces": row.ounces,
                            "contracts": row.contracts,
                            "micros": row.micros,
                        }
                        for row in s.sizing
                    ],
                }
                for s in self.setups
            ],
            "rejects": self.rejects,
        }

    # -- agent-facing text ------------------------------------------------ #

    def render_text(self) -> str:
        ctx, bias = self.ctx, self.bias
        lines = [
            "=== ICT BOARD（1H 偏置 + 15m 扫单；数值由分钟线计算，不得修改）===",
            f"时间：{ctx.clock.now.strftime('%Y-%m-%d %H:%M')} 北京",
            f"最新价：{self.price:,.2f} USD/oz　ATR(15m) {ctx.atr15:,.2f}　ATR(1H) {ctx.atr1h:,.2f}",
            f"时段：{ctx.clock.phase_label}（{ctx.clock.grade} 级）　可交易：{'是' if ctx.clock.tradable else '否'}",
            "",
            "--- 1H 偏置 ---",
            f"{bias.label}（结构 {bias.structure}）　有效期至 {bias.valid_until.strftime('%H:%M') if bias.valid_until else '-'}",
        ]
        if bias.dealing_low is not None:
            lines.append(f"Dealing range {bias.dealing_low:,.2f}-{bias.dealing_high:,.2f}　位置 {bias.position_pct:.0%}（{bias.position}）")
        if bias.draw is not None:
            lines.append(f"Draw on liquidity：{bias.draw.label} {bias.draw.price:,.2f}（{bias.draw_r:.2f}×ATR）")
        for reason in bias.reasons:
            lines.append(f"  · {reason}")

        lines += ["", "--- 流动性池（按距现价排序）---"]
        for pool in sorted(self.pools, key=lambda p: abs(p.price - self.price))[:10]:
            lines.append(
                f"  {pool.price:>9,.2f}  {(pool.price - self.price):>+7.2f}  "
                f"{(pool.price - self.price) / ctx.atr:>+6.2f}ATR  {pool.label}（{pool.touches} 触）"
            )

        lines += ["", "--- 最近扫单 ---"]
        if not self.sweeps:
            lines.append("无。")
        for sweep in self.sweeps[:6]:
            lines.append(
                f"  {sweep.bar_time.strftime('%m-%d %H:%M')}  {sweep.pool.label} {sweep.pool.price:,.2f}  "
                f"扫深 {sweep.pierce:.2f}　回收 {sweep.status}　距今 {sweep.bars_since} 根 15m"
            )

        lines += ["", f"--- 通过的形态（固定 1:{FIXED_RR:g}）---"]
        if not self.setups:
            lines.append("无。")
        for i, setup in enumerate(self.setups, 1):
            lines += [
                f"[{i}] {setup.side_cn}　{setup.pool.label} 扫单回收（评分 {setup.score:.2f}）",
                f"    入场区：{setup.entry_low:,.2f} - {setup.entry_high:,.2f}",
                f"    止损：{setup.stop:,.2f}（{setup.risk:.2f} 美元）",
                f"    目标：{setup.targets[0]:,.2f}（{setup.rr[0]:.1f}R，固定）",
                f"    持有：{setup.hold_minutes} 分钟内　失效：{setup.invalidation}",
            ]
            for warning in setup.warnings:
                lines.append(f"    注意：{warning}")
        return "\n".join(lines)

    # -- console ---------------------------------------------------------- #

    def render(self, console: Console | None = None) -> None:
        console = console or Console()
        console.print(self._header())
        console.print(self._bias_panel())
        console.print(self._pool_table())
        console.print(self._sweep_table())
        for setup in self.setups:
            console.print(self._setup_panel(setup))
        if not self.setups:
            console.print(self._no_trade_panel())

    def _header(self) -> Panel:
        body = Text.assemble(
            ("XAUUSD / GC  ", "bold"),
            (f"{self.price:,.2f}", "bold"),
            f"   1H 偏置 · 15m 触发 · 固定 1:{FIXED_RR:g}\n",
            (f"{self.ctx.clock.phase_label} · {self.stance}", "dim"),
            "\n",
            (
                f"数据 {self.ctx.source} · {self.ctx.bar_age:.0f} 分钟前"
                + (" · 延迟" if self.ctx.stale else "")
                + ("" if self.ctx.has_volume else " · 无量能"),
                "yellow" if self.ctx.stale or not self.ctx.has_volume else "dim",
            ),
        )
        return Panel(body, border_style="cyan", padding=(0, 1))

    def _bias_panel(self) -> Panel:
        bias = self.bias
        table = Table(show_header=False, box=None, padding=(0, 1))
        style = _UP if bias.direction == "long" else (_DOWN if bias.direction == "short" else "yellow")
        table.add_row("1H 判定", f"[{style}]{bias.label}[/]　结构 {bias.structure}　评分 {bias.score:.2f}")
        if bias.dealing_low is not None:
            table.add_row(
                "Dealing range",
                f"{bias.dealing_low:,.2f} - {bias.dealing_high:,.2f}　现价 {bias.position_pct:.0%}（{bias.position}）",
            )
        table.add_row("方向许可", f"[{style}]{bias.side_cn}[/]")
        if bias.draw is not None:
            table.add_row("Draw on liquidity", f"{bias.draw.label} {bias.draw.price:,.2f}（{bias.draw_r:.2f}×ATR）")
        if bias.valid_until is not None:
            table.add_row("有效期", f"至 {bias.valid_until.astimezone(SHANGHAI).strftime('%H:%M')} 北京（下一根 1H 收盘重算）")
        for reason in bias.reasons:
            table.add_row("·", f"[dim]{reason}[/]")
        return Panel(table, title="[bold]1H 偏置[/bold]", border_style="bright_black", title_align="left")

    def _pool_table(self) -> Panel:
        table = Table(box=None, padding=(0, 1), header_style="bold")
        for column in ("价位", "池子", "距现价", "ATR", "侧"):
            table.add_column(column, justify="right" if column in ("价位", "距现价", "ATR") else "left")
        for pool in sorted(self.pools, key=lambda p: p.price, reverse=True)[:12]:
            side = "上方·买方流动性" if pool.side == BUY_SIDE else "下方·卖方流动性"
            table.add_row(
                f"{pool.price:,.2f}",
                pool.label,
                f"{pool.price - self.price:+.2f}",
                f"{(pool.price - self.price) / self.ctx.atr:+.2f}",
                side,
            )
        return Panel(table, title="[bold]流动性池[/bold]", border_style="bright_black", title_align="left")

    def _sweep_table(self) -> Panel:
        table = Table(box=None, padding=(0, 1), header_style="bold")
        for column in ("时间", "池子", "扫深", "状态", "距今"):
            table.add_column(column)
        for sweep in self.sweeps[:6]:
            style = "yellow" if not sweep.reclaimed else "dim"
            table.add_row(
                sweep.bar_time.strftime("%m-%d %H:%M"),
                f"{sweep.pool.label} {sweep.pool.price:,.2f}",
                f"{sweep.pierce:.2f}",
                f"[{style}]{sweep.status}[/]",
                f"{sweep.bars_since} 根",
            )
        if not self.sweeps:
            table.add_row("—", "最近 16 根 15m 没有扫单", "", "", "")
        return Panel(table, title="[bold]15m 扫单监视[/bold]", border_style="bright_black", title_align="left")

    def _setup_panel(self, setup: IctSetup) -> Panel:
        style = _UP if setup.side == "long" else _DOWN
        head = Text.assemble((f"{setup.side_cn} ", style), (setup.pool.label, "bold"), (f"　评分 {setup.score:.2f}", "dim"))
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("入场区", f"[bold]{setup.entry_low:,.2f} - {setup.entry_high:,.2f}[/bold]")
        table.add_row("止损", f"{setup.stop:,.2f}（{setup.risk:.2f} 美元 / {setup.risk / self.ctx.atr:.2f}×ATR）")
        table.add_row("目标", f"{setup.targets[0]:,.2f}（固定 {setup.rr[0]:.1f}R）")
        table.add_row("持有", f"{setup.hold_minutes} 分钟内")
        table.add_row("失效", setup.invalidation)
        table.add_row("依据", f"[dim]{setup.why}[/dim]")
        for warning in setup.warnings:
            table.add_row("注意", f"[yellow]{warning}[/yellow]")
        return Panel(Group(head, table), border_style="red" if setup.side == "long" else "green", padding=(0, 1))

    def _no_trade_panel(self) -> Panel:
        reasons = ([self.freshness_blocker] if self.freshness_blocker else []) + self.rejects
        if not reasons:
            reasons = [self.bias.label]
        body = Text("\n".join(f"· {reason}" for reason in reasons[:6]))
        return Panel(body, title="[bold yellow]不做[/bold yellow]", border_style="yellow", padding=(0, 1))


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def build_ict_board(
    *,
    equity: float = 100_000.0,
    risk_pct: float = 0.5,
    now: datetime | None = None,
    fresh: bool = False,
) -> IctBoard:
    ctx = build_context(now=now, fresh=fresh)
    pools = find_pools(ctx)
    bias = read_bias(ctx, pools)
    sweeps = find_sweeps(ctx, pools)

    # Same rule as the scalp board: a stale price inside an open session is
    # worse than no price, because it looks actionable.
    blocker = ""
    if ctx.clock.is_open and ctx.bar_age > 10.0:
        blocker = f"行情已 {ctx.bar_age:.0f} 分钟未更新，按此价位下单等于赌旧价"
    rejects: list[str] = []
    setups = (
        []
        if blocker
        else build_setups(ctx, bias, sweeps, pools, equity=equity, risk_pct=risk_pct, rejects=rejects)
    )
    return IctBoard(ctx=ctx, bias=bias, pools=pools, sweeps=sweeps, setups=setups, rejects=rejects, freshness_blocker=blocker)
