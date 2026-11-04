"""Assemble and render the intraday board.

The board is the whole product for a discretionary scalper: one screen that
answers, in order, *should I be trading right now* (clock), *where are the
orders* (level map), *which trade am I allowed to take* (regime + setups) and
*how big* (sizing). Everything on it is computed from bars; the model layer only
reads it and writes prose around numbers it is not allowed to change.

:meth:`TapeBoard.render_text` is what the agents receive. It is written as a
labelled key/value dump rather than pretty prose on purpose — a table with
column names survives a model's attention better than a paragraph, and it makes
"the agent invented a price" easy to spot after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tradingagents.gold.intraday import feed
from tradingagents.gold.intraday.clock import SHANGHAI, SessionClock, read_clock
from tradingagents.gold.intraday.levels import LevelMap, build_level_map
from tradingagents.gold.intraday.tape import Regime, Setup, build_setups, read_regime

# Chinese market convention: up is red, down is green.
_UP = "bold red"
_DOWN = "bold green"
_GRADE_STYLE = {"A": "bold red", "B": "yellow", "C": "magenta", "D": "green"}


@dataclass
class TapeBoard:
    clock: SessionClock
    level_map: LevelMap
    regime: Regime
    setups: list[Setup]
    price: float
    prev_close: float | None
    equity: float
    risk_pct: float
    generated_at: datetime
    source: str = "东财分钟线（COMEX 主力连续）"
    has_volume: bool = True
    bar_age_minutes: float = 0.0
    stale: bool = False
    feed_note: str = ""
    # Set when the data is too old to trade off; the board then refuses setups.
    freshness_blocker: str = ""
    # Why each family of setup was skipped. "No trade" must explain itself.
    no_trade_reasons: list[str] = field(default_factory=list)

    # -- derived ---------------------------------------------------------- #

    @property
    def change(self) -> float | None:
        return None if self.prev_close in (None, 0) else self.price - self.prev_close

    @property
    def change_pct(self) -> float | None:
        chg = self.change
        return None if chg is None or not self.prev_close else chg / self.prev_close * 100.0

    @property
    def primary(self) -> Setup | None:
        return self.setups[0] if self.setups else None

    @property
    def stance(self) -> str:
        if not self.clock.tradable:
            return "不可交易"
        if not self.setups:
            return "观望（无形态通过闸门）"
        return f"可执行 · {self.primary.name}"

    # -- machine ---------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        lm = self.level_map
        return {
            "generated_at": self.generated_at.isoformat(),
            "bar_as_of": lm.as_of.isoformat() if lm.as_of is not None else None,
            "price": round(self.price, 2),
            "prev_close": self.prev_close,
            "change": None if self.change is None else round(self.change, 2),
            "change_pct": None if self.change_pct is None else round(self.change_pct, 3),
            "clock": {
                "beijing": self.clock.now.isoformat(),
                "phase": self.clock.phase_label,
                "grade": self.clock.grade,
                "tradable": self.clock.tradable,
                "blockers": self.clock.blockers,
                "minutes_since_open": self.clock.minutes_since_open,
                "minutes_to_halt": self.clock.minutes_to_halt,
                "next_marks": [
                    {"in_minutes": round(minutes, 1), "label": label, "at": at.astimezone(SHANGHAI).isoformat()}
                    for minutes, label, at in self.clock.next_marks
                ],
                "blackout": (
                    None
                    if self.clock.blackout is None
                    else {
                        "label": self.clock.blackout.event.label,
                        "active": self.clock.blackout.active,
                        "minutes": round(self.clock.blackout.minutes_to, 1),
                    }
                ),
            },
            "regime": {
                "code": self.regime.code,
                "label": self.regime.label,
                "er5": round(self.regime.er5, 3),
                "er15": round(self.regime.er15, 3),
                "ema_align": self.regime.ema_align,
                "slope_atr": round(self.regime.slope_atr, 2),
                "rv_1h_pct": round(self.regime.rv_1h, 1),
                "vol_percentile": round(self.regime.vol_pct, 1),
                "vol_basis": self.regime.vol_basis,
                "atr_expansion": round(self.regime.atr_expansion, 2),
                "dist_vwap_sigma": round(self.regime.dist_sigma, 2),
                "volume_state": self.regime.volume_state,
                "bias": self.regime.bias,
                "notes": self.regime.notes,
            },
            "levels": {
                "atr5": round(lm.atr5, 2),
                "atr15": round(lm.atr15, 2),
                "atr_pct": round(lm.atr_pct, 3),
                "day_open": lm.day_open,
                "session_high": lm.session_high,
                "session_low": lm.session_low,
                "asian_high": lm.asian_high,
                "asian_low": lm.asian_low,
                "asian_complete": lm.asian_complete,
                "vwap": None if lm.vwap is None else round(lm.vwap, 2),
                "vwap_sigma": None if lm.vwap_sigma is None else round(lm.vwap_sigma, 2),
                "prior": None
                if lm.prior is None
                else {
                    "high": lm.prior.high,
                    "low": lm.prior.low,
                    "close": lm.prior.close,
                    "settle": lm.prior.settle,
                    "poc": lm.prior.poc,
                    "vah": lm.prior.vah,
                    "val": lm.prior.val,
                },
                "map": [
                    {
                        "name": lv.name,
                        "price": lv.price,
                        "kind": lv.kind,
                        "side": lv.side,
                        "distance": lv.distance,
                        "atr": lv.atr,
                        "confluence": lv.confluence,
                        "note": lv.note,
                    }
                    for lv in lm.levels
                ],
                "notes": lm.notes,
            },
            "setups": [
                {
                    "name": s.name,
                    "side": s.side,
                    "kind": s.kind,
                    "trigger": s.trigger,
                    "entry": [s.entry_low, s.entry_high],
                    "stop": s.stop,
                    "stop_ticks": s.stop_ticks,
                    "targets": s.targets,
                    "rr": s.rr,
                    "hold_minutes": s.hold_minutes,
                    "invalidation": s.invalidation,
                    "why": s.why,
                    "cost_pct_of_t1": s.cost_pct,
                    "score": round(s.score, 2),
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
            "equity": self.equity,
        }

    # -- agent-facing text ------------------------------------------------ #

    def render_text(self) -> str:
        """A labelled dump for the models. Every number they may cite is here."""
        lm, rg, ck = self.level_map, self.regime, self.clock
        lines = [
            "=== BOARD（全部数值由交易所分钟线计算，不得修改或臆造）===",
            f"时间：{ck.now.strftime('%Y-%m-%d %H:%M')} 北京 / {ck.now.astimezone(feed.NEW_YORK).strftime('%H:%M')} 纽约",
            f"最新价：{self.price:,.2f} USD/oz"
            + (f"　日变动 {self.change:+,.2f}（{self.change_pct:+.2f}%）" if self.change is not None else ""),
            f"交易时段：{ck.phase_label}（流动性 {ck.grade} 级，{ck.grade_note}）",
            f"距本交易日起点 {ck.minutes_since_open} 分钟，距日终维护 {ck.minutes_to_halt} 分钟",
            f"可交易：{'是' if ck.tradable else '否'}"
            + (f"　阻断原因：{'；'.join(ck.blockers)}" if ck.blockers else ""),
        ]
        if ck.blackout is not None:
            prefix = "事件窗口" if ck.blackout.active else "下一事件"
            lines.append(f"{prefix}：{ck.blackout.text}")
        lines.append(f"数据源：{self.source}　最后一根 K 线 {self.bar_age_minutes:.0f} 分钟前"
                     + ("　[延迟]" if self.stale else ""))
        if self.feed_note:
            lines.append(f"数据说明：{self.feed_note}")
        if self.freshness_blocker:
            lines.append(f"数据阻断：{self.freshness_blocker}")

        lines += [
            "",
            "--- 结构价位（价格 / 距现价 / 距现价多少 ATR / 性质）---",
            f"当日开盘 {lm.day_open:,.2f}" if lm.day_open else "当日开盘 不可用",
            f"当日高/低 {lm.session_high:,.2f} / {lm.session_low:,.2f}"
            if lm.session_high
            else "当日高低 不可用",
            f"亚洲区间 {lm.asian_low:,.2f} - {lm.asian_high:,.2f}（{'已成型' if lm.asian_complete else '进行中'}）"
            if lm.asian_high
            else "亚洲区间 不可用",
            f"当日 VWAP {lm.vwap:,.2f}（σ={lm.vwap_sigma:,.2f}）" if lm.vwap else "VWAP 不可用",
            f"ATR(5m) {lm.atr5:,.2f}　ATR(15m) {lm.atr15:,.2f}　ATR/价格 {lm.atr_pct:.3f}%",
        ]
        if lm.prior_high and lm.prior_low:
            lines.append(
                f"上一交易日：高 {lm.prior_high:,.2f}　低 {lm.prior_low:,.2f}　收 {lm.prior_close:,.2f}"
                + (
                    f"　POC {lm.prior.poc:,.2f}　价值区 {lm.prior.val:,.2f}-{lm.prior.vah:,.2f}"
                    if lm.prior
                    else "　（无昨日分钟数据，POC/价值区不可用）"
                )
            )
        if lm.daily_atr:
            used = lm.range_used
            lines.append(
                f"20 日均幅 {lm.daily_atr:,.2f}　今日已走 {used:.0%}"
                if used is not None
                else f"20 日均幅 {lm.daily_atr:,.2f}"
            )
        for lv in lm.levels:
            lines.append(
                f"  {lv.price:>9,.2f}  {lv.distance:>+7.2f}  {lv.atr:>+6.2f}ATR  {lv.name}"
                + (f"（{lv.note}）" if lv.note else "")
            )

        lines += [
            "",
            "--- 市场状态 ---",
            f"判定：{rg.label}（{rg.code}）",
            f"效率比 ER(5m) {rg.er5:.2f}　ER(15m) {rg.er15:.2f}　20 根净移动 {rg.slope_atr:+.1f} ATR",
            f"15m 均线：{rg.ema_align}　量能：{rg.volume_state}",
            f"近 1 小时年化实际波动 {rg.rv_1h:.1f}%（{rg.vol_basis}分位 {rg.vol_pct:.0f}%）　ATR 扩张 {rg.atr_expansion:.2f}x",
            f"距 VWAP {rg.dist_sigma:+.2f}σ　综合倾向 {rg.bias:+.2f}",
        ]
        for note in rg.notes:
            lines.append(f"  注：{note}")

        lines += ["", "--- 通过闸门的形态（已按质量排序）---"]
        if not self.setups:
            lines.append("无。当前时段或状态下不应开新仓。")
        for i, s in enumerate(self.setups, 1):
            lines += [
                f"[{i}] {s.name}（{s.side}，{s.kind}，评分 {s.score:.2f}）",
                f"    触发：{s.trigger}",
                f"    入场区：{s.entry_low:,.2f} - {s.entry_high:,.2f}",
                f"    止损：{s.stop:,.2f}（{s.stop_dollars:.2f} 美元 = {s.stop_ticks} 跳）",
                f"    目标：T1 {s.targets[0]:,.2f}（{s.rr[0]:.1f}R）　T2 {s.targets[1]:,.2f}（{s.rr[1]:.1f}R）",
                f"    预期持有：{s.hold_minutes} 分钟以内　往返成本占 T1 {s.cost_pct:.1f}%",
                f"    失效：{s.invalidation}",
                f"    依据：{s.why}",
            ]
        return "\n".join(lines)

    # -- console ---------------------------------------------------------- #

    def render(self, console: Console | None = None) -> None:
        console = console or Console()
        console.print(self._header())
        console.print(self._clock_table())
        console.print(self._regime_table())
        console.print(self._level_table())
        for setup in self.setups:
            console.print(self._setup_panel(setup))
        if not self.setups:
            console.print(self._no_trade_panel())
        if self.primary and self.primary.sizing:
            console.print(self._sizing_table(self.primary))
        console.print(self._events_table())

    # Rich renderers ------------------------------------------------------ #

    def _header(self) -> Panel:
        chg = "" if self.change is None else f"  {self.change:+,.2f} ({self.change_pct:+.2f}%)"
        style = "" if self.change is None else (_UP if self.change >= 0 else _DOWN)
        line = Text()
        line.append("XAUUSD / GC  ", style="bold")
        line.append(f"{self.price:,.2f}", style="bold")
        if chg:
            line.append(chg, style=style)
        line.append(f"   {self.generated_at.strftime('%H:%M:%S')} 北京", style="dim")
        body = Text.assemble(
            line,
            "\n",
            Text(f"{self.clock.phase_label} · 流动性 {self.clock.grade} 级 · {self.stance}", style="dim"),
            "\n",
            Text(
                f"数据 {self.source} · {self.bar_age_minutes:.0f} 分钟前"
                + (" · 延迟" if self.stale else "")
                + ("" if self.has_volume else " · 无量能"),
                style="yellow" if self.stale or not self.has_volume else "dim",
            ),
        )
        return Panel(body, border_style="cyan", padding=(0, 1))

    def _clock_table(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        ck = self.clock
        table.add_row("北京 / 纽约", f"{ck.now.strftime('%H:%M:%S')} / {ck.now.astimezone(feed.NEW_YORK).strftime('%H:%M')}")
        table.add_row("本交易日起点", f"{ck.minutes_since_open} 分钟前　距维护 {ck.minutes_to_halt} 分钟")
        table.add_row("流动性", f"[{_GRADE_STYLE[ck.grade]}]{ck.grade} 级[/]　{ck.grade_note}")
        if ck.blackout is not None:
            style = "bold yellow" if ck.blackout.active else "dim"
            label = "事件窗口" if ck.blackout.active else "下一事件"
            table.add_row(label, f"[{style}]{ck.blackout.text}[/]")
        for minutes, label, at in ck.next_marks[:4]:
            table.add_row("·", f"{int(minutes) // 60}h{int(minutes) % 60:02d}m 后 {label}（{ck.bj(at)}）")
        return Panel(table, title="[bold]时钟[/bold]", border_style="bright_black", title_align="left")

    def _regime_table(self) -> Panel:
        rg = self.regime
        table = Table(show_header=False, box=None, padding=(0, 1))
        style = _UP if rg.bias > 0.15 else (_DOWN if rg.bias < -0.15 else "yellow")
        table.add_row("判定", f"[{style}]{rg.label}[/]")
        table.add_row("效率比", f"ER(5m) {rg.er5:.2f}　ER(15m) {rg.er15:.2f}　净移动 {rg.slope_atr:+.1f} ATR")
        table.add_row("均线 / 量能", f"{rg.ema_align}　{rg.volume_state}")
        table.add_row(
            "波动",
            f"1h 实际波动 {rg.rv_1h:.1f}%（{rg.vol_basis}分位 {rg.vol_pct:.0f}%）　ATR 扩张 {rg.atr_expansion:.2f}x　"
            f"VWAP {rg.dist_sigma:+.2f}σ",
        )
        table.add_row("倾向", f"{rg.bias:+.2f}")
        for note in rg.notes:
            table.add_row("注", f"[dim]{note}[/]")
        return Panel(table, title="[bold]状态[/bold]", border_style="bright_black", title_align="left")

    def _level_table(self) -> Panel:
        lm = self.level_map
        table = Table(box=None, padding=(0, 1), show_edge=False, header_style="bold")
        table.add_column("价位", justify="right")
        table.add_column("层级")
        table.add_column("距现价", justify="right")
        table.add_column("ATR", justify="right")
        table.add_column("备注", style="dim")

        above = [lv for lv in lm.levels if lv.price > lm.price][::-1][:7]
        below = [lv for lv in lm.levels if lv.price <= lm.price][:7]

        def fmt(level) -> tuple[str, str, str, str, str]:
            return (
                f"[bold]{level.price:,.2f}[/bold]",
                level.name,
                f"{level.distance:+.2f}",
                f"{level.atr:+.2f}",
                level.note or "",
            )

        for level in above:
            table.add_row(*fmt(level))
        table.add_row(
            f"[reverse] {lm.price:,.2f} [/reverse]",
            "[bold]现价[/bold]",
            "",
            "",
            f"ATR(5m) {lm.atr5:.2f}　ATR(15m) {lm.atr15:.2f}",
        )
        for level in below:
            table.add_row(*fmt(level))
        title = "价位地图（上=阻力，下=支撑）"
        return Panel(table, title=f"[bold]{title}[/bold]", border_style="bright_black", title_align="left")

    def _setup_panel(self, setup: Setup) -> Panel:
        side_style = _UP if setup.side == "long" else _DOWN
        side_text = "做多" if setup.side == "long" else "做空"
        head = Text.assemble(
            (f"{side_text} ", side_style),
            (setup.name, "bold"),
            (f"　评分 {setup.score:.2f}", "dim"),
        )
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("触发", setup.trigger)
        table.add_row("入场区", f"[bold]{setup.entry_low:,.2f} - {setup.entry_high:,.2f}[/bold]")
        table.add_row("止损", f"[{_DOWN if setup.side == 'long' else _UP}]{setup.stop:,.2f}[/]（{setup.stop_dollars:.2f} 美元 / {setup.stop_ticks} 跳）")
        table.add_row(
            "目标",
            f"T1 {setup.targets[0]:,.2f}（{setup.rr[0]:.1f}R）　T2 {setup.targets[1]:,.2f}（{setup.rr[1]:.1f}R）",
        )
        table.add_row("持有", f"{setup.hold_minutes} 分钟内　成本占 T1 {setup.cost_pct:.1f}%")
        table.add_row("失效", setup.invalidation)
        table.add_row("依据", f"[dim]{setup.why}[/dim]")
        return Panel(Group(head, table), border_style="red" if setup.side == "long" else "green", padding=(0, 1))

    def _no_trade_panel(self) -> Panel:
        if self.freshness_blocker:
            reasons = [self.freshness_blocker]
        else:
            reasons = (self.clock.blockers + self.no_trade_reasons)[:5] or [self.regime.label]
        body = Text("\n".join(f"· {reason}" for reason in reasons) or "· 没有形态同时通过时段、状态与赔率三道闸门")
        return Panel(body, title="[bold yellow]不做[/bold yellow]", border_style="yellow", padding=(0, 1))

    def _sizing_table(self, setup: Setup) -> Panel:
        table = Table(box=None, padding=(0, 1), header_style="bold")
        table.add_column("单笔风险", justify="right")
        table.add_column("金额", justify="right")
        table.add_column("盎司", justify="right")
        table.add_column("GC 手数", justify="right")
        table.add_column("MGC 手数", justify="right")
        for row in setup.sizing:
            highlight = "bold" if abs(row.risk_pct - self.risk_pct) < 1e-6 else ""
            table.add_row(
                f"[{highlight}]{row.risk_pct:.2f}%[/]" if highlight else f"{row.risk_pct:.2f}%",
                f"{row.risk_usd:,.0f}",
                f"{row.ounces:,.2f}",
                f"{row.contracts:.2f}",
                f"{row.micros:.1f}",
            )
        note = (
            f"按账户 {self.equity:,.0f} 美元、止损 {setup.stop_dollars:.2f} 美元/盎司计算。"
            f"GC 1 手 = 100 盎司，MGC 1 手 = 10 盎司。"
        )
        return Panel(
            Group(table, Text(note, style="dim")),
            title="[bold]仓位[/bold]",
            border_style="bright_black",
            title_align="left",
        )

    def _events_table(self) -> Panel:
        rows = []
        for event in self.clock.events[:5]:
            minutes = (event.when - self.clock.now).total_seconds() / 60.0
            when = event.when.astimezone(SHANGHAI).strftime("%m-%d %H:%M")
            if minutes < -event.blackout_after.total_seconds() / 60:
                continue
            tag = "已过" if minutes < 0 else f"{int(minutes) // 60}h{int(minutes) % 60:02d}m 后"
            impact = {"high": "[red]高[/red]", "medium": "[yellow]中[/yellow]", "low": "低"}.get(event.impact, "")
            rows.append((when, event.label, tag, impact, event.source))
        if not rows:
            return Panel(
                Text("未来 36 小时没有规则内事件；CPI / FOMC 需手工写入 events.json", style="dim"),
                title="[bold]事件[/bold]",
                border_style="bright_black",
                title_align="left",
            )
        table = Table(box=None, padding=(0, 1), header_style="bold")
        for column in ("时间（北京）", "事件", "距现在", "影响", "来源"):
            table.add_column(column)
        for row in rows:
            table.add_row(*row)
        return Panel(table, title="[bold]事件[/bold]", border_style="bright_black", title_align="left")


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def build_board(
    *,
    equity: float = 100_000.0,
    risk_pct: float = 0.5,
    now: datetime | None = None,
    fresh: bool = False,
) -> TapeBoard:
    """Fetch bars and compute the full board. Raises ``FeedUnavailable`` if down."""
    one_min = feed.bars("1m", 1200, fresh=fresh)
    five_min = feed.bars("5m", 3000, fresh=fresh)
    fifteen_min = feed.bars("15m", 1200, fresh=fresh)
    daily = feed.daily_bars(120, fresh=fresh)
    if five_min.empty:
        raise feed.FeedUnavailable("5 分钟线为空，无法建立价位地图")

    has_volume = bool(five_min.attrs.get("has_volume", True))
    source = str(five_min.attrs.get("source") or "unknown")
    note = str(five_min.attrs.get("note") or "")
    stale = bool(five_min.attrs.get("stale", False))
    age = feed.frame_age_minutes(one_min if not one_min.empty else five_min)

    level_map = build_level_map(one_min, five_min, fifteen_min, daily)
    if not has_volume:
        level_map.notes.append("数据源无成交量：量能过滤与突破形态不启用，POC 退化为时间分布（TPO）")

    regime = read_regime(five_min, fifteen_min, level_map)
    clock = read_clock(now)

    # A stale price is worse than no price: it looks actionable. Inside an open
    # session, anything older than ten minutes gets no setups at all.
    blocker = ""
    if clock.is_open and age > 10.0:
        blocker = f"行情已 {age:.0f} 分钟未更新，按此价位下单等于赌旧价；先修数据源"
    rejects: list[str] = []
    setups = (
        []
        if blocker
        else build_setups(
            level_map, regime, clock, equity=equity, five_min=five_min, has_volume=has_volume, rejects=rejects
        )
    )

    prev_close = level_map.prior_close
    if prev_close is None and level_map.prior is not None:
        prev_close = level_map.prior.close
    return TapeBoard(
        clock=clock,
        level_map=level_map,
        regime=regime,
        setups=setups,
        price=level_map.price,
        prev_close=prev_close,
        equity=equity,
        risk_pct=risk_pct,
        generated_at=clock.now,
        source=source,
        has_volume=has_volume,
        bar_age_minutes=round(age, 1),
        stale=stale,
        feed_note=note,
        freshness_blocker=blocker,
        no_trade_reasons=rejects,
    )
