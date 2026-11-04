"""The 1H state machine: structure, premium/discount, draw on liquidity.

This is the only component allowed to authorise a direction. It is refreshed on
every 1H close and stays valid until the next one — a 15m sweep against it is
noise, not a signal. Two rules do most of the work:

* **structure** — higher highs and higher lows means longs only, and vice versa;
* **premium/discount** — in a bullish market you buy the discount half of the
  dealing range, never the premium half. Half of all ICT losers are trades taken
  in the right direction at the wrong half of the range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradingagents.gold.intraday.clock import SHANGHAI
from tradingagents.gold.intraday.levels import swings

from tradingagents.gold.ict.liquidity import BUY_SIDE, SELL_SIDE, Pool, nearest

# The dealing range must span at least this many 15m-ATRs to mean anything.
MIN_RANGE_ATR = 1.5


@dataclass
class Bias:
    direction: str  # long | short | none
    structure: str  # bullish | bearish | range
    label: str
    dealing_low: float | None = None
    dealing_high: float | None = None
    position_pct: float | None = None
    position: str = ""  # premium | discount | equilibrium
    draw: Pool | None = None
    draw_r: float | None = None
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    valid_until: datetime | None = None

    @property
    def allowed(self) -> bool:
        return self.direction in ("long", "short")

    @property
    def side_cn(self) -> str:
        return {"long": "只做多", "short": "只做空"}.get(self.direction, "不许开新仓")


def _structure(points: list[tuple[str, float, object]]) -> tuple[str, list[str]]:
    highs = [p for side, p, _ in points if side == "high"][-2:]
    lows = [p for side, p, _ in points if side == "low"][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return "range", ["1H 波段点不足，结构无法判定"]
    higher_high, higher_low = highs[-1] > highs[-2], lows[-1] > lows[-2]
    lower_high, lower_low = highs[-1] < highs[-2], lows[-1] < lows[-2]
    if higher_high and higher_low:
        return "bullish", [f"1H 结构：高点抬高（{highs[-2]:,.2f}→{highs[-1]:,.2f}），低点抬高（{lows[-2]:,.2f}→{lows[-1]:,.2f}）"]
    if lower_high and lower_low:
        return "bearish", [f"1H 结构：高点降低（{highs[-2]:,.2f}→{highs[-1]:,.2f}），低点降低（{lows[-2]:,.2f}→{lows[-1]:,.2f}）"]
    return "range", ["1H 结构：高低点互相包含，属震荡，不做方向单"]


def _next_hour(now: datetime) -> datetime:
    if now.tzinfo is None:
        now = now.replace(tzinfo=SHANGHAI)
    local = now.astimezone(SHANGHAI)
    return (local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))


def read_bias(ctx, pools: list[Pool] | None = None) -> Bias:
    """Classify the 1H frame. Pure function of the bars in ``ctx``."""
    points = swings(ctx.bars1h, k=2, lookback=96)
    structure, reasons = _structure(points)

    # The dealing range is the last 1H swing high and swing low, whichever came
    # first — the market does not care about the order it drew them in.
    last_high = next((p for side, p, _ in reversed(points) if side == "high"), None)
    last_low = next((p for side, p, _ in reversed(points) if side == "low"), None)
    if last_high is None or last_low is None:
        return Bias(direction="none", structure=structure, label="数据不足", reasons=reasons)

    high, low = max(last_high, last_low), min(last_high, last_low)
    span = high - low
    if span < MIN_RANGE_ATR * ctx.atr:
        reasons.append(f"1H dealing range 只有 {span:.2f} 美元（<{MIN_RANGE_ATR}×ATR），太窄不做")
        return Bias(direction="none", structure=structure, label="区间过窄", dealing_low=low, dealing_high=high, reasons=reasons)

    position_pct = (ctx.price - low) / span
    position = "premium" if position_pct > 0.5 else ("discount" if position_pct < 0.5 else "equilibrium")
    reasons.append(
        f"1H dealing range {low:,.2f}-{high:,.2f}，现价位于 {position_pct:.0%}（"
        + {"premium": "溢价区，只找空", "discount": "折价区，只找多", "equilibrium": "中位"}[position]
        + "）"
    )

    direction = "none"
    if structure == "bullish" and position_pct < 0.5:
        direction = "long"
    elif structure == "bearish" and position_pct > 0.5:
        direction = "short"
    elif structure == "bullish":
        reasons.append("结构偏多但价格在 premium 区：等回到 discount 再找多，现在不追")
    elif structure == "bearish":
        reasons.append("结构偏空但价格在 discount 区：等回到 premium 再找空，现在不追")
    else:
        reasons.append("震荡结构：只做区间边缘的扫单，不做趋势单")

    draw = None
    draw_r = None
    if direction in ("long", "short") and pools:
        draw = nearest(pools, ctx.price, BUY_SIDE if direction == "long" else SELL_SIDE)
        if draw is not None:
            draw_r = abs(draw.price - ctx.price) / ctx.atr
            reasons.append(f"流动性吸引（draw）：{draw.label} {draw.price:,.2f}，距现价 {draw_r:.2f}×ATR")

    score = 0.0
    if direction != "none":
        score = 0.5
        score += 0.3 * (1.0 - position_pct if direction == "long" else position_pct)
        if draw_r is not None and 1.0 <= draw_r <= 6.0:
            score += 0.2

    label = {"long": "1H 偏多（折价区）", "short": "1H 偏空（溢价区）"}.get(direction, "1H 无方向")
    return Bias(
        direction=direction,
        structure=structure,
        label=label,
        dealing_low=low,
        dealing_high=high,
        position_pct=position_pct,
        position=position,
        draw=draw,
        draw_r=draw_r,
        score=round(score, 2),
        reasons=reasons,
        valid_until=_next_hour(ctx.now),
    )
