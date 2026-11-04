"""ICT liquidity-sweep model: 1H bias, 15m confirmation.

The intraday scalp module answers "what can I trade in the next five minutes".
This module answers a different question: *where is the market's resting
liquidity, has it just been taken, and does the higher timeframe agree*.

Three pieces, in the order a discretionary ICT trader uses them:

1. :mod:`liquidity` — where the stops sit (equal highs/lows, session extremes,
   prior day, swings). A sweep without a pool is just a wick.
2. :mod:`bias` — the 1H state machine: structure, premium/discount of the
   dealing range, and the draw on liquidity. It is refreshed on every 1H close
   and is the only thing allowed to authorise a direction.
3. :mod:`entry` — the 15m confirmation: pool pierced, price reclaimed, entry on
   the retest, stop beyond the sweep extreme, target at a fixed 1:2.

The fixed 2R is deliberate: it removes target selection from the model, so the
only thing that can be wrong is the read. At 2R the break-even win rate after
costs is ~34.5%, which is the number to hold the model against.
"""
