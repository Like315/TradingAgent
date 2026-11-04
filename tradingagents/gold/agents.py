"""Gold versions of the debate, trader and risk nodes.

The upstream bull/bear prompts ask for "growth potential", "competitive
advantages" and "revenue projections"; the aggressive risk analyst asks about
"innovative benefits". Asked about gold, a model will happily invent a
competitive moat for a metal. These replacements keep the node contracts (state
keys, reply shapes, structured outputs) identical and change only the argument
structure the model is asked to fill:

    bull      real-yield tailwind, dollar topping, policy easing, official demand
    bear      hawkish repricing, ETF outflows, rich residual, crowded longs
    trader    ATR-sized entry/stop/target with event risk named
    risk      1.5x / 0.5x / 1.0x framing of the same gold trade

The Research Manager and Portfolio Manager prompts upstream are already
instrument-agnostic (rating scale + evidence weighing), so they are reused
unchanged.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    opponent_argument_or_opening,
    report_or_absent,
)
from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.gold.analysts import (
    create_gold_macro_analyst,
    create_gold_market_analyst,
    create_gold_news_analyst,
    create_gold_positioning_analyst,
)


def _reports(state) -> dict:
    return {
        "market": report_or_absent(state["market_report"], "technical"),
        "flows": report_or_absent(state["sentiment_report"], "flows/positioning"),
        "news": report_or_absent(state["news_report"], "news"),
        "macro": report_or_absent(state["fundamentals_report"], "macro"),
    }


def _debate_block(state) -> str:
    reports = _reports(state)
    return f"""{get_instrument_context_from_state(state)}
Gold technical report: {reports['market']}
Gold macro report: {reports['macro']}
Gold flows/positioning report: {reports['flows']}
Gold news and event calendar: {reports['news']}"""


def create_gold_bull_researcher(llm: Any):
    def bull_node(state) -> dict:
        debate = state["investment_debate_state"]
        history = debate.get("history", "")
        bull_history = debate.get("bull_history", "")
        current_response = opponent_argument_or_opening(
            debate.get("current_response", ""), "bear analyst"
        )

        prompt = f"""You are the Bull Analyst on a gold-only desk. Build the strongest
evidence-based case for being long gold (GC=F, USD/oz) over the stated horizon.

Argue from these channels, in this order of weight:
1. Real yields: the 10y TIPS real yield is falling, or has stopped rising while
   inflation expectations hold — that lowers gold's opportunity cost.
2. The dollar: the broad dollar is topping, or gold is rising *with* the dollar,
   which means both are being bought as reserve assets.
3. Policy path: the market is pricing more cuts / fewer hikes than the Fed has
   delivered, and the next data prints are likely to confirm it.
4. Official-sector and ETF demand: central-bank buying continues and ETF flow
   proxies turn positive — price-insensitive demand absorbing supply.
5. Positioning: the move is not crowded; there is room for longs to be added.

Rules of engagement:
- Quote the actual numbers from the reports with their dates. Never invent a
  level, a yield, a correlation or a flow figure.
- Where the bear's best point is about volatility, event risk or crowding, do not
  deny it — price it: say how the position should be sized, not that the risk
  isn't real.
- If a channel is genuinely unsupportive (e.g. ETF flows negative, or gold rich
  versus its macro beta), concede it and argue the remaining channels still carry
  the trade. A bull case that ignores its own reports is worthless.

{_debate_block(state)}

Conversation history of the debate: {history}
Last bear argument: {current_response}

Deliver a conversational argument, not a list, and end with the single strongest
falsifiable claim that would prove your case.""" + get_language_instruction()

        response = llm.invoke(prompt)
        argument = f"Bull Analyst: {response.content}"
        return {
            "investment_debate_state": {
                "history": history + "\n" + argument,
                "bull_history": bull_history + "\n" + argument,
                "bear_history": debate.get("bear_history", ""),
                "current_response": argument,
                "count": debate["count"] + 1,
            }
        }

    return bull_node


def create_gold_bear_researcher(llm: Any):
    def bear_node(state) -> dict:
        debate = state["investment_debate_state"]
        history = debate.get("history", "")
        bear_history = debate.get("bear_history", "")
        current_response = opponent_argument_or_opening(
            debate.get("current_response", ""), "bull analyst"
        )

        prompt = f"""You are the Bear Analyst on a gold-only desk. Build the strongest
evidence-based case against being long gold (GC=F, USD/oz) over the stated horizon.

Argue from these channels, in this order of weight:
1. Real yields: the 10y TIPS real yield is rising, or bottoming, which raises the
   opportunity cost of a metal that pays nothing.
2. The dollar: the broad dollar is breaking out, or gold is falling while the
   dollar is soft — a sign the bid is exhausted rather than merely resting.
3. Policy path: the market is underpricing hikes / overpricing cuts, and a hot
   inflation or strong labour print would reprice the path against gold.
4. Demand: ETF flow proxies are negative, or official-sector buying has slowed —
   the price-insensitive bid that carried the rally is fading.
5. Positioning: longs are crowded and the residual versus gold's macro beta is
   rich, so the risk is a long-liquidation move rather than a grind higher.

Rules of engagement:
- Quote the actual numbers from the reports with their dates. Never invent a
  level, a yield, a correlation or a flow figure.
- A bear case for gold is not "gold has no earnings" — the market knows that.
  Attack the *durability of the current bid*.
- Concede the channels that are genuinely bullish and argue they are already in
  the price.

{_debate_block(state)}

Conversation history of the debate: {history}
Last bull argument: {current_response}

Deliver a conversational argument, not a list, and end with the level or data
print that would prove you right.""" + get_language_instruction()

        response = llm.invoke(prompt)
        argument = f"Bear Analyst: {response.content}"
        return {
            "investment_debate_state": {
                "history": history + "\n" + argument,
                "bear_history": bear_history + "\n" + argument,
                "bull_history": debate.get("bull_history", ""),
                "current_response": argument,
                "count": debate["count"] + 1,
            }
        }

    return bear_node


def create_gold_trader(llm: Any):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state) -> dict:
        market_report = (state["market_report"] or "").strip()
        portfolio_context = get_portfolio_context_from_state(state)

        if market_report:
            grounding = (
                "Ground every level in the technical report's price structure — current close, "
                "support/resistance, ATR and the volatility regime — and take direction and "
                "strategy from the research plan. "
            )
            report_section = f"Gold technical report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the trader on a gold-only desk (COMEX front month, USD/oz). "
                    "Turn the research plan into one executable gold trade, or an explicit "
                    "decision to stand aside. "
                    + grounding
                    + "State entry price, stop-loss and target as absolute USD/oz levels "
                    "(for example 4025.0), never as percentages or ranges; if you cannot "
                    "state a number, omit the field. Size from the ATR-based stop distance "
                    "and the account risk, not from a lot count you cannot verify. "
                    "Name the event risk inside the holding period (FOMC, CPI, NFP, "
                    "geopolitical escalation, contract roll). "
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's plan for gold (GC=F). "
                    f"{get_instrument_context_from_state(state)}\n\n"
                    f"{report_section}"
                    f"{portfolio_context}\n\n"
                    f"Proposed Investment Plan:\n{state['investment_plan']}\n\n"
                    "Make an informed, strategic trading decision.\n\n"
                    "## Output\n\n"
                    "Write these sections, in this order, starting with the action on its "
                    "own line:\n\n"
                    "- **Action**: exactly one of Buy / Hold / Sell. A research "
                    "recommendation of Overweight is a Buy and Underweight is a Sell, sized "
                    "by how strong the case is; conflict alone is not a Hold.\n"
                    "- **Reasoning**: why, against the plan and the price structure, and "
                    "what would make you wrong fastest\n"
                    "- **Entry Price**, **Stop Loss**, **Target**, **Position Sizing**: "
                    "absolute USD/oz levels and size as a fraction of account risk"
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm, llm, messages, render_trader_proposal, "Trader"
        )
        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
        }

    return trader_node


_RISK_ROLES = {
    "aggressive": (
        "Aggressive Risk Analyst",
        """You champion the trade's upside. Your job is to make the strongest case for
taking *full* risk on the trader's gold decision: the macro regime is turning, the
crowd is not yet positioned, and a metal with a slow supply response can trend far
further than a mean-reversion model suggests. Attack the conservative and neutral
positions directly — where is their caution actually costing money? Push back on
"wait for confirmation" arguments with the cost of being late in a gold repricing.""",
        "aggressive_history",
        "current_aggressive_response",
    ),
    "conservative": (
        "Conservative Risk Analyst",
        """You defend capital preservation. Your job is to find what breaks: a hawkish
repricing of the policy path, a dollar breakout, an ETF-flow reversal, crowded
longs unwinding, a geopolitical premium evaporating overnight, or a stop placed
inside normal ATR noise. Argue for smaller size, a wider stop or waiting for the
event to pass — and name the specific scenario and the loss it would cause in
USD/oz. Do not argue for zero exposure without saying what evidence would justify
any.""",
        "conservative_history",
        "current_conservative_response",
    ),
    "neutral": (
        "Neutral Risk Analyst",
        """You hold the middle. Weigh the aggressive and conservative cases on evidence,
not on tone: which risks are measurable today (volatility percentile, ATR, event
calendar, flow z-scores) and which are speculative? Propose the *adjusted* trade —
a middle size, a scaled entry, a tighter invalidation — that captures most of the
upside while surviving the scenario the conservative analyst names.""",
        "neutral_history",
        "current_neutral_response",
    ),
}


def _make_gold_risk_analyst(llm: Any, role: str):
    title, stance, own_history_key, own_response_key = _RISK_ROLES[role]
    others = [other for other in _RISK_ROLES if other != role]

    def node(state) -> dict:
        risk = state["risk_debate_state"]
        history = risk.get("history", "")
        own_history = risk.get(own_history_key, "")
        opponents = {
            other: opponent_argument_or_opening(
                risk.get(_RISK_ROLES[other][3], ""),
                f"{_RISK_ROLES[other][0].lower()}",
            )
            for other in others
        }
        reports = _reports(state)

        prompt = f"""As the {title} on a gold-only desk, evaluate the trader's gold decision
(GC=F, USD/oz) and argue your position.

Here is the trader's decision:

{state['trader_investment_plan']}

{stance}

{get_instrument_context_from_state(state)}
{get_portfolio_context_from_state(state)}
Gold technical report: {reports['market']}
Gold macro report: {reports['macro']}
Gold flows/positioning report: {reports['flows']}
Gold news and event calendar: {reports['news']}

Conversation history: {history}
Last conservative argument: {opponents.get('conservative', '')}
Last aggressive argument: {opponents.get('aggressive', '')}
Last neutral argument: {opponents.get('neutral', '')}

If the others have not spoken yet, open with your own case. Argue in gold terms:
real yields, the dollar, policy path, official-sector and ETF demand, volatility
regime, event risk, and USD/oz levels. Quote only numbers that appear in the
reports. Output conversationally, as if speaking — no headings, no tables.""" + get_language_instruction()

        response = llm.invoke(prompt)
        argument = f"{title}: {response.content}"
        new_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk.get("aggressive_history", ""),
            "conservative_history": risk.get("conservative_history", ""),
            "neutral_history": risk.get("neutral_history", ""),
            "latest_speaker": title.split()[0],
            "current_aggressive_response": risk.get("current_aggressive_response", ""),
            "current_conservative_response": risk.get("current_conservative_response", ""),
            "current_neutral_response": risk.get("current_neutral_response", ""),
            "count": risk["count"] + 1,
        }
        new_state[own_history_key] = own_history + "\n" + argument
        new_state[own_response_key] = argument
        return {"risk_debate_state": new_state}

    return node


def create_gold_aggressive_debator(llm: Any):
    return _make_gold_risk_analyst(llm, "aggressive")


def create_gold_conservative_debator(llm: Any):
    return _make_gold_risk_analyst(llm, "conservative")


def create_gold_neutral_debator(llm: Any):
    return _make_gold_risk_analyst(llm, "neutral")


GOLD_ANALYST_BUILDERS = {
    "gold_market": create_gold_market_analyst,
    "gold_macro": create_gold_macro_analyst,
    "gold_positioning": create_gold_positioning_analyst,
    "gold_news": create_gold_news_analyst,
}


def gold_analyst_factories(llm: Any) -> dict:
    """Factory map for the graph builder, one per gold analyst key."""
    return {key: (lambda builder=builder: builder(llm)) for key, builder in GOLD_ANALYST_BUILDERS.items()}


def build_gold_debate_nodes(quick_llm: Any, deep_llm: Any) -> dict:
    """The debate/trader/risk nodes for a gold run.

    The Research Manager and Portfolio Manager are reused from upstream: their
    prompts are instrument-agnostic (rating scale, evidence weighing) and adding
    gold framing there would only risk diverging from the rating contract.
    """
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
    from tradingagents.agents.managers.research_manager import create_research_manager

    return {
        "bull": create_gold_bull_researcher(quick_llm),
        "bear": create_gold_bear_researcher(quick_llm),
        "manager": create_research_manager(deep_llm),
        "trader": create_gold_trader(quick_llm),
        "aggressive": create_gold_aggressive_debator(quick_llm),
        "conservative": create_gold_conservative_debator(quick_llm),
        "neutral": create_gold_neutral_debator(quick_llm),
        "pm": create_portfolio_manager(deep_llm),
    }
