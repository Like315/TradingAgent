"""The four gold analysts.

Each one keeps the upstream node shape (a private message history, its own tools,
a report key) so the rest of the graph — debate, trader, risk, memory — works
unchanged. What differs is that every prompt is written for a metal rather than a
company, and each analyst is given the tool that answers *its* question:

    gold_market       price structure, volatility regime, domestic conversion
    gold_macro        real yields, dollar, policy path, macro beta  -> fundamentals slot
    gold_positioning  ETF flows, cross-asset ratios, seasonality    -> sentiment slot
    gold_news         gold-specific headlines and the event calendar

The macro and positioning analysts write into the ``fundamentals_report`` and
``sentiment_report`` state slots on purpose: the researchers, risk team and
report writer downstream already read those slots, so reusing them means the
gold pipeline needs no changes to the rest of the graph. The slot labels are
re-titled through the gold config.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.tools import (
    get_global_news,
    get_indicators,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
    get_stock_data,
    get_verified_market_snapshot,
)
from tradingagents.gold import prompts
from tradingagents.gold.tools import (
    get_gold_domestic_reference,
    get_gold_flows_and_positioning,
    get_gold_headlines,
    get_gold_macro_dashboard,
    get_gold_volatility_regime,
)

GOLD_MARKET_TOOLS = (
    get_stock_data,
    get_indicators,
    get_verified_market_snapshot,
    get_gold_volatility_regime,
    get_gold_domestic_reference,
)
GOLD_MACRO_TOOLS = (
    get_gold_macro_dashboard,
    get_macro_indicators,
    get_prediction_markets,
    get_news,
)
GOLD_POSITIONING_TOOLS = (
    get_gold_flows_and_positioning,
    get_gold_volatility_regime,
    get_news,
    get_prediction_markets,
)
GOLD_NEWS_TOOLS = (
    # The China-reachable wire feed comes first: on a network where Yahoo news
    # and Polymarket are blocked it is the only source of an actual catalyst,
    # and the upstream tools below then answer "unavailable" honestly.
    get_gold_headlines,
    get_news,
    get_global_news,
    get_macro_indicators,
    get_prediction_markets,
)


def _gold_analyst(llm: Any, system_message: str, tools: tuple, report_key: str):
    """One analyst node: prompt, tools, report — same shape as the upstream nodes."""

    def node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call"
                    " date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join(tool.name for tool in tools))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, tools, state["messages"])
        return {"messages": [result], report_key: report}

    return node


def create_gold_market_analyst(llm: Any):
    system_message = prompts.gold_market_analyst_prompt() + get_language_instruction()
    return _gold_analyst(llm, system_message, GOLD_MARKET_TOOLS, "market_report")


def create_gold_macro_analyst(llm: Any):
    system_message = prompts.gold_macro_analyst_prompt() + get_language_instruction()
    return _gold_analyst(llm, system_message, GOLD_MACRO_TOOLS, "fundamentals_report")


def create_gold_positioning_analyst(llm: Any):
    system_message = prompts.gold_positioning_analyst_prompt() + get_language_instruction()
    return _gold_analyst(llm, system_message, GOLD_POSITIONING_TOOLS, "sentiment_report")


def create_gold_news_analyst(llm: Any):
    system_message = prompts.gold_news_analyst_prompt() + get_language_instruction()
    return _gold_analyst(llm, system_message, GOLD_NEWS_TOOLS, "news_report")
