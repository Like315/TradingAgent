from collections.abc import Iterable
from dataclasses import dataclass

from tradingagents.agents.analysts import fundamentals_analyst, market_analyst, news_analyst
from tradingagents.gold.analysts import (
    GOLD_MACRO_TOOLS,
    GOLD_MARKET_TOOLS,
    GOLD_NEWS_TOOLS,
    GOLD_POSITIONING_TOOLS,
)


@dataclass(frozen=True)
class AnalystNodeSpec:
    key: str
    agent_node: str
    report_key: str
    tools: tuple = ()


@dataclass(frozen=True)
class AnalystExecutionPlan:
    specs: list[AnalystNodeSpec]


ANALYST_NODE_SPECS: dict[str, AnalystNodeSpec] = {
    "market": AnalystNodeSpec(
        key="market",
        agent_node="Market Analyst",
        report_key="market_report",
        tools=market_analyst.TOOLS,
    ),
    "social": AnalystNodeSpec(
        # Saved configs select this analyst as "social". It fetches its
        # sources before calling the model, so it has no tools.
        key="social",
        agent_node="Sentiment Analyst",
        report_key="sentiment_report",
    ),
    "news": AnalystNodeSpec(
        key="news",
        agent_node="News Analyst",
        report_key="news_report",
        tools=news_analyst.TOOLS,
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        agent_node="Fundamentals Analyst",
        report_key="fundamentals_report",
        tools=fundamentals_analyst.TOOLS,
    ),
    # --- gold desk ---------------------------------------------------------
    # The gold analysts write into the same report keys the rest of the graph
    # already reads (market/fundamentals/sentiment/news), so the debate, trader,
    # risk team and report writer need no gold-specific branch: only the prompt
    # and the tools behind each slot differ. See tradingagents/gold/.
    "gold_market": AnalystNodeSpec(
        key="gold_market",
        agent_node="Gold Market Analyst",
        report_key="market_report",
        tools=GOLD_MARKET_TOOLS,
    ),
    "gold_macro": AnalystNodeSpec(
        key="gold_macro",
        agent_node="Gold Macro Analyst",
        report_key="fundamentals_report",
        tools=GOLD_MACRO_TOOLS,
    ),
    "gold_positioning": AnalystNodeSpec(
        key="gold_positioning",
        agent_node="Gold Flows Analyst",
        report_key="sentiment_report",
        tools=GOLD_POSITIONING_TOOLS,
    ),
    "gold_news": AnalystNodeSpec(
        key="gold_news",
        agent_node="Gold News Analyst",
        report_key="news_report",
        tools=GOLD_NEWS_TOOLS,
    ),
}


def build_analyst_execution_plan(
    selected_analysts: Iterable[str],
) -> AnalystExecutionPlan:
    specs: list[AnalystNodeSpec] = []
    for analyst_key in selected_analysts:
        spec = ANALYST_NODE_SPECS.get(analyst_key)
        if spec is None:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        specs.append(spec)

    if not specs:
        raise ValueError("at least one analyst must be selected")

    return AnalystExecutionPlan(specs=specs)


