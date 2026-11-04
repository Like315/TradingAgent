"""The gold desk: instrument lock, config profile, analyst wiring and prompt hygiene.

No network and no model calls: these pin the things that would silently turn the
gold desk back into a generic equity desk — a symbol passing through, an analyst
writing into a slot nobody reads, or equity language leaking into a gold prompt.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.context import analyst_title, build_instrument_context, report_slot_label
from tradingagents.dataflows.config import run_config
from tradingagents.default_config import build_default_config
from tradingagents.gold import prompts
from tradingagents.gold.agents import GOLD_ANALYST_BUILDERS, gold_analyst_factories
from tradingagents.gold.config import GOLD_ANALYSTS, gold_config
from tradingagents.gold.spec import GOLD_SYMBOL, NotGoldError, cny_per_gram, resolve_gold_symbol
from tradingagents.graph.analyst_execution import build_analyst_execution_plan

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "raw",
    ["GC=F", "gc=f", "XAUUSD", "xauusd+", "XAU", "GOLD", "黄金", "伦敦金", "现货黄金", "GCZ26.CMX"],
)
def test_gold_spellings_resolve_to_the_comex_future(raw):
    assert resolve_gold_symbol(raw) == GOLD_SYMBOL


def test_blank_symbol_defaults_to_gold():
    assert resolve_gold_symbol(None) == GOLD_SYMBOL
    assert resolve_gold_symbol("   ") == GOLD_SYMBOL


@pytest.mark.parametrize("raw", ["AAPL", "BTC-USD", "0700.HK", "SPY"])
def test_non_gold_instruments_are_rejected(raw):
    with pytest.raises(NotGoldError):
        resolve_gold_symbol(raw)


@pytest.mark.parametrize("raw", ["AU9999", "AU(T+D)", "沪金", "518880"])
def test_domestic_cny_contracts_are_rejected_with_a_conversion_hint(raw):
    with pytest.raises(NotGoldError) as excinfo:
        resolve_gold_symbol(raw)
    assert "CNY" in str(excinfo.value)


def test_cny_parity_conversion():
    # 4,000 USD/oz at 7.10 CNY/USD is about 913 CNY/g.
    assert cny_per_gram(4000, 7.10) == pytest.approx(913.0, abs=0.5)


def test_gold_config_locks_the_benchmark_to_gold_and_uses_gold_analysts(monkeypatch):
    # The suite's conftest points the state directories at a temp dir; clear them
    # so this asserts the gold profile's own defaults.
    for name in ("TRADINGAGENTS_RESULTS_DIR", "TRADINGAGENTS_CACHE_DIR", "TRADINGAGENTS_MEMORY_LOG_PATH"):
        monkeypatch.delenv(name, raising=False)
    cfg = gold_config()
    assert cfg["gold_mode"] is True
    assert cfg["benchmark_ticker"] == GOLD_SYMBOL
    assert tuple(cfg["analysts"]) == GOLD_ANALYSTS
    assert cfg["data_vendors"]["macro_data"] == "fred"
    assert cfg["instrument_context_note"] == prompts.GOLD_DOMAIN_BRIEF
    # A gold desk's memory log must not mix with equity runs.
    assert "gold" in cfg["memory_log_path"].replace("\\", "/")


def test_gold_config_leaves_the_upstream_default_untouched():
    gold_config()
    assert build_default_config()["benchmark_ticker"] is None
    assert build_default_config().get("gold_mode") is None


def test_gold_config_overrides_and_language():
    cfg = gold_config(language="English", thorough=False, llm_provider="deepseek")
    assert cfg["output_language"] == "English"
    assert cfg["max_debate_rounds"] == 1
    assert cfg["llm_provider"] == "deepseek"


def test_gold_analysts_fill_the_state_slots_the_rest_of_the_graph_reads():
    plan = build_analyst_execution_plan(GOLD_ANALYSTS)
    keys = {spec.report_key for spec in plan.specs}
    # These are the four slots bull/bear, the risk team and the report writer read.
    assert keys == {"market_report", "fundamentals_report", "sentiment_report", "news_report"}
    assert len(plan.specs) == 4


def test_each_gold_analyst_has_tools_and_a_factory():
    factories = gold_analyst_factories(object())
    assert set(factories) == set(GOLD_ANALYSTS) == set(GOLD_ANALYST_BUILDERS)
    for spec in build_analyst_execution_plan(GOLD_ANALYSTS).specs:
        assert spec.tools, f"{spec.key} has no tools"
        for tool in spec.tools:
            assert tool.name and callable(tool.invoke)


def test_gold_tools_are_named_for_what_they_answer():
    names = {
        tool.name
        for spec in build_analyst_execution_plan(GOLD_ANALYSTS).specs
        for tool in spec.tools
    }
    assert {"get_gold_macro_dashboard", "get_gold_flows_and_positioning",
            "get_gold_volatility_regime", "get_gold_domestic_reference"} <= names


def test_instrument_context_carries_the_gold_brief_and_labels():
    with run_config(gold_config()):
        context = build_instrument_context(GOLD_SYMBOL, asset_type="commodity")
        assert "GOLD-ONLY MANDATE" in context
        assert "real yield" in context.lower()
        assert report_slot_label("fundamentals_report", "Fundamentals") == (
            "Gold macro report (real yields, dollar, policy, inflation)"
        )
        assert analyst_title("gold_macro", "gold_macro") == "Gold Macro Analyst"


def test_generic_config_is_unchanged_by_the_gold_profile():
    with run_config({}):
        assert "GOLD-ONLY MANDATE" not in build_instrument_context("AAPL")
        assert report_slot_label("fundamentals_report", "Fundamentals") == "Fundamentals"
        assert analyst_title("market", "Market") == "Market"


@pytest.mark.parametrize(
    "prompt_text",
    [
        prompts.gold_market_analyst_prompt(),
        prompts.gold_macro_analyst_prompt(),
        prompts.gold_positioning_analyst_prompt(),
        prompts.gold_news_analyst_prompt(),
        prompts.gold_trader_prompt(),
        prompts.GOLD_DOMAIN_BRIEF,
    ],
)
def test_prompts_are_gold_specific_and_free_of_equity_framing(prompt_text):
    lowered = prompt_text.lower()
    assert "gold" in lowered
    for equity_phrase in ("competitive advantage", "revenue projection", "p/e", "earnings per share"):
        assert equity_phrase not in lowered


# --------------------------------------------------------------------------- #
# The USD/CNY side of the parity panel, when Yahoo cannot be reached
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_sina_spot_reads_the_latest_print(monkeypatch):
    from tradingagents.gold.vendors import sina

    payload = (
        'var hq_str_fx_susdcny="21:12:00,6.6612000000,6.6620000000,6.6660000000,41.0,'
        '6.6640000000,6.6649000000,6.6608000000,6.6613000000,在岸人民币,-0.0600,'
        '-0.0040,0.0041,此行情由新浪财经计算得出,0.0000,0.0000,,2026-10-29";'
    )
    monkeypatch.setattr(sina, "_get", lambda url, params: _FakeResponse(text=payload))
    assert sina.spot_usdcny() == pytest.approx(6.6613)


@pytest.mark.parametrize("bad", ["66.613", "0.666", "abc"])
def test_sina_spot_refuses_an_implausible_rate(monkeypatch, bad):
    """A reshuffled payload must produce "unavailable", never a wrong conversion."""
    from tradingagents.gold.vendors import sina

    payload = f'var hq_str_fx_susdcny="21:12:00,{bad},{bad},x,0,0,0,0,{bad},在岸人民币";'
    monkeypatch.setattr(sina, "_get", lambda url, params: _FakeResponse(text=payload))
    assert sina.spot_usdcny() is None


def test_eastmoney_quote_scales_by_the_endpoints_own_decimals(monkeypatch):
    from tradingagents.gold.vendors import eastmoney

    payloads = {
        "133.USDCNH": {"data": {"f43": 66590, "f59": 4, "f58": "美元兑离岸人民币", "f60": 66615}},
        "101.GC00Y": {"data": {"f43": 40342, "f59": 1, "f58": "COMEX黄金", "f60": 40516}},
    }

    def fake_get(url, params):
        return _FakeResponse(payload=payloads[params["secid"]])

    monkeypatch.setattr(eastmoney, "_get", fake_get)
    assert eastmoney.quote("CNY=X")["price"] == pytest.approx(6.659)
    assert eastmoney.quote("GC=F")["price"] == pytest.approx(4034.2)
    assert eastmoney.quote("SI=F") is None  # not covered, no request made


def test_spot_falls_back_from_eastmoney_to_sina(monkeypatch):
    from tradingagents.gold import data as gdata
    from tradingagents.gold.vendors import eastmoney, sina

    monkeypatch.setattr(eastmoney, "quote", lambda symbol: {"price": 6.659, "name": "美元兑离岸人民币"})
    assert gdata.spot("CNY=X")[0] == pytest.approx(6.659)
    assert "东财" in gdata.spot("CNY=X")[1]

    monkeypatch.setattr(eastmoney, "quote", lambda symbol: None)
    monkeypatch.setattr(sina, "spot_batch", lambda codes: {codes[0]: 6.662})
    assert gdata.spot("CNY=X")[0] == pytest.approx(6.662)
    assert "新浪" in gdata.spot("CNY=X")[1]

    # The dollar index has its own Sina code, so the same path serves it.
    monkeypatch.setattr(sina, "spot_batch", lambda codes: {"DINIW": 102.35} if "DINIW" in codes else {})
    assert gdata.spot("DX-Y.NYB")[0] == pytest.approx(102.35)

    monkeypatch.setattr(sina, "spot_batch", lambda codes: {})
    assert gdata.spot("CNY=X") == (None, "")
    assert gdata.spot("^GSPC") == (None, "")  # uncovered symbol: no request at all


def test_domestic_reference_keeps_the_parity_panel_without_history(monkeypatch):
    """No daily FX history, no Yahoo: one realtime quote still fills the table."""
    import pandas as pd

    from tradingagents.gold import data as gdata
    from tradingagents.gold import tools as gtools
    from tradingagents.gold.vendors import eastmoney

    monkeypatch.setattr(
        gdata, "closes", lambda symbol, as_of, rows=750: pd.Series([4032.0], index=pd.to_datetime(["2026-10-29"]))
    )
    monkeypatch.setattr(gdata, "fx_closes", lambda as_of, rows=30, symbol="CNY=X": pd.Series(dtype=float))
    monkeypatch.setattr(gdata, "spot", lambda symbol: (6.659, "USD/CNH 实时（东财）"))
    monkeypatch.setattr(eastmoney, "domestic_gold_close", lambda as_of=None, rows=10: pd.Series(dtype=float))
    monkeypatch.setattr(eastmoney, "quote", lambda symbol: None)

    text = gtools._domestic_reference("2026-10-29")
    assert "DATA_UNAVAILABLE" not in text
    assert "6.659" in text
    assert "USD/CNH 实时（东财）" in text
    assert "863.22" in text.replace(",", "")  # CNY/g parity for 4032 @ 6.659


def test_macro_dashboard_quotes_a_realtime_dollar_when_history_is_missing(monkeypatch):
    """A dollar level with no history beats "unavailable" — as long as it is labelled."""
    import pandas as pd

    from tradingagents.gold import data as gdata
    from tradingagents.gold import tools as gtools

    gold = pd.Series([3900.0 + i for i in range(60)], index=pd.bdate_range("2026-08-03", periods=60))

    def fake_closes(symbol, as_of, rows=750):
        if symbol == gtools.DOLLAR_INDEX:
            raise RuntimeError("Yahoo Finance is unreachable")
        return gold

    monkeypatch.setattr(gdata, "closes", fake_closes)
    monkeypatch.setattr(gdata, "spot", lambda symbol: (102.4, "美元指数 实时（东财，仅最新价）"))
    # load_drivers is imported inside the function, so patch it at its source.
    monkeypatch.setattr(
        "tradingagents.gold.macro.load_drivers", lambda trade_date, look_back_days: ({}, {}, [])
    )

    text = gtools._macro_dashboard("2026-10-29", 365)
    assert "102.40" in text
    assert "realtime quote, level only" in text
    assert "do not infer a trend" in text


def test_online_stock_tool_honours_the_domestic_source(monkeypatch):
    """The market analyst's price tool must not pay Yahoo's timeout either."""
    import pandas as pd

    from tradingagents.dataflows.config import run_config
    from tradingagents.dataflows.vendors.yahoo import market

    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2026-10-27", "2026-10-28", "2026-10-29"]),
        "Open": [4080.0, 4073.6, 4081.4], "High": [4090.0, 4097.2, 4081.7],
        "Low": [4060.0, 4036.9, 4027.0], "Close": [4073.1, 4081.2, 4047.3],
        "Volume": [132510.0, 143669.0, 95340.0],
    })
    asked = {"yahoo": False}
    monkeypatch.setattr(market, "domestic_frame", lambda canonical, end_str: frame)
    monkeypatch.setattr(market, "yf_retry", lambda fn: asked.__setitem__("yahoo", True))

    with run_config({"gold_price_source": "domestic"}):
        text = market.get_YFin_data_online("GC=F", "2026-10-25", "2026-10-29")

    assert asked["yahoo"] is False
    assert "2026-10-29" in text
    # The window is cut to what was asked for, not the whole five-year history.
    assert "2026-10-27" in text
    assert "Total records: 3" in text


def test_headlines_refuse_to_rewind_the_wire():
    """A backdated run gets "no archive", never today's news dressed as history."""
    from tradingagents.gold import tools as gtools

    text = gtools._headlines("2026-01-05", limit=5)
    assert text.startswith("DATA_UNAVAILABLE")
    assert "no headline archive" in text


def test_news_flashes_parse_and_the_filter_drops_off_driver_noise(monkeypatch):
    from tradingagents.gold.vendors import news_cn

    payload = {"result": {"data": {"feed": {"list": [
        {"rich_text": "哈塞特：10月CPI和PCE数据应能明确美联储应采取何种行动。", "create_time": "2026-10-29 22:45:05"},
        {"rich_text": "TD Cowen上调微软目标价至650美元", "create_time": "2026-10-29 22:37:00"},
    ]}}}}
    monkeypatch.setattr(news_cn, "_get", lambda url, params: payload)
    monkeypatch.setattr(news_cn, "roll", lambda limit=50: [])

    parsed = news_cn.flash(limit=2)
    assert [item["text"][:2] for item in parsed] == ["哈塞", "TD"]

    kept = news_cn.headlines(limit=10)
    assert len(kept) == 1
    assert "美联储" in kept[0]["text"]
    assert "微软" not in kept[0]["text"]


def test_news_caps_one_speakers_repeated_remarks(monkeypatch):
    """A wire repeats one speaker's every line; two is enough to carry the point."""
    from tradingagents.gold.vendors import news_cn

    flashes = [
        {"ts": f"2026-10-29 22:4{i}:00", "text": f"拉加德：欧元区通胀{i}号表述", "source": "Sina 7x24", "url": ""}
        for i in range(5)
    ]
    monkeypatch.setattr(news_cn, "flash", lambda limit=80: flashes)
    monkeypatch.setattr(news_cn, "roll", lambda limit=50: [])
    assert len(news_cn.headlines(limit=10)) == 2


# --------------------------------------------------------------------------- #
# End to end: the gold graph, scripted models, no network
# --------------------------------------------------------------------------- #
import copy  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from pydantic import Field  # noqa: E402

from tradingagents.agents import context as agent_context  # noqa: E402
from tradingagents.agents import schemas  # noqa: E402
from tradingagents.dataflows import router  # noqa: E402
from tradingagents.dataflows.vendors.yahoo import snapshot as yahoo_snapshot  # noqa: E402
from tradingagents.gold import data as gold_data  # noqa: E402
from tradingagents.graph import trading_graph  # noqa: E402

TRADE_DATE = "2026-01-09"
TEXT = "Report.\n\n**Rating**: Overweight\n\nFINAL TRANSACTION PROPOSAL: **BUY**"

STRUCTURED = {
    schemas.ResearchPlan: schemas.ResearchPlan(
        recommendation=schemas.PortfolioRating.OVERWEIGHT, rationale="r", strategic_actions="a"),
    schemas.TraderProposal: schemas.TraderProposal(action=schemas.TraderAction.BUY, reasoning="r"),
    schemas.PortfolioDecision: schemas.PortfolioDecision(
        rating=schemas.PortfolioRating.OVERWEIGHT, executive_summary="s",
        investment_thesis="Real yields are falling while central banks keep buying."),
}

TOOL_ARGS = {
    "symbol": "GC=F", "ticker": "GC=F", "curr_date": TRADE_DATE,
    "start_date": "2026-01-02", "end_date": TRADE_DATE, "indicator": "rsi",
    "topic": "Fed rate cut", "freq": "quarterly", "look_back_days": 365, "limit": 6,
}


class _Scripted(BaseChatModel):
    """Calls every bound tool once, then answers with TEXT."""

    structured: bool = False
    tools: tuple = ()
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools": tuple(tools)})

    def with_structured_output(self, schema, **kwargs):
        if not self.structured:
            raise NotImplementedError
        return RunnableLambda(lambda _: self.calls.append(1) or STRUCTURED[schema])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(1)
        if self.tools and not isinstance(messages[-1], ToolMessage):
            props = self.tools[0].tool_call_schema.model_json_schema()["properties"] if self.tools else {}
            calls = []
            for index, tool in enumerate(self.tools):
                schema_props = tool.tool_call_schema.model_json_schema()["properties"]
                calls.append({
                    "name": tool.name,
                    "id": f"call_{index}",
                    "args": {k: v for k, v in TOOL_ARGS.items() if k in schema_props},
                })
            message = AIMessage(content="", tool_calls=calls)
        else:
            message = AIMessage(content=TEXT)
        return ChatResult(generations=[ChatGeneration(message=message)])


class _Client:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


@pytest.fixture
def offline_gold(monkeypatch):
    """Every data source answers offline; returns which gold series were touched."""
    seen: set[str] = set()

    for method, vendors in router.VENDOR_METHODS.items():
        for vendor in vendors:
            monkeypatch.setitem(vendors, vendor, lambda *a, _m=method, **k: f"{_m} data")

    prices = pd.DataFrame({
        "Date": pd.bdate_range(end=TRADE_DATE, periods=80),
        "Open": 3000.0, "High": 3020.0, "Low": 2980.0, "Close": 3010.0, "Volume": 150_000,
    })
    monkeypatch.setattr(yahoo_snapshot, "load_ohlcv", lambda *a, **k: prices.copy())

    def fake_closes(symbol, as_of, rows=750):
        seen.add(symbol)
        index = pd.bdate_range(end=TRADE_DATE, periods=rows)
        return pd.Series(np.linspace(2600, 3010, len(index)), index=index)

    def fake_frame(symbol, as_of, fill_gaps=True):
        seen.add(symbol)
        index = pd.bdate_range(end=TRADE_DATE, periods=300)
        close = np.linspace(2800, 3010, len(index))
        return pd.DataFrame({
            "Date": index, "Open": close - 5, "High": close + 12,
            "Low": close - 12, "Close": close, "Volume": 150_000.0,
        })

    def fake_fred(indicator, as_of, look_back_days=1095):
        seen.add(f"FRED:{indicator}")
        index = pd.date_range(end=TRADE_DATE, periods=36, freq="ME")
        return pd.Series(np.linspace(2.4, 1.7, len(index)), index=index)

    monkeypatch.setattr(gold_data, "closes", fake_closes)
    monkeypatch.setattr(gold_data, "price_frame", fake_frame)
    monkeypatch.setattr(gold_data, "fred_series", fake_fred)

    monkeypatch.setattr(agent_context, "get_company_profile", lambda ticker: {"longName": "Gold"})
    agent_context._identity.cache_clear()
    return seen


def _gold_graph(tmp_path, monkeypatch, model, **config):
    cfg = gold_config()
    cfg.update(results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
               memory_log_path=str(tmp_path / "log.md"), **config)
    monkeypatch.setattr(trading_graph, "create_tier_client", lambda c, tier, **k: _Client(model))
    return trading_graph.TradingAgentsGraph(selected_analysts=cfg["analysts"], config=cfg)


@pytest.mark.unit
@pytest.mark.parametrize("structured", [False, True], ids=["free-text", "structured"])
def test_a_full_gold_run_reaches_a_decision(tmp_path, monkeypatch, offline_gold, structured):
    graph = _gold_graph(tmp_path, monkeypatch, _Scripted(structured=structured))

    state, signal = graph.propagate("GC=F", TRADE_DATE, asset_type="commodity")

    assert signal == state["final_rating"] == "Overweight"
    for key in ("market_report", "fundamentals_report", "sentiment_report", "news_report",
                "investment_plan", "trader_investment_plan", "final_trade_decision"):
        assert state[key].strip(), key
    # The gold tools ran (not silently skipped) and read the gold/macro series.
    assert "GC=F" in offline_gold
    assert any(name.startswith("FRED:") for name in offline_gold)
    # Every agent's prompt carried the gold brief, not equity framing.
    assert "GOLD-ONLY MANDATE" in graph.resolve_instrument_context("GC=F", "commodity")


@pytest.mark.unit
def test_a_gold_run_writes_a_report_with_gold_titles(tmp_path, monkeypatch, offline_gold):
    graph = _gold_graph(tmp_path, monkeypatch, _Scripted())
    state, _signal = graph.propagate("GC=F", TRADE_DATE, asset_type="commodity")

    path = graph.save_reports(state, "GC=F", save_path=tmp_path / "report", html=True)
    markdown = path.read_text(encoding="utf-8")

    assert "Gold macro report" in markdown
    assert "Gold flows and positioning report" in markdown
    assert (tmp_path / "report" / "complete_report.html").exists()


@pytest.mark.unit
def test_run_settings_name_the_gold_analysts(tmp_path, monkeypatch, offline_gold):
    graph = _gold_graph(tmp_path, monkeypatch, _Scripted())
    settings = graph.run_settings()

    assert settings["analysts"] == list(GOLD_ANALYSTS)
    # Alpha is measured against holding gold, not against an equity index.
    assert graph.config["benchmark_ticker"] == GOLD_SYMBOL
