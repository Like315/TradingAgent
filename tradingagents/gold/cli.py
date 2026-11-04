"""``tradingagents-gold`` — the gold desk's command line.

    tradingagents-gold                      # analyze gold as of today
    tradingagents-gold --date 2026-10-28    # analyze a specific day
    tradingagents-gold --fast               # one debate round per side
    tradingagents-gold data                 # print the gold data dashboards, no LLM
    tradingagents-gold doctor               # check keys, network and tool health

Differences from the upstream CLI on purpose: there is no ticker prompt (the
instrument is gold), no analyst picker beyond the four gold analysts, and the
data/doctor commands let you verify the data layer without spending a token.
"""

from __future__ import annotations

from datetime import date as _date

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from tradingagents.gold.config import GOLD_ANALYSTS, gold_config, gold_home
from tradingagents.gold.ict.cli import ICT_HELP, ict_command
from tradingagents.gold.intraday.cli import DASHBOARD_HELP, SCALP_HELP, dashboard_command, scalp_command
from tradingagents.gold.spec import GOLD_SYMBOL, NotGoldError, resolve_gold_symbol

app = typer.Typer(
    name="tradingagents-gold",
    help="Gold-only multi-agent research desk (COMEX GC=F, USD/oz).",
    add_completion=False,
)
app.command("scalp", help=SCALP_HELP)(scalp_command)
app.command("dashboard", help=DASHBOARD_HELP)(dashboard_command)
app.command("ict", help=ICT_HELP)(ict_command)
console = Console()

# Chinese market convention: up / bullish is red, down / bearish is green.
_RATING_STYLE = {
    "Buy": "bold red",
    "Overweight": "red",
    "Hold": "yellow",
    "Underweight": "green",
    "Sell": "bold green",
    "REVIEW": "magenta",
}


# Report slot -> the analyst that fills it, so the result panel can tell "this
# analyst was not requested" apart from "this analyst produced nothing".
_SLOT_ANALYST = {
    "technical": "gold_market",
    "macro": "gold_macro",
    "flows": "gold_positioning",
    "news": "gold_news",
}


# Probed only to prove reachability, for providers the OpenAI-compatible
# registry does not describe (it holds the endpoints of everything else).
# ``None`` means the endpoint is supplied per-account, so there is nothing to guess.
_NATIVE_ENDPOINTS: dict[str, str | None] = {
    "anthropic": "https://api.anthropic.com/v1/models",
    "google": "https://generativelanguage.googleapis.com/v1beta/models",
    "azure": None,    # AZURE_OPENAI_ENDPOINT
    "bedrock": None,  # regional AWS endpoint, resolved by the SDK
}


def _endpoint_for(provider: str, base_url: str | None) -> str:
    """The URL doctor should knock on for ``provider`` (empty when unknown)."""
    if base_url:
        return base_url
    try:
        from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

        spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
        if spec is not None:
            # "openai" leaves base_url to the SDK, so name the SDK's default.
            return spec.base_url or ("https://api.openai.com/v1" if provider == "openai" else "")
    except Exception:  # noqa: BLE001 — the probe is a diagnostic, never a gate
        pass
    return _NATIVE_ENDPOINTS.get(provider) or ""


def _print_banner(cfg: dict, trade_date: str, analysts: tuple[str, ...]) -> None:
    provider = cfg.get("quick_think_provider") or cfg.get("llm_provider")
    deep_provider = cfg.get("deep_think_provider") or cfg.get("llm_provider")
    source = cfg.get("gold_price_source", "domestic")
    price_source = {
        "domestic": "Eastmoney/Sina",
        "auto": "Yahoo, then Eastmoney/Sina",
        "yahoo": "Yahoo",
    }.get(source, source)
    console.print(
        Panel.fit(
            f"[bold]Gold desk[/bold] · {GOLD_SYMBOL} (COMEX front month, USD/oz)\n"
            f"analysis date [bold]{trade_date}[/bold] · analysts {', '.join(analysts)}\n"
            f"models {provider}/{cfg.get('quick_think_llm')} · "
            f"deep {deep_provider}/{cfg.get('deep_think_llm')}\n"
            f"data  prices {price_source} · macro FRED/US Treasury · events Polymarket\n"
            f"state {gold_home()}",
            title="tradingagents-gold",
            border_style="yellow",
        )
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    date: str = typer.Option(None, "--date", help="Analysis date YYYY-MM-DD (default: today)"),
    symbol: str = typer.Option(
        GOLD_SYMBOL, "--symbol", help="Gold spelling: GC=F, XAUUSD, 伦敦金 (all resolve to GC=F)"
    ),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated subset of: " + ", ".join(GOLD_ANALYSTS)
    ),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, so sizing uses the real book"
    ),
    provider: str = typer.Option(None, "--provider", help="LLM provider, e.g. openai, deepseek, ollama"),
    quick_model: str = typer.Option(None, "--quick-model", help="Model for analysts/debaters"),
    deep_model: str = typer.Option(None, "--deep-model", help="Model for the managers"),
    language: str = typer.Option("Chinese", "--language", help="Report language"),
    source: str = typer.Option(
        "domestic", "--source",
        help="Price source: domestic (Eastmoney/Sina, default) | auto (Yahoo first) | yahoo",
    ),
    fast: bool = typer.Option(False, "--fast", help="One debate round per side instead of two"),
    checkpoint: bool = typer.Option(False, "--checkpoint", help="Enable resume-after-crash checkpoints"),
    save: bool = typer.Option(True, "--save/--no-save", help="Write the report tree"),
    show: bool = typer.Option(False, "--show/--no-show", help="Print the final decision at the end"),
    html: bool = typer.Option(True, "--html/--no-html", help="Also write complete_report.html"),
):
    """Run the gold desk. This is what a bare `tradingagents-gold` does."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        resolve_gold_symbol(symbol)
    except NotGoldError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    selected = tuple(part.strip() for part in analysts.split(",") if part.strip()) if analysts else GOLD_ANALYSTS
    unknown = [key for key in selected if key not in GOLD_ANALYSTS]
    if unknown:
        console.print(
            f"[red]Unknown analyst(s): {', '.join(unknown)}. "
            f"Available: {', '.join(GOLD_ANALYSTS)}[/red]"
        )
        raise typer.Exit(code=2)

    trade_date = date or _date.today().isoformat()
    overrides: dict = {
        "checkpoint_enabled": checkpoint,
        "output_language": language,
        "gold_price_source": source,
    }
    if provider:
        overrides["llm_provider"] = provider
        overrides["quick_think_provider"] = provider
        overrides["deep_think_provider"] = provider
    if quick_model:
        overrides["quick_think_llm"] = quick_model
    if deep_model:
        overrides["deep_think_llm"] = deep_model

    cfg = gold_config(language=language, thorough=not fast, **overrides)
    _print_banner(cfg, trade_date, selected)

    book = None
    if portfolio:
        from tradingagents.portfolio import load_portfolio

        book = load_portfolio(portfolio)

    from tradingagents.gold.runner import analyze_gold

    try:
        run = analyze_gold(
            trade_date,
            portfolio=book,
            config=cfg,
            analysts=selected,
            save=save,
            html=html,
        )
    except NotGoldError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    rating = run.rating or "REVIEW"
    style = _RATING_STYLE.get(rating, "white")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Rating", f"[{style}]{rating}[/{style}]")
    table.add_row("Gold", GOLD_SYMBOL)
    table.add_row("Date", run.trade_date)
    for slot, length in run.ratings_by_section().items():
        if length:
            shown = f"{length:,} chars"
        elif _SLOT_ANALYST.get(slot) not in selected:
            # An analyst the user did not ask for is not a failure.
            shown = "[dim]not requested[/dim]"
        else:
            shown = "[yellow]missing[/yellow]"
        table.add_row(f"{slot} report", shown)
    if run.report_path:
        table.add_row("Report", str(run.report_path))
    console.print(Panel(table, title="Result", border_style="yellow"))

    if show and run.decision:
        console.print(Markdown(run.decision))
    console.print(
        "[dim]Research output, not investment advice. Gold is leveraged, gaps on "
        "geopolitical events, and the model can be wrong with total confidence.[/dim]"
    )


@app.command("data")
def data(
    date: str = typer.Option(None, "--date", help="As-of date YYYY-MM-DD (default: today)"),
):
    """Print the gold data dashboards (prices, macro, flows, vol) without an LLM call."""
    from tradingagents.gold.tools import (
        _domestic_reference,
        _flows_and_positioning,
        _macro_dashboard,
        _volatility_regime,
    )

    as_of = date or _date.today().isoformat()
    from tradingagents.dataflows.config import set_config

    set_config(gold_config(output_language="Chinese"))
    for name, builder in (
        ("macro dashboard", lambda: _macro_dashboard(as_of, 365)),
        ("flows and positioning", lambda: _flows_and_positioning(as_of)),
        ("volatility regime", lambda: _volatility_regime(as_of)),
        ("domestic reference", lambda: _domestic_reference(as_of)),
    ):
        console.rule(f"[bold]{name}")
        try:
            console.print(Markdown(builder()))
        except Exception as exc:  # noqa: BLE001 — a data preview must not crash
            console.print(f"[yellow]{name} unavailable: {exc}[/yellow]")


@app.command("doctor")
def doctor():
    """Check the environment: Python deps, API keys, data access and tool health."""
    import importlib
    import os
    import platform

    from tradingagents.dataflows.config import set_config

    set_config(gold_config())
    checks: list[tuple[str, bool, str]] = []

    def check(label: str, fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append((label, ok, detail))

    def _deps():
        missing = []
        for module in ("langgraph", "langchain_core", "pandas", "yfinance", "typer", "rich"):
            try:
                importlib.import_module(module)
            except Exception:  # noqa: BLE001
                missing.append(module)
        return (not missing, "all present" if not missing else f"missing: {', '.join(missing)}")

    def _llm_key():
        from tradingagents.dataflows.config import get_config
        from tradingagents.llm_clients.api_key_env import get_api_key_env

        provider = ((get_config() or {}).get("llm_provider") or "openai").lower()
        if provider == "chatgpt":
            return True, "provider chatgpt — sign in with `tradingagents login chatgpt`"
        env_name = get_api_key_env(provider)
        if env_name is None:
            return True, f"provider {provider} — no API key needed"
        if os.getenv(env_name):
            return True, f"{provider}: {env_name} is set"
        return False, f"{provider}: {env_name} is not set (see .env.example)"

    def _prices():
        from tradingagents.gold import data as gdata

        series = gdata.closes(GOLD_SYMBOL, _date.today().isoformat(), rows=5)
        if series.empty:
            return False, "no gold rows returned"
        return True, f"GC=F last {float(series.iloc[-1]):,.2f} on {series.index[-1].date()}"

    def _llm_endpoint():
        """Probe the configured endpoint; any HTTP answer counts as reachable."""
        import requests as _requests

        from tradingagents.dataflows.config import get_config

        cfg = get_config() or {}
        provider = (cfg.get("llm_provider") or "openai").lower()
        url = _endpoint_for(provider, cfg.get("backend_url"))
        if not url:
            return True, f"provider {provider}: endpoint comes from the environment, nothing to probe"
        session = _requests.Session()
        session.trust_env = False
        try:
            status = session.get(url, timeout=8).status_code
            # Any HTTP answer proves reachability; 401/404 just mean the probe
            # sent no key or hit a path that is not an endpoint.
            detail = "reachable" if status < 400 else f"reachable (HTTP {status}, no key sent by the probe)"
            return True, f"{provider} {detail}"
        except Exception as exc:  # noqa: BLE001
            try:
                session.trust_env = True
                status = session.get(url, timeout=8).status_code
                return True, f"{provider} reachable only through the configured proxy (HTTP {status})"
            except Exception:  # noqa: BLE001
                return False, (f"{provider} unreachable from this network: "
                               f"{type(exc).__name__} — use --provider deepseek / qwen / glm")
        finally:
            session.close()

    def _fred():
        from tradingagents.gold import data as gdata

        if not os.getenv("FRED_API_KEY"):
            # Not a blocker: the yields come from the keyless Treasury curves.
            return True, ("FRED_API_KEY not set — yields fall back to the US Treasury "
                          "curves; only Fed funds and CPI need the key")
        series = gdata.fred_series("DFII10", _date.today().isoformat(), look_back_days=30)
        if series.empty:
            return False, "DFII10 returned no observations"
        return True, f"10y real yield {float(series.iloc[-1]):.2f}% on {series.index[-1].date()}"

    def _tool_smoke():
        from tradingagents.gold.tools import _volatility_regime

        text = _volatility_regime(_date.today().isoformat())
        if text.startswith("DATA_UNAVAILABLE"):
            return False, text[:160]
        return True, "volatility regime tool produced a report"

    check("python", lambda: (True, f"{platform.python_version()} on {platform.system()}"))
    check("dependencies", _deps)
    check("llm provider", _llm_key)
    check("gold prices", _prices)
    check("llm endpoint", _llm_endpoint)
    check("macro series (FRED)", _fred)
    check("gold tool smoke", _tool_smoke)

    table = Table(title="tradingagents-gold doctor", show_lines=False)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for label, ok, detail in checks:
        table.add_row(label, "[green]ok[/green]" if ok else "[yellow]warn[/yellow]", detail)
    console.print(table)
    if any(not ok for _label, ok, _detail in checks):
        console.print(
            "[yellow]Some checks warn. Prices come from Eastmoney/Sina (or Yahoo with "
            "--source auto), yields from the US Treasury, so a missing FRED key only "
            "costs Fed funds and CPI. The LLM endpoint must be reachable: if OpenAI is "
            "blocked on this network, use a domestic provider with your own key, e.g. "
            "--provider deepseek / qwen / glm.[/yellow]"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
