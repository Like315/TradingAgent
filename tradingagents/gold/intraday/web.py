"""``tradingagents-gold dashboard`` — a local web UI for the intraday board.

A scalper wants a screen, not a terminal scroll. This module serves one page
that re-renders the same numbers the CLI board prints — session clock, level
map, regime, setups — plus two things the CLI buries in prose:

* a **direction gauge**: every independent vote the arithmetic can cast
  (regime bias, price vs VWAP, EMA alignment, the setups themselves) laid out
  so the trader can disagree with individual votes instead of a black box;
* a **day plan**: the key levels, the if-then scenarios around them and the
  event calendar, written before the open and only re-drawn when the data says
  something changed.

No third-party server dependency: stdlib ``http.server`` with a thread pool.
Board JSON is cached briefly so several panels polling at once do not multiply
vendor requests — the feed's own cooldowns stay untouched.

    tradingagents-gold dashboard                       # http://127.0.0.1:8686
    tradingagents-gold dashboard --equity 50000 --risk 0.3
    tradingagents-gold dashboard --port 9000 --no-browser
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from tradingagents.gold.config import gold_config, gold_home
from tradingagents.gold.intraday import feed
from tradingagents.gold.intraday.board import TapeBoard, build_board

DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")

# Board cache: the page polls faster than the feed's 1-minute bars change.
BOARD_TTL_SECONDS = 15.0
# The LLM card costs ~4 model calls; one per minute is the sane ceiling.
LLM_MIN_INTERVAL_SECONDS = 60.0

# Deliverables: every LLM card is appended here so nothing a model said is
# lost when the page refreshes; daily research reports live in reports/.
CARDS_FILE = Path(gold_home()) / "logs" / "scalp_cards.jsonl"
REPORTS_DIR = Path(gold_home()) / "logs" / "reports"


# --------------------------------------------------------------------------- #
# Direction judgement — arithmetic votes, nothing hidden
# --------------------------------------------------------------------------- #


def judge_direction(data: dict[str, Any]) -> dict[str, Any]:
    """Independent votes for today's direction, in and out of the sum.

    Every vote carries its own weight and its own one-line reason, so the
    trader can discount the ones they disagree with instead of trusting a
    single blended number. ``score`` is -1 (全空) .. +1 (全多).
    """
    price = data["price"]
    regime = data["regime"]
    levels = data["levels"]
    setups = data["setups"]
    votes: list[dict[str, Any]] = []

    def vote(name: str, value: float, weight: float, reason: str) -> None:
        votes.append(
            {"name": name, "value": round(max(-1.0, min(1.0, value)), 3), "weight": weight, "reason": reason}
        )

    # 1. Regime summary bias (already -1..1 from four sub-votes).
    vote("状态倾向", regime["bias"], 0.35, f"{regime['label']}，综合倾向 {regime['bias']:+.2f}")

    # 2. Where price sits against VWAP in sigmas; ±2σ saturates the vote.
    vwap = levels.get("vwap")
    sigma = levels.get("vwap_sigma")
    if vwap and sigma:
        dist = regime.get("dist_vwap_sigma", 0.0)
        vote(
            "VWAP 位置",
            max(-1.0, min(1.0, dist / 2.0)),
            0.20,
            f"价格距 VWAP {dist:+.2f}σ（" + ("上方，偏多" if dist > 0 else "下方，偏空") + "）",
        )
    else:
        vote("VWAP 位置", 0.0, 0.20, "VWAP 不可用，本票弃权")

    # 3. EMA alignment on the 15-minute frame.
    align = regime.get("ema_align", "")
    if "多" in align:
        vote("15m 均线", 1.0, 0.15, align)
    elif "空" in align:
        vote("15m 均线", -1.0, 0.15, align)
    else:
        vote("15m 均线", 0.0, 0.15, align or "均线纠缠")

    # 4. The setups the gates actually passed — the strongest single tell,
    #    because they already carry regime + odds + cost.
    if setups:
        longs = sum(1 for s in setups if s["side"] == "long")
        shorts = len(setups) - longs
        side = 1.0 if longs > shorts else (-1.0 if shorts > longs else 0.0)
        primary = setups[0]
        vote(
            "形态票",
            side,
            0.20,
            f"最佳形态「{primary['name']}」为{'做多' if primary['side'] == 'long' else '做空'}"
            f"（评分 {primary['score']:.2f}，{len(setups)} 个形态：{longs} 多 / {shorts} 空）",
        )
    else:
        vote("形态票", 0.0, 0.20, "无形态通过闸门，本票弃权")

    # 5. Today's drive so far: price vs the session open.
    day_open = levels.get("day_open")
    if day_open:
        atr = levels.get("atr15") or 1.0
        drive = (price - day_open) / (atr * 3.0)
        vote(
            "日内动能",
            max(-1.0, min(1.0, drive)),
            0.10,
            f"价格较开盘 {'高' if price >= day_open else '低'} {abs(price - day_open):.2f} 美元",
        )
    else:
        vote("日内动能", 0.0, 0.10, "当日开盘价不可用，本票弃权")

    total_weight = sum(v["weight"] for v in votes)
    score = sum(v["value"] * v["weight"] for v in votes) / (total_weight or 1.0)
    score = round(max(-1.0, min(1.0, score)), 3)

    if score >= 0.35:
        label, stance = "偏多", "多头环境优先找回调做多"
    elif score <= -0.35:
        label, stance = "偏空", "空头环境优先找反弹做空"
    else:
        label, stance = "中性", "方向不明，优先区间思路或观望"
    if not data["clock"]["tradable"]:
        stance = "当前时段不可交易，判断仅供参考"
    return {
        "score": score,
        "label": label,
        "stance": stance,
        "votes": votes,
        "chaos": regime["code"] in ("chaos", "dead"),
    }


# --------------------------------------------------------------------------- #
# Day plan — if-then scenarios around the levels that actually matter
# --------------------------------------------------------------------------- #


def build_day_plan(data: dict[str, Any]) -> dict[str, Any]:
    """The pre-open style plan, recomputed on every refresh.

    Scenarios are derived from the level map, not invented: each one names the
    level it keys off, the confirmation to wait for, and the target beyond it.
    If the gates already passed setups, they are quoted verbatim — a plan that
    disagrees with the board is worse than no plan.
    """
    levels = data["levels"]
    regime = data["regime"]
    clock = data["clock"]
    price = data["price"]
    map_rows = levels["map"]

    above = [lv for lv in map_rows if lv["price"] > price]
    below = [lv for lv in map_rows if lv["price"] <= price]
    nearest_above = above[0] if above else None
    nearest_below = below[-1] if below else None
    far_above = above[1] if len(above) > 1 else None
    far_below = below[-2] if len(below) > 1 else None

    key_levels: list[dict[str, Any]] = []
    for lv in map_rows[:8]:
        key_levels.append(
            {"price": lv["price"], "name": lv["name"], "distance": lv.get("distance"), "atr": lv.get("atr"), "note": lv.get("note") or ""}
        )
    for label, value in (("当日 VWAP", levels.get("vwap")), ("亚洲区间上沿", levels.get("asian_high")), ("亚洲区间下沿", levels.get("asian_low"))):
        if value:
            key_levels.append({"price": value, "name": label, "distance": round(value - price, 2), "atr": None, "note": ""})
    key_levels.sort(key=lambda lv: lv["price"], reverse=True)

    scenarios: list[dict[str, str]] = []
    setups = data["setups"]
    if setups:
        for s in setups[:3]:
            scenarios.append(
                {
                    "side": s["side"],
                    "title": f"已就绪 · {s['name']}",
                    "trigger": s["trigger"],
                    "entry": f"{s['entry'][0]:,.2f} - {s['entry'][1]:,.2f}",
                    "stop": f"{s['stop']:,.2f}",
                    "targets": f"T1 {s['targets'][0]:,.2f}（{s['rr'][0]:.1f}R） / T2 {s['targets'][1]:,.2f}",
                    "hold": f"{s['hold_minutes']} 分钟内",
                    "invalidation": s["invalidation"],
                }
            )
    else:
        for reject in data.get("no_trade_reasons", [])[:3]:
            scenarios.append({"side": "none", "title": "当前不做", "trigger": reject, "entry": "-", "stop": "-", "targets": "-", "hold": "-", "invalidation": "-"})
        if nearest_above:
            atr_pad = levels.get("atr15") or levels.get("atr5") or 2.0
            far_above = far_above or {
                "name": f"{nearest_above['name']}上方 {2.0:.0f}×ATR",
                "price": nearest_above["price"] + 2.0 * atr_pad,
            }
            scenarios.append(
                {
                    "side": "long",
                    "title": "上破情景",
                    "trigger": f"站稳 {nearest_above['name']} {nearest_above['price']:,.2f} 之上（回踩不破再进，不追第一根）",
                    "entry": f"回踩 {nearest_above['price']:,.2f} 附近确认支撑",
                    "stop": f"跌回 {nearest_above['price'] - (levels.get('atr5') or 1.0):,.2f} 之下",
                    "targets": f"看 {far_above['name']} {far_above['price']:,.2f}",
                    "hold": "视目标距离，剥头皮不过夜",
                    "invalidation": f"假突破：冲高后 15 分钟内跌回 {nearest_above['price']:,.2f} 下方",
                }
            )
        if nearest_below:
            atr_pad = levels.get("atr15") or levels.get("atr5") or 2.0
            far_below = far_below or {
                "name": f"{nearest_below['name']}下方 {2.0:.0f}×ATR",
                "price": nearest_below["price"] - 2.0 * atr_pad,
            }
            scenarios.append(
                {
                    "side": "short",
                    "title": "下破情景",
                    "trigger": f"跌破 {nearest_below['name']} {nearest_below['price']:,.2f} 且反抽不收复",
                    "entry": f"反抽 {nearest_below['price']:,.2f} 附近承压",
                    "stop": f"收复 {nearest_below['price'] + (levels.get('atr5') or 1.0):,.2f} 之上",
                    "targets": f"看 {far_below['name']} {far_below['price']:,.2f}",
                    "hold": "视目标距离，剥头皮不过夜",
                    "invalidation": f"假突破：下探后 15 分钟内收回 {nearest_below['price']:,.2f} 上方",
                }
            )
        if not scenarios:
            scenarios.append(
                {
                    "side": "none",
                    "title": "观望",
                    "trigger": "价位地图过薄，等亚洲区间成型或伦敦开盘后再定",
                    "entry": "-",
                    "stop": "-",
                    "targets": "-",
                    "hold": "-",
                    "invalidation": "-",
                }
            )

    events = []
    for mark in clock.get("next_marks", [])[:5]:
        events.append({"in_minutes": mark["in_minutes"], "label": mark["label"], "at": mark["at"]})
    if clock.get("blackout"):
        bo = clock["blackout"]
        events.insert(0, {"in_minutes": bo["minutes"], "label": ("【事件窗口】" if bo["active"] else "") + bo["label"], "at": ""})

    asian = levels.get("asian_high") and levels.get("asian_low")
    plan_notes = [
        f"今日波动预算：20 日均幅对应的当日已走 {regime['rv_1h_pct']:.0f}%（1h 实际波动），ATR 扩张 {regime['atr_expansion']:.2f}x"
    ]
    if asian:
        plan_notes.append(
            f"亚洲区间 {levels['asian_low']:,.2f}-{levels['asian_high']:,.2f}"
            + ("已成型，伦敦盘通常在此区间外选方向" if levels.get("asian_complete") else "尚未成型，先按区间内高抛低吸对待")
        )
    plan_notes.append("纪律红线：单笔风险不超过设定值；连亏 3 笔停手；当日 -1.5% 停止交易；数据延迟超过 10 分钟不下单。")

    return {
        "key_levels": key_levels,
        "scenarios": scenarios,
        "events": events,
        "notes": plan_notes,
        "asian_complete": bool(levels.get("asian_complete")),
        "regime_label": regime["label"],
        "direction_label": regime["bias"],
    }


def enrich(board: TapeBoard, *, no_trade_reasons: list[str] | None = None) -> dict[str, Any]:
    """Board dict + direction votes + day plan, the full page payload."""
    data = board.to_dict()
    data["no_trade_reasons"] = list(board.no_trade_reasons)
    data["freshness_blocker"] = board.freshness_blocker
    data["clock"]["grade_note"] = board.clock.grade_note
    data["clock"]["is_open"] = board.clock.is_open
    data["source"] = board.source
    data["stale"] = board.stale
    data["bar_age_minutes"] = board.bar_age_minutes
    data["has_volume"] = board.has_volume
    data["feed_note"] = board.feed_note
    data["direction"] = judge_direction(data)
    data["plan"] = build_day_plan(data)
    return data


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Deliverables: card history and daily research reports
# --------------------------------------------------------------------------- #


def _append_card(result: dict[str, Any]) -> None:
    """One JSON line per LLM card, newest last. A card the page lost on
    refresh is a card the trader cannot show when reviewing the day."""
    try:
        CARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        record = {"at": datetime.now().isoformat(timespec="seconds"), **result}
        with CARDS_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass  # history is a convenience; a failed append must not kill the card


def card_history(limit: int = 30) -> list[dict[str, Any]]:
    if not CARDS_FILE.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in CARDS_FILE.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out[::-1]


def report_rating(report_md: Path) -> str:
    try:
        for line in report_md.read_text(encoding="utf-8", errors="replace").splitlines()[:8]:
            if line.startswith("- Rating:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def list_reports() -> list[dict[str, Any]]:
    """Daily multi-agent research runs, newest first."""
    if not REPORTS_DIR.exists():
        return []
    runs = []
    for run_dir in sorted(REPORTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        complete = run_dir / "complete_report.md"
        runs.append(
            {
                "name": run_dir.name,
                "when": run_dir.name.replace("GC=F_", "").replace("_", " "),
                "rating": report_rating(complete) if complete.exists() else "",
                "has_complete": complete.exists(),
            }
        )
    return runs


def read_report(name: str, filename: str) -> dict[str, Any]:
    """Serve one file from one run directory. Both parts are validated —
    a name like ``..\\..`` must never escape the reports folder."""
    run_dir = (REPORTS_DIR / name).resolve()
    if REPORTS_DIR.resolve() not in run_dir.parents or not run_dir.is_dir():
        return {"error": "报告不存在"}
    target = (run_dir / filename).resolve()
    if run_dir not in target.parents or not target.is_file():
        return {"error": "文件不存在"}
    if target.suffix.lower() not in (".md", ".html", ".txt"):
        return {"error": "只支持阅读 md/html/txt"}
    return {"name": name, "file": filename, "content": target.read_text(encoding="utf-8", errors="replace")}


class DashboardState:
    """Shared, lock-guarded state: one board build, many readers."""

    def __init__(self, equity: float, risk_pct: float) -> None:
        self.equity = equity
        self.risk_pct = risk_pct
        self._lock = threading.Lock()
        self._board: TapeBoard | None = None
        self._built_at = 0.0
        self._error: str = ""
        self._llm_lock = threading.Lock()
        self._llm_at = 0.0

    # -- board ------------------------------------------------------------ #

    def get_payload(self, equity: float | None = None, risk: float | None = None) -> dict[str, Any]:
        eq = equity or self.equity
        rk = risk or self.risk_pct
        with self._lock:
            now = time.time()
            if self._board is None or now - self._built_at > BOARD_TTL_SECONDS or self._board.equity != eq or self._board.risk_pct != rk:
                try:
                    # Not fresh=True: TTL caches upstream already keep this
                    # current, and forcing refetches is what earns rate limits.
                    self._board = build_board(equity=eq, risk_pct=rk)
                    self._error = ""
                except feed.FeedUnavailable as exc:
                    self._error = str(exc)
                except Exception as exc:  # noqa: BLE001 — the page must say why it is blank
                    self._error = f"{type(exc).__name__}: {exc}"
                self._built_at = now
            payload: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds")}
            if self._error:
                payload["error"] = self._error
            if self._board is not None:
                payload["board"] = enrich(self._board)
            return payload

    # -- llm card ---------------------------------------------------------- #

    def run_llm(self, force: bool = False) -> dict[str, Any]:
        from tradingagents.gold.intraday.desk import analyze_scalp

        with self._llm_lock:
            wait = LLM_MIN_INTERVAL_SECONDS - (time.time() - self._llm_at)
            if wait > 0 and not force:
                return {"error": f"距上次模型研判还有 {wait:.0f} 秒（限流：每分钟最多一次），勾选“强制”可立即重跑"}
            board = self._board
            if board is None:
                payload = self.get_payload()
                if self._board is None:
                    return {"error": self._error or "行情不可用，无法研判"}
                board = self._board
            self._llm_at = time.time()
            call = analyze_scalp(board, config=gold_config(), force=force)
            result = {
                "verdict": call.verdict,
                "side": call.side,
                "setup": call.setup,
                "entry_low": call.entry_low,
                "entry_high": call.entry_high,
                "stop": call.stop,
                "t1": call.t1,
                "t2": call.t2,
                "rr": call.rr,
                "hold_minutes": call.hold_minutes,
                "risk_pct": call.risk_pct,
                "why": call.why,
                "kill": call.kill,
                "tape_read": call.tape_read,
                "catalyst_read": call.catalyst_read,
                "risk_verdict": call.risk_verdict,
                "risk_reason": call.risk_reason,
                "checks": [{"label": label, "message": message, "ok": ok} for label, message, ok in call.checks],
                "elapsed": round(call.elapsed, 1),
                "price": board.price,
            }
            _append_card(result)
            return result


def make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 — silence per-request noise
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = DASHBOARD_HTML.read_bytes()
                self._send(200, body, "text/html; charset=utf-8")
            elif parsed.path == "/api/board":
                qs = parse_qs(parsed.query)
                equity = float(qs["equity"][0]) if qs.get("equity") else None
                risk = float(qs["risk"][0]) if qs.get("risk") else None
                payload = state.get_payload(equity=equity, risk=risk)
                self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            elif parsed.path == "/api/llm":
                qs = parse_qs(parsed.query)
                force = qs.get("force", ["0"])[0] in ("1", "true")
                self._send(200, json.dumps(state.run_llm(force=force), ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            elif parsed.path == "/api/cards":
                limit = int(parse_qs(parsed.query).get("limit", ["30"])[0])
                self._send(200, json.dumps({"cards": card_history(limit)}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            elif parsed.path == "/api/reports":
                self._send(200, json.dumps({"reports": list_reports()}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            elif parsed.path == "/api/report":
                qs = parse_qs(parsed.query)
                payload = read_report(qs.get("name", [""])[0], qs.get("file", ["complete_report.md"])[0])
                self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            elif parsed.path == "/api/health":
                self._send(200, b'{"ok": true}', "application/json")
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

    return Handler


def serve(
    *,
    equity: float = 100_000.0,
    risk_pct: float = 0.5,
    host: str = "127.0.0.1",
    port: int = 8686,
    open_browser: bool = True,
) -> None:
    state = DashboardState(equity=equity, risk_pct=risk_pct)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    url = f"http://{host}:{port}"
    print(f"黄金日内看板已启动：{url}  （Ctrl-C 退出）")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    serve()
