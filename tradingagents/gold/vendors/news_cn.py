"""Gold-relevant headlines from feeds that answer inside mainland China.

Why this exists: the framework's news tools read Yahoo Finance and Polymarket,
and on many mainland networks both are unreachable — which leaves the news
analyst with no headlines at all and a report that says so. These two Sina
endpoints need no key and answer here:

    7x24 flash   zhibo.sina.com.cn   event-speed macro flashes (one line each)
    finance roll feed.mix.sina.com.cn slower articles with a title and a link

Filtering happens here, not in the prompt: an unfiltered 50-item feed is mostly
noise (single-stock moves, sports-adjacent finance filler), and asking the model
to sift it costs tokens and invites it to invent relevance. An item is kept only
if it mentions gold, the Fed, the dollar, yields, inflation or a listed
geopolitical risk — the drivers the gold brief is organised around.

Everything degrades to an empty list; the caller renders ``DATA_UNAVAILABLE``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

FLASH_URL = "https://zhibo.sina.com.cn/api/zhibo/feed"
ROLL_URL = "https://feed.mix.sina.com.cn/api/roll/get"
TIMEOUT = 15
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn/",
}

# Matched case-insensitively against the headline. Grouped by the driver it maps
# to, so the list can be reviewed against the gold brief rather than guessed.
# Deliberately narrow: a bare "美元" matches "目标价650美元" and a bare "ETF"
# matches a bitcoin fund, and both would push a real gold headline off the list.
KEYWORDS = (
    # the metal itself
    "黄金", "金价", "金银", "白银", "贵金属", "上金所", "沪金", "金饰", "金条", "纸黄金",
    "gold", "bullion", "silver", "xauusd",
    # policy rate and the Fed
    "美联储", "鲍威尔", "fomc", "议息", "降息", "加息", "缩表", "扩表", "点阵图", "联邦基金",
    "fed", "powell", "rate cut", "rate hike",
    # discount rate and the dollar
    "美债", "国债收益率", "美债收益率", "实际利率", "美元指数", "美指", "美元兑", "美元汇率",
    "人民币汇率",
    "t-note", "treasury yield", "real yield", "dollar index", "dxy",
    # inflation and growth
    "通胀", "cpi", "pce", "非农", "失业率", "gdp", "pmi", "衰退", "滞胀",
    "inflation", "payroll", "recession",
    # risk and the official sector
    "避险", "地缘", "俄乌", "中东", "伊朗", "以色列", "关税", "制裁", "停火",
    "央行", "购金", "黄金储备", "增持黄金", "黄金etf", "etf持仓", "etf流入",
    "geopolitic", "tariff", "sanction", "central bank",
)

_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        session = requests.Session()
        session.trust_env = False  # a stale system proxy is the usual failure here
        session.headers.update(HEADERS)
        _SESSION = session
    return _SESSION


def _get(url: str, params: dict) -> dict | None:
    """JSON from ``url``, retried once without proxies; None on any failure."""
    for attempt in range(2):
        if attempt:
            time.sleep(1.0)
        try:
            response = _session().get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError):
            session = requests.Session()
            session.trust_env = False
            session.headers.update(HEADERS)
            try:
                return session.get(url, params=params, timeout=TIMEOUT).json()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Sina news (direct) failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 — a fallback must never raise
            logger.debug("Sina news failed: %s", exc)
    return None


def flash(limit: int = 80) -> list[dict]:
    """7x24 macro flashes, newest first: ``[{ts, text, source}]``."""
    payload = _get(FLASH_URL, {"page": 1, "page_size": limit, "zhibo_id": 152, "dire": "f"})
    try:
        items = payload["result"]["data"]["feed"]["list"]
    except (KeyError, TypeError):
        return []
    out = []
    for item in items:
        text = (item.get("rich_text") or "").strip()
        if not text:
            continue
        out.append({
            "ts": item.get("create_time") or "",
            "text": text,
            "source": "Sina 7x24",
            "url": item.get("docurl") or "",
        })
    return out


def roll(limit: int = 50) -> list[dict]:
    """Finance roll articles, newest first: ``[{ts, title, source, url}]``."""
    payload = _get(ROLL_URL, {"pageid": 153, "lid": 2516, "num": limit, "page": 1})
    try:
        items = payload["result"]["data"]
    except (KeyError, TypeError):
        return []
    out = []
    for item in items:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        stamp = item.get("intime") or item.get("ctime") or ""
        try:
            ts = datetime.fromtimestamp(int(stamp)).strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            ts = str(stamp)
        out.append({
            "ts": ts,
            "text": title,
            "source": item.get("media_name") or "Sina finance",
            "url": item.get("url") or "",
        })
    return out


def is_relevant(text: str) -> bool:
    """Whether a headline touches a gold driver (see ``KEYWORDS``)."""
    lowered = text.lower()
    return any(word.lower() in lowered for word in KEYWORDS)


def headlines(limit: int = 20, flash_limit: int = 80, roll_limit: int = 50) -> list[dict]:
    """Gold-relevant items from both feeds, de-duplicated and newest first.

    Each item is ``{ts, text, source, url}``. Text is clipped to 220 characters:
    a flash is one sentence already, and a long article body would crowd the
    other headlines out of the model's context.

    At most two items share a speaker/section prefix ("欧洲央行行长拉加德："), because
    a wire feed repeats one speaker's every remark and four of them push the
    actual gold stories off the list.
    """
    seen: set[str] = set()
    per_prefix: dict[str, int] = {}
    items: list[dict] = []
    for batch in (flash(flash_limit), roll(roll_limit)):
        for item in batch:
            text = item["text"]
            if not is_relevant(text):
                continue
            key = text[:60]
            if key in seen:
                continue
            prefix = text.split("：")[0].split(":")[0][:14]
            if per_prefix.get(prefix, 0) >= 2:
                continue
            seen.add(key)
            per_prefix[prefix] = per_prefix.get(prefix, 0) + 1
            items.append({**item, "text": text[:220]})
    items.sort(key=lambda row: row["ts"], reverse=True)
    return items[:limit]
