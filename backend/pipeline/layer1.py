"""Layer 1: LLM batch analysis — 50 articles packed into 1 API call.

Strategy:
1. Local keyword extraction: for long descriptions (>500 chars), extract only
   sentences mentioning the company (ticker, name, CEO, products, etc.)
2. Pack 50 articles into a single prompt → 1 API call
3. Get back a compact JSON array — English only (no Chinese, save output tokens)
"""

import json
import re
from typing import List, Dict, Any

import anthropic

from backend.config import settings
from backend.database import get_conn
from backend.llm import get_llm_client_for

# Model configuration - defaults for SiliconFlow
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1"
MODEL = "claude-haiku-4-5-20251001"  # Anthropic Batch API model
BATCH_SIZE = 20  # articles per API call
MAX_OUTPUT_TOKENS = 4096  # enough for 50 articles (~70 tokens each)

# Comprehensive keyword mappings for extraction
# ticker, company name, short name, CEO, key products, subsidiaries
TICKER_KEYWORDS: Dict[str, List[str]] = {
    "BABA": ["alibaba", "ali baba", "baba", "daniel zhang", "joe tsai",
             "taobao", "tmall", "alipay", "ant group", "alicloud",
             "aliyun", "cainiao", "lazada", "ele.me"],
    "AAPL": ["apple", "aapl", "tim cook", "iphone", "ipad", "macbook",
             "apple watch", "vision pro", "app store", "ios", "macos"],
    "TSLA": ["tesla", "tsla", "elon musk", "model 3", "model y",
             "model s", "model x", "cybertruck", "gigafactory",
             "supercharger", "autopilot", "full self-driving", "fsd"],
    "NVDA": ["nvidia", "nvda", "jensen huang", "geforce", "rtx",
             "cuda", "a100", "h100", "h200", "b100", "b200",
             "dgx", "drive", "omniverse", "tensorrt"],
    "GLD": ["spdr gold", "gld", "gold trust", "gold etf", "gold shares"],
    "MSFT": ["microsoft", "msft", "satya nadella", "windows", "azure",
             "office 365", "xbox", "linkedin", "github", "copilot"],
    "GOOGL": ["google", "alphabet", "googl", "goog", "sundar pichai",
              "youtube", "waymo", "deepmind", "gemini", "android",
              "google cloud", "pixel"],
    "AMZN": ["amazon", "amzn", "andy jassy", "aws", "prime",
             "alexa", "kindle", "whole foods"],
    "META": ["meta platforms", "meta", "facebook", "zuckerberg",
             "instagram", "whatsapp", "threads", "oculus", "quest"],
    "AMD":  ["amd", "advanced micro", "lisa su", "radeon", "ryzen",
             "epyc", "xilinx", "instinct"],
}

# Minimum description length to trigger extraction (shorter ones sent in full)
EXTRACT_THRESHOLD = 500


def _get_keywords(symbol: str) -> List[str]:
    """Get all keywords for a ticker. Falls back to just the symbol."""
    kws = [symbol.lower()]
    kws.extend(TICKER_KEYWORDS.get(symbol, []))
    return kws


def _extract_relevant_text(description: str, symbol: str) -> str:
    """For long descriptions, extract only sentences mentioning the company.

    Short descriptions (<500 chars) are returned in full.
    Long descriptions are filtered to company-relevant sentences + 1 neighbor.
    """
    if not description:
        return ""

    desc = description.strip()
    if len(desc) < EXTRACT_THRESHOLD:
        return desc

    keywords = _get_keywords(symbol)
    sentences = re.split(r'(?<=[.!?])\s+', desc)

    # Find sentences with keyword matches
    relevant: set = set()
    for i, sent in enumerate(sentences):
        lower = sent.lower()
        if any(kw in lower for kw in keywords):
            # Keep this sentence + 1 before + 1 after for context
            for j in range(max(0, i - 1), min(len(sentences), i + 2)):
                relevant.add(j)

    if not relevant:
        # No keyword match — just return first 2 sentences
        return " ".join(sentences[:2])

    return " ".join(sentences[i] for i in sorted(relevant))


def _build_batch_prompt(symbol: str, articles: List[Dict[str, Any]]) -> str:
    """Build a single prompt containing up to 50 articles."""
    lines = []
    for i, art in enumerate(articles):
        extract = _extract_relevant_text(art.get("description") or "", symbol)
        lines.append(f"[{i}] {art['title']}")
        if extract:
            lines.append(f"  > {extract}")

    return f"""Rate these {len(articles)} articles for {symbol}. Return JSON array only. Do not include markdown, explanations, or thinking text.

{chr(10).join(lines)}

Format: [{{"i":0,"r":"y"|"n","s":"+"|"-"|"0","score":0,"c":"other","e":"summary","u":"up reason","d":"down reason"}}]
r: "y" = article specifically discusses {symbol}, "n" = irrelevant/brief mention
s: "+" positive, "-" negative, "0" neutral
score: numerical sentiment -2 (very negative) -1 (mildly negative) 0 (neutral) +1 (mildly positive) +2 (very positive). Use 0 if irrelevant.
c: event category — "earnings" (financial results/guidance), "product" (product launch/update), "regulatory" (lawsuit/regulation/policy), "macro" (macroeconomic/interest rates/market), "analyst" (analyst rating/price target), "management" (leadership/strategy/restructuring), "industry" (industry trend/competition), "other" (default)
e: 10-word summary of what happened (empty if irrelevant)
u: why this could push {symbol} stock UP, e.g. "strong earnings beat expectations" (empty if none or irrelevant)
d: why this could push {symbol} stock DOWN, e.g. "antitrust lawsuit threatens App Store revenue" (empty if none or irrelevant)
JSON:"""


def _parse_batch_response(text: str) -> List[Dict[str, Any]]:
    """Extract the result array from common LLM response shapes."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty LLM response")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start < 0 or end <= start:
            raise ValueError(f"no JSON array found; response starts with: {raw[:160]!r}")
        parsed = json.loads(raw[start:end])

    if isinstance(parsed, dict):
        for key in ("results", "items", "articles", "data"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array, got {type(parsed).__name__}")
    return parsed


def get_pending_articles(symbol: str, limit: int = 10000) -> List[Dict[str, Any]]:
    """Get articles that passed Layer 0 but haven't been processed by Layer 1."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT nr.id, nr.title, nr.description
           FROM news_raw nr
           JOIN layer0_results l0 ON nr.id = l0.news_id AND l0.symbol = ?
           WHERE l0.passed = 1
           AND nr.id NOT IN (
               SELECT news_id FROM layer1_results WHERE symbol = ?
           )
           LIMIT ?""",
        (symbol, symbol, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def process_batch_group(
    symbol: str, articles: List[Dict[str, Any]]
) -> Dict[str, int]:
    """Process a group of up to 50 articles in a single API call."""
    conn = get_conn()

    stats = {"processed": 0, "relevant": 0, "irrelevant": 0, "errors": 0}

    prompt = _build_batch_prompt(symbol, articles)

    try:
        # Use unified LLM client
        llm_client = _get_layer1_llm_client()
        text = llm_client.chat_simple(prompt=prompt, max_tokens=MAX_OUTPUT_TOKENS)

        results = _parse_batch_response(text)

        for item in results:
            idx = item.get("i")
            if idx is None or idx >= len(articles):
                stats["errors"] += 1
                continue

            art = articles[idx]
            is_relevant = item.get("r") in ("y", "relevant")
            relevance = "relevant" if is_relevant else "irrelevant"
            raw_s = item.get("s", "0")
            sentiment = {"+": "positive", "-": "negative"}.get(raw_s, "neutral")

            # Parse new fields with backward-compatible defaults
            raw_score = item.get("score")
            if raw_score is not None:
                try:
                    sentiment_score = max(-2.0, min(2.0, float(raw_score)))
                except (ValueError, TypeError):
                    sentiment_score = None
            else:
                sentiment_score = None  # fallback: computed from sentiment in features.py

            event_category = str(item.get("c", "other")) if item.get("c") else "other"
            VALID_CATEGORIES = {"earnings", "product", "regulatory", "macro",
                                "analyst", "management", "industry", "other"}
            if event_category not in VALID_CATEGORIES:
                event_category = "other"

            conn.execute(
                """INSERT OR REPLACE INTO layer1_results
                   (news_id, symbol, relevance, key_discussion, sentiment,
                    sentiment_score, event_category,
                    reason_growth, reason_decrease)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    art["id"],
                    symbol,
                    relevance,
                    item.get("e", ""),
                    sentiment,
                    sentiment_score,
                    event_category,
                    item.get("u", ""),
                    item.get("d", ""),
                ),
            )
            stats["processed"] += 1
            if is_relevant:
                stats["relevant"] += 1
            else:
                stats["irrelevant"] += 1

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        stats["errors"] = len(articles)
        print(f"Batch error for {symbol}: {e}")
    except ValueError as e:
        # LLM client initialization error
        stats["errors"] = len(articles)
        print(f"LLM client error for {symbol}: {e}")

    conn.commit()
    conn.close()
    return stats


def _get_layer1_llm_client():
    """Create the LLM client configured for Layer 1 batch analysis."""
    provider = settings.layer1_llm_provider or settings.llm_provider
    model = settings.layer1_llm_model or settings.llm_model
    return get_llm_client_for(
        provider=provider,
        model=model,
        api_key=settings.layer1_llm_api_key or None,
        base_url=settings.layer1_llm_base_url or None,
    )


def run_layer1(symbol: str, max_articles: int = 10000) -> Dict[str, Any]:
    """Run Layer 1 on all pending articles for a symbol.

    Processes in groups of 50 articles per API call.
    """
    articles = get_pending_articles(symbol, limit=max_articles)
    if not articles:
        return {"status": "no_pending", "total": 0}

    total_stats = {
        "total": len(articles), "processed": 0, "relevant": 0,
        "irrelevant": 0, "errors": 0, "api_calls": 0,
    }

    for i in range(0, len(articles), BATCH_SIZE):
        chunk = articles[i : i + BATCH_SIZE]
        stats = process_batch_group(symbol, chunk)

        total_stats["processed"] += stats["processed"]
        total_stats["relevant"] += stats["relevant"]
        total_stats["irrelevant"] += stats["irrelevant"]
        total_stats["errors"] += stats["errors"]
        total_stats["api_calls"] += 1

        print(f"  [{symbol}] Batch {total_stats['api_calls']}: "
              f"{stats['processed']}/{len(chunk)} ok, {stats['relevant']} relevant")

    return total_stats


# === Batch API support (for very large jobs, 50% cheaper) ===

def submit_batch_api(symbol: str, articles: List[Dict[str, Any]]) -> str:
    """Submit to Anthropic Batch API for async processing."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    requests = []
    for i in range(0, len(articles), BATCH_SIZE):
        chunk = articles[i : i + BATCH_SIZE]
        chunk_ids = "|".join(a["id"] for a in chunk)
        prompt = _build_batch_prompt(symbol, chunk)

        requests.append(
            {
                "custom_id": f"{symbol}|{i}|{chunk_ids}",
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )

    batch = client.messages.batches.create(requests=requests)

    conn = get_conn()
    conn.execute(
        """INSERT INTO batch_jobs (batch_id, symbol, status, total, created_at)
           VALUES (?, ?, ?, ?, datetime('now'))""",
        (batch.id, symbol, batch.processing_status, len(articles)),
    )
    conn.commit()
    conn.close()
    return batch.id


def check_batch_status(batch_id: str) -> Dict[str, Any]:
    """Check the status of a batch job."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    batch = client.messages.batches.retrieve(batch_id)

    conn = get_conn()
    conn.execute(
        "UPDATE batch_jobs SET status = ? WHERE batch_id = ?",
        (batch.processing_status, batch_id),
    )
    conn.commit()
    conn.close()

    return {
        "batch_id": batch.id,
        "status": batch.processing_status,
        "request_counts": {
            "processing": batch.request_counts.processing,
            "succeeded": batch.request_counts.succeeded,
            "errored": batch.request_counts.errored,
            "canceled": batch.request_counts.canceled,
            "expired": batch.request_counts.expired,
        },
    }


def collect_batch_results(batch_id: str) -> Dict[str, int]:
    """Collect results from a completed batch API job."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    conn = get_conn()

    stats = {"processed": 0, "relevant": 0, "irrelevant": 0, "errors": 0}

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        parts = custom_id.split("|", 2)
        if len(parts) < 3:
            stats["errors"] += 1
            continue

        symbol = parts[0]
        article_ids = parts[2].split("|")

        if result.result.type != "succeeded":
            stats["errors"] += len(article_ids)
            continue

        message = result.result.message
        text = message.content[0].text if message.content else "[]"

        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start < 0 or end <= start:
                stats["errors"] += len(article_ids)
                continue

            items = json.loads(text[start:end])

            for item in items:
                idx = item.get("i")
                if idx is None or idx >= len(article_ids):
                    stats["errors"] += 1
                    continue

                is_relevant = item.get("r") in ("y", "relevant")
                relevance = "relevant" if is_relevant else "irrelevant"
                raw_s = item.get("s", "0")
                sentiment = {"+": "positive", "-": "negative"}.get(raw_s, "neutral")

                raw_score = item.get("score")
                if raw_score is not None:
                    try:
                        sentiment_score = max(-2.0, min(2.0, float(raw_score)))
                    except (ValueError, TypeError):
                        sentiment_score = None
                else:
                    sentiment_score = None

                event_category = str(item.get("c", "other")) if item.get("c") else "other"
                if event_category not in {"earnings", "product", "regulatory", "macro",
                                           "analyst", "management", "industry", "other"}:
                    event_category = "other"

                conn.execute(
                    """INSERT OR REPLACE INTO layer1_results
                       (news_id, symbol, relevance, key_discussion, sentiment,
                        sentiment_score, event_category,
                        reason_growth, reason_decrease)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        article_ids[idx],
                        symbol,
                        relevance,
                        item.get("e", ""),
                        sentiment,
                        sentiment_score,
                        event_category,
                        item.get("u", ""),
                        item.get("d", ""),
                    ),
                )
                stats["processed"] += 1
                if is_relevant:
                    stats["relevant"] += 1
                else:
                    stats["irrelevant"] += 1

        except (json.JSONDecodeError, KeyError):
            stats["errors"] += len(article_ids)

    conn.execute(
        "UPDATE batch_jobs SET status = 'collected', finished_at = datetime('now') WHERE batch_id = ?",
        (batch_id,),
    )
    conn.commit()
    conn.close()
    return stats
