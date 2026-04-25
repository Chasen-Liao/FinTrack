"""Process Layer 1 analysis for top N tickers.

Uses the configured LLM provider by default. Anthropic Batch API is used only
when LLM_PROVIDER=anthropic and the base URL is the official Anthropic API.

Usage: python -m backend.batch_submit [--top 50] [--max-per-symbol 200]
"""

import json
import sys
from typing import List, Dict, Any

from backend.config import settings
from backend.database import get_conn
from backend.pipeline.layer1 import (
    get_pending_articles,
    _build_batch_prompt,
    BATCH_SIZE,
    MODEL,
    MAX_OUTPUT_TOKENS,
    run_layer1,
)


def get_top_tickers(n: int = 50) -> List[Dict[str, Any]]:
    """Get top N tickers by Layer 0 passed count, with pending articles."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT l0.symbol, t.name,
               sum(case when l0.passed=1 then 1 else 0 end) as passed
        FROM layer0_results l0
        JOIN tickers t ON l0.symbol = t.symbol
        GROUP BY l0.symbol
        ORDER BY passed DESC
        LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def build_batch_requests(
    symbols: List[str],
) -> tuple[list, dict]:
    """Build all batch requests for given symbols.

    Returns (requests_list, mapping_dict) where mapping_dict maps
    custom_id -> (symbol, [article_ids]).
    """
    all_requests = []
    mapping = {}  # custom_id -> (symbol, article_ids_list)

    for symbol in symbols:
        articles = get_pending_articles(symbol)
        if not articles:
            print(f"  {symbol}: no pending articles, skip")
            continue

        for chunk_idx in range(0, len(articles), BATCH_SIZE):
            chunk = articles[chunk_idx:chunk_idx + BATCH_SIZE]
            custom_id = f"{symbol}_{chunk_idx:05d}"

            prompt = _build_batch_prompt(symbol, chunk)
            article_ids = [a["id"] for a in chunk]

            all_requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })
            mapping[custom_id] = (symbol, article_ids)

        print(f"  {symbol}: {len(articles)} articles -> {len(range(0, len(articles), BATCH_SIZE))} requests")

    return all_requests, mapping


def submit_batch(requests_list: list, mapping: dict) -> str:
    """Submit to Anthropic Batch API and save mapping to database."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    print(f"\nSubmitting {len(requests_list)} requests to Batch API...")
    batch = client.messages.batches.create(requests=requests_list)
    batch_id = batch.id
    print(f"Batch ID: {batch_id}")
    print(f"Status: {batch.processing_status}")

    # Save batch job
    conn = get_conn()
    total_articles = sum(len(v[1]) for v in mapping.values())
    conn.execute(
        """INSERT OR REPLACE INTO batch_jobs
           (batch_id, symbol, status, total, created_at)
           VALUES (?, ?, ?, ?, datetime('now'))""",
        (batch_id, "multi", batch.processing_status, total_articles),
    )

    # Save request mapping
    for custom_id, (symbol, article_ids) in mapping.items():
        conn.execute(
            """INSERT OR REPLACE INTO batch_request_map
               (batch_id, custom_id, symbol, article_ids)
               VALUES (?, ?, ?, ?)""",
            (batch_id, custom_id, symbol, json.dumps(article_ids)),
        )

    conn.commit()
    conn.close()

    return batch_id


def run_configured_provider(symbols: List[str], max_per_symbol: int | None = None) -> Dict[str, int]:
    """Run Layer 1 synchronously through the configured LLM provider."""
    totals = {
        "symbols": 0,
        "total": 0,
        "processed": 0,
        "relevant": 0,
        "irrelevant": 0,
        "errors": 0,
        "api_calls": 0,
    }

    for symbol in symbols:
        pending = get_pending_articles(symbol, limit=max_per_symbol or 10000)
        if not pending:
            print(f"  {symbol}: no pending articles, skip")
            continue

        print(f"  {symbol}: {len(pending)} pending articles")
        stats = run_layer1(symbol, max_articles=len(pending))

        totals["symbols"] += 1
        for key in ("total", "processed", "relevant", "irrelevant", "errors", "api_calls"):
            totals[key] += int(stats.get(key, 0) or 0)

    return totals


def get_layer1_provider() -> str:
    return (settings.layer1_llm_provider or settings.llm_provider or "siliconflow").lower()


def get_layer1_base_url() -> str:
    return (settings.layer1_llm_base_url or settings.llm_base_url or "").strip().rstrip("/")


def should_use_anthropic_batch(provider: str) -> bool:
    """Only the official Anthropic API supports the Batch API path."""
    if provider != "anthropic":
        return False
    base_url = get_layer1_base_url()
    return not base_url or base_url == "https://api.anthropic.com"


def main():
    top_n = 50
    max_per_symbol = None
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--top" and i + 1 < len(sys.argv):
                top_n = int(sys.argv[i + 1])
            elif arg == "--max-per-symbol" and i + 1 < len(sys.argv):
                max_per_symbol = int(sys.argv[i + 1])

    provider = get_layer1_provider()
    use_batch = should_use_anthropic_batch(provider)
    if use_batch:
        print(f"=== Layer 1 Anthropic Batch API Submission (top {top_n} tickers) ===\n")
    else:
        print(f"=== Layer 1 LLM Processing via {provider} (top {top_n} tickers) ===\n")

    # Get top tickers
    tickers = get_top_tickers(top_n)
    symbols = [t["symbol"] for t in tickers]

    total_pending = 0
    for t in tickers:
        total_pending += t["passed"]
    print(f"Top {len(tickers)} tickers, ~{total_pending} Layer0-passed articles")
    print(f"(Already processed by Layer1 will be excluded)\n")

    if not use_batch:
        print("Processing pending articles with configured LLM provider...")
        if max_per_symbol is not None:
            print(f"Limit: up to {max_per_symbol} articles per symbol")
        stats = run_configured_provider(symbols, max_per_symbol=max_per_symbol)
        print(f"\n=== Results ===")
        print(f"Symbols processed: {stats['symbols']}")
        print(f"Total articles: {stats['total']:,}")
        print(f"Processed: {stats['processed']:,}")
        print(f"Relevant: {stats['relevant']:,}")
        print(f"Irrelevant: {stats['irrelevant']:,}")
        print(f"Errors: {stats['errors']:,}")
        print(f"API calls: {stats['api_calls']:,}")
        return

    # Build Anthropic Batch API requests
    print("Building batch requests...")
    requests_list, mapping = build_batch_requests(symbols)

    if not requests_list:
        print("No pending articles to process!")
        return

    total_articles = sum(len(v[1]) for v in mapping.values())

    # Cost estimate
    est_input_tokens = total_articles * 300
    est_output_tokens = total_articles * 80
    est_cost = (est_input_tokens / 1_000_000 * 0.5) + (est_output_tokens / 1_000_000 * 2.5)

    print(f"\n=== Summary ===")
    print(f"Tickers: {len(symbols)}")
    print(f"Total articles: {total_articles:,}")
    print(f"Batch requests: {len(requests_list)}")
    print(f"Estimated cost: ~${est_cost:.2f} (Batch API pricing)")
    print()

    # Submit
    batch_id = submit_batch(requests_list, mapping)

    print(f"\nBatch submitted! ID: {batch_id}")
    print(f"Check status: python -m backend.batch_collect {batch_id}")


if __name__ == "__main__":
    main()
