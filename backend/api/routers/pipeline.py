import logging

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

from backend.database import get_conn
from backend.polygon.client import fetch_ohlc, fetch_news
from backend.pipeline.layer0 import run_layer0
from backend.pipeline.layer1 import (
    get_pending_articles,
    run_layer1,
    check_batch_status,
    collect_batch_results,
    submit_batch_api,
)
from backend.pipeline.alignment import align_news_for_symbol

import json

router = APIRouter()


class FetchRequest(BaseModel):
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None


class ProcessRequest(BaseModel):
    symbol: str
    batch_size: int = 1000
    mode: Literal["sync", "batch"] = "batch"
    include_fetch: bool = False
    start: Optional[str] = None
    end: Optional[str] = None


@router.post("/fetch")
def trigger_fetch(req: FetchRequest, background_tasks: BackgroundTasks):
    """Trigger Polygon data fetch for a symbol."""
    symbol = req.symbol.upper()
    today = datetime.now(timezone.utc).date()
    start = req.start or (today - timedelta(days=2 * 366)).isoformat()
    end = req.end or today.isoformat()

    background_tasks.add_task(_do_fetch, symbol, start, end)
    return {"symbol": symbol, "status": "fetch_started", "start": start, "end": end}


def _do_fetch(symbol: str, start: str, end: str):
    """Background fetch of OHLC + news data."""
    _run_fetch(symbol, start, end)


def _run_fetch(symbol: str, start: str, end: str) -> dict:
    """Fetch OHLC + news data and return lightweight stats."""
    try:
        # OHLC
        ohlc_rows = fetch_ohlc(symbol, start, end)
        conn = get_conn()
        for row in ohlc_rows:
            conn.execute(
                """INSERT OR IGNORE INTO ohlc
                   (symbol, date, open, high, low, close, volume, vwap, transactions)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, row["date"], row["open"], row["high"], row["low"],
                 row["close"], row["volume"], row["vwap"], row["transactions"]),
            )
        conn.execute(
            "UPDATE tickers SET last_ohlc_fetch = ? WHERE symbol = ?",
            (end, symbol),
        )
        conn.commit()

        # News
        articles = fetch_news(symbol, start, end)
        for art in articles:
            news_id = art.get("id")
            if not news_id:
                continue
            tickers = art.get("tickers") or []
            conn.execute(
                """INSERT OR IGNORE INTO news_raw
                   (id, title, description, publisher, author,
                    published_utc, article_url, amp_url, tickers_json, insights_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (news_id, art.get("title"), art.get("description"),
                 art.get("publisher"), art.get("author"), art.get("published_utc"),
                 art.get("article_url"), art.get("amp_url"),
                 json.dumps(tickers),
                 json.dumps(art.get("insights")) if art.get("insights") else None),
            )
            for tk in tickers:
                conn.execute(
                    "INSERT OR IGNORE INTO news_ticker (news_id, symbol) VALUES (?, ?)",
                    (news_id, tk),
                )

        conn.execute(
            "UPDATE tickers SET last_news_fetch = ? WHERE symbol = ?",
            (end, symbol),
        )
        conn.commit()
        conn.close()

        # Run alignment
        align_result = align_news_for_symbol(symbol)
        return {
            "symbol": symbol,
            "status": "fetched",
            "start": start,
            "end": end,
            "ohlc_rows": len(ohlc_rows),
            "news_rows": len(articles),
            "alignment": align_result,
        }
    except Exception:
        logger.exception("Fetch error for %s", symbol)
        raise


def _default_date_range(start: Optional[str], end: Optional[str]) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    return (
        start or (today - timedelta(days=2 * 366)).isoformat(),
        end or today.isoformat(),
    )


def _build_batch_status_response(symbol: str, batch_id: str, l0_stats: dict, fetch_stats: Optional[dict] = None):
    batch_status = check_batch_status(batch_id)
    return {
        "symbol": symbol,
        "mode": "batch",
        "stage": "batch_submitted",
        "message": "Batch analysis submitted",
        "is_done": False,
        "fetch": fetch_stats,
        "layer0": l0_stats,
        "batch": batch_status,
        "batch_id": batch_id,
    }


@router.post("/process")
def trigger_process(req: ProcessRequest):
    """Run optional fetch, Layer 0 filter, then sync or batch Layer 1 analysis."""
    symbol = req.symbol.upper()
    start, end = _default_date_range(req.start, req.end)

    # Step 0: Optional fetch (Polygon API)
    fetch_stats = None
    fetch_error = None
    if req.include_fetch:
        try:
            fetch_stats = _run_fetch(symbol, start, end)
        except Exception as e:
            fetch_error = str(e)
            logger.warning("Fetch partially failed for %s: %s", symbol, e)

    # Step 1: Alignment
    align_result = align_news_for_symbol(symbol)

    # Step 2: Layer 0
    l0_stats = run_layer0(symbol)

    pending_articles = get_pending_articles(symbol, limit=req.batch_size)
    if not pending_articles:
        return {
            "symbol": symbol,
            "mode": req.mode,
            "stage": "completed",
            "message": "No pending articles to analyze",
            "is_done": True,
            "fetch": fetch_stats,
            "fetch_error": fetch_error,
            "alignment": align_result,
            "layer0": l0_stats,
            "layer1": {"status": "no_pending", "total": 0},
            "batch": None,
            "batch_id": None,
        }

    if req.mode == "batch":
        try:
            batch_id = submit_batch_api(symbol, pending_articles)
        except Exception as e:
            logger.warning("Batch submit failed for %s: %s", symbol, e)
            return {
                "symbol": symbol,
                "mode": "batch",
                "stage": "failed",
                "message": f"Batch submit failed: {e}",
                "is_done": False,
                "fetch": fetch_stats,
                "fetch_error": fetch_error,
                "alignment": align_result,
                "layer0": l0_stats,
                "pending_articles": len(pending_articles),
                "batch": None,
                "batch_id": None,
            }
        return {
            "symbol": symbol,
            "mode": "batch",
            "stage": "batch_submitted",
            "message": "Batch analysis submitted",
            "is_done": False,
            "fetch": fetch_stats,
            "fetch_error": fetch_error,
            "alignment": align_result,
            "layer0": l0_stats,
            "pending_articles": len(pending_articles),
            "batch": check_batch_status(batch_id),
            "batch_id": batch_id,
        }

    # Step 3: Run Layer 1 synchronously
    l1_stats = run_layer1(symbol, max_articles=min(req.batch_size, len(pending_articles)))

    return {
        "symbol": symbol,
        "mode": "sync",
        "stage": "completed",
        "message": "Synchronous analysis completed",
        "is_done": True,
        "fetch": fetch_stats,
        "alignment": align_result,
        "layer0": l0_stats,
        "layer1": l1_stats,
        "batch": None,
        "batch_id": None,
    }


@router.get("/batch/{batch_id}")
def get_batch_status(batch_id: str):
    """Check status of a batch job."""
    status = check_batch_status(batch_id)

    # If ended, collect results
    if status["status"] == "ended":
        collect_stats = collect_batch_results(batch_id)
        status["collect_stats"] = collect_stats
        status["status"] = "collected"

    status["stage"] = "completed" if status["status"] == "collected" else "batch_running"
    status["is_done"] = status["status"] == "collected"

    return status
