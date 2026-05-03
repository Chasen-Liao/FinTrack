"""Utilities for generating report-ready ML evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.ml.model import search_xgboost_params


def _round(value: float) -> float:
    return round(float(value), 4)


def summarize_metric_distribution(rows: list[dict]) -> dict:
    """Summarize per-symbol metrics for report tables."""
    valid_auc = [float(row["roc_auc"]) for row in rows if row.get("roc_auc") is not None]
    valid_lift = [float(row["accuracy_lift"]) for row in rows if row.get("accuracy_lift") is not None]
    if not valid_auc:
        return {
            "count": 0,
            "roc_auc_mean": None,
            "roc_auc_median": None,
            "roc_auc_above_0_5_ratio": None,
            "accuracy_lift_mean": None,
        }

    auc = np.array(valid_auc, dtype=float)
    lift = np.array(valid_lift, dtype=float) if valid_lift else np.array([], dtype=float)
    return {
        "count": len(valid_auc),
        "roc_auc_mean": _round(np.mean(auc)),
        "roc_auc_median": _round(np.median(auc)),
        "roc_auc_p25": _round(np.percentile(auc, 25)),
        "roc_auc_p75": _round(np.percentile(auc, 75)),
        "roc_auc_above_0_5_ratio": _round(np.mean(auc > 0.5)),
        "accuracy_lift_mean": _round(np.mean(lift)) if len(lift) else None,
    }


def run_multi_stock_evaluation(
    symbols: list[str],
    horizon: str = "t5",
    target_col: str | None = None,
    metric: str = "roc_auc",
    neutral_band: float | None = None,
    include_market_benchmark: bool = False,
    output_path: str | Path | None = None,
) -> dict:
    """Run per-symbol searches and return aggregate metrics for report use."""
    rows = []
    errors = []
    for symbol in symbols:
        result = search_xgboost_params(
            symbol=symbol,
            horizon=horizon,
            metric=metric,
            neutral_band=neutral_band,
            include_market_benchmark=include_market_benchmark,
            target_col=target_col,
        )
        if "error" in result:
            errors.append({"symbol": symbol, "error": result["error"]})
            continue
        metrics = result["best_metrics"]
        rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "target_col": result.get("target_col", target_col or f"target_{horizon}"),
                "roc_auc": metrics.get("roc_auc"),
                "accuracy": metrics.get("accuracy"),
                "baseline": metrics.get("baseline"),
                "accuracy_lift": metrics.get("accuracy_lift"),
                "f1": metrics.get("f1"),
                "best_params": result.get("best_params", {}),
            }
        )

    payload = {
        "horizon": horizon,
        "target_col": target_col or f"target_{horizon}",
        "metric": metric,
        "neutral_band": neutral_band,
        "include_market_benchmark": include_market_benchmark,
        "summary": summarize_metric_distribution(rows),
        "rows": rows,
        "errors": errors,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload
