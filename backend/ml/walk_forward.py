"""Shared walk-forward evaluation utilities for ML experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    baseline = max(y_true.mean(), 1 - y_true.mean())
    metrics = {
        "accuracy": float(accuracy),
        "baseline": float(baseline),
        "accuracy_lift": float(accuracy - baseline),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def _resolve_fold_sizes(n: int, n_folds: int, min_train: int) -> list[tuple[int, int]]:
    if n < min_train + 20:
        return []

    test_size = (n - min_train) // n_folds
    if test_size < 10:
        n_folds = max(1, (n - min_train) // 10)
        test_size = (n - min_train) // n_folds

    folds = []
    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_end = train_end + test_size if fold < n_folds - 1 else n
        if test_end > train_end:
            folds.append((train_end, test_end))
    return folds


def _slice_rows(values, start: int, end: int):
    if hasattr(values, "iloc"):
        return values.iloc[start:end].copy()
    return values[start:end]


def run_walk_forward_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    model_factory: Callable[[], object],
    transformer_factory: Callable[[], object] | None = None,
    n_folds: int = 5,
    min_train: int = 200,
) -> dict:
    """Run expanding-window validation and return out-of-sample probabilities."""
    n_rows = len(X)
    y = np.asarray(y).astype(int, copy=False)
    folds = _resolve_fold_sizes(n_rows, n_folds, min_train)
    if not folds:
        return {"error": f"Too few rows ({n_rows}) for expanding-window CV"}

    fold_results = []
    all_true = []
    all_pred = []
    all_prob = []
    all_dates = []

    for fold_index, (train_end, test_end) in enumerate(folds, start=1):
        X_train, y_train = _slice_rows(X, 0, train_end), y[:train_end]
        X_test, y_test = _slice_rows(X, train_end, test_end), y[train_end:test_end]

        if transformer_factory is not None:
            transformer = transformer_factory()
            transformer.fit(X_train, y_train)
            X_train = transformer.transform(X_train)
            X_test = transformer.transform(X_test)

        X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float64), copy=True)
        X_test = np.nan_to_num(np.asarray(X_test, dtype=np.float64), copy=True)

        model = model_factory()
        model.fit(X_train, y_train)
        fold_prob = model.predict_proba(X_test)[:, 1]
        fold_pred = (fold_prob >= 0.5).astype(int)

        metrics = _compute_classification_metrics(y_test, fold_pred, fold_prob)
        fold_results.append(
            {
                "fold": fold_index,
                "train_size": int(train_end),
                "test_size": int(test_end - train_end),
                "test_start": dates[train_end],
                "test_end": dates[test_end - 1],
                **{
                    key: (round(value, 4) if isinstance(value, float) else value)
                    for key, value in metrics.items()
                },
            }
        )

        all_true.extend(y_test.tolist())
        all_pred.extend(fold_pred.tolist())
        all_prob.extend(fold_prob.tolist())
        all_dates.extend(dates[train_end:test_end])

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)
    overall = _compute_classification_metrics(y_true, y_pred, y_prob)

    return {
        "n_folds": len(fold_results),
        "total_predictions": len(all_true),
        "folds": fold_results,
        "dates": all_dates,
        "y_true": all_true,
        "predictions": all_pred,
        "probabilities": all_prob,
        **overall,
    }
