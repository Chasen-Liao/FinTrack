import numpy as np
from backend.ml.model import select_best_search_result, _compute_classification_metrics


def test_select_best_search_result_prefers_accuracy_lift_then_f1():
    results = [
        {
            "params": {"max_depth": 3},
            "param_count": 1,
            "accuracy_lift": 0.03,
            "f1": 0.55,
            "accuracy": 0.58,
            "roc_auc": 0.61,
        },
        {
            "params": {"max_depth": 4},
            "param_count": 1,
            "accuracy_lift": 0.03,
            "f1": 0.59,
            "accuracy": 0.57,
            "roc_auc": 0.60,
        },
        {
            "params": {"max_depth": 5},
            "param_count": 1,
            "accuracy_lift": 0.02,
            "f1": 0.70,
            "accuracy": 0.60,
            "roc_auc": 0.66,
        },
    ]

    best = select_best_search_result(results, metric="accuracy_lift")

    assert best["params"] == {"max_depth": 4}


def test_select_best_search_result_prefers_roc_auc_when_requested():
    results = [
        {
            "params": {"max_depth": 3},
            "param_count": 1,
            "accuracy_lift": 0.04,
            "f1": 0.58,
            "accuracy": 0.59,
            "roc_auc": 0.54,
        },
        {
            "params": {"max_depth": 4},
            "param_count": 1,
            "accuracy_lift": 0.02,
            "f1": 0.55,
            "accuracy": 0.57,
            "roc_auc": 0.61,
        },
    ]

    best = select_best_search_result(results, metric="roc_auc")

    assert best["params"] == {"max_depth": 4}


def test_compute_classification_metrics_includes_roc_auc():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = _compute_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["roc_auc"] == 1.0
