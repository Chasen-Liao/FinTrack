import numpy as np
import pandas as pd

from backend.ml.model import (
    _compute_classification_metrics,
    filter_neutral_samples,
    resolve_return_col,
    resolve_target_col,
    select_best_search_result,
    target_path_suffix,
)


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


def test_resolve_target_col_defaults_from_horizon():
    assert resolve_target_col("t5", None) == "target_t5"


def test_resolve_target_col_accepts_explicit_low_noise_target():
    assert resolve_target_col("t5", "target_up_big_t5") == "target_up_big_t5"


def test_resolve_return_col_uses_target_suffix_when_available():
    assert resolve_return_col("t1", "target_up_big_t5") == "future_return_t5"


def test_target_path_suffix_is_empty_for_default_target():
    assert target_path_suffix("t5", None) == ""
    assert target_path_suffix("t5", "target_t5") == ""


def test_target_path_suffix_separates_low_noise_targets():
    assert target_path_suffix("t5", "target_up_big_t5") == "_target_up_big_t5"


def test_filter_neutral_samples_uses_explicit_target_return_horizon():
    df = pd.DataFrame(
        {
            "future_return_t1": [0.001, 0.002, 0.003],
            "future_return_t5": [0.04, 0.004, -0.05],
            "target_up_big_t5": [1, 0, 0],
        }
    )

    filtered = filter_neutral_samples(
        df,
        horizon="t1",
        neutral_band=0.01,
        target_col="target_up_big_t5",
    )

    assert filtered["future_return_t5"].tolist() == [0.04, -0.05]
