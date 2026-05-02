from backend.ml.model import select_best_search_result


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
