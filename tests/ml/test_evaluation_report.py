from backend.ml.evaluation_report import summarize_metric_distribution


def test_summarize_metric_distribution_reports_auc_distribution():
    rows = [
        {"symbol": "AAA", "roc_auc": 0.49, "accuracy_lift": -0.02},
        {"symbol": "BBB", "roc_auc": 0.51, "accuracy_lift": 0.01},
        {"symbol": "CCC", "roc_auc": 0.57, "accuracy_lift": 0.03},
    ]

    summary = summarize_metric_distribution(rows)

    assert summary["count"] == 3
    assert summary["roc_auc_mean"] == 0.5233
    assert summary["roc_auc_median"] == 0.51
    assert summary["roc_auc_above_0_5_ratio"] == 0.6667
    assert summary["accuracy_lift_mean"] == 0.0067
