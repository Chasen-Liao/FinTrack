import numpy as np
import pandas as pd

from backend.ml.features import _add_market_benchmark_features, add_future_return_targets
from backend.ml.model import filter_neutral_samples


def test_add_future_return_targets_creates_direction_and_big_move_labels():
    df = pd.DataFrame(
        {
            "close": [100.0, 103.5, 101.0, 106.0, 109.0, 112.0, 108.0],
        }
    )

    result = add_future_return_targets(df.copy())

    assert "future_return_t5" in result.columns
    assert "target_t5" in result.columns
    assert "target_big1_t5" in result.columns
    assert "target_up_big_t5" in result.columns
    assert "target_down_big_t5" in result.columns
    assert result.loc[0, "target_t5"] == 1
    assert result.loc[0, "target_big1_t5"] == 1
    assert result.loc[0, "target_up_big_t5"] == 1


def test_add_market_benchmark_features_merges_shifted_relative_returns():
    base = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [100.0, 102.0, 101.0],
            "ret_1d": [0.0, 0.01, -0.02],
        }
    )
    benchmark = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "benchmark_ret_1d": [0.0, 0.005, 0.01],
            "benchmark_ret_3d": [0.0, 0.004, 0.008],
            "benchmark_ret_5d": [0.0, 0.003, 0.007],
            "benchmark_ret_10d": [0.0, 0.002, 0.006],
            "benchmark_volatility_5d": [0.1, 0.11, 0.12],
            "benchmark_volatility_10d": [0.2, 0.21, 0.22],
        }
    )

    result = _add_market_benchmark_features(base.copy(), benchmark)

    assert "benchmark_ret_1d" in result.columns
    assert "rel_ret_1d_vs_benchmark" in result.columns
    assert result.loc[1, "benchmark_ret_1d"] == 0.005
    assert result.loc[2, "rel_ret_1d_vs_benchmark"] == -0.03


def test_filter_neutral_samples_drops_small_absolute_returns():
    df = pd.DataFrame(
        {
            "future_return_t5": [0.03, 0.004, -0.002, -0.02],
            "target_t5": [1, 1, 0, 0],
        }
    )

    filtered = filter_neutral_samples(df, "t5", neutral_band=0.01)

    assert filtered["future_return_t5"].tolist() == [0.03, -0.02]


def test_auc_metric_prefers_probabilities_over_hard_labels():
    from sklearn.metrics import roc_auc_score

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.10, 0.40, 0.60, 0.90])

    assert roc_auc_score(y_true, y_prob) == 1.0
