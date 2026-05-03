import numpy as np
import pandas as pd

from backend.ml.walk_forward import run_walk_forward_probabilities


class MeanProbabilityModel:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        probs = np.full(len(X), self.mean_)
        return np.column_stack([1.0 - probs, probs])


class RecordingTransformer:
    fit_lengths = []

    def fit(self, X, y=None):
        self.fit_len_ = len(X)
        RecordingTransformer.fit_lengths.append(len(X))
        return self

    def transform(self, X):
        return X + self.fit_len_


class DataFrameTransformer:
    fit_windows = []

    def fit(self, rows, y=None):
        DataFrameTransformer.fit_windows.append(rows["text"].tolist())
        return self

    def transform(self, rows):
        return pd.DataFrame(
            {
                "value": rows["value"].to_numpy(dtype=float),
                "text_len": rows["text"].str.len().to_numpy(dtype=float),
            }
        )


def test_run_walk_forward_probabilities_returns_metrics_and_folds():
    X = np.arange(60, dtype=float).reshape(-1, 1)
    y = np.array([0, 1] * 30)

    result = run_walk_forward_probabilities(
        X=X,
        y=y,
        dates=[f"2024-01-{(i % 28) + 1:02d}" for i in range(60)],
        model_factory=MeanProbabilityModel,
        n_folds=4,
        min_train=20,
    )

    assert result["n_folds"] == 4
    assert result["total_predictions"] == 40
    assert "roc_auc" in result
    assert len(result["probabilities"]) == 40


def test_run_walk_forward_probabilities_fits_transformer_inside_each_fold():
    RecordingTransformer.fit_lengths = []
    X = np.arange(50, dtype=float).reshape(-1, 1)
    y = np.array([0, 1] * 25)

    run_walk_forward_probabilities(
        X=X,
        y=y,
        dates=[f"2024-02-{(i % 28) + 1:02d}" for i in range(50)],
        model_factory=MeanProbabilityModel,
        transformer_factory=RecordingTransformer,
        n_folds=3,
        min_train=20,
    )

    assert RecordingTransformer.fit_lengths == [20, 30, 40]


def test_run_walk_forward_probabilities_keeps_dataframe_until_transformer_fit():
    DataFrameTransformer.fit_windows = []
    X = pd.DataFrame(
        {
            "value": np.arange(50, dtype=float),
            "text": [f"row-{i}" for i in range(50)],
        }
    )
    y = np.array([0, 1] * 25)

    result = run_walk_forward_probabilities(
        X=X,
        y=y,
        dates=[f"2024-03-{(i % 28) + 1:02d}" for i in range(50)],
        model_factory=MeanProbabilityModel,
        transformer_factory=DataFrameTransformer,
        n_folds=3,
        min_train=20,
    )

    assert result["total_predictions"] == 30
    assert DataFrameTransformer.fit_windows[0] == [f"row-{i}" for i in range(20)]
