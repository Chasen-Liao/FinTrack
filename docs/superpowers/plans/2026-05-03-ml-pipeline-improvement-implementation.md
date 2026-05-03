# ML Pipeline Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ML evidence chain consistent by supporting explicit low-noise targets, shared walk-forward evaluation, leak-safe text features, and a report-ready multi-stock evaluation summary.

**Architecture:** Keep the current `build_features -> train/search -> strategy_backtest` flow, but add small shared helpers so experiments, training, and reporting use the same target and validation rules. The plan avoids overwriting existing model artifacts and focuses on reusable evaluation utilities plus tests.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, XGBoost, pytest

---

## File Structure

- Modify: `backend/ml/model.py`
  Responsibility: accept explicit `target_col`, resolve matching future-return column, and use the same target semantics in train/search paths.
- Create: `backend/ml/walk_forward.py`
  Responsibility: provide shared expanding-window probability evaluation with optional fold-local feature transformers.
- Modify: `backend/ml/experiment.py`
  Responsibility: reuse the shared walk-forward evaluator and stop duplicating metric logic.
- Modify: `backend/ml/train.py`
  Responsibility: expose `--target-col` on CLI and pass it through to train/search calls.
- Modify: `backend/ml/features_v2.py`
  Responsibility: separate raw text loading from text vector fitting so text features can be fit per training fold.
- Create: `backend/ml/evaluation_report.py`
  Responsibility: generate multi-stock aggregate metrics for report evidence.
- Modify: `tests/ml/test_model_search.py`
  Responsibility: cover explicit target resolution and target-aware neutral filtering.
- Create: `tests/ml/test_walk_forward.py`
  Responsibility: cover fold output, metric output, and transformer leakage boundaries.
- Create: `tests/ml/test_text_features.py`
  Responsibility: cover text feature fitting only on training rows.
- Create: `tests/ml/test_evaluation_report.py`
  Responsibility: cover aggregation summary shape and threshold statistics.

## Scope Notes

This plan intentionally does not retrain all saved models. Any implementation that writes into `backend/ml/models/` should be limited to explicit user-run commands after the code changes pass tests.

### Task 1: Add Explicit Target Selection To Model Search And Training

**Files:**
- Modify: `backend/ml/model.py`
- Modify: `backend/ml/train.py`
- Test: `tests/ml/test_model_search.py`

- [ ] **Step 1: Write failing tests for target resolution**

Append these tests to `tests/ml/test_model_search.py`:

```python
import pandas as pd

from backend.ml.model import (
    filter_neutral_samples,
    resolve_return_col,
    resolve_target_col,
)


def test_resolve_target_col_defaults_from_horizon():
    assert resolve_target_col("t5", None) == "target_t5"


def test_resolve_target_col_accepts_explicit_low_noise_target():
    assert resolve_target_col("t5", "target_up_big_t5") == "target_up_big_t5"


def test_resolve_return_col_uses_target_suffix_when_available():
    assert resolve_return_col("t1", "target_up_big_t5") == "future_return_t5"


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ml/test_model_search.py -v`

Expected: FAIL with `ImportError` for `resolve_target_col` or `TypeError` for the new `target_col` argument.

- [ ] **Step 3: Add target helper functions in `backend/ml/model.py`**

Add this code above `filter_neutral_samples`:

```python
def resolve_target_col(horizon: str, target_col: str | None = None) -> str:
    """Return the label column used by train/search paths."""
    if target_col:
        return target_col
    return f"target_{horizon}"


def resolve_return_col(horizon: str, target_col: str | None = None) -> str:
    """Return the future-return column that matches a target column."""
    if target_col:
        for suffix in ("t1", "t2", "t3", "t5"):
            if target_col.endswith(f"_{suffix}"):
                return f"future_return_{suffix}"
    return f"future_return_{horizon}"
```

Replace the existing `filter_neutral_samples` signature and return-column lookup with:

```python
def filter_neutral_samples(df, horizon: str, neutral_band: float | None,
                           target_col: str | None = None):
    """Drop rows where future returns are too small to provide a clear direction."""
    if neutral_band is None or neutral_band <= 0:
        return df

    return_col = resolve_return_col(horizon, target_col)
    if return_col not in df.columns:
        raise ValueError(f"Missing column {return_col} required for neutral filtering")

    return df.loc[df[return_col].abs() >= neutral_band].reset_index(drop=True)
```

- [ ] **Step 4: Thread `target_col` through search and train functions**

Update function signatures in `backend/ml/model.py`:

```python
def search_xgboost_params(
    symbol: str,
    horizon: str = "t1",
    n_folds: int = 5,
    min_train: int = 200,
    param_grid: dict[str, list] | None = None,
    metric: str = "accuracy_lift",
    include_market_benchmark: bool = False,
    neutral_band: float | None = None,
    target_col: str | None = None,
) -> dict:
```

Inside `search_xgboost_params`, replace:

```python
target_col = f"target_{horizon}"
```

with:

```python
target_col = resolve_target_col(horizon, target_col)
```

Replace:

```python
df = filter_neutral_samples(df, horizon, neutral_band)
```

with:

```python
df = filter_neutral_samples(df, horizon, neutral_band, target_col=target_col)
```

Add `"target_col": target_col,` to the returned result dict.

Apply the same pattern to `search_xgboost_params_unified`, `train`, and `train_unified`.

- [ ] **Step 5: Add CLI support in `backend/ml/train.py`**

Add the argument after `--horizon`:

```python
parser.add_argument("--target-col", type=str,
                    help="Explicit target column, such as target_up_big_t5")
```

Pass `target_col=args.target_col` into every call to:

```python
search_xgboost_params(...)
search_xgboost_params_unified(...)
train(...)
```

If `train_unified` is called later from CLI, pass the same argument there too.

- [ ] **Step 6: Run target tests**

Run: `pytest tests/ml/test_model_search.py tests/ml/test_feature_enhancements.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/ml/model.py backend/ml/train.py tests/ml/test_model_search.py
git commit -m "feat: support explicit ml target columns"
```

### Task 2: Extract Shared Walk-Forward Evaluation

**Files:**
- Create: `backend/ml/walk_forward.py`
- Modify: `backend/ml/experiment.py`
- Modify: `backend/ml/model.py`
- Test: `tests/ml/test_walk_forward.py`

- [ ] **Step 1: Write failing tests for shared walk-forward evaluation**

Create `tests/ml/test_walk_forward.py` with:

```python
import numpy as np

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ml/test_walk_forward.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.ml.walk_forward'`.

- [ ] **Step 3: Create `backend/ml/walk_forward.py`**

Create the file with:

```python
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
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int, copy=False)
    folds = _resolve_fold_sizes(len(X), n_folds, min_train)
    if not folds:
        return {"error": f"Too few rows ({len(X)}) for expanding-window CV"}

    fold_results = []
    all_true = []
    all_pred = []
    all_prob = []
    all_dates = []

    for fold_index, (train_end, test_end) in enumerate(folds, start=1):
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        if transformer_factory is not None:
            transformer = transformer_factory()
            transformer.fit(X_train, y_train)
            X_train = transformer.transform(X_train)
            X_test = transformer.transform(X_test)

        X_train = np.nan_to_num(X_train.astype(np.float64), copy=False)
        X_test = np.nan_to_num(X_test.astype(np.float64), copy=False)

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
```

- [ ] **Step 4: Update `backend/ml/model.py` search to use the shared evaluator**

Import the helper:

```python
from backend.ml.walk_forward import run_walk_forward_probabilities
```

Replace `_expanding_window_cv` body with:

```python
def _expanding_window_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    n_folds: int = 5,
    min_train: int = 200,
    model_params: dict | None = None,
) -> dict:
    """Run expanding-window CV and return aggregate and per-fold metrics."""
    return run_walk_forward_probabilities(
        X=X,
        y=y,
        dates=dates,
        model_factory=lambda: _build_xgb_classifier(model_params),
        n_folds=n_folds,
        min_train=min_train,
    )
```

- [ ] **Step 5: Update `backend/ml/experiment.py` to reuse the evaluator**

Import:

```python
from backend.ml.walk_forward import run_walk_forward_probabilities
```

Replace `_expanding_cv` with:

```python
def _expanding_cv(X, y, n_folds=5, min_train=200, model_cls=None, model_kwargs=None):
    """Run expanding-window CV and return aggregate metrics."""
    def make_model():
        if model_cls is None:
            return XGBClassifier(
                max_depth=4, n_estimators=200, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42,
            )
        return model_cls(**(model_kwargs or {}))

    result = run_walk_forward_probabilities(
        X=X,
        y=y,
        dates=[str(i) for i in range(len(y))],
        model_factory=make_model,
        n_folds=n_folds,
        min_train=min_train,
    )
    if "error" in result:
        return None

    return {
        "n": result["total_predictions"],
        "accuracy": round(result["accuracy"], 4),
        "baseline": round(result["baseline"], 4),
        "lift": round(result["accuracy_lift"] * 100, 1),
        "precision": round(result["precision"], 4),
        "recall": round(result["recall"], 4),
        "f1": round(result["f1"], 4),
        "roc_auc": round(result["roc_auc"], 4) if result["roc_auc"] is not None else None,
    }
```

- [ ] **Step 6: Run walk-forward tests**

Run: `pytest tests/ml/test_walk_forward.py tests/ml/test_model_search.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/ml/walk_forward.py backend/ml/model.py backend/ml/experiment.py tests/ml/test_walk_forward.py
git commit -m "feat: add shared walk-forward evaluator"
```

### Task 3: Make Text SVD Features Fit Only Inside Training Windows

**Files:**
- Modify: `backend/ml/features_v2.py`
- Test: `tests/ml/test_text_features.py`

- [ ] **Step 1: Write failing tests for text transformer boundaries**

Create `tests/ml/test_text_features.py` with:

```python
import pandas as pd

from backend.ml.features_v2 import TextSvdFeatureTransformer


def test_text_svd_transformer_fits_only_training_rows():
    train = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "text": ["earnings beat demand", "chip demand rises", "margin demand improves"],
        }
    )
    test = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-05"]),
            "text": ["unseen litigation topic"],
        }
    )

    transformer = TextSvdFeatureTransformer(max_features=20, min_df=1, n_components=2)
    transformer.fit(train)
    transformed = transformer.transform(test)

    assert transformer.fit_dates_ == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert transformed["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-05"]
    assert "text_svd_0" in transformed.columns
    assert "text_svd_1" in transformed.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ml/test_text_features.py -v`

Expected: FAIL with `ImportError` for `TextSvdFeatureTransformer`.

- [ ] **Step 3: Add `TextSvdFeatureTransformer` in `backend/ml/features_v2.py`**

Add this class below imports:

```python
class TextSvdFeatureTransformer:
    """Fit TF-IDF and SVD on a training window, then transform another window."""

    def __init__(self, max_features: int = 500, min_df: int = 3, n_components: int = 10):
        self.max_features = max_features
        self.min_df = min_df
        self.n_components = n_components
        self.vectorizer = None
        self.svd = None
        self.output_columns_: list[str] = []
        self.fit_dates_: list[str] = []

    def fit(self, rows: pd.DataFrame, y=None):
        texts = rows["text"].fillna("").tolist()
        self.fit_dates_ = rows["trade_date"].dt.strftime("%Y-%m-%d").tolist()
        if not any(text.strip() for text in texts):
            self.output_columns_ = []
            return self

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            min_df=self.min_df,
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if n_comp < 1:
            self.output_columns_ = []
            self.svd = None
            return self

        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf_matrix)
        self.output_columns_ = [f"text_svd_{i}" for i in range(n_comp)]
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        result = rows[["trade_date"]].copy()
        if self.vectorizer is None or self.svd is None:
            return result

        matrix = self.vectorizer.transform(rows["text"].fillna("").tolist())
        reduced = self.svd.transform(matrix)
        for i, col in enumerate(self.output_columns_):
            result[col] = reduced[:, i]
        return result
```

- [ ] **Step 4: Refactor `_build_text_features` to use the transformer for full-sample compatibility**

Replace the TF-IDF/SVD block in `_build_text_features` with:

```python
    transformer = TextSvdFeatureTransformer(n_components=n_components)
    transformer.fit(text_df)
    result = transformer.transform(text_df)
    if not transformer.output_columns_:
        return pd.DataFrame({"trade_date": dates})
    return result
```

This keeps the old `build_features_v2(symbol)` behavior while giving walk-forward code a leak-safe transformer to use per fold.

- [ ] **Step 5: Run text feature tests**

Run: `pytest tests/ml/test_text_features.py -v`

Expected: PASS.

- [ ] **Step 6: Run feature and experiment smoke tests**

Run: `pytest tests/ml/test_feature_enhancements.py tests/ml/test_text_features.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/ml/features_v2.py tests/ml/test_text_features.py
git commit -m "feat: add leak-safe text feature transformer"
```

### Task 4: Generate Report-Ready Multi-Stock Evaluation Summary

**Files:**
- Create: `backend/ml/evaluation_report.py`
- Modify: `backend/ml/train.py`
- Test: `tests/ml/test_evaluation_report.py`

- [ ] **Step 1: Write failing tests for aggregate summary**

Create `tests/ml/test_evaluation_report.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ml/test_evaluation_report.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.ml.evaluation_report'`.

- [ ] **Step 3: Create `backend/ml/evaluation_report.py`**

Create the file with:

```python
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
```

- [ ] **Step 4: Add CLI switch in `backend/ml/train.py`**

Import:

```python
from backend.ml.evaluation_report import run_multi_stock_evaluation
```

Add parser arguments:

```python
parser.add_argument("--evaluation-report", action="store_true",
                    help="Generate report-ready aggregate evaluation JSON")
parser.add_argument("--evaluation-output", type=str,
                    default="backend/ml/models/evaluation_summary.json",
                    help="Output path for --evaluation-report")
```

After `symbols` and `horizons` are computed, add:

```python
    if args.evaluation_report:
        if len(horizons) != 1:
            raise SystemExit("--evaluation-report requires --horizon")
        report = run_multi_stock_evaluation(
            symbols=symbols,
            horizon=horizons[0],
            target_col=args.target_col,
            metric=args.metric,
            neutral_band=args.neutral_band,
            include_market_benchmark=args.market_benchmark,
            output_path=args.evaluation_output,
        )
        summary = report["summary"]
        print(
            f"Evaluation report: n={summary['count']} "
            f"auc_mean={summary['roc_auc_mean']} "
            f"auc_median={summary['roc_auc_median']} "
            f"auc>0.5={summary['roc_auc_above_0_5_ratio']}"
        )
        return
```

- [ ] **Step 5: Run evaluation report tests**

Run: `pytest tests/ml/test_evaluation_report.py -v`

Expected: PASS.

- [ ] **Step 6: Run all ML unit tests**

Run: `pytest tests/ml/ -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/ml/evaluation_report.py backend/ml/train.py tests/ml/test_evaluation_report.py
git commit -m "feat: add ml evaluation summary report"
```

### Task 5: Produce A Report Evidence Snapshot Without Overwriting Models

**Files:**
- Create output by command: `backend/ml/models/evaluation_summary_target_up_big_t5.json`
- No code file changes expected in this task

- [ ] **Step 1: Run aggregate evaluation for the low-noise T+5 target**

Run:

```bash
python -m backend.ml.train --evaluation-report --horizon t5 --target-col target_up_big_t5 --metric roc_auc --evaluation-output backend/ml/models/evaluation_summary_target_up_big_t5.json
```

Expected output includes:

```text
Evaluation report: n=
auc_mean=
auc_median=
auc>0.5=
```

- [ ] **Step 2: Inspect generated JSON**

Run:

```bash
python -m json.tool backend/ml/models/evaluation_summary_target_up_big_t5.json
```

Expected: valid JSON with top-level keys `horizon`, `target_col`, `summary`, `rows`, and `errors`.

- [ ] **Step 3: Decide whether to keep the generated artifact**

If this is intended as a course-report evidence snapshot, commit it:

```bash
git add backend/ml/models/evaluation_summary_target_up_big_t5.json
git commit -m "docs: add ml evaluation evidence snapshot"
```

If the artifact is only a local exploratory run, leave it uncommitted and mention the generated path in the final handoff.

## Self-Review Checklist

- Spec coverage: Task 1 covers explicit low-noise target selection; Task 2 covers shared walk-forward evaluation; Task 3 covers leak-safe text features; Task 4 covers report-ready aggregate metrics; Task 5 covers evidence generation without overwriting existing models.
- Placeholder scan: This plan contains no placeholder markers, no open-ended test instructions, and no missing file paths.
- Type consistency: `target_col`, `neutral_band`, `roc_auc`, `accuracy_lift`, and `run_walk_forward_probabilities` use the same names across tasks.
- Scope check: The plan is a single implementation track focused on ML evidence quality. Deep learning experiments and full report rewrites remain outside this plan.
