"""XGBoost model training and prediction."""

import json
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from backend.ml.features import (
    build_features,
    build_features_multi,
    FEATURE_COLS,
    FEATURE_COLS_WITH_MARKET,
    resolve_feature_cols,
)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_XGB_PARAMS = {
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "eval_metric": "logloss",
    "random_state": 42,
}

DEFAULT_SEARCH_GRID = {
    "max_depth": [3, 4],
    "n_estimators": [150, 300],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
}


def _build_xgb_classifier(model_params: dict | None = None) -> XGBClassifier:
    params = DEFAULT_XGB_PARAMS.copy()
    if model_params:
        params.update(model_params)
    return XGBClassifier(**params)


def _fit_xgb_with_train_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: dict | None = None,
) -> XGBClassifier:
    """Fit XGBoost with a small chronological validation tail for early stopping."""
    X_train = np.nan_to_num(X_train.astype(np.float64), copy=False)
    y_train = y_train.astype(int, copy=False)

    model = _build_xgb_classifier(model_params)

    if len(X_train) >= 80:
        val_size = max(10, min(len(X_train) // 5, 40))
        split_idx = len(X_train) - val_size
        X_fit, X_val = X_train[:split_idx], X_train[split_idx:]
        y_fit, y_val = y_train[:split_idx], y_train[split_idx:]
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


def _make_param_grid(grid: dict[str, list] | None = None) -> list[dict]:
    search_grid = grid or DEFAULT_SEARCH_GRID
    keys = list(search_grid.keys())
    values = [search_grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


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


def filter_neutral_samples(df, horizon: str, neutral_band: float | None):
    """Drop rows where future returns are too small to provide a clear direction."""
    if neutral_band is None or neutral_band <= 0:
        return df

    return_col = f"future_return_{horizon}"
    if return_col not in df.columns:
        raise ValueError(f"Missing column {return_col} required for neutral filtering")

    return df.loc[df[return_col].abs() >= neutral_band].reset_index(drop=True)


def _expanding_window_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    n_folds: int = 5,
    min_train: int = 200,
    model_params: dict | None = None,
) -> dict:
    """Run expanding-window CV and return aggregate and per-fold metrics."""
    n = len(X)
    if n < min_train + 20:
        return {"error": f"Too few rows ({n}) for expanding-window CV"}

    test_size = (n - min_train) // n_folds
    if test_size < 10:
        n_folds = max(1, (n - min_train) // 10)
        test_size = (n - min_train) // n_folds

    folds = []
    all_true = []
    all_pred = []
    all_prob = []

    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_end = train_end + test_size if fold < n_folds - 1 else n

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        if len(X_test) == 0:
            continue

        model = _fit_xgb_with_train_validation(X_train, y_train, model_params)
        X_test = np.nan_to_num(X_test.astype(np.float64), copy=False)
        fold_prob = model.predict_proba(X_test)[:, 1]
        fold_pred = (fold_prob >= 0.5).astype(int)

        metrics = _compute_classification_metrics(y_test, fold_pred, fold_prob)
        folds.append({
            "fold": fold + 1,
            "train_size": int(train_end),
            "test_size": int(test_end - train_end),
            "test_start": dates[train_end],
            "test_end": dates[test_end - 1],
            **{
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in metrics.items()
            },
        })

        all_true.extend(y_test.tolist())
        all_pred.extend(fold_pred.tolist())
        all_prob.extend(fold_prob.tolist())

    if not all_true:
        return {"error": "No out-of-sample predictions generated"}

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)
    overall = _compute_classification_metrics(y_true, y_pred, y_prob)

    return {
        "n_folds": len(folds),
        "total_predictions": len(all_true),
        "folds": folds,
        **overall,
    }


def select_best_search_result(results: list[dict], metric: str = "accuracy_lift") -> dict:
    """Pick the best search result with stable tie-breaking."""
    if not results:
        raise ValueError("No search results to rank")

    def score_value(item: dict, key: str) -> float:
        value = item.get(key)
        return float("-inf") if value is None else float(value)

    ranked = sorted(
        results,
        key=lambda item: (
            score_value(item, metric),
            score_value(item, "f1"),
            score_value(item, "accuracy"),
            score_value(item, "roc_auc"),
            -int(item.get("param_count", 0)),
        ),
        reverse=True,
    )
    return ranked[0]


def _search_suffix(include_market_benchmark: bool, neutral_band: float | None) -> str:
    parts = []
    if include_market_benchmark:
        parts.append("market")
    if neutral_band is not None and neutral_band > 0:
        parts.append(f"neutral{int(round(neutral_band * 10000))}")
    return "" if not parts else "_" + "_".join(parts)


def search_xgboost_params(
    symbol: str,
    horizon: str = "t1",
    n_folds: int = 5,
    min_train: int = 200,
    param_grid: dict[str, list] | None = None,
    metric: str = "accuracy_lift",
    include_market_benchmark: bool = False,
    neutral_band: float | None = None,
) -> dict:
    """Search XGBoost parameters with expanding-window validation."""
    target_col = f"target_{horizon}"
    df = build_features(symbol, include_market_benchmark=include_market_benchmark)
    if df.empty or len(df) < min_train + 20:
        return {"error": f"Not enough data for {symbol} ({len(df)} rows)"}

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df = filter_neutral_samples(df, horizon, neutral_band)
    if df.empty or len(df) < min_train + 20:
        return {"error": f"Not enough rows for {symbol}/{horizon} after neutral filtering ({len(df)} rows)"}

    feature_cols = FEATURE_COLS_WITH_MARKET if include_market_benchmark else FEATURE_COLS
    X = df[feature_cols].values
    y = df[target_col].values
    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()

    candidates = []
    for params in _make_param_grid(param_grid):
        cv_result = _expanding_window_cv(
            X=X,
            y=y,
            dates=dates,
            n_folds=n_folds,
            min_train=min_train,
            model_params=params,
        )
        if "error" in cv_result:
            continue
        candidate = {
            "params": params,
            "param_count": len(params),
            "accuracy": round(cv_result["accuracy"], 4),
            "baseline": round(cv_result["baseline"], 4),
            "accuracy_lift": round(cv_result["accuracy_lift"], 4),
            "precision": round(cv_result["precision"], 4),
            "recall": round(cv_result["recall"], 4),
            "f1": round(cv_result["f1"], 4),
            "roc_auc": round(cv_result["roc_auc"], 4) if cv_result["roc_auc"] is not None else None,
            "n_folds": cv_result["n_folds"],
            "total_predictions": cv_result["total_predictions"],
            "folds": cv_result["folds"],
        }
        candidates.append(candidate)

    if not candidates:
        return {"error": f"No valid parameter search results for {symbol}/{horizon}"}

    best = select_best_search_result(candidates, metric=metric)
    result = {
        "symbol": symbol,
        "horizon": horizon,
        "metric": metric,
        "include_market_benchmark": include_market_benchmark,
        "neutral_band": neutral_band,
        "searched_at": datetime.now().isoformat(),
        "candidate_count": len(candidates),
        "default_params": DEFAULT_XGB_PARAMS,
        "best_params": best["params"],
        "best_metrics": {
            "accuracy": best["accuracy"],
            "baseline": best["baseline"],
            "accuracy_lift": best["accuracy_lift"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "roc_auc": best["roc_auc"],
        },
        "top_candidates": sorted(
            candidates,
            key=lambda item: (
                item.get(metric, float("-inf")),
                item.get("f1", float("-inf")),
                item.get("accuracy", float("-inf")),
            ),
            reverse=True,
        )[:10],
    }

    suffix = _search_suffix(include_market_benchmark, neutral_band)
    out_path = MODELS_DIR / f"{symbol}_{horizon}_xgb_search{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result


def search_xgboost_params_unified(
    horizon: str = "t5",
    symbols: list[str] | None = None,
    n_folds: int = 5,
    min_train: int = 200,
    param_grid: dict[str, list] | None = None,
    metric: str = "roc_auc",
    include_market_benchmark: bool = False,
    neutral_band: float | None = None,
) -> dict:
    """Search XGBoost params on combined multi-ticker data with expanding-window CV."""
    target_col = f"target_{horizon}"
    df = build_features_multi(symbols, include_market_benchmark=include_market_benchmark)
    if df.empty or len(df) < min_train + 20:
        return {"error": f"Not enough combined data for unified search ({len(df)} rows)"}

    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df = filter_neutral_samples(df, horizon, neutral_band)
    if df.empty or len(df) < min_train + 20:
        return {"error": f"Not enough combined rows after neutral filtering ({len(df)} rows)"}

    feature_cols = FEATURE_COLS_WITH_MARKET if include_market_benchmark else FEATURE_COLS
    X = df[feature_cols].values
    y = df[target_col].values
    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()

    candidates = []
    for params in _make_param_grid(param_grid):
        cv_result = _expanding_window_cv(
            X=X,
            y=y,
            dates=dates,
            n_folds=n_folds,
            min_train=min_train,
            model_params=params,
        )
        if "error" in cv_result:
            continue
        candidates.append(
            {
                "params": params,
                "param_count": len(params),
                "accuracy": round(cv_result["accuracy"], 4),
                "baseline": round(cv_result["baseline"], 4),
                "accuracy_lift": round(cv_result["accuracy_lift"], 4),
                "precision": round(cv_result["precision"], 4),
                "recall": round(cv_result["recall"], 4),
                "f1": round(cv_result["f1"], 4),
                "roc_auc": round(cv_result["roc_auc"], 4) if cv_result["roc_auc"] is not None else None,
                "n_folds": cv_result["n_folds"],
                "total_predictions": cv_result["total_predictions"],
            }
        )

    if not candidates:
        return {"error": f"No valid unified search results for {horizon}"}

    best = select_best_search_result(candidates, metric=metric)
    return {
        "symbol": "UNIFIED",
        "horizon": horizon,
        "metric": metric,
        "include_market_benchmark": include_market_benchmark,
        "neutral_band": neutral_band,
        "candidate_count": len(candidates),
        "best_params": best["params"],
        "best_metrics": {
            "accuracy": best["accuracy"],
            "baseline": best["baseline"],
            "accuracy_lift": best["accuracy_lift"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "roc_auc": best["roc_auc"],
        },
    }


def train(symbol: str, horizon: str = "t1", model_params: dict | None = None,
          include_market_benchmark: bool = False,
          neutral_band: float | None = None) -> dict:
    """Train XGBoost for a single symbol/horizon. Returns metrics dict."""
    target_col = f"target_{horizon}"

    df = build_features(symbol, include_market_benchmark=include_market_benchmark)
    if df.empty or len(df) < 60:
        return {"error": f"Not enough data for {symbol} ({len(df)} rows)"}

    # Drop rows where target is NaN (last few days)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df = filter_neutral_samples(df, horizon, neutral_band)
    if df.empty or len(df) < 60:
        return {"error": f"Not enough data for {symbol}/{horizon} after neutral filtering ({len(df)} rows)"}

    feature_cols = FEATURE_COLS_WITH_MARKET if include_market_benchmark else FEATURE_COLS
    X = df[feature_cols].values
    y = df[target_col].values
    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()

    # Time-series split: last 20% for test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = _fit_xgb_with_train_validation(X_train, y_train, model_params=model_params)
    X_test = np.nan_to_num(X_test.astype(np.float64), copy=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    # Feature importance
    importances = model.feature_importances_
    top_features = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    try:
        roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
    except ValueError:
        roc_auc = None

    meta = {
        "symbol": symbol,
        "horizon": horizon,
        "accuracy": round(accuracy, 4),
        "baseline": round(baseline, 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "train_size": split_idx,
        "test_size": len(y_test),
        "train_start": dates[0],
        "train_end": dates[split_idx - 1],
        "test_start": dates[split_idx],
        "test_end": dates[-1],
        "params": {**DEFAULT_XGB_PARAMS, **(model_params or {})},
        "include_market_benchmark": include_market_benchmark,
        "neutral_band": neutral_band,
        "feature_count": len(feature_cols),
        "roc_auc": roc_auc,
        "top_features": [{"name": n, "importance": round(v, 4)} for n, v in top_features],
        "trained_at": datetime.now().isoformat(),
    }

    # Save
    model_path = MODELS_DIR / f"{symbol}_{horizon}.joblib"
    meta_path = MODELS_DIR / f"{symbol}_{horizon}_meta.json"

    # 使用 XGBoost 原生格式保存，更稳定且跨版本兼容
    booster = model.get_booster()
    json_path = MODELS_DIR / f"{symbol}_{horizon}_xgboost.json"
    booster.save_model(str(json_path))

    # 同时保留 joblib 以保持向后兼容
    joblib.dump(model, model_path)

    meta_path.write_text(json.dumps(meta, indent=2))

    return meta


def train_unified(horizon: str = "t1", symbols: list[str] | None = None,
                  model_params: dict | None = None,
                  include_market_benchmark: bool = False,
                  neutral_band: float | None = None) -> dict:
    """Train a single XGBoost on ALL tickers combined. Returns metrics dict."""
    target_col = f"target_{horizon}"

    df = build_features_multi(symbols, include_market_benchmark=include_market_benchmark)
    if df.empty or len(df) < 100:
        return {"error": f"Not enough combined data ({len(df)} rows)"}

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df = filter_neutral_samples(df, horizon, neutral_band)
    if df.empty or len(df) < 100:
        return {"error": f"Not enough combined rows after neutral filtering ({len(df)} rows)"}

    feature_cols = FEATURE_COLS_WITH_MARKET if include_market_benchmark else FEATURE_COLS
    X = df[feature_cols].values
    y = df[target_col].values
    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()
    syms = df["symbol"].tolist()

    # Time-series split: sort by date, last 20% for test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = _fit_xgb_with_train_validation(X_train, y_train, model_params=model_params)
    X_test = np.nan_to_num(X_test.astype(np.float64), copy=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    importances = model.feature_importances_
    top_features = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    try:
        roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
    except ValueError:
        roc_auc = None

    meta = {
        "symbol": "UNIFIED",
        "horizon": horizon,
        "accuracy": round(accuracy, 4),
        "baseline": round(baseline, 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "train_size": split_idx,
        "test_size": len(y_test),
        "train_start": dates[0],
        "train_end": dates[split_idx - 1],
        "test_start": dates[split_idx],
        "test_end": dates[-1],
        "tickers": sorted(set(syms)),
        "params": {**DEFAULT_XGB_PARAMS, **(model_params or {})},
        "include_market_benchmark": include_market_benchmark,
        "neutral_band": neutral_band,
        "feature_count": len(feature_cols),
        "roc_auc": roc_auc,
        "top_features": [{"name": n, "importance": round(v, 4)} for n, v in top_features],
        "trained_at": datetime.now().isoformat(),
    }

    model_path = MODELS_DIR / f"UNIFIED_{horizon}.joblib"
    meta_path = MODELS_DIR / f"UNIFIED_{horizon}_meta.json"

    # 使用 XGBoost 原生格式保存，更稳定且跨版本兼容
    booster = model.get_booster()
    json_path = MODELS_DIR / f"UNIFIED_{horizon}_xgboost.json"
    booster.save_model(str(json_path))

    # 同时保留 joblib 以保持向后兼容
    joblib.dump(model, model_path)

    meta_path.write_text(json.dumps(meta, indent=2))

    return meta


def predict(symbol: str, horizon: str = "t1") -> dict:
    """Load model and predict direction for the latest trading day."""
    model_path = MODELS_DIR / f"{symbol}_{horizon}.joblib"
    meta_path = MODELS_DIR / f"{symbol}_{horizon}_meta.json"
    json_path = MODELS_DIR / f"{symbol}_{horizon}_xgboost.json"

    # Fall back to unified model if per-ticker model missing
    if not model_path.exists():
        model_path = MODELS_DIR / f"UNIFIED_{horizon}.joblib"
        meta_path = MODELS_DIR / f"UNIFIED_{horizon}_meta.json"
        json_path = MODELS_DIR / f"UNIFIED_{horizon}_xgboost.json"
    if not model_path.exists() and not json_path.exists():
        return {"error": f"No model for {symbol}/{horizon}. Run training first."}

    # 优先使用原生 XGBoost JSON 格式（跨版本兼容）
    from xgboost import XGBClassifier
    if json_path.exists():
        booster = XGBClassifier()
        booster.load_model(str(json_path))
        model = booster
    else:
        model = joblib.load(model_path)

    meta = json.loads(meta_path.read_text())

    model_cols = resolve_feature_cols(model)
    include_market_benchmark = len(model_cols) == len(FEATURE_COLS_WITH_MARKET)

    df = build_features(symbol, include_market_benchmark=include_market_benchmark)
    if df.empty:
        return {"error": f"No feature data for {symbol}"}

    # Use the correct feature columns for this model's expected input
    last_row = df.iloc[-1]
    X = last_row[model_cols].values.reshape(1, -1).astype(np.float64)

    proba = model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])

    # Top feature contributions for this prediction
    importances = model.feature_importances_
    feature_values = {col: float(last_row[col]) for col in model_cols}
    top = sorted(
        zip(model_cols, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "symbol": symbol,
        "horizon": horizon,
        "direction": "up" if pred_class == 1 else "down",
        "confidence": round(confidence, 4),
        "date": str(last_row["trade_date"].date()),
        "top_features": [
            {"name": n, "value": round(feature_values[n], 4), "importance": round(imp, 4)}
            for n, imp in top
        ],
        "model_accuracy": meta["accuracy"],
        "baseline_accuracy": meta["baseline"],
    }
