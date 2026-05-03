"""Comparative experiment: test multiple feature sets, models, and targets."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from backend.ml.features_v2 import (
    build_features_v2,
    FEATURE_COLS,
    FEATURE_COLS_V2_MARKET,
    FEATURE_COLS_V2_CANDLE,
    TextSvdFeatureTransformer,
    load_text_by_date,
)
from backend.ml.walk_forward import run_walk_forward_probabilities


class TextSvdAppendingTransformer:
    """Append fold-local text SVD columns to numeric feature columns."""

    def __init__(self, numeric_cols: list[str]):
        self.numeric_cols = numeric_cols
        self.text_transformer = TextSvdFeatureTransformer()

    def fit(self, rows: pd.DataFrame, y=None):
        self.text_transformer.fit(rows[["trade_date", "text"]], y)
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        numeric = rows[self.numeric_cols].reset_index(drop=True)
        text_features = self.text_transformer.transform(rows[["trade_date", "text"]])
        text_features = text_features.drop(columns=["trade_date"], errors="ignore").reset_index(drop=True)
        if text_features.empty:
            return numeric
        return pd.concat([numeric, text_features], axis=1)


def _expanding_cv(X, y, dates=None, n_folds=5, min_train=200, model_cls=None,
                  model_kwargs=None, transformer_factory=None):
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
        dates=dates or [str(i) for i in range(len(y))],
        model_factory=make_model,
        transformer_factory=transformer_factory,
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


def run_experiment(symbol: str):
    """Run all experiment combinations for a single ticker."""
    print(f"\n{'='*60}")
    print(f"  {symbol} — Comparative Experiment")
    print(f"{'='*60}")

    df = build_features_v2(symbol, use_text=False)
    if df.empty or len(df) < 250:
        print(f"  Not enough data: {len(df)} rows")
        return

    text_by_date = load_text_by_date(symbol)
    if text_by_date.empty:
        df["text"] = ""
    else:
        df = df.merge(text_by_date, on="trade_date", how="left")
        df["text"] = df["text"].fillna("")

    # Define experiments
    feature_sets = {
        "v1_base":   {"cols": FEATURE_COLS, "use_text": False},
        "v2_market": {"cols": FEATURE_COLS_V2_MARKET, "use_text": False},
        "v2_candle": {"cols": FEATURE_COLS_V2_CANDLE, "use_text": False},
        "v2_full":   {"cols": FEATURE_COLS_V2_CANDLE, "use_text": True},
    }

    targets = {
        "direction_t1":     "target_t1",
        "direction_t2":     "target_t2",
        "direction_t3":     "target_t3",
        "direction_t5":     "target_t5",
        "big_move_1pct":    "target_big1_t1",
        "big_move_2pct":    "target_big2_t1",
        "up_big_1pct":      "target_up_big_t1",
        "up_big_3pct_t5":   "target_up_big_t5",
    }

    models = {
        "XGBoost":  (None, None),
        "LogReg":   (LogisticRegression, {"max_iter": 1000, "C": 0.1, "random_state": 42}),
        "RF":       (RandomForestClassifier, {"n_estimators": 200, "max_depth": 6, "random_state": 42}),
    }

    # Run combinations
    results = []
    for target_name, target_col in targets.items():
        sub = df.dropna(subset=[target_col]).reset_index(drop=True)
        if len(sub) < 250:
            continue
        y = sub[target_col].values

        dates = sub["trade_date"].dt.strftime("%Y-%m-%d").tolist()

        for feat_name, feat_spec in feature_sets.items():
            # Only use columns that exist
            valid_cols = [c for c in feat_spec["cols"] if c in sub.columns]
            if feat_spec["use_text"]:
                X = sub[valid_cols + ["trade_date", "text"]].copy()
                transformer_factory = lambda cols=valid_cols: TextSvdAppendingTransformer(cols)
            else:
                X = sub[valid_cols].values.astype(np.float64)
                transformer_factory = None

            for model_name, (model_cls, model_kw) in models.items():
                r = _expanding_cv(
                    X,
                    y,
                    dates=dates,
                    n_folds=5,
                    min_train=200,
                    model_cls=model_cls,
                    model_kwargs=model_kw,
                    transformer_factory=transformer_factory,
                )
                if r is None:
                    continue
                results.append({
                    "target": target_name,
                    "features": feat_name,
                    "model": model_name,
                    **r,
                })

    # Print results sorted by auc then lift
    results.sort(
        key=lambda x: (
            float("-inf") if x["roc_auc"] is None else x["roc_auc"],
            x["lift"],
            x["f1"],
        ),
        reverse=True,
    )

    print(f"\n{'Target':<18} {'Features':<12} {'Model':<8} {'AUC':>6} {'Acc':>6} {'Base':>6} {'Lift':>6} {'F1':>6}")
    print("-" * 82)
    for r in results:
        lift_str = f"{r['lift']:+.1f}pp"
        auc_str = "n/a" if r["roc_auc"] is None else f"{r['roc_auc']:5.3f}"
        print(f"{r['target']:<18} {r['features']:<12} {r['model']:<8} "
              f"{auc_str:>6} {r['accuracy']*100:5.1f}% {r['baseline']*100:5.1f}% {lift_str:>6} "
              f"{r['f1']*100:5.1f}%")

    # Top 5
    print(f"\n  Top 5 by AUC:")
    for i, r in enumerate(results[:5]):
        auc_str = "n/a" if r["roc_auc"] is None else f"{r['roc_auc']:.3f}"
        print(f"  {i+1}. {r['target']} + {r['features']} + {r['model']}: "
              f"auc={auc_str} acc={r['accuracy']*100:.1f}% lift={r['lift']:+.1f}pp f1={r['f1']*100:.1f}%")

    return results


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["NVDA", "AAPL", "TSLA"]
    for sym in tickers:
        run_experiment(sym)
