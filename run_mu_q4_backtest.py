"""Q4 2025 strategy backtest for MU.

Train on data up to 2025-09-30, predict Q4 2025 (Oct-Dec),
and compute strategy metrics for multiple thresholds.
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.strategy_backtest import (
    build_strategy_features,
    simulate_equity_curve,
    calculate_annual_return,
    calculate_max_drawdown,
    FEATURE_COLS,
)
from xgboost import XGBClassifier

CUTOFF_DATE = "2025-09-30"
START_DATE = "2025-10-01"
END_DATE = "2025-12-31"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
HORIZONS = ["t1", "t5"]
FEE_RATE = 0.001
SLIPPAGE = 0.001


def run_q4_backtest():
    symbol = "MU"
    print(f"=== {symbol} Q4 2025 Strategy Backtest ===")
    print(f"Training cutoff: {CUTOFF_DATE}")
    print(f"Test period: {START_DATE} ~ {END_DATE}")
    print()

    # Build features
    df = build_strategy_features(symbol)
    if df.empty:
        print("ERROR: No feature data for MU")
        return

    print(f"Total feature rows: {len(df)}")
    print(f"Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print()

    # Ensure trade_date is datetime
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    results = {}

    for horizon in HORIZONS:
        target_col = f"target_{horizon}"
        if target_col not in df.columns:
            print(f"  {horizon}: target column not found, skipping")
            continue

        work = df.dropna(subset=[target_col]).copy()
        work = work.sort_values("trade_date").reset_index(drop=True)

        # Split by date: train up to CUTOFF_DATE, test in Q4 2025
        cutoff = pd.Timestamp(CUTOFF_DATE)
        start = pd.Timestamp(START_DATE)
        end = pd.Timestamp(END_DATE)

        train_df = work[work["trade_date"] <= cutoff]
        test_df = work[(work["trade_date"] >= start) & (work["trade_date"] <= end)]

        if train_df.empty or test_df.empty:
            print(f"  {horizon}: train={len(train_df)}, test={len(test_df)} - insufficient data")
            continue

        print(f"--- Horizon: {horizon} ---")
        print(f"  Train rows: {len(train_df)} (up to {CUTOFF_DATE})")
        print(f"  Test rows:  {len(test_df)} ({START_DATE} ~ {END_DATE})")

        # Train XGBoost
        X_train = train_df[FEATURE_COLS].values.astype(np.float64)
        y_train = train_df[target_col].values.astype(int)
        X_test = test_df[FEATURE_COLS].values.astype(np.float64)
        np.nan_to_num(X_train, copy=False)
        np.nan_to_num(X_test, copy=False)

        model = XGBClassifier(
            max_depth=4,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )

        # Use early stopping with a validation tail
        if len(X_train) >= 80:
            val_size = max(10, min(len(X_train) // 5, 40))
            split_idx = len(X_train) - val_size
            model.fit(
                X_train[:split_idx], y_train[:split_idx],
                eval_set=[(X_train[split_idx:], y_train[split_idx:])],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, verbose=False)

        # Predict probabilities
        up_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (up_probs >= 0.5).astype(int)
        y_true = test_df[target_col].values

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, up_probs)
        except ValueError:
            auc = None
        baseline = max(y_true.mean(), 1 - y_true.mean())

        print(f"  Model accuracy: {acc:.1%} (baseline: {baseline:.1%}, lift: {(acc-baseline)*100:+.1f}pp)")
        print(f"  Precision: {prec:.1%}, Recall: {rec:.1%}, F1: {f1:.1%}")
        if auc is not None:
            print(f"  AUC: {auc:.4f}")
        print()

        # Strategy simulation for each threshold
        results[horizon] = {"model": {"accuracy": round(acc, 4), "baseline": round(baseline, 4),
                                       "lift_pp": round((acc - baseline) * 100, 1),
                                       "precision": round(prec, 4), "recall": round(rec, 4),
                                       "f1": round(f1, 4), "roc_auc": round(auc, 4) if auc else None},
                            "strategies": {}}

        for threshold in THRESHOLDS:
            rows = []
            for i, (_, row) in enumerate(test_df.iterrows()):
                prob = float(up_probs[i])
                trade_date = row["trade_date"]
                date_text = trade_date.strftime("%Y-%m-%d") if hasattr(trade_date, "strftime") else str(trade_date)
                rows.append({
                    "date": date_text,
                    "close": float(row["close"]),
                    "prob_up": prob,
                    "signal": 1 if prob >= threshold else 0,
                })

            sim = simulate_equity_curve(rows, fee_rate=FEE_RATE, slippage=SLIPPAGE)
            equity_curve = sim["equity_curve"]
            annual_return = calculate_annual_return(equity_curve)
            max_drawdown = calculate_max_drawdown(equity_curve)

            buy_hold = 0.0
            if len(rows) >= 2 and rows[0]["close"] > 0:
                buy_hold = rows[-1]["close"] / rows[0]["close"] - 1.0

            strat = {
                "threshold": threshold,
                "annual_return": round(annual_return, 6),
                "max_drawdown": round(max_drawdown, 6),
                "cumulative_return": round(sim["cumulative_return"], 6),
                "buy_hold_return": round(buy_hold, 6),
                "trade_count": sim["trade_count"],
                "win_rate": round(sim["win_rate"], 4),
                "meets_annual_return_target": annual_return > 0.20,
                "meets_drawdown_target": max_drawdown < 0.20,
            }
            strat["meets_course_target"] = (
                strat["meets_annual_return_target"] and strat["meets_drawdown_target"]
            )

            results[horizon]["strategies"][str(threshold)] = strat

            print(f"  Threshold {threshold:.2f}: "
                  f"cum_ret={sim['cumulative_return']:+.2%} "
                  f"ann_ret={annual_return:+.2%} "
                  f"max_dd={max_drawdown:.2%} "
                  f"trades={sim['trade_count']} "
                  f"win_rate={sim['win_rate']:.1%} "
                  f"buy_hold={buy_hold:+.2%} "
                  f"{'✓' if strat['meets_course_target'] else '✗'}")

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find best strategy
    best = None
    best_score = -999
    for horizon, data in results.items():
        for thr_str, strat in data["strategies"].items():
            score = strat["annual_return"] - strat["max_drawdown"]
            if score > best_score:
                best_score = score
                best = {"horizon": horizon, **strat}

    if best:
        print(f"\nBest strategy: {best['horizon']} / threshold {best['threshold']}")
        print(f"  Annual return: {best['annual_return']:+.2%}")
        print(f"  Max drawdown:  {best['max_drawdown']:.2%}")
        print(f"  Cumulative return: {best['cumulative_return']:+.2%}")
        print(f"  Buy & hold return: {best['buy_hold_return']:+.2%}")
        print(f"  Trade count: {best['trade_count']}")
        print(f"  Win rate: {best['win_rate']:.1%}")
        print(f"  Meets course target: {'YES' if best['meets_course_target'] else 'NO'}")

    # Save results
    MODELS_DIR = PROJECT_ROOT / "backend" / "ml" / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / "MU_q4_2025_strategy_backtest.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    import pandas as pd
    run_q4_backtest()
