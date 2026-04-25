"""Single-stock AI trading strategy backtest.

This module converts model probabilities into long/cash trading signals and
computes course-required strategy metrics such as annualized return and maximum
drawdown. Historical strategy scans use expanding-window out-of-sample
predictions to avoid evaluating returns on a model trained with future rows.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "pokieticker.db"
MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
DEFAULT_HORIZONS = ["t1", "t5"]
DEFAULT_FEE_RATE = 0.001
MIN_TRADE_COUNT = 8

FEATURE_COLS = [
    "n_articles", "n_relevant", "n_positive", "n_negative", "n_neutral",
    "sentiment_score", "relevance_ratio", "positive_ratio", "negative_ratio", "has_news",
    "sentiment_score_3d", "sentiment_score_5d", "sentiment_score_10d",
    "positive_ratio_3d", "positive_ratio_5d", "positive_ratio_10d",
    "negative_ratio_3d", "negative_ratio_5d", "negative_ratio_10d",
    "news_count_3d", "news_count_5d", "news_count_10d",
    "sentiment_momentum_3d",
    "ret_1d", "ret_3d", "ret_5d", "ret_10d",
    "volatility_5d", "volatility_10d",
    "volume_ratio_5d", "gap", "ma5_vs_ma20", "rsi_14", "day_of_week",
]


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _threshold_label(threshold: float) -> str:
    return f"{int(round(threshold * 100)):02d}"


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return value


def _iter_records(rows: Any) -> list[dict]:
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return list(rows)


def calculate_annual_return(equity_curve: list[dict]) -> float:
    """Calculate annualized return from trading-day equity observations."""
    if len(equity_curve) < 2:
        return 0.0

    start_equity = float(equity_curve[0]["equity"])
    end_equity = float(equity_curve[-1]["equity"])
    if start_equity <= 0 or end_equity <= 0:
        return 0.0

    periods = max(len(equity_curve) - 1, 1)
    years = periods / 252.0
    return (end_equity / start_equity) ** (1.0 / years) - 1.0


def calculate_max_drawdown(equity_curve: list[dict]) -> float:
    """Calculate maximum drawdown as a positive decimal."""
    peak = 0.0
    max_drawdown = 0.0
    for point in equity_curve:
        equity = float(point["equity"])
        peak = max(peak, equity)
        if peak <= 0:
            continue
        drawdown = (peak - equity) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def simulate_equity_curve(rows: list[dict], fee_rate: float = DEFAULT_FEE_RATE) -> dict:
    """Simulate daily long/cash equity from close prices and binary signals.

    The signal on day N becomes the position for day N to day N+1. Fees are
    charged whenever position changes.
    """
    if len(rows) < 2:
        equity_curve = [
            {"date": r["date"], "equity": 1.0, "position": int(r.get("signal", 0))}
            for r in rows
        ]
        return {
            "equity_curve": equity_curve,
            "trades": [],
            "trade_count": 0,
            "win_rate": 0.0,
            "average_trade_return": 0.0,
            "best_trade_return": 0.0,
            "worst_trade_return": 0.0,
            "cumulative_return": 0.0,
        }

    equity = 1.0
    prev_position = 0
    entry_equity = None
    entry_date = None
    trades = []
    equity_curve = [{"date": rows[0]["date"], "equity": equity, "position": 0}]

    for i in range(1, len(rows)):
        prev = rows[i - 1]
        curr = rows[i]
        desired_position = int(prev.get("signal", 0))

        if desired_position != prev_position:
            equity *= 1.0 - fee_rate
            if desired_position == 1:
                entry_equity = equity
                entry_date = prev["date"]
            elif prev_position == 1 and entry_equity is not None:
                trade_return = equity / entry_equity - 1.0
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": prev["date"],
                    "return": trade_return,
                })
                entry_equity = None
                entry_date = None
            prev_position = desired_position

        prev_close = float(prev["close"])
        curr_close = float(curr["close"])
        daily_return = curr_close / prev_close - 1.0 if prev_close > 0 else 0.0
        if prev_position == 1:
            equity *= 1.0 + daily_return

        equity_curve.append({
            "date": curr["date"],
            "equity": equity,
            "position": prev_position,
        })

    if prev_position == 1 and entry_equity is not None:
        equity *= 1.0 - fee_rate
        trade_return = equity / entry_equity - 1.0
        trades.append({
            "entry_date": entry_date,
            "exit_date": rows[-1]["date"],
            "return": trade_return,
        })
        equity_curve[-1]["equity"] = equity

    trade_returns = [float(t["return"]) for t in trades]
    wins = [r for r in trade_returns if r > 0]

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trade_returns) if trade_returns else 0.0,
        "average_trade_return": sum(trade_returns) / len(trade_returns) if trade_returns else 0.0,
        "best_trade_return": max(trade_returns) if trade_returns else 0.0,
        "worst_trade_return": min(trade_returns) if trade_returns else 0.0,
        "cumulative_return": equity_curve[-1]["equity"] - 1.0 if equity_curve else 0.0,
    }


def choose_best_strategy(candidates: list[dict]) -> dict:
    """Choose the best candidate for course reporting."""
    if not candidates:
        return {"error": "No strategy candidates"}

    passing = [
        c for c in candidates
        if c.get("meets_course_target") and int(c.get("trade_count", 0)) >= MIN_TRADE_COUNT
    ]
    if passing:
        return sorted(
            passing,
            key=lambda c: (
                float(c.get("annual_return", 0.0)),
                -float(c.get("max_drawdown", 1.0)),
                int(c.get("trade_count", 0)),
            ),
            reverse=True,
        )[0]

    def closest_score(candidate: dict) -> tuple:
        annual_return = float(candidate.get("annual_return", 0.0))
        max_drawdown = float(candidate.get("max_drawdown", 1.0))
        trade_count = int(candidate.get("trade_count", 0))
        annual_gap = min(annual_return - 0.20, 0.0)
        drawdown_gap = min(0.20 - max_drawdown, 0.0)
        trade_penalty = 0.0 if trade_count >= MIN_TRADE_COUNT else -0.25
        return (annual_gap + drawdown_gap + trade_penalty, annual_return, -max_drawdown, trade_count)

    return sorted(candidates, key=closest_score, reverse=True)[0]


def build_strategy_result_from_predictions(
    symbol: str,
    horizon: str,
    threshold: float,
    fee_rate: float,
    df: Any,
    up_probabilities: np.ndarray,
) -> dict:
    """Build a strategy result from feature rows and up probabilities."""
    records = _iter_records(df)
    if not records or len(records) != len(up_probabilities):
        return {"error": "Feature rows and probabilities do not align"}

    rows = []
    for i, row in enumerate(records):
        prob = float(up_probabilities[i])
        trade_date = row["trade_date"]
        date_text = trade_date.strftime("%Y-%m-%d") if hasattr(trade_date, "strftime") else str(trade_date)
        rows.append({
            "date": date_text,
            "close": float(row["close"]),
            "prob_up": prob,
            "signal": 1 if prob >= threshold else 0,
        })

    simulation = simulate_equity_curve(rows, fee_rate=fee_rate)
    equity_curve = simulation["equity_curve"]
    annual_return = calculate_annual_return(equity_curve)
    max_drawdown = calculate_max_drawdown(equity_curve)
    buy_hold_return = 0.0
    if len(rows) >= 2 and rows[0]["close"] > 0:
        buy_hold_return = rows[-1]["close"] / rows[0]["close"] - 1.0

    result = {
        "symbol": symbol,
        "horizon": horizon,
        "threshold": float(threshold),
        "fee_rate": float(fee_rate),
        "start_date": rows[0]["date"],
        "end_date": rows[-1]["date"],
        "annual_return": _round_metric(annual_return),
        "max_drawdown": _round_metric(max_drawdown),
        "cumulative_return": _round_metric(simulation["cumulative_return"]),
        "buy_hold_return": _round_metric(buy_hold_return),
        "win_rate": _round_metric(simulation["win_rate"]),
        "trade_count": int(simulation["trade_count"]),
        "average_trade_return": _round_metric(simulation["average_trade_return"]),
        "best_trade_return": _round_metric(simulation["best_trade_return"]),
        "worst_trade_return": _round_metric(simulation["worst_trade_return"]),
        "meets_annual_return_target": annual_return > 0.20,
        "meets_drawdown_target": max_drawdown < 0.20,
        "equity_curve": equity_curve,
        "trades": simulation["trades"],
    }
    result["meets_course_target"] = (
        result["meets_annual_return_target"] and result["meets_drawdown_target"]
    )
    return result


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_news_features(symbol: str) -> Any:
    import pandas as pd

    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT na.trade_date,
               COUNT(*) AS n_articles,
               SUM(CASE WHEN l1.relevance IN ('high','medium') THEN 1 ELSE 0 END) AS n_relevant,
               SUM(CASE WHEN l1.sentiment = 'positive' THEN 1 ELSE 0 END) AS n_positive,
               SUM(CASE WHEN l1.sentiment = 'negative' THEN 1 ELSE 0 END) AS n_negative,
               SUM(CASE WHEN l1.sentiment = 'neutral' THEN 1 ELSE 0 END) AS n_neutral
        FROM news_aligned na
        JOIN layer1_results l1 ON na.news_id = l1.news_id AND na.symbol = l1.symbol
        WHERE na.symbol = ?
        GROUP BY na.trade_date
        ORDER BY na.trade_date
        """,
        (symbol,),
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    total = df["n_articles"].clip(lower=1)
    df["sentiment_score"] = (df["n_positive"] - df["n_negative"]) / total
    df["relevance_ratio"] = df["n_relevant"] / total
    df["positive_ratio"] = df["n_positive"] / total
    df["negative_ratio"] = df["n_negative"] / total
    df["has_news"] = 1
    return df


def _load_ohlc(symbol: str) -> Any:
    import pandas as pd

    conn = _get_conn()
    rows = conn.execute(
        "SELECT date, open, high, low, close, volume FROM ohlc WHERE symbol = ? ORDER BY date",
        (symbol,),
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_strategy_features(symbol: str) -> Any:
    """Build the same daily features used by the existing XGBoost pipeline."""
    import pandas as pd

    ohlc = _load_ohlc(symbol)
    if ohlc.empty or len(ohlc) < 30:
        return pd.DataFrame()

    news = _load_news_features(symbol)
    df = ohlc.rename(columns={"date": "trade_date"})
    if not news.empty:
        df = df.merge(news, on="trade_date", how="left")
    else:
        for col in ["n_articles", "n_relevant", "n_positive", "n_negative",
                    "n_neutral", "sentiment_score", "relevance_ratio",
                    "positive_ratio", "negative_ratio", "has_news"]:
            df[col] = 0

    news_cols = ["n_articles", "n_relevant", "n_positive", "n_negative",
                 "n_neutral", "sentiment_score", "relevance_ratio",
                 "positive_ratio", "negative_ratio", "has_news"]
    df[news_cols] = df[news_cols].fillna(0)

    for w in [3, 5, 10]:
        df[f"sentiment_score_{w}d"] = df["sentiment_score"].rolling(w, min_periods=1).mean()
        df[f"positive_ratio_{w}d"] = df["positive_ratio"].rolling(w, min_periods=1).mean()
        df[f"negative_ratio_{w}d"] = df["negative_ratio"].rolling(w, min_periods=1).mean()
        df[f"news_count_{w}d"] = df["n_articles"].rolling(w, min_periods=1).sum()
    df["sentiment_momentum_3d"] = df["sentiment_score_3d"] - df["sentiment_score_10d"]

    close = df["close"]
    df["ret_1d"] = close.pct_change(1).shift(1)
    df["ret_3d"] = close.pct_change(3).shift(1)
    df["ret_5d"] = close.pct_change(5).shift(1)
    df["ret_10d"] = close.pct_change(10).shift(1)
    df["volatility_5d"] = close.pct_change().rolling(5).std().shift(1)
    df["volatility_10d"] = close.pct_change().rolling(10).std().shift(1)

    avg_vol_5 = df["volume"].rolling(5).mean().shift(1)
    df["volume_ratio_5d"] = df["volume"].shift(1) / avg_vol_5.clip(lower=1)
    df["gap"] = (df["open"] / close.shift(1) - 1).shift(1)

    ma5 = close.rolling(5).mean().shift(1)
    ma20 = close.rolling(20).mean().shift(1)
    df["ma5_vs_ma20"] = ma5 / ma20.clip(lower=0.01) - 1

    delta = close.diff().shift(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.clip(lower=1e-10)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    df["day_of_week"] = df["trade_date"].dt.dayofweek

    df["target_t1"] = (close.shift(-1) > close).astype(int)
    df["target_t2"] = (close.shift(-2) > close).astype(int)
    df["target_t3"] = (close.shift(-3) > close).astype(int)
    df["target_t5"] = (close.shift(-5) > close).astype(int)

    return df.dropna(subset=["ret_10d", "rsi_14"]).reset_index(drop=True)


def load_stored_oos_predictions(symbol: str, horizon: str) -> tuple[list[dict], np.ndarray]:
    """Load existing expanding-window predictions saved by backend.ml.backtest."""
    path = MODELS_DIR / f"{symbol}_{horizon}_backtest.json"
    if not path.exists():
        return [], np.array([])

    data = json.loads(path.read_text(encoding="utf-8"))
    predictions = data.get("daily_predictions") or []
    if not predictions:
        return [], np.array([])

    conn = _get_conn()
    rows = conn.execute(
        "SELECT date, close FROM ohlc WHERE symbol = ? ORDER BY date",
        (symbol,),
    ).fetchall()
    conn.close()
    close_by_date = {r["date"]: float(r["close"]) for r in rows}

    result_rows = []
    probabilities = []
    for item in predictions:
        date = item.get("date")
        if date not in close_by_date:
            continue
        predicted = int(item.get("predicted", 0))
        result_rows.append({
            "trade_date": datetime.strptime(date, "%Y-%m-%d"),
            "close": close_by_date[date],
        })
        probabilities.append(float(predicted))

    return result_rows, np.array(probabilities)


def generate_oos_probabilities(
    df: Any,
    horizon: str,
    n_folds: int = 5,
    min_train: int = 200,
) -> tuple[Any, np.ndarray]:
    """Generate expanding-window out-of-sample up probabilities."""
    from xgboost import XGBClassifier

    target_col = f"target_{horizon}"
    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    n = len(work)
    if n < min_train + 20:
        return work.iloc[0:0], np.array([])

    test_size = (n - min_train) // n_folds
    if test_size < 10:
        n_folds = max(1, (n - min_train) // 10)
        test_size = (n - min_train) // n_folds

    frames = []
    probs = []
    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_end = train_end + test_size if fold < n_folds - 1 else n
        train_df = work.iloc[:train_end]
        test_df = work.iloc[train_end:test_end]
        if test_df.empty:
            continue

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
        model.fit(X_train, y_train, verbose=False)
        fold_probs = model.predict_proba(X_test)[:, 1]
        frames.append(test_df)
        probs.extend(fold_probs.tolist())

    if not frames:
        return work.iloc[0:0], np.array([])

    import pandas as pd

    return pd.concat(frames, ignore_index=True), np.array(probs)


def run_strategy_backtest(
    symbol: str,
    horizon: str = "t5",
    threshold: float = 0.60,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> dict:
    """Run out-of-sample strategy backtest for one symbol/horizon/threshold."""
    symbol = symbol.upper()
    df = build_strategy_features(symbol)
    target_col = f"target_{horizon}"
    if df.empty or target_col not in df.columns:
        return {"error": f"No feature data for {symbol}/{horizon}"}

    if len(df.dropna(subset=[target_col])) < 220:
        return {"error": f"Not enough feature rows for {symbol}/{horizon}"}

    prediction_source = "expanding_window_xgboost"
    try:
        oos_df, up_probabilities = generate_oos_probabilities(df, horizon)
    except Exception as exc:
        oos_df, up_probabilities = load_stored_oos_predictions(symbol, horizon)
        prediction_source = f"stored_backtest_predictions:{exc.__class__.__name__}"

    is_empty = oos_df.empty if hasattr(oos_df, "empty") else len(oos_df) == 0
    if is_empty or len(up_probabilities) == 0:
        oos_df, up_probabilities = load_stored_oos_predictions(symbol, horizon)
        prediction_source = "stored_backtest_predictions"

    is_empty = oos_df.empty if hasattr(oos_df, "empty") else len(oos_df) == 0
    if is_empty or len(up_probabilities) == 0:
        return {"error": f"Unable to generate out-of-sample probabilities for {symbol}/{horizon}"}

    result = build_strategy_result_from_predictions(
        symbol=symbol,
        horizon=horizon,
        threshold=threshold,
        fee_rate=fee_rate,
        df=oos_df,
        up_probabilities=up_probabilities,
    )
    result["oos_method"] = "expanding_window"
    result["prediction_source"] = prediction_source
    result["min_train"] = 200

    if "error" not in result:
        MODELS_DIR.mkdir(exist_ok=True)
        threshold_label = _threshold_label(threshold)
        out_path = MODELS_DIR / f"{symbol}_{horizon}_thr{threshold_label}_strategy_backtest.json"
        out_path.write_text(json.dumps(result, indent=2, default=_to_jsonable), encoding="utf-8")

    return result


def _discover_symbols() -> list[str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM ohlc GROUP BY symbol HAVING COUNT(*) >= 220 ORDER BY symbol"
    ).fetchall()
    conn.close()
    return [r["symbol"] for r in rows]


def scan_strategy_space(
    symbols: list[str] | None = None,
    horizons: list[str] | None = None,
    thresholds: list[float] | None = None,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> dict:
    """Scan symbol/horizon/threshold combinations and save the best result."""
    symbols = symbols or _discover_symbols()
    horizons = horizons or DEFAULT_HORIZONS
    thresholds = thresholds or DEFAULT_THRESHOLDS

    candidates = []
    errors = []
    for symbol in symbols:
        for horizon in horizons:
            for threshold in thresholds:
                result = run_strategy_backtest(symbol, horizon, threshold, fee_rate)
                if "error" in result:
                    errors.append({
                        "symbol": symbol,
                        "horizon": horizon,
                        "threshold": threshold,
                        "error": result["error"],
                    })
                    continue
                candidates.append(result)

    best = choose_best_strategy(candidates)
    output = {
        "best": best,
        "candidate_count": len(candidates),
        "error_count": len(errors),
        "passing_count": sum(1 for c in candidates if c.get("meets_course_target")),
        "errors": errors[:50],
        "top_candidates": sorted(
            candidates,
            key=lambda c: (
                bool(c.get("meets_course_target")),
                float(c.get("annual_return", 0.0)),
                -float(c.get("max_drawdown", 1.0)),
            ),
            reverse=True,
        )[:20],
    }

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / "strategy_best.json"
    out_path.write_text(json.dumps(output, indent=2, default=_to_jsonable), encoding="utf-8")
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run AI strategy return backtests")
    parser.add_argument("--symbol", type=str, help="Run a single symbol")
    parser.add_argument("--horizon", type=str, default="t5", choices=DEFAULT_HORIZONS)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--scan", action="store_true", help="Scan all available symbols")
    args = parser.parse_args()

    if args.scan:
        result = scan_strategy_space(fee_rate=args.fee_rate)
        best = result.get("best", {})
        print(json.dumps({
            "candidate_count": result["candidate_count"],
            "passing_count": result["passing_count"],
            "best_symbol": best.get("symbol"),
            "best_horizon": best.get("horizon"),
            "best_threshold": best.get("threshold"),
            "annual_return": best.get("annual_return"),
            "max_drawdown": best.get("max_drawdown"),
            "meets_course_target": best.get("meets_course_target"),
        }, indent=2))
        return

    if not args.symbol:
        parser.error("--symbol is required unless --scan is used")

    result = run_strategy_backtest(
        args.symbol,
        horizon=args.horizon,
        threshold=args.threshold,
        fee_rate=args.fee_rate,
    )
    print(json.dumps(result, indent=2, default=_to_jsonable))


if __name__ == "__main__":
    main()
