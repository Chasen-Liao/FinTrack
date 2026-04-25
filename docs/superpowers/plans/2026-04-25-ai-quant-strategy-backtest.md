# AI Quant Strategy Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real single-stock AI strategy backtest that scans existing symbols and outputs annualized return, maximum drawdown, equity curve, and course-target pass/fail status.

**Architecture:** Add a focused backend module `backend/ml/strategy_backtest.py` that reuses the existing database, feature builder, and XGBoost configuration. Strategy results must be based on expanding-window out-of-sample probabilities, not predictions from a model trained on the full history. Keep strategy return backtesting separate from the existing classification backtest so current prediction APIs remain stable. Add tests for metric math and selection logic before implementation.

**Tech Stack:** Python, pandas, numpy, xgboost, pytest, SQLite through the existing `backend.database` helper.

---

## File Structure

- Create `backend/ml/strategy_backtest.py`
  - Owns strategy backtest data models, metric calculations, out-of-sample probability generation, per-symbol backtest, scanning, JSON output, and CLI.
- Create `tests/ml/test_strategy_backtest.py`
  - Unit tests for annualized return, maximum drawdown, no-trade handling, fee impact, and best-candidate selection.
- Optionally modify `backend/api/routers/predict.py`
  - Expose saved JSON results through API endpoints after the CLI works.
- Optionally modify `requirements.txt`
  - Add `pytest` if the local environment cannot run `python -m pytest`.
- Do not modify `backend/ml/backtest.py`
  - It remains the classification accuracy backtest.

## Prerequisite Checks

- [ ] **Step 1: Confirm pytest is available**

Run:

```bash
python -m pytest --version
```

Expected:

```text
pytest <version>
```

If this fails with `No module named pytest`, add this line to `requirements.txt`:

```text
pytest>=8.0.0
```

Then install project dependencies in the active virtual environment:

```bash
pip install -r requirements.txt
```

- [ ] **Step 2: Create the test directory**

Create this directory before adding tests:

```text
tests/ml
```

## Task 1: Add Metric Calculation Tests

**Files:**
- Create: `tests/ml/test_strategy_backtest.py`
- Create later: `backend/ml/strategy_backtest.py`

- [ ] **Step 1: Write failing tests for annualized return and maximum drawdown**

Create `tests/ml/test_strategy_backtest.py` with:

```python
import math

from backend.ml.strategy_backtest import calculate_annual_return, calculate_max_drawdown


def test_calculate_annual_return_from_equity_curve():
    equity_curve = [
        {"date": "2024-01-02", "equity": 1.0},
        {"date": "2024-07-01", "equity": 1.10},
        {"date": "2024-12-31", "equity": 1.21},
    ]

    result = calculate_annual_return(equity_curve)

    assert result > 0.20
    assert result < 0.22


def test_calculate_max_drawdown_from_equity_curve():
    equity_curve = [
        {"date": "2024-01-02", "equity": 1.00},
        {"date": "2024-01-03", "equity": 1.20},
        {"date": "2024-01-04", "equity": 0.90},
        {"date": "2024-01-05", "equity": 1.10},
    ]

    result = calculate_max_drawdown(equity_curve)

    assert math.isclose(result, 0.25, rel_tol=1e-9)
```

- [ ] **Step 2: Run tests and verify they fail because the module does not exist**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'backend.ml.strategy_backtest'
```

- [ ] **Step 3: Create minimal metric functions**

Create `backend/ml/strategy_backtest.py` with:

```python
"""Single-stock AI trading strategy backtest.

This module converts existing model predictions into long/cash trading signals
and computes course-required strategy metrics such as annualized return and
maximum drawdown.
"""

from __future__ import annotations

from datetime import datetime


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def calculate_annual_return(equity_curve: list[dict]) -> float:
    """Calculate annualized return from trading-day equity observations.

    Returns 0.0 when there is not enough data or the initial equity is invalid.
    The annualization basis is 252 trading days, which is the common reporting
    convention for strategy backtests.
    """
    if len(equity_curve) < 2:
        return 0.0

    start = equity_curve[0]
    end = equity_curve[-1]
    start_equity = float(start["equity"])
    end_equity = float(end["equity"])
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
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit metric foundation**

Run:

```bash
git add backend/ml/strategy_backtest.py tests/ml/test_strategy_backtest.py
git commit -m "test: add strategy backtest metric foundation"
```

Expected:

```text
[branch commit] test: add strategy backtest metric foundation
```

## Task 2: Add Strategy Simulation Tests

**Files:**
- Modify: `tests/ml/test_strategy_backtest.py`
- Modify: `backend/ml/strategy_backtest.py`

- [ ] **Step 1: Add failing tests for no-trade and fee impact behavior**

Append to `tests/ml/test_strategy_backtest.py`:

```python
from backend.ml.strategy_backtest import simulate_equity_curve


def test_simulate_equity_curve_handles_no_trades():
    rows = [
        {"date": "2024-01-02", "close": 100.0, "signal": 0},
        {"date": "2024-01-03", "close": 110.0, "signal": 0},
        {"date": "2024-01-04", "close": 105.0, "signal": 0},
    ]

    result = simulate_equity_curve(rows, fee_rate=0.001)

    assert result["cumulative_return"] == 0.0
    assert result["trade_count"] == 0
    assert result["equity_curve"][-1]["equity"] == 1.0
    assert result["trades"] == []


def test_simulate_equity_curve_applies_transaction_fees():
    rows = [
        {"date": "2024-01-02", "close": 100.0, "signal": 0},
        {"date": "2024-01-03", "close": 100.0, "signal": 1},
        {"date": "2024-01-04", "close": 110.0, "signal": 1},
        {"date": "2024-01-05", "close": 110.0, "signal": 0},
    ]

    no_fee = simulate_equity_curve(rows, fee_rate=0.0)
    with_fee = simulate_equity_curve(rows, fee_rate=0.001)

    assert no_fee["cumulative_return"] > with_fee["cumulative_return"]
    assert with_fee["trade_count"] == 1
```

- [ ] **Step 2: Run tests and verify the new tests fail**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
ImportError: cannot import name 'simulate_equity_curve'
```

- [ ] **Step 3: Implement `simulate_equity_curve`**

Add this code to `backend/ml/strategy_backtest.py`:

```python
def simulate_equity_curve(rows: list[dict], fee_rate: float = 0.001) -> dict:
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
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit strategy simulation**

Run:

```bash
git add backend/ml/strategy_backtest.py tests/ml/test_strategy_backtest.py
git commit -m "feat: simulate long cash strategy equity"
```

Expected:

```text
[branch commit] feat: simulate long cash strategy equity
```

## Task 3: Add Candidate Selection Tests

**Files:**
- Modify: `tests/ml/test_strategy_backtest.py`
- Modify: `backend/ml/strategy_backtest.py`

- [ ] **Step 1: Add failing tests for best-candidate selection**

Append to `tests/ml/test_strategy_backtest.py`:

```python
from backend.ml.strategy_backtest import choose_best_strategy


def test_choose_best_strategy_prefers_course_passing_candidate_with_enough_trades():
    candidates = [
        {
            "symbol": "AAA",
            "horizon": "t1",
            "threshold": 0.5,
            "annual_return": 0.80,
            "max_drawdown": 0.05,
            "trade_count": 1,
            "meets_course_target": True,
        },
        {
            "symbol": "BBB",
            "horizon": "t5",
            "threshold": 0.6,
            "annual_return": 0.25,
            "max_drawdown": 0.12,
            "trade_count": 18,
            "meets_course_target": True,
        },
    ]

    result = choose_best_strategy(candidates)

    assert result["symbol"] == "BBB"


def test_choose_best_strategy_returns_closest_when_none_pass():
    candidates = [
        {
            "symbol": "AAA",
            "horizon": "t1",
            "threshold": 0.5,
            "annual_return": 0.18,
            "max_drawdown": 0.12,
            "trade_count": 20,
            "meets_course_target": False,
        },
        {
            "symbol": "BBB",
            "horizon": "t5",
            "threshold": 0.6,
            "annual_return": 0.10,
            "max_drawdown": 0.10,
            "trade_count": 18,
            "meets_course_target": False,
        },
    ]

    result = choose_best_strategy(candidates)

    assert result["symbol"] == "AAA"
    assert result["meets_course_target"] is False
```

- [ ] **Step 2: Run tests and verify the new tests fail**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
ImportError: cannot import name 'choose_best_strategy'
```

- [ ] **Step 3: Implement `choose_best_strategy`**

Add this code to `backend/ml/strategy_backtest.py`:

```python
MIN_TRADE_COUNT = 8


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
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Commit selection logic**

Run:

```bash
git add backend/ml/strategy_backtest.py tests/ml/test_strategy_backtest.py
git commit -m "feat: select best course strategy candidate"
```

Expected:

```text
[branch commit] feat: select best course strategy candidate
```

## Task 4: Implement Per-Symbol Strategy Backtest

**Files:**
- Modify: `backend/ml/strategy_backtest.py`
- Modify: `tests/ml/test_strategy_backtest.py`

- [ ] **Step 1: Add a failing integration-shaped test with a fake model**

Append to `tests/ml/test_strategy_backtest.py`:

```python
import numpy as np
import pandas as pd

from backend.ml.strategy_backtest import build_strategy_result_from_predictions


def test_build_strategy_result_from_predictions_marks_course_target():
    df = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=6, freq="D"),
        "close": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
    })
    probabilities = np.array([0.80, 0.80, 0.80, 0.80, 0.80, 0.80])

    result = build_strategy_result_from_predictions(
        symbol="UP",
        horizon="t1",
        threshold=0.70,
        fee_rate=0.0,
        df=df,
        up_probabilities=probabilities,
    )

    assert result["symbol"] == "UP"
    assert result["horizon"] == "t1"
    assert result["threshold"] == 0.70
    assert result["cumulative_return"] > 0.09
    assert result["max_drawdown"] == 0.0
    assert result["meets_drawdown_target"] is True
    assert result["meets_annual_return_target"] is True
    assert result["meets_course_target"] is True
```

- [ ] **Step 2: Run tests and verify the new test fails**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
ImportError: cannot import name 'build_strategy_result_from_predictions'
```

- [ ] **Step 3: Implement result construction from predictions**

Add imports near the top of `backend/ml/strategy_backtest.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from backend.database import get_conn
from backend.ml.features import FEATURE_COLS, build_features
```

Add constants and helper code:

```python
MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
DEFAULT_HORIZONS = ["t1", "t5"]
DEFAULT_FEE_RATE = 0.001


def _to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _threshold_label(threshold: float) -> str:
    return f"{int(round(threshold * 100)):02d}"
```

Add:

```python
def build_strategy_result_from_predictions(
    symbol: str,
    horizon: str,
    threshold: float,
    fee_rate: float,
    df: pd.DataFrame,
    up_probabilities: np.ndarray,
) -> dict:
    """Build a strategy result from feature rows and up probabilities."""
    if df.empty or len(df) != len(up_probabilities):
        return {"error": "Feature rows and probabilities do not align"}

    rows = []
    work = df.reset_index(drop=True).copy()
    for i, row in work.iterrows():
        prob = float(up_probabilities[i])
        trade_date = row["trade_date"]
        if hasattr(trade_date, "strftime"):
            date_text = trade_date.strftime("%Y-%m-%d")
        else:
            date_text = str(trade_date)
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
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
7 passed
```

- [ ] **Step 5: Implement out-of-sample per-symbol backtest**

Add:

```python
def generate_oos_probabilities(
    df: pd.DataFrame,
    horizon: str,
    n_folds: int = 5,
    min_train: int = 200,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate expanding-window out-of-sample up probabilities.

    Each probability is produced by a model trained only on rows before the
    prediction date. This prevents sample-in backtest metrics.
    """
    target_col = f"target_{horizon}"
    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    n = len(work)
    if n < min_train + 20:
        return pd.DataFrame(), np.array([])

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
        return pd.DataFrame(), np.array([])
    return pd.concat(frames, ignore_index=True), np.array(probs)


def run_strategy_backtest(
    symbol: str,
    horizon: str = "t5",
    threshold: float = 0.60,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> dict:
    """Run out-of-sample strategy backtest for one symbol/horizon/threshold."""
    symbol = symbol.upper()

    df = build_features(symbol)
    target_col = f"target_{horizon}"
    if df.empty or target_col not in df.columns:
        return {"error": f"No feature data for {symbol}/{horizon}"}

    if len(df.dropna(subset=[target_col])) < 220:
        return {"error": f"Not enough feature rows for {symbol}/{horizon}"}

    oos_df, up_probabilities = generate_oos_probabilities(df, horizon)
    if oos_df.empty or len(up_probabilities) == 0:
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
    result["min_train"] = 200

    if "error" not in result:
        MODELS_DIR.mkdir(exist_ok=True)
        threshold_label = _threshold_label(threshold)
        out_path = MODELS_DIR / f"{symbol}_{horizon}_thr{threshold_label}_strategy_backtest.json"
        out_path.write_text(json.dumps(result, indent=2, default=_to_jsonable), encoding="utf-8")

    return result
```

- [ ] **Step 6: Run one real out-of-sample symbol backtest**

Run:

```bash
python -c "from backend.ml.strategy_backtest import run_strategy_backtest; import json; print(json.dumps(run_strategy_backtest('TSLA','t5',0.6), indent=2)[:2000])"
```

Expected:

```text
{
  "symbol": "TSLA",
  "horizon": "t5",
  "oos_method": "expanding_window",
  ...
}
```

- [ ] **Step 7: Commit per-symbol backtest**

Run:

```bash
git add backend/ml/strategy_backtest.py tests/ml/test_strategy_backtest.py backend/ml/models/TSLA_t5_thr60_strategy_backtest.json
git commit -m "feat: backtest model predictions as trading strategy"
```

Expected:

```text
[branch commit] feat: backtest model predictions as trading strategy
```

## Task 5: Implement Strategy Scanner and CLI

**Files:**
- Modify: `backend/ml/strategy_backtest.py`

- [ ] **Step 1: Add symbol discovery and scan functions**

Add:

```python
def _discover_symbols() -> list[str]:
    conn = get_conn()
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
```

- [ ] **Step 2: Add CLI**

Add:

```python
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run AI strategy return backtests")
    parser.add_argument("--symbol", type=str, help="Run a single symbol")
    parser.add_argument("--horizon", type=str, default="t5", choices=DEFAULT_HORIZONS)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--scan", action="store_true", help="Scan all available model symbols")
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
```

- [ ] **Step 3: Run all tests**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
7 passed
```

- [ ] **Step 4: Run scanner**

Run:

```bash
python -m backend.ml.strategy_backtest --scan
```

Expected:

```text
{
  "candidate_count": <positive integer>,
  "passing_count": <integer>,
  "best_symbol": "<symbol>",
  "annual_return": <number>,
  "max_drawdown": <number>,
  "meets_course_target": true or false
}
```

- [ ] **Step 5: Inspect generated best result**

Run:

```bash
python -c "import json; d=json.load(open('backend/ml/models/strategy_best.json', encoding='utf-8')); print(json.dumps(d['best'], indent=2)[:2000])"
```

Expected:

```text
{
  "symbol": "...",
  "horizon": "...",
  "annual_return": ...,
  "max_drawdown": ...,
  "meets_course_target": ...
}
```

- [ ] **Step 6: Commit scanner and generated best JSON**

Run:

```bash
git add backend/ml/strategy_backtest.py backend/ml/models/strategy_best.json backend/ml/models/*_strategy_backtest.json
git commit -m "feat: scan for best course strategy"
```

Expected:

```text
[branch commit] feat: scan for best course strategy
```

## Task 6: Optional API Endpoints

**Files:**
- Modify: `backend/api/routers/predict.py`

- [ ] **Step 1: Add endpoints for saved strategy JSON**

Add these route functions to `backend/api/routers/predict.py` after `get_backtest`:

```python
@router.get("/{symbol}/strategy-backtest")
def get_strategy_backtest(
    symbol: str,
    horizon: str = Query("t5", pattern="^t[15]$"),
    threshold: float = Query(0.60, ge=0.0, le=1.0),
):
    """Get saved strategy return backtest for a symbol."""
    sym = symbol.upper()
    threshold_label = f"{int(round(threshold * 100)):02d}"
    path = MODELS_DIR / f"{sym}_{horizon}_thr{threshold_label}_strategy_backtest.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No strategy backtest for {sym}/{horizon}/{threshold}. Run python -m backend.ml.strategy_backtest --symbol {sym} --horizon {horizon} --threshold {threshold}.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/strategy-best")
def get_best_strategy():
    """Get saved best strategy scan result."""
    path = MODELS_DIR / "strategy_best.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No best strategy result. Run python -m backend.ml.strategy_backtest --scan.",
        )
    return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run backend import smoke test**

Run:

```bash
python -c "from backend.api.main import app; print('ok')"
```

Expected:

```text
ok
```

- [ ] **Step 3: Commit API endpoints**

Run:

```bash
git add backend/api/routers/predict.py
git commit -m "feat: expose strategy backtest results"
```

Expected:

```text
[branch commit] feat: expose strategy backtest results
```

## Task 7: Final Verification and Report Evidence

**Files:**
- Read: `backend/ml/models/strategy_best.json`
- Read: `backend/ml/models/{SYMBOL}_{HORIZON}_thr{THRESHOLD}_strategy_backtest.json`

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python -m pytest tests/ml/test_strategy_backtest.py -v
```

Expected:

```text
7 passed
```

- [ ] **Step 2: Run full Python test suite**

Run:

```bash
python -m pytest
```

Expected:

```text
All discovered tests pass.
```

- [ ] **Step 3: Run strategy scan fresh**

Run:

```bash
python -m backend.ml.strategy_backtest --scan
```

Expected:

```text
candidate_count is greater than 0
best_symbol is not null
annual_return is present
max_drawdown is present
meets_course_target is present
```

- [ ] **Step 4: Summarize course metrics**

Run:

```bash
python -c "import json; d=json.load(open('backend/ml/models/strategy_best.json', encoding='utf-8'))['best']; print('symbol=',d.get('symbol')); print('horizon=',d.get('horizon')); print('threshold=',d.get('threshold')); print('annual_return=',d.get('annual_return')); print('max_drawdown=',d.get('max_drawdown')); print('cumulative_return=',d.get('cumulative_return')); print('trade_count=',d.get('trade_count')); print('meets_course_target=',d.get('meets_course_target'))"
```

Expected:

```text
symbol= <best symbol>
horizon= <t1 or t5>
threshold= <threshold>
annual_return= <decimal>
max_drawdown= <decimal>
cumulative_return= <decimal>
trade_count= <integer>
meets_course_target= true or false
```

- [ ] **Step 5: Commit final generated evidence if changed**

Run:

```bash
git status --short
git add backend/ml/models/strategy_best.json backend/ml/models/*_strategy_backtest.json
git commit -m "chore: update strategy backtest evidence"
```

Expected when files changed:

```text
[branch commit] chore: update strategy backtest evidence
```

Expected when no files changed:

```text
nothing to commit
```

## Self-Review

- Spec coverage:
  - Single-stock scanner: Task 5.
  - Long-only trading rule: Task 2 and Task 4.
  - Annualized return and maximum drawdown: Task 1 and Task 4.
  - JSON evidence output: Task 4 and Task 5.
  - Optional API: Task 6.
  - Verification: Task 7.
- Placeholder scan:
  - No unfinished markers or undefined placeholder steps are used.
- Type consistency:
  - `calculate_annual_return`, `calculate_max_drawdown`, `simulate_equity_curve`, `choose_best_strategy`, `build_strategy_result_from_predictions`, `run_strategy_backtest`, and `scan_strategy_space` are introduced before later tasks use them.
