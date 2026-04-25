# AI Quant Strategy Backtest Design

## Goal

Improve the current project so it can support the course requirement for "Project 7: Quantitative Trading Strategy Based on AI Algorithms".

The feature must produce evidence for these two target metrics:

- Annualized return greater than 20%.
- Maximum drawdown less than 20%.

The priority is course deliverability. The system should search the existing historical data and trained models for a single-stock strategy configuration that satisfies the target metrics if one exists.

## Current Context

The project already includes:

- Polygon-based OHLC and news collection.
- SQLite storage for tickers, OHLC, raw news, aligned news, and layer analysis results.
- XGBoost models for per-symbol direction prediction.
- LSTM experiment support for selected symbols.
- Existing prediction accuracy backtests.
- React/FastAPI UI and API for news, prediction, and analysis.

The current gap is that `backend/ml/backtest.py` evaluates classification quality, not real trading performance. It reports accuracy, baseline, precision, recall, and F1, but it does not compute annualized return, maximum drawdown, trading equity curve, or strategy-level returns.

## Scope

Add a single-stock AI strategy backtest module.

The module will:

1. Scan existing symbols with available OHLC data and trained model files.
2. Evaluate strategy configurations across prediction horizons and confidence thresholds.
3. Generate long-only trading signals from expanding-window out-of-sample AI model probabilities.
4. Simulate trades and an equity curve.
5. Calculate annualized return, maximum drawdown, cumulative return, win rate, trade count, average trade return, best trade, worst trade, and buy-and-hold benchmark return.
6. Save the best strategy result to JSON for report use.

Out of scope for the first implementation:

- Short selling.
- Portfolio rotation across multiple stocks.
- Intraday data.
- Real order execution.
- Frontend chart integration unless the backend result is already complete.

## Strategy Object

The strategy does not start with a fixed symbol.

The system will scan the existing stock universe and select one best-performing single-stock strategy. The selected symbol becomes the report's main research object.

This is aligned with the course-deliverability goal: the report can state that the system automatically selected the best representative stock from the available dataset.

## Trading Rules

The first implementation uses a long-only signal:

```text
If predicted direction is up and predicted up probability >= confidence threshold:
    enter or hold a long position.
Otherwise:
    stay in cash.
```

Candidate parameters:

- Horizon: `t1`, `t5`.
- Confidence threshold: `0.50`, `0.55`, `0.60`, `0.65`, `0.70`.
- Transaction fee: `0.1%` per side by default.

The horizon determines the model target and the intended holding period:

- `t1`: one trading day.
- `t5`: five trading days.

If overlapping signals occur, the first implementation may use a simple daily position model: position is `1` on days where the signal is active, otherwise `0`. This keeps the strategy understandable for the course report.

## Data Flow

1. Load OHLC rows for a symbol from the existing SQLite database.
2. Build features using the existing feature pipeline.
3. Train XGBoost models in an expanding-window loop so each tested date is predicted only by models trained on earlier rows.
4. Generate out-of-sample predicted probabilities for each eligible test date.
5. Convert probabilities into position signals using the selected threshold.
6. Calculate daily strategy return:

```text
strategy_return = position_previous_day * stock_daily_return - transaction_cost
```

7. Build equity curve from daily strategy returns.
8. Calculate performance metrics.
9. Save per-configuration and best-strategy JSON outputs.

## Metrics

The strategy output must include:

- `symbol`
- `horizon`
- `threshold`
- `fee_rate`
- `start_date`
- `end_date`
- `annual_return`
- `max_drawdown`
- `cumulative_return`
- `buy_hold_return`
- `win_rate`
- `trade_count`
- `average_trade_return`
- `best_trade_return`
- `worst_trade_return`
- `meets_annual_return_target`
- `meets_drawdown_target`
- `meets_course_target`
- `equity_curve`
- `trades`

Target checks:

```text
meets_annual_return_target = annual_return > 0.20
meets_drawdown_target = max_drawdown < 0.20
meets_course_target = both checks are true
```

Maximum drawdown is represented as a positive decimal. For example, `0.18` means an 18% drawdown.

## Best Strategy Selection

The scanner evaluates:

```text
symbol x horizon x confidence threshold
```

Selection order:

1. Prefer strategies where `meets_course_target` is true.
2. Among passing strategies, prefer higher annual return.
3. Use lower maximum drawdown as a tie-breaker.
4. Require a minimum reasonable trade count so the result is not based on one isolated trade.
5. If no strategy passes, return the closest result and mark it as not meeting the course target.

The primary output file:

```text
backend/ml/models/strategy_best.json
```

Per-symbol outputs:

```text
backend/ml/models/{SYMBOL}_{HORIZON}_thr{THRESHOLD}_strategy_backtest.json
```

## API Option

After the core module works, the prediction API can expose:

```text
GET /api/predict/{symbol}/strategy-backtest
GET /api/predict/strategy-best
```

This is optional for the first implementation. JSON files and command output are sufficient for the course report.

## Testing

Tests should cover the metric calculations and selection logic before production code is added.

Required behaviors:

- Annualized return is calculated correctly from an equity curve.
- Maximum drawdown is calculated correctly.
- No-trade periods do not crash the backtest.
- Transaction fees reduce returns.
- Scanner selects a passing strategy over a non-passing strategy.
- If no strategy passes, scanner still returns the closest candidate and marks it as not passing.

## Verification

Implementation is not complete until these checks run successfully:

```text
python -m pytest
python -m backend.ml.strategy_backtest --scan
```

The final report should use fresh output from `strategy_best.json`, not manually edited metrics.

## Report Positioning

The course report can describe the strategy as:

```text
This project builds a single-stock AI quantitative trading strategy.
The system scans the available stock universe and selects the best-performing representative stock.
The AI model predicts future price direction from price, volume, technical indicators, and news sentiment features.
When the model predicts an upward move with sufficient confidence, the strategy enters a long position; otherwise it holds cash.
The strategy is evaluated using annualized return, maximum drawdown, cumulative return, win rate, and trade count.
```

If the best strategy meets the target metrics, the report should present it as the final result. If no strategy meets both metrics, the report should present the closest strategy and explain optimization directions.
