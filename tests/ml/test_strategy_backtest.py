import math
from datetime import datetime, timedelta

import numpy as np

from backend.ml.strategy_backtest import (
    build_strategy_result_from_predictions,
    calculate_annual_return,
    calculate_max_drawdown,
    choose_best_strategy,
    simulate_equity_curve,
)


def test_calculate_annual_return_from_equity_curve():
    equity_curve = [
        {
            "date": (datetime(2024, 1, 2) + timedelta(days=i)).strftime("%Y-%m-%d"),
            "equity": 1.0 + (0.21 * i / 252),
        }
        for i in range(253)
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


def test_build_strategy_result_from_predictions_marks_course_target():
    df = [
        {"trade_date": datetime(2024, 1, 1), "close": 100.0},
        {"trade_date": datetime(2024, 1, 2), "close": 102.0},
        {"trade_date": datetime(2024, 1, 3), "close": 104.0},
        {"trade_date": datetime(2024, 1, 4), "close": 106.0},
        {"trade_date": datetime(2024, 1, 5), "close": 108.0},
        {"trade_date": datetime(2024, 1, 6), "close": 110.0},
    ]
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
