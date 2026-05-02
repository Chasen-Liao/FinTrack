"""CLI entry point: python -m backend.ml.train [--symbol SYM] [--backtest] [--lstm]"""

import argparse
import time
import json

from backend.database import get_conn
from backend.ml.model import train, search_xgboost_params, search_xgboost_params_unified
from backend.ml.backtest import run_backtest

HORIZONS = ["t1", "t5"]
SEARCH_METRICS = ["accuracy_lift", "roc_auc", "f1"]

# Best LSTM configs per ticker (from experiments)
LSTM_CONFIGS = {
    "TSLA": {"target_col": "target_t3", "seq_len": 10, "exclude_neutral": False},
    # Add more tickers here as LSTM proves beneficial
}


def get_symbols() -> list[str]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM tickers WHERE last_ohlc_fetch IS NOT NULL ORDER BY symbol"
    ).fetchall()
    conn.close()
    return [r["symbol"] for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--symbol", type=str, help="Train only this ticker")
    parser.add_argument("--backtest", action="store_true", help="Run backtest after training")
    parser.add_argument("--lstm", action="store_true", help="Also train LSTM for configured tickers")
    parser.add_argument("--search-params", action="store_true", help="Search XGBoost params with expanding-window CV")
    parser.add_argument("--horizon", choices=HORIZONS, help="Search or train only one horizon")
    parser.add_argument("--folds", type=int, default=5, help="Expanding-window fold count for search")
    parser.add_argument("--min-train", type=int, default=200, help="Minimum training rows for expanding-window search")
    parser.add_argument("--apply-best-params", action="store_true", help="Train final model with searched best params")
    parser.add_argument("--market-benchmark", action="store_true", help="Add equal-weight market benchmark features")
    parser.add_argument("--neutral-band", type=float, help="Drop samples whose absolute future return is below this threshold")
    parser.add_argument("--metric", choices=SEARCH_METRICS, default="accuracy_lift",
                        help="Primary metric used to rank expanding-window search candidates")
    parser.add_argument("--unified", action="store_true", help="Use the combined multi-ticker training/search path")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else get_symbols()
    horizons = [args.horizon] if args.horizon else HORIZONS
    print(f"Training for {len(symbols)} ticker(s): {', '.join(symbols)}")

    t0 = time.time()
    for sym in symbols:
        for h in horizons:
            if args.search_params and args.unified:
                search = search_xgboost_params_unified(
                    horizon=h,
                    symbols=symbols,
                    n_folds=args.folds,
                    min_train=args.min_train,
                    metric=args.metric,
                    include_market_benchmark=args.market_benchmark,
                    neutral_band=args.neutral_band,
                )
                if "error" in search:
                    print(f"  UNIFIED/{h} search: {search['error']}")
                    continue
                best = search["best_metrics"]
                print(
                    f"  UNIFIED/{h} search ({args.metric}): "
                    f"lift={best['accuracy_lift']:.2%} "
                    f"auc={best['roc_auc']:.4f} "
                    f"acc={best['accuracy']:.1%} "
                    f"f1={best['f1']:.1%}"
                )
                continue

            if args.search_params:
                search = search_xgboost_params(
                    sym,
                    h,
                    n_folds=args.folds,
                    min_train=args.min_train,
                    metric=args.metric,
                    include_market_benchmark=args.market_benchmark,
                    neutral_band=args.neutral_band,
                )
                if "error" in search:
                    print(f"  {sym}/{h} search: {search['error']}")
                    continue

                best = search["best_metrics"]
                print(
                    f"  {sym}/{h} search ({args.metric}): "
                    f"lift={best['accuracy_lift']:.2%} "
                    f"auc={best['roc_auc']:.4f} "
                    f"acc={best['accuracy']:.1%} "
                    f"f1={best['f1']:.1%} "
                    f"params={json.dumps(search['best_params'], ensure_ascii=False)}"
                )

                if not args.apply_best_params:
                    continue

                result = train(
                    sym,
                    h,
                    model_params=search["best_params"],
                    include_market_benchmark=args.market_benchmark,
                    neutral_band=args.neutral_band,
                )
            else:
                result = train(
                    sym,
                    h,
                    include_market_benchmark=args.market_benchmark,
                    neutral_band=args.neutral_band,
                )
            if "error" in result:
                print(f"  {sym}/{h}: {result['error']}")
            else:
                print(f"  {sym}/{h}: acc={result['accuracy']:.1%} baseline={result['baseline']:.1%} "
                      f"(train={result['train_size']}, test={result['test_size']})")

            if args.backtest and "error" not in result:
                bt = run_backtest(sym, h)
                if "error" in bt:
                    print(f"    backtest: {bt['error']}")
                else:
                    print(f"    backtest: {bt['n_folds']} folds, "
                          f"acc={bt['overall_accuracy']:.1%} baseline={bt['overall_baseline']:.1%}")

        # LSTM training for configured tickers
        if args.lstm and sym in LSTM_CONFIGS:
            from backend.ml.lstm_model import train_and_save_lstm
            cfg = LSTM_CONFIGS[sym]
            print(f"  {sym}/LSTM: training {cfg['target_col']} seq={cfg['seq_len']}...")
            result = train_and_save_lstm(sym, **cfg, epochs=50)
            if "error" in result:
                print(f"    LSTM: {result['error']}")
            else:
                print(f"    LSTM: saved ({result['train_size']} sequences)")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
