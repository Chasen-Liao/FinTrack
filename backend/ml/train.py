"""CLI entry point: python -m backend.ml.train [--symbol SYM] [--backtest] [--lstm]"""

import argparse
import time
import json

from backend.database import get_conn
from backend.ml.model import train, search_xgboost_params, search_xgboost_params_unified
from backend.ml.backtest import run_backtest
from backend.ml.evaluation_report import run_multi_stock_evaluation

HORIZONS = ["t1", "t5"]
SEARCH_METRICS = ["accuracy_lift", "roc_auc", "f1"]

# Best LSTM configs per ticker (from experiments)
LSTM_CONFIGS = {
    "TSLA": {"target_col": "target_t3", "seq_len": 10, "exclude_neutral": False},
    "MU":   {"target_col": "target_t3", "seq_len": 10, "exclude_neutral": True},
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
    parser.add_argument("--lstm-only", action="store_true", help="Only train LSTM, skip XGBoost")
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
    parser.add_argument("--target-col", type=str,
                        help="Explicit target column, such as target_up_big_t5")
    parser.add_argument("--evaluation-report", action="store_true",
                        help="Generate report-ready aggregate evaluation JSON")
    parser.add_argument("--evaluation-output", type=str,
                        default="backend/ml/models/evaluation_summary.json",
                        help="Output path for --evaluation-report")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else get_symbols()
    horizons = [args.horizon] if args.horizon else HORIZONS

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

    print(f"Training for {len(symbols)} ticker(s): {', '.join(symbols)}")

    t0 = time.time()
    for sym in symbols:
        # Skip XGBoost if --lstm-only
        if not args.lstm_only:
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
                        target_col=args.target_col,
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
                        target_col=args.target_col,
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
                        target_col=args.target_col,
                    )
                else:
                    result = train(
                        sym,
                        h,
                        include_market_benchmark=args.market_benchmark,
                        neutral_band=args.neutral_band,
                        target_col=args.target_col,
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
        if (args.lstm or args.lstm_only) and sym in LSTM_CONFIGS:
            from backend.ml.lstm_model import train_and_save_lstm, run_lstm_backtest
            from pathlib import Path

            cfg = dict(LSTM_CONFIGS[sym])
            if args.target_col:
                cfg["target_col"] = args.target_col
            print(f"  {sym}/LSTM: training {cfg['target_col']} seq={cfg['seq_len']}...")
            result = train_and_save_lstm(sym, **cfg, epochs=50)
            if "error" in result:
                print(f"    LSTM: {result['error']}")
            else:
                print(f"    LSTM: saved ({result['train_size']} sequences)")

            # Run expanding-window backtest and save results
            print(f"  {sym}/LSTM: running backtest...")
            bt = run_lstm_backtest(sym, **cfg)
            if "error" in bt:
                print(f"    LSTM backtest: {bt['error']}")
            else:
                models_dir = Path(__file__).parent / "models"
                neutral_suffix = "exneutral" if cfg.get("exclude_neutral") else "allnews"
                bt_path = models_dir / (
                    f"{sym}_lstm_{cfg['target_col']}_seq{cfg['seq_len']}_{neutral_suffix}_backtest.json"
                )
                bt_path.write_text(json.dumps(bt, indent=2))
                print(f"    LSTM backtest: {bt['n_folds']} folds, "
                      f"acc={bt['overall_accuracy']:.1%} baseline={bt['overall_baseline']:.1%} "
                      f"lift={bt['lift']:+.1f}pp "
                      f"prec={bt['overall_precision']:.1%} rec={bt['overall_recall']:.1%} "
                      f"f1={bt['overall_f1']:.1%}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
