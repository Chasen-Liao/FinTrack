"""Forecast module: aggregate recent news window → predict future trend.

Combines:
1. Recent news aggregation (7d or 30d window)
2. XGBoost model prediction
3. Similar historical period search
4. Statistical conclusion generation
"""

import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib

from backend.database import get_conn
from backend.ml.features import build_features, FEATURE_COLS, resolve_feature_cols

MODELS_DIR = Path(__file__).parent / "models"


def _load_recent_news(symbol: str, window_days: int, ref_date: str | None = None) -> list[dict]:
    """Load recent news articles within the window.

    Args:
        ref_date: Reference date (YYYY-MM-DD). If None, uses the latest
                  available trade_date for this symbol in the database.
    """
    conn = get_conn()
    if ref_date is None:
        row = conn.execute(
            "SELECT MAX(trade_date) FROM news_aligned WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row and row[0]:
            ref_date = row[0]
        else:
            ref_date = datetime.now().strftime("%Y-%m-%d")
    ref_dt = datetime.strptime(ref_date, "%Y-%m-%d") if isinstance(ref_date, str) else ref_date
    cutoff = (ref_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        """SELECT na.news_id, na.trade_date, nr.title,
                  l1.sentiment, l1.chinese_summary,
                  l1.relevance, l1.key_discussion,
                  na.ret_t0, na.ret_t1
           FROM news_aligned na
           JOIN news_raw nr ON na.news_id = nr.id
           LEFT JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = na.symbol
           WHERE na.symbol = ? AND na.trade_date >= ? AND na.trade_date <= ?
           ORDER BY na.trade_date DESC
           LIMIT 200""",
        (symbol, cutoff, ref_date),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _compute_window_features(df: pd.DataFrame, window_days: int) -> np.ndarray | None:
    """Average the feature vectors over the last `window_days` trading days."""
    if df.empty:
        return None
    # Take last N rows (trading days, not calendar days)
    n_rows = min(window_days, len(df))
    window_df = df.iloc[-n_rows:]
    vec = window_df[FEATURE_COLS].mean().values.astype(np.float64)
    np.nan_to_num(vec, copy=False)
    return vec


def _find_similar_periods(
    df: pd.DataFrame, window_vec: np.ndarray, window_days: int, top_k: int = 10
) -> list[dict]:
    """Slide a window over history, compute average feature vector for each
    position, then find the most similar windows to the current one."""
    n = len(df)
    if n < window_days + 10:
        return []

    # Build sliding window averages
    X_raw = df[FEATURE_COLS].values.astype(np.float64)
    np.nan_to_num(X_raw, copy=False)

    # Compute rolling mean for each feature using cumsum for efficiency
    cumsum = np.vstack([np.zeros((1, X_raw.shape[1])), np.cumsum(X_raw, axis=0)])
    # window_vecs[i] = mean of rows [i, i+window_days)
    max_start = n - window_days
    window_vecs = (cumsum[window_days:n + 1] - cumsum[:max_start + 1]) / window_days

    # Normalize all vectors (including the target)
    all_vecs = np.vstack([window_vecs, window_vec.reshape(1, -1)])
    mean = np.mean(all_vecs, axis=0)
    std = np.std(all_vecs, axis=0)
    std[std < 1e-10] = 1.0
    all_norm = (all_vecs - mean) / std

    target_norm = all_norm[-1]
    history_norm = all_norm[:-1]

    # Cosine similarity
    norms = np.linalg.norm(history_norm, axis=1)
    norms[norms < 1e-10] = 1.0
    target_n = np.linalg.norm(target_norm)
    if target_n < 1e-10:
        target_n = 1.0
    sims = history_norm @ target_norm / (norms * target_n)

    # Exclude windows that overlap with the current period (last window_days*2 rows)
    exclude_start = max(0, len(sims) - window_days * 2)
    sims[exclude_start:] = -999

    # Top K
    top_indices = np.argsort(sims)[::-1][:top_k]

    # Get forward returns after each historical window
    conn = get_conn()
    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()

    results = []
    for idx in top_indices:
        if sims[idx] < -900:
            continue
        period_start = dates[idx]
        period_end = dates[min(idx + window_days - 1, n - 1)]
        # Forward return: price change in the window_days after this period
        after_start = idx + window_days
        after_end_t5 = min(after_start + 5, n)
        after_end_t10 = min(after_start + 10, n)

        if after_start >= n:
            continue

        close_vals = df["close"].values
        period_close = close_vals[min(idx + window_days - 1, n - 1)]

        ret_t5 = None
        ret_t10 = None
        if after_end_t5 > after_start:
            ret_t5 = round((close_vals[after_end_t5 - 1] / period_close - 1) * 100, 2)
        if after_end_t10 > after_start:
            ret_t10 = round((close_vals[after_end_t10 - 1] / period_close - 1) * 100, 2)

        # Sentiment in that historical window
        window_slice = df.iloc[idx:idx + window_days]
        avg_sentiment = float(window_slice["sentiment_score"].mean())

        results.append({
            "period_start": period_start,
            "period_end": period_end,
            "similarity": round(float(sims[idx]), 4),
            "avg_sentiment": round(avg_sentiment, 3),
            "n_articles": int(window_slice["n_articles"].sum()),
            "ret_after_5d": ret_t5,
            "ret_after_10d": ret_t10,
        })

    conn.close()
    return results


def generate_forecast(symbol: str, window_days: int = 7) -> dict:
    """Generate a complete forecast report for a symbol.

    Args:
        symbol: Ticker symbol
        window_days: Look-back window (7 or 30)

    Returns:
        Complete forecast with prediction, similar periods, recent news, conclusion.
    """
    symbol = symbol.upper()
    df = build_features(symbol)
    if df.empty:
        return {"error": f"No feature data for {symbol}"}

    # Use last available trade date as reference (not today's date)
    last_date = df.iloc[-1]["trade_date"].strftime("%Y-%m-%d")

    # 1. Recent news
    recent_news = _load_recent_news(symbol, window_days, ref_date=last_date)
    n_pos = sum(1 for n in recent_news if n.get("sentiment") == "positive")
    n_neg = sum(1 for n in recent_news if n.get("sentiment") == "negative")
    n_neu = sum(1 for n in recent_news if n.get("sentiment") == "neutral")
    n_total = len(recent_news)

    # Score articles by composite: relevance + sentiment strength + price move
    def _impact_score(n):
        score = 0.0
        # Relevant articles score higher
        if n.get("relevance") == "relevant":
            score += 2.0
        # Non-neutral sentiment scores higher
        sent = n.get("sentiment")
        if sent in ("positive", "negative"):
            score += 1.5
        elif sent == "neutral":
            score += 0.3
        # else (None) → 0
        # Price move magnitude
        if n.get("ret_t0") is not None:
            score += min(abs(n["ret_t0"]) * 10, 2.0)  # cap at 2.0
        return score

    impact_candidates = [
        n for n in recent_news
        if n.get("sentiment") in ("positive", "negative")  # exclude neutral/None
        and n.get("ret_t0") is not None
    ]
    # Fallback: if too few strong-sentiment articles, include neutral ones too
    if len(impact_candidates) < 5:
        impact_candidates = [
            n for n in recent_news
            if n.get("sentiment") is not None and n.get("ret_t0") is not None
        ]
    impact_sorted = sorted(impact_candidates, key=_impact_score, reverse=True)

    news_summary = {
        "total": n_total,
        "positive": n_pos,
        "negative": n_neg,
        "neutral": n_neu,
        "sentiment_ratio": round((n_pos - n_neg) / max(n_total, 1), 3),
        # Top headlines (most recent)
        "top_headlines": [
            {
                "date": n["trade_date"],
                "title": (n["title"] or "")[:100],
                "sentiment": n.get("sentiment", "unknown"),
                "summary": (n.get("chinese_summary") or "")[:120],
            }
            for n in recent_news[:10]
        ],
        # Most impactful articles (by price move magnitude)
        "top_impact": [
            {
                "news_id": n["news_id"],
                "date": n["trade_date"],
                "title": (n["title"] or "")[:120],
                "sentiment": n.get("sentiment", "unknown"),
                "relevance": n.get("relevance"),
                "key_discussion": (n.get("key_discussion") or "")[:150],
                "ret_t0": round(n["ret_t0"] * 100, 2) if n.get("ret_t0") else None,
                "ret_t1": round(n["ret_t1"] * 100, 2) if n.get("ret_t1") else None,
            }
            for n in impact_sorted[:5]
        ],
    }

    # 2. Window feature vector (average of last N trading days)
    window_vec = _compute_window_features(df, window_days)
    if window_vec is None:
        return {"error": "Cannot compute features"}

    # 3. Model predictions
    prediction = None

    # 3a. Check for LSTM model (best for some tickers like TSLA)
    try:
        from backend.ml.lstm_model import predict_lstm
        lstm_result = predict_lstm(symbol)
    except ImportError:
        lstm_result = None
    if lstm_result is not None:
        h = lstm_result["horizon"]  # e.g. "t3"
        if prediction is None:
            prediction = {}
        prediction[h] = {
            "direction": lstm_result["direction"],
            "confidence": lstm_result["confidence"],
            "model_type": "LSTM",
            "top_drivers": [],  # LSTM doesn't have per-feature importances
            "model_accuracy": None,
            "baseline_accuracy": None,
        }

    # 3b. XGBoost predictions for t1/t5
    for horizon in ["t1", "t5"]:
        model_path = MODELS_DIR / f"{symbol}_{horizon}.joblib"
        meta_path = MODELS_DIR / f"{symbol}_{horizon}_meta.json"
        json_path = MODELS_DIR / f"{symbol}_{horizon}_xgboost.json"
        if not model_path.exists() and not json_path.exists():
            continue

        # 优先使用原生 XGBoost JSON 格式（跨版本兼容）
        from xgboost import XGBClassifier
        if json_path.exists():
            booster = XGBClassifier()
            booster.load_model(str(json_path))
            model = booster
        else:
            model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text())

        # Use the correct feature columns for this model's expected input
        model_cols = resolve_feature_cols(model)
        last_row = df.iloc[-1]
        X = last_row[model_cols].values.reshape(1, -1).astype(np.float64)
        np.nan_to_num(X, copy=False)

        proba = model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        # Instance-level feature contribution (deviation from training mean)
        feature_means = df[model_cols].mean()
        feature_stds = df[model_cols].std().clip(lower=1e-10)
        importances = model.feature_importances_

        contributions = []
        for i, col in enumerate(model_cols):
            val = float(last_row[col]) if pd.notna(last_row[col]) else 0.0
            z = (val - feature_means[col]) / feature_stds[col]
            contrib = abs(z) * importances[i]
            contributions.append({
                "name": col,
                "value": round(val, 4),
                "importance": round(float(importances[i]), 4),
                "z_score": round(float(z), 2),
                "contribution": round(float(contrib), 4),
            })
        contributions.sort(key=lambda x: x["contribution"], reverse=True)

        if prediction is None:
            prediction = {}
        prediction[horizon] = {
            "direction": "up" if pred_class == 1 else "down",
            "confidence": round(confidence, 4),
            "model_type": "XGBoost",
            "top_drivers": contributions[:6],
            "model_accuracy": meta.get("accuracy", 0),
            "baseline_accuracy": meta.get("baseline", 0),
        }

    if prediction is None:
        return {"error": f"No trained model for {symbol}"}

    # 4. Similar historical periods
    similar_periods = _find_similar_periods(df, window_vec, window_days, top_k=10)

    # Stats from similar periods
    rets_5 = [p["ret_after_5d"] for p in similar_periods if p["ret_after_5d"] is not None]
    rets_10 = [p["ret_after_10d"] for p in similar_periods if p["ret_after_10d"] is not None]

    similar_stats = {
        "count": len(similar_periods),
        "up_ratio_5d": round(sum(1 for r in rets_5 if r > 0) / max(len(rets_5), 1), 2),
        "up_ratio_10d": round(sum(1 for r in rets_10 if r > 0) / max(len(rets_10), 1), 2),
        "avg_ret_5d": round(float(np.mean(rets_5)), 2) if rets_5 else None,
        "avg_ret_10d": round(float(np.mean(rets_10)), 2) if rets_10 else None,
    }

    # 5. Generate conclusion (pure statistics, no AI API)
    conclusion = _build_conclusion(
        symbol, window_days, news_summary, prediction, similar_stats
    )

    last_date = df.iloc[-1]["trade_date"].strftime("%Y-%m-%d")

    return {
        "symbol": symbol,
        "window_days": window_days,
        "forecast_date": last_date,
        "news_summary": news_summary,
        "prediction": prediction,
        "similar_periods": similar_periods,
        "similar_stats": similar_stats,
        "conclusion": conclusion,
    }


def _build_conclusion(
    symbol: str,
    window_days: int,
    news_summary: dict,
    prediction: dict,
    similar_stats: dict,
) -> str:
    """Build a Chinese-language conclusion from statistical signals."""
    parts = []

    window_label = f"近{window_days}天" if window_days <= 7 else f"近{window_days}天（约1个月）"
    n = news_summary["total"]
    ratio = news_summary["sentiment_ratio"]

    # News summary
    if n == 0:
        parts.append(f"{symbol}在{window_label}内没有相关新闻。")
    else:
        tone = "偏正面" if ratio > 0.1 else "偏负面" if ratio < -0.1 else "中性"
        parts.append(
            f"{symbol}在{window_label}内有{n}篇相关新闻，"
            f"正面{news_summary['positive']}篇/负面{news_summary['negative']}篇，"
            f"整体情绪{tone}（{ratio:+.2f}）。"
        )

    # Model prediction
    horizon_labels = [
        ("短期(T+1)", "t1"), ("中期(T+3)", "t3"), ("中期(T+5)", "t5"),
    ]
    for h_label, h_key in horizon_labels:
        p = prediction.get(h_key)
        if not p:
            continue
        d = "看涨" if p["direction"] == "up" else "看跌"
        conf = p["confidence"] * 100
        model_tag = f"[{p.get('model_type', 'XGBoost')}]" if p.get("model_type") else ""
        parts.append(f"{model_tag}模型{h_label}预测：{d}，置信度{conf:.0f}%。")

    # Similar periods
    if similar_stats["count"] > 0:
        ur5 = similar_stats.get("up_ratio_5d")
        ar5 = similar_stats.get("avg_ret_5d")
        if ur5 is not None and ar5 is not None:
            parts.append(
                f"在{similar_stats['count']}个历史相似时期中，"
                f"{ur5*100:.0f}%在后续5天上涨，"
                f"平均收益为{ar5:+.1f}%。"
            )

    # Overall judgment
    signals = []
    t1 = prediction.get("t1", {})
    if t1:
        signals.append(1 if t1["direction"] == "up" else -1)
    t3 = prediction.get("t3", {})
    if t3:
        signals.append(1 if t3["direction"] == "up" else -1)
    t5 = prediction.get("t5", {})
    if t5:
        signals.append(1 if t5["direction"] == "up" else -1)
    if similar_stats.get("up_ratio_5d") is not None:
        signals.append(1 if similar_stats["up_ratio_5d"] > 0.5 else -1)
    if ratio > 0.1:
        signals.append(1)
    elif ratio < -0.1:
        signals.append(-1)

    if signals:
        avg_signal = sum(signals) / len(signals)
        if avg_signal > 0.3:
            parts.append("多信号综合评估：偏多。")
        elif avg_signal < -0.3:
            parts.append("多信号综合评估：偏空。")
        else:
            parts.append("多信号综合评估：方向不明，建议观望。")

    # Strip trailing periods from each part, then re-join
    parts = [p.rstrip("。") for p in parts if p.strip()]
    return "。".join(parts) + "。"
