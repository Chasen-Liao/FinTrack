"""Enhanced feature engineering v2.

New features over v1:
1. Market-wide sentiment (cross-ticker aggregate)
2. Candlestick patterns (doji, hammer, engulfing, etc.)
3. TF-IDF on news text (key_discussion + title) → PCA top components
4. New targets: big_move_1pct, big_move_2pct, direction_if_big
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from backend.database import get_conn
from backend.ml.features import build_features, FEATURE_COLS, _target_from_return


class TextSvdFeatureTransformer:
    """Fit TF-IDF and SVD on a training window, then transform another window."""

    def __init__(self, max_features: int = 500, min_df: int = 3, n_components: int = 10):
        self.max_features = max_features
        self.min_df = min_df
        self.n_components = n_components
        self.vectorizer = None
        self.svd = None
        self.output_columns_: list[str] = []
        self.fit_dates_: list[str] = []

    def fit(self, rows: pd.DataFrame, y=None):
        texts = rows["text"].fillna("").tolist()
        self.fit_dates_ = rows["trade_date"].dt.strftime("%Y-%m-%d").tolist()
        if not any(text.strip() for text in texts):
            self.output_columns_ = []
            return self

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            min_df=self.min_df,
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if n_comp < 1:
            self.output_columns_ = []
            self.svd = None
            return self

        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf_matrix)
        self.output_columns_ = [f"text_svd_{i}" for i in range(n_comp)]
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        result = rows[["trade_date"]].copy()
        if self.vectorizer is None or self.svd is None:
            return result

        matrix = self.vectorizer.transform(rows["text"].fillna("").tolist())
        reduced = self.svd.transform(matrix)
        for i, col in enumerate(self.output_columns_):
            result[col] = reduced[:, i]
        return result


def _load_market_sentiment() -> pd.DataFrame:
    """Aggregate sentiment across ALL tickers per trading date."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT na.trade_date,
               COUNT(*) AS mkt_articles,
               SUM(CASE WHEN l1.sentiment = 'positive' THEN 1 ELSE 0 END) AS mkt_positive,
               SUM(CASE WHEN l1.sentiment = 'negative' THEN 1 ELSE 0 END) AS mkt_negative,
               COUNT(DISTINCT na.symbol) AS mkt_tickers_active
        FROM news_aligned na
        JOIN layer1_results l1 ON na.news_id = l1.news_id AND na.symbol = l1.symbol
        GROUP BY na.trade_date
        ORDER BY na.trade_date
        """
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    total = df["mkt_articles"].clip(lower=1)
    df["mkt_sentiment"] = (df["mkt_positive"] - df["mkt_negative"]) / total
    df["mkt_positive_ratio"] = df["mkt_positive"] / total
    # Rolling market sentiment
    df["mkt_sentiment_3d"] = df["mkt_sentiment"].rolling(3, min_periods=1).mean()
    df["mkt_sentiment_5d"] = df["mkt_sentiment"].rolling(5, min_periods=1).mean()
    df["mkt_momentum"] = df["mkt_sentiment_3d"] - df["mkt_sentiment_5d"]
    return df


def load_text_by_date(symbol: str) -> pd.DataFrame:
    """Load concatenated article text per trading date for fold-local fitting."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT na.trade_date,
               GROUP_CONCAT(COALESCE(l1.key_discussion, '') || ' ' || COALESCE(nr.title, ''), ' ') AS text
        FROM news_aligned na
        JOIN news_raw nr ON na.news_id = nr.id
        LEFT JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = na.symbol
        WHERE na.symbol = ?
        GROUP BY na.trade_date
        ORDER BY na.trade_date
        """,
        (symbol,),
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(columns=["trade_date", "text"])

    df = pd.DataFrame([dict(r) for r in rows])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["text"] = df["text"].fillna("")
    return df


def _add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick pattern features from OHLC data."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).clip(lower=1e-10)

    # Body ratio: how much of the range is body (0=doji, 1=marubozu)
    df["candle_body_ratio"] = body / rng

    # Direction: 1=bullish candle, 0=bearish
    df["candle_bullish"] = (c > o).astype(int)

    # Upper shadow ratio
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    df["candle_upper_shadow"] = upper_shadow / rng

    # Lower shadow ratio
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    df["candle_lower_shadow"] = lower_shadow / rng

    # Doji: body < 10% of range
    df["candle_doji"] = (df["candle_body_ratio"] < 0.1).astype(int)

    # Hammer: small body at top, long lower shadow
    df["candle_hammer"] = ((df["candle_lower_shadow"] > 0.6) & (df["candle_body_ratio"] < 0.3)).astype(int)

    # 2-day patterns (shifted to avoid leakage)
    prev_bullish = df["candle_bullish"].shift(1)
    prev_body = body.shift(1)
    # Engulfing: current body larger than previous and opposite direction
    df["candle_engulfing"] = ((body > prev_body) & (df["candle_bullish"] != prev_bullish)).astype(int).shift(1)

    # Consecutive: same direction streak
    df["candle_streak"] = df["candle_bullish"].rolling(3, min_periods=1).sum().shift(1)  # 0-3 bullish in last 3

    # All candle features shifted by 1 to prevent leakage (use yesterday's pattern)
    for col in ["candle_body_ratio", "candle_bullish", "candle_upper_shadow",
                "candle_lower_shadow", "candle_doji", "candle_hammer"]:
        df[col] = df[col].shift(1)

    return df


def _build_text_features(symbol: str, dates: pd.Series, n_components: int = 10) -> pd.DataFrame:
    """Build TF-IDF → SVD features from news text per trading day."""
    text_df = load_text_by_date(symbol)
    if text_df.empty:
        return pd.DataFrame({"trade_date": dates})

    transformer = TextSvdFeatureTransformer(n_components=n_components)
    transformer.fit(text_df)
    if not transformer.output_columns_:
        return pd.DataFrame({"trade_date": dates})

    return transformer.transform(text_df)


def build_features_v2(symbol: str, use_text: bool = False) -> pd.DataFrame:
    """Build enhanced feature matrix with market, candle, and text features."""
    df = build_features(symbol)
    if df.empty:
        return df

    # 1. Market-wide sentiment
    mkt = _load_market_sentiment()
    if not mkt.empty:
        df = df.merge(mkt, on="trade_date", how="left")
        for col in ["mkt_articles", "mkt_positive", "mkt_negative", "mkt_tickers_active",
                     "mkt_sentiment", "mkt_positive_ratio", "mkt_sentiment_3d",
                     "mkt_sentiment_5d", "mkt_momentum"]:
            df[col] = df[col].fillna(0)
    else:
        for col in ["mkt_sentiment", "mkt_positive_ratio", "mkt_sentiment_3d",
                     "mkt_sentiment_5d", "mkt_momentum"]:
            df[col] = 0

    # 2. Candlestick patterns
    df = _add_candle_patterns(df)

    # 3. Text features (TF-IDF → SVD)
    if use_text:
        text_feats = _build_text_features(symbol, df["trade_date"])
        if "text_svd_0" in text_feats.columns:
            df = df.merge(text_feats, on="trade_date", how="left")
            text_cols = [c for c in text_feats.columns if c.startswith("text_svd_")]
            df[text_cols] = df[text_cols].fillna(0)

    # 4. New targets
    close = df["close"]
    ret_t1 = close.shift(-1) / close - 1
    ret_t3 = close.shift(-3) / close - 1

    # Big move targets: |return| > threshold
    df["target_big1_t1"] = _target_from_return(ret_t1, ret_t1.abs() > 0.01)  # >1% move
    df["target_big2_t1"] = _target_from_return(ret_t1, ret_t1.abs() > 0.02)  # >2% move
    df["target_big1_t3"] = _target_from_return(ret_t3, ret_t3.abs() > 0.02)  # >2% in 3 days

    # Direction only when big move (more signal)
    df["target_up_big_t1"] = _target_from_return(ret_t1, ret_t1 > 0.01)   # up >1%
    df["target_down_big_t1"] = _target_from_return(ret_t1, ret_t1 < -0.01)  # down >1%

    # Drop NaN from new features
    df = df.dropna(subset=["candle_body_ratio"]).reset_index(drop=True)

    return df


# Feature column sets for different experiments
FEATURE_COLS_V2_MARKET = FEATURE_COLS + [
    "mkt_sentiment", "mkt_positive_ratio", "mkt_sentiment_3d", "mkt_sentiment_5d", "mkt_momentum",
]

FEATURE_COLS_V2_CANDLE = FEATURE_COLS_V2_MARKET + [
    "candle_body_ratio", "candle_bullish", "candle_upper_shadow",
    "candle_lower_shadow", "candle_doji", "candle_hammer",
    "candle_engulfing", "candle_streak",
]

def get_feature_cols_v2_full(df: pd.DataFrame) -> list[str]:
    """Get all v2 feature columns including text SVD components."""
    text_cols = sorted([c for c in df.columns if c.startswith("text_svd_")])
    return FEATURE_COLS_V2_CANDLE + text_cols
