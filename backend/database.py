import sqlite3
from backend.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickers (
    symbol        TEXT PRIMARY KEY,
    name          TEXT,
    sector        TEXT,
    last_ohlc_fetch   TEXT,
    last_news_fetch   TEXT
);

CREATE TABLE IF NOT EXISTS ohlc (
    symbol        TEXT NOT NULL,
    date          TEXT NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    volume        REAL,
    vwap          REAL,
    transactions  INTEGER,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS news_raw (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    description   TEXT,
    publisher     TEXT,
    author        TEXT,
    published_utc TEXT,
    article_url   TEXT,
    amp_url       TEXT,
    tickers_json  TEXT,
    insights_json TEXT
);

CREATE TABLE IF NOT EXISTS news_ticker (
    news_id       TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    PRIMARY KEY (news_id, symbol),
    FOREIGN KEY (news_id) REFERENCES news_raw(id)
);

CREATE TABLE IF NOT EXISTS layer0_results (
    news_id       TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    passed        INTEGER NOT NULL,
    reason        TEXT,
    PRIMARY KEY (news_id, symbol)
);

CREATE TABLE IF NOT EXISTS layer1_results (
    news_id       TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    relevance     TEXT,
    key_discussion      TEXT,
    chinese_summary     TEXT,
    sentiment           TEXT,
    sentiment_score     REAL,
    event_category      TEXT,
    discussion          TEXT,
    reason_growth       TEXT,
    reason_decrease     TEXT,
    PRIMARY KEY (news_id, symbol)
);

CREATE TABLE IF NOT EXISTS layer2_results (
    news_id       TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    discussion    TEXT,
    growth_reasons  TEXT,
    decrease_reasons TEXT,
    created_at    TEXT,
    PRIMARY KEY (news_id, symbol)
);

CREATE TABLE IF NOT EXISTS news_aligned (
    news_id       TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    trade_date    TEXT NOT NULL,
    published_utc TEXT,
    ret_t0        REAL,
    ret_t1        REAL,
    ret_t3        REAL,
    ret_t5        REAL,
    ret_t10       REAL,
    PRIMARY KEY (news_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_news_aligned_symbol_date ON news_aligned(symbol, trade_date);

CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id      TEXT PRIMARY KEY,
    symbol        TEXT,
    status        TEXT,
    total         INTEGER,
    completed     INTEGER DEFAULT 0,
    created_at    TEXT,
    finished_at   TEXT
);

CREATE TABLE IF NOT EXISTS batch_request_map (
    batch_id      TEXT NOT NULL,
    custom_id     TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    article_ids   TEXT NOT NULL,
    PRIMARY KEY (batch_id, custom_id)
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def run_migrations():
    """Add new columns to existing tables (safe to run repeatedly)."""
    conn = get_conn()
    cursor = conn.execute("PRAGMA table_info(layer1_results)")
    existing = {row[1] for row in cursor.fetchall()}
    if "sentiment_score" not in existing:
        conn.execute("ALTER TABLE layer1_results ADD COLUMN sentiment_score REAL")
    if "event_category" not in existing:
        conn.execute("ALTER TABLE layer1_results ADD COLUMN event_category TEXT")
    conn.commit()
    conn.close()


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.close()
    run_migrations()
    print(f"Database initialized at {settings.database_path}")


if __name__ == "__main__":
    init_db()
