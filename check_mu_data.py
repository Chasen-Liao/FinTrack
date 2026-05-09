"""Quick check: MU data availability in database."""
import sqlite3

conn = sqlite3.connect("pokieticker.db")
conn.row_factory = sqlite3.Row

# Check OHLC date range for MU
rows = conn.execute(
    "SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as cnt FROM ohlc WHERE symbol='MU'"
).fetchall()
for r in rows:
    print(f"MU OHLC: {r['min_date']} ~ {r['max_date']}, count={r['cnt']}")

# Check news_aligned date range for MU
rows2 = conn.execute(
    "SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date, COUNT(*) as cnt FROM news_aligned WHERE symbol='MU'"
).fetchall()
for r in rows2:
    print(f"MU News: {r['min_date']} ~ {r['max_date']}, count={r['cnt']}")

# Check Q4 2025 specifically
rows3 = conn.execute(
    "SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as cnt FROM ohlc WHERE symbol='MU' AND date >= '2025-10-01' AND date <= '2025-12-31'"
).fetchall()
for r in rows3:
    print(f"MU Q4 2025 OHLC: {r['min_date']} ~ {r['max_date']}, count={r['cnt']}")

# Check how many total rows to understand full data span
rows4 = conn.execute(
    "SELECT date, close FROM ohlc WHERE symbol='MU' AND date >= '2025-10-01' AND date <= '2025-12-31' ORDER BY date"
).fetchall()
print(f"\nQ4 2025 trading days:")
for r in rows4:
    print(f"  {r['date']}: close={r['close']}")

conn.close()
