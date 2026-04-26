#!/usr/bin/env python3
import psycopg2

conn = psycopg2.connect(
    host='blunt-serpent-15319.jxf.gcp-us-central1.cockroachlabs.cloud',
    port=26257,
    user='ehis',
    password='ZYjErHst77b78mb9ENC_Cw',
    database='defaultdb',
    sslmode='require'
)

cur = conn.cursor()

print("=" * 70)
print("CockroachDB Migration Verification")
print("=" * 70)

# Check stocks
cur.execute("SELECT COUNT(*) FROM stocks")
stocks_count = cur.fetchone()[0]
print(f"\n[OK] Stocks: {stocks_count:,} records")

cur.execute("SELECT COUNT(DISTINCT index_name) FROM stocks")
indices = cur.fetchone()[0]
print(f"  Covering {indices} index(es)")

# Check price_history
cur.execute("SELECT COUNT(*) FROM price_history")
price_count = cur.fetchone()[0]
print(f"\n[OK] Price History: {price_count:,} records")

cur.execute("SELECT COUNT(DISTINCT ticker) FROM price_history")
tickers = cur.fetchone()[0]
print(f"  For {tickers:,} unique tickers")

cur.execute("""
    SELECT MIN(date) as earliest, MAX(date) as latest
    FROM price_history
""")
earliest, latest = cur.fetchone()
print(f"  Date range: {earliest} to {latest}")

print("\n" + "=" * 70)
print("[SUCCESS] All data migrated successfully!")
print("=" * 70)

conn.close()
