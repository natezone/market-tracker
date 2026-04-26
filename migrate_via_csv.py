#!/usr/bin/env python3
"""
Migrate data via CSV export (more reliable for large datasets)
"""
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OLD_DB_URL = "postgresql://postgres.htodaqaeaithxpnoaufj:Niceguynatezone2017$@aws-0-us-west-2.pooler.supabase.com:6543/postgres"
NEW_DB_URL = os.environ.get('DATABASE_URL')

print("=" * 70)
print("MIGRATION: Supabase -> CockroachDB (via CSV)")
print("=" * 70)

# Parse URLs
def parse_db_url(url):
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'user': parsed.username,
        'password': parsed.password,
        'database': parsed.path.lstrip('/'),
        'sslmode': 'allow' if 'sslmode' not in url else 'require'
    }

old_params = parse_db_url(OLD_DB_URL)
new_params = parse_db_url(NEW_DB_URL)
old_params['sslmode'] = 'allow'

print("\n[1/5] Exporting data from Supabase...")
try:
    old_conn = psycopg2.connect(**old_params)

    # Read stocks (exclude id column, it will be auto-generated)
    print("  - Reading stocks...")
    stocks_df = pd.read_sql_query("SELECT * FROM stocks", old_conn)
    stocks_df = stocks_df.drop('id', axis=1)  # Drop id column
    print(f"    Loaded {len(stocks_df)} records")

    # Read price history in chunks
    print("  - Reading price history...")
    chunk_size = 100000
    price_chunks = []
    offset = 0

    while True:
        chunk = pd.read_sql_query(
            f"SELECT ticker, date, open, high, low, close, volume FROM price_history ORDER BY id LIMIT {chunk_size} OFFSET {offset}",
            old_conn
        )
        if len(chunk) == 0:
            break
        price_chunks.append(chunk)
        print(f"    Loaded {offset + len(chunk):,} records")
        offset += chunk_size

    price_df = pd.concat(price_chunks, ignore_index=True) if price_chunks else pd.DataFrame()
    old_conn.close()
    print(f"  Total price records: {len(price_df):,}")

except Exception as e:
    print(f"[ERROR] Failed to export from Supabase: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[2/5] Saving to CSV...")
try:
    stocks_df.to_csv('stocks_export.csv', index=False)
    price_df.to_csv('price_history_export.csv', index=False)
    print(f"  - stocks_export.csv ({len(stocks_df)} rows)")
    print(f"  - price_history_export.csv ({len(price_df)} rows)")
except Exception as e:
    print(f"[ERROR] Failed to save CSV: {e}")
    exit(1)

print("\n[3/5] Connecting to CockroachDB...")
try:
    new_conn = psycopg2.connect(**new_params)
    new_cur = new_conn.cursor()
    print("[OK] Connected")
except Exception as e:
    print(f"[ERROR] Failed to connect: {e}")
    exit(1)

print("\n[4/5] Recreating tables...")
try:
    # Drop and recreate
    new_cur.execute("DROP TABLE IF EXISTS price_history CASCADE")
    new_cur.execute("DROP TABLE IF EXISTS stocks CASCADE")

    new_cur.execute("""
        CREATE TABLE stocks (
            ticker VARCHAR(20),
            index_name VARCHAR(50),
            company_name VARCHAR(255),
            sector VARCHAR(100),
            industry VARCHAR(100),
            status VARCHAR(50),
            last_date DATE,
            last_close FLOAT,
            pe_ratio FLOAT,
            data_points INTEGER,
            rising_3day BOOLEAN,
            declining_3day BOOLEAN,
            rising_7day BOOLEAN,
            declining_7day BOOLEAN,
            rising_14day BOOLEAN,
            declining_14day BOOLEAN,
            rising_21day BOOLEAN,
            declining_21day BOOLEAN,
            pct_1d FLOAT,
            pct_3d FLOAT,
            pct_5d FLOAT,
            pct_21d FLOAT,
            pct_63d FLOAT,
            pct_252d FLOAT,
            ann_vol_pct FLOAT,
            rsi FLOAT,
            high_52w FLOAT,
            low_52w FLOAT,
            pct_from_52w_high FLOAT,
            pct_from_52w_low FLOAT,
            avg_volume_20d FLOAT,
            volume_vs_avg FLOAT,
            updated_at TIMESTAMP
        )
    """)

    new_cur.execute("""
        CREATE TABLE price_history (
            ticker VARCHAR(20),
            date DATE,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT
        )
    """)

    new_conn.commit()
    print("[OK] Tables created")
except Exception as e:
    new_conn.rollback()
    print(f"[ERROR] Failed to create tables: {e}")
    exit(1)

print("\n[5/5] Importing from CSV...")
try:
    # Import stocks
    print("  - Importing stocks...")
    with open('stocks_export.csv', 'r') as f:
        new_cur.copy_expert("COPY stocks FROM STDIN WITH (FORMAT CSV, HEADER)", f)
    new_conn.commit()
    print(f"    Imported {len(stocks_df):,} stocks")

    # Import price history
    print("  - Importing price history...")
    with open('price_history_export.csv', 'r') as f:
        new_cur.copy_expert("COPY price_history FROM STDIN WITH (FORMAT CSV, HEADER)", f)
    new_conn.commit()
    print(f"    Imported {len(price_df):,} price records")

    new_conn.close()

    print("\n" + "=" * 70)
    print("[SUCCESS] Migration complete!")
    print("=" * 70)

except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
