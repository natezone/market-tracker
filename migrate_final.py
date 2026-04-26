#!/usr/bin/env python3
"""
Final migration: Supabase -> CockroachDB
"""
import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

OLD_DB_URL = "postgresql://postgres.htodaqaeaithxpnoaufj:Niceguynatezone2017$@aws-0-us-west-2.pooler.supabase.com:6543/postgres"
NEW_DB_URL = os.environ.get('DATABASE_URL')

print("=" * 70)
print("MIGRATION: Supabase -> CockroachDB (Final)")
print("=" * 70)

# Modify URLs for SQLAlchemy
old_sa_url = OLD_DB_URL.replace('postgres://', 'postgresql://') if 'postgres://' in OLD_DB_URL else OLD_DB_URL

print("\n[1/4] Exporting data from Supabase...")
try:
    old_engine = create_engine(old_sa_url, connect_args={"sslmode": "allow"})

    # Read stocks
    print("  - Reading stocks...")
    stocks_df = pd.read_sql_table('stocks', old_engine)
    stocks_df = stocks_df.drop('id', axis=1)
    print(f"    Loaded {len(stocks_df)} records")

    # Read price history
    print("  - Reading price history...")
    price_df = pd.read_sql_table('price_history', old_engine)
    price_df = price_df.drop('id', axis=1)  # Drop id column
    print(f"    Loaded {len(price_df)} records")

    old_engine.dispose()

except Exception as e:
    print(f"[ERROR] Export failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Parse CockroachDB URL
from urllib.parse import urlparse
parsed = urlparse(NEW_DB_URL)
new_params = {
    'host': parsed.hostname,
    'port': parsed.port or 5432,
    'user': parsed.username,
    'password': parsed.password,
    'database': parsed.path.lstrip('/'),
    'sslmode': 'require'
}

print("\n[2/4] Connecting to CockroachDB...")
try:
    conn = psycopg2.connect(**new_params)
    cur = conn.cursor()
    print("[OK] Connected")
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    exit(1)

print("\n[3/4] Recreating tables...")
try:
    cur.execute("DROP TABLE IF EXISTS price_history CASCADE")
    cur.execute("DROP TABLE IF EXISTS stocks CASCADE")

    cur.execute("""
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

    cur.execute("""
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

    conn.commit()
    print("[OK] Tables created")
except Exception as e:
    conn.rollback()
    print(f"[ERROR] Failed to create tables: {e}")
    exit(1)

print("\n[4/4] Importing data...")
try:
    # Import stocks (convert NaT/NaN to None)
    print("  - Importing stocks...")
    # Replace NaT and NaN with None
    stocks_df = stocks_df.astype(object).where(pd.notna(stocks_df), None)
    stocks_data = [tuple(None if pd.isna(x) else x for x in row) for row in stocks_df.values]
    insert_query = f"""
        INSERT INTO stocks ({','.join(stocks_df.columns)})
        VALUES ({','.join(['%s'] * len(stocks_df.columns))})
    """
    cur.executemany(insert_query, stocks_data)
    conn.commit()
    print(f"    Saved {len(stocks_data):,} stocks")

    # Import price history (convert NaT/NaN to None)
    print("  - Importing price history...")
    price_df = price_df.astype(object).where(pd.notna(price_df), None)
    price_data = [tuple(None if pd.isna(x) else x for x in row) for row in price_df.values]
    insert_query = f"""
        INSERT INTO price_history ({','.join(price_df.columns)})
        VALUES ({','.join(['%s'] * len(price_df.columns))})
    """

    # Insert in chunks to avoid memory issues
    chunk_size = 10000
    for i in range(0, len(price_data), chunk_size):
        chunk = price_data[i:i+chunk_size]
        cur.executemany(insert_query, chunk)
        conn.commit()
        print(f"    Saved {i + len(chunk):,}/{len(price_data):,} records")

    conn.close()

    print("\n" + "=" * 70)
    print("[SUCCESS] Migration complete!")
    print("=" * 70)
    print(f"Imported: {len(stocks_data):,} stocks + {len(price_data):,} price records")

except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
