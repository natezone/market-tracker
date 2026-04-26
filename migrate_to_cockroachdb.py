#!/usr/bin/env python3
"""
Migrate data from Supabase PostgreSQL to CockroachDB
"""
import os
import sys
from dotenv import load_dotenv
import psycopg2
import psycopg2.errors

load_dotenv()

# Old Supabase PostgreSQL
OLD_DB_URL = "postgresql://postgres.htodaqaeaithxpnoaufj:Niceguynatezone2017$@aws-0-us-west-2.pooler.supabase.com:6543/postgres"

# New CockroachDB
NEW_DB_URL = os.environ.get('DATABASE_URL')

print("=" * 70)
print("MIGRATION: Supabase PostgreSQL -> CockroachDB (Full Data)")
print("=" * 70)

# Parse URLs and extract connection params
def parse_db_url(url):
    """Parse database URL to connection parameters"""
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

# For Supabase, use allow mode due to cert issues on Windows
old_params['sslmode'] = 'allow'

print("\n[1/4] Connecting to Supabase PostgreSQL...")
try:
    old_conn = psycopg2.connect(**old_params)
    print("[OK] Connected to Supabase")
except Exception as e:
    print(f"[ERROR] Failed to connect to Supabase: {e}")
    sys.exit(1)

print("\n[2/4] Connecting to CockroachDB...")
try:
    new_conn = psycopg2.connect(**new_params)
    print("[OK] Connected to CockroachDB")
except Exception as e:
    print(f"[ERROR] Failed to connect to CockroachDB: {e}")
    sys.exit(1)

# Clear and recreate tables
print("\n[3/4] Clearing tables in CockroachDB...")
try:
    cur = new_conn.cursor()
    cur.execute("DROP TABLE IF EXISTS price_history CASCADE")
    cur.execute("DROP TABLE IF EXISTS stocks CASCADE")
    new_conn.commit()
    print("[OK] Tables cleared")

    # Create stocks table (matching Supabase schema)
    cur.execute("""
        CREATE TABLE stocks (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            index_name VARCHAR(50) NOT NULL,
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

    # Create price_history table (without unique constraint for bulk insert)
    cur.execute("""
        CREATE TABLE price_history (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
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
    sys.exit(1)

# Migrate data
print("\n[4/4] Migrating data...")
try:
    old_cur = old_conn.cursor()
    new_cur = new_conn.cursor()

    # Migrate stocks table
    print("  - Migrating stocks table...")
    old_cur.execute("SELECT * FROM stocks")
    stocks_data = old_cur.fetchall()
    stocks_count = len(stocks_data)
    print(f"    Found {stocks_count} stock records")

    if stocks_count > 0:
        col_names = [desc[0] for desc in old_cur.description]
        placeholders = ','.join(['%s'] * len(col_names))
        col_str = ','.join(col_names)
        insert_query = f"INSERT INTO stocks ({col_str}) VALUES ({placeholders})"

        for i, row in enumerate(stocks_data):
            try:
                new_cur.execute(insert_query, row)
            except Exception as e:
                print(f"    Warning: Row {i} failed: {e}")

        new_conn.commit()
        print(f"    [OK] Saved {stocks_count} stocks to CockroachDB")

    # Migrate price_history table
    print("  - Migrating price_history table...")
    old_cur.execute("SELECT COUNT(*) FROM price_history")
    price_count = old_cur.fetchone()[0]
    print(f"    Total price history records: {price_count:,}")

    chunk_size = 50000
    offset = 0
    total_migrated = 0

    while offset < price_count:
        try:
            old_cur.execute(f"""
                SELECT ticker, date, open, high, low, close, volume
                FROM price_history
                ORDER BY id
                LIMIT {chunk_size} OFFSET {offset}
            """)

            rows = old_cur.fetchall()
            if not rows:
                break

            insert_query = """
                INSERT INTO price_history (ticker, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            failed = 0
            for row in rows:
                try:
                    new_cur.execute(insert_query, row)
                except Exception as e:
                    failed += 1

            new_conn.commit()
            total_migrated += len(rows) - failed
            offset += chunk_size
            print(f"    Processed {offset:,}/{price_count:,} ({total_migrated:,} saved, {failed} failed)", flush=True)

        except psycopg2.InterfaceError as e:
            # Reconnect if connection dropped
            print(f"    Connection lost: {e}, reconnecting...", flush=True)
            try:
                new_conn.close()
            except:
                pass
            new_conn = psycopg2.connect(**new_params)
            new_cur = new_conn.cursor()

    print(f"    [OK] Migrated {total_migrated:,} price history records to CockroachDB")

except Exception as e:
    print(f"[ERROR] Migration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] Migration complete!")
print("=" * 70)
print(f"Migrated: {stocks_count:,} stocks + {total_migrated:,} price history records")

old_cur.close()
new_cur.close()
old_conn.close()
new_conn.close()
