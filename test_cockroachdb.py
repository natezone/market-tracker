#!/usr/bin/env python3
import psycopg2
import sys

db_url = "postgresql://ehis:ZYjErHst77b78mb9ENC_Cw@blunt-serpent-15319.jxf.gcp-us-central1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"

# Parse URL
from urllib.parse import urlparse
parsed = urlparse(db_url)

params = {
    'host': parsed.hostname,
    'port': parsed.port or 5432,
    'user': parsed.username,
    'password': parsed.password,
    'database': parsed.path.lstrip('/'),
    'sslmode': 'require'
}

print(f"Connecting to {params['host']}:{params['port']}...", flush=True)

try:
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    print("[OK] Connected!", flush=True)

    # Check tables
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    tables = cur.fetchall()
    print(f"[INFO] Tables in database: {[t[0] for t in tables]}", flush=True)

    # Check stocks count
    cur.execute("SELECT COUNT(*) FROM stocks")
    count = cur.fetchone()[0]
    print(f"[INFO] Stocks in table: {count}", flush=True)

    conn.close()
    print("[SUCCESS] Connection test passed!", flush=True)
except Exception as e:
    print(f"[ERROR] {e}", flush=True)
    sys.exit(1)
