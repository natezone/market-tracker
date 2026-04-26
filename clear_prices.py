#!/usr/bin/env python3
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Parse URL
from urllib.parse import urlparse
parsed = urlparse(os.environ.get('DATABASE_URL'))
params = {
    'host': parsed.hostname,
    'port': parsed.port or 5432,
    'user': parsed.username,
    'password': parsed.password,
    'database': parsed.path.lstrip('/'),
    'sslmode': 'require'
}

conn = psycopg2.connect(**params)
cur = conn.cursor()

print("Clearing price_history table...")
cur.execute("DELETE FROM price_history")
conn.commit()

# Check counts
cur.execute("SELECT COUNT(*) FROM stocks")
stocks = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM price_history")
prices = cur.fetchone()[0]

print(f"[OK] Stocks: {stocks}, Price history: {prices}")
conn.close()
