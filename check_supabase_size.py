#!/usr/bin/env python3
import psycopg2

conn = psycopg2.connect(
    host='aws-0-us-west-2.pooler.supabase.com',
    port=6543,
    user='postgres.htodaqaeaithxpnoaufj',
    password='Niceguynatezone2017$',
    database='postgres',
    sslmode='allow'
)

cur = conn.cursor()

# Check price history count
cur.execute("SELECT COUNT(*) FROM price_history")
count = cur.fetchone()[0]
print(f"Supabase price_history records: {count:,}")

# Check a few records
cur.execute("""
    SELECT ticker, date, COUNT(*)
    FROM price_history
    GROUP BY ticker, date
    HAVING COUNT(*) > 1
    LIMIT 5
""")
duplicates = cur.fetchall()
if duplicates:
    print(f"Found {len(duplicates)} duplicate ticker-date combinations")

conn.close()
