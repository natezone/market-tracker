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
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name='stocks'
    ORDER BY ordinal_position
""")

print("Stocks table columns:")
for col, dtype in cur.fetchall():
    print(f"  {col}: {dtype}")

conn.close()
