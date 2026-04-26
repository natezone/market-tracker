from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv('DATABASE_URL')

if db_url:
    engine = create_engine(db_url)
    
    # Calculate cutoff date (3 years ago)
    cutoff_date = (datetime.now() - timedelta(days=3*365)).date()
    print(f"[INFO] Removing price history older than {cutoff_date}...")
    
    with engine.begin() as conn:
        # Count rows before
        result = conn.execute(text("SELECT COUNT(*) FROM price_history"))
        before = result.scalar()
        
        # Delete old data
        conn.execute(text(f"DELETE FROM price_history WHERE date < '{cutoff_date}'"))
        
        # Count rows after
        result = conn.execute(text("SELECT COUNT(*) FROM price_history"))
        after = result.scalar()
        
        removed = before - after
        print(f"[OK] Removed {removed:,} rows")
        print(f"[OK] Remaining rows: {after:,}")
    
    print(f"\n[SUCCESS] Database cleanup complete!")
    print(f"Now storing only 3 years of price history (since {cutoff_date})")
else:
    print("[ERROR] No DATABASE_URL found")
