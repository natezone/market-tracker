#!/usr/bin/env python
"""Load existing CSV data into PostgreSQL"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from market_tracker import PostgreSQLManager

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
indices = ['SP500', 'NASDAQ100', 'COMBINED', 'DOW30', 'SP400', 'SP600']

# Create SQLAlchemy engine
database_url = os.getenv('DATABASE_URL')
if not database_url:
    print("❌ DATABASE_URL not found in .env")
    exit(1)

engine = create_engine(database_url)
pg_manager = PostgreSQLManager(engine)
print("🔄 Loading CSV data into PostgreSQL...\n")

for index in indices:
    index_dir = os.path.join(DATA_DIR, index)
    metrics_file = os.path.join(index_dir, 'latest_metrics.csv')
    history_dir = os.path.join(index_dir, 'history')

    if not os.path.exists(metrics_file):
        print(f"⚠️  Metrics file not found: {metrics_file}")
        continue

    print(f"\n{'='*60}")
    print(f"Processing: {index}")
    print(f"{'='*60}")

    try:
        # Load metrics
        print(f"Loading metrics from {metrics_file}...")
        metrics_df = pd.read_csv(metrics_file)
        print(f"  Loaded {len(metrics_df)} metrics records")

        # Save metrics to PostgreSQL
        pg_manager.save_metrics(metrics_df, index)
        print(f"  ✅ Saved metrics to PostgreSQL")

        # Load price history for each ticker
        if os.path.exists(history_dir):
            history_files = [f for f in os.listdir(history_dir) if f.endswith('.csv')]
            print(f"Loading {len(history_files)} price history files...")

            for i, history_file in enumerate(history_files, 1):
                ticker = history_file.replace('.csv', '')
                history_path = os.path.join(history_dir, history_file)

                try:
                    price_df = pd.read_csv(history_path, index_col='Date', parse_dates=True)
                    pg_manager.save_price_history(ticker, price_df)
                except Exception as e:
                    # Skip duplicate key errors and continue
                    if 'duplicate' not in str(e).lower() and 'unique' not in str(e).lower():
                        print(f"    Error loading {ticker}: {e}")

                # Progress indicator
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(history_files)} price history files")

            print(f"  ✅ Loaded all price history files")

        print(f"✅ {index} complete")

    except Exception as e:
        print(f"❌ {index} failed: {e}")

print("\n" + "="*60)
print("✅ PostgreSQL population from CSV complete!")
print("="*60)
