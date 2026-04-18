#!/usr/bin/env python
"""Refresh PostgreSQL with latest data from CSV files"""

from market_tracker import run_cli

indices = ['SP500', 'NASDAQ100', 'COMBINED', 'DOW30', 'SP400', 'SP600']

print("🔄 Refreshing PostgreSQL with latest data...\n")

for index in indices:
    print(f"\n{'='*60}")
    print(f"Processing: {index}")
    print(f"{'='*60}")
    try:
        run_cli(consecutive_days=7, index_key=index)
        print(f"✅ {index} complete")
    except Exception as e:
        print(f"❌ {index} failed: {e}")

print("\n" + "="*60)
print("✅ PostgreSQL refresh complete!")
print("="*60)
