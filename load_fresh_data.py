#!/usr/bin/env python3
"""
Load fresh market data directly into CockroachDB
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

# Import from market_tracker
from market_tracker import load_csv_data_to_postgres

print("=" * 70)
print("Loading Fresh Market Data to CockroachDB")
print("=" * 70)

try:
    load_csv_data_to_postgres()
    print("\n" + "=" * 70)
    print("[SUCCESS] Fresh data loaded to CockroachDB!")
    print("=" * 70)
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
