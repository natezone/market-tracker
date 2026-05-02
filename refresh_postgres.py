#!/usr/bin/env python3
"""
Refresh PostgreSQL with latest market data
Runs the market tracker CLI in data refresh mode for all indices
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if DATABASE_URL is set
if not os.environ.get('DATABASE_URL'):
    print("❌ DATABASE_URL environment variable not found")
    sys.exit(1)

print("=" * 60)
print("MARKET DATA REFRESH - ALL INDICES")
print("=" * 60)

# Import and run market_tracker CLI
from market_tracker import run_cli

# Run CLI for all indices
try:
    print("\nStarting data refresh for all market indices...\n")
    run_cli(consecutive_days=7, index_key="ALL")
    print("\n" + "=" * 60)
    print("✅ Data refresh completed successfully!")
    print("=" * 60)
except Exception as e:
    print(f"\n❌ Error during data refresh: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
