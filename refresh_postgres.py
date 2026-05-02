#!/usr/bin/env python3
"""
Refresh PostgreSQL with latest market data
Runs the market tracker CLI in data refresh mode
"""

import sys
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if DATABASE_URL is set
if not os.environ.get('DATABASE_URL'):
    print("❌ DATABASE_URL environment variable not found")
    sys.exit(1)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Refresh PostgreSQL market data')
parser.add_argument(
    '--index',
    default='ALL',
    choices=['ALL', 'SP500', 'SP400', 'SP600', 'NASDAQ100', 'DOW30', 'COMBINED'],
    help='Market index to refresh (default: ALL)'
)
parser.add_argument(
    '--consecutive-days',
    type=int,
    default=7,
    help='Consecutive days for trend analysis (default: 7)'
)

args = parser.parse_args()

print("=" * 60)
print(f"MARKET DATA REFRESH - {args.index}")
print("=" * 60)

# Import and run market_tracker CLI
from market_tracker import run_cli

# Run CLI with specified index
try:
    print(f"\nStarting data refresh for {args.index}...\n")
    run_cli(consecutive_days=args.consecutive_days, index_key=args.index)
    print("\n" + "=" * 60)
    print("✅ Data refresh completed successfully!")
    print("=" * 60)
except Exception as e:
    print(f"\n❌ Error during data refresh: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
