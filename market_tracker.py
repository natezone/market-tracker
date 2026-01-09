from email import parser
import os
import sys
import time
import math
import argparse
from datetime import datetime, timedelta
from venv import logger
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
from datetime import timezone, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from textblob import TextBlob
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, MetaData, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import os
from stock_recommendations import (
    generate_recommendations,
    calculate_stock_score,
    build_diversified_portfolio,
    InvestmentStrategy,
    RecommendationLevel,
    ScoringWeights,
    MarketRegime,
    ConfidenceLevel,
    MarketRegimeData,              
    RegimeAdjustmentMultipliers,   
    TimeHorizon,                   
    CatalystType,                  
    RiskManagementGuidance,        
    detect_market_regime,
    calculate_market_breadth
)

try:
    from enhanced_data_fetcher import (
        fetch_all_enhanced_data,
        fetch_enhanced_data_batch,
        merge_enhanced_with_base_metrics
    )
    ENHANCED_DATA_AVAILABLE = True
except ImportError:
    ENHANCED_DATA_AVAILABLE = False
    print("Note: Enhanced data fetcher not available. Some features will be limited.")

# Load environment variables
load_dotenv()

# Try importing Streamlit and Plotly for web interface
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None
    print("Note: Streamlit/Plotly not installed. Web interface unavailable.")

# One-time flag to avoid spamming warnings when cache is unavailable
_cache_cache_warning_logged = False

# Provide a safe no-op cache decorator when Streamlit isn't available
if STREAMLIT_AVAILABLE:
    cache_data = st.cache_data
else:
    def cache_data(ttl=None, show_spinner=True):
        global _cache_cache_warning_logged
        if not _cache_cache_warning_logged:
            print("Note: Streamlit cache unavailable; caching disabled. Install streamlit for caching.")
            _cache_cache_warning_logged = True
        def decorator(func):
            return func
        return decorator

# ---------------------------
# Configuration
# ---------------------------
UNIVERSE = "SP500"
DATA_DIR = "data"
BATCH_SIZE = 50
MIN_CONSECUTIVE = 30
DOWNLOAD_PERIOD = "max"
VERBOSE = True
MIN_MARKET_CAP = 0

# Sentiment Analysis Configuration
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', None)
sentiment_analyzer = SentimentIntensityAnalyzer()

# ---------------------------
# Shared Helper Functions
# ---------------------------
def ensure_dir(d):
    """Create directory if it doesn't exist"""
    os.makedirs(d, exist_ok=True)

# PostgreSQL Database configuration
POSTGRES_URL = os.environ.get('DATABASE_URL') 

if POSTGRES_URL:
    if POSTGRES_URL.startswith('postgres://'):
        POSTGRES_URL = POSTGRES_URL.replace('postgres://', 'postgresql://', 1)
    
    try:
        engine = create_engine(
            POSTGRES_URL,
            connect_args={
                "sslmode": "require",  # Supabase requires SSL
            },
            pool_pre_ping=True,  
            pool_recycle=300, 
            pool_size=5, 
            max_overflow=10
        )
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ PostgreSQL engine created successfully")
        
    except Exception as e:
        print(f"⚠️ PostgreSQL connection failed: {str(e)[:200]}")
        print("📁 Will use CSV fallback mode")
        engine = None
else:
    print("ℹ️ No DATABASE_URL found. Using CSV mode.")
    engine = None

class PostgreSQLManager:
    """Manage PostgreSQL database operations"""
    
    def __init__(self, engine):
        self.engine = engine
        self.metadata = MetaData()
        self._tables_created = False
    
    def _create_tables(self):
        """Create tables if they don't exist"""
        if not self.engine:
            return
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    index_name VARCHAR(50) NOT NULL,
                    company_name VARCHAR(255),
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    status VARCHAR(20),
                    last_date DATE,
                    last_close FLOAT,
                    pe_ratio FLOAT,
                    data_points INTEGER,
                    rising_3day BOOLEAN,
                    declining_3day BOOLEAN,
                    rising_7day BOOLEAN,
                    declining_7day BOOLEAN,
                    rising_14day BOOLEAN,
                    declining_14day BOOLEAN,
                    rising_21day BOOLEAN,
                    declining_21day BOOLEAN,
                    pct_1d FLOAT,
                    pct_3d FLOAT,
                    pct_5d FLOAT,
                    pct_21d FLOAT,
                    pct_63d FLOAT,
                    pct_252d FLOAT,
                    ann_vol_pct FLOAT,
                    rsi FLOAT,
                    high_52w FLOAT,
                    low_52w FLOAT,
                    pct_from_52w_high FLOAT,
                    pct_from_52w_low FLOAT,
                    avg_volume_20d FLOAT,
                    volume_vs_avg FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, index_name)
                );
                
                -- Existing indexes
                CREATE INDEX IF NOT EXISTS idx_ticker ON stocks(ticker);
                CREATE INDEX IF NOT EXISTS idx_index ON stocks(index_name);
                CREATE INDEX IF NOT EXISTS idx_sector ON stocks(sector);
                CREATE INDEX IF NOT EXISTS idx_pct_21d ON stocks(pct_21d);

                -- Performance optimization indexes
                CREATE INDEX IF NOT EXISTS idx_composite_score ON stocks(composite_score DESC NULLS LAST);
                CREATE INDEX IF NOT EXISTS idx_sector_score ON stocks(sector, pct_21d DESC);
                CREATE INDEX IF NOT EXISTS idx_updated_at ON stocks(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_status ON stocks(status) WHERE status = 'ok';
                CREATE INDEX IF NOT EXISTS idx_rising_declining ON stocks(rising_7day, declining_7day) WHERE status = 'ok';

                -- Composite index for common dashboard query
                CREATE INDEX IF NOT EXISTS idx_dashboard_query ON stocks(index_name, status, pct_21d DESC) 
                    WHERE status = 'ok';
            """))
    
    def save_metrics(self, metrics_df, index_name):
        """Save metrics to PostgreSQL"""
        if not self.engine or metrics_df.empty:
            return
        
        try:
            # Make a clean copy
            df = metrics_df.copy()
            
            # Add index_name column
            df['index_name'] = index_name
            
            # Rename 52w columns if they exist
            if '52w_high' in df.columns:
                df = df.rename(columns={
                    '52w_high': 'high_52w',
                    '52w_low': 'low_52w'
                })
            
            # Define valid columns that match database schema
            valid_columns = [
                'ticker', 'index_name', 'company_name', 'sector', 'industry', 'status',
                'last_date', 'last_close', 'pe_ratio', 'data_points',
                'rising_3day', 'declining_3day', 'rising_7day', 'declining_7day',
                'rising_14day', 'declining_14day', 'rising_21day', 'declining_21day',
                'pct_1d', 'pct_3d', 'pct_5d', 'pct_21d', 'pct_63d', 'pct_252d',
                'ann_vol_pct', 'rsi', 'high_52w', 'low_52w', 'pct_from_52w_high',
                'pct_from_52w_low', 'avg_volume_20d', 'volume_vs_avg'
            ]
            
            # Keep only valid columns that exist in DataFrame
            df = df[[col for col in valid_columns if col in df.columns]]
            
            print(f"📊 Saving {len(df)} stocks with {len(df.columns)} columns to PostgreSQL...")
            
            # Delete existing records for this index
            with self.engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM stocks WHERE index_name = :idx"),
                    {"idx": index_name}
                )
            
            # Insert new data (suppress verbose output)
            df.to_sql(
                'stocks',
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=100
            )
            
            # Verify the save
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM stocks WHERE index_name = :idx"),
                    {"idx": index_name}
                )
                count = result.scalar()
                
                if count > 0:
                    print(f"✓ Saved {count} stocks to PostgreSQL for {index_name}")
                else:
                    print(f"⚠️ Warning: 0 rows in database after save!")
            
        except Exception as e:
            print(f"❌ Error saving to PostgreSQL: {e}")
            import traceback
            traceback.print_exc()
    
    def load_metrics(self, index_name):
        """Load metrics from PostgreSQL"""
        if not self.engine:
            return pd.DataFrame()
        
        try:
            query = text("""
                SELECT * FROM stocks 
                WHERE index_name = :idx 
                ORDER BY ticker
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"idx": index_name})
            
            print(f"🔍 DEBUG load_metrics: Loaded {len(df)} rows for {index_name}")
            
            if df.empty:
                print(f"⚠️ WARNING: No data found for index_name = '{index_name}'")
                print(f"   Checking what's in the database...")
                
                # Check what index_names exist
                with self.engine.connect() as conn:
                    check = conn.execute(text("SELECT DISTINCT index_name, COUNT(*) FROM stocks GROUP BY index_name"))
                    print(f"   Available indices:")
                    for row in check:
                        print(f"     - {row[0]}: {row[1]} stocks")
            
            # Drop PostgreSQL-specific columns
            df = df.drop(columns=['id', 'updated_at'], errors='ignore')
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading from PostgreSQL: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_stats(self):
        """Get database statistics"""
        if not self.engine:
            return {}
        
        try:
            with self.engine.connect() as conn:
                # Total stocks
                result = conn.execute(text("SELECT COUNT(*) FROM stocks"))
                total = result.scalar()
                
                # Stocks by index
                result = conn.execute(text("""
                    SELECT index_name, COUNT(*) as count 
                    FROM stocks 
                    GROUP BY index_name
                """))
                by_index = [{"index_name": row[0], "count": row[1]} for row in result]
                
                # Last update
                result = conn.execute(text("""
                    SELECT MAX(updated_at) FROM stocks
                """))
                last_update = result.scalar()
                
                return {
                    'total_stocks': total,
                    'stocks_by_index': by_index,
                    'last_update': str(last_update) if last_update else None
                }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

# Initialize PostgreSQL manager
pg_manager = None

if engine:
    try:
        pg_manager = PostgreSQLManager(engine)
        print("✅ PostgreSQL manager initialized")
    except Exception as e:
        print(f"⚠️ PostgreSQL init failed: {str(e)[:200]}")
        print("📁 Falling back to CSV mode")
        engine = None
        pg_manager = None

if not pg_manager:
    print("ℹ️ Running in CSV-only mode")

def sanitize_filename_windows(filename):
    """
    Sanitize filename for Windows compatibility.
    Renames reserved names like CON, PRN, AUX, NUL, COM1-9, LPT1-9
    """
    # Windows reserved names
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    # Get the base name without extension
    base_name = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1]
    
    if base_name.upper() in reserved_names:
        return f"{base_name}_stock{extension}"
    return filename

def fetch_sp500_tickers():
    """Scrape S&P500 tickers from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()

        tables = pd.read_html(r.text)
        if not tables:
            raise RuntimeError("No tables found on S&P500 Wikipedia page")

        df = tables[0]

        # Standardize ticker symbols
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        tickers = df['Symbol'].tolist()

        # Extract sector, industry, and company name info
        sectors = dict(zip(df['Symbol'], df.get('GICS Sector', df['Symbol'])))
        industries = dict(zip(df['Symbol'], df.get('GICS Sub-Industry', df['Symbol'])))
        
        # Try different column names for company
        if 'Security' in df.columns:
            company_names = dict(zip(df['Symbol'], df['Security']))
        elif 'Company' in df.columns:
            company_names = dict(zip(df['Symbol'], df['Company']))
        else:
            company_names = dict(zip(df['Symbol'], df['Symbol']))

        return tickers, df, sectors, industries, company_names  

    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return [], pd.DataFrame(), {}, {}, {}  

def fetch_sp400_tickers():
    """Scrape S&P MidCap 400 tickers from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()

        tables = pd.read_html(r.text)
        if not tables:
            raise RuntimeError("No tables found on S&P 400 Wikipedia page")

        df = tables[0]

        # Standardize ticker symbols
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        tickers = df['Symbol'].tolist()

        # Extract sector, industry, and company name info with fallbacks
        sectors = dict(zip(df['Symbol'], df.get('GICS Sector', ['Unknown'] * len(df))))
        industries = dict(zip(df['Symbol'], df.get('GICS Sub-Industry', ['Unknown'] * len(df))))
        
        # Try different column names for company
        if 'Security' in df.columns:
            company_names = dict(zip(df['Symbol'], df['Security']))
        elif 'Company' in df.columns:
            company_names = dict(zip(df['Symbol'], df['Company']))
        else:
            company_names = dict(zip(df['Symbol'], df['Symbol']))

        return tickers, df, sectors, industries, company_names

    except Exception as e:
        print(f"Error fetching S&P 400 tickers: {e}")
        return [], pd.DataFrame(), {}, {}, {}

def fetch_sp600_tickers():
    """Scrape S&P SmallCap 600 tickers from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()

        tables = pd.read_html(r.text)
        if not tables:
            raise RuntimeError("No tables found on S&P 600 Wikipedia page")

        df = tables[0]

        # Standardize ticker symbols
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        tickers = df['Symbol'].tolist()

        # Extract sector, industry, and company name info with fallbacks
        sectors = dict(zip(df['Symbol'], df.get('GICS Sector', ['Unknown'] * len(df))))
        industries = dict(zip(df['Symbol'], df.get('GICS Sub-Industry', ['Unknown'] * len(df))))
        
        # Try different column names for company
        if 'Security' in df.columns:
            company_names = dict(zip(df['Symbol'], df['Security']))
        elif 'Company' in df.columns:
            company_names = dict(zip(df['Symbol'], df['Company']))
        else:
            company_names = dict(zip(df['Symbol'], df['Symbol']))

        return tickers, df, sectors, industries, company_names

    except Exception as e:
        print(f"Error fetching S&P 600 tickers: {e}")
        return [], pd.DataFrame(), {}, {}, {}

def fetch_nasdaq100_tickers():
    """Scrape Nasdaq-100 tickers from Wikipedia with robust column detection"""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        
        tables = pd.read_html(r.text)
        if not tables:
            raise RuntimeError("No tables found on Nasdaq-100 Wikipedia page")
        
        # Try to find the components table
        df = None
        for i, table in enumerate(tables):
            # Skip None entries 
            if table is None:
                continue
            
            # Skip empty DataFrames
            if isinstance(table, pd.DataFrame) and table.empty:
                continue
                
            cols_str = [str(col).lower() for col in table.columns]
            if any('ticker' in col for col in cols_str):
                df = table
                break
        
        if df is None:
            # Fallback: find first non-None, non-empty DataFrame
            for table in tables:
                if table is not None and isinstance(table, pd.DataFrame) and not table.empty:
                    df = table
                    break
        
        if df is None or df.empty:
            raise RuntimeError("Could not find valid Nasdaq-100 components table")
        
        # Find ticker column
        ticker_col = None
        for col in df.columns:
            if 'ticker' in str(col).lower() or 'symbol' in str(col).lower():
                ticker_col = col
                break
        
        if ticker_col is None:
            raise RuntimeError(f"Could not find ticker column in: {list(df.columns)}")
        
        # Standardize ticker symbols
        df[ticker_col] = df[ticker_col].astype(str).str.replace('.', '-', regex=False)
        tickers = df[ticker_col].tolist()
        
        # Extract company names
        company_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if 'company' in col_str or ('name' in col_str and 'ticker' not in col_str):
                company_col = col
                break
        
        if company_col:
            company_names = dict(zip(df[ticker_col], df[company_col].astype(str)))
        else:
            company_names = {ticker: ticker for ticker in tickers}
        
        # Extract sector
        sector_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if 'sector' in col_str:
                sector_col = col
                break
        
        if sector_col:
            sectors = dict(zip(df[ticker_col], df[sector_col].astype(str)))
        else:
            sectors = {ticker: "Unknown" for ticker in tickers}
        
        # Extract industry
        industry_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if 'industry' in col_str or 'sub-industry' in col_str:
                industry_col = col
                break
        
        if industry_col:
            industries = dict(zip(df[ticker_col], df[industry_col].astype(str)))
        else:
            industries = {ticker: "Unknown" for ticker in tickers}
        
        print(f"Successfully fetched {len(tickers)} Nasdaq-100 tickers")
        
        return tickers, df, sectors, industries, company_names
        
    except Exception as e:
        print(f"Error fetching Nasdaq-100 tickers: {e}")
        import traceback
        traceback.print_exc()
        return [], pd.DataFrame(), {}, {}, {}

def fetch_dow30_tickers():
    """Scrape Dow 30 tickers from Wikipedia with robust column detection"""
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        
        tables = pd.read_html(r.text)
        if not tables:
            raise RuntimeError("No tables found on Dow Jones Wikipedia page")
        
        # Find the components table
        df = None
        for i, table in enumerate(tables):
            cols_str = [str(col).lower() for col in table.columns]
            if any(keyword in ' '.join(cols_str) for keyword in ['symbol', 'ticker', 'company']):
                df = table
                break
        
        if df is None:
            raise RuntimeError("Could not find Dow components table")
        
        # Find the ticker/symbol column
        ticker_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if 'symbol' in col_str or 'ticker' in col_str:
                ticker_col = col
                break
        
        if ticker_col is None:
            raise RuntimeError(f"Could not find ticker column in: {list(df.columns)}")
        
        # Standardize ticker symbols
        df[ticker_col] = df[ticker_col].astype(str).str.replace('.', '-', regex=False)
        tickers = df[ticker_col].tolist()
        
        # Extract company names
        company_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if 'company' in col_str or ('name' in col_str and 'ticker' not in col_str):
                company_col = col
                break
        
        if company_col:
            company_names = dict(zip(df[ticker_col], df[company_col].astype(str)))
        else:
            company_names = {ticker: ticker for ticker in tickers}
        
        # Extract industry/sector info
        industry_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if 'industry' in col_str or 'sector' in col_str:
                industry_col = col
                break
        
        if industry_col:
            industries = dict(zip(df[ticker_col], df[industry_col].astype(str)))
            sectors = industries.copy()  # Dow doesn't separate sector/industry
        else:
            industries = {ticker: "Unknown" for ticker in tickers}
            sectors = {ticker: "Unknown" for ticker in tickers}
        
        print(f"Successfully fetched {len(tickers)} Dow 30 tickers")
        
        return tickers, df, sectors, industries, company_names
        
    except Exception as e:
        print(f"Error fetching Dow 30 tickers: {e}")
        import traceback
        traceback.print_exc()
        return [], pd.DataFrame(), {}, {}, {}

def fetch_combined_tickers():
    """Fetch all tickers from S&P 500, MidCap 400, SmallCap 600, Nasdaq-100, and Dow 30 (deduplicated)"""
    all_tickers = []
    all_company_names = {}
    all_sectors = {}
    all_industries = {}
    all_sources = {}  # Track which index each ticker comes from
    
    # Fetch S&P 500
    sp_tickers, sp_df, sp_sectors, sp_industries, sp_names = fetch_sp500_tickers()
    for ticker in sp_tickers:
        all_tickers.append(ticker)
        all_company_names[ticker] = sp_names.get(ticker, "Unknown")
        all_sectors[ticker] = sp_sectors.get(ticker, "Unknown")
        all_industries[ticker] = sp_industries.get(ticker, "Unknown")
        all_sources[ticker] = "S&P 500"
    
    # Fetch S&P MidCap 400
    sp4_tickers, sp4_df, sp4_sectors, sp4_industries, sp4_names = fetch_sp400_tickers()
    for ticker in sp4_tickers:
        if ticker not in all_tickers:
            all_tickers.append(ticker)
            all_company_names[ticker] = sp4_names.get(ticker, ticker)
            all_sectors[ticker] = sp4_sectors.get(ticker, "Unknown")
            all_industries[ticker] = sp4_industries.get(ticker, "Unknown")
            all_sources[ticker] = "S&P MidCap 400"
        else:
            all_sources[ticker] += ", S&P MidCap 400"
    
    # Fetch S&P SmallCap 600
    sp6_tickers, sp6_df, sp6_sectors, sp6_industries, sp6_names = fetch_sp600_tickers()
    for ticker in sp6_tickers:
        if ticker not in all_tickers:
            all_tickers.append(ticker)
            all_company_names[ticker] = sp6_names.get(ticker, ticker)
            all_sectors[ticker] = sp6_sectors.get(ticker, "Unknown")
            all_industries[ticker] = sp6_industries.get(ticker, "Unknown")
            all_sources[ticker] = "S&P SmallCap 600"
        else:
            all_sources[ticker] += ", S&P SmallCap 600"
    
    # Fetch Nasdaq-100
    nq_tickers, nq_df, nq_sectors, nq_industries, nq_names = fetch_nasdaq100_tickers()
    for ticker in nq_tickers:
        if ticker not in all_tickers:
            all_tickers.append(ticker)
            all_company_names[ticker] = nq_names.get(ticker, "Unknown")
            all_sectors[ticker] = nq_sectors.get(ticker, "Unknown")
            all_industries[ticker] = nq_industries.get(ticker, "Unknown")
            all_sources[ticker] = "Nasdaq-100"
        else:
            all_sources[ticker] += ", Nasdaq-100"
    
    # Fetch Dow 30
    dow_tickers, dow_df, dow_sectors, dow_industries, dow_names = fetch_dow30_tickers()
    for ticker in dow_tickers:
        if ticker not in all_tickers:
            all_tickers.append(ticker)
            all_company_names[ticker] = dow_names.get(ticker, "Unknown")
            all_sectors[ticker] = dow_sectors.get(ticker, "Unknown")
            all_industries[ticker] = dow_industries.get(ticker, "Unknown")
            all_sources[ticker] = "Dow 30"
        else:
            all_sources[ticker] += ", Dow 30"
    
    # Create combined dataframe
    combined_df = pd.DataFrame({
        'Symbol': all_tickers,
        'Company': [all_company_names[t] for t in all_tickers],
        'GICS Sector': [all_sectors[t] for t in all_tickers],
        'GICS Sub-Industry': [all_industries[t] for t in all_tickers],
        'Source Index': [all_sources[t] for t in all_tickers]
    })
    
    return all_tickers, combined_df, all_sectors, all_industries, all_company_names

def batch(iterable, n=1):
    """Batch an iterable into chunks of size n"""
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:i+n]

def download_history(tickers, period="max", interval="1d", threads=True, progress=True):
    """Download historical data for multiple tickers with optimized batching"""
    if len(tickers) == 0:
        return {}

    out = {}
    
    # Optimize batch size based on number of tickers
    if len(tickers) < 100:
        batch_size = 25  # Smaller batches for better error handling
    else:
        batch_size = BATCH_SIZE

    for chunk in batch(tickers, batch_size):
        chunk_start_time = time.time()
        
        for attempt in range(3):
            try:
                data = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    threads=threads,
                    progress=progress,
                    timeout=30
                )
                break
            except Exception as e:
                wait = (attempt + 1) * 2
                if VERBOSE:
                    print(f"Download error (attempt {attempt + 1}/3), retrying in {wait}s: {e}")
                time.sleep(wait)
        else:
            print(f"❌ Failed to download chunk after 3 attempts: {chunk}")
            # Add empty DataFrames for failed tickers
            for t in chunk:
                out[t] = pd.DataFrame()
            continue

        # Process downloaded data
        if len(chunk) == 1:
            t = chunk[0]
            df = data.copy()
            if not df.empty:
                df.index = pd.to_datetime(df.index)
            out[t] = df
        else:
            for t in chunk:
                try:
                    # yfinance returns a multiindex DataFrame for multiple tickers
                    df = data[t].dropna(how='all')
                    df.index = pd.to_datetime(df.index)
                    out[t] = df
                except Exception:
                    out[t] = pd.DataFrame()
        
        # Rate limiting between chunks
        chunk_duration = time.time() - chunk_start_time
        if chunk_duration < 1.0:  # If chunk was too fast, slow down
            time.sleep(1.0 - chunk_duration)

    if VERBOSE:
        successful = sum(1 for df in out.values() if not df.empty)
        print(f"✓ Downloaded data for {successful}/{len(tickers)} tickers")

    return out

# ---------------------------
# Color Coding Functions
# ---------------------------

def get_performance_color(value, metric_type='return', use_gradient=True):
    """
    Get color based on performance metric
    
    Args:
        value: The metric value
        metric_type: Type of metric ('return', 'rsi', 'volatility', 'volume')
        use_gradient: Whether to use gradient colors or simple red/green
    
    Returns:
        Color string (hex code)
    """
    if pd.isna(value):
        return '#888888'  # Gray for missing data
    
    if metric_type == 'return':
        # For returns: negative = red, positive = green
        if value < -5:
            return '#8B0000' if use_gradient else '#FF0000'  # Dark red
        elif value < -2:
            return '#DC143C' if use_gradient else '#FF0000'  # Crimson
        elif value < -1:
            return '#FF6B6B' if use_gradient else '#FF0000'  # Light red
        elif value < 0:
            return '#FFB3B3' if use_gradient else '#FF0000'  # Very light red
        elif value == 0:
            return '#D3D3D3'  # Light gray
        elif value < 1:
            return '#B3FFB3' if use_gradient else '#00FF00'  # Very light green
        elif value < 2:
            return '#66FF66' if use_gradient else '#00FF00'  # Light green
        elif value < 5:
            return '#32CD32' if use_gradient else '#00FF00'  # Lime green
        else:
            return '#006400' if use_gradient else '#00FF00'  # Dark green
    
    elif metric_type == 'rsi':
        # For RSI: <30 = oversold (good buy), >70 = overbought (bad)
        if value < 20:
            return '#006400'  # Dark green (very oversold)
        elif value < 30:
            return '#32CD32'  # Green (oversold)
        elif value < 40:
            return '#90EE90'  # Light green
        elif value < 60:
            return '#FFFF99'  # Yellow (neutral)
        elif value < 70:
            return '#FFA500'  # Orange
        elif value < 80:
            return '#FF6347'  # Tomato
        else:
            return '#8B0000'  # Dark red (very overbought)
    
    elif metric_type == 'volatility':
        # For volatility: lower = better (green), higher = riskier (red)
        if value < 15:
            return '#006400' if use_gradient else '#00FF00'  # Dark green
        elif value < 25:
            return '#32CD32' if use_gradient else '#00FF00'  # Green
        elif value < 35:
            return '#90EE90' if use_gradient else '#FFFF00'  # Light green
        elif value < 50:
            return '#FFFF99'  # Yellow
        elif value < 75:
            return '#FFA500'  # Orange
        else:
            return '#8B0000' if use_gradient else '#FF0000'  # Dark red
    
    elif metric_type == 'volume':
        # For volume vs average: higher = more interest
        if value > 100:
            return '#006400'  # Dark green (high volume)
        elif value > 50:
            return '#32CD32'  # Green
        elif value > 20:
            return '#90EE90'  # Light green
        elif value > -20:
            return '#FFFF99'  # Yellow (normal)
        elif value > -50:
            return '#FFA500'  # Orange
        else:
            return '#FF6347'  # Red (low volume)

def get_pe_ratio_color(pe_value, use_gradient=True):
    """
    Get color based on P/E ratio
    
    Args:
        pe_value: The P/E ratio value
        use_gradient: Whether to use gradient colors
    
    Returns:
        Color string (hex code)
    """
    if pd.isna(pe_value) or pe_value <= 0:
        return '#888888'  # Gray for missing/negative data
    
    # P/E ratio color scale (lower = better value, higher = overvalued)
    if pe_value < 10:
        return '#006400' if use_gradient else '#00FF00'  # Dark green (undervalued)
    elif pe_value < 15:
        return '#32CD32' if use_gradient else '#00FF00'  # Green (good value)
    elif pe_value < 20:
        return '#90EE90' if use_gradient else '#00FF00'  # Light green (fair)
    elif pe_value < 25:
        return '#FFFF99'  # Yellow (neutral)
    elif pe_value < 30:
        return '#FFA500'  # Orange (expensive)
    elif pe_value < 40:
        return '#FF6347'  # Tomato (overvalued)
    else:
        return '#8B0000' if use_gradient else '#FF0000'  # Dark red (very overvalued)

def create_gradient_colorscale(values, metric_type='return'):
    """Create a custom colorscale for plotly based on values"""
    if metric_type == 'return':
        return [
            [0.0, '#8B0000'],    # Dark red
            [0.25, '#FF6B6B'],   # Light red
            [0.45, '#FFB3B3'],   # Very light red
            [0.5, '#D3D3D3'],    # Gray (neutral)
            [0.55, '#B3FFB3'],   # Very light green
            [0.75, '#66FF66'],   # Light green
            [1.0, '#006400']     # Dark green
        ]
    elif metric_type == 'rsi':
        return [
            [0.0, '#006400'],    # Dark green (oversold)
            [0.3, '#90EE90'],    # Light green
            [0.5, '#FFFF99'],    # Yellow (neutral)
            [0.7, '#FFA500'],    # Orange
            [1.0, '#8B0000']     # Dark red (overbought)
        ]

def create_colored_metric_card(title, value, metric_type='return', format_func=None):
    """Create a colored metric card for Streamlit"""
    if format_func:
        display_value = format_func(value)
    else:
        display_value = f"{value:.2f}%" if metric_type == 'return' else f"{value:.1f}"
    
    color = get_performance_color(value, metric_type)
    text_color = 'white' if color in ['#8B0000', '#006400', '#DC143C'] else 'black'
    
    # Use HTML to create colored metric
    st.markdown(f"""
        <div style="
            background-color: {color};
            color: {text_color};
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 5px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h4 style="margin: 0; font-size: 14px;">{title}</h4>
            <h2 style="margin: 5px 0; font-size: 24px;">{display_value}</h2>
        </div>
    """, unsafe_allow_html=True)

def create_enhanced_bar_chart(df, x_col, y_col, title, metric_type='return'):
    """Create a bar chart with gradient colors"""
    colors = [get_performance_color(val, metric_type) for val in df[y_col]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color=colors,
            text=df[y_col].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                          f'{title}: %{{y:.2f}}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False,
        yaxis_title=f"{title} (%)"
    )
    
    return fig

def create_sector_heatmap(sector_data, value_col, title):
    """Create a heatmap for sector performance"""
    fig = go.Figure(data=go.Heatmap(
        z=sector_data.values.reshape(1, -1),
        x=sector_data.index,
        y=['Performance'],
        colorscale=create_gradient_colorscale(sector_data, 'return'),
        text=sector_data.apply(lambda x: f"{x:.1f}%").values.reshape(1, -1),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>Performance: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=200,
        xaxis_tickangle=-45
    )
    
    return fig

def add_custom_css():
    """Add custom CSS for better color styling"""
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
        }
        
        .performance-good {
            background: linear-gradient(135deg, #d4ff88 0%, #4caf50 100%);
            color: white;
        }
        
        .performance-bad {
            background: linear-gradient(135deg, #ff8a80 0%, #f44336 100%);
            color: white;
        }
        
        .performance-neutral {
            background: linear-gradient(135deg, #e0e0e0 0%, #9e9e9e 100%);
            color: black;
        }
        
        /* Custom table styling */
        .stDataFrame [data-testid="stTable"] {
            background-color: white;
            border-radius: 5px;
        }
        
        /* Gradient backgrounds for metric cards */
        .metric-card-positive {
            background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 5px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-card-negative {
            background: linear-gradient(135deg, #f44336 0%, #c62828 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 5px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-card-neutral {
            background: linear-gradient(135deg, #9e9e9e 0%, #616161 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 5px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

def show_color_legend():
    """Display enhanced color legend with visual graphics for performance interpretation"""
    with st.expander("🎨 Color Legend & Definitions"):
        
        # Performance Colors Section
        st.markdown("### 📈 **Performance Colors**")
        st.markdown("*Based on percentage gains/losses over selected time period*")
        
        perf_cols = st.columns(4)
        with perf_cols[0]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #006400 0%, #228B22 100%); 
                        color: white; padding: 8px; border-radius: 5px; text-align: center; margin: 2px;">
                <strong>🟢 Dark Green</strong><br>
                Excellent (>5% gains)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #32CD32 0%, #00FF00 100%); 
                        color: black; padding: 8px; border-radius: 5px; text-align: center; margin: 2px;">
                <strong>🟢 Green</strong><br>
                Good (2-5% gains)
            </div>
            """, unsafe_allow_html=True)
            
        with perf_cols[1]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #90EE90 0%, #98FB98 100%); 
                        color: black; padding: 8px; border-radius: 5px; text-align: center; margin: 2px;">
                <strong>🟢 Light Green</strong><br>
                Positive (0-2% gains)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #D3D3D3 0%, #A9A9A9 100%); 
                        color: black; padding: 8px; border-radius: 5px; text-align: center; margin: 2px;">
                <strong>⚪ Gray</strong><br>
                Neutral (0% change)
            </div>
            """, unsafe_allow_html=True)
            
        with perf_cols[2]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFFF99 0%, #FFD700 100%); 
                        color: black; padding: 8px; border-radius: 5px; text-align: center; margin: 2px;">
                <strong>🟡 Yellow</strong><br>
                Caution Zone
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFB3B3 0%, #FFA0A0 100%); 
                        color: black; padding: 8px; border-radius: 5px; text-align: center; margin: 2px;">
                <strong>🔴 Light Red</strong><br>
                Negative (0-2% loss)
            </div>
            """, unsafe_allow_html=True)
            
        with perf_cols[3]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FF4444 100%); 
                        color: white; padding: 8px; border-radius: 5px; text-align: center; margin: 2px;">
                <strong>🔴 Red</strong><br>
                Poor (2-5% loss)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8B0000 0%, #DC143C 100%); 
                        color: white; padding: 8px; border-radius: 5px; text-align: center; margin: 2px;">
                <strong>🔴 Dark Red</strong><br>
                Very Poor (>5% loss)
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()

        st.markdown("### 💰 **P/E Ratio Colors** (Price-to-Earnings)")
        st.markdown("*Measures stock valuation - lower ratios may indicate better value*")
        
        pe_cols = st.columns(3)
        with pe_cols[0]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #006400 0%, #32CD32 100%); 
                        color: white; padding: 12px; border-radius: 5px; text-align: center;">
                <strong>🟢 Green (P/E < 15)</strong><br>
                <em>Potentially Undervalued</em><br>
                Good value relative to earnings<br>
                May be a bargain
            </div>
            """, unsafe_allow_html=True)
            
        with pe_cols[1]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFFF99 0%, #FFA500 100%); 
                        color: black; padding: 12px; border-radius: 5px; text-align: center;">
                <strong>🟡 Yellow (P/E 15-30)</strong><br>
                <em>Fair Value</em><br>
                Average market valuation<br>
                Reasonable price
            </div>
            """, unsafe_allow_html=True)
            
        with pe_cols[2]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8B0000 0%, #FF6347 100%); 
                        color: white; padding: 12px; border-radius: 5px; text-align: center;">
                <strong>🔴 Red (P/E > 30)</strong><br>
                <em>Potentially Overvalued</em><br>
                High price relative to earnings<br>
                Premium valuation
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # RSI Colors Section
        st.markdown("### 📊 **RSI Colors** (Relative Strength Index)")
        st.markdown("*Technical indicator measuring overbought/oversold conditions (0-100 scale)*")
        
        rsi_cols = st.columns(3)
        with rsi_cols[0]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #006400 0%, #32CD32 100%); 
                        color: white; padding: 12px; border-radius: 5px; text-align: center;">
                <strong>🟢 Green (RSI < 30)</strong><br>
                <em>Oversold</em><br>
                Potential buying opportunity<br>
                Stock may be undervalued
            </div>
            """, unsafe_allow_html=True)
            
        with rsi_cols[1]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFFF99 0%, #FFD700 100%); 
                        color: black; padding: 12px; border-radius: 5px; text-align: center;">
                <strong>🟡 Yellow (RSI 30-70)</strong><br>
                <em>Neutral Zone</em><br>
                Normal trading range<br>
                No clear signal
            </div>
            """, unsafe_allow_html=True)
            
        with rsi_cols[2]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8B0000 0%, #FF6347 100%); 
                        color: white; padding: 12px; border-radius: 5px; text-align: center;">
                <strong>🔴 Red (RSI > 70)</strong><br>
                <em>Overbought</em><br>
                Potential selling signal<br>
                Stock may be overvalued
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Volatility Colors Section
        st.markdown("### 📈📉 **Volatility Colors** (Annualized)")
        st.markdown("*Measures price fluctuation risk - higher volatility = higher risk/reward potential*")
        
        vol_cols = st.columns(3)
        with vol_cols[0]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #006400 0%, #32CD32 100%); 
                        color: white; padding: 12px; border-radius: 5px; text-align: center;">
                <strong>🟢 Green (< 25%)</strong><br>
                <em>Low Risk</em><br>
                Stable, predictable moves<br>
                Conservative investment
            </div>
            """, unsafe_allow_html=True)
            
        with vol_cols[1]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFFF99 0%, #FFA500 100%); 
                        color: black; padding: 12px; border-radius: 5px; text-align: center;">
                <strong>🟡 Yellow (25-50%)</strong><br>
                <em>Medium Risk</em><br>
                Moderate price swings<br>
                Balanced risk/reward
            </div>
            """, unsafe_allow_html=True)
            
        with vol_cols[2]:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #8B0000 0%, #FF6347 100%); 
                        color: white; padding: 12px; border-radius: 5px; text-align: center;">
                <strong>🔴 Red (> 50%)</strong><br>
                <em>High Risk</em><br>
                Large price fluctuations<br>
                High risk/reward potential
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Usage Tips
        st.markdown("### 💡 **How to Use These Colors**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **For Stock Screening:**
            - Look for green performance + green RSI (oversold winners)
            - Avoid red performance + red RSI (overbought losers)
            - Consider volatility for risk tolerance
            """)
            
        with col2:
            st.markdown("""
            **Risk Management:**
            - Green volatility = Safer for conservative portfolios
            - Red volatility = Higher potential gains but bigger losses
            - Yellow = Balanced middle ground
            """)
        
        st.info("💡 **Tip:** Colors provide quick visual cues, but always research fundamentals before investing!")

def apply_color_styling_to_dataframe(df, metrics_columns=None):
    """
    Apply color styling to any dataframe based on column types
    """
    if df.empty:
        return df
    
    # Default column type mapping
    default_metrics = {
        'pct_1d': 'return',
        'pct_3d': 'return', 
        'pct_5d': 'return',
        'pct_21d': 'return',
        'pct_63d': 'return',
        'pct_252d': 'return',
        'ann_vol_pct': 'volatility',
        'rsi': 'rsi',
        'pe_ratio': 'pe_ratio', 
        'volume_vs_avg': 'volume',
        'Mean Return': 'return',
        'Median Return': 'return',
        'Avg Volatility': 'volatility',
        'pct_from_52w_high': 'return',
        'pct_from_52w_low': 'return'
    }
    
    if metrics_columns:
        default_metrics.update(metrics_columns)
    
    def highlight_performance(row):
        colors = []
        for col in df.columns:
            if col in default_metrics:
                metric_type = default_metrics[col]
                if metric_type == 'pe_ratio':
                    color = get_pe_ratio_color(row[col]) 
                else:
                    color = get_performance_color(row[col], metric_type)
                text_color = 'white' if color in ['#8B0000', '#006400', '#DC143C'] else 'black'
                colors.append(f'background-color: {color}; color: {text_color}')
            else:
                colors.append('background-color: white; color: black')
        return colors
    
    return df.style.apply(highlight_performance, axis=1)

def format_and_style_dataframe(df, format_dict=None, metrics_columns=None):
    """
    Format and apply color styling to dataframe
    """
    if df.empty:
        return df
    
    # Default formatting
    default_format = {}
    
    # Auto-detect formatting needs
    for col in df.columns:
        if 'last_close' in col or 'price' in col.lower() or '52w_' in col:
            default_format[col] = '${:.2f}'
        elif 'pe_ratio' in col.lower():
            default_format[col] = '{:.2f}x'
        elif any(x in col for x in ['pct_', 'return', 'vol', 'volatility']):
            default_format[col] = '{:.2f}%'
        elif 'rsi' in col.lower():
            default_format[col] = '{:.1f}'
    
    if format_dict:
        default_format.update(format_dict)
    
    # Apply styling first, then formatting
    styled_df = apply_color_styling_to_dataframe(df.copy(), metrics_columns)
    
    if default_format:
        styled_df = styled_df.format(default_format)
    
    return styled_df

def render_opportunity_recommendations_mode(valid_metrics, hist):
    """
    Render the Opportunity Ranking interface
    With Market Regime Awareness and Confidence Scoring
    """
    st.subheader("🎯 Opportunity Ranking")
    
    # Methodology explanation
    with st.expander("📚 How Opportunity Ranking Works"):
        st.markdown("""
        ### Three-Question Framework
        
        Each stock is evaluated to answer:
        
        1. **Is this stock attractive right now?**
           - Composite Score (0-100) across Technical, Fundamental, Sentiment, and Risk
        
        2. **Why is it attractive right now?**
           - Top 3 contributing factors auto-identified
           - Detailed score breakdown
        
        3. **How confident should I be given current market conditions?**
           - Confidence Level (High/Medium/Low) based on data quality
           - Market Regime Fit (Bull/Neutral/Bear alignment)
        
        ---
        
        ### Market Regime Awareness
        
        Your recommendations are **adjusted for market conditions**:
        
        | Regime | Characteristics | Best Strategies |
        |--------|----------------|-----------------|
        | 🟢 **Bull Market** | Strong uptrend, low volatility | Growth, Momentum |
        | 🟡 **Neutral Market** | Mixed signals, moderate volatility | Balanced, Value |
        | 🔴 **Bear Market** | Decline or high volatility | Value, Contrarian |
        
        **Score Adjustment Example:**
        - Base Score: 75
        - Strategy: Growth
        - Regime: Bull Market
        - **Adjusted Score: 75 × 1.05 = 78.75** ✅
        
        ---
        
        ### Confidence Scoring
        
        Confidence reflects:
        - ✅ **Data Completeness**: All metrics available?
        - ✅ **Sentiment Quality**: Sufficient news articles?
        - ✅ **Volatility**: Predictable price behavior?
        - ✅ **Score Consistency**: Aligned signals across factors?
        
        | Confidence | Meaning |
        |-----------|---------|
        | 🟢 **High** | Strong data, clear signals |
        | 🟡 **Medium** | Good data, some uncertainty |
        | 🔴 **Low** | Limited data or conflicting signals |
        """)
    
    # -------------------- MARKET REGIME DETECTION --------------------
    st.markdown("### 🌍 Current Market Regime")
    
    # Calculate market-level metrics
    with st.spinner("Analyzing market conditions..."):
        index_metrics = pd.Series({
            'pct_21d': valid_metrics['pct_21d'].median(),
            'pct_63d': valid_metrics['pct_63d'].median() if 'pct_63d' in valid_metrics.columns else 0,
            'ann_vol_pct': valid_metrics['ann_vol_pct'].median() if 'ann_vol_pct' in valid_metrics.columns else 25,
            'breadth': calculate_market_breadth(valid_metrics)
        })
        
        market_regime = detect_market_regime(index_metrics)
    
    # Display regime
    regime_colors = {
        "Bull Market": "#006400",
        "Neutral Market": "#FFD700",
        "Bear Market": "#8B0000"
    }
    
    regime_color = regime_colors.get(market_regime.regime.value, "#888888")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: {regime_color}; color: white; padding: 15px; 
                    border-radius: 10px; text-align: center;">
            <h3 style="margin: 0;">{market_regime.regime.value}</h3>
            <p style="margin: 5px 0;">Confidence: {market_regime.confidence:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("21-Day Return", f"{market_regime.pct_21d:.2f}%")
    
    with col3:
        st.metric("63-Day Return", f"{market_regime.pct_63d:.2f}%")
    
    with col4:
        st.metric("Market Volatility", f"{market_regime.volatility:.1f}%")
    
    st.caption(f"📊 Market Breadth: {market_regime.breadth:.1f}% of stocks above 50-day average")
    
    st.divider()
    
    # -------------------- STRATEGY SELECTION --------------------
    st.markdown("### 🎲 Select Investment Strategy")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        strategy = st.selectbox(
            "Investment Strategy",
            options=[
                "Balanced (All-around best)",
                "Growth (High momentum)",
                "Value (Undervalued picks)",
                "Momentum (Trending stocks)",
                "Contrarian (Oversold gems)"
            ],
            help="Strategy determines score weighting and regime adjustments"
        )
    
    with col2:
        top_n = st.slider(
            "Number of Opportunities",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
    
    # Parse strategy
    strategy_map = {
        "Balanced (All-around best)": InvestmentStrategy.BALANCED,
        "Growth (High momentum)": InvestmentStrategy.GROWTH,
        "Value (Undervalued picks)": InvestmentStrategy.VALUE,
        "Momentum (Trending stocks)": InvestmentStrategy.MOMENTUM,
        "Contrarian (Oversold gems)": InvestmentStrategy.CONTRARIAN
    }
    
    selected_strategy = strategy_map[strategy]
    
    # Show regime fit
    st.info(f"""
    **Strategy-Regime Alignment:**  
    {selected_strategy.value.capitalize()} strategy in {market_regime.regime.value} → 
    Score multiplier: **{RegimeAdjustmentMultipliers.for_strategy(selected_strategy).get_multiplier(market_regime.regime):.2f}x**
    """)
    
    # -------------------- ADVANCED OPTIONS --------------------
    with st.expander("⚙️ Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_score = st.number_input(
                "Minimum Score",
                min_value=0,
                max_value=100,
                value=50,
                step=5
            )
        
        with col2:
            min_confidence = st.selectbox(
                "Minimum Confidence",
                ["Any", "Medium or Higher", "High Only"],
                index=0
            )
        
        with col3:
            max_volatility = st.number_input(
                "Max Volatility (%)",
                min_value=10,
                max_value=100,
                value=60,
                step=5
            )
    
    # -------------------- GENERATE RECOMMENDATIONS --------------------
    if st.button("🚀 Rank Opportunities", type="primary", use_container_width=True):
        with st.spinner("Analyzing opportunities with market regime awareness..."):
            try:
                # Generate recommendations with regime awareness
                recommendations, regime_data = generate_recommendations(
                    valid_metrics,
                    strategy=selected_strategy,
                    include_sentiment=True,
                    top_n=top_n,
                    index_metrics=index_metrics
                )
                
                if recommendations.empty:
                    st.warning("No opportunities found. Try adjusting filters.")
                    return
                
                # Apply filters
                recommendations = recommendations[
                    recommendations['composite_score'] >= min_score
                ]
                
                # Confidence filter
                if min_confidence == "High Only":
                    recommendations = recommendations[
                        recommendations['confidence_level'] == "High Confidence"
                    ]
                elif min_confidence == "Medium or Higher":
                    recommendations = recommendations[
                        recommendations['confidence_level'].isin(["High Confidence", "Medium Confidence"])
                    ]
                
                # Volatility filter
                if 'volatility' in recommendations.columns:
                    recommendations = recommendations[
                        (recommendations['volatility'].isna()) | 
                        (recommendations['volatility'] <= max_volatility)
                    ]
                
                # Store in session state
                st.session_state.ai_recommendations = recommendations
                st.session_state.market_regime = regime_data
                
                st.success(f"✅ Ranked {len(recommendations)} opportunities!")
                
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                import traceback
                with st.expander("🐛 Debug"):
                    st.code(traceback.format_exc())
                return
    
    # -------------------- DISPLAY RECOMMENDATIONS --------------------
    if 'ai_recommendations' in st.session_state and not st.session_state.ai_recommendations.empty:
        recommendations = st.session_state.ai_recommendations
        regime_data = st.session_state.get('market_regime', market_regime)
        
        # Summary
        st.markdown("### 📊 Opportunity Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = recommendations['composite_score'].mean()
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        with col2:
            high_conf = (recommendations['confidence_level'] == 'High Confidence').sum()
            st.metric("High Confidence", high_conf)
        
        with col3:
            buy_signals = recommendations[
                recommendations['recommendation_str'].isin(['Strong Buy', 'Buy'])
            ].shape[0]
            st.metric("Buy Signals", buy_signals)
        
        with col4:
            avg_adj = recommendations['regime_multiplier'].mean()
            st.metric("Avg Regime Adj", f"{avg_adj:.2f}x")
        
        # Tabs
        tabs = st.tabs([
            "🎯 Ranked Opportunities",
            "📊 Detailed Analysis",
            "📈 Visual Insights",
            "💼 Portfolio Builder",
            "📥 Export"
        ])
        
        with tabs[0]:
            st.markdown("### 🏆 Top Ranked Opportunities")
            
            # Display top opportunities
            for idx, (_, row) in enumerate(recommendations.head(15).iterrows(), 1):
                
                # Confidence color
                conf_colors = {
                    'High Confidence': '#006400',
                    'Medium Confidence': '#FFD700',
                    'Low Confidence': '#FF6347'
                }
                conf_color = conf_colors.get(row['confidence_level'], '#888888')
                
                # Recommendation color
                rec_colors = {
                    'Strong Buy': '#006400',
                    'Buy': '#32CD32',
                    'Hold': '#FFD700',
                    'Sell': '#FF6347',
                    'Strong Sell': '#8B0000'
                }
                rec_color = rec_colors.get(row['recommendation_str'], '#888888')
                
                with st.container():
                    # Header row
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **#{idx} {row['ticker']} - {row['company_name']}**  
                        <small>{row['sector']}</small>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: {rec_color}; padding: 8px; border-radius: 5px; 
                                    text-align: center; color: white;">
                            <strong>{row['recommendation_str']}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="background: {conf_color}; padding: 8px; border-radius: 5px; 
                                    text-align: center; color: white;">
                            {row['confidence_level'].replace(' Confidence', '')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.metric("Score", f"{row['composite_score']:.1f}")
                    
                    # Details row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.caption("💰 **Price & Valuation**")
                        if row['current_price']:
                            st.write(f"${row['current_price']:.2f}")
                        if row['pe_ratio']:
                            st.write(f"P/E: {row['pe_ratio']:.1f}")
                    
                    with col2:
                        st.caption("📈 **Score Breakdown**")
                        st.write(f"Tech: {row['technical_score']:.0f}")
                        st.write(f"Fund: {row['fundamental_score']:.0f}")
                    
                    with col3:
                        st.caption("🌍 **Market Fit**")
                        st.write(f"{row['market_regime']}")
                        st.write(f"Adj: {row['regime_multiplier']:.2f}x")
                    
                    with col4:
                        st.caption("⚡ **Key Strengths**")
                        factors = row.get('top_factors', [])
                        if factors:
                            for factor in factors[:2]:
                                st.write(f"• {factor}")
                    
                    st.divider()
        
        with tabs[1]:
            st.markdown("### 🔍 Detailed Scoring Analysis")
            
            # Comprehensive table
            display_cols = [
                'ticker', 'company_name', 'composite_score', 'base_score',
                'recommendation_str', 'confidence_level', 'confidence_score',
                'technical_score', 'fundamental_score', 'sentiment_score', 'risk_score',
                'regime_multiplier', 'current_price', 'pe_ratio', 'volatility'
            ]
            
            display_df = recommendations[[col for col in display_cols if col in recommendations.columns]].copy()
            
            # Format
            format_dict = {
                'composite_score': '{:.1f}',
                'base_score': '{:.1f}',
                'confidence_score': '{:.2f}',
                'technical_score': '{:.0f}',
                'fundamental_score': '{:.0f}',
                'sentiment_score': '{:.0f}',
                'risk_score': '{:.0f}',
                'regime_multiplier': '{:.2f}',
                'current_price': '${:.2f}',
                'pe_ratio': '{:.1f}',
                'volatility': '{:.1f}%'
            }
            
            styled_df = display_df.style.format(format_dict)
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Score distributions
            st.markdown("### 📊 Score Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    recommendations,
                    x='composite_score',
                    nbins=20,
                    title="Composite Score Distribution",
                    color='confidence_level',
                    color_discrete_map={
                        'High Confidence': '#006400',
                        'Medium Confidence': '#FFD700',
                        'Low Confidence': '#FF6347'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    recommendations,
                    x='recommendation_str',
                    y='composite_score',
                    color='confidence_level',
                    title="Score by Recommendation Level",
                    color_discrete_map={
                        'High Confidence': '#006400',
                        'Medium Confidence': '#FFD700',
                        'Low Confidence': '#FF6347'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.markdown("### 📈 Visual Opportunity Analysis")
            
            # Risk vs Opportunity
            fig = px.scatter(
                recommendations,
                x='volatility',
                y='composite_score',
                size='confidence_score',
                color='recommendation_str',
                hover_data=['ticker', 'company_name', 'sector', 'confidence_level'],
                title="Risk vs Opportunity (size = confidence)",
                labels={
                    'volatility': 'Volatility (Risk %)',
                    'composite_score': 'Opportunity Score'
                },
                color_discrete_map={
                    'Strong Buy': '#006400',
                    'Buy': '#32CD32',
                    'Hold': '#FFD700',
                    'Sell': '#FF6347',
                    'Strong Sell': '#8B0000'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Regime adjustment impact
            st.markdown("### 🌍 Regime Adjustment Impact")
            
            recommendations['adjustment_impact'] = (
                recommendations['composite_score'] - recommendations['base_score']
            )
            
            fig = px.bar(
                recommendations.head(20),
                x='ticker',
                y='adjustment_impact',
                color='adjustment_impact',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                title="Top 20: Regime Adjustment Impact on Scores",
                labels={'adjustment_impact': 'Score Change from Regime'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence breakdown
            st.markdown("### 🎯 Confidence Analysis")
            
            conf_dist = recommendations['confidence_level'].value_counts()
            
            fig = px.pie(
                values=conf_dist.values,
                names=conf_dist.index,
                title="Confidence Level Distribution",
                color=conf_dist.index,
                color_discrete_map={
                    'High Confidence': '#006400',
                    'Medium Confidence': '#FFD700',
                    'Low Confidence': '#FF6347'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            st.markdown("### 💼 Build Opportunity Portfolio")
            
            st.info("""
            **Smart Diversification**: Build a portfolio that balances high-score opportunities 
            with sector diversification and confidence levels.
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                portfolio_size = st.slider("Portfolio Size", 5, 30, 10)
            
            with col2:
                max_per_sector = st.slider("Max per Sector", 1, 5, 3)
            
            with col3:
                min_portfolio_score = st.slider("Min Score", 50, 90, 65)
            
            if st.button("🏗️ Build Portfolio", use_container_width=True):
                with st.spinner("Optimizing portfolio..."):
                    # Filter by score first
                    eligible = recommendations[
                        recommendations['composite_score'] >= min_portfolio_score
                    ]
                    
                    if eligible.empty:
                        st.warning("No stocks meet the minimum score requirement.")
                    else:
                        # Build diversified portfolio
                        portfolio = []
                        sector_counts = {}
                        
                        # Prioritize high confidence, then high score
                        eligible = eligible.sort_values(
                            ['confidence_score', 'composite_score'],
                            ascending=[False, False]
                        )
                        
                        for _, row in eligible.iterrows():
                            sector = row['sector']
                            if len(portfolio) >= portfolio_size:
                                break
                            
                            if sector_counts.get(sector, 0) < max_per_sector:
                                portfolio.append(row)
                                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                        
                        if portfolio:
                            portfolio_df = pd.DataFrame(portfolio)
                            st.session_state.opportunity_portfolio = portfolio_df
                            st.success(f"✅ Built portfolio with {len(portfolio_df)} opportunities!")
            
            # Display portfolio
            if 'opportunity_portfolio' in st.session_state:
                portfolio = st.session_state.opportunity_portfolio
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Stocks", len(portfolio))
                
                with col2:
                    st.metric("Avg Score", f"{portfolio['composite_score'].mean():.1f}")
                
                with col3:
                    high_conf = (portfolio['confidence_level'] == 'High Confidence').sum()
                    st.metric("High Confidence", high_conf)
                
                with col4:
                    st.metric("Avg Volatility", f"{portfolio['volatility'].mean():.1f}%")
                
                # Sector breakdown
                st.markdown("#### 📊 Portfolio Composition")
                
                sector_dist = portfolio['sector'].value_counts()
                fig = px.pie(
                    values=sector_dist.values,
                    names=sector_dist.index,
                    title="Sector Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Holdings table
                st.markdown("#### 📋 Portfolio Holdings")
                
                holdings_cols = [
                    'ticker', 'company_name', 'composite_score', 'confidence_level',
                    'recommendation_str', 'current_price', 'sector'
                ]
                
                st.dataframe(
                    portfolio[[col for col in holdings_cols if col in portfolio.columns]],
                    use_container_width=True
                )
        
        with tabs[4]:
            st.markdown("### 📥 Export Opportunities")
            
            # Prepare export
            export_cols = [
                'ticker', 'company_name', 'sector',
                'composite_score', 'base_score', 'regime_multiplier',
                'recommendation_str', 'confidence_level', 'confidence_score',
                'technical_score', 'fundamental_score', 'sentiment_score', 'risk_score',
                'market_regime', 'current_price', 'pe_ratio', 'volatility',
                'top_factors'
            ]
            
            export_df = recommendations[[col for col in export_cols if col in recommendations.columns]]
            
            # Convert list columns to strings
            if 'top_factors' in export_df.columns:
                export_df['top_factors'] = export_df['top_factors'].apply(
                    lambda x: '; '.join(x) if isinstance(x, list) else x
                )
            
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Opportunities (CSV)",
                data=csv,
                file_name=f"opportunity_ranking_{selected_strategy.value}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.dataframe(export_df, use_container_width=True, height=400)
    
    else:
        st.info("👆 Configure your preferences and click 'Rank Opportunities' to begin!")

# ---------------------------
# Metrics Calculation
# ---------------------------
def compute_metrics_for_ticker(df, consecutive_days=7):
    """Calculate comprehensive metrics for a single ticker with configurable consecutive period"""
    metrics = {}

    if df is None or df.empty:
        return None

    df = df.sort_index()
    if 'Close' not in df.columns:
        return None
    closes = df['Close'].dropna()

    if closes.empty:
        return None

    last_date = closes.index[-1]
    metrics['last_date'] = last_date.strftime('%Y-%m-%d')
    metrics['last_close'] = float(closes.iloc[-1])
    metrics['pe_ratio'] = np.nan  # Default value

    # Configurable consecutive rising/decline check
    if len(closes) >= consecutive_days:
        last_n = closes.iloc[-consecutive_days:]
        increasing = all(last_n.values[i] > last_n.values[i-1] for i in range(1, len(last_n)))
        decreasing = all(last_n.values[i] < last_n.values[i-1] for i in range(1, len(last_n)))
    else:
        increasing = False
        decreasing = False

    metrics[f'rising_{consecutive_days}day'] = bool(increasing)
    metrics[f'declining_{consecutive_days}day'] = bool(decreasing)

    # Keep original 7-day metrics for backward compatibility
    if consecutive_days != 7:
        if len(closes) >= 7:
            last_7 = closes.iloc[-7:]
            increasing_7 = all(last_7.values[i] > last_7.values[i-1] for i in range(1, len(last_7)))
            decreasing_7 = all(last_7.values[i] < last_7.values[i-1] for i in range(1, len(last_7)))
        else:
            increasing_7 = False
            decreasing_7 = False

        metrics['rising_7day'] = bool(increasing_7)
        metrics['declining_7day'] = bool(decreasing_7)

    # Percentage changes
    def pct_change(closes, days):
        if len(closes) < days + 1:
            return np.nan
        return (closes.iloc[-1] / closes.iloc[-(days+1)] - 1.0) * 100.0

    metrics['pct_1d'] = pct_change(closes, 1)
    metrics['pct_3d'] = pct_change(closes, 3)
    metrics['pct_5d'] = pct_change(closes, 5)
    metrics['pct_21d'] = pct_change(closes, 21)
    metrics['pct_63d'] = pct_change(closes, 63)
    metrics['pct_252d'] = pct_change(closes, 252)

    # Add configurable period percentage change if different from standard periods
    if consecutive_days not in [1, 3, 5, 7, 21, 63, 252]:
        metrics[f'pct_{consecutive_days}d'] = pct_change(closes, consecutive_days)

    # Add validation
    if consecutive_days < 2:
        raise ValueError("consecutive_days must be >= 2")
    if consecutive_days > len(df):
        return None  # Not enough data

    # Historical averages
    today = closes.index[-1]

    def mean_since(delta):
        start = today - delta
        rv = closes[closes.index >= start]
        return float(rv.mean()) if not rv.empty else np.nan

    metrics['avg_1y'] = mean_since(relativedelta(years=1))
    metrics['avg_2y'] = mean_since(relativedelta(years=2))
    metrics['avg_5y'] = mean_since(relativedelta(years=5))
    metrics['avg_max'] = float(closes.mean())

    # Cumulative return
    first = closes.iloc[0]
    last = closes.iloc[-1]
    metrics['cum_return_from_start_pct'] = (last / first - 1.0) * 100.0

    # Volatility
    daily_ret = closes.pct_change().dropna()
    if len(daily_ret) > 1:
        metrics['ann_vol_pct'] = np.std(daily_ret) * np.sqrt(252) * 100.0
    else:
        metrics['ann_vol_pct'] = np.nan

    # RSI
    metrics['rsi'] = calculate_rsi(closes)

    # 52-week high/low
    if len(closes) >= 252:
        metrics['high_52w'] = float(closes.iloc[-252:].max())
        metrics['low_52w'] = float(closes.iloc[-252:].min())
        metrics['pct_from_52w_high'] = ((last - metrics['high_52w']) / metrics['high_52w']) * 100
        metrics['pct_from_52w_low'] = ((last - metrics['low_52w']) / metrics['low_52w']) * 100

    # Volume metrics if available
    if 'Volume' in df.columns:
        volumes = df['Volume'].dropna()
        if len(volumes) >= 20:
            metrics['avg_volume_20d'] = float(volumes.iloc[-20:].mean())
            metrics['volume_vs_avg'] = (float(volumes.iloc[-1]) / metrics['avg_volume_20d'] - 1) * 100 if metrics['avg_volume_20d'] > 0 else 0

    metrics['data_points'] = len(closes)

    return metrics

def fetch_pe_ratios(tickers, metrics_df):
    """
    Fetch P/E ratios for all tickers and update metrics dataframe
    
    Args:
        tickers: List of ticker symbols
        metrics_df: DataFrame with existing metrics
    
    Returns:
        Updated DataFrame with pe_ratio column
    """
    if VERBOSE:
        print(f"\nFetching P/E ratios for {len(tickers)} tickers...")
    
    pe_data = {}
    
    # Fetching in batches to avoid overwhelming the API
    for chunk in tqdm(batch(tickers, 50), desc="Fetching P/E ratios"):
        for ticker in chunk:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Try multiple P/E ratio fields
                pe_ratio = info.get('trailingPE', None)
                if pe_ratio is None:
                    pe_ratio = info.get('forwardPE', None)
                
                # Validate P/E ratio (should be positive and reasonable)
                if pe_ratio is not None and pe_ratio > 0 and pe_ratio < 1000:
                    pe_data[ticker] = float(pe_ratio)
                else:
                    pe_data[ticker] = np.nan
                    
            except Exception as e:
                if VERBOSE:
                    print(f"Error fetching P/E for {ticker}: {e}")
                pe_data[ticker] = np.nan
            
            time.sleep(0.1)  # Rate limiting
    
    # Update metrics dataframe
    metrics_df['pe_ratio'] = metrics_df['ticker'].map(pe_data)
    
    if VERBOSE:
        valid_pe_count = metrics_df['pe_ratio'].notna().sum()
        print(f"✓ Fetched P/E ratios for {valid_pe_count}/{len(tickers)} stocks")
    
    return metrics_df

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    if len(prices) < period + 1:
        return np.nan

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else np.nan

def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    if df.empty:
        return df

    # Moving averages
    df = df.copy()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal"]

    # RSI (series)
    try:
        df['RSI'] = df['Close'].rolling(window=15).apply(lambda s: calculate_rsi(s), raw=False)
    except Exception:
        # fallback - compute last RSI using closing prices
        df['RSI'] = np.nan

    # Momentum
    df['Momentum'] = df['Close'] / df['Close'].shift(14) - 1

    return df

def format_currency(val):
    """Format value as currency"""
    try:
        return f'${float(val):.2f}'
    except:
        return str(val)

def format_percentage(val):
    """Format value as percentage"""
    try:
        return f'{float(val):.2f}%'
    except:
        return str(val)

def format_number(val, decimals=1):
    """Format value as number with specified decimals"""
    try:
        return f'{float(val):.{decimals}f}'
    except:
        return str(val)
    
# ---------------------------
# Advanced Analytics Functions
# ---------------------------

def calculate_beta(stock_returns, market_returns):
    """Calculate beta coefficient"""
    if len(stock_returns) != len(market_returns) or len(stock_returns) < 10:
        return np.nan
    
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return np.nan
    
    return covariance / market_variance

def calculate_alpha(stock_returns, market_returns, risk_free_rate=0.02):
    """Calculate alpha coefficient"""
    if len(stock_returns) != len(market_returns) or len(stock_returns) < 10:
        return np.nan
    
    beta = calculate_beta(stock_returns, market_returns)
    if np.isnan(beta):
        return np.nan
    
    stock_avg_return = np.mean(stock_returns) * 252  # Annualized
    market_avg_return = np.mean(market_returns) * 252  # Annualized
    
    expected_return = risk_free_rate + beta * (market_avg_return - risk_free_rate)
    alpha = stock_avg_return - expected_return
    
    return alpha

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    if len(returns) < 10:
        return np.nan
    
    excess_returns = np.mean(returns) * 252 - risk_free_rate
    volatility = np.std(returns) * np.sqrt(252)
    
    if volatility == 0:
        return np.nan
    
    return excess_returns / volatility

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    if len(prices) < 2:
        return np.nan
    
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    max_drawdown = drawdown.min()
    
    return abs(max_drawdown) * 100

def calculate_var(returns, confidence_level=0.05):
    """Calculate Value at Risk"""
    if len(returns) < 30:
        return np.nan
    
    return np.percentile(returns, confidence_level * 100) * 100

def calculate_correlation_matrix(data_dict):
    """Calculate correlation matrix for multiple stocks"""
    if len(data_dict) < 2:
        return pd.DataFrame()
    
    returns_df = pd.DataFrame()
    
    for ticker, df in data_dict.items():
        if df is not None and not df.empty and 'Close' in df.columns:
            # Remove duplicate indices before processing
            df_clean = df[~df.index.duplicated(keep='first')]
            
            # Calculate returns
            returns = df_clean['Close'].pct_change().dropna()
            
            if len(returns) > 50:  # Minimum data requirement
                # Ensure the returns series has a unique index
                if returns.index.duplicated().any():
                    returns = returns[~returns.index.duplicated(keep='first')]
                
                returns_df[ticker] = returns
    
    if returns_df.empty or len(returns_df.columns) < 2:
        return pd.DataFrame()
    
    # Align all series to common dates
    returns_df = returns_df.dropna(how='all')  # Remove rows with all NaN
    
    # Check if we have enough data after alignment
    if len(returns_df) < 50:
        return pd.DataFrame()
    
    return returns_df.corr()

def get_sector_comparison_data(valid_metrics, selected_tickers):
    """Get sector comparison data for selected tickers"""
    comparison_data = []
    
    for ticker in selected_tickers:
        ticker_data = valid_metrics[valid_metrics['ticker'] == ticker]
        if not ticker_data.empty:
            row = ticker_data.iloc[0]
            comparison_data.append({
                'ticker': ticker,
                'company_name': row.get('company_name', 'Unknown'),
                'sector': row.get('sector', 'Unknown'),
                'last_close': row.get('last_close', 0),
                'pe_ratio': row.get('pe_ratio', np.nan),
                'pct_1d': row.get('pct_1d', 0),
                'pct_5d': row.get('pct_5d', 0),
                'pct_21d': row.get('pct_21d', 0),
                'pct_252d': row.get('pct_252d', 0),
                'ann_vol_pct': row.get('ann_vol_pct', 0),
                'rsi': row.get('rsi', 50)
            })
    
    return pd.DataFrame(comparison_data)

# ---------------------------
# Advanced Mode UI Functions
# ---------------------------

def render_comparison_mode(valid_metrics, hist, horizon_col, horizon_label, current_index):
    """Render the Comparison Mode interface"""
    st.subheader("🔄 Stock Comparison Analysis")
    
    # Stock selection with index-specific state management
    available_tickers = sorted(valid_metrics['ticker'].unique())

    # Use index-specific state keys to avoid conflicts
    state_key = f'comparison_selected_{current_index}'
    selection_history_key = f'selection_history_{current_index}'

    # Initialize session state for this specific index
    if state_key not in st.session_state:
        st.session_state[state_key] = []
        st.session_state[selection_history_key] = []

    # Validate stored tickers against current data
    valid_stored_tickers = [t for t in st.session_state[state_key] if t in available_tickers]

    # Track if we had to clear invalid selections
    selections_cleared = len(valid_stored_tickers) != len(st.session_state[state_key])

    # Update session state
    st.session_state[state_key] = valid_stored_tickers

    # Show informative message if selections were cleared
    if selections_cleared and len(st.session_state[state_key]) == 0:
        st.info(f"ℹ️ Previous selections cleared due to index change to {index_selection}")
    elif selections_cleared:
        removed_count = len(st.session_state[selection_history_key]) - len(valid_stored_tickers)
        st.warning(f"⚠️ {removed_count} previously selected ticker(s) not found in {index_selection}")

    # Update selection history
    st.session_state[selection_history_key] = valid_stored_tickers.copy()
    
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Map current_index to display name
        index_display_names = {
            "SP500": "S&P 500",
            "SP400": "S&P MidCap 400",
            "SP600": "S&P SmallCap 600",
            "NASDAQ100": "Nasdaq-100",
            "DOW30": "Dow Jones 30",
            "COMBINED": "Combined Indices"
        }
        index_display = index_display_names.get(current_index, current_index)
        
        selected_tickers = st.multiselect(
            "Select stocks to compare (2-20 stocks)",
            options=available_tickers,
            default=st.session_state[state_key],
            max_selections=20,
            key=f"stock_selector_{current_index}",
            help=f"Select 2-20 stocks from {index_display} to compare"
        )
        
        # Update session state when user manually changes selection
        st.session_state[state_key] = selected_tickers

    with col2:
        # Add top performers button
        if st.button("⭐ Top Performers", key=f"add_top_performers_{current_index}", use_container_width=True):
            available_return_cols = [col for col in ['pct_21d', 'pct_5d', 'pct_1d'] 
                                    if col in valid_metrics.columns]
            
            if available_return_cols:
                top_10 = valid_metrics.nlargest(10, available_return_cols[0])['ticker'].tolist()
                
                # Merge with existing selections
                current = st.session_state[state_key]
                updated = list(dict.fromkeys(current + top_10))[:20]
                
                # Update state
                st.session_state[state_key] = updated
                
                st.success(f"✅ Added {len(updated) - len(current)} top performers!")
                st.rerun()

    with col3:
        # Clear selections button
        if st.button("🗑️ Clear All", key=f"clear_selections_{current_index}", use_container_width=True):
            st.session_state[state_key] = []
            st.info("Selections cleared")
            st.rerun()
    
    if len(selected_tickers) < 2:
        st.info("👆 Please select at least 2 stocks to compare")
        return
    
    # Get comparison data
    comparison_df = get_sector_comparison_data(valid_metrics, selected_tickers)
    
    if comparison_df.empty:
        st.error("No data available for selected stocks")
        return
    
    # Display comparison table
    st.subheader("📊 Performance Comparison")
    
    styled_comparison = format_and_style_dataframe(comparison_df)
    st.dataframe(styled_comparison, use_container_width=True)
    
    # Performance charts
    st.subheader("📈 Visual Comparison")
    
    tabs = st.tabs(["Returns Comparison", "Risk vs Return", "Correlation Analysis", "Top Performers"])
    
    with tabs[0]:
        # Returns comparison chart
        metrics_to_plot = ['pct_1d', 'pct_5d', 'pct_21d', 'pct_252d']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if available_metrics:
            fig = go.Figure()
            
            for ticker in selected_tickers:
                ticker_data = comparison_df[comparison_df['ticker'] == ticker]
                if not ticker_data.empty:
                    values = [ticker_data[metric].iloc[0] for metric in available_metrics]
                    fig.add_trace(go.Scatter(
                        x=available_metrics,
                        y=values,
                        mode='lines+markers',
                        name=ticker,
                        line=dict(width=3)
                    ))
            
            fig.update_layout(
                title="Return Comparison Across Time Periods",
                xaxis_title="Time Period",
                yaxis_title="Return (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        # Risk vs Return scatter plot
        if 'ann_vol_pct' in comparison_df.columns and 'pct_252d' in comparison_df.columns:
            hover_data_list = ['sector', 'ann_vol_pct', 'pct_252d', 'last_close']
            if 'company_name' in comparison_df.columns:
                hover_data_list.insert(0, 'company_name')
            
            fig = px.scatter(
                comparison_df,
                x='ann_vol_pct',
                y='pct_252d',
                text='ticker',
                title="Risk vs Return Analysis (1 Year)",
                labels={
                    'ann_vol_pct': 'Volatility (%)', 
                    'pct_252d': '1-Year Return (%)',
                    'last_close': 'Price ($)',
                    'company_name': 'Company'
                },
                size='last_close',
                color='sector',
                hover_data=hover_data_list
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for risk vs return analysis")
    
    with tabs[2]:
        # Correlation analysis
        if len(selected_tickers) >= 2:
            correlation_data = {}
            for ticker in selected_tickers:
                if ticker in hist and not hist[ticker].empty:
                    correlation_data[ticker] = hist[ticker]
            
            if len(correlation_data) >= 2:
                corr_matrix = calculate_correlation_matrix(correlation_data)
                
                if not corr_matrix.empty:
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Stock Price Correlation Matrix",
                        color_continuous_scale="RdBu"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Interpretation:**")
                    st.markdown("- Values close to 1: Stocks move together")
                    st.markdown("- Values close to -1: Stocks move in opposite directions")
                    st.markdown("- Values close to 0: Little correlation")
                else:
                    st.info("Unable to calculate correlations with available data")
            else:
                st.info("Insufficient historical data for correlation analysis")

    with tabs[3]:
        # Top Performers
        st.subheader("🏆 Top 20 Performers")
        
        if horizon_col and horizon_col in valid_metrics.columns:
            top_20 = valid_metrics.nlargest(20, horizon_col)
            
            st.info(f"Showing top 20 performers over {horizon_label}")
            
            top_20_tickers = top_20['ticker'].tolist()
            top_20_comparison = get_sector_comparison_data(valid_metrics, top_20_tickers)
            
            if not top_20_comparison.empty:
                display_df = top_20_comparison.copy()
                display_df['rank'] = range(1, len(display_df) + 1)
                
                cols = ['rank', 'ticker']
                if 'company_name' in display_df.columns:
                    cols.append('company_name')
                cols += [col for col in display_df.columns if col not in cols]
                display_df = display_df[cols]
                
                styled_top20 = format_and_style_dataframe(
                    display_df,
                    format_dict={'rank': '{:.0f}'}
                )
                st.dataframe(styled_top20, use_container_width=True)
                
                if horizon_col in top_20.columns:
                    fig = create_enhanced_bar_chart(
                        top_20.head(20), 'ticker', horizon_col,
                        f"Top 20 Performers ({horizon_label})", 'return'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                sector_counts = top_20['sector'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sector Breakdown")
                    fig = px.pie(
                        values=sector_counts.values,
                        names=sector_counts.index,
                        title="Top Performers by Sector"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Performance Stats")
                    avg_return = top_20[horizon_col].mean()
                    median_return = top_20[horizon_col].median()
                    min_return = top_20[horizon_col].min()
                    max_return = top_20[horizon_col].max()
                    
                    st.metric("Average Return", f"{avg_return:.2f}%")
                    st.metric("Median Return", f"{median_return:.2f}%")
                    st.metric("Range", f"{min_return:.2f}% to {max_return:.2f}%")
            
        else:
            st.error("No performance data available for top performers analysis")

def render_historical_analysis_mode(valid_metrics, hist):
    """Render the Historical Analysis Mode interface"""
    st.subheader("📅 Historical Analysis")
    
    # Stock selection
    available_tickers = sorted(valid_metrics['ticker'].unique())
    
    # Create display options with company names
    if 'company_name' in valid_metrics.columns:
        ticker_to_company = valid_metrics.set_index('ticker')['company_name'].to_dict()
        ticker_options = {
            f"{ticker_to_company.get(ticker, ticker)} ({ticker})": ticker 
            for ticker in available_tickers
        }
        selected_display = st.selectbox(
            "Select company for historical analysis",
            options=list(ticker_options.keys()),
            key="historical_ticker"
        )
        selected_ticker = ticker_options[selected_display]
    else:
        selected_ticker = st.selectbox(
            "Select stock for historical analysis",
            options=available_tickers,
            key="historical_ticker"
        )
    
    if not selected_ticker or selected_ticker not in hist:
        st.info("Please select a valid stock")
        return
    
    df = hist[selected_ticker]
    if df.empty or 'Close' not in df.columns:
        st.error("No historical data available for selected stock")
        return
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="hist_start_date"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="hist_end_date"
        )
    
    # Filter data by date range
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    filtered_df = df.loc[mask].copy()
    
    if filtered_df.empty:
        st.error("No data available for selected date range")
        return
    
    # Add technical indicators
    filtered_df = add_technical_indicators(filtered_df)
    
    # Calculate additional metrics
    returns = filtered_df['Close'].pct_change().dropna()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = (filtered_df['Close'].iloc[-1] / filtered_df['Close'].iloc[0] - 1) * 100
        st.metric("Total Return", f"{total_return:.2f}%")
    
    with col2:
        annualized_vol = returns.std() * np.sqrt(252) * 100
        st.metric("Annualized Volatility", f"{annualized_vol:.2f}%")
    
    with col3:
        max_dd = calculate_max_drawdown(filtered_df['Close'])
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    
    with col4:
        sharpe = calculate_sharpe_ratio(returns)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
    
    # Charts
    tabs = st.tabs(["Price & Volume", "Returns Analysis", "Drawdown Analysis", "Seasonal Patterns"])
    
    with tabs[0]:
        # Price and volume chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], 
                                name='Close Price'), row=1, col=1)
        
        if 'MA20' in filtered_df.columns:
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['MA20'], 
                                   name='20-day MA', line=dict(dash='dash')), row=1, col=1)
        
        if 'Volume' in filtered_df.columns:
            fig.add_trace(go.Bar(x=filtered_df.index, y=filtered_df['Volume'], 
                               name='Volume'), row=2, col=1)
        
        fig.update_layout(height=600, title=f"{selected_ticker} Price and Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        # Returns distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns * 100, nbinsx=50, name='Daily Returns'))
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns statistics
        st.subheader("Returns Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Mean Daily Return: {returns.mean() * 100:.3f}%")
            st.write(f"Median Daily Return: {returns.median() * 100:.3f}%")
            st.write(f"Std Dev Daily Return: {returns.std() * 100:.3f}%")
        
        with col2:
            st.write(f"Skewness: {stats.skew(returns):.3f}")
            st.write(f"Kurtosis: {stats.kurtosis(returns):.3f}")
            st.write(f"Best Day: {returns.max() * 100:.2f}%")
            st.write(f"Worst Day: {returns.min() * 100:.2f}%")
    
    with tabs[2]:
        # Drawdown analysis
        peak = filtered_df['Close'].expanding().max()
        drawdown = (filtered_df['Close'] - peak) / peak * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df.index, y=drawdown, 
                               fill='tozeroy', name='Drawdown'))
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        # Seasonal analysis
        if len(filtered_df) > 252:  # At least 1 year of data
            monthly_returns = returns.groupby(returns.index.month).mean() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=monthly_returns.values,
                name='Average Monthly Return'
            ))
            fig.update_layout(
                title="Seasonal Pattern - Average Monthly Returns",
                xaxis_title="Month",
                yaxis_title="Average Return (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for seasonal analysis (need at least 1 year)")

def render_risk_management_mode(valid_metrics, hist):
    """Render the Risk Management Mode interface"""
    st.subheader("⚠️ Risk Management Analysis")
    
    # Portfolio construction
    st.markdown("### 📋 Portfolio Construction")
    
    available_tickers = sorted(valid_metrics['ticker'].unique())
    
    # Portfolio input
    portfolio_stocks = []
    weights = []
    
    num_stocks = st.slider("Number of stocks in portfolio", 2, 10, 5, key="portfolio_size")
    
    col1, col2 = st.columns(2)
    
    for i in range(num_stocks):
        with col1:
            # Create display options with company names
            if 'company_name' in valid_metrics.columns:
                ticker_to_company = valid_metrics.set_index('ticker')['company_name'].to_dict()
                ticker_options = {
                    f"{ticker_to_company.get(ticker, ticker)} ({ticker})": ticker 
                    for ticker in available_tickers
                }
                stock_display = st.selectbox(
                    f"Stock {i+1}",
                    options=list(ticker_options.keys()),
                    key=f"portfolio_stock_{i}"
                )
                stock = ticker_options[stock_display]
            else:
                stock = st.selectbox(
                    f"Stock {i+1}",
                    options=available_tickers,
                    key=f"portfolio_stock_{i}"
                )
        with col2:
            weight = st.number_input(
                f"Weight {i+1} (%)",
                min_value=0.0,
                max_value=100.0,
                value=100.0/num_stocks,
                step=0.1,
                key=f"portfolio_weight_{i}"
            )
        
        if stock:
            portfolio_stocks.append(stock)
            weights.append(weight/100)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w/total_weight for w in weights]
    
    if len(portfolio_stocks) < 2 or total_weight == 0:
        st.info("Please select at least 2 stocks with valid weights")
        return
    
    # Calculate portfolio metrics
    st.markdown("### 📊 Portfolio Risk Metrics")
    
    # Get historical data for portfolio stocks
    portfolio_data = {}
    portfolio_returns = pd.DataFrame()
    
    for stock in portfolio_stocks:
        if stock in hist and not hist[stock].empty:
            portfolio_data[stock] = hist[stock]
            returns = hist[stock]['Close'].pct_change().dropna()
            if len(returns) > 100:  # Minimum data requirement
                portfolio_returns[stock] = returns
    
    if portfolio_returns.empty:
        st.error("Insufficient historical data for risk analysis")
        return
    
    # Align dates and calculate portfolio returns
    portfolio_returns = portfolio_returns.dropna()
    
    if len(portfolio_returns) < 50:
        st.error("Insufficient overlapping data for portfolio analysis")
        return
    
    # Calculate weighted portfolio returns
    valid_stocks = list(portfolio_returns.columns)
    valid_weights = [weights[portfolio_stocks.index(stock)] for stock in valid_stocks if stock in portfolio_stocks]
    
    if len(valid_weights) != len(valid_stocks):
        st.error("Mismatch between stocks and weights")
        return
    
    # Normalize weights for valid stocks
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()
    
    portfolio_return_series = (portfolio_returns[valid_stocks] * valid_weights).sum(axis=1)
    
    # Calculate metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        portfolio_vol = portfolio_return_series.std() * np.sqrt(252) * 100
        st.metric("Portfolio Volatility", f"{portfolio_vol:.2f}%")
    
    with col2:
        portfolio_var = calculate_var(portfolio_return_series)
        st.metric("1-Day VaR (95%)", f"{portfolio_var:.2f}%")
    
    with col3:
        portfolio_sharpe = calculate_sharpe_ratio(portfolio_return_series)
        st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}" if not np.isnan(portfolio_sharpe) else "N/A")
    
    with col4:
        # Convert to price series for drawdown calculation
        portfolio_prices = (1 + portfolio_return_series).cumprod()
        portfolio_dd = calculate_max_drawdown(portfolio_prices)
        st.metric("Max Drawdown", f"{portfolio_dd:.2f}%")
    
    # Risk decomposition
    tabs = st.tabs(["Correlation Matrix", "Individual Stock Risk", "VaR Analysis", "Stress Testing"])
    
    with tabs[0]:
        # Correlation matrix
        corr_matrix = portfolio_returns[valid_stocks].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Portfolio Correlation Matrix",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
    # Individual stock contributions
        individual_metrics = []
        
        for stock in valid_stocks:
            stock_returns = portfolio_returns[stock]
            weight = valid_weights[valid_stocks.index(stock)]
            
            # Get market data for beta calculation 
            market_returns = portfolio_returns.mean(axis=1)  # Simple market proxy
            
            # Get company name from valid_metrics
            company_name = valid_metrics[valid_metrics['ticker'] == stock]['company_name'].iloc[0] if 'company_name' in valid_metrics.columns and not valid_metrics[valid_metrics['ticker'] == stock].empty else 'Unknown'
            pe_ratio = valid_metrics[valid_metrics['ticker'] == stock]['pe_ratio'].iloc[0] if 'pe_ratio' in valid_metrics.columns and not valid_metrics[valid_metrics['ticker'] == stock].empty else np.nan
            
            metrics = {
                'Stock': stock,
                'Company Name': company_name,
                'P/E Ratio': f"{pe_ratio:.2f}x" if not pd.isna(pe_ratio) else 'N/A',
                'Weight': f"{weight*100:.1f}%",
                'Volatility': f"{stock_returns.std() * np.sqrt(252) * 100:.2f}%",
                'Beta': f"{calculate_beta(stock_returns, market_returns):.2f}",
                'Alpha': f"{calculate_alpha(stock_returns, market_returns):.2f}%",
                'VaR': f"{calculate_var(stock_returns):.2f}%"
            }
            individual_metrics.append(metrics)
        
        st.dataframe(pd.DataFrame(individual_metrics), use_container_width=True)
    
    with tabs[2]:
        # VaR analysis
        confidence_levels = [0.01, 0.05, 0.10]
        var_results = []
        
        for conf in confidence_levels:
            var_value = calculate_var(portfolio_return_series, conf)
            var_results.append({
                'Confidence Level': f"{(1-conf)*100:.0f}%",
                'Daily VaR': f"{var_value:.2f}%",
                'Weekly VaR': f"{var_value * np.sqrt(5):.2f}%",
                'Monthly VaR': f"{var_value * np.sqrt(22):.2f}%"
            })
        
        st.dataframe(pd.DataFrame(var_results), use_container_width=True)
        
        # VaR visualization
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=portfolio_return_series * 100, nbinsx=50, name='Portfolio Returns'))
        
        for conf in confidence_levels:
            var_line = calculate_var(portfolio_return_series, conf)
            fig.add_vline(x=var_line, line_dash="dash", 
                         annotation_text=f"VaR {(1-conf)*100:.0f}%")
        
        fig.update_layout(
            title="Portfolio Return Distribution with VaR Levels",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        # Stress testing
        st.markdown("**Scenario Analysis**")
        
        scenarios = {
            "Market Crash (-20%)": -0.20,
            "Moderate Decline (-10%)": -0.10,
            "Volatility Spike (+50% vol)": 0.0,
            "Bull Market (+15%)": 0.15
        }
        
        stress_results = []
        # Portfolio construction
        st.markdown("### 📋 Portfolio Construction")

        col1, col2 = st.columns(2)
        with col1:
            base_portfolio_value = st.number_input(
                "Portfolio Value ($)",
                min_value=500,
                max_value=10000000,
                value=100000,
                step=1000,
                format="%d",
                key="portfolio_value"
            )
        with col2:
            st.metric("Portfolio Value", f"${base_portfolio_value:,}")

        available_tickers = sorted(valid_metrics['ticker'].unique())
        
        for scenario_name, market_shock in scenarios.items():
            if "Volatility" in scenario_name:
                # Simulate higher volatility
                shocked_returns = portfolio_return_series * 1.5  # 50% higher volatility
                portfolio_impact = shocked_returns.std() * np.sqrt(252) * 100
                value_impact = f"{portfolio_impact:.2f}% vol"
                impact_display = f"{portfolio_impact:.2f}% vol"
            else:
                # Simulate market shock
                avg_beta = np.mean([calculate_beta(portfolio_returns[stock], portfolio_returns.mean(axis=1)) 
                                for stock in valid_stocks if not np.isnan(calculate_beta(portfolio_returns[stock], portfolio_returns.mean(axis=1)))])
                if np.isnan(avg_beta):
                    avg_beta = 1.0
                
                portfolio_impact = market_shock * avg_beta
                new_portfolio_value = base_portfolio_value * (1 + portfolio_impact)
                value_change = new_portfolio_value - base_portfolio_value
                
                value_impact = f"${new_portfolio_value:,.0f}"
                impact_display = f"{portfolio_impact*100:.2f}% (${value_change:+,.0f})"
            
            stress_results.append({
                'Scenario': scenario_name,
                'Portfolio Impact': impact_display,
                'New Portfolio Value': value_impact
            })
            
        st.dataframe(pd.DataFrame(stress_results), use_container_width=True)

# ---------------------------
# Sentiment Analysis Module
# ---------------------------

def get_news_sentiment(ticker, company_name, days_back=7):
    """
    Fetch news articles and calculate sentiment score for a ticker
    
    Args:
        ticker: Stock ticker symbol
        company_name: Full company name for better search results
        days_back: Number of days to look back for news
    
    Returns:
        Dictionary with sentiment metrics
    """
    if not NEWS_API_KEY:
        return {
            'sentiment_score': 0,
            'sentiment_label': 'Neutral',
            'article_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'error': 'No API key configured'
        }
    
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        search_query = f'{ticker} OR "{company_name}"'
        
        articles = newsapi.get_everything(
            q=search_query,
            language='en',
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            sort_by='relevancy',
            page_size=50
        )
        
        if not articles or articles['totalResults'] == 0:
            return {
                'sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles['articles']:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            if text.strip():
                vader_scores = sentiment_analyzer.polarity_scores(text)
                compound_score = vader_scores['compound']
                
                sentiments.append(compound_score)
                
                if compound_score >= 0.05:
                    positive_count += 1
                elif compound_score <= -0.05:
                    negative_count += 1
                else:
                    neutral_count += 1
        
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            
            if avg_sentiment >= 0.05:
                sentiment_label = 'Positive'
            elif avg_sentiment <= -0.05:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
        else:
            avg_sentiment = 0
            sentiment_label = 'Neutral'
        
        return {
            'sentiment_score': float(avg_sentiment),
            'sentiment_label': sentiment_label,
            'article_count': len(sentiments),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
        
    except Exception as e:
        if VERBOSE:
            print(f"Error fetching sentiment for {ticker}: {e}")
        return {
            'sentiment_score': 0,
            'sentiment_label': 'Neutral',
            'article_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'error': str(e)
        }

def get_yfinance_news_sentiment(ticker):
    """
    Use yfinance news (no API key needed, but limited data)
    Enhanced with better error handling and fallback methods
    """
    try:
        stock = yf.Ticker(ticker)
        news = None
        
        try:
            news = stock.news
        except (AttributeError, KeyError) as e:
            logger.debug(f"Method 1 failed for {ticker}: {e}")
        
        if not news:
            try:
                news = stock.get_news()
            except (AttributeError, KeyError) as e:
                logger.debug(f"Method 2 failed for {ticker}: {e}")
        
        # Method 2: Try get_news() method if available
        if not news:
            try:
                news = stock.get_news()
            except:
                pass
        
        # Method 3: Check the info dict for news
        if not news:
            try:
                info = stock.info
                if 'news' in info:
                    news = info['news']
            except:
                pass
        
        # If still no news, return neutral with debug info
        if not news:
            return {
                'sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'article_count': 0,
                'source': 'yfinance',
                'debug': 'No news available from yfinance'
            }
        
        # Process news articles
        sentiments = []
        
        for article in news[:20]:  # Limit to 20 articles
            # Handle different article formats
            if isinstance(article, dict):
                title = article.get('title', '') or article.get('headline', '')
            else:
                title = str(article)
            
            if title and len(title) > 5:  # Ensure title has content
                try:
                    vader_scores = sentiment_analyzer.polarity_scores(title)
                    sentiments.append(vader_scores['compound'])
                except Exception as e:
                    if VERBOSE:
                        print(f"Error analyzing sentiment for {ticker}: {e}")
                    continue
        
        # Calculate average sentiment
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            
            if avg_sentiment >= 0.05:
                sentiment_label = 'Positive'
            elif avg_sentiment <= -0.05:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
        else:
            avg_sentiment = 0
            sentiment_label = 'Neutral'
        
        return {
            'sentiment_score': float(avg_sentiment),
            'sentiment_label': sentiment_label,
            'article_count': len(sentiments),
            'source': 'yfinance',
            'debug': f'Processed {len(sentiments)} articles from {len(news)} available'
        }
        
    except Exception as e:
        error_msg = str(e)
        if VERBOSE:
            print(f"Error fetching yfinance sentiment for {ticker}: {error_msg}")
        
        return {
            'sentiment_score': 0,
            'sentiment_label': 'Neutral',
            'article_count': 0,
            'source': 'yfinance',
            'error': error_msg,
            'debug': 'Exception occurred during sentiment analysis'
        }

def get_google_news_sentiment(ticker, company_name):
    """
    Fallback: Use Google News RSS feed (free, no API key needed)
    More reliable than yfinance
    """
    try:
        import feedparser
        
        # Google News RSS feed
        query = f"{ticker} stock OR {company_name}"
        # URL encode the query
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        # Parse the RSS feed
        feed = feedparser.parse(url)
        
        if not feed.entries:
            return {
                'sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'article_count': 0,
                'source': 'google_news',
                'debug': 'No articles found in Google News RSS'
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Analyze up to 20 articles
        for entry in feed.entries[:20]:
            title = entry.get('title', '')
            
            if title and len(title) > 5:
                vader_scores = sentiment_analyzer.polarity_scores(title)
                compound_score = vader_scores['compound']
                sentiments.append(compound_score)
                
                if compound_score >= 0.05:
                    positive_count += 1
                elif compound_score <= -0.05:
                    negative_count += 1
                else:
                    neutral_count += 1
        
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            
            if avg_sentiment >= 0.05:
                sentiment_label = 'Positive'
            elif avg_sentiment <= -0.05:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
        else:
            avg_sentiment = 0
            sentiment_label = 'Neutral'
        
        return {
            'sentiment_score': float(avg_sentiment),
            'sentiment_label': sentiment_label,
            'article_count': len(sentiments),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'source': 'google_news',
            'debug': f'Successfully processed {len(sentiments)} articles from Google News'
        }
        
    except Exception as e:
        if VERBOSE:
            print(f"Error with Google News for {ticker}: {e}")
        return {
            'sentiment_score': 0,
            'sentiment_label': 'Neutral',
            'article_count': 0,
            'source': 'google_news',
            'error': str(e),
            'debug': 'Exception occurred'
        }
    
def analyze_sentiment_mixed(valid_metrics, horizon_col, max_newsapi_requests=75):
    """
    Smart sentiment analysis - uses Google News by default, NewsAPI for top movers if key available
    
    Args:
        valid_metrics: DataFrame with stock metrics
        horizon_col: Column to determine "top movers" (only used if NewsAPI available)
        max_newsapi_requests: Max NewsAPI requests (default 75, only used if NewsAPI available)
    
    Returns:
        DataFrame with sentiment data
    """
    sentiment_data = []
    all_tickers = valid_metrics['ticker'].tolist()
    company_names = dict(zip(valid_metrics['ticker'], valid_metrics['company_name']))
    
    # DEFAULT: Use Google News for everything if no NewsAPI key
    if not NEWS_API_KEY:
        print("\n📰 Using Google News RSS for all stocks (free, no API key required)")
        print(f"Analyzing sentiment for {len(all_tickers)} stocks...")
        
        for ticker in tqdm(all_tickers, desc="Google News Analysis"):
            company_name = company_names.get(ticker, ticker)
            sentiment = get_google_news_sentiment(ticker, company_name)
            sentiment['ticker'] = ticker
            sentiment_data.append(sentiment)
            time.sleep(0.3)  # Light rate limiting
        
        print(f"✓ Completed: {len(all_tickers)} stocks analyzed with Google News RSS")
        return pd.DataFrame(sentiment_data)
    
    # PREMIUM: NewsAPI available - use mixed approach
    print("\n📊 Mixed Strategy: NewsAPI + Google News")
    
    valid_metrics['abs_return'] = valid_metrics[horizon_col].abs()
    top_movers = valid_metrics.nlargest(max_newsapi_requests, 'abs_return')
    
    newsapi_tickers = set(top_movers['ticker'].tolist())
    
    newsapi_count = 0
    google_news_count = 0
    
    print(f"  • Top {max_newsapi_requests} movers: NewsAPI (premium)")
    print(f"  • Remaining {len(all_tickers) - max_newsapi_requests}: Google News (free)")
    
    for ticker in tqdm(all_tickers, desc="Mixed Analysis"):
        company_name = company_names.get(ticker, ticker)
        
        if ticker in newsapi_tickers and newsapi_count < max_newsapi_requests:
            sentiment = get_news_sentiment(ticker, company_name)
            sentiment['source'] = 'NewsAPI'
            newsapi_count += 1
            time.sleep(0.5)
        else:
            sentiment = get_google_news_sentiment(ticker, company_name)
            google_news_count += 1
            time.sleep(0.3)
        
        sentiment['ticker'] = ticker
        sentiment_data.append(sentiment)
    
    print(f"✓ Completed: {newsapi_count} NewsAPI + {google_news_count} Google News")
    
    return pd.DataFrame(sentiment_data)
    
    # PREMIUM: NewsAPI available - use mixed approach
    print("\n📊 Mixed Strategy: NewsAPI + Google News")
    
    valid_metrics['abs_return'] = valid_metrics[horizon_col].abs()
    top_movers = valid_metrics.nlargest(max_newsapi_requests, 'abs_return')
    
    newsapi_tickers = set(top_movers['ticker'].tolist())
    
    newsapi_count = 0
    google_news_count = 0
    
    print(f"  • Top {max_newsapi_requests} movers: NewsAPI (premium)")
    print(f"  • Remaining {len(all_tickers) - max_newsapi_requests}: Google News (free)")
    
    for ticker in tqdm(all_tickers, desc="Mixed Analysis"):
        company_name = company_names.get(ticker, ticker)
        
        if ticker in newsapi_tickers and newsapi_count < max_newsapi_requests:
            sentiment = get_news_sentiment(ticker, company_name)
            sentiment['source'] = 'NewsAPI'
            newsapi_count += 1
            time.sleep(0.5)
        else:
            sentiment = get_google_news_sentiment(ticker, company_name)
            google_news_count += 1
            time.sleep(0.3)
        
        sentiment['ticker'] = ticker
        sentiment_data.append(sentiment)
    
    print(f"✓ Completed: {newsapi_count} NewsAPI + {google_news_count} Google News")
    
    return pd.DataFrame(sentiment_data)

def get_sentiment_strategy_summary(sentiment_df):
    """Get summary of which source was used"""
    if sentiment_df.empty or 'source' not in sentiment_df.columns:
        return None
    
    summary = sentiment_df['source'].value_counts()
    
    return {
        'newsapi_count': summary.get('NewsAPI', 0),
        'google_news_count': summary.get('google_news', 0), 
        'total': len(sentiment_df)
    }

def filter_high_quality_sentiment(sentiment_df, min_articles=3):
    """Filter sentiment data to only high-quality results"""
    mask = (
        (sentiment_df['source'] == 'google_news') |
        ((sentiment_df['source'] == 'NewsAPI') & (sentiment_df['article_count'] >= min_articles))
    )
    
    return sentiment_df[mask]

def get_sentiment_color(sentiment_score):
    """Get color for sentiment visualization"""
    if pd.isna(sentiment_score):
        return '#888888'
    
    if sentiment_score >= 0.5:
        return '#006400'
    elif sentiment_score >= 0.05:
        return '#32CD32'
    elif sentiment_score > -0.05:
        return '#FFFF99'
    elif sentiment_score > -0.5:
        return '#FF6347'
    else:
        return '#8B0000'

def create_sentiment_chart(sentiment_df, top_n=20):
    """Create sentiment visualization chart"""
    if sentiment_df.empty:
        return None
    
    sorted_df = sentiment_df.nlargest(top_n, 'sentiment_score')
    
    colors = [get_sentiment_color(score) for score in sorted_df['sentiment_score']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_df['ticker'],
            y=sorted_df['sentiment_score'],
            marker_color=colors,
            text=sorted_df['sentiment_label'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                          'Sentiment: %{y:.3f}<br>' +
                          'Articles: %{customdata}<extra></extra>',
            customdata=sorted_df['article_count']
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Stocks by News Sentiment',
        xaxis_title='Ticker',
        yaxis_title='Sentiment Score',
        height=400,
        yaxis=dict(range=[-1, 1])
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def merge_sentiment_with_metrics(metrics_df, sentiment_df):
    """Merge sentiment data with existing metrics"""
    return metrics_df.merge(
        sentiment_df[['ticker', 'sentiment_score', 'sentiment_label', 'article_count', 'source']],
        on='ticker',
        how='left'
    )

def get_sector_sentiment_summary(sentiment_df, metrics_df):
    """Calculate average sentiment by sector"""
    merged = sentiment_df.merge(
        metrics_df[['ticker', 'sector']], 
        on='ticker', 
        how='left'
    )
    
    sector_summary = merged.groupby('sector').agg({
        'sentiment_score': 'mean',
        'article_count': 'sum',
        'ticker': 'count'
    }).round(3)
    
    sector_summary.columns = ['Avg Sentiment', 'Total Articles', 'Stock Count']
    sector_summary = sector_summary.sort_values('Avg Sentiment', ascending=False)
    
    return sector_summary

def create_sentiment_comparison_chart(sentiment_df):
    """Create chart comparing NewsAPI vs yfinance sentiment quality"""
    if 'source' not in sentiment_df.columns:
        return None
    
    source_summary = sentiment_df.groupby('source').agg({
        'sentiment_score': 'mean',
        'article_count': 'mean',
        'ticker': 'count'
    }).round(3)
    
    source_summary.columns = ['Avg Sentiment', 'Avg Articles', 'Stock Count']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Stock Count',
        x=source_summary.index,
        y=source_summary['Stock Count'],
        yaxis='y',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Avg Articles',
        x=source_summary.index,
        y=source_summary['Avg Articles'],
        yaxis='y2',
        marker_color='orange'
    ))
    
    fig.update_layout(
        title='Sentiment Data Sources Comparison',
        xaxis_title='Source',
        yaxis=dict(title='Stock Count', side='left'),
        yaxis2=dict(title='Avg Articles per Stock', overlaying='y', side='right'),
        height=400,
        barmode='group'
    )
    
    return fig

# ---------------------------
# CLI Main Function
# ---------------------------
def run_cli(consecutive_days=7, index_key="SP500"):
    """Run the CLI version of the market tracker with configurable consecutive period"""
    
    # Allow running all indices with 'ALL'
    if index_key == 'ALL':
        indices = ['SP500', 'SP400', 'SP600', 'NASDAQ100', 'DOW30', 'COMBINED']
        for idx in indices:
            print(f"\n=== Running index: {idx} ===")
            run_cli(consecutive_days=consecutive_days, index_key=idx)
        return

    # Create index-specific data directory
    index_data_dir = os.path.join(DATA_DIR, index_key)
    ensure_dir(index_data_dir)

    index_names = {
            "SP500": "S&P 500",
            "SP400": "S&P MidCap 400",
            "SP600": "S&P SmallCap 600",
            "NASDAQ100": "Nasdaq-100",
            "DOW30": "Dow Jones 30",
            "COMBINED": "Combined (S&P 500 + MidCap 400 + SmallCap 600 + Nasdaq-100 + Dow 30)"
        }
    
    index_display = index_names.get(index_key, "S&P 500")
    
    print("=" * 60)
    print(f"{index_display} Market Tracker - CLI Mode (Consecutive: {consecutive_days} days)")
    print("=" * 60)

    print(f"\nFetching {index_display} tickers...")
    
    # Fetch appropriate index
    if index_key == "SP400":
        tickers, sp_df, sectors, industries, company_names = fetch_sp400_tickers()
    elif index_key == "SP600":
        tickers, sp_df, sectors, industries, company_names = fetch_sp600_tickers()
    elif index_key == "NASDAQ100":
        tickers, sp_df, sectors, industries, company_names = fetch_nasdaq100_tickers()
    elif index_key == "DOW30":
        tickers, sp_df, sectors, industries, company_names = fetch_dow30_tickers()
    elif index_key == "COMBINED":
        tickers, sp_df, sectors, industries, company_names = fetch_combined_tickers()
    else:  # Default to S&P 500
        tickers, sp_df, sectors, industries, company_names = fetch_sp500_tickers()

    if not tickers:
        print("Failed to fetch tickers. Exiting.")
        return

    print(f"Found {len(tickers)} tickers")

    # Save constituents snapshot
    sp_df.to_csv(os.path.join(index_data_dir, f"{index_key}_constituents_snapshot.csv"), index=False)

    # Download historical data
    print("\nDownloading historical price data...")
    print("This may take several minutes depending on your connection speed.")

    hist = download_history(tickers, period=DOWNLOAD_PERIOD, interval="1d")

    metrics_rows = []
    per_ticker_dir = os.path.join(index_data_dir, "history")
    ensure_dir(per_ticker_dir)

    print(f"\nProcessing ticker data with {consecutive_days}-day consecutive analysis...")

    def process_single_ticker(ticker_data):
        """Process a single ticker for parallel execution"""
        t, df, consecutive_days, company_names, sectors, industries, per_ticker_dir = ticker_data
        
        if df is None or df.empty:
            return {
                "ticker": t,
                "company_name": company_names.get(t, "Unknown"),  
                "status": "no_data",
                "sector": sectors.get(t, "Unknown"),
                "industry": industries.get(t, "Unknown")
            }
        
        # Save per-ticker CSV
        try:
            safe_filename = sanitize_filename_windows(f"{t}.csv")
            df.to_csv(os.path.join(per_ticker_dir, safe_filename))
        except Exception:
            pass

        metrics = compute_metrics_for_ticker(df, consecutive_days)

        if metrics is None:
            return {
                "ticker": t,
                "company_name": company_names.get(t, "Unknown"),  
                "status": "no_data",
                "sector": sectors.get(t, "Unknown"),
                "industry": industries.get(t, "Unknown")
            }
        else:
            metrics['ticker'] = t
            metrics['company_name'] = company_names.get(t, "Unknown") 
            metrics['status'] = 'ok'
            metrics['sector'] = sectors.get(t, "Unknown")
            metrics['industry'] = industries.get(t, "Unknown")
            return metrics

    # Prepare data for parallel processing
    ticker_data_list = [
        (t, hist.get(t, pd.DataFrame()), consecutive_days, company_names, sectors, industries, per_ticker_dir)
        for t in tickers
    ]

    # Use parallel processing (use fewer workers to avoid overwhelming CPU)
    max_workers = min(4, os.cpu_count() or 1)  # Max 4 workers
    print(f"Using {max_workers} parallel workers for metrics calculation...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process in parallel with progress bar
        results = list(tqdm(
            executor.map(process_single_ticker, ticker_data_list),
            total=len(ticker_data_list),
            desc="Computing metrics"
        ))

    metrics_rows = results

    df_metrics = pd.DataFrame(metrics_rows)
    print("\nFetching P/E ratios...")
    df_metrics = fetch_pe_ratios(tickers, df_metrics)

    # Save to PostgreSQL
    print("\n" + "=" * 60)
    print("SAVING TO DATABASE")
    print("=" * 60)

    save_successful = False

    if pg_manager:
        print(f"📊 Saving {len(df_metrics)} stocks to PostgreSQL...")
        try:
            pg_manager.save_metrics(df_metrics, index_key)
            print(f"✅ Successfully saved to PostgreSQL!")
            save_successful = True
        except Exception as e:
            print(f"❌ PostgreSQL save failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"⚠️ Falling back to CSV-only mode for this run")
    else:
        print("⚠️ No PostgreSQL connection. Using CSV-only mode.")

    # Always save CSV backup regardless of database status
    try:
        master_csv = os.path.join(index_data_dir, "latest_metrics.csv")
        ensure_dir(index_data_dir)
        df_metrics.to_csv(master_csv, index=False)
        print(f"✓ Saved CSV backup to {master_csv}")
    except Exception as csv_error:
        print(f"❌ CRITICAL: CSV backup also failed: {csv_error}")
        # try to save to a temp location
        try:
            temp_csv = os.path.join(DATA_DIR, f"emergency_backup_{index_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df_metrics.to_csv(temp_csv, index=False)
            print(f"⚠️ Saved emergency backup to {temp_csv}")
        except:
            print(f"❌ FATAL: All save methods failed!")

    print("=" * 60 + "\n")

    # Dynamic column ordering based on consecutive_days
    cols_order = [
        'ticker', 'company_name', 'status', 'sector', 'industry', 'last_date', 'last_close', 'pe_ratio',
        'data_points', f'rising_{consecutive_days}day', f'declining_{consecutive_days}day'
    ]

    # Add 7-day columns if different from consecutive_days
    if consecutive_days != 7:
        cols_order.extend(['rising_7day', 'declining_7day'])

    cols_order.extend([
        'pct_1d', 'pct_3d', 'pct_5d', 'pct_21d', 'pct_63d', 'pct_252d'
    ])
    
    # Add consecutive_days percentage if not in standard list
    if consecutive_days not in [1, 3, 5, 7, 21, 63, 252]:
        cols_order.insert(-6, f'pct_{consecutive_days}d')
    
    cols_order.extend([
        'avg_1y', 'avg_2y', 'avg_5y', 'avg_max', 'cum_return_from_start_pct', 
        'ann_vol_pct', 'rsi', '52w_high', '52w_low', 'pct_from_52w_high', 
        'pct_from_52w_low', 'avg_volume_20d', 'volume_vs_avg'
    ])

    cols = [c for c in cols_order if c in df_metrics.columns]
    cols += [c for c in df_metrics.columns if c not in cols]
    df_metrics = df_metrics[cols]

    # Save to PostgreSQL
    if pg_manager:
        pg_manager.save_metrics(df_metrics, index_key)
    else:
        # Fallback to CSV only
        st.warning("No database connection. Data saved to CSV only.")

    # Save master CSV
    master_csv = os.path.join(index_data_dir, "latest_metrics.csv")
    ensure_dir(index_data_dir)
    df_metrics.to_csv(master_csv, index=False)
    print(f"\n✓ Saved master metrics to {master_csv}")

    # Rising and declining lists using configurable period
    rising_col = f'rising_{consecutive_days}day'
    declining_col = f'declining_{consecutive_days}day'

    # Rising and declining lists using configurable period
    if consecutive_days in [1, 3, 5, 21, 63, 252]:
        # Use the standard column name
        pct_col = f'pct_{consecutive_days}d'
    elif f'pct_{consecutive_days}d' in df_metrics.columns:
        # Use the custom column if it exists
        pct_col = f'pct_{consecutive_days}d'
    else:
        # Fallback to the closest available column
        if consecutive_days <= 1:
            pct_col = 'pct_1d'
        elif consecutive_days <= 3:
            pct_col = 'pct_3d'
        elif consecutive_days <= 5:
            pct_col = 'pct_5d'
        elif consecutive_days <= 21:
            pct_col = 'pct_21d'
        elif consecutive_days <= 63:
            pct_col = 'pct_63d'
        else:
            pct_col = 'pct_252d'

    # Verify the column exists before using it
    if pct_col not in df_metrics.columns:
        print(f"Warning: Column {pct_col} not found. Available columns: {list(df_metrics.columns)}")
        # Use pct_5d as final fallback
        pct_col = 'pct_5d' if 'pct_5d' in df_metrics.columns else df_metrics.columns[0]

    print(f"Using {pct_col} for sorting (consecutive days: {consecutive_days})")

    rising = df_metrics[
        (df_metrics['status'] == 'ok') &
        (df_metrics[rising_col] == True)
    ].sort_values(by=pct_col, ascending=False)

    declining = df_metrics[
        (df_metrics['status'] == 'ok') &
        (df_metrics[declining_col] == True)
    ].sort_values(by=pct_col)

    rising.to_csv(os.path.join(index_data_dir, f"rising_{consecutive_days}day.csv"), index=False)
    declining.to_csv(os.path.join(index_data_dir, f"declining_{consecutive_days}day.csv"), index=False)

    # Print summary with configurable period
    print("\n" + "=" * 60)
    print("MARKET SUMMARY")
    print("=" * 60)

    valid_metrics = df_metrics[df_metrics['status'] == 'ok'].copy()
    # Validate consecutive day columns exist
    if rising_col not in df_metrics.columns:
        available_rising_cols = [col for col in df_metrics.columns if col.startswith('rising_') and col.endswith('day')]
        if available_rising_cols:
            # Use first available as fallback
            actual_rising_col = available_rising_cols[0]
            actual_declining_col = actual_rising_col.replace('rising_', 'declining_')
            fallback_days = actual_rising_col.replace('rising_', '').replace('day', '')
            st.info(f"Using {fallback_days}-day data. Set consecutive days to {fallback_days} or update data for {consecutive_days}-day analysis.")
            rising_col = actual_rising_col
            declining_col = actual_declining_col
    if not valid_metrics.empty:
        print(f"\nOverall Statistics:")
        print(f"  • Total stocks processed: {len(valid_metrics)}")
        print(f"  • Rising ({consecutive_days}-day): {len(rising)} stocks")
        print(f"  • Declining ({consecutive_days}-day): {len(declining)} stocks")

        print(f"\nTop 5 Rising Stocks ({consecutive_days}-day consecutive):")
        for _, row in rising.head().iterrows():
            pct_val = row.get(pct_col, 0)
            print(f"  • {row['ticker']}: +{pct_val:.2f}% ({row['sector']})")

        print(f"\nTop 5 Declining Stocks ({consecutive_days}-day consecutive):")
        for _, row in declining.head().iterrows():
            pct_val = row.get(pct_col, 0)
            print(f"  • {row['ticker']}: {pct_val:.2f}% ({row['sector']})")

    print("\n✅ Analysis complete! Check the 'data' directory for all output files.")
    print("=" * 60)

# ---------------------------
# Streamlit Web Interface
# ---------------------------
def plot_ticker_price_rsi(ticker_csv_path, ticker):
    """
    Helper: load per-ticker data and produce a 2-row plot: price and RSI
    
    Args:
        ticker_csv_path: Path to CSV file with historical data
        ticker: Stock ticker symbol
    """
    # Initialize empty DataFrame
    df = pd.DataFrame()
    
    # Try to load historical data from CSV
    if os.path.exists(ticker_csv_path):
        try:
            df = pd.read_csv(ticker_csv_path, parse_dates=True, index_col=0)
        except Exception as e:
            st.error(f"❌ Error loading data for {ticker}: {e}")
            st.info("💡 Try updating the data using the 'Update Data' button in the sidebar.")
            return
    else:
        st.warning(f"⚠️ Data file not found for {ticker}")
        st.info(f"Expected location: `{ticker_csv_path}`")
        return
    
    # Validate data
    if df.empty:
        st.info(f"ℹ️ No historical data available for {ticker}.")
        return
    
    if 'Close' not in df.columns:
        st.error(f"❌ Invalid data format for {ticker}: missing 'Close' column")
        st.caption(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Add technical indicators
    try:
        df = add_technical_indicators(df)
    except Exception as e:
        st.error(f"❌ Error calculating technical indicators: {e}")
        # Continue with basic plot even if indicators fail
    
    # Build subplot with price and RSI
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3], 
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Add close price
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name=f"{ticker} Close", line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'MA20' in df.columns and df['MA20'].notna().any():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(dash='dash', color='orange')),
            row=1, col=1
        )
    
    if 'MA50' in df.columns and df['MA50'].notna().any():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(dash='dot', color='red')),
            row=1, col=1
        )
    
    # Add RSI if available
    if 'RSI' in df.columns and df['RSI'].notna().any():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    else:
        st.caption("⚠️ RSI indicator not available")
    
    # Update layout
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    fig.update_layout(
        height=600, 
        title_text=f"{ticker} - Price & Technical Indicators",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Show data summary
    with st.expander("📊 Data Summary"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Points", len(df))
        
        with col2:
            date_range = (df.index.max() - df.index.min()).days
            st.metric("Date Range", f"{date_range} days")
        
        with col3:
            current_price = df['Close'].iloc[-1]
            st.metric("Latest Price", f"${current_price:.2f}")
        
        with col4:
            if 'RSI' in df.columns:
                current_rsi = df['RSI'].iloc[-1]
                if pd.notna(current_rsi):
                    st.metric("Current RSI", f"{current_rsi:.1f}")

# ENHANCED DATA FETCHING AND CACHING
def fetch_and_enhance_metrics(valid_metrics: pd.DataFrame, 
                              force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch enhanced data and merge with base metrics
    
    This function:
    1. Checks if enhanced_data_fetcher.py is available
    2. Uses cached data if less than 24 hours old
    3. Fetches fresh comprehensive data if needed
    4. Merges enhanced data with base metrics
    
    Args:
        valid_metrics: Base metrics DataFrame from your existing system
        force_refresh: Force fetch even if cached (useful for daily updates)
    
    Returns:
        Enhanced metrics DataFrame (or base metrics if enhancement fails)
    """
    # Check if enhanced data fetcher is available
    if not ENHANCED_DATA_AVAILABLE:
        st.warning("⚠️ Enhanced data fetcher not installed. Using base metrics only.")
        with st.expander("ℹ️ How to Enable Enhanced Features"):
            st.markdown("""
            To enable comprehensive fundamental and institutional analysis:
            
            1. Ensure `enhanced_data_fetcher.py` is in the same directory as `market_tracker.py`
            2. The enhanced fetcher provides:
               - Forward P/E ratios
               - Earnings dates and beat rates
               - Revenue/earnings growth metrics
               - Profit margins and ROE
               - Institutional ownership data
               - Insider trading activity
               - Short interest data
               - Analyst recommendations
            
            **Note:** All data comes from free yfinance API (no API key needed)
            """)
        return valid_metrics
    
    cache_file = os.path.join(DATA_DIR, "enhanced_metrics_cache.pkl")
    
    # Check cache (24-hour validity)
    if not force_refresh and os.path.exists(cache_file):
        cache_age = time.time() - os.path.getmtime(cache_file)
        hours_old = cache_age / 3600
        
        if cache_age < 86400:  # 24 hours
            st.info(f"📦 Loading enhanced data from cache ({hours_old:.1f} hours old)...")
            try:
                enhanced_df = pd.read_pickle(cache_file)
                merged = merge_enhanced_with_base_metrics(valid_metrics, enhanced_df)
                st.success(f"✅ Loaded {len(enhanced_df)} enhanced records from cache")
                
                # Show what's included
                with st.expander("📊 Enhanced Data Summary"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        has_earnings = enhanced_df['days_to_earnings'].notna().sum()
                        st.metric("Stocks with Earnings Data", has_earnings)
                    
                    with col2:
                        has_institutional = enhanced_df['institutional_ownership_pct'].notna().sum()
                        st.metric("Institutional Data", has_institutional)
                    
                    with col3:
                        has_analyst = enhanced_df['analyst_target_price'].notna().sum()
                        st.metric("Analyst Coverage", has_analyst)
                
                return merged
                
            except Exception as e:
                st.warning(f"⚠️ Cache error: {e}. Fetching fresh data...")
    
    # Fetch fresh data
    st.info("🔄 Fetching comprehensive fundamental and institutional data...")
    st.caption("""
    **What's being fetched (FREE via yfinance):**
    - ⏰ Earnings dates and history
    - 📊 Forward P/E and valuation metrics
    - 📈 Revenue/earnings growth
    - 💰 Profit margins and ROE
    - 🏢 Institutional ownership
    - 👔 Insider trading activity
    - 📉 Short interest data
    - 🎯 Analyst recommendations
    
    **Time estimate:** ~3-5 minutes for 500 stocks
    """)
    
    tickers = valid_metrics['ticker'].tolist()
    
    # Show progress estimate
    total_tickers = len(tickers)
    estimated_minutes = (total_tickers * 0.3) / 60  # ~0.3 seconds per ticker
    
    st.info(f"📊 Fetching data for {total_tickers} tickers (est. {estimated_minutes:.1f} minutes)")
    
    try:
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data in batches with progress updates
        from enhanced_data_fetcher import fetch_enhanced_data_batch
        
        status_text.text(f"Fetching enhanced data for {total_tickers} stocks...")
        enhanced_df = fetch_enhanced_data_batch(tickers, verbose=False)
        
        progress_bar.progress(100)
        status_text.text("✅ Enhanced data fetch complete!")
        
        # Save to cache
        try:
            enhanced_df.to_pickle(cache_file)
            st.success(f"✅ Cached {len(enhanced_df)} enhanced records for future use")
        except Exception as e:
            st.warning(f"⚠️ Could not cache data: {e}")
        
        # Merge with base metrics
        merged = merge_enhanced_with_base_metrics(valid_metrics, enhanced_df)
        
        # Show summary
        with st.expander("📊 Enhanced Data Summary"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                has_earnings = enhanced_df['days_to_earnings'].notna().sum()
                pct_earnings = (has_earnings / len(enhanced_df) * 100)
                st.metric("Earnings Data", f"{has_earnings} ({pct_earnings:.0f}%)")
            
            with col2:
                has_institutional = enhanced_df['institutional_ownership_pct'].notna().sum()
                pct_inst = (has_institutional / len(enhanced_df) * 100)
                st.metric("Institutional", f"{has_institutional} ({pct_inst:.0f}%)")
            
            with col3:
                has_analyst = enhanced_df['analyst_target_price'].notna().sum()
                pct_analyst = (has_analyst / len(enhanced_df) * 100)
                st.metric("Analyst Data", f"{has_analyst} ({pct_analyst:.0f}%)")
            
            with col4:
                has_insider = enhanced_df['insider_buy_sell_ratio'].notna().sum()
                pct_insider = (has_insider / len(enhanced_df) * 100)
                st.metric("Insider Data", f"{has_insider} ({pct_insider:.0f}%)")
        
        st.balloons()
        
        return merged
        
    except Exception as e:
        st.error(f"❌ Error fetching enhanced data: {e}")
        
        with st.expander("🐛 Error Details"):
            import traceback
            st.code(traceback.format_exc())
        
        st.warning("⚠️ Falling back to base metrics only.")
        st.info("💡 You can still use the recommendation system with base metrics, but some features will be limited.")
        
        return valid_metrics

# PostgreSQL data loading with caching
@cache_data(ttl=3600, show_spinner=False)
def load_data_from_postgres(index_name):
    """Load data from PostgreSQL with intelligent caching"""
    if pg_manager:
        try:
            df = pg_manager.load_metrics(index_name)
            
            if df is not None and not df.empty:
                # Cache metadata for faster subsequent checks
                cache_metadata = {
                    'row_count': len(df),
                    'last_updated': df['updated_at'].max() if 'updated_at' in df.columns else None,
                    'cached_at': pd.Timestamp.now()
                }
                st.session_state[f'cache_meta_{index_name}'] = cache_metadata
                return df
        except Exception as e:
            print(f"PostgreSQL load failed: {e}")
    
    # Fallback to CSV if PostgreSQL unavailable or empty
    csv_path = os.path.join(DATA_DIR, index_name, "latest_metrics.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Parse date columns if present
            if 'last_date' in df.columns:
                df['last_date'] = pd.to_datetime(df['last_date'], errors='coerce')
            return df
        except Exception as e:
            print(f"CSV load failed: {e}")
    
    return None

# Add cache invalidation function
def invalidate_cache(index_name=None):
    """Manually invalidate cached data"""
    if index_name:
        # Clear specific index cache
        cache_key = f'cache_meta_{index_name}'
        if cache_key in st.session_state:
            del st.session_state[cache_key]
    else:
        # Clear all caches
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith('cache_meta_')]
        for key in keys_to_delete:
            del st.session_state[key]
    
    # Clear Streamlit's cache_data
    load_data_from_postgres.clear()

# STREAMLIT WEB INTERFACE
def run_streamlit():
    """Run the Streamlit web interface"""

    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit is not installed. Please install with:")
        print("pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="Market Tracker",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Add custom CSS
    add_custom_css()

    # Sidebar - Controls
    st.sidebar.header("⚙️ Settings")

    # 0. Index selection
    index_selection = st.sidebar.selectbox(
        "Select Index",
        ["S&P 500", "S&P MidCap 400", "S&P SmallCap 600", "Nasdaq-100", "Dow Jones 30", "Combined Stock Indices"],
        key="index_selector",
        help="Choose which market index to analyze"
    )
    
    # Map display name to index key
    index_map = {
        "S&P 500": "SP500",
        "S&P MidCap 400": "SP400",
        "S&P SmallCap 600": "SP600",
        "Nasdaq-100": "NASDAQ100",
        "Dow Jones 30": "DOW30",
        "Combined Stock Indices": "COMBINED"
    }
    current_index = index_map[index_selection]
    
    # Get current index-specific data directory
    index_data_dir = os.path.join(DATA_DIR, current_index)

    # Auto-fetch data if not present
    if not os.path.exists(os.path.join(index_data_dir, "latest_metrics.csv")):
        if 'data_fetched' not in st.session_state:
            with st.spinner(f"First time setup: Fetching {index_selection} data... This will take about 5-10 minutes."):
                try:
                    run_cli(consecutive_days=7, index_key=current_index)
                    st.session_state.data_fetched = True
                    st.success("✅ Data fetched successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to fetch data: {e}")
                    st.info("Click 'Fetch All Data' in the sidebar to try again.")
    # Header
    st.title(f"📈 {index_selection} Market Tracker")

    # Check for existing data
    data_exists = os.path.exists(os.path.join(index_data_dir, "latest_metrics.csv"))

    # 1. View mode
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Dashboard", "Sector Analysis", "Top Movers", "Technical Screener", 
        "Sentiment Analysis", "Opportunity Ranking",
        "Comparison Mode", "Historical Analysis", "Risk Management", "Data Export"],
        key="view_mode_radio"
    )

    # 2. Load data and set up time horizon
    if data_exists:
        # Load from PostgreSQL (fast) or fallback to CSV
        df_metrics = load_data_from_postgres(current_index)
        
        if df_metrics is not None:
            valid_metrics = df_metrics[df_metrics['status'] == 'ok'].copy()         
            
            # Time horizon configuration
            horizon_map_full = {
                "1 Day": "pct_1d",
                "3 Days": "pct_3d", 
                "5 Days": "pct_5d",
                "1 Month": "pct_21d",
                "3 Months": "pct_63d",
                "1 Year": "pct_252d"
            }
            
            horizon_map = {label: col for label, col in horizon_map_full.items() 
                        if col in valid_metrics.columns}
            
            if horizon_map:
                default_horizon = "5 Days" if "5 Days" in horizon_map else list(horizon_map.keys())[0]
                
                horizon_label = st.sidebar.select_slider(
                    "Time Horizon",
                    options=list(horizon_map.keys()),
                    value=default_horizon,
                    key="horizon_slider"
                )
                horizon_col = horizon_map[horizon_label]
            else:
                st.error("No valid return columns found in data")
                horizon_col = None
                horizon_label = "N/A"
        else:
            data_exists = False
            valid_metrics = None
            horizon_col = None
            horizon_label = "N/A"
    else:
        valid_metrics = None
        horizon_col = None
        horizon_label = "N/A"

    # 3. Consecutive days slider 
    if 'consecutive_days' not in st.session_state:
        st.session_state.consecutive_days = 7
    
    consecutive_days = st.sidebar.slider(
        "Consecutive Days Lookback",
        min_value=2,
        max_value=126,
        value=st.session_state.consecutive_days,
        step=1,
        help="Number of consecutive days to analyze for rising/declining trends",
        key="consecutive_days_slider"
    )
    
    # Update session state when slider changes
    st.session_state.consecutive_days = consecutive_days

    # Add preset buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("3d", key="preset_3d"):
            st.session_state.consecutive_days = 3
            st.rerun()
    with col2:
        if st.button("1w", key="preset_1w"):
            st.session_state.consecutive_days = 7
            st.rerun()
    with col3:
        if st.button("1m", key="preset_1m"):
            st.session_state.consecutive_days = 21
            st.rerun()

    st.sidebar.info(f"Current: {consecutive_days} days ({consecutive_days/5:.1f} weeks)")

    # Add warning for very long periods
    if consecutive_days > 63:
        st.sidebar.warning("⚠️ Long periods may have fewer matching stocks")

    # Load historical data if data exists
    if data_exists:
        @cache_data(ttl=3600)  # Cache for 1 hour
        def load_historical_data(index_key):
            hist_dir = os.path.join(DATA_DIR, index_key, "history")
            hist = {}
            
            if os.path.exists(hist_dir):
                for file in os.listdir(hist_dir):
                    if file.endswith('.csv'):
                        ticker = file.replace('.csv', '')
                        try:
                            df = pd.read_csv(os.path.join(hist_dir, file), 
                                        parse_dates=True, index_col=0)
                            if not df.empty:
                                hist[ticker] = df
                        except Exception:
                            continue
            return hist
        
        # Load historical data
        hist = load_historical_data(current_index)
        
        # Show data freshness info
        last_modified = os.path.getmtime(os.path.join(index_data_dir, "latest_metrics.csv"))
        last_update = datetime.fromtimestamp(last_modified)
        last_update_est = last_update - timedelta(hours=5)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Data last updated: {last_update_est.strftime('%Y-%m-%d at %I:%M %p EST')}")
        with col2:
            st.caption("Updates: 9AM & 5PM EST daily")
    else:
        hist = {}
        st.error(f"No data available for {index_selection}. Click 'Fetch All Data' below.")

    # Show color legend
    show_color_legend()

    if not data_exists:
        st.sidebar.warning("No data found. Run initial data fetch first.")
        if st.sidebar.button("📥 Fetch All Data", key="fetch_all_data_btn"):
            # Create progress container
            progress_container = st.container()
            
            with progress_container:
                st.info("Starting data fetch...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Stage 1: Fetch tickers
                status_text.text("📋 Fetching ticker list...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                # Stage 2: Download historical data
                status_text.text("📊 Downloading historical data (this may take 3-5 minutes)...")
                progress_bar.progress(30)
                
                # Run the actual fetch
                try:
                    run_cli(consecutive_days, current_index)
                    
                    # Stage 3: Processing
                    status_text.text("⚙️ Processing metrics...")
                    progress_bar.progress(80)
                    time.sleep(0.5)
                    
                    # Stage 4: Complete
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    
                    st.success("Data fetched successfully!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Fetch failed: {e}")
                    status_text.text("Failed - check logs")
                    
    else:
        # Show last update time
        if os.path.exists(os.path.join(index_data_dir, "latest_metrics.csv")):
            last_modified = os.path.getmtime(os.path.join(index_data_dir, "latest_metrics.csv"))
            last_update = datetime.fromtimestamp(last_modified)
            hours_since = (datetime.now() - last_update).total_seconds() / 3600
            
            if hours_since < 1:
                st.sidebar.info(f"✅ Data current ({int(hours_since * 60)} min ago)")
            elif hours_since < 6:
                st.sidebar.info(f"📊 Data updated {hours_since:.1f}h ago")
            else:
                st.sidebar.warning(f"⚠️ Data is {hours_since:.1f}h old - consider updating")
        
        if st.sidebar.button("🔄 Update Data", key="update_data_btn"):
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"🔄 Updating {index_selection} data...")
                progress_bar.progress(20)
                
                try:
                    run_cli(consecutive_days, current_index)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Update complete!")
                    
                    # Invalidate cache
                    invalidate_cache(current_index)
                    
                    st.success("Data updated!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Update failed: {e}")
                    status_text.text("Failed - check logs")

    if data_exists:
        # Dynamic column names based on consecutive_days
        rising_col = f'rising_{consecutive_days}day'
        declining_col = f'declining_{consecutive_days}day'
        
        # Check if columns exist, if not use 3-day as fallback
        if rising_col not in df_metrics.columns:
            rising_col = 'rising_3day'
            declining_col = 'declining_3day'
            st.warning(f"Data not available for {consecutive_days}-day analysis. Using 3-day data. Please update data to use custom period.")

        # Check if we have valid horizon column
        if not horizon_col or horizon_col not in valid_metrics.columns:
            st.error("No valid return columns found in data.")
            if st.button("🔄 Regenerate Data", key="regenerate_data_btn"):
                with st.spinner("Regenerating data with all metrics..."):
                    run_cli()
                st.success("Data regenerated!")
                st.rerun()
            return

    # Database status indicator
    if pg_manager:
        with st.sidebar.expander("📊 Database Info"):
            stats = pg_manager.get_stats()
            
            if stats:
                st.metric("Total Stocks", f"{stats['total_stocks']:,}")
                
                if stats.get('last_update'):
                    last_update = stats['last_update']
                    st.caption(f"Last updated: {last_update}")
                
                if stats.get('stocks_by_index'):
                    st.caption("**Stocks by Index:**")
                    for item in stats['stocks_by_index']:
                        st.caption(f"• {item['index_name']}: {item['count']}")
    else:
        with st.sidebar.expander("⚠️ Database Status"):
            st.warning("No PostgreSQL connection")
            st.info("Using CSV fallback mode")
            
            if not os.environ.get('DATABASE_URL'):
                st.markdown("""
                **Setup PostgreSQL:**
                1. Get free database from:
                   - [Supabase](https://supabase.com) or
                   - [Neon](https://neon.tech)
                2. Add to Streamlit secrets:
                ```
                   DATABASE_URL = "postgresql://..."
                ```
                3. Restart app
                """)

    # ---------- Dashboard ----------
    if view_mode == "Dashboard":
        st.subheader("📊 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Safe column access with fallback
        if rising_col not in valid_metrics.columns:
            available_rising_cols = [col for col in valid_metrics.columns if col.startswith('rising_') and col.endswith('day')]
            if available_rising_cols:
                rising_col = available_rising_cols[0]
                declining_col = rising_col.replace('rising_', 'declining_')
                fallback_days = rising_col.replace('rising_', '').replace('day', '')
                st.warning(f"⚠️ {consecutive_days}-day data not available. Showing {fallback_days}-day data. Click 'Update Data' to generate {consecutive_days}-day analysis.")
            else:
                st.error("❌ No consecutive day trend data found. Please update the data.")
                rising_count = 0
                declining_count = 0
        
        # Safe column access
        if rising_col in valid_metrics.columns and declining_col in valid_metrics.columns:
            rising_count = len(valid_metrics[valid_metrics[rising_col] == True])
            declining_count = len(valid_metrics[valid_metrics[declining_col] == True])
        else:
            rising_count = 0
            declining_count = 0
        
        with col1:
            rising_pct = (rising_count/len(valid_metrics)*100)
            create_colored_metric_card(
                f"Rising ({consecutive_days}d)", 
                rising_pct, 
                'return',
                lambda x: f"{rising_count} ({x:.1f}%)"
            )
        
        with col2:
            declining_pct = (declining_count/len(valid_metrics)*100)
            create_colored_metric_card(
                f"Declining ({consecutive_days}d)", 
                -declining_pct,  # Negative to show as red
                'return',
                lambda x: f"{declining_count} ({abs(x):.1f}%)"
            )

        with col3:
            avg_volatility = valid_metrics['ann_vol_pct'].mean() if 'ann_vol_pct' in valid_metrics.columns else np.nan
            if not pd.isna(avg_volatility):
                create_colored_metric_card("Avg Volatility", avg_volatility, 'volatility', format_percentage)

        with col4:
            if 'rsi' in valid_metrics.columns:
                avg_rsi = valid_metrics['rsi'].mean()
                create_colored_metric_card("Avg RSI", avg_rsi, 'rsi', lambda x: f"{x:.1f}")

        # Enhanced top movers with color gradients
        st.subheader(f"🚀 Top Movers ({horizon_label})")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Top Gainers")
            top_gainers = valid_metrics.nlargest(10, horizon_col)
            fig_gainers = create_enhanced_bar_chart(
                top_gainers, 'ticker', horizon_col, 
                f"Top Gainers ({horizon_label})", 'return'
            )
            st.plotly_chart(fig_gainers, use_container_width=True)

        with col2:
            st.markdown("### 📉 Top Decliners")
            top_losers = valid_metrics.nsmallest(10, horizon_col)
            fig_losers = create_enhanced_bar_chart(
                top_losers, 'ticker', horizon_col, 
                f"Top Decliners ({horizon_label})", 'return'
            )
            st.plotly_chart(fig_losers, use_container_width=True)

        # Add sector heatmap
        st.subheader("🏢 Sector Performance Heatmap")
        sector_perf = valid_metrics.groupby('sector')[horizon_col].mean().sort_values(ascending=False)
        fig_heatmap = create_sector_heatmap(sector_perf, horizon_col, f"Sector Performance ({horizon_label})")
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Enhanced data table with colors
        st.subheader("📋 All Stocks")

        # Build display columns safely as some datasets may not have pe_ratio
        display_cols = ['ticker', 'company_name', 'sector', 'last_close', horizon_col, 'ann_vol_pct']
        # Include P/E if available or add a placeholder column so table layout is consistent
        if 'pe_ratio' in valid_metrics.columns:
            display_cols.insert(4, 'pe_ratio')
        else:
            valid_metrics = valid_metrics.copy()
            valid_metrics['pe_ratio'] = np.nan
            display_cols.insert(4, 'pe_ratio')

        if 'rsi' in valid_metrics.columns:
            display_cols.append('rsi')

        display_df = valid_metrics[display_cols].copy()
        styled_df = format_and_style_dataframe(display_df)
        st.dataframe(styled_df, use_container_width=True, height=400)

    # ---------- Sector Analysis ----------
    elif view_mode == "Sector Analysis":
        st.subheader(f"🏢 {index_selection} Sector Performance Analysis")

        sector_perf = valid_metrics.groupby('sector').agg({
            horizon_col: ['mean', 'median', 'std'],
            'ticker': 'count',
            'ann_vol_pct': 'mean'
        }).round(2)

        sector_perf.columns = ['Mean Return', 'Median Return', 'Std Dev', 'Count', 'Avg Volatility']
        sector_perf = sector_perf.sort_values('Mean Return', ascending=False)

        # Enhanced Bar chart with colors
        fig_sector = create_enhanced_bar_chart(
            sector_perf.reset_index(), 'sector', 'Mean Return',
            f"Average Sector Returns ({horizon_label})", 'return'
        )
        fig_sector.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_sector, use_container_width=True)

        styled_sector = format_and_style_dataframe(
            sector_perf,
            format_dict={'Count': '{:.0f}'},
            metrics_columns={
                'Mean Return': 'return',
                'Median Return': 'return', 
                'Std Dev': 'volatility',
                'Avg Volatility': 'volatility'
            }
        )
        st.dataframe(styled_sector, use_container_width=True)

    # ---------- Top Movers ----------
    elif view_mode == "Top Movers":
        st.subheader(f"🎯 {index_selection} Market Movers Analysis")
        
        tabs = st.tabs([f"Rising Stocks ({consecutive_days}d)", f"Declining Stocks ({consecutive_days}d)", "Most Volatile", "52-Week Highs/Lows"])
        
        with tabs[0]:
            if rising_col in valid_metrics.columns:
                rising = valid_metrics[valid_metrics[rising_col] == True]
            else:
                rising = pd.DataFrame()
            if not rising.empty:
                # Sort by appropriate percentage column
                sort_col = 'pct_5d' if consecutive_days <= 5 else 'pct_21d'
                if sort_col in rising.columns:
                    rising = rising.sort_values(sort_col, ascending=False)
            
            st.metric(f"Total Rising ({consecutive_days}-day consecutive)", len(rising))
            if not rising.empty:
                display_cols = ['ticker', 'company_name', 'sector', 'last_close', 'pe_ratio', 'pct_1d', 'pct_3d', 'pct_5d']
                if 'pct_21d' in rising.columns:
                    display_cols.append('pct_21d')
                # Filter to only existing columns
                display_cols = [col for col in display_cols if col in rising.columns]
                
                rising_display = rising[display_cols].head(20)
                styled_rising = format_and_style_dataframe(rising_display)
                st.dataframe(styled_rising, use_container_width=True)
        
        with tabs[1]:
            if declining_col in valid_metrics.columns:
                declining = valid_metrics[valid_metrics[declining_col] == True]
            else:
                declining = pd.DataFrame()
            if not declining.empty:
                sort_col = 'pct_5d' if consecutive_days <= 5 else 'pct_21d'
                if sort_col in declining.columns:
                    declining = declining.sort_values(sort_col)
            
            st.metric(f"Total Declining ({consecutive_days}-day consecutive)", len(declining))
            if not declining.empty:
                display_cols = ['ticker', 'company_name', 'sector', 'last_close', 'pe_ratio', 'pct_1d', 'pct_3d', 'pct_5d']
                if 'pct_21d' in declining.columns:
                    display_cols.append('pct_21d')
                # Filter to only existing columns
                display_cols = [col for col in display_cols if col in declining.columns]
                
                declining_display = declining[display_cols].head(20)
                styled_declining = format_and_style_dataframe(declining_display)
                st.dataframe(styled_declining, use_container_width=True)

        with tabs[2]:
            if 'ann_vol_pct' in valid_metrics.columns:
                # Build column list with company_name if available
                volatile_cols = ['ticker']
                if 'company_name' in valid_metrics.columns:
                    volatile_cols.append('company_name')
                volatile_cols.extend(['sector', 'last_close', 'pe_ratio', 'ann_vol_pct', horizon_col])
                
                most_volatile = valid_metrics.nlargest(20, 'ann_vol_pct')[volatile_cols].copy()
                
                # Display metric with company name if available
                top_ticker = most_volatile.iloc[0]['ticker']
                top_vol = most_volatile.iloc[0]['ann_vol_pct']
                if 'company_name' in most_volatile.columns:
                    top_company = most_volatile.iloc[0]['company_name']
                    st.metric("Highest Volatility Stock", f"{top_ticker} - {top_company} ({top_vol:.1f}%)")
                else:
                    st.metric("Highest Volatility Stock", f"{top_ticker} ({top_vol:.1f}%)")
                
                # Format the data
                styled_volatile = format_and_style_dataframe(most_volatile)
                st.dataframe(styled_volatile, use_container_width=True)
            else:
                st.info("Volatility data not available")

        with tabs[3]:
            if 'pct_from_52w_high' in valid_metrics.columns:
                near_highs = valid_metrics.nlargest(20, 'pct_from_52w_high')[['ticker', 'company_name', 'sector', 'pe_ratio', 'last_close', 'high_52w', 'pct_from_52w_high']].copy()
                near_lows = valid_metrics.nsmallest(20, 'pct_from_52w_low')[['ticker', 'company_name', 'sector', 'pe_ratio', 'last_close', 'low_52w', 'pct_from_52w_low']].copy()

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Near 52-Week Highs")
                    # Format the data
                    styled_highs = format_and_style_dataframe(near_highs)
                    st.dataframe(styled_highs, use_container_width=True)
                    
                with col2:
                    st.markdown("### Near 52-Week Lows")
                    # Format the data
                    styled_lows = format_and_style_dataframe(near_lows)
                    st.dataframe(styled_lows, use_container_width=True)
            else:
                st.info("52-week high/low data not available")

    # ---------- Technical Screener ----------
    elif view_mode == "Technical Screener":
        st.subheader(f"🔍 {index_selection} Technical Stock Screener")

        col1, col2, col3 = st.columns(3)

        with col1:
            min_return = st.number_input(
                f"Min {horizon_label} Return (%)",
                value=-100.0,
                max_value=100.0,
                step=1.0,
                key="ts_min_return"
            )
            max_return = st.number_input(
                f"Max {horizon_label} Return (%)",
                value=100.0,
                min_value=-100.0,
                step=1.0,
                key="ts_max_return"
            )

        with col2:
            min_volatility = st.number_input(
                "Min Volatility (%)",
                value=0.0,
                max_value=200.0,
                step=1.0,
                key="ts_min_vol"
            )
            max_volatility = st.number_input(
                "Max Volatility (%)",
                value=200.0,
                min_value=0.0,
                step=1.0,
                key="ts_max_vol"
            )

        with col3:
            min_price = st.number_input(
                "Min Price ($)",
                value=0.0,
                max_value=10000.0,
                step=1.0,
                key="ts_min_price"
            )
            max_price = st.number_input(
                "Max Price ($)",
                value=10000.0,
                min_value=0.0,
                step=1.0,
                key="ts_max_price"
            )

        # RSI filters
        col1, col2 = st.columns(2)
        with col1:
            if 'rsi' in valid_metrics.columns:
                min_rsi = st.number_input("Min RSI", value=0.0, max_value=100.0, step=1.0, key="ts_min_rsi")
            else:
                min_rsi = 0
        with col2:
            if 'rsi' in valid_metrics.columns:
                max_rsi = st.number_input("Max RSI", value=100.0, min_value=0.0, step=1.0, key="ts_max_rsi")
            else:
                max_rsi = 100
    
        # Add P/E ratio filters
        col1, col2 = st.columns(2)
        with col1:
            if 'pe_ratio' in valid_metrics.columns:
                min_pe = st.number_input("Min P/E Ratio", value=0.0, max_value=200.0, step=1.0, key="ts_min_pe")
            else:
                min_pe = 0
        with col2:
            if 'pe_ratio' in valid_metrics.columns:
                max_pe = st.number_input("Max P/E Ratio", value=200.0, min_value=0.0, step=1.0, key="ts_max_pe")
            else:
                max_pe = 200

        # Sector filter
        all_sectors = sorted(valid_metrics['sector'].unique())
        selected_sectors = st.multiselect(
            "Filter by Sectors",
            options=all_sectors,
            default=all_sectors,
            key="ts_sector_multiselect"
        )

        # Additional filters
        col1, col2 = st.columns(2)
        with col1:
            only_rising = st.checkbox(f"Only Rising ({consecutive_days}-day)", False, key="ts_only_rising")
        with col2:
            only_declining = st.checkbox(f"Only Declining ({consecutive_days}-day)", False, key="ts_only_declining")

        # Apply filters
        screened_df = valid_metrics[
            (valid_metrics[horizon_col] >= min_return) &
            (valid_metrics[horizon_col] <= max_return) &
            (valid_metrics.get('ann_vol_pct', 0) >= min_volatility) &
            (valid_metrics.get('ann_vol_pct', 0) <= max_volatility) &
            (valid_metrics['sector'].isin(selected_sectors)) &
            (valid_metrics['last_close'] >= min_price) &
            (valid_metrics['last_close'] <= max_price)
        ].copy()
    
        if 'pe_ratio' in valid_metrics.columns:
            screened_df = screened_df[
                (screened_df['pe_ratio'].isna()) | 
                ((screened_df['pe_ratio'] >= min_pe) & (screened_df['pe_ratio'] <= max_pe))
            ]

        if 'rsi' in valid_metrics.columns:
            screened_df = screened_df[
                (screened_df['rsi'] >= min_rsi) &
                (screened_df['rsi'] <= max_rsi)
            ]

        # Apply consecutive filters
        if only_rising and rising_col in screened_df.columns:
            screened_df = screened_df[screened_df[rising_col] == True]
        if only_declining and declining_col in screened_df.columns:
            screened_df = screened_df[screened_df[declining_col] == True]

        # Sort options
        sort_columns = [horizon_col, 'ann_vol_pct', 'last_close']
        if 'rsi' in screened_df.columns:
            sort_columns.append('rsi')

        sort_by = st.selectbox("Sort By", options=sort_columns, key="ts_sort_by")
        sort_order = st.radio("Sort Order", ["Descending", "Ascending"], horizontal=True, key="ts_sort_order")

        screened_df = screened_df.sort_values(sort_by, ascending=(sort_order == "Ascending"))

        # Results
        st.subheader(f"📋 Screener Results ({len(screened_df)} stocks)")

        if not screened_df.empty:
            display_cols = ['ticker', 'company_name', 'sector', 'last_close', 'pe_ratio', horizon_col, 'ann_vol_pct']
            if 'rsi' in screened_df.columns:
                display_cols.append('rsi')

            # Format the screener results
            screener_results = screened_df[display_cols]
            styled_screener = format_and_style_dataframe(screener_results)
            st.dataframe(styled_screener, use_container_width=True, height=400)

            # Enhanced Visualization with color coding
            if len(screened_df) > 1 and 'ann_vol_pct' in screened_df.columns:
                # Build hover data
                hover_data_list = ['ticker', 'sector', 'last_close']
                if 'company_name' in screened_df.columns:
                    hover_data_list.insert(0, 'company_name')
                
                fig = px.scatter(
                    screened_df,
                    x='ann_vol_pct',
                    y=horizon_col,
                    color=horizon_col,
                    color_continuous_scale=create_gradient_colorscale(screened_df[horizon_col], 'return'),
                    size='last_close',
                    hover_data=hover_data_list,
                    title=f"{horizon_label} Return vs Volatility",
                    labels={
                        'ann_vol_pct': 'Annual Volatility (%)',
                        horizon_col: f'{horizon_label} Return (%)',
                        'last_close': 'Price ($)',
                        'company_name': 'Company'
                    }
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)

            # Per-ticker plotting: choose one from screened results
            tickers_list = screened_df['ticker'].tolist()
            if tickers_list:
                # Create display options with company names
                if 'company_name' in screened_df.columns:
                    ticker_options = {
                        f"{row['company_name']} ({row['ticker']})": row['ticker'] 
                        for _, row in screened_df[['ticker', 'company_name']].iterrows()
                    }
                    chosen_display = st.selectbox("Select company to chart", options=list(ticker_options.keys()), key="ts_ticker_chart")
                    chosen_ticker = ticker_options[chosen_display]
                else:
                    chosen_ticker = st.selectbox("Select ticker to chart", options=tickers_list, key="ts_ticker_chart")
                
                # Load per-ticker CSV from data/<index>/history/<ticker>.csv and plot
                ticker_csv_path = os.path.join(index_data_dir, "history", f"{chosen_ticker}.csv")
                plot_ticker_price_rsi(ticker_csv_path, chosen_ticker)
        else:
            st.info("No stocks match the selected criteria")

    # ---------- Sentiment Analysis ----------
    elif view_mode == "Sentiment Analysis":
        st.subheader("📰 Market Sentiment Analysis")
        
        # Strategy explanation
        with st.expander("ℹ️ How Sentiment Analysis Works"):
            if NEWS_API_KEY:
                st.markdown("""
                **🎯 Mixed Source Strategy** (NewsAPI key detected)
                
                You have NewsAPI configured! This gives you the best of both worlds:
                
                1. **NewsAPI (Premium)**: For top movers
                - Comprehensive article coverage (50+ articles per stock)
                - Better quality and more sources
                - Limited to 75-100 stocks/day (free tier)
                - Automatically targets the most volatile stocks
                
                2. **Google News RSS (Free)**: For remaining stocks
                - Free and unlimited
                - Good general sentiment coverage
                - 10-20 articles per stock
                
                **Result:** Premium coverage where it matters most, free coverage for everything else!
                """)
            else:
                st.markdown("""
                **📰 Google News RSS Strategy** (No API key required)
                
                Using 100% free sentiment analysis:
                
                - **No API key needed** - works out of the box
                - **No rate limits** - analyze as many stocks as you want
                - **Reliable data** - Google News aggregates from major sources
                - **10-20 articles per stock** - sufficient for sentiment analysis
                
                ---
                
                **💡 Want premium coverage?**
                
                Get a free NewsAPI key for enhanced analysis of top movers:
                1. Sign up at [newsapi.org](https://newsapi.org) (free tier available)
                2. Add `NEWS_API_KEY=your_key_here` to your `.env` file
                3. Restart the app
                
                With NewsAPI, you'll get 50+ articles for the most important stocks!
                """)
        
        # Configuration section
        st.markdown("### ⚙️ Configuration")
        
        if NEWS_API_KEY:
            # Show NewsAPI controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_newsapi = st.slider(
                    "NewsAPI stocks (top movers)", 
                    10, 100, 75,
                    help="How many top movers to analyze with NewsAPI"
                )
                if max_newsapi > 100:
                    st.warning("⚠️ Free tier limit is 100 requests/day")
            
            with col2:
                min_articles = st.number_input(
                    "Min articles (NewsAPI only)", 
                    0, 20, 3,
                    help="Filter out NewsAPI results with too few articles"
                )
            
            with col3:
                st.success("✅ NewsAPI Active")
                st.caption(f"Premium analysis enabled")
                if horizon_col:
                    st.info(f"Top movers: **{horizon_label}** returns")
        else:
            # Show Google News only info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("📰 **Free Mode:** Using Google News RSS for all stocks")
                st.caption("No API key required • No rate limits • Unlimited stocks")
            
            with col2:
                if st.button("🔑 Setup NewsAPI", key="setup_newsapi"):
                    st.markdown("""
                    **Quick Setup:**
                    1. Get key: [newsapi.org](https://newsapi.org)
                    2. Add to `.env`: `NEWS_API_KEY=your_key`
                    3. Restart app
                    """)
            
            # Set defaults for Google News only mode
            max_newsapi = 0
            min_articles = 0
        
        # Analyze button
        st.markdown("### 🔍 Run Analysis")
        
        if st.button("📊 Analyze Market Sentiment", key="analyze_sentiment_btn", type="primary"):
            with st.spinner("Analyzing market sentiment... This may take 5-10 minutes."):
                try:
                    sentiment_df = analyze_sentiment_mixed(
                        valid_metrics, 
                        horizon_col,
                        max_newsapi_requests=max_newsapi
                    )
                    
                    # Filter quality only if using NewsAPI
                    if NEWS_API_KEY:
                        sentiment_df_filtered = filter_high_quality_sentiment(
                            sentiment_df, 
                            min_articles=min_articles
                        )
                    else:
                        # Keep all Google News results
                        sentiment_df_filtered = sentiment_df
                    
                    st.session_state.sentiment_df = sentiment_df_filtered
                    st.session_state.sentiment_df_raw = sentiment_df
                    
                    # Success message based on mode
                    strategy = get_sentiment_strategy_summary(sentiment_df)
                    if strategy:
                        if NEWS_API_KEY:
                            st.success(f"""
                            ✅ **Sentiment analysis complete!**
                            
                            - 🎯 NewsAPI: {strategy['newsapi_count']} stocks (premium)
                            - 📰 Google News: {strategy['google_news_count']} stocks (free)
                            - 📊 Total: {strategy['total']} stocks analyzed
                            - ⭐ High quality: {len(sentiment_df_filtered)} stocks
                            """)
                        else:
                            st.success(f"""
                            ✅ **Sentiment analysis complete!**
                            
                            - 📰 Google News: {strategy['google_news_count']} stocks
                            - 📊 Total: {strategy['total']} stocks analyzed
                            - 💯 100% free coverage (no API key required)
                            """)
                    
                except Exception as e:
                    st.error(f"❌ Error during sentiment analysis: {e}")
                    with st.expander("🐛 Debug Information"):
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display results
        if 'sentiment_df' in st.session_state and not st.session_state.sentiment_df.empty:
            sentiment_df = st.session_state.sentiment_df
            sentiment_df_raw = st.session_state.get('sentiment_df_raw', sentiment_df)
            
            merged_data = merge_sentiment_with_metrics(valid_metrics, sentiment_df)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_sentiment = sentiment_df['sentiment_score'].mean()
                color = get_sentiment_color(avg_sentiment)
                st.markdown(f"""
                    <div style="background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;">
                        <h4>Market Sentiment</h4>
                        <h2>{avg_sentiment:.3f}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                positive_pct = (sentiment_df['sentiment_label'] == 'Positive').sum() / len(sentiment_df) * 100
                st.metric("📈 Positive Stocks", f"{positive_pct:.1f}%")
            
            with col3:
                negative_pct = (sentiment_df['sentiment_label'] == 'Negative').sum() / len(sentiment_df) * 100
                st.metric("📉 Negative Stocks", f"{negative_pct:.1f}%")
            
            with col4:
                total_articles = sentiment_df['article_count'].sum()
                st.metric("📰 Total Articles", f"{total_articles:,}")
            
            # Tabs for results
            tabs = st.tabs([
                "Top Sentiment", 
                "Sector Sentiment", 
                "Sentiment vs Performance", 
                "Data Sources",
                "Detailed Table"
            ])
            
            with tabs[0]:
                st.subheader("🏆 Top 20 Stocks by Sentiment")
                
                top_20 = sentiment_df.nlargest(20, 'sentiment_score')
                
                fig = create_sentiment_chart(top_20, 20)
                st.plotly_chart(fig, use_container_width=True)
                
                # Source breakdown
                if NEWS_API_KEY:
                    col1, col2 = st.columns(2)
                    with col1:
                        newsapi_in_top = (top_20['source'] == 'NewsAPI').sum()
                        st.metric("🎯 NewsAPI Coverage", newsapi_in_top)
                    with col2:
                        google_news_in_top = (top_20['source'] == 'google_news').sum()
                        st.metric("📰 Google News Coverage", google_news_in_top)
                else:
                    st.info("📰 All sentiment data from Google News RSS (free)")
            
            with tabs[1]:
                sector_sentiment = get_sector_sentiment_summary(sentiment_df, valid_metrics)
                
                st.subheader("🏢 Sentiment by Sector")
                
                # Try to use background_gradient with fall back to simple display
                try:
                    styled_sector = sector_sentiment.style.background_gradient(
                        subset=['Avg Sentiment'],
                        cmap='RdYlGn',
                        vmin=-1,
                        vmax=1
                    )
                    st.dataframe(styled_sector, use_container_width=True)
                except ImportError:
                    # Fallback: Use custom color function instead
                    def color_sentiment(val):
                        """Color cells based on sentiment value"""
                        if pd.isna(val):
                            return ''
                        color = get_sentiment_color(val)
                        return f'background-color: {color}'
                    
                    styled_sector = sector_sentiment.style.applymap(
                        color_sentiment, 
                        subset=['Avg Sentiment']
                    )
                    st.dataframe(styled_sector, use_container_width=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=sector_sentiment.index,
                        y=sector_sentiment['Avg Sentiment'],
                        marker_color=[get_sentiment_color(s) for s in sector_sentiment['Avg Sentiment']],
                        text=sector_sentiment['Avg Sentiment'].apply(lambda x: f"{x:.3f}"),
                        textposition='outside'
                    )
                ])
                fig.update_layout(
                    title="Average Sentiment by Sector",
                    xaxis_title="Sector",
                    yaxis_title="Average Sentiment",
                    xaxis_tickangle=-45,
                    height=400,
                    yaxis=dict(range=[-1, 1])
                )
                fig.add_hline(y=0, line_dash="dash", opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[2]:
                if horizon_col in merged_data.columns:
                    st.subheader(f"📊 Sentiment vs {horizon_label} Performance")
                    
                    plot_data = merged_data.dropna(subset=['sentiment_score', horizon_col])
                    
                    fig = px.scatter(
                        plot_data,
                        x='sentiment_score',
                        y=horizon_col,
                        color='source',
                        size='article_count',
                        hover_data=['ticker', 'company_name', 'sector'],
                        title=f"Sentiment vs Performance Correlation",
                        labels={
                            'sentiment_score': 'Sentiment Score',
                            horizon_col: f'{horizon_label} Return (%)',
                            'source': 'Data Source'
                        },
                        color_discrete_map={
                            'NewsAPI': '#1f77b4', 
                            'google_news': '#ff7f0e'
                        }
                    )
                    fig.add_hline(y=0, line_dash="dash", opacity=0.3)
                    fig.add_vline(x=0, line_dash="dash", opacity=0.3)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        corr_all = plot_data[['sentiment_score', horizon_col]].corr().iloc[0, 1]
                        st.metric("📊 Overall Correlation", f"{corr_all:.3f}")
                    
                    with col2:
                        if NEWS_API_KEY:
                            newsapi_data = plot_data[plot_data['source'] == 'NewsAPI']
                            if len(newsapi_data) > 10:
                                corr_newsapi = newsapi_data[['sentiment_score', horizon_col]].corr().iloc[0, 1]
                                st.metric("🎯 NewsAPI Correlation", f"{corr_newsapi:.3f}")
                            else:
                                st.info("Not enough NewsAPI data")
                        else:
                            google_data = plot_data[plot_data['source'] == 'google_news']
                            if len(google_data) > 10:
                                corr_google = google_data[['sentiment_score', horizon_col]].corr().iloc[0, 1]
                                st.metric("📰 Google News Correlation", f"{corr_google:.3f}")
            
            with tabs[3]:
                st.subheader("📊 Data Source Analysis")
                
                if NEWS_API_KEY:
                    fig = create_sentiment_comparison_chart(sentiment_df_raw)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 NewsAPI Stocks (Top Movers)**")
                        newsapi_stocks = sentiment_df_raw[sentiment_df_raw['source'] == 'NewsAPI']
                        if not newsapi_stocks.empty:
                            st.dataframe(
                                newsapi_stocks[['ticker', 'sentiment_score', 'article_count']].head(10),
                                use_container_width=True
                            )
                        else:
                            st.info("No NewsAPI data")
                    
                    with col2:
                        st.markdown("**📰 Google News Stocks**")
                        google_stocks = sentiment_df_raw[sentiment_df_raw['source'] == 'google_news']
                        if not google_stocks.empty:
                            st.dataframe(
                                google_stocks[['ticker', 'sentiment_score', 'article_count']].head(10),
                                use_container_width=True
                            )
                        else:
                            st.info("No Google News data")
                    
                    st.markdown("**📈 Quality Metrics by Source**")
                    quality_comparison = sentiment_df_raw.groupby('source').agg({
                        'article_count': ['mean', 'median', 'max'],
                        'sentiment_score': ['mean', 'std'],
                        'ticker': 'count'
                    }).round(3)
                    st.dataframe(quality_comparison, use_container_width=True)
                else:
                    st.info("📰 **All sentiment data from Google News RSS**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_articles = sentiment_df_raw['article_count'].mean()
                        st.metric("Avg Articles/Stock", f"{avg_articles:.1f}")
                    
                    with col2:
                        total_stocks = len(sentiment_df_raw)
                        st.metric("Total Stocks", total_stocks)
                    
                    with col3:
                        with_articles = (sentiment_df_raw['article_count'] > 0).sum()
                        coverage = with_articles / total_stocks * 100
                        st.metric("Coverage", f"{coverage:.1f}%")
                    
                    st.markdown("**📊 Article Distribution**")
                    fig = px.histogram(
                        sentiment_df_raw,
                        x='article_count',
                        nbins=20,
                        title="Distribution of Articles per Stock",
                        labels={'article_count': 'Articles per Stock'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tabs[4]:
                st.subheader("📋 All Sentiment Data")
                
                display_cols = ['ticker', 'company_name', 'sentiment_score', 'sentiment_label', 
                            'article_count', 'source']
                
                if horizon_col in merged_data.columns:
                    display_cols.insert(4, horizon_col)
                
                if 'sector' in merged_data.columns:
                    display_cols.insert(3, 'sector')
                
                if 'pe_ratio' in merged_data.columns:
                    display_cols.insert(3, 'pe_ratio')
                
                display_data = merged_data[display_cols].sort_values('sentiment_score', ascending=False)
                
                st.dataframe(
                    display_data.style.background_gradient(
                        subset=['sentiment_score'],
                        cmap='RdYlGn',
                        vmin=-1,
                        vmax=1
                    ),
                    use_container_width=True,
                    height=400
                )
                
                csv = display_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sentiment Data",
                    data=csv,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("👆 Click 'Analyze Market Sentiment' to start analysis")
            
            if NEWS_API_KEY:
                st.markdown("""
                **📊 What you'll get:**
                - Comprehensive sentiment for top movers (NewsAPI)
                - Free sentiment for remaining stocks (Google News)
                - Sector sentiment breakdown
                - Sentiment vs performance correlation
                - Exportable data
                """)
            else:
                st.markdown("""
                **📊 What you'll get:**
                - Free sentiment analysis for all stocks
                - Google News RSS coverage (10-20 articles/stock)
                - Sector sentiment breakdown
                - Sentiment vs performance correlation
                - Exportable data
                
                **✨ 100% free - no API key required!**
                """)

    # ---------- Opportunity Ranking Recommendation ----------
    elif view_mode == "Opportunity Ranking":
        st.subheader("🎯 Opportunity Ranking (Enhanced)")
        
        # Option to fetch enhanced data
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("""
            **Enhanced Mode includes:**
            - 📊 Forward-looking fundamentals (earnings, growth, margins)
            - 🏢 Institutional & insider activity
            - 📈 Sector-relative performance
            - ⚡ Catalyst detection
            - 💰 Risk management (position sizing, stop loss)
            - ⏰ Time horizon optimization
            """)
        
        with col2:
            use_enhanced = st.checkbox(
                "Use Enhanced Data",
                value=True,
                help="Fetch comprehensive data (takes 3-5 min first time)"
            )
        
        if use_enhanced:
            # Fetch enhanced data
            if st.button("🔄 Fetch/Refresh Enhanced Data"):
                enhanced_metrics = fetch_and_enhance_metrics(valid_metrics, force_refresh=True)
                st.session_state.enhanced_metrics = enhanced_metrics
            
            # Check if enhanced data is available
            if 'enhanced_metrics' not in st.session_state:
                st.warning("⚠️ Enhanced data not loaded. Click 'Fetch Enhanced Data' button above.")
                enhanced_metrics = valid_metrics  # Fallback to base
            else:
                enhanced_metrics = st.session_state.enhanced_metrics
        else:
            enhanced_metrics = valid_metrics
        
        # rendering function
        render_opportunity_recommendations_mode(enhanced_metrics, hist)
    
    # ---------- Data Export ----------
    elif view_mode == "Data Export":
        st.subheader("📥 Data Export")

        st.markdown("### Available Data Files")

        files = {
            f"{index_selection} Metrics": "latest_metrics.csv",
            f"{index_selection} Constituents": f"{current_index}_constituents_snapshot.csv",
            f"Rising Stocks ({consecutive_days}-day)": f"rising_{consecutive_days}day.csv",
            f"Declining Stocks ({consecutive_days}-day)": f"declining_{consecutive_days}day.csv"
        }

        for name, filename in files.items():
            filepath = os.path.join(index_data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = f.read()
                st.download_button(
                    label=f"📄 Download {name}",
                    data=data,
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}"
                )

        st.markdown("### Custom Export")

        # Custom filtering for export
        export_sectors = st.multiselect(
            "Select Sectors for Export",
            options=sorted(valid_metrics['sector'].unique()),
            default=sorted(valid_metrics['sector'].unique()),
            key="export_sectors"
        )

        # Set default columns including company_name if available
        default_export_cols = ['ticker']
        if 'company_name' in valid_metrics.columns:
            default_export_cols.append('company_name')
        default_export_cols.extend(['sector', 'last_close', 'pe_ratio', 'pct_1d', 'pct_5d', 'ann_vol_pct'])
        
        export_cols = st.multiselect(
            "Select Columns for Export",
            options=list(valid_metrics.columns),
            default=default_export_cols,
            key="export_cols"
        )

        if export_sectors and export_cols:
            export_df = valid_metrics[valid_metrics['sector'].isin(export_sectors)][export_cols]

            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Custom Export",
                data=csv,
                file_name=f"sp500_custom_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_custom_export"
            )

            styled_preview = format_and_style_dataframe(export_df.head(10))
            st.dataframe(styled_preview, use_container_width=True)
    # ---------- Comparison Mode ----------
    elif view_mode == "Comparison Mode":
        render_comparison_mode(valid_metrics, hist, horizon_col, horizon_label, current_index)

    # ---------- Historical Analysis ---------- 
    elif view_mode == "Historical Analysis":
        render_historical_analysis_mode(valid_metrics, hist)

    # ---------- Risk Management ----------
    elif view_mode == "Risk Management":
        render_risk_management_mode(valid_metrics, hist)

    # Footer
    st.divider()
    st.caption("""
    Created by Ehiremen Nathaniel Omoarebun
             
    **Disclaimer:** This tool is for informational purposes only and should not be considered as financial advice.
    Data provided by Yahoo Finance and may be delayed. Always do your own research before making investment decisions.
    """)

# ---------------------------
# Main Entry Point
# ---------------------------
def main():
    """Main entry point for the application"""
    
    # Simple detection: if no arguments provided and streamlit is available, run web mode
    if len(sys.argv) == 1 and STREAMLIT_AVAILABLE:
        run_streamlit()
        return
    
    # Parse command line arguments for CLI mode
    parser = argparse.ArgumentParser(
        description="Market Tracker - CLI and Web Interface"
    )

    parser.add_argument(
        '--mode',
        choices=['cli', 'web'],
        default='cli',
        help='Run mode: cli for command line, web for Streamlit interface'
    )

    parser.add_argument(
        '--consecutive-days',
        type=int,
        default=7,
        help='Number of consecutive days for trend analysis (default: 7)'
    )
    
    parser.add_argument(
            '--index',
            choices=['SP500', 'SP400', 'SP600', 'NASDAQ100', 'DOW30', 'COMBINED', 'ALL'],
            default='SP500',
            help='Market index to track: SP500, SP400 (MidCap), SP600 (SmallCap), NASDAQ100, DOW30, COMBINED, or ALL to run all indices (default: SP500)'
        )

    args = parser.parse_args()

    # Run appropriate mode
    if args.mode == 'cli':
        run_cli(args.consecutive_days, args.index)
    else:
        run_streamlit()

if __name__ == "__main__":
    main()