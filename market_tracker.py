import os
import sys
import time
import math
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Try importing Streamlit and Plotly for web interface
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Note: Streamlit/Plotly not installed. Web interface unavailable.")

# ---------------------------
# Configuration
# ---------------------------
UNIVERSE = "SP500"
DATA_DIR = "data"
BATCH_SIZE = 50
MIN_CONSECUTIVE = 3
DOWNLOAD_PERIOD = "max"
VERBOSE = True
MIN_MARKET_CAP = 0

# ---------------------------
# Shared Helper Functions
# ---------------------------
def ensure_dir(d):
    """Create directory if it doesn't exist"""
    os.makedirs(d, exist_ok=True)

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

        # Extract sector and industry info
        sectors = dict(zip(df['Symbol'], df['GICS Sector']))
        industries = dict(zip(df['Symbol'], df['GICS Sub-Industry']))

        return tickers, df, sectors, industries

    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return [], pd.DataFrame(), {}, {}

def batch(iterable, n=1):
    """Batch an iterable into chunks of size n"""
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:i+n]

def download_history(tickers, period="max", interval="1d", threads=True, progress=True):
    """Download historical data for multiple tickers"""
    if len(tickers) == 0:
        return {}

    out = {}

    for chunk in batch(tickers, BATCH_SIZE):
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
                    print(f"Download error, retrying in {wait}s: {e}")
                time.sleep(wait)
        else:
            print("Failed to download chunk:", chunk)
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

    return out

# ---------------------------
# Metrics Calculation
# ---------------------------
def compute_metrics_for_ticker(df):
    """Calculate comprehensive metrics for a single ticker"""
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

    # 3-day consecutive rising/decline check
    if len(closes) >= MIN_CONSECUTIVE:
        last_n = closes.iloc[-MIN_CONSECUTIVE:]
        increasing = all(last_n.values[i] > last_n.values[i-1] for i in range(1, len(last_n)))
        decreasing = all(last_n.values[i] < last_n.values[i-1] for i in range(1, len(last_n)))
    else:
        increasing = False
        decreasing = False

    metrics['rising_3day'] = bool(increasing)
    metrics['declining_3day'] = bool(decreasing)

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
        metrics['52w_high'] = float(closes.iloc[-252:].max())
        metrics['52w_low'] = float(closes.iloc[-252:].min())
        metrics['pct_from_52w_high'] = ((last - metrics['52w_high']) / metrics['52w_high']) * 100
        metrics['pct_from_52w_low'] = ((last - metrics['52w_low']) / metrics['52w_low']) * 100

    # Volume metrics if available
    if 'Volume' in df.columns:
        volumes = df['Volume'].dropna()
        if len(volumes) >= 20:
            metrics['avg_volume_20d'] = float(volumes.iloc[-20:].mean())
            metrics['volume_vs_avg'] = (float(volumes.iloc[-1]) / metrics['avg_volume_20d'] - 1) * 100 if metrics['avg_volume_20d'] > 0 else 0

    metrics['data_points'] = len(closes)

    return metrics

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
    df['RSI'] = df['Close'].diff().apply(lambda x: x if x>0 else 0).rolling(window=14).mean() / \
                (-df['Close'].diff().apply(lambda x: x if x<0 else 0)).rolling(window=14).mean()
    # The above is not the classic RS->RSI formula; easier to compute RSI using helper:
    try:
        df['RSI'] = df['Close'].rolling(window=15).apply(lambda s: calculate_rsi(s), raw=False)
    except Exception:
        # fallback - compute last RSI using closing prices
        df['RSI'] = np.nan

    # Momentum
    df['Momentum'] = df['Close'] / df['Close'].shift(14) - 1

    return df

# ---------------------------
# CLI Main Function
# ---------------------------
def run_cli():
    """Run the CLI version of the market tracker"""
    ensure_dir(DATA_DIR)

    print("=" * 60)
    print("S&P 500 Market Tracker - CLI Mode")
    print("=" * 60)

    print("\nFetching S&P 500 tickers...")
    tickers, sp_df, sectors, industries = fetch_sp500_tickers()

    if not tickers:
        print("Failed to fetch tickers. Exiting.")
        return

    print(f"Found {len(tickers)} tickers")

    # Save constituents snapshot
    sp_df.to_csv(os.path.join(DATA_DIR, "sp500_constituents_snapshot.csv"), index=False)

    # Download historical data
    print("\nDownloading historical price data...")
    print("This may take several minutes depending on your connection speed.")

    hist = download_history(tickers, period=DOWNLOAD_PERIOD, interval="1d")

    metrics_rows = []
    per_ticker_dir = os.path.join(DATA_DIR, "history")
    ensure_dir(per_ticker_dir)

    print("\nProcessing ticker data...")

    for t in tqdm(tickers, desc="Computing metrics"):
        df = hist.get(t, pd.DataFrame())

        if df is None or df.empty:
            metrics = None
        else:
            # Save per-ticker CSV
            try:
                df.to_csv(os.path.join(per_ticker_dir, f"{t}.csv"))
            except Exception:
                pass

            metrics = compute_metrics_for_ticker(df)

        if metrics is None:
            metrics_rows.append({
                "ticker": t,
                "status": "no_data",
                "sector": sectors.get(t, "Unknown"),
                "industry": industries.get(t, "Unknown")
            })
        else:
            metrics['ticker'] = t
            metrics['status'] = 'ok'
            metrics['sector'] = sectors.get(t, "Unknown")
            metrics['industry'] = industries.get(t, "Unknown")
            metrics_rows.append(metrics)

    df_metrics = pd.DataFrame(metrics_rows)

    # Order columns
    cols_order = [
        'ticker', 'status', 'sector', 'industry', 'last_date', 'last_close',
        'data_points', 'rising_3day', 'declining_3day', 'pct_1d', 'pct_3d',
        'pct_5d', 'pct_21d', 'pct_63d', 'pct_252d', 'avg_1y', 'avg_2y',
        'avg_5y', 'avg_max', 'cum_return_from_start_pct', 'ann_vol_pct', 'rsi',
        '52w_high', '52w_low', 'pct_from_52w_high', 'pct_from_52w_low',
        'avg_volume_20d', 'volume_vs_avg'
    ]

    cols = [c for c in cols_order if c in df_metrics.columns]
    cols += [c for c in df_metrics.columns if c not in cols]
    df_metrics = df_metrics[cols]

    # Save master CSV
    master_csv = os.path.join(DATA_DIR, "latest_metrics.csv")
    ensure_dir(DATA_DIR)
    df_metrics.to_csv(master_csv, index=False)
    print(f"\n✓ Saved master metrics to {master_csv}")

    # Rising and declining lists
    rising = df_metrics[
        (df_metrics['status'] == 'ok') &
        (df_metrics['rising_3day'] == True)
    ].sort_values(by='pct_3d', ascending=False)

    declining = df_metrics[
        (df_metrics['status'] == 'ok') &
        (df_metrics['declining_3day'] == True)
    ].sort_values(by='pct_3d')

    rising.to_csv(os.path.join(DATA_DIR, "rising_3day.csv"), index=False)
    declining.to_csv(os.path.join(DATA_DIR, "declining_3day.csv"), index=False)

    # Additional analysis files

    # Top gainers/losers
    valid_metrics = df_metrics[df_metrics['status'] == 'ok'].copy()

    if not valid_metrics.empty:
        # Daily top movers
        if 'pct_1d' in valid_metrics.columns:
            top_gainers_1d = valid_metrics.nlargest(20, 'pct_1d')[['ticker', 'sector', 'last_close', 'pct_1d']]
            top_losers_1d = valid_metrics.nsmallest(20, 'pct_1d')[['ticker', 'sector', 'last_close', 'pct_1d']]

            top_gainers_1d.to_csv(os.path.join(DATA_DIR, "top_gainers_1d.csv"), index=False)
            top_losers_1d.to_csv(os.path.join(DATA_DIR, "top_losers_1d.csv"), index=False)

        # Most volatile stocks
        if 'ann_vol_pct' in valid_metrics.columns:
            most_volatile = valid_metrics.nlargest(20, 'ann_vol_pct')[['ticker', 'sector', 'last_close', 'ann_vol_pct']]
            most_volatile.to_csv(os.path.join(DATA_DIR, "most_volatile.csv"), index=False)

        # Sector performance summary
        # guard missing cols
        agg_map = {}
        if 'pct_1d' in valid_metrics.columns:
            agg_map['pct_1d'] = 'mean'
        if 'pct_5d' in valid_metrics.columns:
            agg_map['pct_5d'] = 'mean'
        if 'pct_21d' in valid_metrics.columns:
            agg_map['pct_21d'] = 'mean'
        if 'ann_vol_pct' in valid_metrics.columns:
            agg_map['ann_vol_pct'] = 'mean'
        agg_map['ticker'] = 'count'

        sector_summary = valid_metrics.groupby('sector').agg(agg_map).round(2)
        # rename columns safely
        col_rename = {}
        if 'pct_1d' in sector_summary.columns: col_rename['pct_1d'] = 'avg_1d_return'
        if 'pct_5d' in sector_summary.columns: col_rename['pct_5d'] = 'avg_5d_return'
        if 'pct_21d' in sector_summary.columns: col_rename['pct_21d'] = 'avg_21d_return'
        if 'ann_vol_pct' in sector_summary.columns: col_rename['ann_vol_pct'] = 'avg_volatility'
        if 'ticker' in sector_summary.columns: col_rename['ticker'] = 'stock_count'
        sector_summary = sector_summary.rename(columns=col_rename)
        if 'avg_5d_return' in sector_summary.columns:
            sector_summary = sector_summary.sort_values('avg_5d_return', ascending=False)
        sector_summary.to_csv(os.path.join(DATA_DIR, "sector_summary.csv"))

        # Print summary statistics
        print("\n" + "=" * 60)
        print("MARKET SUMMARY")
        print("=" * 60)

        print(f"\n📊 Overall Statistics:")
        print(f"  • Total stocks processed: {len(valid_metrics)}")
        print(f"  • Rising (3-day): {len(rising)} stocks")
        print(f"  • Declining (3-day): {len(declining)} stocks")

        if 'pct_1d' in valid_metrics.columns:
            gainers_1d = len(valid_metrics[valid_metrics['pct_1d'] > 0])
            print(f"  • Daily gainers: {gainers_1d} ({gainers_1d/len(valid_metrics)*100:.1f}%)")
            print(f"  • Daily losers: {len(valid_metrics) - gainers_1d} ({(len(valid_metrics)-gainers_1d)/len(valid_metrics)*100:.1f}%)")

        print(f"\n📈 Market Performance (Averages):")
        if 'pct_1d' in valid_metrics.columns:
            print(f"  • 1-day return: {valid_metrics['pct_1d'].mean():.2f}%")
        if 'pct_5d' in valid_metrics.columns:
            print(f"  • 5-day return: {valid_metrics['pct_5d'].mean():.2f}%")
        if 'pct_21d' in valid_metrics.columns:
            print(f"  • 21-day return: {valid_metrics['pct_21d'].mean():.2f}%")
        if 'ann_vol_pct' in valid_metrics.columns:
            print(f"  • Annual volatility: {valid_metrics['ann_vol_pct'].mean():.1f}%")

        if 'pct_1d' in locals():
            print(f"\n🏆 Top 5 Daily Gainers:")
            for _, row in top_gainers_1d.head().iterrows():
                print(f"  • {row['ticker']}: +{row['pct_1d']:.2f}% ({row['sector']})")

        if 'top_losers_1d' in locals():
            print(f"\n📉 Top 5 Daily Losers:")
            for _, row in top_losers_1d.head().iterrows():
                print(f"  • {row['ticker']}: {row['pct_1d']:.2f}% ({row['sector']})")

        if 'avg_5d_return' in sector_summary.columns:
            print(f"\n🏢 Best Performing Sectors (5-day):")
            for sector, row in sector_summary.head(3).iterrows():
                print(f"  • {sector}: {row['avg_5d_return']:.2f}% ({int(row['stock_count'])} stocks)")

    print("\n✅ Analysis complete! Check the 'data' directory for all output files.")
    print("=" * 60)

# ---------------------------
# Streamlit Web Interface
# ---------------------------
def plot_ticker_price_rsi(ticker_csv_path, ticker):
    """Helper: load per-ticker csv and produce a 2-row plot: price and RSI"""
    if not os.path.exists(ticker_csv_path):
        st.info("No per-ticker history file available for plotting.")
        return

    df = pd.read_csv(ticker_csv_path, parse_dates=True, index_col=0)
    if df.empty or 'Close' not in df.columns:
        st.info("Insufficient data in per-ticker file.")
        return

    # Add RSI and Momentum
    df = add_technical_indicators(df)
    # build subplot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.7, 0.3], specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=f"{ticker} Close"), row=1, col=1)
    # add MA20/50 if present
    if 'MA20' in df.columns:
        fig.add_trace(go.Line(x=df.index, y=df['MA20'], name='MA20', line=dict(dash='dash')), row=1, col=1)
    if 'MA50' in df.columns:
        fig.add_trace(go.Line(x=df.index, y=df['MA50'], name='MA50', line=dict(dash='dot')), row=1, col=1)

    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_layout(height=600, title_text=f"{ticker} Price + RSI")
    st.plotly_chart(fig, use_container_width=True)

def run_streamlit():
    """Run the Streamlit web interface"""

    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit is not installed. Please install with:")
        print("pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="S&P 500 Market Tracker",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("📈 S&P 500 Market Tracker")
    with col2:
        st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col3:
        if st.button("🔄 Refresh", key="refresh_btn_streamlit"):
            # clear cached data (if using st.cache_data) and rerun
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.runtime.scriptrunner.rerun()

    # Check for existing data
    data_exists = os.path.exists(os.path.join(DATA_DIR, "latest_metrics.csv"))

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    if not data_exists:
        st.sidebar.warning("No data found. Run initial data fetch first.")
        if st.sidebar.button("📥 Fetch All Data", key="fetch_all_data_btn"):
            with st.spinner("Fetching data... This may take several minutes."):
                run_cli()
            st.success("Data fetched successfully!")
            st.runtime.scriptrunner.rerun()
    else:
        if st.sidebar.button("🔄 Update Data", key="update_data_btn"):
            with st.spinner("Updating data..."):
                run_cli()
            st.success("Data updated!")
            st.runtime.scriptrunner.rerun()

    if data_exists:
        # Load data
        df_metrics = pd.read_csv(os.path.join(DATA_DIR, "latest_metrics.csv"))
        valid_metrics = df_metrics[df_metrics['status'] == 'ok'].copy()

        # Debug: Show available columns
        st.sidebar.text("Available columns:")
        pct_cols = [col for col in valid_metrics.columns if col.startswith('pct_')]
        st.sidebar.text(", ".join(pct_cols))

        # View mode
        view_mode = st.sidebar.radio(
            "View Mode",
            ["Dashboard", "Sector Analysis", "Top Movers", "Technical Screener", "Data Export"],
            key="view_mode_radio"
        )

        # Time horizon - check which columns are available (use this single horizon selector)
        horizon_map_full = {
            "1 Day": "pct_1d",
            "3 Days": "pct_3d",
            "5 Days": "pct_5d",
            "1 Month": "pct_21d",
            "3 Months": "pct_63d",
            "1 Year": "pct_252d"
        }

        # Filter to only available columns
        horizon_map = {label: col for label, col in horizon_map_full.items() if col in valid_metrics.columns}

        # If no pct columns found, show error
        if not horizon_map:
            st.error("No return columns found in data. Available columns: " + ", ".join(valid_metrics.columns.tolist()))
            if st.button("🔄 Regenerate Data", key="regenerate_data_btn"):
                with st.spinner("Regenerating data with all metrics..."):
                    run_cli()
                st.success("Data regenerated!")
                st.runtime.scriptrunner.rerun()
            return

        # Default horizon choice
        default_horizon = "5 Days"
        if "5 Days" in horizon_map:
            default_horizon = "5 Days"
        elif "3 Days" in horizon_map:
            default_horizon = "3 Days"
        else:
            default_horizon = list(horizon_map.keys())[0]

        # Single select_slider (unique key)
        horizon_label = st.sidebar.select_slider(
            "Time Horizon",
            options=list(horizon_map.keys()),
            value=default_horizon,
            key="horizon_slider_single"
        )
        horizon_col = horizon_map[horizon_label]

        # Show debug
        st.sidebar.text(f"Selected column: {horizon_col}")
        st.sidebar.text(f"Column exists: {horizon_col in valid_metrics.columns}")

        # ---------- Dashboard ----------
        if view_mode == "Dashboard":
            # Market Overview
            st.subheader("📊 Market Overview")

            col1, col2, col3, col4 = st.columns(4)

            gainers = len(valid_metrics[valid_metrics[horizon_col] > 0])
            total = len(valid_metrics)

            with col1:
                st.metric(
                    "Market Breadth",
                    f"{gainers}/{total}",
                    f"{(gainers/total*100):.1f}% advancing"
                )

            with col2:
                avg_return = valid_metrics[horizon_col].mean()
                st.metric(f"Avg {horizon_label} Return", f"{avg_return:.2f}%")

            with col3:
                avg_volatility = valid_metrics['ann_vol_pct'].mean() if 'ann_vol_pct' in valid_metrics.columns else np.nan
                st.metric("Avg Volatility", f"{avg_volatility:.1f}%" if not pd.isna(avg_volatility) else "n/a")

            with col4:
                if 'rsi' in valid_metrics.columns:
                    avg_rsi = valid_metrics['rsi'].mean()
                    st.metric("Avg RSI", f"{avg_rsi:.1f}")

            # Top movers
            st.subheader(f"🚀 Top Movers ({horizon_label})")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📈 Top Gainers")
                top_gainers = valid_metrics.nlargest(10, horizon_col)[['ticker', 'sector', horizon_col, 'last_close']]

                fig_gainers = go.Figure(data=[
                    go.Bar(
                        x=top_gainers['ticker'],
                        y=top_gainers[horizon_col],
                        text=top_gainers[horizon_col].apply(lambda x: f"{x:.1f}%"),
                        textposition='outside',
                        marker_color='lightgreen',
                        hovertemplate='<b>%{x}</b><br>' +
                                      'Return: %{y:.2f}%<br>' +
                                      'Price: $%{customdata[0]:.2f}<br>' +
                                      'Sector: %{customdata[1]}<extra></extra>',
                        customdata=top_gainers[['last_close', 'sector']].values
                    )
                ])
                fig_gainers.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title=f"Return % ({horizon_label})"
                )
                st.plotly_chart(fig_gainers, use_container_width=True)

            with col2:
                st.markdown("### 📉 Top Decliners")
                top_losers = valid_metrics.nsmallest(10, horizon_col)[['ticker', 'sector', horizon_col, 'last_close']]

                fig_losers = go.Figure(data=[
                    go.Bar(
                        x=top_losers['ticker'],
                        y=top_losers[horizon_col],
                        text=top_losers[horizon_col].apply(lambda x: f"{x:.1f}%"),
                        textposition='outside',
                        marker_color='lightcoral',
                        hovertemplate='<b>%{x}</b><br>' +
                                      'Return: %{y:.2f}%<br>' +
                                      'Price: $%{customdata[0]:.2f}<br>' +
                                      'Sector: %{customdata[1]}<extra></extra>',
                        customdata=top_losers[['last_close', 'sector']].values
                    )
                ])
                fig_losers.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title=f"Return % ({horizon_label})"
                )
                st.plotly_chart(fig_losers, use_container_width=True)

            # Data table
            st.subheader("📋 All Stocks")

            display_cols = ['ticker', 'sector', 'last_close', horizon_col, 'ann_vol_pct']
            if 'rsi' in valid_metrics.columns:
                display_cols.append('rsi')

            display_df = valid_metrics[display_cols].copy()
            # rename for display
            renamed_cols = ['Ticker', 'Sector', 'Price', f'{horizon_label} %', 'Volatility']
            if 'rsi' in valid_metrics.columns:
                renamed_cols.append('RSI')
            display_df.columns = renamed_cols

            st.dataframe(
                display_df.style.format({
                    'Price': '${:.2f}',
                    f'{horizon_label} %': '{:.2f}%',
                    'Volatility': '{:.1f}%',
                    'RSI': '{:.1f}' if 'RSI' in display_df.columns else None
                }),
                use_container_width=True,
                height=400
            )

        # ---------- Sector Analysis ----------
        elif view_mode == "Sector Analysis":
            st.subheader("🏢 Sector Performance Analysis")

            sector_perf = valid_metrics.groupby('sector').agg({
                horizon_col: ['mean', 'median', 'std'],
                'ticker': 'count',
                'ann_vol_pct': 'mean'
            }).round(2)

            sector_perf.columns = ['Mean Return', 'Median Return', 'Std Dev', 'Count', 'Avg Volatility']
            sector_perf = sector_perf.sort_values('Mean Return', ascending=False)

            # Bar chart
            fig_sector = go.Figure(data=[
                go.Bar(
                    x=sector_perf.index,
                    y=sector_perf['Mean Return'],
                    text=sector_perf['Mean Return'].apply(lambda x: f"{x:.2f}%"),
                    textposition='outside',
                    marker_color=sector_perf['Mean Return'].apply(
                        lambda x: 'lightgreen' if x > 0 else 'lightcoral'
                    )
                )
            ])
            fig_sector.update_layout(
                title=f"Average Sector Returns ({horizon_label})",
                xaxis_title="Sector",
                yaxis_title="Average Return (%)",
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_sector, use_container_width=True)

            # Table
            st.dataframe(
                sector_perf.style.format({
                    'Mean Return': '{:.2f}%',
                    'Median Return': '{:.2f}%',
                    'Std Dev': '{:.2f}%',
                    'Avg Volatility': '{:.1f}%'
                }),
                use_container_width=True
            )

        # ---------- Top Movers ----------
        elif view_mode == "Top Movers":
            st.subheader("🎯 Market Movers Analysis")

            tabs = st.tabs(["Rising Stocks", "Declining Stocks", "Most Volatile", "52-Week Highs/Lows"])

            with tabs[0]:
                rising = valid_metrics[valid_metrics['rising_3day'] == True].sort_values('pct_3d', ascending=False)
                st.metric("Total Rising (3-day consecutive)", len(rising))
                if not rising.empty:
                    st.dataframe(
                        rising[['ticker', 'sector', 'last_close', 'pct_1d', 'pct_3d', 'pct_5d']].head(20),
                        use_container_width=True
                    )

            with tabs[1]:
                declining = valid_metrics[valid_metrics['declining_3day'] == True].sort_values('pct_3d')
                st.metric("Total Declining (3-day consecutive)", len(declining))
                if not declining.empty:
                    st.dataframe(
                        declining[['ticker', 'sector', 'last_close', 'pct_1d', 'pct_3d', 'pct_5d']].head(20),
                        use_container_width=True
                    )

            with tabs[2]:
                if 'ann_vol_pct' in valid_metrics.columns:
                    most_volatile = valid_metrics.nlargest(20, 'ann_vol_pct')[['ticker', 'sector', 'last_close', 'ann_vol_pct', horizon_col]]
                    st.metric("Highest Volatility Stock", f"{most_volatile.iloc[0]['ticker']} ({most_volatile.iloc[0]['ann_vol_pct']:.1f}%)")
                    st.dataframe(most_volatile, use_container_width=True)
                else:
                    st.info("Volatility data not available")

            with tabs[3]:
                if 'pct_from_52w_high' in valid_metrics.columns:
                    near_highs = valid_metrics.nlargest(20, 'pct_from_52w_high')[['ticker', 'sector', 'last_close', '52w_high', 'pct_from_52w_high']]
                    near_lows = valid_metrics.nsmallest(20, 'pct_from_52w_low')[['ticker', 'sector', 'last_close', '52w_low', 'pct_from_52w_low']]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Near 52-Week Highs")
                        st.dataframe(near_highs, use_container_width=True)
                    with col2:
                        st.markdown("### Near 52-Week Lows")
                        st.dataframe(near_lows, use_container_width=True)
                else:
                    st.info("52-week high/low data not available")

        # ---------- Technical Screener ----------
        elif view_mode == "Technical Screener":
            st.subheader("🔍 Technical Stock Screener")

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
                if 'rsi' in valid_metrics.columns:
                    min_rsi = st.number_input("Min RSI", value=0.0, max_value=100.0, step=1.0, key="ts_min_rsi")
                    max_rsi = st.number_input("Max RSI", value=100.0, min_value=0.0, step=1.0, key="ts_max_rsi")
                else:
                    min_rsi, max_rsi = 0, 100

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
                only_rising = st.checkbox("Only Rising (3-day)", False, key="ts_only_rising")
                only_declining = st.checkbox("Only Declining (3-day)", False, key="ts_only_declining")

            with col2:
                min_price = st.number_input("Min Price ($)", value=0.0, step=1.0, key="ts_min_price")
                max_price = st.number_input("Max Price ($)", value=10000.0, step=1.0, key="ts_max_price")

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

            if 'rsi' in valid_metrics.columns:
                screened_df = screened_df[
                    (screened_df['rsi'] >= min_rsi) &
                    (screened_df['rsi'] <= max_rsi)
                ]

            if only_rising:
                screened_df = screened_df[screened_df['rising_3day'] == True]
            if only_declining:
                screened_df = screened_df[screened_df['declining_3day'] == True]

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
                display_cols = ['ticker', 'sector', 'last_close', horizon_col, 'ann_vol_pct']
                if 'rsi' in screened_df.columns:
                    display_cols.append('rsi')

                st.dataframe(
                    screened_df[display_cols].style.format({
                        'last_close': '${:.2f}',
                        horizon_col: '{:.2f}%',
                        'ann_vol_pct': '{:.1f}%',
                        'rsi': '{:.1f}' if 'rsi' in display_cols else None
                    }),
                    use_container_width=True,
                    height=400
                )

                # Visualization
                if len(screened_df) > 1:
                    fig = px.scatter(
                        screened_df,
                        x='ann_vol_pct' if 'ann_vol_pct' in screened_df.columns else None,
                        y=horizon_col,
                        color='sector',
                        size='last_close',
                        hover_data=['ticker'],
                        title=f"{horizon_label} Return vs Volatility",
                        labels={
                            'ann_vol_pct': 'Annual Volatility (%)',
                            horizon_col: f'{horizon_label} Return (%)'
                        }
                    )
                    # add horizontal zero line if y exists
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    st.plotly_chart(fig, use_container_width=True)

                # Per-ticker plotting: choose one from screened results
                tickers_list = screened_df['ticker'].tolist()
                if tickers_list:
                    chosen_ticker = st.selectbox("Select ticker to chart", options=tickers_list, key="ts_ticker_chart")
                    # Load per-ticker CSV from data/history/<ticker>.csv and plot
                    ticker_csv_path = os.path.join(DATA_DIR, "history", f"{chosen_ticker}.csv")
                    plot_ticker_price_rsi(ticker_csv_path, chosen_ticker)
            else:
                st.info("No stocks match the selected criteria")

        # ---------- Data Export ----------
        elif view_mode == "Data Export":
            st.subheader("📥 Data Export")

            st.markdown("### Available Data Files")

            files = {
                "Master Metrics": "latest_metrics.csv",
                "S&P 500 Constituents": "sp500_constituents_snapshot.csv",
                "Rising Stocks (3-day)": "rising_3day.csv",
                "Declining Stocks (3-day)": "declining_3day.csv",
                "Top Daily Gainers": "top_gainers_1d.csv",
                "Top Daily Losers": "top_losers_1d.csv",
                "Most Volatile": "most_volatile.csv",
                "Sector Summary": "sector_summary.csv"
            }

            for name, filename in files.items():
                filepath = os.path.join(DATA_DIR, filename)
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

            export_cols = st.multiselect(
                "Select Columns for Export",
                options=list(valid_metrics.columns),
                default=['ticker', 'sector', 'last_close', 'pct_1d', 'pct_5d', 'ann_vol_pct'],
                key="export_cols"
            )

            if export_sectors and export_cols:
                export_df = valid_metrics[valid_metrics['sector'].isin(export_sectors)][export_cols]

                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Custom Export",
                    data=csv,
                    file_name=f"custom_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_custom_export"
                )

                st.dataframe(export_df.head(10), use_container_width=True)

    # Footer
    st.divider()
    st.caption("""
    Created by: Ehiremen Nathaniel Omoarebun 

    **Disclaimer:** This tool is for informational purposes only and should not be considered as financial advice.
    Data provided by Yahoo Finance and may be delayed. Always do your own research before making investment decisions.
    """)

# ---------------------------
# Main Entry Point
# ---------------------------
def main():
    """Main entry point for the application"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="S&P 500 Market Tracker - CLI and Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  CLI mode:   python %(prog)s --mode cli
  Web mode:   python %(prog)s --mode web
  Or:         streamlit run %(prog)s -- --mode web
        """
    )

    parser.add_argument(
        '--mode',
        choices=['cli', 'web'],
        default='cli',
        help='Run mode: cli for command line, web for Streamlit interface'
    )

    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory for storing data files (default: data)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for downloading tickers (default: 50)'
    )

    # Handle streamlit's extra arguments
    if 'streamlit' in sys.modules:
        # If running through streamlit, default to web mode
        args = parser.parse_args(['--mode', 'web'])
    else:
        args = parser.parse_args()

    # Update global settings
    global DATA_DIR, BATCH_SIZE
    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size

    # Run appropriate mode
    if args.mode == 'cli':
        run_cli()
    else:
        run_streamlit()

if __name__ == "__main__":
    main()
