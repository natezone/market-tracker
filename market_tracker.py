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
MIN_CONSECUTIVE = 30
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
    """Display color legend for performance interpretation"""
    with st.expander("Color Legend"):
        st.markdown("""
        **Performance Colors:**
        - Dark Green: Excellent performance (>5% gains)
        - Green: Good performance (2-5% gains)  
        - Light Green: Positive performance (0-2% gains)
        - Gray: Neutral (0% change)
        - Yellow: Caution zone
        - Light Red: Negative performance (0-2% loss)
        - Red: Poor performance (2-5% loss)
        - Dark Red: Very poor performance (>5% loss)
        
        **RSI Colors:**
        - Green: Oversold (potential buy, RSI < 30)
        - Yellow: Neutral (30-70)
        - Red: Overbought (potential sell, RSI > 70)
        
        **Volatility Colors:**
        - Green: Low risk (< 25%)
        - Yellow: Medium risk (25-50%)
        - Red: High risk (> 50%)
        """)

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
# CLI Main Function
# ---------------------------
def run_cli(consecutive_days=7, index_key="SP500"):
    """Run the CLI version of the market tracker with configurable consecutive period"""
    ensure_dir(DATA_DIR)

    print("=" * 60)
    print(f"S&P 500 Market Tracker - CLI Mode (Consecutive: {consecutive_days} days)")
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

    print(f"\nProcessing ticker data with {consecutive_days}-day consecutive analysis...")

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

            metrics = compute_metrics_for_ticker(df, consecutive_days)

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

    # Dynamic column ordering based on consecutive_days
    cols_order = [
        'ticker', 'status', 'sector', 'industry', 'last_date', 'last_close',
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

    # Save master CSV
    master_csv = os.path.join(DATA_DIR, "latest_metrics.csv")
    ensure_dir(DATA_DIR)
    df_metrics.to_csv(master_csv, index=False)
    print(f"\n✓ Saved master metrics to {master_csv}")

    # Rising and declining lists using configurable period
    rising_col = f'rising_{consecutive_days}day'
    declining_col = f'declining_{consecutive_days}day'
    pct_col = f'pct_{consecutive_days}d' if consecutive_days not in [1, 3, 5, 7, 21, 63, 252] else f'pct_{consecutive_days}d'

    # Use closest available percentage column
    if pct_col not in df_metrics.columns:
        if consecutive_days <= 3:
            pct_col = 'pct_3d'
        elif consecutive_days <= 5:
            pct_col = 'pct_5d'
        elif consecutive_days <= 7:
            pct_col = 'pct_7d'
        elif consecutive_days <= 21:
            pct_col = 'pct_21d'
        else:
            pct_col = 'pct_63d'

    rising = df_metrics[
        (df_metrics['status'] == 'ok') &
        (df_metrics[rising_col] == True)
    ].sort_values(by=pct_col, ascending=False)

    declining = df_metrics[
        (df_metrics['status'] == 'ok') &
        (df_metrics[declining_col] == True)
    ].sort_values(by=pct_col)

    rising.to_csv(os.path.join(DATA_DIR, f"rising_{consecutive_days}day.csv"), index=False)
    declining.to_csv(os.path.join(DATA_DIR, f"declining_{consecutive_days}day.csv"), index=False)

    # Print summary with configurable period
    print("\n" + "=" * 60)
    print("MARKET SUMMARY")
    print("=" * 60)

    valid_metrics = df_metrics[df_metrics['status'] == 'ok'].copy()
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
    st.plotly_chart(fig, width='stretch')

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

    # Add custom CSS
    add_custom_css()

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
            st.rerun()

    # Check for existing data
    data_exists = os.path.exists(os.path.join(DATA_DIR, "latest_metrics.csv"))

    # Sidebar - Controls
    st.sidebar.header("⚙️ Settings")

    # 1. View mode
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Dashboard", "Sector Analysis", "Top Movers", "Technical Screener", "Data Export"],
        key="view_mode_radio"
    )

    # Load data to check available columns
    if data_exists:
        df_metrics = pd.read_csv(os.path.join(DATA_DIR, "latest_metrics.csv"))
        valid_metrics = df_metrics[df_metrics['status'] == 'ok'].copy()
        
        # 2. Time horizon slider
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

        if horizon_map:
            # Default horizon choice
            default_horizon = "5 Days"
            if "5 Days" in horizon_map:
                default_horizon = "5 Days"
            elif "3 Days" in horizon_map:
                default_horizon = "3 Days"
            else:
                default_horizon = list(horizon_map.keys())[0]

            # Time horizon selector
            horizon_label = st.sidebar.select_slider(
                "Time Horizon",
                options=list(horizon_map.keys()),
                value=default_horizon,
                key="horizon_slider_single"
            )
            horizon_col = horizon_map[horizon_label]
        else:
            horizon_label = "N/A"
            horizon_col = None
    else:
        horizon_label = "N/A"
        horizon_col = None

    # 3. Consecutive days slider
    consecutive_days = st.sidebar.slider(
        "Consecutive Days Lookback",
        min_value=2,
        max_value=126,  # ~6 months of trading days
        value=3,
        step=1,
        help="Number of consecutive days to analyze for rising/declining trends"
    )

    # Add preset buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("3d", key="preset_3d"):
            consecutive_days = 3
    with col2:
        if st.button("1w", key="preset_1w"):
            consecutive_days = 5
    with col3:
        if st.button("1m", key="preset_1m"):
            consecutive_days = 21

    st.sidebar.info(f"Current: {consecutive_days} days ({consecutive_days/5:.1f} weeks)")

    # Add warning for very long periods
    if consecutive_days > 63:
        st.sidebar.warning("⚠️ Long periods may have fewer matching stocks")

    # Show color legend
    show_color_legend()

    if not data_exists:
        st.sidebar.warning("No data found. Run initial data fetch first.")
        if st.sidebar.button("📥 Fetch All Data", key="fetch_all_data_btn"):
            with st.spinner("Fetching data... This may take several minutes."):
                run_cli(consecutive_days)
            st.success("Data fetched successfully!")
            st.rerun()
    else:
        if st.sidebar.button("🔄 Update Data", key="update_data_btn"):
            with st.spinner(f"Updating data with {consecutive_days}-day analysis..."):
                run_cli(consecutive_days)
            st.success("Data updated!")
            st.rerun()

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

        # ---------- Dashboard ----------
        if view_mode == "Dashboard":
            st.subheader("📊 Market Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Use dynamic column names
            rising_count = len(valid_metrics[valid_metrics.get(rising_col, False) == True])
            declining_count = len(valid_metrics[valid_metrics.get(declining_col, False) == True])
            
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

            display_cols = ['ticker', 'sector', 'last_close', horizon_col, 'ann_vol_pct']
            if 'rsi' in valid_metrics.columns:
                display_cols.append('rsi')

            display_df = valid_metrics[display_cols].copy()
            
            # Apply color coding function
            def highlight_performance(row):
                colors = []
                for col in display_df.columns:
                    if col == horizon_col:
                        color = get_performance_color(row[col], 'return')
                    elif col == 'ann_vol_pct':
                        color = get_performance_color(row[col], 'volatility')
                    elif col == 'rsi':
                        color = get_performance_color(row[col], 'rsi')
                    else:
                        color = 'white'
                    
                    text_color = 'white' if color in ['#8B0000', '#006400', '#DC143C'] else 'black'
                    colors.append(f'background-color: {color}; color: {text_color}')
                return colors

            styled_df = display_df.style.apply(highlight_performance, axis=1)
            
            # Format the data
            format_dict = {
                'last_close': '${:.2f}',
                horizon_col: '{:.2f}%',
                'ann_vol_pct': '{:.2f}%'
            }
            if 'rsi' in display_df.columns:
                format_dict['rsi'] = '{:.1f}'
            
            styled_df = styled_df.format(format_dict)
            
            st.dataframe(styled_df, use_container_width=True, height=400)

        # ---------- Sector Analysis ----------
        elif view_mode == "Sector Analysis":
            st.subheader("🏢 S&P 500 Sector Performance Analysis")

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

            # Format sector performance table
            formatted_sector = sector_perf.copy()
            formatted_sector['Mean Return'] = formatted_sector['Mean Return'].apply(format_percentage)
            formatted_sector['Median Return'] = formatted_sector['Median Return'].apply(format_percentage)
            formatted_sector['Std Dev'] = formatted_sector['Std Dev'].apply(format_percentage)
            formatted_sector['Avg Volatility'] = formatted_sector['Avg Volatility'].apply(format_percentage)
            
            st.dataframe(formatted_sector, use_container_width=True)

        # ---------- Top Movers ----------
        elif view_mode == "Top Movers":
            st.subheader("🎯 S&P 500 Market Movers Analysis")
            
            tabs = st.tabs([f"Rising Stocks ({consecutive_days}d)", f"Declining Stocks ({consecutive_days}d)", "Most Volatile", "52-Week Highs/Lows"])
            
            with tabs[0]:
                rising = valid_metrics[valid_metrics.get(rising_col, False) == True]
                if not rising.empty:
                    # Sort by appropriate percentage column
                    sort_col = 'pct_5d' if consecutive_days <= 5 else 'pct_21d'
                    if sort_col in rising.columns:
                        rising = rising.sort_values(sort_col, ascending=False)
                
                st.metric(f"Total Rising ({consecutive_days}-day consecutive)", len(rising))
                if not rising.empty:
                    display_cols = ['ticker', 'sector', 'last_close', 'pct_1d', 'pct_3d', 'pct_5d']
                    if 'pct_21d' in rising.columns:
                        display_cols.append('pct_21d')
                    # Filter to only existing columns
                    display_cols = [col for col in display_cols if col in rising.columns]
                    
                    # Take head first, then format
                    rising_display = rising[display_cols].head(20).copy()
                    
                    # Format the data
                    rising_display['last_close'] = rising_display['last_close'].apply(format_currency)
                    for col in display_cols:
                        if col.startswith('pct_'):
                            rising_display[col] = rising_display[col].apply(format_percentage)
                    
                    st.dataframe(rising_display, use_container_width=True)
            
            with tabs[1]:
                declining = valid_metrics[valid_metrics.get(declining_col, False) == True]
                if not declining.empty:
                    sort_col = 'pct_5d' if consecutive_days <= 5 else 'pct_21d'
                    if sort_col in declining.columns:
                        declining = declining.sort_values(sort_col)
                
                st.metric(f"Total Declining ({consecutive_days}-day consecutive)", len(declining))
                if not declining.empty:
                    display_cols = ['ticker', 'sector', 'last_close', 'pct_1d', 'pct_3d', 'pct_5d']
                    if 'pct_21d' in declining.columns:
                        display_cols.append('pct_21d')
                    # Filter to only existing columns
                    display_cols = [col for col in display_cols if col in declining.columns]
                    
                    # Take head first, then format
                    declining_display = declining[display_cols].head(20).copy()
                    
                    # Format the data
                    declining_display['last_close'] = declining_display['last_close'].apply(format_currency)
                    for col in display_cols:
                        if col.startswith('pct_'):
                            declining_display[col] = declining_display[col].apply(format_percentage)
                    
                    st.dataframe(declining_display, use_container_width=True)

            with tabs[2]:
                if 'ann_vol_pct' in valid_metrics.columns:
                    most_volatile = valid_metrics.nlargest(20, 'ann_vol_pct')[['ticker', 'sector', 'last_close', 'ann_vol_pct', horizon_col]].copy()
                    st.metric("Highest Volatility Stock", f"{most_volatile.iloc[0]['ticker']} ({most_volatile.iloc[0]['ann_vol_pct']:.1f}%)")
                    
                    # Format the data
                    most_volatile['last_close'] = most_volatile['last_close'].apply(format_currency)
                    most_volatile['ann_vol_pct'] = most_volatile['ann_vol_pct'].apply(format_percentage)
                    most_volatile[horizon_col] = most_volatile[horizon_col].apply(format_percentage)
                    
                    st.dataframe(most_volatile, use_container_width=True)
                else:
                    st.info("Volatility data not available")

            with tabs[3]:
                if 'pct_from_52w_high' in valid_metrics.columns:
                    near_highs = valid_metrics.nlargest(20, 'pct_from_52w_high')[['ticker', 'sector', 'last_close', '52w_high', 'pct_from_52w_high']].copy()
                    near_lows = valid_metrics.nsmallest(20, 'pct_from_52w_low')[['ticker', 'sector', 'last_close', '52w_low', 'pct_from_52w_low']].copy()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Near 52-Week Highs")
                        # Format the data
                        near_highs['last_close'] = near_highs['last_close'].apply(format_currency)
                        near_highs['52w_high'] = near_highs['52w_high'].apply(format_currency)
                        near_highs['pct_from_52w_high'] = near_highs['pct_from_52w_high'].apply(format_percentage)
                        st.dataframe(near_highs, use_container_width=True)
                        
                    with col2:
                        st.markdown("### Near 52-Week Lows")
                        # Format the data
                        near_lows['last_close'] = near_lows['last_close'].apply(format_currency)
                        near_lows['52w_low'] = near_lows['52w_low'].apply(format_currency)
                        near_lows['pct_from_52w_low'] = near_lows['pct_from_52w_low'].apply(format_percentage)
                        st.dataframe(near_lows, use_container_width=True)
                else:
                    st.info("52-week high/low data not available")

        # ---------- Technical Screener ----------
        elif view_mode == "Technical Screener":
            st.subheader("🔍 S&P 500 Technical Stock Screener")

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

            if 'rsi' in valid_metrics.columns:
                screened_df = screened_df[
                    (screened_df['rsi'] >= min_rsi) &
                    (screened_df['rsi'] <= max_rsi)
                ]

            # Apply consecutive filters
            if only_rising:
                screened_df = screened_df[screened_df.get(rising_col, False) == True]
            if only_declining:
                screened_df = screened_df[screened_df.get(declining_col, False) == True]

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

                # Format the screener results
                formatted_screener = screened_df[display_cols].copy()
                formatted_screener['last_close'] = formatted_screener['last_close'].apply(format_currency)
                formatted_screener[horizon_col] = formatted_screener[horizon_col].apply(format_percentage)
                if 'ann_vol_pct' in formatted_screener.columns:
                    formatted_screener['ann_vol_pct'] = formatted_screener['ann_vol_pct'].apply(format_percentage)
                if 'rsi' in formatted_screener.columns:
                    formatted_screener['rsi'] = formatted_screener['rsi'].apply(lambda x: format_number(x, 1))

                st.dataframe(formatted_screener, use_container_width=True, height=400)

                # Enhanced Visualization with color coding
                if len(screened_df) > 1 and 'ann_vol_pct' in screened_df.columns:
                    fig = px.scatter(
                        screened_df,
                        x='ann_vol_pct',
                        y=horizon_col,
                        color=horizon_col,
                        color_continuous_scale=create_gradient_colorscale(screened_df[horizon_col], 'return'),
                        size='last_close',
                        hover_data=['ticker', 'sector'],
                        title=f"{horizon_label} Return vs Volatility",
                        labels={
                            'ann_vol_pct': 'Annual Volatility (%)',
                            horizon_col: f'{horizon_label} Return (%)'
                        }
                    )
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
                "S&P 500 Metrics": "latest_metrics.csv",
                "S&P 500 Constituents": "sp500_constituents_snapshot.csv",
                f"Rising Stocks ({consecutive_days}-day)": f"rising_{consecutive_days}day.csv",
                f"Declining Stocks ({consecutive_days}-day)": f"declining_{consecutive_days}day.csv"
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
                    file_name=f"sp500_custom_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_custom_export"
                )

                st.dataframe(export_df.head(10), use_container_width=True)

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
        '--consecutive-days',
        type=int,
        default=3,
        help='Number of consecutive days for trend analysis (default: 3)'
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
        run_cli(args.consecutive_days)
    else:
        run_streamlit()

if __name__ == "__main__":
    main()