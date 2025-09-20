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
from datetime import timezone, timedelta
import scipy.stats as stats
from scipy.optimize import minimize

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
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 50:  # Minimum data requirement
                returns_df[ticker] = returns
    
    if returns_df.empty or len(returns_df.columns) < 2:
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
                'sector': row.get('sector', 'Unknown'),
                'last_close': row.get('last_close', 0),
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

def render_comparison_mode(valid_metrics, hist):
    """Render the Comparison Mode interface"""
    st.subheader("🔄 Stock Comparison Analysis")
    
    # Stock selection
    available_tickers = sorted(valid_metrics['ticker'].unique())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Initialize session state for selected tickers if not exists
        if 'comparison_selected_tickers' not in st.session_state:
            st.session_state.comparison_selected_tickers = []
        
        selected_tickers = st.multiselect(
            "Select stocks to compare (2-15 stocks)",
            options=available_tickers,
            default=st.session_state.comparison_selected_tickers,
            max_selections=15,
            key="comparison_tickers"
        )
        
        # Update session state when selection changes
        st.session_state.comparison_selected_tickers = selected_tickers

    with col2:
        if st.button("Add Top Performers", key="add_top_performers"):
            # Get top performers based on available data
            available_return_cols = [col for col in ['pct_21d', 'pct_5d', 'pct_1d'] if col in valid_metrics.columns]
            if available_return_cols:
                top_performers = valid_metrics.nlargest(10, available_return_cols[0])['ticker'].tolist()
                
                # Combine existing selections with top performers
                current_selections = st.session_state.comparison_selected_tickers
                new_selections = list(set(current_selections + top_performers))[:15]  # Limit to 15

                # Update session state
                st.session_state.comparison_selected_tickers = new_selections

                st.success(f"Added top 10 performers: {', '.join(top_performers[:10])}")
                st.rerun()  
    
    if len(selected_tickers) < 2:
        st.info("Please select at least 2 stocks to compare")
        return
    
    # Get comparison data
    comparison_df = get_sector_comparison_data(valid_metrics, selected_tickers)
    
    if comparison_df.empty:
        st.error("No data available for selected stocks")
        return
    
    # Display comparison table
    st.subheader("📊 Performance Comparison")
    
    # Format the comparison table
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
            fig = px.scatter(
                comparison_df,
                x='ann_vol_pct',
                y='pct_252d',
                text='ticker',
                title="Risk vs Return Analysis (1 Year)",
                labels={'ann_vol_pct': 'Volatility (%)', 'pct_252d': '1-Year Return (%)'},
                size='last_close',
                color='sector'
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for risk vs return analysis")
    
    with tabs[2]:
        # Correlation analysis
        if len(selected_tickers) >= 2:
            # Get historical data for correlation
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
        # Top Performers - Auto-select top 20 stocks
        st.subheader("🏆 Top 20 Performers")
        
        # Determine which time horizon column to use
        horizon_map = {
            "1 Day": "pct_1d",
            "3 Days": "pct_3d", 
            "5 Days": "pct_5d",
            "1 Month": "pct_21d",
            "3 Months": "pct_63d",
            "1 Year": "pct_252d"
        }

        # Use a default or detect from available columns
        available_horizons = [col for col in horizon_map.values() if col in valid_metrics.columns]
        
        if available_horizons:
            # Use the first available horizon or get from sidebar selection
            current_horizon_col = available_horizons[2] if len(available_horizons) > 2 else available_horizons[0]  # Default to 5-day if available
            
            # Get top 20 performers
            top_20 = valid_metrics.nlargest(20, current_horizon_col)
            
            # Display summary
            horizon_name = [k for k, v in horizon_map.items() if v == current_horizon_col][0] if current_horizon_col in horizon_map.values() else "Selected Period"
            st.info(f"Showing top 20 performers over {horizon_name}")
            
            # Auto-populate comparison
            top_20_tickers = top_20['ticker'].tolist()
            top_20_comparison = get_sector_comparison_data(valid_metrics, top_20_tickers)
            
            if not top_20_comparison.empty:
                # Format display
                display_df = top_20_comparison.copy()
                display_df['last_close'] = display_df['last_close'].apply(lambda x: f"${x:.2f}")
                display_df['rank'] = range(1, len(display_df) + 1)
                
                # Reorder columns to show rank first
                cols = ['rank'] + [col for col in display_df.columns if col != 'rank']
                display_df = display_df[cols]
                
                for col in ['pct_1d', 'pct_5d', 'pct_21d', 'pct_252d', 'ann_vol_pct']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
                
                if 'rsi' in display_df.columns:
                    display_df['rsi'] = display_df['rsi'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Performance chart for top 20
                if current_horizon_col in top_20.columns:
                    fig = create_enhanced_bar_chart(
                        top_20.head(20), 'ticker', current_horizon_col,
                        f"Top 20 Performers ({horizon_name})", 'return'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sector breakdown of top performers
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
                    avg_return = top_20[current_horizon_col].mean()
                    median_return = top_20[current_horizon_col].median()
                    min_return = top_20[current_horizon_col].min()
                    max_return = top_20[current_horizon_col].max()
                    
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
            
            # Get market data for beta calculation (using SPY as proxy)
            market_returns = portfolio_returns.mean(axis=1)  # Simple market proxy
            
            metrics = {
                'Stock': stock,
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
    st.title("📈 S&P 500 Market Tracker")

    # Show data freshness info
    if os.path.exists(os.path.join(DATA_DIR, "latest_metrics.csv")):
        last_modified = os.path.getmtime(os.path.join(DATA_DIR, "latest_metrics.csv"))
        last_update = datetime.fromtimestamp(last_modified)
        
        # Convert to EST (subtract 5 hours)
        last_update_est = last_update - timedelta(hours=5)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Data last updated: {last_update_est.strftime('%Y-%m-%d at %I:%M %p EST')}")
        with col2:
            st.caption("Updates: 9AM & 5PM EST daily")
    else:
        st.error("No data available. Check GitHub Actions workflow.")

    # Check for existing data
    data_exists = os.path.exists(os.path.join(DATA_DIR, "latest_metrics.csv"))

    # Sidebar - Controls
    st.sidebar.header("⚙️ Settings")

    # 1. View mode
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Dashboard", "Sector Analysis", "Top Movers", "Technical Screener", 
        "Comparison Mode", "Historical Analysis", "Risk Management", "Data Export"],
        key="view_mode_radio"
    )

    # Load data to check available columns
    if data_exists:
        df_metrics = pd.read_csv(os.path.join(DATA_DIR, "latest_metrics.csv"))
        valid_metrics = df_metrics[df_metrics['status'] == 'ok'].copy()

    # Load historical data for advanced modes
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def load_historical_data():
            hist_dir = os.path.join(DATA_DIR, "history")
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
        hist = load_historical_data()
        
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
        value=7,
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
            consecutive_days = 7
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

            display_cols = ['ticker', 'sector', 'last_close', horizon_col, 'ann_vol_pct']
            if 'rsi' in valid_metrics.columns:
                display_cols.append('rsi')

            display_df = valid_metrics[display_cols].copy()
            styled_df = format_and_style_dataframe(display_df)
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
            st.subheader("🎯 S&P 500 Market Movers Analysis")
            
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
                    display_cols = ['ticker', 'sector', 'last_close', 'pct_1d', 'pct_3d', 'pct_5d']
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
                    display_cols = ['ticker', 'sector', 'last_close', 'pct_1d', 'pct_3d', 'pct_5d']
                    if 'pct_21d' in declining.columns:
                        display_cols.append('pct_21d')
                    # Filter to only existing columns
                    display_cols = [col for col in display_cols if col in declining.columns]
                    
                    declining_display = declining[display_cols].head(20)
                    styled_declining = format_and_style_dataframe(declining_display)
                    st.dataframe(styled_declining, use_container_width=True)

            with tabs[2]:
                if 'ann_vol_pct' in valid_metrics.columns:
                    most_volatile = valid_metrics.nlargest(20, 'ann_vol_pct')[['ticker', 'sector', 'last_close', 'ann_vol_pct', horizon_col]].copy()
                    st.metric("Highest Volatility Stock", f"{most_volatile.iloc[0]['ticker']} ({most_volatile.iloc[0]['ann_vol_pct']:.1f}%)")
                    
                    # Format the data
                    styled_volatile = format_and_style_dataframe(most_volatile)
                    st.dataframe(styled_volatile, use_container_width=True)
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
                display_cols = ['ticker', 'sector', 'last_close', horizon_col, 'ann_vol_pct']
                if 'rsi' in screened_df.columns:
                    display_cols.append('rsi')

                # Format the screener results
                screener_results = screened_df[display_cols]
                styled_screener = format_and_style_dataframe(screener_results)
                st.dataframe(styled_screener, use_container_width=True, height=400)

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

                styled_preview = format_and_style_dataframe(export_df.head(10))
                st.dataframe(styled_preview, use_container_width=True)
        # ---------- Comparison Mode ----------
        elif view_mode == "Comparison Mode":
            render_comparison_mode(valid_metrics, hist)

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