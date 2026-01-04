# Stock Market Tracker

A comprehensive market tracking tool with both CLI and Streamlit web interfaces for S&P 500, NASDAQ-100, and Dow Jones 30 analysis, portfolio management, and risk assessment.

## Features

### Core Analysis
- Real-time S&P 500, NASDAQ-100, and Dow Jones 30 data tracking with automated updates
- Technical analysis indicators (RSI, Moving Averages, MACD, Bollinger Bands)
- Sector performance analysis with color-coded visualizations
- Advanced stock screener with multiple filters
- Historical data analysis with custom date ranges

### Advanced Trading Modes
- **Comparison Mode**: Side-by-side analysis of 2-10 stocks with correlation analysis
- **Historical Analysis**: Deep time-series analysis with drawdown and seasonal patterns
- **Risk Management**: Portfolio risk assessment with VaR calculations and stress testing
- **Top Performers**: Automated identification of best performing stocks

### Data & Export
- Automated data updates (9 AM & 5 PM EST daily)
- CSV export capabilities for all analysis
- Color-coded performance indicators for quick visual assessment
- Real-time portfolio valuation and scenario analysis

## Installation
```bash
git clone https://github.com/your-username/market-tracker.git
cd market-tracker
pip install -r requirements.txt
```
## Web Interface Modes
- **Dashboard**
  - Market overview with rising/declining stock counts
  - Top gainers and losers visualization
  - Sector performance heatmap
  - Color-coded performance tables

- **Sector Analysis**
  - Comprehensive sector breakdown with statistics
  - Average returns, volatility, and stock counts by sector
  - Interactive sector performance charts

- **Top Movers**
  - Rising and declining stocks with consecutive day tracking
  - Most volatile stocks analysis
  - 52-week highs and lows identification

- **Technical Screener**
  - Advanced filtering by price, volatility, RSI, and returns
  - Sector-based filtering
  - Risk vs return scatter plots
  - Individual stock technical charts

- **Sentiment Analysis**
  - News sentiment analysis for all stocks using NLP
  - Mixed source strategy: NewsAPI (premium) + Google News RSS (free)
  - 100% free mode with Google News (no API key required)
  - Sentiment vs performance correlation analysis
  - Sector sentiment breakdown and heatmaps
  - Top positive/negative stocks identification
  - Data source quality comparison
  - Exportable sentiment data with article counts

- **Comparison Mode**
  - Select and compare 2-15 stocks side-by-side
  - Performance correlation analysis
  - Risk vs return visualization
  - Top 20 performers auto-selection

- **Historical Analysis**
  - Custom date range analysis
  - Drawdown analysis and maximum drawdown calculation
  - Seasonal pattern identification
  - Return distribution analysis with statistics

- **Risk Management**
  - Portfolio construction with customizable weights
  - Value at Risk (VaR) calculations at multiple confidence levels
  - Beta and Alpha calculations vs market
  - Stress testing with market crash scenarios
  - Portfolio correlation analysis

- **Data Export**
  - Download all metrics in CSV format
  - Custom filtering for exports
  - Index-specific data files
  - Rising/declining stock lists

## Automated Data Updates
The system includes GitHub Actions automation:
- **Schedule:** 9:00 AM and 5:00 PM EST daily
- **Process:** Automatically fetches latest market data and commits to repository

## Dependencies
```
streamlit
pandas
numpy
yfinance
plotly
requests
beautifulsoup4
tqdm
python-dateutil
scipy
```

## Data Sources
- **Stock Data**: Yahoo Finance
- **S&P 500 Constituents:** Wikipedia (automatically updated)
- **Technical Indicators:** Calculated from price data
- **Risk Metrics:** Computed using scipy statistical functions

## File Structure
```
market-tracker/
├── market_tracker.py           # Main application file
├── requirements.txt            # Python dependencies
├── README.md                   
├── .github/workflows/          # GitHub Actions automation
│   └── update-data.yml
└── data/                       # Generated data files
    ├── latest_metrics.csv      # Main analysis results
    ├── sp500_constituents_snapshot.csv
    └── history/                # Individual stock history
        ├── AAPL.csv
        ├── MSFT.csv
        └── ...
```
## Risk Disclaimer
This tool is for informational and educational purposes only. It should not be considered as financial advice. All data is provided by Yahoo Finance and may be delayed. Users should:

- Conduct their own research before making investment decisions
- Understand that past performance does not guarantee future results
- Consider consulting with financial professionals for investment advice
- Be aware that all investments carry risk of loss

### Author
Created by Ehiremen Nathaniel Omoarebun

**Live Demo:** https://eno-market-tracker.streamlit.app