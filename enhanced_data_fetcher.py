# ENHANCED DATA FETCHER
# Comprehensive fundamental, institutional, and forward-looking data

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import time
from tqdm import tqdm


# FORWARD-LOOKING FUNDAMENTAL DATA
def fetch_forward_fundamentals(ticker: str) -> Dict:
    """
    Fetch forward-looking fundamental data

    FREE from yfinance:
    - Earnings dates and beat rate
    - Forward P/E
    - Revenue growth
    - Profit margins
    - Cash flow metrics
    - Balance sheet health
    
    Returns:
        Dictionary with fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # EARNINGS ANALYSIS
        earnings_data = {}
        
        try:
            # Earnings calendar
            calendar = stock.calendar
            
            # Handle both DataFrame and dict returns from yfinance
            if calendar is not None:
                # Check if it's a DataFrame
                if hasattr(calendar, 'empty') and not calendar.empty:
                    earnings_dates = calendar.get('Earnings Date', [])
                    if earnings_dates and len(earnings_dates) > 0:
                        next_earnings = pd.to_datetime(earnings_dates[0])
                        days_to_earnings = (next_earnings - pd.Timestamp.now()).days
                        earnings_data['days_to_earnings'] = days_to_earnings
                        earnings_data['next_earnings_date'] = next_earnings.strftime('%Y-%m-%d')
                
                # Check if it's a dict
                elif isinstance(calendar, dict):
                    earnings_date = calendar.get('Earnings Date', None)
                    if earnings_date:
                        # Handle list of dates
                        if isinstance(earnings_date, list) and len(earnings_date) > 0:
                            next_earnings = pd.to_datetime(earnings_date[0])
                        else:
                            next_earnings = pd.to_datetime(earnings_date)
                        
                        days_to_earnings = (next_earnings - pd.Timestamp.now()).days
                        earnings_data['days_to_earnings'] = days_to_earnings
                        earnings_data['next_earnings_date'] = next_earnings.strftime('%Y-%m-%d')
            
            # Earnings history (beat rate)
            earnings_history = stock.earnings_history
            if earnings_history is not None and not earnings_history.empty:
                # Calculate beat rate
                beats = (earnings_history['epsActual'] > earnings_history['epsEstimate']).sum()
                total = len(earnings_history)
                earnings_data['earnings_beat_rate'] = beats / total if total > 0 else 0.5
                earnings_data['earnings_history_count'] = total
                
                # Recent surprise
                if len(earnings_history) > 0:
                    latest = earnings_history.iloc[0]
                    if pd.notna(latest['epsActual']) and pd.notna(latest['epsEstimate']) and latest['epsEstimate'] != 0:
                        surprise_pct = ((latest['epsActual'] - latest['epsEstimate']) / 
                                      abs(latest['epsEstimate']) * 100)
                        earnings_data['last_earnings_surprise_pct'] = surprise_pct
            else:
                earnings_data['earnings_beat_rate'] = 0.5
                earnings_data['earnings_history_count'] = 0
                
        except Exception as e:
            # Silently handle earnings fetch failures
            earnings_data = {
                'days_to_earnings': None,
                'earnings_beat_rate': 0.5,
                'earnings_history_count': 0
            }
        
        # VALUATION METRICS
        valuation = {
            'forward_pe': info.get('forwardPE'),
            'trailing_pe': info.get('trailingPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'enterprise_value': info.get('enterpriseValue'),
            'ev_to_revenue': info.get('enterpriseToRevenue'),
            'ev_to_ebitda': info.get('enterpriseToEbitda')
        }
        
        # GROWTH METRICS 
        growth = {
            'revenue_growth': info.get('revenueGrowth'),  # YoY
            'earnings_growth': info.get('earningsGrowth'),  # YoY
            'revenue_per_share': info.get('revenuePerShare'),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth')
        }
        
        # Try to get quarterly financials for better growth metrics
        try:
            quarterly_financials = stock.quarterly_financials
            if quarterly_financials is not None and not quarterly_financials.empty:
                # Revenue growth (QoQ)
                if 'Total Revenue' in quarterly_financials.index:
                    revenues = quarterly_financials.loc['Total Revenue']
                    if len(revenues) >= 2:
                        recent_revenue = revenues.iloc[0]
                        prev_revenue = revenues.iloc[1]
                        if prev_revenue != 0:
                            growth['revenue_growth_qoq'] = ((recent_revenue - prev_revenue) / 
                                                           abs(prev_revenue) * 100)
        except:
            pass
        
        # PROFITABILITY METRICS
        profitability = {
            'profit_margins': info.get('profitMargins'),
            'gross_margins': info.get('grossMargins'),
            'operating_margins': info.get('operatingMargins'),
            'ebitda_margins': info.get('ebitdaMargins'),
            'return_on_assets': info.get('returnOnAssets'),
            'return_on_equity': info.get('returnOnEquity')
        }
        
        # CASH FLOW METRICS
        cash_flow = {
            'free_cash_flow': info.get('freeCashflow'),
            'operating_cash_flow': info.get('operatingCashflow'),
            'fcf_per_share': None  # Calculate if possible
        }
        
        # Calculate FCF per share
        if cash_flow['free_cash_flow'] and info.get('sharesOutstanding'):
            cash_flow['fcf_per_share'] = (cash_flow['free_cash_flow'] / 
                                         info.get('sharesOutstanding'))
        
        # BALANCE SHEET HEALTH
        balance_sheet = {
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'book_value_per_share': info.get('bookValue')
        }
        
        # Interest coverage (EBITDA / Interest Expense)
        ebitda = info.get('ebitda')
        interest_expense = info.get('interestExpense')
        if ebitda and interest_expense and interest_expense != 0:
            balance_sheet['interest_coverage'] = ebitda / abs(interest_expense)
        
        # COMBINE ALL
        fundamental_data = {
            **earnings_data,
            **valuation,
            **growth,
            **profitability,
            **cash_flow,
            **balance_sheet
        }
        
        return fundamental_data
        
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return {}

# INSTITUTIONAL AND INSIDER DATA
def fetch_institutional_data(ticker: str) -> Dict:
    """
    Fetch institutional ownership and insider trading data
    
    FREE from yfinance:
    - Institutional ownership %
    - Number of institutions
    - Short interest
    - Insider transactions (limited)
    
    Returns:
        Dictionary with institutional metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        institutional = {
            'institutional_ownership_pct': info.get('heldPercentInstitutions'),
            'insider_ownership_pct': info.get('heldPercentInsiders'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'float_shares': info.get('floatShares'),
            'shares_short': info.get('sharesShort'),
            'short_percent_of_float': info.get('shortPercentOfFloat'),
            'short_ratio': info.get('shortRatio'),  # Days to cover
            'shares_short_prior_month': info.get('sharesShortPriorMonth')
        }
        
        # Calculate short interest change
        if (institutional['shares_short'] and 
            institutional['shares_short_prior_month'] and
            institutional['shares_short_prior_month'] != 0):
            short_change = ((institutional['shares_short'] - 
                           institutional['shares_short_prior_month']) /
                          institutional['shares_short_prior_month'] * 100)
            institutional['short_interest_change_pct'] = short_change
        
        # Try to get insider transactions (limited free data)
        try:
            insider_trades = stock.insider_transactions
            if insider_trades is not None and not insider_trades.empty:
                # Last 6 months
                recent_cutoff = datetime.now() - timedelta(days=180)
                recent_trades = insider_trades[
                    pd.to_datetime(insider_trades['Start Date']) >= recent_cutoff
                ]
                
                if not recent_trades.empty:
                    # Count buys vs sells
                    buys = recent_trades[recent_trades['Transaction'] == 'Buy']
                    sells = recent_trades[recent_trades['Transaction'] == 'Sale']
                    
                    buy_value = buys['Value'].sum() if 'Value' in buys.columns else 0
                    sell_value = sells['Value'].sum() if 'Value' in sells.columns else 0
                    
                    institutional['insider_buys_6m'] = len(buys)
                    institutional['insider_sells_6m'] = len(sells)
                    institutional['insider_buy_value_6m'] = buy_value
                    institutional['insider_sell_value_6m'] = sell_value
                    
                    # Calculate ratio
                    if len(sells) > 0:
                        institutional['insider_buy_sell_ratio'] = len(buys) / len(sells)
                    else:
                        institutional['insider_buy_sell_ratio'] = len(buys) if len(buys) > 0 else 1.0
                else:
                    institutional['insider_buys_6m'] = 0
                    institutional['insider_sells_6m'] = 0
                    institutional['insider_buy_sell_ratio'] = 1.0
        except:
            institutional['insider_buys_6m'] = 0
            institutional['insider_sells_6m'] = 0
            institutional['insider_buy_sell_ratio'] = 1.0
        
        # Try to get institutional holders (top holders)
        try:
            inst_holders = stock.institutional_holders
            if inst_holders is not None and not inst_holders.empty:
                institutional['num_institutions'] = len(inst_holders)
                
                # Calculate concentration (top 5 holders %)
                if 'Shares' in inst_holders.columns and institutional['shares_outstanding']:
                    top5_shares = inst_holders.head(5)['Shares'].sum()
                    institutional['top5_institutional_pct'] = (
                        top5_shares / institutional['shares_outstanding'] * 100
                    )
        except:
            pass
        
        return institutional
        
    except Exception as e:
        print(f"Error fetching institutional data for {ticker}: {e}")
        return {}

# ANALYST AND RECOMMENDATION DATA
def fetch_analyst_data(ticker: str) -> Dict:
    """
    Fetch analyst recommendations and price targets
    
    FREE from yfinance (limited):
    - Analyst recommendations
    - Price targets
    - Recommendation trends
    
    Returns:
        Dictionary with analyst metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        analyst = {
            'analyst_target_price': info.get('targetMeanPrice'),
            'analyst_target_high': info.get('targetHighPrice'),
            'analyst_target_low': info.get('targetLowPrice'),
            'num_analyst_opinions': info.get('numberOfAnalystOpinions'),
            'recommendation_mean': info.get('recommendationMean'),  # 1=Strong Buy, 5=Sell
            'recommendation_key': info.get('recommendationKey')
        }
        
        # Calculate upside to target
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if current_price and analyst['analyst_target_price']:
            analyst['upside_to_target_pct'] = (
                (analyst['analyst_target_price'] - current_price) / current_price * 100
            )
        
        # Try to get recommendation trend
        try:
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                # Recent recommendations (last 3 months)
                recent_cutoff = datetime.now() - timedelta(days=90)
                recent_recs = recommendations[
                    recommendations.index >= recent_cutoff
                ]
                
                if not recent_recs.empty:
                    # Count upgrades vs downgrades
                    if 'To Grade' in recent_recs.columns and 'From Grade' in recent_recs.columns:
                        upgrade_keywords = ['buy', 'outperform', 'overweight']
                        downgrade_keywords = ['sell', 'underperform', 'underweight']
                        
                        upgrades = 0
                        downgrades = 0
                        
                        for _, row in recent_recs.iterrows():
                            to_grade = str(row.get('To Grade', '')).lower()
                            from_grade = str(row.get('From Grade', '')).lower()
                            
                            # Check if it's an upgrade
                            if (any(word in to_grade for word in upgrade_keywords) and
                                not any(word in from_grade for word in upgrade_keywords)):
                                upgrades += 1
                            # Check if it's a downgrade
                            elif (any(word in to_grade for word in downgrade_keywords) and
                                  not any(word in from_grade for word in downgrade_keywords)):
                                downgrades += 1
                        
                        analyst['upgrades_3m'] = upgrades
                        analyst['downgrades_3m'] = downgrades
                        analyst['net_upgrades_3m'] = upgrades - downgrades
        except:
            pass
        
        return analyst
        
    except Exception as e:
        print(f"Error fetching analyst data for {ticker}: {e}")
        return {}

# COMPREHENSIVE FETCH FUNCTION
def fetch_all_enhanced_data(ticker: str, verbose: bool = False, max_retries: int = 3) -> Dict:
    """
    Fetch all enhanced data for a single ticker with retry logic
    
    Args:
        ticker: Stock ticker symbol
        verbose: Print progress
        max_retries: Maximum retry attempts for failed requests
    
    Returns:
        Dictionary with all enhanced metrics
    """
    if verbose:
        print(f"Fetching enhanced data for {ticker}...")
    
    all_data = {'ticker': ticker}
    
    def fetch_with_retry(fetch_func, category_name):
        """Helper function to fetch with exponential backoff"""
        for attempt in range(max_retries):
            try:
                data = fetch_func(ticker)
                if verbose:
                    print(f"  ✓ {category_name} fetched")
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    if verbose:
                        print(f"  ⚠️ {category_name} failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if verbose:
                        print(f"  ✗ {category_name} failed after {max_retries} attempts: {e}")
                    return {}
    
    # Fetch each category with retry logic
    fundamentals = fetch_with_retry(fetch_forward_fundamentals, "Fundamentals")
    all_data.update(fundamentals)
    time.sleep(0.3)  # Increased base rate limit
    
    institutional = fetch_with_retry(fetch_institutional_data, "Institutional")
    all_data.update(institutional)
    time.sleep(0.3)
    
    analyst = fetch_with_retry(fetch_analyst_data, "Analyst")
    all_data.update(analyst)
    
    return all_data

def fetch_enhanced_data_batch(tickers: List[str], 
                              verbose: bool = True,
                              fail_threshold: float = 0.3) -> pd.DataFrame:
    """
    Fetch enhanced data for multiple tickers with failure tracking
    
    Args:
        tickers: List of ticker symbols
        verbose: Show progress bar
        fail_threshold: Stop if this % of requests fail (0.3 = 30%)
    
    Returns:
        DataFrame with enhanced metrics
    """
    all_data = []
    failed_count = 0
    total_processed = 0
    
    iterator = tqdm(tickers, desc="Fetching enhanced data") if verbose else tickers
    
    for ticker in iterator:
        total_processed += 1
        data = fetch_all_enhanced_data(ticker, verbose=False, max_retries=3)
        
        # Track failures
        if not data or len(data) <= 1:  # Only has 'ticker' key = failed
            failed_count += 1
        
        all_data.append(data)
        
        # Check if we're hitting rate limits (too many failures)
        if total_processed >= 10:  # After 10 requests, check failure rate
            failure_rate = failed_count / total_processed
            if failure_rate > fail_threshold:
                print(f"\n⚠️ WARNING: {failure_rate:.0%} failure rate detected!")
                print(f"   This may indicate rate limiting or network issues.")
                print(f"   Increasing delay between requests...")
                time.sleep(2)  # Longer delay when rate limited
            else:
                time.sleep(0.5)  # Normal rate limiting
        else:
            time.sleep(0.5)
    
    # Summary
    if verbose:
        success_count = total_processed - failed_count
        print(f"\n✅ Successfully fetched: {success_count}/{total_processed} ({success_count/total_processed:.0%})")
        if failed_count > 0:
            print(f"⚠️ Failed: {failed_count}/{total_processed} ({failed_count/total_processed:.0%})")
    
    return pd.DataFrame(all_data)

# INTEGRATION WITH EXISTING METRICS
def merge_enhanced_with_base_metrics(base_metrics_df: pd.DataFrame,
                                    enhanced_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge enhanced data with existing base metrics
    
    Args:
        base_metrics_df: Your existing metrics DataFrame
        enhanced_df: Enhanced data from fetch_enhanced_data_batch
    
    Returns:
        Merged DataFrame
    """
    merged = base_metrics_df.merge(
        enhanced_df,
        on='ticker',
        how='left',
        suffixes=('', '_enhanced')
    )
    
    return merged

