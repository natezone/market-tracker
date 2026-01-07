# STOCK RECOMMENDATION ENGINE
# Multi-factor scoring system using technical, fundamental, and sentiment data

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

# RECOMMENDATION METHODOLOGY
"""
STOCK RECOMMENDATION SCORING METHODOLOGY
=========================================

This system uses a WEIGHTED MULTI-FACTOR MODEL that combines:

1. **Technical Analysis** (30% weight)
   - Momentum indicators (RSI, price trends)
   - Volume analysis
   - 52-week positioning

2. **Fundamental Analysis** (40% weight)
   - Valuation metrics (P/E ratio)
   - Growth metrics (returns over time)
   - Consistency (trend stability)

3. **Sentiment Analysis** (20% weight)
   - News sentiment
   - Article volume
   - Sentiment trend

4. **Risk Assessment** (10% weight)
   - Volatility
   - Drawdown risk
   - Sector correlation

SCORING SYSTEM:
- Each factor scored 0-100
- Weighted combination → Final Score (0-100)
- Risk-adjusted using volatility
- Categorized into BUY/HOLD/SELL recommendations

INVESTMENT STRATEGIES:
- Growth: High momentum + high sentiment
- Value: Low P/E + undervalued + positive trend
- Momentum: Strong recent performance + volume
- Contrarian: Oversold + positive fundamentals
- Balanced: Best overall risk-adjusted scores
"""

import pandas as pd
import numpy as np

def safe_get(row, column, default=0):
    """Safely get a value from a row, return default if None or missing"""
    try:
        # Try multiple column name variations
        variations = [column]
        
        # Add 52-week column variations
        if column == '52w_high':
            variations.append('high_52w')
        elif column == '52w_low':
            variations.append('low_52w')
        elif column == 'high_52w':
            variations.append('52w_high')
        elif column == 'low_52w':
            variations.append('52w_low')
        
        for col_name in variations:
            if col_name in row:
                val = row.get(col_name)
                if val is not None and not pd.isna(val):
                    return float(val)
        
        return default
        
    except Exception as e:
        return default
    
# CONFIGURATION AND ENUMS
class RecommendationLevel(Enum):
    """Recommendation levels"""
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"

class InvestmentStrategy(Enum):
    """Investment strategy types"""
    GROWTH = "growth"
    VALUE = "value"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"
    BALANCED = "balanced"
    DIVIDEND = "dividend"  

class TimeHorizon(Enum):
    """Investment time horizons"""
    SWING_TRADE = "Swing (1-4 weeks)"
    SHORT_TERM = "Short-term (1-3 months)"
    MEDIUM_TERM = "Medium-term (3-12 months)"
    LONG_TERM = "Long-term (1+ years)"

class CatalystType(Enum):
    """Types of potential catalysts"""
    EARNINGS_SOON = "Earnings in next 2 weeks"
    ANALYST_UPGRADE = "Recent analyst upgrade"
    BREAKING_RESISTANCE = "Breaking 52-week high"
    OVERSOLD_BOUNCE = "Oversold RSI reversal"
    INSIDER_BUYING = "Strong insider buying"
    SHORT_SQUEEZE_SETUP = "Potential short squeeze"
    EARNINGS_BEAT = "Strong earnings beat history"
    SECTOR_LEADER = "Sector outperformer"
    
@dataclass
class ScoringWeights:
    """Configurable weights for different factors"""
    technical: float = 0.30
    fundamental: float = 0.40
    sentiment: float = 0.20
    risk: float = 0.10
    
    def validate(self):
        """Ensure weights sum to 1.0"""
        total = self.technical + self.fundamental + self.sentiment + self.risk
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

@dataclass
class RecommendationThresholds:
    """Score thresholds for recommendation levels"""
    strong_buy: float = 80.0
    buy: float = 65.0
    hold_upper: float = 50.0
    hold_lower: float = 35.0
    sell: float = 20.0


# MARKET REGIME AND CONFIDENCE SYSTEM
class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "Bull Market"
    NEUTRAL = "Neutral Market"
    BEAR = "Bear Market"
    UNKNOWN = "Unknown"

class ConfidenceLevel(Enum):
    """Confidence in recommendation"""
    HIGH = "High Confidence"
    MEDIUM = "Medium Confidence"
    LOW = "Low Confidence"

@dataclass
class MarketRegimeData:
    """Market regime classification data"""
    regime: MarketRegime
    pct_21d: float
    pct_63d: float
    volatility: float
    breadth: float  # % stocks above MA
    confidence: float  # Confidence in regime classification
    
    def __str__(self):
        return f"{self.regime.value} (Confidence: {self.confidence:.0%})"

@dataclass
class RegimeAdjustmentMultipliers:
    """Regime-based score adjustments by strategy"""
    bull: float
    neutral: float
    bear: float
    
    @classmethod
    def for_strategy(cls, strategy: InvestmentStrategy):
        """Get multipliers for specific strategy"""
        multipliers = {
            InvestmentStrategy.GROWTH: cls(bull=1.05, neutral=1.00, bear=0.85),
            InvestmentStrategy.MOMENTUM: cls(bull=1.05, neutral=0.95, bear=0.80),
            InvestmentStrategy.BALANCED: cls(bull=1.00, neutral=1.00, bear=0.90),
            InvestmentStrategy.VALUE: cls(bull=0.95, neutral=1.05, bear=1.05),
            InvestmentStrategy.CONTRARIAN: cls(bull=0.90, neutral=1.00, bear=1.10),
            InvestmentStrategy.DIVIDEND: cls(bull=0.95, neutral=1.00, bear=1.05),
        }
        return multipliers.get(strategy, cls(bull=1.00, neutral=1.00, bear=1.00))
    
    def get_multiplier(self, regime: MarketRegime) -> float:
        """Get multiplier for specific regime"""
        if regime == MarketRegime.BULL:
            return self.bull
        elif regime == MarketRegime.BEAR:
            return self.bear
        else:
            return self.neutral

# MARKET REGIME DETECTION
def detect_market_regime(index_metrics: pd.Series) -> MarketRegimeData:
    """
    Detect current market regime using index-level data
    
    Args:
        index_metrics: Series with market index data
            Required fields: pct_21d, pct_63d, ann_vol_pct
            Optional: breadth (% stocks above 50 DMA)
    
    Returns:
        MarketRegimeData with regime classification
    """
    pct_21d = index_metrics.get('pct_21d', 0)
    pct_63d = index_metrics.get('pct_63d', 0)
    volatility = index_metrics.get('ann_vol_pct', 25.0)
    breadth = index_metrics.get('breadth', 50.0)  # Default to neutral
    
    # Regime classification logic
    regime = MarketRegime.NEUTRAL
    confidence = 0.5
    
    # BULL MARKET: Strong uptrend, low volatility
    if pct_21d > 3 and pct_63d > 8 and volatility < 25:
        regime = MarketRegime.BULL
        confidence = min(0.95, 0.6 + (pct_63d / 20) + ((25 - volatility) / 50))
    
    # BEAR MARKET: Significant decline or high volatility
    elif pct_63d < -5 or volatility > 35:
        regime = MarketRegime.BEAR
        confidence = min(0.95, 0.6 + (abs(pct_63d) / 20) + ((volatility - 25) / 50))
    
    # NEUTRAL: Mixed signals
    else:
        regime = MarketRegime.NEUTRAL
        # Lower confidence in neutral (harder to classify)
        confidence = 0.5 + (breadth - 50) / 200  # Slight adjustment based on breadth
    
    return MarketRegimeData(
        regime=regime,
        pct_21d=pct_21d,
        pct_63d=pct_63d,
        volatility=volatility,
        breadth=breadth,
        confidence=max(0.3, min(0.95, confidence))
    )

def calculate_market_breadth(metrics_df: pd.DataFrame) -> float:
    """
    Calculate market breadth (% stocks above 50-day MA)
    
    Args:
        metrics_df: DataFrame with stock metrics
    
    Returns:
        Percentage of stocks above their 50-day average (0-100)
    """
    if 'avg_1y' not in metrics_df.columns or 'last_close' not in metrics_df.columns:
        return 50.0  # Default to neutral
    
    valid_stocks = metrics_df[
        (metrics_df['status'] == 'ok') &
        (metrics_df['avg_1y'].notna()) &
        (metrics_df['last_close'].notna())
    ]
    
    if len(valid_stocks) == 0:
        return 50.0
    
    above_ma = (valid_stocks['last_close'] > valid_stocks['avg_1y']).sum()
    breadth = (above_ma / len(valid_stocks)) * 100
    
    return breadth

# SCORING FUNCTIONS
def score_rsi(rsi_value: float) -> float:
    """
    Score RSI indicator (Relative Strength Index)
    
    RSI Interpretation:
    - 0-30: Oversold (potential buy) → High score
    - 30-70: Neutral → Medium score
    - 70-100: Overbought (potential sell) → Low score
    
    Returns: Score 0-100
    """
    if pd.isna(rsi_value):
        return 50.0  # Neutral if missing
    
    if rsi_value <= 30:
        # Oversold - excellent buying opportunity
        return 85.0 + (30 - rsi_value) / 2  # 85-100
    elif rsi_value <= 40:
        # Slightly oversold - good
        return 70.0 + (40 - rsi_value) * 1.5  # 70-85
    elif rsi_value <= 60:
        # Neutral zone
        return 50.0 + (50 - abs(rsi_value - 50)) * 0.4  # 42-58
    elif rsi_value <= 70:
        # Slightly overbought - caution
        return 40.0 - (rsi_value - 60) * 3  # 10-40
    else:
        # Overbought - avoid
        return max(0, 30.0 - (rsi_value - 70))  # 0-30

def score_momentum(pct_5d: float, pct_21d: float, pct_63d: float) -> float:
    """
    Score price momentum across multiple timeframes
    
    Methodology:
    - Recent momentum (5d): 40% weight
    - Medium-term (21d): 35% weight
    - Long-term (63d): 25% weight
    - Penalize extreme volatility
    
    Returns: Score 0-100
    """
    def normalize_return(ret: float, max_expected: float = 10.0) -> float:
        """Normalize return to 0-100 scale"""
        if pd.isna(ret):
            return 50.0
        # Cap extreme values
        ret = np.clip(ret, -50, 50)
        # Scale: -50% → 0, 0% → 50, +50% → 100
        return 50 + (ret / max_expected * 50)
    
    score_5d = normalize_return(pct_5d, max_expected=5.0)
    score_21d = normalize_return(pct_21d, max_expected=10.0)
    score_63d = normalize_return(pct_63d, max_expected=20.0)
    
    # Weighted combination
    momentum_score = (
        score_5d * 0.40 +
        score_21d * 0.35 +
        score_63d * 0.25
    )
    
    return momentum_score

def score_pe_ratio(pe_ratio: float) -> float:
    """
    Score P/E ratio for valuation
    
    P/E Interpretation:
    - < 10: Undervalued → High score
    - 10-15: Fair value → Good score
    - 15-25: Average → Medium score
    - 25-40: Expensive → Low score
    - > 40: Overvalued → Very low score
    
    Returns: Score 0-100
    """
    if pd.isna(pe_ratio) or pe_ratio <= 0:
        return 50.0  # Neutral if missing or negative
    
    if pe_ratio < 10:
        return 95.0  # Excellent value
    elif pe_ratio < 15:
        return 85.0  # Good value
    elif pe_ratio < 20:
        return 70.0  # Fair value
    elif pe_ratio < 25:
        return 55.0  # Slightly expensive
    elif pe_ratio < 30:
        return 40.0  # Expensive
    elif pe_ratio < 40:
        return 25.0  # Very expensive
    else:
        return max(0, 20.0 - (pe_ratio - 40) * 0.5)  # Extremely overvalued

def score_52_week_position(pct_from_high: float, pct_from_low: float) -> float:
    """
    Score position relative to 52-week range
    
    Ideal position: 20-40% from highs (room to grow, not at bottom)
    
    Returns: Score 0-100
    """
    if pd.isna(pct_from_high):
        return 50.0
    
    # Convert to distance from low (0-100 scale)
    if not pd.isna(pct_from_low):
        position_pct = pct_from_low / (pct_from_low - pct_from_high) * 100
    else:
        # Approximate from high
        position_pct = 100 + pct_from_high
    
    # Optimal is 60-80% up from low (20-40% from high)
    if 60 <= position_pct <= 80:
        return 90.0  # Ideal position
    elif 40 <= position_pct < 60:
        return 75.0  # Good position (some value left)
    elif 80 < position_pct <= 90:
        return 70.0  # Near highs but not overextended
    elif 20 <= position_pct < 40:
        return 60.0  # Lower but establishing base
    elif position_pct > 90:
        return 40.0  # Too close to highs
    else:
        return 50.0  # Near lows (risky)

def score_volume_trend(volume_vs_avg: float) -> float:
    """
    Score volume relative to average
    
    Interpretation:
    - High volume with price increase = Bullish
    - Low volume = Lack of conviction
    
    Returns: Score 0-100
    """
    if pd.isna(volume_vs_avg):
        return 50.0
    
    if volume_vs_avg > 50:
        return 85.0  # Very high volume (strong conviction)
    elif volume_vs_avg > 20:
        return 75.0  # High volume
    elif volume_vs_avg > 0:
        return 60.0  # Above average
    elif volume_vs_avg > -20:
        return 45.0  # Slightly below average
    elif volume_vs_avg > -40:
        return 30.0  # Low volume
    else:
        return 20.0  # Very low volume (lack of interest)

def score_volatility(ann_vol_pct: float, target_vol: float = 25.0) -> float:
    """
    Score volatility (lower is better for risk-averse)
    
    Target volatility: 25% (market average)
    
    Returns: Score 0-100
    """
    if pd.isna(ann_vol_pct):
        return 50.0
    
    if ann_vol_pct < 15:
        return 90.0  # Very low volatility
    elif ann_vol_pct < 25:
        return 75.0  # Low volatility
    elif ann_vol_pct < 35:
        return 60.0  # Moderate volatility
    elif ann_vol_pct < 50:
        return 40.0  # High volatility
    else:
        return max(0, 30.0 - (ann_vol_pct - 50) * 0.5)  # Very high volatility

# CONFIDENCE SCORING
def calculate_confidence_score(stock_data: pd.Series, 
                               score_components: Dict) -> Tuple[float, ConfidenceLevel]:
    """
    Calculate confidence in recommendation based on data quality and consistency
    
    Factors:
    1. Data completeness (missing PE, sentiment, etc.)
    2. Sentiment article count
    3. Volatility (high vol = lower confidence)
    4. Score component dispersion (consistent scores = higher confidence)
    
    Returns:
        Tuple of (confidence_score 0-1, ConfidenceLevel enum)
    """
    confidence_factors = []
    
    # Factor 1: Data Completeness (0-1)
    required_fields = ['pe_ratio', 'rsi', 'pct_21d', 'pct_63d', 'ann_vol_pct']
    optional_fields = ['sentiment_score', 'article_count', 'volume_vs_avg']
    
    required_present = sum(1 for field in required_fields 
                          if not pd.isna(stock_data.get(field)))
    optional_present = sum(1 for field in optional_fields 
                          if not pd.isna(stock_data.get(field)))
    
    data_completeness = (
        (required_present / len(required_fields)) * 0.7 +
        (optional_present / len(optional_fields)) * 0.3
    )
    confidence_factors.append(data_completeness)
    
    # Factor 2: Sentiment Data Quality (0-1)
    article_count = stock_data.get('article_count', 0)
    if article_count >= 10:
        sentiment_quality = 1.0
    elif article_count >= 5:
        sentiment_quality = 0.85
    elif article_count >= 2:
        sentiment_quality = 0.65
    elif article_count >= 1:
        sentiment_quality = 0.5
    else:
        sentiment_quality = 0.3  # No sentiment data
    
    confidence_factors.append(sentiment_quality)
    
    # Factor 3: Volatility Confidence (0-1)
    # Lower volatility = higher confidence
    volatility = stock_data.get('ann_vol_pct', 25.0)
    if volatility < 20:
        vol_confidence = 1.0
    elif volatility < 30:
        vol_confidence = 0.85
    elif volatility < 40:
        vol_confidence = 0.65
    elif volatility < 60:
        vol_confidence = 0.45
    else:
        vol_confidence = 0.25
    
    confidence_factors.append(vol_confidence)
    
    # Factor 4: Score Consistency (0-1)
    # Lower dispersion in component scores = higher confidence
    scores = [
        score_components.get('technical_score', 50),
        score_components.get('fundamental_score', 50),
        score_components.get('sentiment_score', 50),
        score_components.get('risk_score', 50)
    ]
    
    score_std = np.std(scores)
    # Low std (consistent) = high confidence
    if score_std < 10:
        consistency_confidence = 1.0
    elif score_std < 20:
        consistency_confidence = 0.8
    elif score_std < 30:
        consistency_confidence = 0.6
    else:
        consistency_confidence = 0.4
    
    confidence_factors.append(consistency_confidence)
    
    # Weighted combination
    confidence_score = (
        data_completeness * 0.25 +
        sentiment_quality * 0.25 +
        vol_confidence * 0.30 +
        consistency_confidence * 0.20
    )
    
    # Classify confidence level
    if confidence_score >= 0.75:
        confidence_level = ConfidenceLevel.HIGH
    elif confidence_score >= 0.55:
        confidence_level = ConfidenceLevel.MEDIUM
    else:
        confidence_level = ConfidenceLevel.LOW
    
    return confidence_score, confidence_level

def score_sentiment(sentiment_score: float, article_count: int) -> float:
    """
    Score news sentiment
    
    Combines:
    - Sentiment polarity (-1 to +1)
    - Article volume (confidence in sentiment)
    
    Returns: Score 0-100
    """
    if pd.isna(sentiment_score):
        return 50.0  # Neutral if missing
    
    # Base score from sentiment (-1 to +1 → 0 to 100)
    base_score = (sentiment_score + 1) * 50
    
    # Confidence adjustment based on article count
    if article_count >= 10:
        confidence = 1.0  # High confidence
    elif article_count >= 5:
        confidence = 0.85
    elif article_count >= 2:
        confidence = 0.7
    else:
        confidence = 0.5  # Low confidence, regress to neutral
    
    return base_score * confidence + 50 * (1 - confidence)

def score_trend_consistency(rising_3d: bool, rising_7d: bool, rising_14d: bool, 
                           rising_21d: bool) -> float:
    """
    Score trend consistency across multiple timeframes
    
    Consistent uptrend = Higher score
    Mixed signals = Lower score
    
    Returns: Score 0-100
    """
    # Count rising periods
    rising_count = sum([rising_3d, rising_7d, rising_14d, rising_21d])
    
    # More consistent = higher score
    consistency_map = {
        4: 95.0,  # All rising
        3: 75.0,  # Mostly rising
        2: 55.0,  # Mixed
        1: 35.0,  # Mostly declining
        0: 15.0   # All declining
    }
    
    return consistency_map.get(rising_count, 50.0)

def safe_get_enhanced_data(row: pd.Series) -> Dict:
    """
    Safely extract enhanced data from a row, with defaults for missing values
    
    Args:
        row: DataFrame row with stock data
    
    Returns:
        Dictionary with enhanced data (or defaults)
    """
    return {
        # Earnings data
        'days_to_earnings': row.get('days_to_earnings'),
        'earnings_beat_rate': row.get('earnings_beat_rate', 0.5),
        'last_earnings_surprise_pct': row.get('last_earnings_surprise_pct'),
        
        # Valuation
        'forward_pe': row.get('forward_pe'),
        
        # Growth metrics
        'revenue_growth': row.get('revenue_growth'),
        'earnings_growth': row.get('earnings_growth'),
        'revenue_growth_qoq': row.get('revenue_growth_qoq'),
        
        # Profitability
        'profit_margins': row.get('profit_margins'),
        'gross_margins': row.get('gross_margins'),
        'return_on_equity': row.get('return_on_equity'),
        'free_cash_flow': row.get('free_cash_flow'),
        
        # Balance sheet
        'debt_to_equity': row.get('debt_to_equity'),
        'current_ratio': row.get('current_ratio'),
        'interest_coverage': row.get('interest_coverage'),
        
        # Institutional
        'institutional_ownership_pct': row.get('institutional_ownership_pct'),
        'insider_buy_sell_ratio': row.get('insider_buy_sell_ratio', 1.0),
        'short_percent_of_float': row.get('short_percent_of_float'),
        'short_interest_change_pct': row.get('short_interest_change_pct'),
        
        # Analyst
        'analyst_target_price': row.get('analyst_target_price'),
        'upside_to_target_pct': row.get('upside_to_target_pct'),
        'recommendation_key': row.get('recommendation_key'),
        'upgrades_3m': row.get('upgrades_3m', 0),
        'downgrades_3m': row.get('downgrades_3m', 0),
        
        # Insider trading
        'insider_buys_6m': row.get('insider_buys_6m', 0),
        'insider_sells_6m': row.get('insider_sells_6m', 0),
    }

# COMPREHENSIVE SCORING FUNCTION
def calculate_stock_score(stock_data: pd.Series,
                                   enhanced_data: Dict,
                                   sector_df: pd.DataFrame,
                                   weights: ScoringWeights = ScoringWeights(),
                                   market_regime: Optional[MarketRegimeData] = None,
                                   strategy: InvestmentStrategy = InvestmentStrategy.BALANCED,
                                   time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM) -> Dict:
    """
    ENHANCED comprehensive scoring with all 6 critical dimensions
    
    Args:
        stock_data: Base metrics (your existing data)
        enhanced_data: Enhanced data from fetch_all_enhanced_data()
        sector_df: All stocks for sector comparison
        weights: Scoring weights
        market_regime: Market regime data
        strategy: Investment strategy
        time_horizon: Investment time horizon
    
    Returns:
        Dictionary with comprehensive scoring and risk management
    """
    weights.validate()

    # Technical Score
    rsi_score = score_rsi(stock_data.get('rsi', np.nan))
    momentum_score = score_momentum(
        stock_data.get('pct_5d', np.nan),
        stock_data.get('pct_21d', np.nan),
        stock_data.get('pct_63d', np.nan)
    )
    position_score = score_52_week_position(
        stock_data.get('pct_from_52w_high', np.nan),
        stock_data.get('pct_from_52w_low', np.nan)
    )
    volume_score = score_volume_trend(stock_data.get('volume_vs_avg', np.nan))
    
    technical_score = (
        rsi_score * 0.35 +
        momentum_score * 0.35 +
        position_score * 0.20 +
        volume_score * 0.10
    )
    
    # Fundamental Score
    pe_score = score_pe_ratio(stock_data.get('pe_ratio', np.nan))
    
    # Add forward P/E if available
    forward_pe = enhanced_data.get('forward_pe')
    if forward_pe and forward_pe > 0:
        forward_pe_score = score_pe_ratio(forward_pe)
        pe_score = (pe_score * 0.6 + forward_pe_score * 0.4)  # Blend
    
    pct_252d = stock_data.get('pct_252d', np.nan)
    if pd.isna(pct_252d):
        growth_score = 50.0
    else:
        growth_score = np.clip(50 + pct_252d, 0, 100)
    
    consistency_score = score_trend_consistency(
        stock_data.get('rising_3day', False),
        stock_data.get('rising_7day', False),
        stock_data.get('rising_14day', False),
        stock_data.get('rising_21day', False)
    )
    
    fundamental_score = (
        pe_score * 0.40 +
        growth_score * 0.35 +
        consistency_score * 0.25
    )
    
    # Sentiment Score
    sentiment_score_val = score_sentiment(
        stock_data.get('sentiment_score', np.nan),
        stock_data.get('article_count', 0)
    )
    
    # Risk Score
    volatility_score = score_volatility(stock_data.get('ann_vol_pct', np.nan))
    risk_score = volatility_score
    
    # FORWARD-LOOKING METRICS
    # Earnings catalyst
    earnings_catalyst_score = score_earnings_catalyst(
        enhanced_data.get('days_to_earnings'),
        enhanced_data.get('earnings_beat_rate', 0.5),
        enhanced_data.get('last_earnings_surprise_pct')
    )
    
    # Growth quality
    growth_quality_score = score_growth_quality(
        enhanced_data.get('revenue_growth'),
        enhanced_data.get('earnings_growth'),
        enhanced_data.get('revenue_growth_qoq')
    )
    
    # Profitability
    profitability_score = score_profitability(
        enhanced_data.get('profit_margins'),
        enhanced_data.get('gross_margins'),
        enhanced_data.get('return_on_equity'),
        enhanced_data.get('free_cash_flow')
    )
    
    # Balance sheet health
    balance_sheet_score = score_balance_sheet_health(
        enhanced_data.get('debt_to_equity'),
        enhanced_data.get('current_ratio'),
        enhanced_data.get('interest_coverage')
    )
    
    forward_looking_score = (
        earnings_catalyst_score * 0.25 +
        growth_quality_score * 0.30 +
        profitability_score * 0.25 +
        balance_sheet_score * 0.20
    )
    
    # INSTITUTIONAL INTEREST METRICS
    institutional_score = score_institutional_activity(
        enhanced_data.get('institutional_ownership_pct'),
        enhanced_data.get('insider_buy_sell_ratio', 1.0),
        enhanced_data.get('short_percent_of_float'),
        enhanced_data.get('short_interest_change_pct')
    )
    
    # RELATIVE STRENGTH METRICS
    sector_percentiles = calculate_sector_percentile(
        stock_data,
        sector_df,
        metrics=['pct_21d', 'pct_63d', 'pe_ratio', 'ann_vol_pct']
    )
    
    relative_strength_score = score_relative_strength(sector_percentiles)
    
    # CATALYSTS
    catalysts = detect_catalysts(stock_data, enhanced_data, sector_percentiles)
    catalyst_score = score_catalysts(catalysts)
    
    # COMPOSITE SCORE (9 DIMENSIONS)
    # Updated weights to include new dimensions
    base_composite_score = (
        technical_score * 0.15 +           
        fundamental_score * 0.20 +         
        sentiment_score_val * 0.10 +       
        risk_score * 0.08 +                
        forward_looking_score * 0.20 +     
        institutional_score * 0.10 +       
        relative_strength_score * 0.10 +   
        catalyst_score * 0.07              
    )
    
    # Risk adjustment
    ann_vol = stock_data.get('ann_vol_pct', 25.0)
    if ann_vol > 50:
        volatility_penalty = (ann_vol - 50) * 0.3
        base_composite_score = max(0, base_composite_score - volatility_penalty)
    
    # REGIME ADJUSTMENT
    regime_multiplier = 1.0
    regime_name = "Unknown"
    regime_confidence = 0.0
    
    if market_regime:
        regime_name = market_regime.regime.value
        regime_confidence = market_regime.confidence
        multipliers = RegimeAdjustmentMultipliers.for_strategy(strategy)
        regime_multiplier = multipliers.get_multiplier(market_regime.regime)
    
    adjusted_composite_score = min(100.0, base_composite_score * regime_multiplier)
    
    # TIME HORIZON ADJUSTMENT
    horizon_adjusted_score, horizon_explanation = adjust_score_for_time_horizon(
        adjusted_composite_score,
        stock_data,
        enhanced_data,
        time_horizon
    )
    
    # CONFIDENCE CALCULATION
    score_components = {
        'technical_score': technical_score,
        'fundamental_score': fundamental_score,
        'sentiment_score': sentiment_score_val,
        'risk_score': risk_score,
        'forward_looking_score': forward_looking_score,
        'institutional_score': institutional_score,
        'relative_strength_score': relative_strength_score,
        'catalyst_score': catalyst_score
    }
    
    confidence_score, confidence_level = calculate_confidence_score(
        stock_data,
        score_components
    )

    # TOP CONTRIBUTING FACTORS
    factor_scores = {
        'Strong Technical Momentum': technical_score if technical_score > 70 else 0,
        'Attractive Valuation (P/E)': pe_score if pe_score > 70 else 0,
        'Positive News Sentiment': sentiment_score_val if sentiment_score_val > 70 else 0,
        'Low Volatility': volatility_score if volatility_score > 75 else 0,
        'Oversold (RSI)': rsi_score if rsi_score > 85 else 0,
        'High Growth Potential': growth_quality_score if growth_quality_score > 75 else 0,
        'Earnings Catalyst': earnings_catalyst_score if earnings_catalyst_score > 70 else 0,
        'Strong Profitability': profitability_score if profitability_score > 75 else 0,
        'Healthy Balance Sheet': balance_sheet_score if balance_sheet_score > 75 else 0,
        'Institutional Interest': institutional_score if institutional_score > 70 else 0,
        'Sector Leader': relative_strength_score if relative_strength_score > 80 else 0,
        'Multiple Catalysts': catalyst_score if catalyst_score > 70 else 0,
    }
    
    top_factors = sorted(
        [(k, v) for k, v in factor_scores.items() if v > 0],
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    contributing_factors = [factor[0] for factor in top_factors]
    if not contributing_factors:
        contributing_factors = ['Balanced across all metrics']
    
    # RISK MANAGEMENT
    risk_mgmt = calculate_risk_management(
        stock_data,
        horizon_adjusted_score,
        confidence_score,
        portfolio_value=100000  # Default, can be parameterized
    )
    
    # RETURN COMPREHENSIVE RESULT 
    return {
        # Identification
        'ticker': stock_data.get('ticker', 'UNKNOWN'),
        'company_name': stock_data.get('company_name', 'Unknown'),
        'sector': stock_data.get('sector', 'Unknown'),
        
        # Core Scores
        'base_score': round(base_composite_score, 2),
        'regime_adjusted_score': round(adjusted_composite_score, 2),
        'composite_score': round(horizon_adjusted_score, 2),
        
        # Dimension Breakdown
        'technical_score': round(technical_score, 2),
        'fundamental_score': round(fundamental_score, 2),
        'sentiment_score': round(sentiment_score_val, 2),
        'risk_score': round(risk_score, 2),
        'forward_looking_score': round(forward_looking_score, 2),
        'institutional_score': round(institutional_score, 2),
        'relative_strength_score': round(relative_strength_score, 2),
        'catalyst_score': round(catalyst_score, 2),
        
        # Sub-scores
        'earnings_catalyst_score': round(earnings_catalyst_score, 2),
        'growth_quality_score': round(growth_quality_score, 2),
        'profitability_score': round(profitability_score, 2),
        'balance_sheet_score': round(balance_sheet_score, 2),
        
        # Regime & Time Horizon
        'market_regime': regime_name,
        'regime_multiplier': round(regime_multiplier, 3),
        'regime_confidence': round(regime_confidence, 2),
        'time_horizon': time_horizon.value,
        'horizon_adjustment': horizon_explanation,
        
        # Confidence
        'confidence_score': round(confidence_score, 2),
        'confidence_level': confidence_level.value,
        
        # Catalysts
        'catalysts': [c.value for c in catalysts],
        'catalyst_count': len(catalysts),
        
        # Sector Relative
        'sector_percentiles': sector_percentiles,
        
        # Explainability
        'top_factors': contributing_factors,
        
        # Risk Management
        'position_size_pct': risk_mgmt.suggested_position_size_pct,
        'position_size_dollars': risk_mgmt.position_size_dollars,
        'stop_loss_price': risk_mgmt.stop_loss_price,
        'stop_loss_pct': risk_mgmt.stop_loss_pct,
        'take_profit_1': risk_mgmt.take_profit_1,
        'take_profit_2': risk_mgmt.take_profit_2,
        'take_profit_3': risk_mgmt.take_profit_3,
        'max_loss_dollars': risk_mgmt.max_loss_dollars,
        'risk_reward_ratio': risk_mgmt.risk_reward_ratio,
        
        # Original metrics
        'current_price': stock_data.get('last_close', None),
        'pe_ratio': stock_data.get('pe_ratio', None),
        'forward_pe': enhanced_data.get('forward_pe'),
        'volatility': round(ann_vol, 2) if not pd.isna(ann_vol) else None,
        
        # Enhanced fundamentals
        'revenue_growth': enhanced_data.get('revenue_growth'),
        'earnings_growth': enhanced_data.get('earnings_growth'),
        'profit_margins': enhanced_data.get('profit_margins'),
        'roe': enhanced_data.get('return_on_equity'),
        'debt_to_equity': enhanced_data.get('debt_to_equity'),
        
        # Institutional
        'institutional_ownership': enhanced_data.get('institutional_ownership_pct'),
        'insider_buy_sell_ratio': enhanced_data.get('insider_buy_sell_ratio'),
        'short_interest': enhanced_data.get('short_percent_of_float'),
        
        # Analyst
        'analyst_target_price': enhanced_data.get('analyst_target_price'),
        'upside_to_target': enhanced_data.get('upside_to_target_pct'),
        'analyst_recommendation': enhanced_data.get('recommendation_key'),
    }


# ENHANCED SCORING FUNCTIONS
# FORWARD-LOOKING ANALYSIS

def score_earnings_catalyst(days_to_earnings: Optional[int],
                           beat_rate: float = 0.5,
                           last_surprise_pct: Optional[float] = None) -> float:
    """
    Score upcoming earnings as a catalyst
    
    High score if:
    - Earnings coming soon (7-14 days)
    - History of beating estimates
    - Last earnings was a beat
    
    Returns: Score 0-100
    """
    if days_to_earnings is None:
        return 50.0  # Neutral if unknown
    
    base_score = 50.0
    
    # 1. Proximity to earnings (sweet spot: 7-14 days before)
    if 7 <= days_to_earnings <= 14:
        proximity_bonus = 20.0
    elif 3 <= days_to_earnings < 7:
        proximity_bonus = 15.0  # Very close (some risk)
    elif 14 < days_to_earnings <= 21:
        proximity_bonus = 10.0
    elif days_to_earnings < 3:
        proximity_bonus = 5.0  # Too close (risky)
    else:
        proximity_bonus = 0.0  # Too far away
    
    # 2. Historical beat rate
    if beat_rate >= 0.75:  # Beats 75%+ of time
        beat_bonus = 20.0
    elif beat_rate >= 0.60:
        beat_bonus = 15.0
    elif beat_rate >= 0.50:
        beat_bonus = 5.0
    else:
        beat_bonus = -10.0  # History of missing
    
    # 3. Last earnings surprise
    surprise_bonus = 0.0
    if last_surprise_pct is not None:
        if last_surprise_pct > 10:  # Beat by >10%
            surprise_bonus = 15.0
        elif last_surprise_pct > 5:
            surprise_bonus = 10.0
        elif last_surprise_pct > 0:
            surprise_bonus = 5.0
        elif last_surprise_pct < -10:  # Missed by >10%
            surprise_bonus = -15.0
    
    total_score = base_score + proximity_bonus + beat_bonus + surprise_bonus
    
    return np.clip(total_score, 0, 100)

def score_growth_quality(revenue_growth: Optional[float],
                        earnings_growth: Optional[float],
                        revenue_growth_qoq: Optional[float] = None) -> float:
    """
    Score quality and sustainability of growth
    
    High score if:
    - Strong revenue AND earnings growth
    - Accelerating growth (QoQ improving)
    - Positive free cash flow
    
    Returns: Score 0-100
    """
    if revenue_growth is None and earnings_growth is None:
        return 50.0
    
    base_score = 50.0
    
    # 1. Revenue growth (YoY)
    if revenue_growth is not None:
        revenue_growth_pct = revenue_growth * 100  # Convert to percentage
        
        if revenue_growth_pct > 20:
            revenue_score = 90.0
        elif revenue_growth_pct > 15:
            revenue_score = 80.0
        elif revenue_growth_pct > 10:
            revenue_score = 70.0
        elif revenue_growth_pct > 5:
            revenue_score = 60.0
        elif revenue_growth_pct > 0:
            revenue_score = 50.0
        else:
            revenue_score = 30.0  # Declining revenue
        
        base_score = revenue_score
    
    # 2. Earnings growth alignment
    if earnings_growth is not None and revenue_growth is not None:
        earnings_growth_pct = earnings_growth * 100
        
        # Check if earnings growing faster than revenue (margin expansion)
        if earnings_growth_pct > revenue_growth * 100:
            base_score += 10.0  # Bonus for margin expansion
        elif earnings_growth_pct < 0 < revenue_growth * 100:
            base_score -= 15.0  # Penalty: revenue up but earnings down
    
    # 3. Growth acceleration (QoQ)
    if revenue_growth_qoq is not None and revenue_growth is not None:
        yoy_pct = revenue_growth * 100
        
        # If QoQ growth is accelerating
        if revenue_growth_qoq > yoy_pct:
            base_score += 10.0  # Accelerating growth
        elif revenue_growth_qoq < yoy_pct * 0.5:
            base_score -= 10.0  # Decelerating growth
    
    return np.clip(base_score, 0, 100)

def score_profitability(profit_margins: Optional[float],
                       gross_margins: Optional[float],
                       roe: Optional[float],
                       fcf: Optional[float]) -> float:
    """
    Score profitability and efficiency
    
    High score if:
    - High and improving margins
    - Strong ROE
    - Positive free cash flow
    
    Returns: Score 0-100
    """
    scores = []
    
    # 1. Profit margins
    if profit_margins is not None:
        margin_pct = profit_margins * 100
        
        if margin_pct > 20:
            scores.append(95.0)
        elif margin_pct > 15:
            scores.append(85.0)
        elif margin_pct > 10:
            scores.append(70.0)
        elif margin_pct > 5:
            scores.append(55.0)
        elif margin_pct > 0:
            scores.append(40.0)
        else:
            scores.append(20.0)
    
    # 2. Return on Equity
    if roe is not None:
        roe_pct = roe * 100
        
        if roe_pct > 20:
            scores.append(90.0)
        elif roe_pct > 15:
            scores.append(75.0)
        elif roe_pct > 10:
            scores.append(60.0)
        elif roe_pct > 0:
            scores.append(45.0)
        else:
            scores.append(25.0)
    
    # 3. Free Cash Flow (positive vs negative)
    if fcf is not None:
        if fcf > 1e9:  # > $1B FCF
            scores.append(85.0)
        elif fcf > 0:
            scores.append(65.0)
        else:
            scores.append(30.0)  # Negative FCF
    
    if scores:
        return np.mean(scores)
    else:
        return 50.0

def score_balance_sheet_health(debt_to_equity: Optional[float],
                               current_ratio: Optional[float],
                               interest_coverage: Optional[float]) -> float:
    """
    Score balance sheet health and financial stability
    
    High score if:
    - Low debt
    - Good liquidity
    - Strong interest coverage
    
    Returns: Score 0-100
    """
    scores = []
    
    # 1. Debt to Equity
    if debt_to_equity is not None:
        if debt_to_equity < 0.3:
            scores.append(95.0)  # Very low debt
        elif debt_to_equity < 0.5:
            scores.append(85.0)  # Low debt
        elif debt_to_equity < 1.0:
            scores.append(70.0)  # Moderate debt
        elif debt_to_equity < 2.0:
            scores.append(50.0)  # High debt
        else:
            scores.append(30.0)  # Very high debt
    
    # 2. Current Ratio (liquidity)
    if current_ratio is not None:
        if current_ratio > 2.0:
            scores.append(90.0)  # Excellent liquidity
        elif current_ratio > 1.5:
            scores.append(80.0)  # Good liquidity
        elif current_ratio > 1.0:
            scores.append(60.0)  # Adequate liquidity
        else:
            scores.append(35.0)  # Poor liquidity
    
    # 3. Interest Coverage
    if interest_coverage is not None:
        if interest_coverage > 10:
            scores.append(95.0)  # Very safe
        elif interest_coverage > 5:
            scores.append(80.0)  # Safe
        elif interest_coverage > 2.5:
            scores.append(60.0)  # Adequate
        else:
            scores.append(35.0)  # Risky
    
    if scores:
        return np.mean(scores)
    else:
        return 50.0

# INSTITUTIONAL ACTIVITY
def score_institutional_activity(institutional_ownership_pct: Optional[float],
                                insider_buy_sell_ratio: float = 1.0,
                                short_interest_pct: Optional[float] = None,
                                short_interest_change_pct: Optional[float] = None) -> float:
    """
    Score institutional and insider activity
    
    High score if:
    - Moderate-high institutional ownership (40-80%)
    - Insider buying > selling
    - Low/declining short interest
    
    Returns: Score 0-100
    """
    base_score = 50.0
    
    # 1. Institutional ownership (sweet spot: 40-80%)
    if institutional_ownership_pct is not None:
        inst_pct = institutional_ownership_pct * 100 if institutional_ownership_pct < 1 else institutional_ownership_pct
        
        if 40 <= inst_pct <= 80:
            inst_score = 75.0  # Optimal range
        elif 30 <= inst_pct < 40 or 80 < inst_pct <= 90:
            inst_score = 65.0  # Good
        elif 20 <= inst_pct < 30:
            inst_score = 55.0  # Moderate
        elif inst_pct > 90:
            inst_score = 50.0  # Too much institutional (less room for growth)
        else:
            inst_score = 45.0  # Low institutional interest
        
        base_score = inst_score
    
    # 2. Insider activity
    if insider_buy_sell_ratio > 2.0:  # 2x more buying than selling
        base_score += 20.0
    elif insider_buy_sell_ratio > 1.5:
        base_score += 12.0
    elif insider_buy_sell_ratio > 1.0:
        base_score += 5.0
    elif insider_buy_sell_ratio < 0.5:  # Heavy selling
        base_score -= 15.0
    elif insider_buy_sell_ratio < 0.8:
        base_score -= 8.0
    
    # 3. Short interest
    if short_interest_pct is not None:
        short_pct = short_interest_pct * 100 if short_interest_pct < 1 else short_interest_pct
        
        if short_pct < 3:
            base_score += 10.0  # Low short interest
        elif short_pct < 5:
            base_score += 5.0
        elif short_pct > 15:
            base_score -= 15.0  # High short interest (risky)
        elif short_pct > 10:
            base_score -= 10.0
    
    # 4. Short interest trend
    if short_interest_change_pct is not None:
        if short_interest_change_pct < -20:  # Shorts covering significantly
            base_score += 15.0  # Potential short squeeze
        elif short_interest_change_pct < -10:
            base_score += 8.0
        elif short_interest_change_pct > 20:  # Shorts increasing
            base_score -= 10.0  # Bearish signal
    
    return np.clip(base_score, 0, 100)

# RELATIVE STRENGTH
def calculate_sector_percentile(stock_data: pd.Series, 
                               sector_df: pd.DataFrame,
                               metrics: List[str] = ['pct_21d', 'pct_63d', 'pe_ratio']) -> Dict[str, float]:
    """
    Calculate stock's percentile rank within its sector
    
    Args:
        stock_data: Series with stock metrics
        sector_df: DataFrame with all stocks in same sector
        metrics: List of metrics to rank
    
    Returns:
        Dictionary with percentile ranks (0-100) for each metric
    """
    from scipy import stats
    
    sector = stock_data.get('sector', 'Unknown')
    sector_stocks = sector_df[sector_df['sector'] == sector]
    
    if len(sector_stocks) < 10:
        # Not enough peers for meaningful comparison
        return {f'{metric}_sector_percentile': 50.0 for metric in metrics}
    
    percentiles = {}
    
    for metric in metrics:
        if metric not in stock_data or metric not in sector_stocks.columns:
            percentiles[f'{metric}_sector_percentile'] = 50.0
            continue
        
        stock_value = stock_data[metric]
        sector_values = sector_stocks[metric].dropna()
        
        if len(sector_values) < 5 or pd.isna(stock_value):
            percentiles[f'{metric}_sector_percentile'] = 50.0
            continue
        
        # Higher is better for returns, lower is better for valuation
        reverse_metrics = ['pe_ratio', 'ann_vol_pct', 'debt_to_equity']
        
        percentile = stats.percentileofscore(sector_values, stock_value, kind='rank')
        
        if metric in reverse_metrics:
            percentile = 100 - percentile
        
        percentiles[f'{metric}_sector_percentile'] = percentile
    
    return percentiles

def score_relative_strength(sector_percentiles: Dict[str, float]) -> float:
    """
    Score based on sector-relative performance
    
    High score if stock is in top quartile of sector
    
    Returns: Score 0-100
    """
    if not sector_percentiles:
        return 50.0
    
    # Average all percentiles
    avg_percentile = np.mean(list(sector_percentiles.values()))
    
    # Map percentile to score with bonus for top performers
    if avg_percentile >= 90:
        return 95.0  # Top 10% in sector
    elif avg_percentile >= 75:
        return 85.0  # Top quartile
    elif avg_percentile >= 60:
        return 70.0  # Above average
    elif avg_percentile >= 40:
        return 55.0  # Average
    elif avg_percentile >= 25:
        return 40.0  # Below average
    else:
        return 30.0  # Bottom quartile

# CATALYST DETECTION
def detect_catalysts(stock_data: pd.Series, 
                    enhanced_data: Dict,
                    sector_percentiles: Dict) -> List[CatalystType]:
    """
    Detect potential catalysts for price movement
    
    Returns:
        List of identified catalysts
    """
    catalysts = []
    
    # 1. Earnings catalyst
    days_to_earnings = enhanced_data.get('days_to_earnings')
    beat_rate = enhanced_data.get('earnings_beat_rate', 0.5)
    
    if days_to_earnings is not None and 0 < days_to_earnings <= 14:
        if beat_rate >= 0.6:
            catalysts.append(CatalystType.EARNINGS_SOON)
            if beat_rate >= 0.75:
                catalysts.append(CatalystType.EARNINGS_BEAT)
    
    # 2. Analyst upgrade
    upgrades = enhanced_data.get('upgrades_3m', 0)
    downgrades = enhanced_data.get('downgrades_3m', 0)
    
    if upgrades > downgrades and upgrades >= 2:
        catalysts.append(CatalystType.ANALYST_UPGRADE)
    
    # 3. Technical breakout
    pct_from_high = stock_data.get('pct_from_52w_high', -20)
    volume_vs_avg = stock_data.get('volume_vs_avg', 0)
    
    if pct_from_high > -2 and volume_vs_avg > 30:
        # Within 2% of 52-week high on high volume
        catalysts.append(CatalystType.BREAKING_RESISTANCE)
    
    # 4. Oversold bounce
    rsi = stock_data.get('rsi', 50)
    pct_5d = stock_data.get('pct_5d', 0)
    
    if rsi < 35 and pct_5d > 2:
        # RSI oversold but starting to recover
        catalysts.append(CatalystType.OVERSOLD_BOUNCE)
    
    # 5. Insider buying
    insider_ratio = enhanced_data.get('insider_buy_sell_ratio', 1.0)
    insider_buys = enhanced_data.get('insider_buys_6m', 0)
    
    if insider_ratio > 2.0 and insider_buys >= 3:
        catalysts.append(CatalystType.INSIDER_BUYING)
    
    # 6. Short squeeze setup
    short_pct = enhanced_data.get('short_percent_of_float')
    short_change = enhanced_data.get('short_interest_change_pct')
    
    if short_pct is not None and short_change is not None:
        short_val = short_pct * 100 if short_pct < 1 else short_pct
        if short_val > 15 and short_change < -15:
            # High short interest declining (shorts covering)
            catalysts.append(CatalystType.SHORT_SQUEEZE_SETUP)
    
    # 7. Sector leadership
    pct_21d_percentile = sector_percentiles.get('pct_21d_sector_percentile', 50)
    if pct_21d_percentile >= 85:
        catalysts.append(CatalystType.SECTOR_LEADER)
    
    return catalysts

def score_catalysts(catalysts: List[CatalystType]) -> float:
    """
    Score based on number and quality of catalysts
    
    Returns: Score 0-100
    """
    if not catalysts:
        return 50.0
    
    base_score = 55.0
    
    # Different catalysts have different weights
    catalyst_weights = {
        CatalystType.EARNINGS_SOON: 12,
        CatalystType.EARNINGS_BEAT: 8,
        CatalystType.ANALYST_UPGRADE: 10,
        CatalystType.BREAKING_RESISTANCE: 10,
        CatalystType.OVERSOLD_BOUNCE: 8,
        CatalystType.INSIDER_BUYING: 12,
        CatalystType.SHORT_SQUEEZE_SETUP: 15,
        CatalystType.SECTOR_LEADER: 10
    }
    
    for catalyst in catalysts:
        base_score += catalyst_weights.get(catalyst, 8)
    
    return min(100, base_score)

# RISK MANAGEMENT
@dataclass
class RiskManagementGuidance:
    """Position sizing and risk parameters"""
    suggested_position_size_pct: float  # % of portfolio
    position_size_dollars: float  # Dollar amount
    stop_loss_price: float
    stop_loss_pct: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    max_loss_dollars: float
    risk_reward_ratio: float
    confidence_adjustment: str  # "Increase", "Normal", "Decrease"

def calculate_risk_management(stock_data: pd.Series,
                             composite_score: float,
                             confidence_score: float,
                             portfolio_value: float = 100000) -> RiskManagementGuidance:
    """
    Calculate position sizing and risk parameters
    
    Based on:
    - Volatility
    - Composite score
    - Confidence level
    - Portfolio size
    
    Returns:
        RiskManagementGuidance with specific levels
    """
    current_price = stock_data.get('last_close', 100)
    volatility = stock_data.get('ann_vol_pct', 25)
    
    # POSITION SIZE
    # Base position size on volatility
    if volatility < 15:
        base_position_pct = 0.08  # 8% (low vol)
    elif volatility < 25:
        base_position_pct = 0.06  # 6%
    elif volatility < 35:
        base_position_pct = 0.04  # 4%
    elif volatility < 50:
        base_position_pct = 0.03  # 3%
    else:
        base_position_pct = 0.02  # 2% (high vol)
    
    # Adjust for composite score (higher score = larger position)
    if composite_score >= 85:
        score_multiplier = 1.3
    elif composite_score >= 75:
        score_multiplier = 1.15
    elif composite_score >= 65:
        score_multiplier = 1.0
    elif composite_score >= 55:
        score_multiplier = 0.85
    else:
        score_multiplier = 0.7
    
    # Adjust for confidence (higher confidence = larger position)
    if confidence_score >= 0.75:
        confidence_multiplier = 1.2
        confidence_adjustment = "Increase"
    elif confidence_score >= 0.55:
        confidence_multiplier = 1.0
        confidence_adjustment = "Normal"
    else:
        confidence_multiplier = 0.8
        confidence_adjustment = "Decrease"
    
    final_position_pct = base_position_pct * score_multiplier * confidence_multiplier
    final_position_pct = min(0.10, final_position_pct)  # Cap at 10%
    
    position_dollars = portfolio_value * final_position_pct
    
    # STOP LOSS
    # Stop loss based on ATR (Average True Range approximation)
    daily_vol_pct = volatility / np.sqrt(252)
    atr_multiplier = 2.0
    
    # Stop loss = 2 x daily volatility or 8% (whichever is tighter for low vol stocks)
    volatility_stop_pct = atr_multiplier * daily_vol_pct
    hard_stop_pct = 8.0
    
    stop_loss_pct = min(volatility_stop_pct, hard_stop_pct)
    stop_loss_price = current_price * (1 - stop_loss_pct / 100)
    
    # TAKE PROFIT LEVELS
    # Risk = distance from entry to stop
    risk_per_share = current_price - stop_loss_price
    
    # Take profit at 1.5:1, 2.5:1, and 4:1 risk/reward
    tp1 = current_price + (risk_per_share * 1.5)
    tp2 = current_price + (risk_per_share * 2.5)
    tp3 = current_price + (risk_per_share * 4.0)
    
    # MAX LOSS 
    shares = position_dollars / current_price
    max_loss_dollars = shares * risk_per_share
    
    return RiskManagementGuidance(
        suggested_position_size_pct=round(final_position_pct * 100, 2),
        position_size_dollars=round(position_dollars, 2),
        stop_loss_price=round(stop_loss_price, 2),
        stop_loss_pct=round(stop_loss_pct, 2),
        take_profit_1=round(tp1, 2),
        take_profit_2=round(tp2, 2),
        take_profit_3=round(tp3, 2),
        max_loss_dollars=round(max_loss_dollars, 2),
        risk_reward_ratio=2.5,
        confidence_adjustment=confidence_adjustment
    )

# TIME HORIZON OPTIMIZATION
def adjust_score_for_time_horizon(base_score: float,
                                  stock_data: pd.Series,
                                  enhanced_data: Dict,
                                  horizon: TimeHorizon) -> Tuple[float, str]:
    """
    Adjust score based on investment time horizon
    
    Returns:
        Tuple of (adjusted_score, explanation)
    """
    adjusted_score = base_score
    explanations = []
    
    if horizon == TimeHorizon.SWING_TRADE:
        # Swing trading: Emphasize technicals, momentum, volume
        rsi = stock_data.get('rsi', 50)
        volume_vs_avg = stock_data.get('volume_vs_avg', 0)
        pct_5d = stock_data.get('pct_5d', 0)
        
        # Favor momentum extremes
        if rsi < 30 or rsi > 70:
            adjusted_score *= 1.1
            explanations.append("Strong RSI signal for swing trade")
        
        # Require volume
        if volume_vs_avg < 10:
            adjusted_score *= 0.85
            explanations.append("Low volume reduces swing trade appeal")
        else:
            explanations.append("Good volume for swing trading")
        
        # Recent momentum
        if abs(pct_5d) > 5:
            adjusted_score *= 1.05
            explanations.append("Strong 5-day momentum")
    
    elif horizon == TimeHorizon.SHORT_TERM:
        # Short-term: Balance of technicals and fundamentals
        pct_21d = stock_data.get('pct_21d', 0)
        catalysts = enhanced_data.get('catalyst_count', 0)
        
        # Recent trend
        if pct_21d > 8:
            adjusted_score *= 1.08
            explanations.append("Strong recent uptrend")
        
        # Catalysts matter
        if catalysts >= 2:
            adjusted_score *= 1.1
            explanations.append(f"{catalysts} near-term catalysts identified")
    
    elif horizon == TimeHorizon.MEDIUM_TERM:
        # Medium-term: Fundamentals + growth
        earnings_growth = enhanced_data.get('earnings_growth')
        revenue_growth = enhanced_data.get('revenue_growth')
        
        if earnings_growth and revenue_growth:
            if earnings_growth > 0.15 and revenue_growth > 0.10:
                adjusted_score *= 1.12
                explanations.append("Strong earnings and revenue growth")
            elif earnings_growth < 0 or revenue_growth < 0:
                adjusted_score *= 0.88
                explanations.append("Negative growth concerns")
    
    else:  # LONG_TERM
        # Long-term: Fundamentals, profitability, balance sheet
        pe_ratio = stock_data.get('pe_ratio')
        roe = enhanced_data.get('return_on_equity')
        debt_to_equity = enhanced_data.get('debt_to_equity')
        profit_margins = enhanced_data.get('profit_margins')
        
        # Value preference for long-term
        if pe_ratio and pe_ratio < 15:
            adjusted_score *= 1.1
            explanations.append("Attractive valuation for long-term")
        
        # Profitability required
        if profit_margins and profit_margins < 0:
            adjusted_score *= 0.75
            explanations.append("Unprofitable - risky for long-term")
        elif profit_margins and profit_margins > 0.15:
            adjusted_score *= 1.08
            explanations.append("Strong profit margins")
        
        # Balance sheet health
        if debt_to_equity and debt_to_equity < 0.5:
            adjusted_score *= 1.05
            explanations.append("Strong balance sheet")
        elif debt_to_equity and debt_to_equity > 2.0:
            adjusted_score *= 0.90
            explanations.append("High debt levels")
        
        # Quality metrics
        if roe and roe > 0.15:
            adjusted_score *= 1.08
            explanations.append("Excellent return on equity")
    
    explanation = " | ".join(explanations) if explanations else "Standard horizon adjustment"
    
    return min(100, adjusted_score), explanation

# RECOMMENDATION CLASSIFICATION
def classify_recommendation(score: float, 
                           thresholds: RecommendationThresholds = RecommendationThresholds()
                           ) -> RecommendationLevel:
    """
    Classify score into recommendation level
    
    Args:
        score: Composite score (0-100)
        thresholds: Recommendation thresholds
    
    Returns:
        RecommendationLevel enum
    """
    if score >= thresholds.strong_buy:
        return RecommendationLevel.STRONG_BUY
    elif score >= thresholds.buy:
        return RecommendationLevel.BUY
    elif score >= thresholds.hold_lower:
        return RecommendationLevel.HOLD
    elif score >= thresholds.sell:
        return RecommendationLevel.SELL
    else:
        return RecommendationLevel.STRONG_SELL

# STRATEGY-SPECIFIC RECOMMENDATIONS
def get_strategy_recommendations(metrics_df: pd.DataFrame, 
                                strategy: InvestmentStrategy,
                                top_n: int = 10) -> pd.DataFrame:
    """
    Get recommendations optimized for specific investment strategy
    
    Args:
        metrics_df: DataFrame with all stock metrics
        strategy: Investment strategy type
        top_n: Number of top recommendations to return
    
    Returns:
        DataFrame with top recommendations for strategy
    """
    
    if strategy == InvestmentStrategy.GROWTH:
        # Growth: High momentum + positive sentiment + willing to accept higher P/E
        weights = ScoringWeights(
            technical=0.40,    # Emphasize momentum
            fundamental=0.30,
            sentiment=0.25,
            risk=0.05          # Less concerned with volatility
        )
        # Prefer stocks with strong upward momentum
        filtered_df = metrics_df[
            (metrics_df['pct_21d'] > 5) &  # Positive 3-week return
            (metrics_df['pct_63d'] > 10)   # Strong 3-month return
        ].copy()
    
    elif strategy == InvestmentStrategy.VALUE:
        # Value: Low P/E + undervalued + positive trend
        weights = ScoringWeights(
            technical=0.20,
            fundamental=0.50,  # Emphasize valuation
            sentiment=0.15,
            risk=0.15
        )
        # Prefer undervalued stocks with reasonable P/E
        filtered_df = metrics_df[
            (metrics_df['pe_ratio'] < 20) &
            (metrics_df['pe_ratio'] > 0) &
            (metrics_df['pct_from_52w_low'] > 10)  # Not at absolute bottom
        ].copy()
    
    elif strategy == InvestmentStrategy.MOMENTUM:
        # Momentum: Strong recent performance + high volume
        weights = ScoringWeights(
            technical=0.50,    # Heavy on technicals
            fundamental=0.20,
            sentiment=0.20,
            risk=0.10
        )
        # Prefer stocks with strong recent momentum
        filtered_df = metrics_df[
            (metrics_df['rising_7day'] == True) &
            (metrics_df['volume_vs_avg'] > 0)
        ].copy()
    
    elif strategy == InvestmentStrategy.CONTRARIAN:
        # Contrarian: Oversold + good fundamentals
        weights = ScoringWeights(
            technical=0.25,
            fundamental=0.45,
            sentiment=0.10,    # Less weight on sentiment
            risk=0.20
        )
        # Prefer oversold stocks with good fundamentals
        filtered_df = metrics_df[
            (metrics_df['rsi'] < 40) &  # Oversold
            (metrics_df['pe_ratio'] < 25) &
            (metrics_df['pe_ratio'] > 0)
        ].copy()
    
    else:  # BALANCED
        # Balanced: Use default weights
        weights = ScoringWeights()
        filtered_df = metrics_df.copy()
    
    # Calculate scores for filtered stocks
    scores = []
    for _, row in filtered_df.iterrows():
        score_dict = calculate_stock_score(row, weights)
        scores.append(score_dict)
    
    if not scores:
        return pd.DataFrame()
    
    scores_df = pd.DataFrame(scores)
    
    # Add recommendation level
    scores_df['recommendation'] = scores_df['composite_score'].apply(
        classify_recommendation
    )
    scores_df['recommendation_str'] = scores_df['recommendation'].apply(
        lambda x: x.value
    )
    
    # Sort by score and return top N
    scores_df = scores_df.sort_values('composite_score', ascending=False)
    
    return scores_df.head(top_n)

# PORTFOLIO OPTIMIZATION
def build_diversified_portfolio(metrics_df: pd.DataFrame,
                               portfolio_size: int = 10,
                               max_per_sector: int = 3,
                               min_score: float = 65.0) -> pd.DataFrame:
    """
    Build a diversified portfolio with sector constraints
    
    Args:
        metrics_df: DataFrame with stock metrics
        portfolio_size: Target number of stocks
        max_per_sector: Maximum stocks per sector
        min_score: Minimum composite score to consider
    
    Returns:
        DataFrame with diversified portfolio recommendations
    """
    # Calculate scores for all stocks
    scores = []
    for _, row in metrics_df.iterrows():
        enhanced_data = safe_get_enhanced_data(row)
        score_dict = calculate_stock_score(
            stock_data=row,
            enhanced_data=enhanced_data,
            sector_df=metrics_df,
            weights=ScoringWeights(),
            market_regime=None,
            strategy=InvestmentStrategy.BALANCED,
            time_horizon=TimeHorizon.MEDIUM_TERM
        )
        if score_dict['composite_score'] >= min_score:
            scores.append(score_dict)
    
    if not scores:
        return pd.DataFrame()
    
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values('composite_score', ascending=False)
    
    # Diversification: Limit stocks per sector
    portfolio = []
    sector_counts = {}
    
    for _, row in scores_df.iterrows():
        sector = row['sector']
        current_count = sector_counts.get(sector, 0)
        
        if current_count < max_per_sector and len(portfolio) < portfolio_size:
            portfolio.append(row)
            sector_counts[sector] = current_count + 1
    
    portfolio_df = pd.DataFrame(portfolio)
    
    # Add recommendation levels
    portfolio_df['recommendation'] = portfolio_df['composite_score'].apply(
        classify_recommendation
    )
    portfolio_df['recommendation_str'] = portfolio_df['recommendation'].apply(
        lambda x: x.value
    )
    
    # Calculate portfolio statistics
    avg_score = portfolio_df['composite_score'].mean()
    avg_volatility = portfolio_df['volatility'].mean()
    
    print(f"\n{'='*70}")
    print(f"DIVERSIFIED PORTFOLIO ({len(portfolio_df)} stocks)")
    print(f"{'='*70}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Volatility: {avg_volatility:.2f}%")
    print(f"\nSector Distribution:")
    for sector, count in sector_counts.items():
        print(f"  {sector}: {count} stocks")
    print(f"{'='*70}\n")
    
    return portfolio_df

# MAIN RECOMMENDATION FUNCTION
def generate_recommendations(metrics_df: pd.DataFrame,
                            strategy: InvestmentStrategy = InvestmentStrategy.BALANCED,
                            include_sentiment: bool = True,
                            top_n: int = 20,
                            index_metrics: Optional[pd.Series] = None,
                            time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM) -> Tuple[pd.DataFrame, MarketRegimeData]:
    """
    Generate regime-aware stock recommendations
    """
    # Validate required columns
    required_cols = ['ticker', 'last_close', 'pct_5d', 'pct_21d', 'sector']
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter valid stocks
    valid_df = metrics_df[metrics_df['status'] == 'ok'].copy()
    
    if valid_df.empty:
        print("No valid stocks to analyze")
        return pd.DataFrame(), MarketRegimeData(
            regime=MarketRegime.UNKNOWN,
            pct_21d=0, pct_63d=0, volatility=0, breadth=50, confidence=0
        )
    
    # Detect market regime
    if index_metrics is not None:
        market_regime = detect_market_regime(index_metrics)
    else:
        aggregate_metrics = pd.Series({
            'pct_21d': valid_df['pct_21d'].median(),
            'pct_63d': valid_df['pct_63d'].median() if 'pct_63d' in valid_df.columns else 0,
            'ann_vol_pct': valid_df['ann_vol_pct'].median() if 'ann_vol_pct' in valid_df.columns else 25,
            'breadth': calculate_market_breadth(valid_df)
        })
        market_regime = detect_market_regime(aggregate_metrics)
    
    print(f"\n{'='*70}")
    print(f"MARKET REGIME: {market_regime}")
    print(f"{'='*70}\n")
    
    # Apply strategy filters
    if strategy == InvestmentStrategy.GROWTH:
        weights = ScoringWeights(technical=0.40, fundamental=0.30, sentiment=0.25, risk=0.05)
        filtered_df = valid_df[
            (valid_df['pct_21d'] > 5) &
            (valid_df.get('pct_63d', 0) > 10)
        ].copy()
    elif strategy == InvestmentStrategy.VALUE:
        weights = ScoringWeights(technical=0.20, fundamental=0.50, sentiment=0.15, risk=0.15)
        filtered_df = valid_df[
            (valid_df['pe_ratio'] < 20) &
            (valid_df['pe_ratio'] > 0) &
            (valid_df.get('pct_from_52w_low', 0) > 10)
        ].copy()
    elif strategy == InvestmentStrategy.MOMENTUM:
        weights = ScoringWeights(technical=0.50, fundamental=0.20, sentiment=0.20, risk=0.10)
        filtered_df = valid_df[
            (valid_df.get('rising_7day', False) == True) &
            (valid_df.get('volume_vs_avg', 0) > 0)
        ].copy()
    elif strategy == InvestmentStrategy.CONTRARIAN:
        weights = ScoringWeights(technical=0.25, fundamental=0.45, sentiment=0.10, risk=0.20)
        filtered_df = valid_df[
            (valid_df.get('rsi', 50) < 40) &
            (valid_df['pe_ratio'] < 25) &
            (valid_df['pe_ratio'] > 0)
        ].copy()
    else:  # BALANCED
        weights = ScoringWeights()
        filtered_df = valid_df.copy()
    
    # Calculate scores
    scores = []
    for _, row in filtered_df.iterrows():
        try:
            # Safely extract enhanced data
            enhanced_data = safe_get_enhanced_data(row)
            
            # Calculate score
            score_dict = calculate_stock_score(
                stock_data=row,
                enhanced_data=enhanced_data,
                sector_df=filtered_df,
                weights=weights,
                market_regime=market_regime,
                strategy=strategy,
                time_horizon=time_horizon
            )
            scores.append(score_dict)
            
        except Exception as e:
            print(f"Error scoring {row.get('ticker', 'UNKNOWN')}: {e}")
            continue
    
    if not scores:
        return pd.DataFrame(), market_regime
    
    scores_df = pd.DataFrame(scores)
    
    # Add recommendation level
    scores_df['recommendation'] = scores_df['composite_score'].apply(
        classify_recommendation
    )
    scores_df['recommendation_str'] = scores_df['recommendation'].apply(
        lambda x: x.value
    )
    
    # Sort by score
    scores_df = scores_df.sort_values('composite_score', ascending=False)
    
    # Add metadata
    scores_df['strategy'] = strategy.value
    scores_df['generated_at'] = pd.Timestamp.now()
    
    return scores_df.head(top_n), market_regime
