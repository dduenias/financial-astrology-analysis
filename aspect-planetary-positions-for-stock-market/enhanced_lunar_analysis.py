"""
Enhanced Lunar Cycle Analysis Tool
---------------------------------
An advanced framework for analyzing market behavior around significant lunar events, planetary aspects,
and their combined effects.

This enhanced script builds on the lunar_cycle_analysis.py by adding:
1. Market Regime Analysis - Separate bull, bear, and sideways markets
2. Volatility Correlation - Analyze relationship between lunar factors and market volatility
3. Sector-Specific Analysis - Compare effectiveness across different markets
4. Signal Optimization - Create weighted scoring system for lunar factors
5. Duration Analysis - Measure how long trends last after lunar signals
6. Planetary Integration - Combine lunar and planetary signals for comprehensive analysis

Author: [Your Name]
Date: April 2025
"""

import swisseph as swe
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from math import fmod
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import time
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
import json

# Suppress warnings
warnings.filterwarnings("ignore")

# Import functions from your original lunar cycle analysis
from lunar_cycle_analysis import (
    initialize_ephemeris,
    get_julian_day,
    get_date_from_julian,
    find_declination_events,
    find_true_node_direction_changes,
    find_moon_phases,
    find_apogee_perigee,
    find_void_of_course,
    get_zodiac_sign
)

# Import functions from planetary aspects analysis if available
try:
    from major_planetary_aspects_analysis import (
        find_aspects_between_planets,
        get_planet_name,
        get_aspect_name,
        get_aspect_symbol
    )
    PLANETARY_ASPECTS_AVAILABLE = True
except ImportError:
    PLANETARY_ASPECTS_AVAILABLE = False
    print("Planetary aspects module not found. Running with lunar analysis only.")

# Global configuration
CONFIG = {
    "default_window_days": 10,
    "max_trend_duration_days": 60,
    "market_regime_ma_window": 200,
    "volatility_window": 20,
    "bull_bear_threshold": 5,  # % above/below MA to define bull/bear market
    "turning_point_threshold": 3,  # % move to confirm trend reversal
    "cluster_proximity_days": 3,  # Days between events to consider them part of same cluster
    "score_threshold_very_strong": 0.8,
    "score_threshold_strong": 0.6,
    "score_threshold_moderate": 0.4,
    "score_threshold_weak": 0.2
}

# --------------------------------------
# MARKET REGIME ANALYSIS FUNCTIONS
# --------------------------------------

def calculate_technical_indicators(price_data):
    """
    Calculate technical indicators for market regime analysis.
    
    Args:
        price_data: DataFrame with OHLC price data
        
    Returns:
        DataFrame with technical indicators added
    """
    df = price_data.copy()
    
    # Make sure we have required columns
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            if col == 'Open' and 'Close' in df.columns:
                df['Open'] = df['Close'].shift(1)
            elif col == 'High' and 'Close' in df.columns:
                df['High'] = df['Close']
            elif col == 'Low' and 'Close' in df.columns:
                df['Low'] = df['Close']
            else:
                print(f"Warning: {col} column not found in price data.")
    
    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate ATR (Average True Range) for volatility
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close'] * 100  # ATR as percentage of price
    
    # Calculate MACD (Moving Average Convergence Divergence)
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Clean up NaN values
    df = df.dropna()
    
    return df

def identify_market_regime(price_data, ma_window=200, threshold=5):
    """
    Identify bull, bear, and sideways market regimes.
    
    Args:
        price_data: DataFrame with price history and technical indicators
        ma_window: Window size for moving average to determine trend
        threshold: Percentage above/below MA to define bull/bear market
        
    Returns:
        DataFrame with market regime labels added
    """
    df = price_data.copy()
    
    # Make sure we have moving average
    if f'MA{ma_window}' not in df.columns:
        df[f'MA{ma_window}'] = df['Close'].rolling(window=ma_window).mean()
    
    # Calculate percentage distance from MA
    ma_col = f'MA{ma_window}'
    df['MA_Pct'] = ((df['Close'] - df[ma_col]) / df[ma_col]) * 100
    
    # Determine market regime based on distance from MA
    df['Market_Regime'] = 'Sideways'
    df.loc[df['MA_Pct'] > threshold, 'Market_Regime'] = 'Bull'
    df.loc[df['MA_Pct'] < -threshold, 'Market_Regime'] = 'Bear'
    
    # Add trend strength based on indicators
    if 'ATR_Pct' in df.columns and 'RSI' in df.columns:
        # High ATR and RSI > 70 -> Strong Bull
        df.loc[(df['Market_Regime'] == 'Bull') & 
               (df['ATR_Pct'] > df['ATR_Pct'].rolling(window=50).mean()) &
               (df['RSI'] > 70), 'Market_Regime'] = 'Strong_Bull'
        
        # High ATR and RSI < 30 -> Strong Bear
        df.loc[(df['Market_Regime'] == 'Bear') & 
               (df['ATR_Pct'] > df['ATR_Pct'].rolling(window=50).mean()) &
               (df['RSI'] < 30), 'Market_Regime'] = 'Strong_Bear'
    
    return df

def analyze_lunar_effectiveness_by_regime(clusters, market_data):
    """
    Analyze effectiveness of lunar clusters by market regime.
    
    Args:
        clusters: List of lunar clusters
        market_data: DataFrame with price history and market regime labels
        
    Returns:
        Dictionary with effectiveness statistics by regime
    """
    results = {
        'Bull': {'count': 0, 'turning_points': 0, 'highs': 0, 'lows': 0},
        'Bear': {'count': 0, 'turning_points': 0, 'highs': 0, 'lows': 0},
        'Sideways': {'count': 0, 'turning_points': 0, 'highs': 0, 'lows': 0},
        'Strong_Bull': {'count': 0, 'turning_points': 0, 'highs': 0, 'lows': 0},
        'Strong_Bear': {'count': 0, 'turning_points': 0, 'highs': 0, 'lows': 0}
    }
    
    # Process each cluster
    for cluster in clusters:
        cluster_date = cluster['date']
        
        # Find closest matching date in market data
        try:
            closest_idx = find_closest_date(market_data, cluster_date)
            
            if closest_idx is not None:
                regime = market_data.loc[closest_idx, 'Market_Regime']
                is_turning_point = cluster.get('is_turning_point', False)
                pattern = cluster.get('pattern', 'Unknown')
                
                # Update counts
                results[regime]['count'] += 1
                
                if is_turning_point:
                    results[regime]['turning_points'] += 1
                
                if pattern == 'High':
                    results[regime]['highs'] += 1
                elif pattern == 'Low':
                    results[regime]['lows'] += 1
        except Exception as e:
            print(f"Error analyzing cluster {cluster_date}: {e}")
    
    # Calculate percentages
    for regime in results:
        if results[regime]['count'] > 0:
            results[regime]['turning_point_pct'] = (results[regime]['turning_points'] / 
                                                  results[regime]['count']) * 100
            
            results[regime]['high_pct'] = (results[regime]['highs'] / 
                                         results[regime]['count']) * 100
            
            results[regime]['low_pct'] = (results[regime]['lows'] / 
                                        results[regime]['count']) * 100
    
    return results

# --------------------------------------
# VOLATILITY CORRELATION FUNCTIONS
# --------------------------------------

def calculate_market_volatility(price_data, window=20):
    """
    Calculate market volatility indicators.
    
    Args:
        price_data: DataFrame with price history
        window: Window size for rolling volatility calculation
        
    Returns:
        DataFrame with volatility indicators added
    """
    df = price_data.copy()
    
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change() * 100
    
    # Calculate rolling standard deviation of returns (annualized)
    df['Volatility'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
    
    # Calculate historical volatility percentile (0-100)
    # This helps identify if current volatility is high or low relative to history
    lookback = min(len(df), 252)  # Use up to 1 year of history
    df['Vol_Percentile'] = df['Volatility'].rolling(window=lookback).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1])
    )
    
    return df

def analyze_volatility_around_lunar_events(clusters, price_data, window_days=10):
    """
    Analyze volatility behavior around lunar clusters.
    
    Args:
        clusters: List of lunar clusters
        price_data: DataFrame with price history and volatility data
        window_days: Number of days to analyze before/after each event
        
    Returns:
        Tuple: (List of volatility results, Dictionary of average impact by event type)
    """
    volatility_results = []
    
    # Ensure we have volatility data
    if 'Volatility' not in price_data.columns:
        price_data = calculate_market_volatility(price_data)
    
    # Process each cluster
    for cluster in clusters:
        cluster_date = cluster['date']
        
        # Get volatility before and after cluster
        try:
            # Convert pandas timestamp to datetime.date if needed
            if isinstance(cluster_date, pd.Timestamp):
                cluster_date = cluster_date.date()
            
            # Convert to Timestamp for DataFrame indexing
            pd_date = pd.Timestamp(cluster_date)
            
            # Define windows
            pre_start = pd_date - pd.Timedelta(days=window_days)
            pre_end = pd_date
            post_start = pd_date
            post_end = pd_date + pd.Timedelta(days=window_days)
            
            # Extract data
            pre_window = price_data[(price_data.index >= pre_start) & (price_data.index < pre_end)]
            post_window = price_data[(price_data.index > post_start) & (price_data.index <= post_end)]
            
            # Skip if not enough data
            if len(pre_window) < 3 or len(post_window) < 3:
                continue
            
            # Calculate volatility metrics
            pre_vol = pre_window['Volatility'].mean()
            post_vol = post_window['Volatility'].mean()
            vol_change_pct = ((post_vol - pre_vol) / pre_vol) * 100 if pre_vol != 0 else 0
            
            # Find volatility peak within window after event
            vol_peak = post_window['Volatility'].max()
            days_to_peak = None
            if not post_window.empty:
                peak_idx = post_window['Volatility'].idxmax()
                days_to_peak = (peak_idx - pd_date).days
            
            # Track volatility convergence/divergence with price
            if 'Returns' in post_window.columns:
                # Correlation between volatility and absolute returns
                abs_returns = post_window['Returns'].abs()
                vol_return_corr = abs_returns.corr(post_window['Volatility'])
            else:
                vol_return_corr = None
            
            # Create result
            volatility_results.append({
                'cluster_date': cluster_date,
                'pre_volatility': pre_vol,
                'post_volatility': post_vol,
                'vol_change_pct': vol_change_pct,
                'volatility_peak': vol_peak,
                'days_to_peak': days_to_peak,
                'vol_return_correlation': vol_return_corr,
                'event_types': cluster.get('event_types', []),
                'is_turning_point': cluster.get('is_turning_point', False),
                'pattern': cluster.get('pattern', 'Unknown')
            })
            
        except Exception as e:
            print(f"Error analyzing volatility for {cluster_date}: {e}")
    
    # Analyze impact by event type
    event_vol_impact = {}
    for result in volatility_results:
        for event_type in result['event_types']:
            if event_type not in event_vol_impact:
                event_vol_impact[event_type] = []
            event_vol_impact[event_type].append(result['vol_change_pct'])
    
    # Calculate summary statistics
    for event_type, impacts in event_vol_impact.items():
        if impacts:
            event_vol_impact[event_type] = {
                'mean_impact': np.mean(impacts),
                'median_impact': np.median(impacts),
                'std_dev': np.std(impacts),
                'max_impact': np.max(impacts),
                'min_impact': np.min(impacts),
                'count': len(impacts)
            }
    
    return volatility_results, event_vol_impact

# --------------------------------------
# SIGNAL OPTIMIZATION FUNCTIONS
# --------------------------------------

def create_weighted_scoring_system(historical_results, min_weight=0.1):
    """
    Create a weighted scoring system based on historical effectiveness.
    
    Args:
        historical_results: Dictionary with event type effectiveness statistics
        min_weight: Minimum weight to assign to any factor
        
    Returns:
        Dictionary with weights for each lunar factor
    """
    weights = {}
    
    # Extract effectiveness scores for each event type
    for event_type, stats in historical_results.items():
        # Get effectiveness percentage
        if isinstance(stats, dict) and 'effectiveness' in stats:
            effectiveness = stats['effectiveness']
        elif isinstance(stats, (int, float)):
            effectiveness = stats
        else:
            effectiveness = 50  # Default value if not found
        
        # Apply sigmoid function to create more separation
        # The sigmoid stretches the weights to create more differentiation
        transformed = 1 / (1 + np.exp(-0.1 * (effectiveness - 50)))
        
        # Ensure minimum weight
        weights[event_type] = max(transformed, min_weight)
    
    # Normalize weights to sum to 1
    sum_weights = sum(weights.values())
    for event_type in weights:
        weights[event_type] /= sum_weights
    
    return weights

def score_lunar_cluster(cluster, weights):
    """
    Score a lunar cluster based on the weighted system.
    
    Args:
        cluster: Dictionary with cluster information
        weights: Dictionary with weights for each event type
        
    Returns:
        Score for the cluster (0-1)
    """
    score = 0
    used_weights = []
    
    # Get event types
    event_types = cluster.get('event_types', [])
    
    # Sum weighted scores
    for event_type in event_types:
        if event_type in weights:
            score += weights[event_type]
            used_weights.append(weights[event_type])
    
    # If no weights were applied, return 0
    if not used_weights:
        return 0
    
    # Normalize by sum of weights to keep score 0-1
    score = score / sum(used_weights) if sum(used_weights) > 0 else 0
    
    return score

def categorize_signal_strength(score):
    """
    Categorize signal strength based on score.
    
    Args:
        score: Signal score (0-1)
        
    Returns:
        String indicating signal strength category
    """
    if score > CONFIG["score_threshold_very_strong"]:
        return "Very Strong"
    elif score > CONFIG["score_threshold_strong"]:
        return "Strong"
    elif score > CONFIG["score_threshold_moderate"]:
        return "Moderate"
    elif score > CONFIG["score_threshold_weak"]:
        return "Weak"
    else:
        return "Very Weak"

# --------------------------------------
# DURATION ANALYSIS FUNCTIONS
# --------------------------------------

def analyze_trend_duration(clusters, price_data, max_days=60, threshold_pct=3):
    """
    Measure how long trends last after lunar clusters that create turning points.
    
    Args:
        clusters: List of lunar clusters
        price_data: DataFrame with price history
        max_days: Maximum number of days to look ahead
        threshold_pct: Percentage move to confirm trend reversal
        
    Returns:
        Tuple: (List of duration results, Dictionary of duration statistics by event type)
    """
    duration_results = []
    
    # Process each cluster
    for cluster in clusters:
        cluster_date = cluster['date']
        
        # Skip if not a turning point
        if not cluster.get('is_turning_point', False):
            continue
        
        # Get pattern type
        pattern = cluster.get('pattern', None)
        if pattern not in ['High', 'Low']:
            continue
        
        # Look ahead for trend reversal
        try:
            # Convert pandas timestamp to datetime.date if needed
            if isinstance(cluster_date, pd.Timestamp):
                cluster_date = cluster_date.date()
            
            # Convert to Timestamp for DataFrame indexing
            pd_date = pd.Timestamp(cluster_date)
            
            # Find closest price data point
            closest_idx = find_closest_date(price_data, cluster_date)
            
            if closest_idx is None:
                continue
                
            # Get price at event and establish baseline
            event_price = price_data.loc[closest_idx, 'Close']
            
            # Get future prices
            future_end = pd_date + pd.Timedelta(days=max_days)
            future_data = price_data[(price_data.index > pd_date) & (price_data.index <= future_end)]
            
            # Skip if not enough future data
            if len(future_data) < 3:
                continue
            
            # Calculate trend duration
            duration = 0
            max_move = 0
            reversal_date = None
            
            if pattern == 'High':
                # For High, look for price to drop below event price by threshold
                for idx, row in future_data.iterrows():
                    days_delta = (idx - pd_date).days
                    pct_move = ((row['Close'] - event_price) / event_price) * 100
                    
                    # Track maximum adverse move
                    if pct_move < max_move:
                        max_move = pct_move
                    
                    # Check for trend reversal
                    if pct_move < -threshold_pct:
                        duration = days_delta
                        reversal_date = idx.date()
                        break
                
                # If no reversal found, set duration to max
                if reversal_date is None:
                    duration = max_days
                    max_move = min(max_move, 0)  # Only consider downside moves
            
            elif pattern == 'Low':
                # For Low, look for price to rise above event price by threshold
                for idx, row in future_data.iterrows():
                    days_delta = (idx - pd_date).days
                    pct_move = ((row['Close'] - event_price) / event_price) * 100
                    
                    # Track maximum favorable move
                    if pct_move > max_move:
                        max_move = pct_move
                    
                    # Check for trend reversal
                    if pct_move > threshold_pct:
                        duration = days_delta
                        reversal_date = idx.date()
                        break
                
                # If no reversal found, set duration to max
                if reversal_date is None:
                    duration = max_days
                    max_move = max(max_move, 0)  # Only consider upside moves
            
            # Create result
            duration_results.append({
                'cluster_date': cluster_date,
                'pattern': pattern,
                'trend_duration_days': duration,
                'max_price_move_pct': max_move,
                'reversal_date': reversal_date,
                'event_types': cluster.get('event_types', [])
            })
        
        except Exception as e:
            print(f"Error analyzing trend duration for {cluster_date}: {e}")
    
    # Analyze duration by event type
    event_durations = {}
    
    for result in duration_results:
        for event_type in result['event_types']:
            if event_type not in event_durations:
                event_durations[event_type] = []
            
            event_durations[event_type].append({
                'duration': result['trend_duration_days'],
                'move': result['max_price_move_pct'],
                'pattern': result['pattern']
            })
    
    # Calculate summary statistics
    event_duration_stats = {}
    for event_type, durations in event_durations.items():
        high_durations = [d['duration'] for d in durations if d['pattern'] == 'High']
        low_durations = [d['duration'] for d in durations if d['pattern'] == 'Low']
        
        high_moves = [d['move'] for d in durations if d['pattern'] == 'High']
        low_moves = [d['move'] for d in durations if d['pattern'] == 'Low']
        
        event_duration_stats[event_type] = {
            'count': len(durations),
            'high_count': len(high_durations),
            'low_count': len(low_durations),
            'mean_duration': np.mean([d['duration'] for d in durations]) if durations else None,
            'mean_high_duration': np.mean(high_durations) if high_durations else None,
            'mean_low_duration': np.mean(low_durations) if low_durations else None,
            'mean_high_move': np.mean(high_moves) if high_moves else None,
            'mean_low_move': np.mean(low_moves) if low_moves else None
        }
    
    return duration_results, event_duration_stats

# --------------------------------------
# PLANETARY INTEGRATION FUNCTIONS
# --------------------------------------

def find_planetary_aspects(start_date, end_date, planet_pairs, aspects_to_find):
    """
    Wrapper for planetary aspects function that handles the case when module is not available.
    
    Args:
        start_date: Beginning of search range
        end_date: End of search range
        planet_pairs: List of planet ID pairs to check for aspects
        aspects_to_find: List of aspect angles to search for
        
    Returns:
        List of planetary aspects found or empty list if module not available
    """
    if not PLANETARY_ASPECTS_AVAILABLE:
        print("Warning: Planetary aspects module not available.")
        return []
    
    try:
        # Call the imported function
        aspects = find_aspects_between_planets(
            start_date, 
            end_date, 
            planet_pairs, 
            aspects_to_find, 
            orb=1.0, 
            is_heliocentric=False
        )
        return aspects
    except Exception as e:
        print(f"Error finding planetary aspects: {e}")
        return []

def create_planetary_weights(planetary_aspects_results):
    """
    Create weights for planetary aspects based on historical effectiveness.
    
    Args:
        planetary_aspects_results: Dictionary with aspect effectiveness statistics
        
    Returns:
        Dictionary with weights for each planetary aspect
    """
    if not planetary_aspects_results:
        return {}
    
    weights = {}
    
    # Process each aspect type
    for aspect_detail, stats in planetary_aspects_results.items():
        if isinstance(stats, dict) and 'effectiveness' in stats:
            effectiveness = stats['effectiveness']
        elif isinstance(stats, (int, float)):
            effectiveness = stats
        else:
            effectiveness = 50  # Default value
        
        # Apply sigmoid function to create more separation
        transformed = 1 / (1 + np.exp(-0.1 * (effectiveness - 50)))
        
        # Ensure minimum weight
        weights[aspect_detail] = max(transformed, 0.1)
    
    # Normalize weights to sum to 1
    sum_weights = sum(weights.values())
    if sum_weights > 0:
        for aspect_type in weights:
            weights[aspect_type] /= sum_weights
    
    return weights

def integrate_lunar_and_planetary_signals(lunar_clusters, planetary_aspects, price_data, 
                                         lunar_weights, planetary_weights, window_days=3):
    """
    Integrate lunar and planetary signals into a combined scoring system.
    
    Args:
        lunar_clusters: List of lunar clusters
        planetary_aspects: List of planetary aspects
        price_data: DataFrame with price history
        lunar_weights: Dictionary with weights for each lunar factor
        planetary_weights: Dictionary with weights for each planetary aspect
        window_days: Maximum days between events to consider them coincident
        
    Returns:
        List of integrated signals with combined scores
    """
    integrated_signals = []
    
    # Skip if no planetary aspects
    if not planetary_aspects:
        # Just score lunar clusters
        for cluster in lunar_clusters:
            lunar_score = score_lunar_cluster(cluster, lunar_weights)
            
            integrated_signals.append({
                'date': cluster['date'],
                'lunar_score': lunar_score,
                'planetary_score': 0,
                'combined_score': lunar_score,
                'signal_strength': categorize_signal_strength(lunar_score),
                'lunar_event_types': cluster.get('event_types', []),
                'planetary_aspects': [],
                'is_turning_point': cluster.get('is_turning_point', False),
                'pattern': cluster.get('pattern', 'Unknown')
            })
        
        return integrated_signals
    
    # Create a date index for planetary aspects
    aspect_dates = {}
    
    for aspect in planetary_aspects:
        aspect_date = aspect['date']
        
        if aspect_date not in aspect_dates:
            aspect_dates[aspect_date] = []
        
        aspect_dates[aspect_date].append(aspect)
    
    # Process each lunar cluster
    for cluster in lunar_clusters:
        cluster_date = cluster['date']
        
        # Find nearby planetary aspects
        nearby_aspects = []
        
        for days_delta in range(-window_days, window_days + 1):
            check_date = cluster_date + datetime.timedelta(days=days_delta)
            
            if check_date in aspect_dates:
                nearby_aspects.extend(aspect_dates[check_date])
        
        # Calculate lunar score
        lunar_score = score_lunar_cluster(cluster, lunar_weights)
        
        # Calculate planetary score
        planetary_score = 0
        used_weights = []
        
        for aspect in nearby_aspects:
            # Create aspect key
            p1 = get_planet_name(aspect['planet1']) if PLANETARY_ASPECTS_AVAILABLE else str(aspect['planet1'])
            p2 = get_planet_name(aspect['planet2']) if PLANETARY_ASPECTS_AVAILABLE else str(aspect['planet2'])
            a = get_aspect_name(aspect['aspect']) if PLANETARY_ASPECTS_AVAILABLE else str(aspect['aspect'])
            
            aspect_key = f"{a}_{p1}_{p2}"
            
            if aspect_key in planetary_weights:
                weight = planetary_weights[aspect_key]
                planetary_score += weight
                used_weights.append(weight)
        
        # Normalize planetary score
        if used_weights:
            planetary_score = planetary_score / sum(used_weights)
        
        # Calculate combined score (weighted average)
        # Give more weight to lunar factors if they're more effective
        if lunar_score > 0 or planetary_score > 0:
            lunar_weight = 0.6  # Adjust this based on comparative effectiveness
            planetary_weight = 0.4
            
            combined_score = (lunar_score * lunar_weight + 
                             planetary_score * planetary_weight) / (lunar_weight + planetary_weight)
        else:
            combined_score = 0
        
        # Get closest price data
        closest_idx = find_closest_date(price_data, cluster_date)
        price_info = {}
        
        if closest_idx is not None:
            price = price_data.loc[closest_idx, 'Close']
            
            # Calculate future price movement (10 days)
            future_cutoff = pd.Timestamp(cluster_date) + pd.Timedelta(days=10)
            future_data = price_data[(price_data.index > pd.Timestamp(cluster_date)) & 
                                   (price_data.index <= future_cutoff)]
            
            if not future_data.empty:
                future_price = future_data.iloc[-1]['Close']
                price_move_pct = ((future_price - price) / price) * 100
                price_info = {
                    'price': price,
                    'future_price': future_price,
                    'price_move_pct': price_move_pct,
                    'trend_direction': 'Up' if price_move_pct > 0 else 'Down'
                }
        
        # Create integrated signal
        signal = {
            'date': cluster_date,
            'lunar_score': lunar_score,
            'planetary_score': planetary_score,
            'combined_score': combined_score,
            'signal_strength': categorize_signal_strength(combined_score),
            'lunar_event_types': cluster.get('event_types', []),
            'planetary_aspects': nearby_aspects,
            'is_turning_point': cluster.get('is_turning_point', False),
            'pattern': cluster.get('pattern', 'Unknown'),
            'price_info': price_info
        }
        
        integrated_signals.append(signal)
    
    return integrated_signals

# --------------------------------------
# HELPER FUNCTIONS
# --------------------------------------

def find_closest_date(price_data, target_date):
    """
    Find the closest date in price data to the target date.
    
    Args:
        price_data: DataFrame with price data indexed by date
        target_date: Date to find closest match for
        
    Returns:
        Index of closest date or None if not found
    """
    # Convert target_date to pd.Timestamp if it's not already
    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.Timestamp(target_date)
    
    # Maximum days to look (forward and backward)
    max_days = 5
    
    # Try exact match first
    if target_date in price_data.index:
        return target_date
    
    # Look for closest match within window
    for days in range(1, max_days + 1):
        # Check day after
        check_date = target_date + pd.Timedelta(days=days)
        if check_date in price_data.index:
            return check_date
        
        # Check day before
        check_date = target_date - pd.Timedelta(days=days)
        if check_date in price_data.index:
            return check_date
    
    return None

def organize_output_directories(market_symbol, start_date, end_date):
    """
    Create an organized directory structure for analysis outputs.
    
    Args:
        market_symbol: Symbol for the market being analyzed
        start_date: Start date of analysis
        end_date: End date of analysis
        
    Returns:
        Dictionary with paths to different output directories
    """
    # Format dates for folder name
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Create main directory structure
    base_dir = "ENHANCED_LUNAR"  # Root folder in script directory
    analysis_dir = f"{base_dir}/analysis_{market_symbol}_{start_str}_{end_str}"
    csv_dir = f"{analysis_dir}/CSV"
    visualization_dir = f"{analysis_dir}/Visualizations"
    
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Return paths
    return {
        'base_dir': base_dir,
        'analysis_dir': analysis_dir,
        'csv_dir': csv_dir,
        'visualization_dir': visualization_dir
    }

def load_historical_effectiveness(filename=None):
    """
    Load historical effectiveness data from file or return default values.
    
    Args:
        filename: Path to JSON file with historical effectiveness data
        
    Returns:
        Dictionary with effectiveness data for lunar factors and planetary aspects
    """
    # Default effectiveness values based on analysis of GOLD, SILVER, and NASDAQ
    default_lunar_effectiveness = {
        'apogee': 59.1,           # Strong across markets
        'first_quarter': 55.6,    # Effective in GOLD
        'declination_zero': 52.4, # Good in GOLD
        'true_node_retrograde_to_direct': 49.0,
        'voc_start': 48.0,
        'voc_end': 48.3,
        'true_node_direct_to_retrograde': 47.5,
        'last_quarter': 46.0,
        'full_moon': 45.6,
        'perigee': 45.5,
        'declination_min': 43.5,
        'declination_max': 43.5,
        'new_moon': 40.0
    }
    
    # Default planetary aspect effectiveness (conservative estimates)
    default_planetary_effectiveness = {
        'Conjunction_Jupiter_Saturn': 62.0,
        'Opposition_Jupiter_Saturn': 58.0,
        'Square_Jupiter_Saturn': 55.0,
        'Trine_Jupiter_Saturn': 53.0,
        'Sextile_Jupiter_Saturn': 51.0,
        'Conjunction_Jupiter_Uranus': 60.0,
        'Opposition_Jupiter_Uranus': 57.0,
        'Square_Jupiter_Uranus': 54.0,
        'Conjunction_Saturn_Pluto': 59.0,
        'Square_Venus_Mars': 52.0
    }
    
    if filename and os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            lunar_effectiveness = data.get('lunar_effectiveness', default_lunar_effectiveness)
            planetary_effectiveness = data.get('planetary_effectiveness', default_planetary_effectiveness)
        except Exception as e:
            print(f"Error loading effectiveness data: {e}")
            lunar_effectiveness = default_lunar_effectiveness
            planetary_effectiveness = default_planetary_effectiveness
    else:
        lunar_effectiveness = default_lunar_effectiveness
        planetary_effectiveness = default_planetary_effectiveness
    
    return {
        'lunar_effectiveness': lunar_effectiveness,
        'planetary_effectiveness': planetary_effectiveness
    }

def save_effectiveness_data(effectiveness_data, filename):
    """
    Save effectiveness data to a JSON file.
    
    Args:
        effectiveness_data: Dictionary with effectiveness statistics
        filename: File to save to
    """
    try:
        with open(filename, 'w') as f:
            json.dump(effectiveness_data, f, indent=4)
        print(f"Effectiveness data saved to {filename}")
    except Exception as e:
        print(f"Error saving effectiveness data: {e}")

# --------------------------------------
# VISUALIZATION FUNCTIONS
# --------------------------------------

def visualize_market_regimes(price_data, visualization_dir, market_symbol):
    """
    Create visualization of market regimes over time.
    
    Args:
        price_data: DataFrame with price and regime data
        visualization_dir: Directory to save visualization
        market_symbol: Symbol for the market being analyzed
    """
    plt.figure(figsize=(14, 8))
    
    # Plot price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(price_data.index, price_data['Close'], 'k-', label='Price')
    
    if 'MA200' in price_data.columns:
        ax1.plot(price_data.index, price_data['MA200'], 'b-', 
                alpha=0.7, label='200-day MA')
    
    # Color the background based on regime
    if 'Market_Regime' in price_data.columns:
        # Define colors for regimes
        colors = {
            'Bull': 'lightgreen',
            'Strong_Bull': 'green',
            'Bear': 'lightcoral',
            'Strong_Bear': 'red',
            'Sideways': 'lightgray'
        }
        
        # Find regime change points
        regimes = price_data['Market_Regime'].tolist()
        dates = price_data.index.tolist()
        
        # Initialize with first regime
        current_regime = regimes[0]
        start_idx = 0
        
        # Plot regime backgrounds
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                # End of a regime, plot background
                ax1.axvspan(dates[start_idx], dates[i-1], 
                           alpha=0.2, color=colors.get(current_regime, 'lightgray'))
                
                # Start new regime
                current_regime = regimes[i]
                start_idx = i
        
        # Plot last regime
        ax1.axvspan(dates[start_idx], dates[-1], 
                   alpha=0.2, color=colors.get(current_regime, 'lightgray'))
    
    ax1.set_title(f'{market_symbol} Price and Market Regimes')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot volatility
    if 'Volatility' in price_data.columns:
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(price_data.index, price_data['Volatility'], 'r-', label='Volatility')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{visualization_dir}/{market_symbol}_market_regimes.png")
    print(f"Market regime visualization saved to {visualization_dir}/{market_symbol}_market_regimes.png")
    plt.close()

def visualize_lunar_clusters(lunar_clusters, market_data, visualization_dir, market_symbol):
    """
    Create visualizations of market behavior around lunar clusters.
    
    Args:
        lunar_clusters: List of lunar clusters
        market_data: DataFrame with price history
        visualization_dir: Directory to save visualizations
        market_symbol: Symbol for the market being analyzed
    """
    # Create individual charts for each cluster
    for i, cluster in enumerate(lunar_clusters):
        cluster_date = cluster['date']
        
        # Skip if no market data
        if find_closest_date(market_data, cluster_date) is None:
            continue
        
        # Convert to pd.Timestamp for indexing
        pd_date = pd.Timestamp(cluster_date)
        
        # Define window
        window_days = CONFIG['default_window_days']
        start_date = pd_date - pd.Timedelta(days=window_days)
        end_date = pd_date + pd.Timedelta(days=window_days)
        
        # Get data within window
        mask = (market_data.index >= start_date) & (market_data.index <= end_date)
        data = market_data.loc[mask].copy()
        
        # Skip if not enough data
        if len(data) < 3:
            continue
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot price data
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'])
        
        # Mark cluster date
        closest_idx = find_closest_date(data, cluster_date)
        if closest_idx is not None:
            event_price = data.loc[closest_idx, 'Close']
            
            plt.axvline(x=closest_idx, color='r', linestyle='--')
            plt.plot(closest_idx, event_price, 'ro', markersize=8)
            
            # Add regime if available
            if 'Market_Regime' in data.columns:
                regime = data.loc[closest_idx, 'Market_Regime']
                plt.title(f"Lunar Cluster on {cluster_date.strftime('%Y-%m-%d')} - {cluster['event_count']} Events - {regime} Market")
            else:
                plt.title(f"Lunar Cluster on {cluster_date.strftime('%Y-%m-%d')} - {cluster['event_count']} Events")
            
            # Add annotation for events
            y_offset = 0
            for event_type in cluster['event_types']:
                plt.annotate(event_type, 
                           xy=(closest_idx, event_price), 
                           xytext=(10, 10 + y_offset),
                           textcoords="offset points",
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                
                y_offset += 15
        
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # Plot volatility if available
        if 'Volatility' in data.columns:
            plt.subplot(2, 1, 2)
            plt.plot(data.index, data['Volatility'], 'r-')
            
            if closest_idx is not None:
                plt.axvline(x=closest_idx, color='r', linestyle='--')
            
            plt.ylabel('Volatility (%)')
            plt.grid(True, alpha=0.3)
        # Otherwise plot returns
        elif 'Returns' in data.columns:
            plt.subplot(2, 1, 2)
            plt.bar(data.index, data['Returns'])
            
            if closest_idx is not None:
                plt.axvline(x=closest_idx, color='r', linestyle='--')
            
            plt.ylabel('Daily % Change')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart
        chart_filename = f"{visualization_dir}/cluster_{i+1}_{cluster_date.strftime('%Y-%m-%d')}.png"
        plt.savefig(chart_filename)
        print(f"Chart saved as {chart_filename}")
        
        plt.close()
    
    # Create summary chart with all clusters
    plt.figure(figsize=(14, 8))
    
    # Plot price for full period
    plt.plot(market_data.index, market_data['Close'], 'k-', alpha=0.7)
    
    # Color regions by regime if available
    if 'Market_Regime' in market_data.columns:
        # Define colors for regimes
        colors = {
            'Bull': 'lightgreen',
            'Strong_Bull': 'green',
            'Bear': 'lightcoral',
            'Strong_Bear': 'red',
            'Sideways': 'lightgray'
        }
        
        # Find regime change points
        regimes = market_data['Market_Regime'].tolist()
        dates = market_data.index.tolist()
        
        # Initialize with first regime
        current_regime = regimes[0]
        start_idx = 0
        
        # Plot regime backgrounds
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                # End of a regime, plot background
                plt.axvspan(dates[start_idx], dates[i-1], 
                           alpha=0.2, color=colors.get(current_regime, 'lightgray'))
                
                # Start new regime
                current_regime = regimes[i]
                start_idx = i
        
        # Plot last regime
        plt.axvspan(dates[start_idx], dates[-1], 
                   alpha=0.2, color=colors.get(current_regime, 'lightgray'))
    
    # Mark each cluster
    for i, cluster in enumerate(lunar_clusters):
        cluster_date = cluster['date']
        pd_date = pd.Timestamp(cluster_date)
        
        closest_idx = find_closest_date(market_data, cluster_date)
        if closest_idx is not None:
            price = market_data.loc[closest_idx, 'Close']
            
            # Use different marker for turning points
            if cluster.get('is_turning_point', False):
                plt.plot(closest_idx, price, 'ro', markersize=8)
            else:
                plt.plot(closest_idx, price, 'bo', markersize=6)
            
            # Add label with cluster number
            if i % 5 == 0:  # Label every 5th cluster to avoid overcrowding
                plt.annotate(f"{i+1}", 
                           xy=(closest_idx, price), 
                           xytext=(0, 10),
                           textcoords="offset points",
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    plt.title(f"All Lunar Clusters - {market_symbol}")
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # Create legend
    plt.plot([], [], 'ro', markersize=8, label='Turning Point')
    plt.plot([], [], 'bo', markersize=6, label='Non-Turning Point')
    
    if 'Market_Regime' in market_data.columns:
        for regime, color in colors.items():
            plt.plot([], [], 's', markersize=10, color=color, alpha=0.2, label=regime)
    
    plt.legend(loc='best')
    
    # Save the summary chart
    summary_filename = f"{visualization_dir}/{market_symbol}_lunar_clusters_summary.png"
    plt.savefig(summary_filename)
    print(f"Summary chart saved as {summary_filename}")
    
    plt.close()

def visualize_integrated_signals(integrated_signals, market_data, visualization_dir, market_symbol):
    """
    Create visualization of integrated lunar and planetary signals.
    
    Args:
        integrated_signals: List of integrated signals
        market_data: DataFrame with price history
        visualization_dir: Directory to save visualization
        market_symbol: Symbol for the market being analyzed
    """
    plt.figure(figsize=(14, 8))
    
    # Plot price
    plt.plot(market_data.index, market_data['Close'], 'k-', alpha=0.7)
    
    # Define colors for signal strength
    colors = {
        'Very Strong': 'darkred',
        'Strong': 'red',
        'Moderate': 'orange',
        'Weak': 'yellow',
        'Very Weak': 'lightgray'
    }
    
    # Mark each signal
    for signal in integrated_signals:
        date = signal['date']
        strength = signal['signal_strength']
        score = signal['combined_score']
        is_turning_point = signal.get('is_turning_point', False)
        
        # Find closest data point
        closest_idx = find_closest_date(market_data, date)
        if closest_idx is None:
            continue
            
        price = market_data.loc[closest_idx, 'Close']
        
        # Mark with colored dot based on strength
        color = colors.get(strength, 'gray')
        
        # Use marker shape based on turning point
        marker = 'o' if is_turning_point else 's'
        size = 8 if is_turning_point else 6
        
        plt.plot(closest_idx, price, marker, color=color, markersize=size)
        
        # Add score annotation to some signals (avoid overcrowding)
        if score > 0.6:  # Only annotate stronger signals
            plt.annotate(f"{score:.2f}", 
                       xy=(closest_idx, price), 
                       xytext=(0, 10),
                       textcoords="offset points",
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    plt.title(f"Integrated Signals - {market_symbol}")
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # Create legend
    for strength, color in colors.items():
        plt.plot([], [], 'o', color=color, markersize=8, label=strength)
    
    plt.plot([], [], 'o', color='gray', markersize=8, label='Turning Point')
    plt.plot([], [], 's', color='gray', markersize=6, label='Non-Turning Point')
    
    plt.legend(loc='best')
    
    # Save the chart
    signals_filename = f"{visualization_dir}/{market_symbol}_integrated_signals.png"
    plt.savefig(signals_filename)
    print(f"Integrated signals chart saved as {signals_filename}")
    
    plt.close()

# Continuing from where the script was cut off - visualize_effectiveness_heatmap function
def visualize_effectiveness_heatmap(effectiveness_data, visualization_dir, market_symbol):
    """
    Create heatmap visualization of effectiveness by factor and market regime.
    
    Args:
        effectiveness_data: Dictionary with effectiveness statistics
        visualization_dir: Directory to save visualization
        market_symbol: Symbol for the market being analyzed
    """
    # Extract regime effectiveness if available
    if 'regime_effectiveness' in effectiveness_data:
        regime_data = effectiveness_data['regime_effectiveness']
        
        # Create data for heatmap
        regimes = ['Bull', 'Bear', 'Sideways', 'Strong_Bull', 'Strong_Bear']
        factors = []
        data = []
        
        # Gather data
        for factor, stats in regime_data.items():
            if isinstance(stats, dict):
                factors.append(factor)
                row = []
                
                for regime in regimes:
                    if regime in stats and 'turning_point_pct' in stats[regime]:
                        row.append(stats[regime]['turning_point_pct'])
                    else:
                        row.append(0)
                
                data.append(row)
        
        # Create heatmap
        if factors and data:
            plt.figure(figsize=(12, 8))
            
            # Create heatmap
            ax = sns.heatmap(data, annot=True, fmt=".1f", 
                           xticklabels=regimes, 
                           yticklabels=factors, 
                           cmap="YlGnBu")
            
            plt.title(f"Lunar Factor Effectiveness by Market Regime - {market_symbol}")
            plt.tight_layout()
            
            # Save the heatmap
            heatmap_filename = f"{visualization_dir}/{market_symbol}_effectiveness_heatmap.png"
            plt.savefig(heatmap_filename)
            print(f"Effectiveness heatmap saved as {heatmap_filename}")
            
            plt.close()
    
    # Create event type effectiveness chart
    if 'event_type_effectiveness' in effectiveness_data:
        event_effectiveness = effectiveness_data['event_type_effectiveness']
        
        # Sort by effectiveness
        sorted_events = sorted(event_effectiveness.items(), 
                             key=lambda x: x[1]['effectiveness'] if isinstance(x[1], dict) and 'effectiveness' in x[1] else 0, 
                             reverse=True)
        
        # Extract data
        events = []
        effectiveness_values = []
        counts = []
        
        for event, stats in sorted_events:
            if isinstance(stats, dict) and 'effectiveness' in stats:
                events.append(event)
                effectiveness_values.append(stats['effectiveness'])
                counts.append(stats.get('count', 0))
        
        # Create bar chart
        if events and effectiveness_values:
            plt.figure(figsize=(12, 8))
            
            # Plot bars
            bars = plt.bar(events, effectiveness_values, alpha=0.7)
            
            # Add count annotations
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(i, bar.get_height() + 1, f"n={count}", 
                       ha='center', va='bottom', fontsize=8)
            
            plt.title(f"Effectiveness of Lunar Factors - {market_symbol}")
            plt.ylabel("Effectiveness (%)")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save the chart
            chart_filename = f"{visualization_dir}/{market_symbol}_event_effectiveness.png"
            plt.savefig(chart_filename)
            print(f"Event effectiveness chart saved as {chart_filename}")
            
            plt.close()

def visualize_duration_analysis(duration_results, visualization_dir, market_symbol):
    """
    Create visualization of trend duration after turning points.
    
    Args:
        duration_results: List of trend duration analysis results
        visualization_dir: Directory to save visualization
        market_symbol: Symbol for the market being analyzed
    """
    if not duration_results:
        print("No duration results to visualize.")
        return
    
    # Create scatter plot of duration vs. price move
    plt.figure(figsize=(12, 8))
    
    # Split by pattern
    high_results = [r for r in duration_results if r['pattern'] == 'High']
    low_results = [r for r in duration_results if r['pattern'] == 'Low']
    
    # Plot high patterns (trend duration after highs)
    if high_results:
        durations = [r['trend_duration_days'] for r in high_results]
        moves = [r['max_price_move_pct'] for r in high_results]
        plt.scatter(durations, moves, color='red', label='After High', alpha=0.7)
    
    # Plot low patterns (trend duration after lows)
    if low_results:
        durations = [r['trend_duration_days'] for r in low_results]
        moves = [r['max_price_move_pct'] for r in low_results]
        plt.scatter(durations, moves, color='green', label='After Low', alpha=0.7)
    
    plt.title(f"Trend Duration and Price Movement After Lunar Signals - {market_symbol}")
    plt.xlabel("Duration (Days)")
    plt.ylabel("Maximum Price Move (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the chart
    duration_filename = f"{visualization_dir}/{market_symbol}_trend_duration.png"
    plt.savefig(duration_filename)
    print(f"Trend duration chart saved as {duration_filename}")
    
    plt.close()
    
    # Create bar chart of average duration by event type
    if hasattr(duration_results, 'event_duration_stats'):
        event_duration_stats = duration_results.event_duration_stats
        
        # Sort by mean duration
        sorted_events = sorted(event_duration_stats.items(), 
                             key=lambda x: x[1]['mean_duration'] if 'mean_duration' in x[1] else 0, 
                             reverse=True)
        
        # Extract data
        events = []
        mean_durations = []
        high_durations = []
        low_durations = []
        
        for event, stats in sorted_events:
            if 'mean_duration' in stats:
                events.append(event)
                mean_durations.append(stats['mean_duration'])
                high_durations.append(stats.get('mean_high_duration', 0))
                low_durations.append(stats.get('mean_low_duration', 0))
        
        # Create grouped bar chart
        if events and mean_durations:
            plt.figure(figsize=(12, 8))
            
            # Plot grouped bars
            x = np.arange(len(events))
            width = 0.25
            
            plt.bar(x - width, mean_durations, width, label='Overall', alpha=0.7)
            plt.bar(x, high_durations, width, label='After High', color='red', alpha=0.7)
            plt.bar(x + width, low_durations, width, label='After Low', color='green', alpha=0.7)
            
            plt.title(f"Average Trend Duration by Event Type - {market_symbol}")
            plt.ylabel("Average Duration (Days)")
            plt.xticks(x, events, rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save the chart
            chart_filename = f"{visualization_dir}/{market_symbol}_duration_by_event.png"
            plt.savefig(chart_filename)
            print(f"Duration by event chart saved as {chart_filename}")
            
            plt.close()

# --------------------------------------
# BACKTESTING & PERFORMANCE FUNCTIONS
# --------------------------------------

def backtest_lunar_signals(integrated_signals, price_data, window_days=10):
    """
    Backtest the performance of integrated lunar signals.
    
    Args:
        integrated_signals: List of integrated signals with scores
        price_data: DataFrame with price history
        window_days: Number of days to look ahead for performance measurement
        
    Returns:
        DataFrame with backtest results
    """
    results = []
    
    # Process each signal
    for signal in integrated_signals:
        signal_date = signal['date']
        signal_strength = signal['signal_strength']
        combined_score = signal['combined_score']
        
        # Skip if we've reached recent dates without enough future data
        cutoff_date = price_data.index.max() - pd.Timedelta(days=window_days)
        if pd.Timestamp(signal_date) > cutoff_date:
            continue
        
        try:
            # Find closest price data point
            closest_idx = find_closest_date(price_data, signal_date)
            
            if closest_idx is None:
                continue
                
            # Get price at signal
            signal_price = price_data.loc[closest_idx, 'Close']
            
            # Define windows for performance measurement
            look_ahead = [1, 3, 5, 10, 15, 20]  # Days to look ahead
            performance = {}
            
            # Calculate performance for each look-ahead period
            for days in look_ahead:
                future_date = pd.Timestamp(signal_date) + pd.Timedelta(days=days)
                
                # Find closest future date in data
                future_indices = price_data.index[price_data.index >= future_date]
                if len(future_indices) > 0:
                    future_idx = future_indices[0]
                    future_price = price_data.loc[future_idx, 'Close']
                    
                    # Calculate return
                    pct_change = ((future_price - signal_price) / signal_price) * 100
                    performance[f'return_{days}d'] = pct_change
                else:
                    performance[f'return_{days}d'] = None
            
            # Add volatility if available
            if 'Volatility' in price_data.columns:
                future_indices = price_data.index[(price_data.index >= pd.Timestamp(signal_date)) & 
                                             (price_data.index <= pd.Timestamp(signal_date) + pd.Timedelta(days=window_days))]
                
                if len(future_indices) > 0:
                    volatility = price_data.loc[future_indices, 'Volatility'].mean()
                    performance['avg_volatility'] = volatility
            
            # Add market regime if available
            if 'Market_Regime' in price_data.columns:
                regime = price_data.loc[closest_idx, 'Market_Regime']
                performance['market_regime'] = regime
            
            # Combine results
            result = {
                'signal_date': signal_date,
                'signal_strength': signal_strength,
                'combined_score': combined_score,
                'is_turning_point': signal.get('is_turning_point', False),
                'pattern': signal.get('pattern', 'Unknown'),
                'price': signal_price
            }
            result.update(performance)
            
            results.append(result)
            
        except Exception as e:
            print(f"Error backtesting signal for {signal_date}: {e}")
    
    # Convert to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        return results_df
    else:
        return pd.DataFrame()

def analyze_backtest_performance(backtest_results):
    """
    Analyze performance of lunar signals from backtest results.
    
    Args:
        backtest_results: DataFrame with backtest results
        
    Returns:
        Dictionary with performance statistics
    """
    if backtest_results.empty:
        return {}
    
    stats = {}
    
    # Get return columns
    return_cols = [col for col in backtest_results.columns if col.startswith('return_')]
    
    # Calculate overall statistics
    stats['overall'] = {}
    for col in return_cols:
        # Skip columns with insufficient data
        valid_returns = backtest_results[col].dropna()
        if len(valid_returns) < 5:
            continue
            
        stats['overall'][col] = {
            'mean': valid_returns.mean(),
            'median': valid_returns.median(),
            'std_dev': valid_returns.std(),
            'positive_pct': (valid_returns > 0).mean() * 100,
            'count': len(valid_returns)
        }
    
    # Calculate statistics by signal strength
    stats['by_strength'] = {}
    strength_categories = backtest_results['signal_strength'].unique()
    
    for strength in strength_categories:
        filtered = backtest_results[backtest_results['signal_strength'] == strength]
        
        stats['by_strength'][strength] = {}
        for col in return_cols:
            valid_returns = filtered[col].dropna()
            if len(valid_returns) < 3:
                continue
                
            stats['by_strength'][strength][col] = {
                'mean': valid_returns.mean(),
                'median': valid_returns.median(),
                'positive_pct': (valid_returns > 0).mean() * 100,
                'count': len(valid_returns)
            }
    
    # Calculate statistics by market regime if available
    if 'market_regime' in backtest_results.columns:
        stats['by_regime'] = {}
        regimes = backtest_results['market_regime'].unique()
        
        for regime in regimes:
            filtered = backtest_results[backtest_results['market_regime'] == regime]
            
            stats['by_regime'][regime] = {}
            for col in return_cols:
                valid_returns = filtered[col].dropna()
                if len(valid_returns) < 3:
                    continue
                    
                stats['by_regime'][regime][col] = {
                    'mean': valid_returns.mean(),
                    'median': valid_returns.median(),
                    'positive_pct': (valid_returns > 0).mean() * 100,
                    'count': len(valid_returns)
                }
    
    # Calculate statistics by pattern type
    if 'pattern' in backtest_results.columns:
        stats['by_pattern'] = {}
        patterns = backtest_results['pattern'].unique()
        
        for pattern in patterns:
            filtered = backtest_results[backtest_results['pattern'] == pattern]
            
            stats['by_pattern'][pattern] = {}
            for col in return_cols:
                valid_returns = filtered[col].dropna()
                if len(valid_returns) < 3:
                    continue
                    
                stats['by_pattern'][pattern][col] = {
                    'mean': valid_returns.mean(),
                    'median': valid_returns.median(),
                    'positive_pct': (valid_returns > 0).mean() * 100,
                    'count': len(valid_returns)
                }
    
    return stats

def visualize_backtest_performance(backtest_results, performance_stats, visualization_dir, market_symbol):
    """
    Create visualizations of backtest performance.
    
    Args:
        backtest_results: DataFrame with backtest results
        performance_stats: Dictionary with performance statistics
        visualization_dir: Directory to save visualizations
        market_symbol: Symbol for the market being analyzed
    """
    if backtest_results.empty:
        print("No backtest results to visualize.")
        return
    
    # Get return columns
    return_cols = [col for col in backtest_results.columns if col.startswith('return_')]
    
    # Create bar chart of overall performance by timeframe
    if 'overall' in performance_stats:
        overall_stats = performance_stats['overall']
        
        # Extract data
        timeframes = []
        mean_returns = []
        positive_pcts = []
        
        for col in return_cols:
            if col in overall_stats:
                days = col.split('_')[1].replace('d', '')
                timeframes.append(f"{days} days")
                mean_returns.append(overall_stats[col]['mean'])
                positive_pcts.append(overall_stats[col]['positive_pct'])
        
        # Create chart
        if timeframes and mean_returns:
            plt.figure(figsize=(12, 8))
            
            # Plot dual axis chart
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Plot mean returns
            bars = ax1.bar(timeframes, mean_returns, color='blue', alpha=0.7)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.2f}%', ha='center', va='bottom')
            
            ax1.set_ylabel('Mean Return (%)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Create second y-axis for positive percentage
            ax2 = ax1.twinx()
            ax2.plot(timeframes, positive_pcts, 'r-', marker='o', linewidth=2)
            ax2.set_ylabel('Positive Return %', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.title(f"Overall Signal Performance - {market_symbol}")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save the chart
            chart_filename = f"{visualization_dir}/{market_symbol}_overall_performance.png"
            plt.savefig(chart_filename)
            print(f"Overall performance chart saved as {chart_filename}")
            
            plt.close()
    
    # Create chart of performance by signal strength
    if 'by_strength' in performance_stats:
        strength_stats = performance_stats['by_strength']
        
        # Select a specific return timeframe (e.g., 10 days)
        target_return = 'return_10d'
        if target_return in return_cols:
            # Extract data
            strengths = []
            mean_returns = []
            counts = []
            
            for strength, stats in strength_stats.items():
                if target_return in stats:
                    strengths.append(strength)
                    mean_returns.append(stats[target_return]['mean'])
                    counts.append(stats[target_return]['count'])
            
            # Create chart
            if strengths and mean_returns:
                plt.figure(figsize=(12, 8))
                
                # Define colors based on return values
                colors = ['green' if ret > 0 else 'red' for ret in mean_returns]
                
                # Plot bars
                bars = plt.bar(strengths, mean_returns, color=colors, alpha=0.7)
                
                # Add count annotations
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    plt.text(i, bar.get_height() + 0.1, f"n={count}", 
                           ha='center', va='bottom', fontsize=8)
                
                plt.title(f"10-Day Return by Signal Strength - {market_symbol}")
                plt.ylabel("Mean Return (%)")
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                # Save the chart
                chart_filename = f"{visualization_dir}/{market_symbol}_strength_performance.png"
                plt.savefig(chart_filename)
                print(f"Strength performance chart saved as {chart_filename}")
                
                plt.close()
    
    # Create scatter plot of signal score vs. return
    plt.figure(figsize=(12, 8))
    
    # Choose a specific return period
    return_period = 'return_10d'
    if return_period in backtest_results.columns:
        # Plot scatter
        plt.scatter(backtest_results['combined_score'], 
                   backtest_results[return_period],
                   alpha=0.7)
        
        # Add trend line
        valid_data = backtest_results.dropna(subset=[return_period, 'combined_score'])
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['combined_score'], valid_data[return_period], 1)
            p = np.poly1d(z)
            plt.plot(valid_data['combined_score'], p(valid_data['combined_score']), "r--", alpha=0.7)
        
        plt.title(f"Signal Score vs. 10-Day Return - {market_symbol}")
        plt.xlabel("Signal Score")
        plt.ylabel("10-Day Return (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        chart_filename = f"{visualization_dir}/{market_symbol}_score_vs_return.png"
        plt.savefig(chart_filename)
        print(f"Score vs. return chart saved as {chart_filename}")
        
        plt.close()

# --------------------------------------
# SECTOR COMPARISON FUNCTIONS
# --------------------------------------

def compare_markets(market_data_dict, integrated_signals_dict, visualization_dir):
    """
    Compare effectiveness of lunar signals across different markets/sectors.
    
    Args:
        market_data_dict: Dictionary mapping market symbols to price DataFrames
        integrated_signals_dict: Dictionary mapping market symbols to signals
        visualization_dir: Directory to save visualizations
        
    Returns:
        Dictionary with comparison statistics
    """
    if not market_data_dict or not integrated_signals_dict:
        return {}
    
    comparison = {
        'effectiveness_by_market': {},
        'correlation': {}
    }
    
    # Calculate effectiveness stats for each market
    for symbol, signals in integrated_signals_dict.items():
        if symbol not in market_data_dict:
            continue
            
        price_data = market_data_dict[symbol]
        
        # Count turning points
        total_signals = len(signals)
        turning_points = sum(1 for s in signals if s.get('is_turning_point', False))
        
        # Calculate effectiveness percentage
        if total_signals > 0:
            effectiveness = (turning_points / total_signals) * 100
        else:
            effectiveness = 0
            
        comparison['effectiveness_by_market'][symbol] = {
            'total_signals': total_signals,
            'turning_points': turning_points,
            'effectiveness_pct': effectiveness
        }
    
    # Calculate correlation of returns after signals
    markets = list(market_data_dict.keys())
    
    for i in range(len(markets)):
        for j in range(i+1, len(markets)):
            market1 = markets[i]
            market2 = markets[j]
            
            if market1 not in integrated_signals_dict or market2 not in integrated_signals_dict:
                continue
                
            # Collect return data for common dates
            returns_data = []
            
            for signal1 in integrated_signals_dict[market1]:
                date1 = signal1['date']
                
                # Find matching signal in market2
                matching_signals = [s for s in integrated_signals_dict[market2] 
                                  if abs((s['date'] - date1).days) <= 3]
                
                if matching_signals:
                    # Take the closest match
                    signal2 = min(matching_signals, key=lambda s: abs((s['date'] - date1).days))
                    
                    # Get 10-day returns if available
                    if 'price_info' in signal1 and 'price_info' in signal2:
                        if 'price_move_pct' in signal1['price_info'] and 'price_move_pct' in signal2['price_info']:
                            returns_data.append({
                                'date': date1,
                                f'{market1}_return': signal1['price_info']['price_move_pct'],
                                f'{market2}_return': signal2['price_info']['price_move_pct']
                            })
            
            # Calculate correlation
            if returns_data:
                df = pd.DataFrame(returns_data)
                corr = df[f'{market1}_return'].corr(df[f'{market2}_return'])
                
                comparison['correlation'][f'{market1}_{market2}'] = {
                    'correlation': corr,
                    'sample_size': len(returns_data)
                }
    
    # Create visualization of effectiveness by market
    effectiveness_data = comparison['effectiveness_by_market']
    
    if effectiveness_data:
        plt.figure(figsize=(12, 8))
        
        # Extract data
        markets = []
        effectiveness_values = []
        counts = []
        
        for market, stats in sorted(effectiveness_data.items(), key=lambda x: x[1]['effectiveness_pct'], reverse=True):
            markets.append(market)
            effectiveness_values.append(stats['effectiveness_pct'])
            counts.append(stats['total_signals'])
        
        # Plot bars
        bars = plt.bar(markets, effectiveness_values, alpha=0.7)
        
        # Add count annotations
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(i, bar.get_height() + 1, f"n={count}", 
                   ha='center', va='bottom', fontsize=8)
        
        plt.title("Lunar Signal Effectiveness by Market")
        plt.ylabel("Effectiveness (%)")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        chart_filename = f"{visualization_dir}/market_effectiveness_comparison.png"
        plt.savefig(chart_filename)
        print(f"Market effectiveness comparison saved as {chart_filename}")
        
        plt.close()
    
    # Create correlation heatmap
    if comparison['correlation']:
        # Extract correlation data
        correlations = {}
        
        for pair, data in comparison['correlation'].items():
            markets = pair.split('_')
            market1 = markets[0]
            market2 = markets[1]
            
            if market1 not in correlations:
                correlations[market1] = {}
            
            if market2 not in correlations:
                correlations[market2] = {}
            
            correlations[market1][market2] = data['correlation']
            correlations[market2][market1] = data['correlation']
            
            # Add diagonal elements (self-correlation = 1)
            for market in markets:
                if market not in correlations:
                    correlations[market] = {}
                correlations[market][market] = 1.0
        
        # Convert to DataFrame
        market_list = list(correlations.keys())
        corr_matrix = pd.DataFrame(index=market_list, columns=market_list)
        
        for market1 in market_list:
            for market2 in market_list:
                if market1 in correlations and market2 in correlations[market1]:
                    corr_matrix.loc[market1, market2] = correlations[market1][market2]
                else:
                    corr_matrix.loc[market1, market2] = np.nan
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title("Return Correlation After Lunar Signals")
        plt.tight_layout()
        
        # Save the heatmap
        heatmap_filename = f"{visualization_dir}/market_correlation_heatmap.png"
        plt.savefig(heatmap_filename)
        print(f"Market correlation heatmap saved as {heatmap_filename}")
        
        plt.close()
    
    return comparison

# --------------------------------------
# MAIN EXECUTION FUNCTIONS
# --------------------------------------


def get_market_data(symbol, start_date, end_date):
    """
    Get market data for a symbol and calculate technical indicators.
    
    Args:
        symbol: Market symbol to analyze
        start_date: Beginning of analysis period
        end_date: End of analysis period
        
    Returns:
        DataFrame with price data and indicators
    """
    # Format symbol for Yahoo Finance
    if symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
        yf_symbol = '^GSPC'
    elif symbol.upper() in ['DJI', 'DJIA', 'DOW']:
        yf_symbol = '^DJI'
    elif symbol.upper() in ['IXIC', 'NASDAQ', 'NDX']:
        yf_symbol = '^IXIC'
    elif symbol.upper() in ['RUT', 'RUSSELL', 'RUSSELL2000']:
        yf_symbol = '^RUT'   # Russell 2000
    elif symbol.upper() in ['GOLD', 'XAU']:
        yf_symbol = 'GC=F'  # Gold Futures
    elif symbol.upper() in ['SILVER', 'XAG']:
        yf_symbol = 'SI=F'  # Silver Futures
    elif symbol.upper() in ['WHEAT', 'ZW', 'W']:
        yf_symbol = 'ZW=F'   # Wheat Futures
    elif symbol.upper() in ['CORN', 'ZC', 'C']:
        yf_symbol = 'ZC=F'   # Corn Futures
    elif symbol.upper() in ['SOYBEANS', 'ZS', 'S']:
        yf_symbol = 'ZS=F'   # Soybean Futures
    elif symbol.upper() in ['COTTON', 'CT']:
        yf_symbol = 'CT=F'   # Cotton Futures
    elif symbol.upper() in ['SUGAR', 'SB']:
        yf_symbol = 'SB=F'   # Sugar Futures
    elif symbol.upper() in ['COFFEE', 'KC']:
        yf_symbol = 'KC=F'   # Coffee Futures
    elif symbol.upper() in ['COCOA', 'CC']:
        yf_symbol = 'CC=F'   # Cocoa Futures
    else:
        yf_symbol = symbol
    
    try:
        # Extend date range for calculation of indicators
        extended_start = start_date - datetime.timedelta(days=365)  # Extra year for MA calculation
        
        # Get market data - first attempt with download method
        print(f"Attempting to download data for {yf_symbol}...")
        try:
            data = yf.download(yf_symbol, start=extended_start, end=end_date, progress=False)
            print(f"Successfully downloaded {len(data)} rows of data for {symbol}")
        except Exception as e:
            print(f"Error with yf.download method: {e}")
            print("Trying with alternate Ticker.history method...")
            
            # Second attempt with Ticker.history method
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(start=extended_start, end=end_date)
            print(f"Successfully retrieved {len(data)} rows of data with Ticker.history method")
        
        if data.empty:
            print(f"No data available for {symbol} in the specified date range.")
            return pd.DataFrame()
        
        # Add a short delay to avoid rate limiting
        time.sleep(2)
        
        # Calculate technical indicators
        data = calculate_technical_indicators(data)
        
        # Identify market regimes
        data = identify_market_regime(data, 
                                    ma_window=CONFIG['market_regime_ma_window'],
                                    threshold=CONFIG['bull_bear_threshold'])
        
        # Calculate volatility
        data = calculate_market_volatility(data, window=CONFIG['volatility_window'])
        
        # Trim to requested date range
        mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
        trimmed_data = data.loc[mask].copy()
        
        return trimmed_data
        
    except Exception as e:
        print(f"Error getting market data for {symbol}: {e}")
        return pd.DataFrame()

def analyze_multiple_markets(market_symbols, start_date, end_date, window_days=10, cluster_threshold=2):
    """
    Run enhanced lunar analysis on multiple markets/sectors.
    
    Args:
        market_symbols: List of market symbols to analyze
        start_date: Beginning of analysis period
        end_date: End of analysis period
        window_days: Number of days to analyze before/after each event
        cluster_threshold: Minimum number of lunar factors for a cluster
        
    Returns:
        Tuple: (Market data dictionary, Results dictionary)
    """
    # Load historical effectiveness data
    effectiveness_data = load_historical_effectiveness()
    
    # Create weight systems
    lunar_weights = create_weighted_scoring_system(effectiveness_data['lunar_effectiveness'])
    planetary_weights = create_weighted_scoring_system(effectiveness_data['planetary_effectiveness'])
    
    # Store results for each market
    market_data_dict = {}
    results_dict = {}
    
    # Process each market
    for symbol in market_symbols:
        print(f"\nAnalyzing {symbol}...")
        
        # Create directories
        directories = organize_output_directories(symbol, start_date, end_date)
        
        # Get market data
        market_data = get_market_data(symbol, start_date, end_date)
        
        if market_data.empty:
            print(f"Skipping {symbol} due to insufficient data.")
            continue
            
        market_data_dict[symbol] = market_data
        
        # Run lunar analysis
        lunar_results = run_lunar_analysis(symbol, start_date, end_date, market_data, 
                                        window_days, cluster_threshold, directories)
        
        if PLANETARY_ASPECTS_AVAILABLE:
            # Run planetary aspects analysis if available
            planetary_results = run_planetary_analysis(symbol, start_date, end_date, market_data, 
                                                    window_days, directories)
            
            # Integrate lunar and planetary signals
            integrated_signals = integrate_lunar_and_planetary_signals(
                lunar_results.get('lunar_clusters', []),
                planetary_results.get('planetary_aspects', []),
                market_data,
                lunar_weights,
                planetary_weights
            )
        else:
            # Just use lunar signals
            integrated_signals = []
            for cluster in lunar_results.get('lunar_clusters', []):
                lunar_score = score_lunar_cluster(cluster, lunar_weights)
                
                integrated_signals.append({
                    'date': cluster['date'],
                    'lunar_score': lunar_score,
                    'planetary_score': 0,
                    'combined_score': lunar_score,
                    'signal_strength': categorize_signal_strength(lunar_score),
                    'lunar_event_types': cluster.get('event_types', []),
                    'planetary_aspects': [],
                    'is_turning_point': cluster.get('is_turning_point', False),
                    'pattern': cluster.get('pattern', 'Unknown')
                })
        
        # Backtest performance
        backtest_results = backtest_lunar_signals(integrated_signals, market_data, window_days)
        performance_stats = analyze_backtest_performance(backtest_results)
        
        # Save results
        results_dict[symbol] = {
            'lunar_results': lunar_results,
            'planetary_results': planetary_results if PLANETARY_ASPECTS_AVAILABLE else None,
            'integrated_signals': integrated_signals,
            'backtest_results': backtest_results,
            'performance_stats': performance_stats
        }
        
        # Visualize results
        visualize_market_regimes(market_data, directories['visualization_dir'], symbol)
        visualize_lunar_clusters(lunar_results.get('lunar_clusters', []), market_data, 
                               directories['visualization_dir'], symbol)
        visualize_integrated_signals(integrated_signals, market_data,
                                   directories['visualization_dir'], symbol)
        visualize_backtest_performance(backtest_results, performance_stats,
                                      directories['visualization_dir'], symbol)
        
        # If we have duration results
        if 'duration_results' in lunar_results:
            visualize_duration_analysis(lunar_results['duration_results'],
                                       directories['visualization_dir'], symbol)
        
        # Save results to CSV
        save_results_to_csv(symbol, results_dict[symbol], directories['csv_dir'])
    
    # Compare markets if we have multiple
    if len(market_symbols) > 1:
        # Create master visualization directory
        master_dir = f"ENHANCED_LUNAR/MARKET_COMPARISON_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        os.makedirs(master_dir, exist_ok=True)
        
        # Compare markets
        comparison_stats = compare_markets(
            market_data_dict,
            {symbol: results['integrated_signals'] for symbol, results in results_dict.items()},
            master_dir
        )
        
        # Add to results
        results_dict['comparison'] = comparison_stats
    
    return market_data_dict, results_dict

def run_lunar_analysis(symbol, start_date, end_date, market_data, window_days, cluster_threshold, directories):
    """
    Run lunar cycle analysis for a market.
    
    Args:
        symbol: Market symbol to analyze
        start_date: Beginning of analysis period
        end_date: End of analysis period
        market_data: DataFrame with price history
        window_days: Number of days to analyze before/after each event
        cluster_threshold: Minimum number of lunar factors for a cluster
        directories: Dictionary with output directories
        
    Returns:
        Dictionary with lunar analysis results
    """
    # Results container
    results = {}
    
    print("Calculating lunar events...")
    lunar_events = {}
    
    # Calculate each lunar factor
    print("Finding Moon declination events...")
    lunar_events['declination'] = find_declination_events(start_date, end_date)
    print(f"Found {len(lunar_events['declination'])} declination events.")
    
    print("Finding True Node direction changes...")
    lunar_events['true_node'] = find_true_node_direction_changes(start_date, end_date)
    print(f"Found {len(lunar_events['true_node'])} True Node direction changes.")
    
    print("Finding Moon phases...")
    lunar_events['moon_phase'] = find_moon_phases(start_date, end_date)
    print(f"Found {len(lunar_events['moon_phase'])} Moon phases.")
    
    print("Finding Apogee/Perigee points...")
    lunar_events['distance'] = find_apogee_perigee(start_date, end_date)
    print(f"Found {len(lunar_events['distance'])} distance extremes.")
    
    print("Finding Moon void of course periods...")
    lunar_events['voc'] = find_void_of_course(start_date, end_date)
    print(f"Found {len(lunar_events['voc'])} void of course events.")
    
    # Store all lunar events
    results['lunar_events'] = lunar_events
    
    # Identify clusters
    print("\nIdentifying lunar clusters...")
    from lunar_cycle_analysis import identify_lunar_clusters
    lunar_clusters = identify_lunar_clusters(lunar_events, threshold=cluster_threshold)
    print(f"Found {len(lunar_clusters)} lunar clusters.")
    
    # Store clusters
    results['lunar_clusters'] = lunar_clusters
    
    # Get market data for each cluster
    print("\nAnalyzing market behavior around clusters...")
    from lunar_cycle_analysis import analyze_market_behavior
    
    cluster_dates = [cluster['date'] for cluster in lunar_clusters]
    market_analyses = {}
    
    for cluster_date in cluster_dates:
        market_analyses[cluster_date] = analyze_market_behavior(market_data, cluster_date)
    
    # Store market analyses
    results['market_analyses'] = market_analyses
    
    # Update cluster objects with analysis results
    for cluster in lunar_clusters:
        cluster_date = cluster['date']
        if cluster_date in market_analyses:
            analysis = market_analyses[cluster_date]
            cluster['is_turning_point'] = analysis.get('is_turning_point', False)
            cluster['pattern'] = analysis.get('pattern', 'Unknown')
    
    # Analyze effectiveness by market regime
    print("\nAnalyzing effectiveness by market regime...")
    regime_effectiveness = analyze_lunar_effectiveness_by_regime(lunar_clusters, market_data)
    results['regime_effectiveness'] = regime_effectiveness
    
    # Analyze volatility around lunar events
    print("\nAnalyzing volatility correlations...")
    volatility_results, vol_impact = analyze_volatility_around_lunar_events(
        lunar_clusters, market_data, window_days)
    
    results['volatility_results'] = volatility_results
    results['volatility_impact'] = vol_impact
    
    # Analyze trend duration
    print("\nAnalyzing trend duration after turning points...")
    duration_results, duration_stats = analyze_trend_duration(
        lunar_clusters, market_data, CONFIG['max_trend_duration_days'], CONFIG['turning_point_threshold'])
    
    results['duration_results'] = duration_results
    results['duration_stats'] = duration_stats
    
    return results

def run_planetary_analysis(symbol, start_date, end_date, market_data, window_days, directories):
    """
    Run planetary aspects analysis for a market.
    
    Args:
        symbol: Market symbol to analyze
        start_date: Beginning of analysis period
        end_date: End of analysis period
        market_data: DataFrame with price history
        window_days: Number of days to analyze before/after each aspect
        directories: Dictionary with output directories
        
    Returns:
        Dictionary with planetary analysis results
    """
    if not PLANETARY_ASPECTS_AVAILABLE:
        print("Planetary aspects module not available. Skipping planetary analysis.")
        return {}
    
    # Results container
    results = {}
    
    # Define planets to analyze
    planet_map = {
        'Mercury': swe.MERCURY,
        'Venus': swe.VENUS,
        'Mars': swe.MARS,
        'Jupiter': swe.JUPITER,
        'Saturn': swe.SATURN,
        'Uranus': swe.URANUS,
        'Neptune': swe.NEPTUNE,
        'Pluto': swe.PLUTO
    }
    
    # Define aspects to find
    aspects_to_find = [0, 60, 90, 120, 180]  # Conjunction, Sextile, Square, Trine, Opposition
    
    # Define planet pairs
    default_pairs = [
        # Traditional major pairs
        (planet_map['Jupiter'], planet_map['Saturn']),
        (planet_map['Jupiter'], planet_map['Uranus']),
        (planet_map['Saturn'], planet_map['Uranus']),
        (planet_map['Saturn'], planet_map['Pluto']),
        # Personal/social planet pairs
        (planet_map['Venus'], planet_map['Mars']),
        (planet_map['Mercury'], planet_map['Jupiter']),
        (planet_map['Venus'], planet_map['Jupiter']),
        (planet_map['Mars'], planet_map['Jupiter']),
        (planet_map['Mars'], planet_map['Saturn'])
    ]
    
    print("Finding planetary aspects...")
    planetary_aspects = find_planetary_aspects(start_date, end_date, default_pairs, aspects_to_find)
    print(f"Found {len(planetary_aspects)} planetary aspects.")
    
    # Store aspects
    results['planetary_aspects'] = planetary_aspects
    
    return results

def save_results_to_csv(symbol, results, csv_dir):
    """
    Save analysis results to CSV files.
    
    Args:
        symbol: Market symbol analyzed
        results: Dictionary with analysis results
        csv_dir: Directory to save CSV files
    """
    # Save lunar clusters
    lunar_clusters = results.get('lunar_results', {}).get('lunar_clusters', [])
    
    if lunar_clusters:
        clusters_data = []
        
        for cluster in lunar_clusters:
            # Basic fields
            row = {
                'Date': cluster['date'],
                'Event_Count': cluster.get('event_count', 0),
                'Event_Types': ','.join(cluster.get('event_types', [])),
                'Is_Turning_Point': cluster.get('is_turning_point', False),
                'Pattern': cluster.get('pattern', 'Unknown')
            }
            
            # Add lunar score if available
            for signal in results.get('integrated_signals', []):
                if signal['date'] == cluster['date']:
                    row['Lunar_Score'] = signal.get('lunar_score', 0)
                    row['Signal_Strength'] = signal.get('signal_strength', 'Unknown')
                    break
            
            clusters_data.append(row)
        
        # Save to CSV
        clusters_df = pd.DataFrame(clusters_data)
        clusters_df.to_csv(f"{csv_dir}/{symbol}_lunar_clusters.csv", index=False)
        print(f"Lunar clusters saved to {csv_dir}/{symbol}_lunar_clusters.csv")
    
    # Save planetary aspects
    planetary_aspects = results.get('planetary_results', {}).get('planetary_aspects', [])
    
    if planetary_aspects:
        aspects_data = []
        
        for aspect in planetary_aspects:
            # Get planet and aspect names
            if PLANETARY_ASPECTS_AVAILABLE:
                planet1 = get_planet_name(aspect['planet1'])
                planet2 = get_planet_name(aspect['planet2'])
                aspect_name = get_aspect_name(aspect['aspect'])
            else:
                planet1 = str(aspect['planet1'])
                planet2 = str(aspect['planet2'])
                aspect_name = str(aspect['aspect'])
            
            row = {
                'Date': aspect['date'],
                'Planet1': planet1,
                'Planet2': planet2,
                'Aspect': aspect_name,
                'Is_Heliocentric': aspect.get('is_heliocentric', False),
                'Orb': aspect.get('orb', 0)
            }
            
            if 'sequence_number' in aspect:
                row['Sequence_Number'] = aspect['sequence_number']
                row['Total_In_Sequence'] = aspect['total_in_sequence']
            
            aspects_data.append(row)
        
        # Save to CSV
        aspects_df = pd.DataFrame(aspects_data)
        aspects_df.to_csv(f"{csv_dir}/{symbol}_planetary_aspects.csv", index=False)
        print(f"Planetary aspects saved to {csv_dir}/{symbol}_planetary_aspects.csv")
    
    # Save integrated signals
    integrated_signals = results.get('integrated_signals', [])
    
    if integrated_signals:
        signals_data = []
        
        for signal in integrated_signals:
            row = {
                'Date': signal['date'],
                'Lunar_Score': signal.get('lunar_score', 0),
                'Planetary_Score': signal.get('planetary_score', 0),
                'Combined_Score': signal.get('combined_score', 0),
                'Signal_Strength': signal.get('signal_strength', 'Unknown'),
                'Is_Turning_Point': signal.get('is_turning_point', False),
                'Pattern': signal.get('pattern', 'Unknown')
            }
            
            # Add price info if available
            if 'price_info' in signal:
                price_info = signal['price_info']
                for key, value in price_info.items():
                    row[f'Price_{key}'] = value
            
            signals_data.append(row)
        
        # Save to CSV
        signals_df = pd.DataFrame(signals_data)
        signals_df.to_csv(f"{csv_dir}/{symbol}_integrated_signals.csv", index=False)
        print(f"Integrated signals saved to {csv_dir}/{symbol}_integrated_signals.csv")
    
    # Save backtest results
    backtest_results = results.get('backtest_results')
    
    if backtest_results is not None and not backtest_results.empty:
        backtest_results.to_csv(f"{csv_dir}/{symbol}_backtest_results.csv", index=False)
        print(f"Backtest results saved to {csv_dir}/{symbol}_backtest_results.csv")
    
    # Save performance statistics
    performance_stats = results.get('performance_stats')
    
    if performance_stats:
        with open(f"{csv_dir}/{symbol}_performance_stats.json", 'w') as f:
            json.dump(performance_stats, f, indent=4, default=str)
        print(f"Performance statistics saved to {csv_dir}/{symbol}_performance_stats.json")

def main():
    """Main function to run the enhanced lunar analysis."""
    print("Enhanced Lunar Cycle Analysis Tool")
    print("---------------------------------")
    
    # Get date range
    print("Enter start date (YYYY-MM-DD):")
    start_date_str = input()
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format. Using default of 1 year ago.")
        start_date = datetime.datetime.now().date() - datetime.timedelta(days=365)
    
    print("Enter end date (YYYY-MM-DD):")
    end_date_str = input()
    try:
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format. Using default of today.")
        end_date = datetime.datetime.now().date()
    
    # Get market symbols
    print("Enter market symbol(s) to analyze (comma-separated, e.g., SPX,GOLD,SILVER):")
    symbol_input = input()
    market_symbols = [s.strip() for s in symbol_input.split(',')]
    
    if not market_symbols:
        print("No market symbols provided. Using S&P 500 (SPX) as default.")
        market_symbols = ['SPX']
    
    # Get lunar factors to analyze
    print("Select lunar factors to analyze (comma-separated, e.g., 1,2,3,4,5):")
    print("1. Moon declination (max/zero)")
    print("2. True Node direction changes")
    print("3. Moon phases (New, Full, Quarter)")
    print("4. Apogee/Perigee")
    print("5. Moon void of course")
    print("Or enter 'all' to include all factors:")
    
    factors_input = input().strip()
    if factors_input.lower() == 'all':
        lunar_factors = [1, 2, 3, 4, 5]
    else:
        try:
            lunar_factors = [int(f.strip()) for f in factors_input.split(',')]
        except ValueError:
            print("Invalid input. Using all factors as default.")
            lunar_factors = [1, 2, 3, 4, 5]
    
    # Get cluster threshold
    print("Minimum number of lunar factors required to consider a date as a 'cluster' (2-5):")
    try:
        cluster_threshold = int(input())
        if cluster_threshold < 2 or cluster_threshold > 5:
            print("Value out of range. Using default of 2.")
            cluster_threshold = 2
    except ValueError:
        print("Invalid input. Using default of 2.")
        cluster_threshold = 2
    
    # Get analysis window
    print("Number of days to analyze before/after each lunar cluster (default 10):")
    try:
        window_days = int(input())
        if window_days < 1:
            print("Value too small. Using default of 10.")
            window_days = 10
    except ValueError:
        print("Invalid input. Using default of 10.")
        window_days = 10
    
    # Run analysis
    print("\nRunning enhanced lunar analysis...")
    start_time = time.time()
    
    market_data_dict, results_dict = analyze_multiple_markets(
        market_symbols, start_date, end_date, window_days, cluster_threshold)
    
    # Calculate time elapsed
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.1f} seconds.")
    
    print("\nResults have been saved to the corresponding directories.")
    print("Visualizations can be found in the 'Visualizations' subdirectories.")

if __name__ == "__main__":
    main()