#!/usr/bin/env python3
"""
Lunar Cycle and Planetary Aspects Confluence Analysis

This script analyzes the confluence of lunar cycle events and planetary aspects
to identify potential correlations with market turning points.

It combines lunar factors (Moon phases, declination, etc.) with planetary aspects
(conjunctions, oppositions, etc.) and tests whether specific combinations have
predictive value for market reversals.
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
import warnings
import random
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")

# Import lunar cycle analysis functions
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

# Check if planetary aspects module is available
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
    print("Planetary aspects module not found. Using alternative implementation.")
    
    # Provide basic planetary aspect functions if module not available
    def get_planet_name(planet_id):
        """Get the name of a planet from its Swiss Ephemeris ID."""
        names = {
            swe.MERCURY: "Mercury",
            swe.VENUS: "Venus",
            swe.MARS: "Mars",
            swe.JUPITER: "Jupiter",
            swe.SATURN: "Saturn",
            swe.URANUS: "Uranus",
            swe.NEPTUNE: "Neptune",
            swe.PLUTO: "Pluto"
        }
        return names.get(planet_id, str(planet_id))
    
    def get_aspect_name(aspect):
        """Get the name of an aspect from its angle."""
        names = {
            0: "Conjunction",
            60: "Sextile",
            90: "Square",
            120: "Trine",
            180: "Opposition"
        }
        return names.get(aspect, f"{aspect}°")
    
    def get_aspect_symbol(aspect):
        """Return the symbol for a given aspect."""
        symbols = {
            0: "☌",   # Conjunction
            60: "⚹",  # Sextile
            90: "□",  # Square
            120: "△", # Trine
            180: "☍"  # Opposition
        }
        return symbols.get(aspect, str(aspect) + "°")

# Global configuration
CONFIG = {
    "default_start_date": datetime.datetime(2000, 1, 1).date(),
    "default_end_date": datetime.datetime(2020, 12, 31).date(),
    "default_market": "SPX",
    "confluence_windows": [1, 3, 5],  # Days for confluence window
    "aspect_orbs": {
        "tight": 1.0,   # 1 degree orb
        "standard": 2.0  # 2 degree orb
    },
    "turning_point_definitions": {
        "short": {"percent": 3, "days": 5},
        "medium": {"percent": 5, "days": 10},
        "long": {"percent": 8, "days": 20}
    }
}

def find_planetary_aspects(start_date, end_date, orb=1.0, focus_planets=None):
    """
    Find all major aspects between planets in the given date range.
    
    Args:
        start_date: Beginning of search range
        end_date: End of search range
        orb: Maximum allowed orb in degrees
        focus_planets: Optional list of specific planets to focus on
        
    Returns:
        List of dictionaries with aspect details
    """
    # Define planets to analyze - use focus_planets if provided
    if focus_planets:
        planets = focus_planets
    else:
        planets = [
            swe.MERCURY,
            swe.VENUS,
            swe.MARS,
            swe.JUPITER,
            swe.SATURN,
            swe.URANUS,
            swe.NEPTUNE,
            swe.PLUTO
        ]

# 1. Add aspect weighting by planet type
planet_weights = {
    swe.MERCURY: 1.0,
    swe.VENUS: 1.0,
    swe.MARS: 1.0,
    swe.JUPITER: 1.0,
    swe.SATURN: 1.0,
    swe.URANUS: 1.0,
    swe.NEPTUNE: 1.0,
    swe.PLUTO: 1.0
}

# 2. Add more aspect types
aspects_to_find = [
    0,    # Conjunction
    30,   # Semi-sextile
    45,   # Semi-square
    60,   # Sextile
    72,   # Quintile
    90,   # Square
    120,  # Trine
    135,  # Sesqui-square
    150,  # Quincunx
    180   # Opposition
]

# 3. Add a custom filter parameter
def find_planetary_aspects(start_date, end_date, orb=1.0, 
                          focus_planets=None, focus_aspects=None):
    # If focus planets specified, use only those
    if focus_planets:
        planets = focus_planets
    else:
        planets = [
            swe.MERCURY,
            swe.VENUS,
            swe.MARS,
            swe.JUPITER,
            swe.SATURN,
            swe.URANUS,
            swe.NEPTUNE,
            swe.PLUTO
        ]
    
    # Define aspects to find
    aspects_to_find = [0, 60, 90, 120, 180]  # Conjunction, Sextile, Square, Trine, Opposition
    
    # Create all possible planet pairs
    planet_pairs = []
    for i in range(len(planets)):
        for j in range(i+1, len(planets)):
            planet_pairs.append((planets[i], planets[j]))
    
    # Use imported function if available
    if PLANETARY_ASPECTS_AVAILABLE:
        return find_aspects_between_planets(
            start_date, 
            end_date, 
            planet_pairs, 
            aspects_to_find, 
            orb=orb, 
            is_heliocentric=False
        )
    else:
        # Basic implementation if module not available
        print("Using simplified planetary aspect calculation.")
        # This is a simplified implementation - in a real app you would
        # need to implement the full aspect calculation logic
        return []

def get_market_data(symbol, start_date, end_date):
    """
    Get market data from Yahoo Finance.
    
    Args:
        symbol: Market symbol to analyze
        start_date: Beginning of analysis period
        end_date: End of analysis period
        
    Returns:
        DataFrame with price history
    """
    # Format symbol for Yahoo Finance
    if symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
        yf_symbol = '^GSPC'
    elif symbol.upper() in ['DJI', 'DJIA', 'DOW']:
        yf_symbol = '^DJI'
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
    elif symbol.upper() in ['OIL', 'CRUDE', 'CL']:
        yf_symbol = 'CL=F'  # Crude Oil Futures
    elif symbol.upper() in ['EURUSD', 'EUR/USD', 'EUR']:
        yf_symbol = 'EURUSD=X'  # Euro to USD exchange rate
    elif symbol.upper() in ['BTCUSD', 'BTC/USD', 'BITCOIN']:
        yf_symbol = 'BTC-USD'  # Bitcoin to USD 
    elif symbol.upper() in ['DXY', 'DOLLAR', 'USD']:
        yf_symbol = 'DX-Y.NYB'  # US Dollar Index
    else:
        yf_symbol = symbol
    
    try:
        # Extend date range for technical indicators
        extended_start = start_date - datetime.timedelta(days=200)  # Extra days for MA calculation
        
        # Get market data - first attempt with download method
        print(f"Downloading data for {yf_symbol}...")
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
        time.sleep(1)
        
        # Calculate moving averages for market regime identification
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change() * 100
        
        # Calculate MA_Pct safely
        try:
            # Handle NaN values properly
            data['MA_Pct'] = ((data['Close'] - data['MA200']) / data['MA200']) * 100
            
            # Identify market regimes
            data['Market_Regime'] = 'Sideways'
            data.loc[data['MA_Pct'] > 5, 'Market_Regime'] = 'Bull'
            data.loc[data['MA_Pct'] < -5, 'Market_Regime'] = 'Bear'
        except Exception as e:
            print(f"Warning: Could not calculate market regimes: {e}")
            # Ensure Market_Regime column exists even if calculation fails
            data['Market_Regime'] = 'Unknown'
        
        # Trim to requested date range
        mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
        trimmed_data = data.loc[mask].copy()
        
        return trimmed_data
        
    except Exception as e:
        print(f"Error getting market data for {symbol}: {e}")
        return pd.DataFrame()

def identify_turning_points(price_data, definition):
    """
    Identify market turning points based on specified definition.
    
    Args:
        price_data: DataFrame with price history
        definition: Dictionary with percent and days thresholds
        
    Returns:
        DataFrame with turning points
    """
    percent = definition['percent']
    days = definition['days']
    
    turning_points = []
    
    # Loop through each date in the price data
    for i in range(len(price_data)):
        if i < 10 or i >= len(price_data) - days:
            continue  # Skip edges of data where we can't look back/forward enough
            
        current_date = price_data.index[i]
        date = current_date.date()
        price = float(price_data['Close'].iloc[i])
        
        # Look back to check if it's a potential high or low
        prev_window = price_data.iloc[i-10:i]
        if len(prev_window) == 0:
            continue
            
        prev_max = float(prev_window['Close'].max())
        prev_min = float(prev_window['Close'].min())
        
        # Get future window
        if i + days + 1 > len(price_data):
            continue  # Not enough future data
            
        future_window = price_data.iloc[i+1:i+days+1]
        if len(future_window) < days:
            continue  # Not enough future data
            
        # Check for high (price higher than previous and followed by decline)
        if price >= prev_max:
            future_min = float(future_window['Close'].min())
            decline_pct = ((future_min - price) / price) * 100
            
            if decline_pct <= -percent:
                # Find the index of minimum price in future window
                min_idx_position = future_window['Close'].values.argmin()
                if min_idx_position < len(future_window):
                    min_date = future_window.index[min_idx_position]
                    days_to_min = (min_date - current_date).days
                else:
                    days_to_min = None
                
                turning_points.append({
                    'date': date,
                    'price': price,
                    'type': 'High',
                    'percent_move': decline_pct,
                    'days_to_move': days_to_min
                })
        
        # Check for low (price lower than previous and followed by advance)
        if price <= prev_min:
            future_max = float(future_window['Close'].max())
            advance_pct = ((future_max - price) / price) * 100
            
            if advance_pct >= percent:
                # Find the index of maximum price in future window
                max_idx_position = future_window['Close'].values.argmax()
                if max_idx_position < len(future_window):
                    max_date = future_window.index[max_idx_position]
                    days_to_max = (max_date - current_date).days
                else:
                    days_to_max = None
                
                turning_points.append({
                    'date': date,
                    'price': price,
                    'type': 'Low',
                    'percent_move': advance_pct,
                    'days_to_move': days_to_max
                })
    
    return pd.DataFrame(turning_points)

def find_lunar_events(start_date, end_date):
    """
    Find all lunar cycle events in the given date range.
    
    Args:
        start_date: Beginning of search range
        end_date: End of search range
        
    Returns:
        Dictionary with lunar events by type
    """
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
    
    # Combine all events into a flat list
    all_events = []
    for category, events in lunar_events.items():
        for event in events:
            event['category'] = category
            all_events.append(event)
    
    # Sort by date
    all_events.sort(key=lambda x: x['date'])
    
    return {
        'by_category': lunar_events,
        'all_events': all_events
    }

def find_confluences(lunar_events, planetary_aspects, window_days):
    """
    Find confluences of lunar events and planetary aspects within specified window.
    
    Args:
        lunar_events: List of lunar cycle events
        planetary_aspects: List of planetary aspects
        window_days: Number of days to consider for confluence
        
    Returns:
        List of confluence events
    """
    confluences = []
    
    # Create date indexes for faster lookup
    aspect_dates = {}
    for aspect in planetary_aspects:
        aspect_date = aspect['date']
        if aspect_date not in aspect_dates:
            aspect_dates[aspect_date] = []
        aspect_dates[aspect_date].append(aspect)
    
    # Check each lunar event for nearby planetary aspects
    for lunar_event in lunar_events:
        lunar_date = lunar_event['date']
        
        # Define the window range
        start_window = lunar_date - datetime.timedelta(days=window_days)
        end_window = lunar_date + datetime.timedelta(days=window_days)
        
        # Find aspects within the window
        nearby_aspects = []
        
        for days_delta in range(-window_days, window_days + 1):
            check_date = lunar_date + datetime.timedelta(days=days_delta)
            
            if check_date in aspect_dates:
                for aspect in aspect_dates[check_date]:
                    # Calculate days between events
                    days_between = abs((aspect['date'] - lunar_date).days)
                    
                    nearby_aspects.append({
                        'aspect': aspect,
                        'days_between': days_between
                    })
        
        # Create a confluence if there are nearby aspects
        if nearby_aspects:
            for nearby in nearby_aspects:
                confluences.append({
                    'lunar_event': lunar_event,
                    'planetary_aspect': nearby['aspect'],
                    'days_between': nearby['days_between'],
                    'confluence_date': lunar_date,  # Use lunar date as reference
                    'window_days': window_days
                })
    
    return confluences

def analyze_confluence_effectiveness(confluences, turning_points, market_data, tp_definition):
    """
    Analyze effectiveness of confluences in predicting turning points.
    
    Args:
        confluences: List of confluence events
        turning_points: DataFrame with turning points
        market_data: DataFrame with price history
        tp_definition: Dictionary with turning point definition parameters
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    
    # Convert turning points dates to a set for faster lookup
    tp_dates = set(turning_points['date'])
    
    # Analyze each confluence
    for confluence in confluences:
        # Get basic information
        confluence_date = confluence['confluence_date']
        lunar_type = confluence['lunar_event']['type']
        lunar_category = confluence['lunar_event']['category']
        
        aspect_type = get_aspect_name(confluence['planetary_aspect']['aspect'])
        planet1 = get_planet_name(confluence['planetary_aspect']['planet1'])
        planet2 = get_planet_name(confluence['planetary_aspect']['planet2'])
        
        aspect_description = f"{planet1}-{planet2} {aspect_type}"
        
        # Determine if this is a turning point
        is_turning_point = False
        turning_point_type = None
        percent_move = None
        days_to_move = None
        
        # Check for exact match
        if confluence_date in tp_dates:
            is_turning_point = True
            tp_info = turning_points[turning_points['date'] == confluence_date].iloc[0]
            turning_point_type = tp_info['type']
            percent_move = tp_info['percent_move']
            days_to_move = tp_info['days_to_move']
        else:
            # Check nearby dates (within window)
            window_days = tp_definition['days'] // 2  # Half the evaluation window
            
            for days_delta in range(-window_days, window_days + 1):
                check_date = confluence_date + datetime.timedelta(days=days_delta)
                
                if check_date in tp_dates:
                    is_turning_point = True
                    tp_info = turning_points[turning_points['date'] == check_date].iloc[0]
                    turning_point_type = tp_info['type']
                    percent_move = tp_info['percent_move']
                    days_to_move = tp_info['days_to_move']
                    break
        
        # Get market regime if available
        market_regime = None
        closest_idx = None
        
        for idx in market_data.index:
            if idx.date() == confluence_date:
                closest_idx = idx
                break
            elif idx.date() > confluence_date:
                closest_idx = idx
                break
        
        if closest_idx is not None and 'Market_Regime' in market_data.columns:
            market_regime = market_data.loc[closest_idx, 'Market_Regime']
        
        # Create result entry
        result = {
            'date': confluence_date,
            'lunar_event': lunar_type,
            'lunar_category': lunar_category,
            'planetary_aspect': aspect_description,
            'days_between': confluence['days_between'],
            'is_turning_point': is_turning_point,
            'turning_point_type': turning_point_type,
            'percent_move': percent_move,
            'days_to_move': days_to_move,
            'market_regime': market_regime,
            'confluence_key': f"{lunar_type} + {aspect_description}"
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def generate_random_dates(start_date, end_date, num_dates):
    """
    Generate random dates within the specified range.
    
    Args:
        start_date: Beginning of date range
        end_date: End of date range
        num_dates: Number of random dates to generate
        
    Returns:
        List of random datetime.date objects
    """
    # Convert to timestamps for easier random generation
    start_ts = datetime.datetime.combine(start_date, datetime.time()).timestamp()
    end_ts = datetime.datetime.combine(end_date, datetime.time()).timestamp()
    
    # Generate random timestamps
    random_timestamps = [random.uniform(start_ts, end_ts) for _ in range(num_dates)]
    
    # Convert back to dates
    random_dates = [datetime.datetime.fromtimestamp(ts).date() for ts in random_timestamps]
    
    return random_dates

def calculate_statistical_significance(results_df, market_data, tp_definition, num_samples=1000):
    """
    Calculate statistical significance of confluence results compared to random dates.
    
    Args:
        results_df: DataFrame with confluence analysis results
        market_data: DataFrame with price history
        tp_definition: Dictionary with turning point definition parameters
        num_samples: Number of random samples to generate
        
    Returns:
        Dictionary with statistical test results
    """
    # Count turning points in confluences
    confluence_count = len(results_df)
    confluence_tp_count = results_df['is_turning_point'].sum()
    confluence_tp_rate = confluence_tp_count / confluence_count if confluence_count > 0 else 0
    
    # Generate random dates
    start_date = market_data.index[0].date()
    end_date = market_data.index[-1].date()
    random_dates = generate_random_dates(start_date, end_date, num_samples)
    
    # Identify turning points for random dates
    random_tp_count = 0
    
    # Get turning points from market data
    turning_points = identify_turning_points(market_data, tp_definition)
    tp_dates = set(turning_points['date'])
    
    # Check how many random dates are near turning points
    for random_date in random_dates:
        # Check window around the random date
        window_days = tp_definition['days'] // 2  # Half the evaluation window
        
        for days_delta in range(-window_days, window_days + 1):
            check_date = random_date + datetime.timedelta(days=days_delta)
            
            if check_date in tp_dates:
                random_tp_count += 1
                break
    
    random_tp_rate = random_tp_count / num_samples
    
    # Perform chi-squared test
    # Create 2x2 contingency table:
    # [TP confluences, non-TP confluences]
    # [TP random dates, non-TP random dates]
    observed = np.array([
        [confluence_tp_count, confluence_count - confluence_tp_count],
        [random_tp_count, num_samples - random_tp_count]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    # Calculate relative effectiveness
    relative_effectiveness = confluence_tp_rate / random_tp_rate if random_tp_rate > 0 else float('inf')
    
    return {
        'confluence_tp_rate': confluence_tp_rate,
        'random_tp_rate': random_tp_rate,
        'relative_effectiveness': relative_effectiveness,
        'chi2': chi2,
        'p_value': p_value,
        'is_significant': p_value < 0.05
    }

def summarize_combinations(results_df, min_occurrences=3, inner_planets_only=False):
    """
    Summarize effectiveness of specific lunar+planetary combinations.
    
    Args:
        results_df: DataFrame with confluence analysis results
        min_occurrences: Minimum number of occurrences to consider a combination
        
    Returns:
        DataFrame with summary statistics by combination
    """
    # Filter results if inner_planets_only is True
    if inner_planets_only:
        inner_planet_aspects = []
        for idx, row in results_df.iterrows():
            aspect_desc = row['planetary_aspect']
            # Check if aspect involves Mercury, Venus or Mars
            if any(p in aspect_desc for p in ['Mercury', 'Venus', 'Mars']):
                inner_planet_aspects.append(idx)
        
        # Filter dataframe to only include inner planet aspects
        results_df = results_df.loc[inner_planet_aspects]

    # Group by confluence key
    summary = []
    
    for key, group in results_df.groupby('confluence_key'):
        occurrences = len(group)
        
        if occurrences >= min_occurrences:
            successful = group['is_turning_point'].sum()
            success_rate = successful / occurrences
            
            # Calculate average percent move where applicable
            avg_move = group.loc[group['percent_move'].notnull(), 'percent_move'].mean()
            
            # Get regime effectiveness if applicable
            regime_stats = {}
            
            # Only try to group by market_regime if it's a valid column with simple values
            if 'market_regime' in group.columns:
                try:
                    # Check if market_regime contains simple values
                    sample_regime = group['market_regime'].iloc[0]
                    if isinstance(sample_regime, (str, int, float)) or sample_regime is None:
                        for regime, regime_group in group.groupby('market_regime'):
                            regime_occurrences = len(regime_group)
                            regime_successful = regime_group['is_turning_point'].sum()
                            regime_success_rate = regime_successful / regime_occurrences if regime_occurrences > 0 else 0
                            
                            regime_stats[regime] = {
                                'occurrences': regime_occurrences,
                                'success_rate': regime_success_rate
                            }
                except (TypeError, ValueError):
                    # If groupby fails, just skip the market regime analysis
                    pass
            
            lunar_event = group['lunar_event'].iloc[0]
            lunar_category = group['lunar_category'].iloc[0]
            planetary_aspect = group['planetary_aspect'].iloc[0]
            
            summary.append({
                'combination': key,
                'lunar_event': lunar_event,
                'lunar_category': lunar_category,
                'planetary_aspect': planetary_aspect,
                'occurrences': occurrences,
                'successful': successful,
                'success_rate': success_rate,
                'avg_percent_move': avg_move,
                'regime_stats': regime_stats
            })
    
    # Convert to DataFrame and sort by success rate
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('success_rate', ascending=False)
        return summary_df
    else:
        # Return empty DataFrame with the same columns
        return pd.DataFrame(columns=[
            'combination', 'lunar_event', 'lunar_category', 'planetary_aspect',
            'occurrences', 'successful', 'success_rate', 'avg_percent_move', 'regime_stats'
        ])

def generate_trading_rules(summary_df, stats_results, window_days, threshold=0.7):
    """
    Generate plain-language trading rules from analysis results.
    
    Args:
        summary_df: DataFrame with summary statistics by combination
        stats_results: Dictionary with statistical significance results
        window_days: Window size used for confluence analysis
        threshold: Minimum success rate to include in rules
        
    Returns:
        List of trading rule strings
    """
    trading_rules = []
    
    # Only consider statistically significant results
    if not stats_results.get('is_significant', False):
        trading_rules.append("CAUTION: Analysis results did not achieve statistical significance.")
        trading_rules.append(f"Confluence turning point rate: {stats_results['confluence_tp_rate']:.1%}")
        trading_rules.append(f"Random date turning point rate: {stats_results['random_tp_rate']:.1%}")
        trading_rules.append(f"P-value: {stats_results['p_value']:.4f} (threshold: 0.05)")
        return trading_rules
    
    # Add overall statistics
    trading_rules.append(f"OVERALL: Lunar-Planetary confluences are {stats_results['relative_effectiveness']:.1f}x more effective than random dates")
    trading_rules.append(f"Confluence turning point rate: {stats_results['confluence_tp_rate']:.1%}")
    trading_rules.append(f"Statistical significance: p-value {stats_results['p_value']:.4f}")
    trading_rules.append("")
    
    # Add specific combination rules
    if not summary_df.empty:
        trading_rules.append("TOP COMBINATIONS:")
        
        for _, row in summary_df.iterrows():
            if row['success_rate'] >= threshold:
                rule = f"When a {row['lunar_event']} occurs within {window_days} days of a {row['planetary_aspect']}, "
                rule += f"a market turning point occurred in {row['success_rate']:.1%} of instances "
                rule += f"({row['successful']} out of {row['occurrences']} occurrences)"
                
                # Add regime-specific information if available
                regime_info = row.get('regime_stats', {})
                if regime_info:
                    best_regime = None
                    best_rate = 0
                    
                    for regime, stats in regime_info.items():
                        if stats['occurrences'] >= 3 and stats['success_rate'] > best_rate:
                            best_regime = regime
                            best_rate = stats['success_rate']
                    
                    if best_regime:
                        rule += f"\n    → Best in {best_regime} markets: {best_rate:.1%} success rate"
                
                # Add price movement
                if pd.notnull(row['avg_percent_move']):
                    rule += f"\n    → Average price move: {abs(row['avg_percent_move']):.1f}%"
                
                trading_rules.append(rule)
                trading_rules.append("")
    
    if len(trading_rules) <= 5:  # Just the overall stats
        trading_rules.append("No combinations met the minimum success threshold.")
    
    return trading_rules

def visualize_results(results_df, summary_df, market_data, stats_results, directory, market_symbol, confluence_window, aspect_orb, tp_definition):
    """
    Create visualizations of analysis results.
    
    Args:
        results_df: DataFrame with confluence analysis results
        summary_df: DataFrame with summary statistics
        market_data: DataFrame with price history
        stats_results: Dictionary with statistical significance results
        directory: Directory to save visualizations
        market_symbol: Symbol for the market being analyzed
        confluence_window: Number of days for confluence window
        aspect_orb: Maximum orb for planetary aspects
        tp_definition: Dictionary with turning point definition parameters
    """
    os.makedirs(directory, exist_ok=True)
    
    # Get date info for parameters
    start_date = min(market_data.index).strftime('%Y-%m-%d') if not market_data.empty else "Unknown"
    end_date = max(market_data.index).strftime('%Y-%m-%d') if not market_data.empty else "Unknown"
    
    # Create parameter text for the text box
    param_text = "ANALYSIS PARAMETERS\n"
    param_text += "-------------------\n"
    param_text += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    param_text += f"Market Symbol: {market_symbol.upper()}\n"
    param_text += f"Date Range: {start_date} to {end_date}\n"
    param_text += f"Confluence Window: {confluence_window} days\n"
    param_text += f"Aspect Orb: {aspect_orb}°\n"
    param_text += f"Turning Point Definition: {tp_definition['percent']}% move within {tp_definition['days']} days"
    
    # 1. Timeline chart showing confluences and turning points
    plt.figure(figsize=(14, 8))
    
    # Plot price data
    plt.plot(market_data.index, market_data['Close'], 'k-', alpha=0.7)
    
    # Mark each confluence
    for _, confluence in results_df.iterrows():
        date = confluence['date']
        is_tp = confluence['is_turning_point']
        
        # Find closest market data point
        closest_idx = None
        min_diff = float('inf')
        
        for idx in market_data.index:
            diff = abs((idx.date() - date).days)
            if diff < min_diff:
                min_diff = diff
                closest_idx = idx
        
        if closest_idx is not None:
            price = market_data.loc[closest_idx, 'Close']
            
            # Use different colors for turning points vs non-turning points
            color = 'red' if is_tp else 'blue'
            marker = 'o' if is_tp else 's'
            
            plt.plot(closest_idx, price, marker=marker, color=color, markersize=6)
    
    plt.title(f"Lunar-Planetary Confluences and Market Turning Points for {market_symbol.upper()}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    
    # Create legend
    plt.plot([], [], 'ro', markersize=6, label='Turning Point')
    plt.plot([], [], 'bs', markersize=6, label='Non-Turning Point')
    
    # Add statistics text box
    stats_text = f"Market: {market_symbol.upper()}\n"
    stats_text += f"Statistical Significance: {'Yes (p < 0.05)' if stats_results.get('is_significant', False) else 'No (p > 0.05)'}\n"
    stats_text += f"Confluence Rate: {stats_results.get('confluence_tp_rate', 0)*100:.1f}%\n"
    stats_text += f"Random Date Rate: {stats_results.get('random_tp_rate', 0)*100:.1f}%\n"
    stats_text += f"P-value: {stats_results.get('p_value', 1):.4f}"
    
    # Position the text box in figure coords
    plt.figtext(0.15, 0.02, stats_text, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save chart
    plt.savefig(f"{directory}/timeline_chart.png")
    plt.close()
    
    # 2. Success rate by combination (top 10)
    if not summary_df.empty and len(summary_df) > 0:
        # Sort by success rate
        top_combos = summary_df.sort_values('success_rate', ascending=False).head(10)
        
        plt.figure(figsize=(12, 8))
        
        # Create bars with colors based on success rate
        bars = plt.bar(top_combos['combination'], top_combos['success_rate'] * 100)
        
        # Color bars based on success rate
        for i, bar in enumerate(bars):
            success_rate = top_combos.iloc[i]['success_rate']
            if success_rate >= 0.8:
                bar.set_color('green')
            elif success_rate >= 0.6:
                bar.set_color('yellowgreen')
            elif success_rate >= 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add occurrence counts as text
        for i, v in enumerate(top_combos['success_rate']):
            occurrences = top_combos.iloc[i]['occurrences']
            plt.text(i, v * 100 + 2, f"n={occurrences}", ha='center')

        # Customize x-axis
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Combination')
        plt.ylabel('Success Rate (%)')
        plt.title(f'Top 10 Lunar-Planetary Combinations for {market_symbol.upper()} by Success Rate')
        
        # Add statistics text box
        stats_text = f"Market: {market_symbol.upper()}\n"
        stats_text += f"Statistical Significance: {'Yes (p < 0.05)' if stats_results.get('is_significant', False) else 'No (p > 0.05)'}\n"
        stats_text += f"Confluence Rate: {stats_results.get('confluence_tp_rate', 0)*100:.1f}%\n"
        stats_text += f"Random Date Rate: {stats_results.get('random_tp_rate', 0)*100:.1f}%\n"
        stats_text += f"P-value: {stats_results.get('p_value', 1):.4f}"
        
        # Position the text box in figure coords
        plt.figtext(0.15, 0.01, stats_text, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        plt.tight_layout()
        
        # Save chart
        plt.savefig(f"{directory}/top_combinations.png")
        plt.close()
    
    # 3. Statistical significance visualization
    plt.figure(figsize=(8, 6))
    
    labels = ['Confluences', 'Random Dates']
    rates = [stats_results['confluence_tp_rate'] * 100, stats_results['random_tp_rate'] * 100]
    
    bars = plt.bar(labels, rates, color=['blue', 'gray'])
    
    # Add significance annotation
    if stats_results['is_significant']:
        plt.text(0.5, max(rates) + 5, 'Statistically Significant (p < 0.05)', 
                ha='center', color='green', fontweight='bold')
    else:
        plt.text(0.5, max(rates) + 5, 'Not Statistically Significant (p > 0.05)', 
                ha='center', color='red', fontweight='bold')
    
    # Add p-value
    plt.text(0.5, max(rates) + 2, f"p-value: {stats_results['p_value']:.4f}", ha='center')
    plt.text(0.5, max(rates) - 5, f"Market: {market_symbol.upper()}", ha='center', fontweight='bold')
    
    plt.ylabel('Turning Point Rate (%)')
    plt.title(f'Confluence Effectiveness vs. Random Dates for {market_symbol.upper()}')
    
    # Save chart
    plt.savefig(f"{directory}/statistical_significance.png")
    plt.close()
    
    # Skip the market regime visualization that was causing problems
    print("Note: Market regime visualization has been skipped to avoid errors.")

def save_to_csv(results_df, summary_df, tp_definition, directory):
    """
    Save analysis results to CSV files.
    
    Args:
        results_df: DataFrame with confluence analysis results
        summary_df: DataFrame with summary statistics
        tp_definition: Dictionary with turning point definition parameters
        directory: Directory to save CSV files
    """
    os.makedirs(directory, exist_ok=True)
    
    # Save all confluences
    results_df.to_csv(f"{directory}/all_confluences.csv", index=False)
    
    # Save summary of combinations
    if not summary_df.empty:
        # Convert regime_stats to separate columns
        if 'regime_stats' in summary_df.columns:
            # Extract regime information into separate columns
            regimes = []
            for stats in summary_df['regime_stats']:
                for regime, regime_data in stats.items():
                    regime_col = f"{regime}_success_rate"
                    if regime_col not in summary_df.columns:
                        summary_df[regime_col] = None
                    
                    idx = summary_df[summary_df['regime_stats'] == stats].index
                    summary_df.loc[idx, regime_col] = regime_data['success_rate']
            
            # Drop regime_stats column
            summary_df = summary_df.drop('regime_stats', axis=1)
        
        summary_df.to_csv(f"{directory}/combination_summary.csv", index=False)
    
    # Save turning point definition
    pd.DataFrame([tp_definition]).to_csv(f"{directory}/turning_point_definition.csv", index=False)

def save_trading_rules(trading_rules, directory):
    """
    Save trading rules to a text file.
    
    Args:
        trading_rules: List of trading rule strings
        directory: Directory to save file
    """
    os.makedirs(directory, exist_ok=True)
    
    with open(f"{directory}/trading_rules.txt", 'w', encoding='utf-8') as f:
        for rule in trading_rules:
            f.write(rule + '\n')

def run_confluence_analysis(market_symbol, start_date, end_date, confluence_window, aspect_orb, tp_definition, focus_planets=None):
    """
    Run complete analysis of lunar-planetary confluences.
    
    Args:
        market_symbol: Symbol for the market to analyze
        start_date: Beginning of analysis period
        end_date: End of analysis period
        confluence_window: Number of days for confluence window
        aspect_orb: Maximum orb for planetary aspects
        tp_definition: Dictionary with turning point definition parameters
        focus_planets: Optional list of specific planets to focus on
        
    Returns:
        Dictionary with analysis results
    """

    print(f"\nRunning analysis for {market_symbol} from {start_date} to {end_date}")
    print(f"Confluence window: {confluence_window} days, Aspect orb: {aspect_orb}°")
    print(f"Turning point definition: {tp_definition['percent']}% move within {tp_definition['days']} days")
    
    # Create output directories
    analysis_name = f"{market_symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    base_dir = f"LUNAR_PLANETARY/{analysis_name}/window{confluence_window}_orb{aspect_orb}"
    csv_dir = f"{base_dir}/CSV"
    viz_dir = f"{base_dir}/Visualizations"
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # Save analysis parameters to a file
    parameters = [
        "ANALYSIS PARAMETERS",
        "-------------------",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Market Symbol: {market_symbol}",
        f"Date Range: {start_date} to {end_date}",
        f"Confluence Window: {confluence_window} days",
        f"Aspect Orb: {aspect_orb}°",
        f"Turning Point Definition: {tp_definition['percent']}% move within {tp_definition['days']} days"
    ]
    
    # Write parameters to file
    with open(f"{base_dir}/parameters.txt", 'w') as f:
        for line in parameters:
            f.write(line + '\n')
    
    # Step 1: Get market data
    print("\nRetrieving market data...")
    market_data = get_market_data(market_symbol, start_date, end_date)
    
    if market_data.empty:
        print(f"Error: No market data available for {market_symbol}.")
        return None
    
    print(f"Retrieved {len(market_data)} days of market data.")
    
    # Step 2: Find planetary aspects
    print("\nFinding planetary aspects...")
    planetary_aspects = find_planetary_aspects(start_date, end_date, orb=aspect_orb, focus_planets=focus_planets)
    print(f"Found {len(planetary_aspects)} planetary aspects.")
    
    # Step 3: Find lunar events
    print("\nFinding lunar events...")
    lunar_events = find_lunar_events(start_date, end_date)
    all_lunar_events = lunar_events['all_events']
    print(f"Found {len(all_lunar_events)} total lunar events.")
    
    # Step 4: Identify market turning points
    print("\nIdentifying market turning points...")
    turning_points = identify_turning_points(market_data, tp_definition)
    print(f"Found {len(turning_points)} turning points.")
    
    # Step 5: Find confluences
    print("\nFinding lunar-planetary confluences...")
    confluences = find_confluences(all_lunar_events, planetary_aspects, confluence_window)
    print(f"Found {len(confluences)} confluences.")
    
    if len(confluences) == 0:
        print("Error: No confluences found. Try adjusting parameters.")
        return None
    
    # Step 6: Analyze effectiveness
    print("\nAnalyzing confluence effectiveness...")
    results_df = analyze_confluence_effectiveness(confluences, turning_points, market_data, tp_definition)
    print(f"Analyzed {len(results_df)} confluences.")
    
    # Step 7: Calculate statistical significance
    print("\nCalculating statistical significance...")
    stats_results = calculate_statistical_significance(results_df, market_data, tp_definition)
    significance_msg = "statistically significant" if stats_results['is_significant'] else "not statistically significant"
    print(f"Results are {significance_msg} (p-value: {stats_results['p_value']:.4f}).")
    print(f"Confluence turning point rate: {stats_results['confluence_tp_rate']:.1%}")
    print(f"Random date turning point rate: {stats_results['random_tp_rate']:.1%}")
    
    # Step 8: Summarize combinations
    print("\nSummarizing lunar-planetary combinations...")
    summary_df = summarize_combinations(results_df)
    print(f"Summarized {len(summary_df)} unique combinations.")
    
    # Step 9: Generate trading rules
    print("\nGenerating trading rules...")
    trading_rules = generate_trading_rules(summary_df, stats_results, confluence_window)
    
    # Step 10: Create visualizations
    print("\nCreating visualizations...")
    try:
        visualize_results(results_df, summary_df, market_data, stats_results, viz_dir, 
                        market_symbol, confluence_window, aspect_orb, tp_definition)
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")

    # Step 11: Save results to CSV
    print("\nSaving results to CSV...")
    save_to_csv(results_df, summary_df, tp_definition, csv_dir)
    save_trading_rules(trading_rules, csv_dir)
    
    print(f"\nAnalysis complete. Results saved to {base_dir}")
    
    return {
        'market_symbol': market_symbol,
        'results_df': results_df,
        'summary_df': summary_df,
        'stats_results': stats_results,
        'trading_rules': trading_rules,
        'base_dir': base_dir
    }

def main():
    """Main function to run the lunar-planetary confluence analysis."""
    print("Lunar Cycle and Planetary Aspects Confluence Analysis")
    print("---------------------------------------------------")
    
    # Get market symbol
    print("\nEnter market symbol (default: SPX):")
    symbol_input = input().strip()
    market_symbol = symbol_input if symbol_input else CONFIG['default_market']
    
    # Get date range
    print("\nEnter start date (YYYY-MM-DD, default: 2000-01-01):")
    start_date_str = input().strip()
    if start_date_str:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    else:
        start_date = CONFIG['default_start_date']
    
    print("\nEnter end date (YYYY-MM-DD, default: 2020-12-31):")
    end_date_str = input().strip()
    if end_date_str:
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    else:
        end_date = CONFIG['default_end_date']
    
    # Choose confluence window
    print("\nSelect confluence window (days between lunar and planetary events):")
    for i, window in enumerate(CONFIG['confluence_windows']):
        print(f"{i+1}. {window} days")
    
    window_choice = input("Enter choice (default: 1): ").strip()
    try:
        window_idx = int(window_choice) - 1
        if window_idx < 0 or window_idx >= len(CONFIG['confluence_windows']):
            window_idx = 0
    except:
        window_idx = 0
    
    confluence_window = CONFIG['confluence_windows'][window_idx]
    
    # Choose aspect orb
    print("\nSelect aspect orb precision:")
    print("1. Tight (1°)")
    print("2. Standard (2°)")
    
    orb_choice = input("Enter choice (default: 1): ").strip()
    if orb_choice == "2":
        aspect_orb = CONFIG['aspect_orbs']['standard']
    else:
        aspect_orb = CONFIG['aspect_orbs']['tight']
    
    # Choose turning point definition
    print("\nSelect turning point definition:")
    print("1. Short-term (3% move within 5 days)")
    print("2. Medium-term (5% move within 10 days)")
    print("3. Long-term (8% move within 20 days)")
    
    tp_choice = input("Enter choice (default: 2): ").strip()
    if tp_choice == "1":
        tp_definition = CONFIG['turning_point_definitions']['short']
    elif tp_choice == "3":
        tp_definition = CONFIG['turning_point_definitions']['long']
    else:
        tp_definition = CONFIG['turning_point_definitions']['medium']
    
    # ADD THE NEW CODE HERE - Planet focus selection
    print("\nSelect planet focus:")
    print("1. All planets")
    print("2. Fast-moving planets only (Mercury, Venus, Mars)")
    print("3. Slow-moving planets only (Jupiter through Pluto)")
    
    focus_choice = input("Enter choice (default: 1): ").strip()
    
    # Set up planet focus based on choice
    if focus_choice == "2":
        focus_planets = [swe.MERCURY, swe.VENUS, swe.MARS]
    elif focus_choice == "3":
        focus_planets = [swe.JUPITER, swe.SATURN, swe.URANUS, swe.NEPTUNE, swe.PLUTO]
    else:
        focus_planets = None
    
    # Run analysis - modify this call to include the new focus_planets parameter
    results = run_confluence_analysis(
        market_symbol,
        start_date,
        end_date,
        confluence_window,
        aspect_orb,
        tp_definition,
        focus_planets  # Pass the new parameter
    )
    
    if results:
        # Display trading rules
        print("\nTRADING RULES:")
        print("=============")
        for rule in results['trading_rules']:
            print(rule)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()