"""
Major Planetary Aspects Analysis
--------------------------------
Analyzes market behavior around major aspects between outer planets
(Venus, Mars, Jupiter, Saturn, Uranus, Pluto).

This script:
1. Finds all occurrences of selected aspects between specified planets
2. Analyzes market behavior around these aspect dates
3. Visualizes the results and identifies patterns
4. Exports data for further analysis

Built on previous financial astrology scripts
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
import argparse
# Add this near the top of the script, right after imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def initialize_ephemeris():
    """Initialize the Swiss Ephemeris with proper path."""
    # Let Swiss Ephemeris download files automatically:
    swe.set_ephe_path(None)

def find_aspects_between_planets(start_date, end_date, planet_pairs, aspects_to_find, orb=1.0, is_heliocentric=False):
    """
    Find dates when specified planet pairs form specific aspects within a date range,
    including multiple aspects due to retrograde motion.
    """
    # Initialize ephemeris
    initialize_ephemeris()
    
    # Convert dates to Julian days
    jd_start = swe.julday(start_date.year, start_date.month, start_date.day, 0)
    jd_end = swe.julday(end_date.year, end_date.month, end_date.day, 0)
    
    # Step size for scanning (1 day)
    step = 1
    
    # Store found aspects
    found_aspects = []
    
    # Set up flags for calculating speed (to detect retrograde)
    flag = swe.FLG_SWIEPH | swe.FLG_SPEED
    if is_heliocentric:
        flag |= swe.FLG_HELCTR
    
    # For each planet pair, scan for aspects
    for planet1_id, planet2_id in planet_pairs:
        # Track when we enter and exit orb ranges to find exact aspect dates
        active_aspects = {}
        
        # Track aspect groups (for retrograde motion)
        aspect_groups = {}
        
        # Scan through the date range
        current_jd = jd_start
        while current_jd <= jd_end:
            # Get planet positions and speeds
            pos1, status1 = swe.calc_ut(current_jd, planet1_id, flag)
            pos2, status2 = swe.calc_ut(current_jd, planet2_id, flag)
            
            # Extract longitudes
            lon1 = pos1[0]
            lon2 = pos2[0]
            
            # Extract speeds (for retrograde detection)
            speed1 = pos1[3]  # Daily speed in longitude
            speed2 = pos2[3]
            
            # Determine if planets are retrograde
            is_retrograde1 = speed1 < 0
            is_retrograde2 = speed2 < 0
            
            # Create an identifier for the planet motions
            motion_key = f"{'R' if is_retrograde1 else 'D'}-{'R' if is_retrograde2 else 'D'}"
            
            # Calculate angular difference, ensuring it's in the range 0-180
            diff = abs(lon1 - lon2) % 360
            if diff > 180:
                diff = 360 - diff
            
            # Check each aspect we're looking for
            for aspect in aspects_to_find:
                # Combined aspect and motion key
                aspect_motion_key = f"{aspect}-{motion_key}"
                
                # Check if we're within orb of the aspect
                if abs(diff - aspect) <= orb:
                    # If we're not tracking this aspect-motion yet, add it
                    if aspect_motion_key not in active_aspects:
                        active_aspects[aspect_motion_key] = {
                            'start_jd': current_jd,
                            'min_diff': abs(diff - aspect),
                            'min_diff_jd': current_jd,
                            'motion_key': motion_key
                        }
                    # Otherwise update the minimum difference if needed
                    elif abs(diff - aspect) < active_aspects[aspect_motion_key]['min_diff']:
                        active_aspects[aspect_motion_key]['min_diff'] = abs(diff - aspect)
                        active_aspects[aspect_motion_key]['min_diff_jd'] = current_jd
                
                # If we're tracking this aspect but now out of orb, record it
                elif aspect_motion_key in active_aspects:
                    # Get the date when the aspect was most exact
                    exact_jd = active_aspects[aspect_motion_key]['min_diff_jd']
                    
                    # Convert to calendar date
                    y, m, d, h = swe.revjul(exact_jd, swe.GREG_CAL)
                    exact_date = datetime.date(y, m, d)
                    
                    # Create group key for tracking related aspects (same aspect angle)
                    group_key = f"{planet1_id}-{planet2_id}-{aspect}"
                    
                    # Initialize group if needed
                    if group_key not in aspect_groups:
                        aspect_groups[group_key] = []
                    
                    # Add to aspect group
                    aspect_groups[group_key].append({
                        'planet1': planet1_id,
                        'planet2': planet2_id,
                        'aspect': aspect,
                        'date': exact_date,
                        'is_heliocentric': is_heliocentric,
                        'orb': active_aspects[aspect_motion_key]['min_diff'],
                        'motion_key': motion_key,
                        'is_retrograde1': is_retrograde1,
                        'is_retrograde2': is_retrograde2
                    })
                    
                    # Remove from active tracking
                    del active_aspects[aspect_motion_key]
            
            # Move to next day
            current_jd += step
        
        # Check if any aspects are still active at the end of the range
        for aspect_motion_key, data in active_aspects.items():
            aspect_info = aspect_motion_key.split('-')
            aspect = float(aspect_info[0])
            motion_key = data['motion_key']
            
            # Parse motion key to get retrograde status
            planet_motions = motion_key.split('-')
            is_retrograde1 = planet_motions[0] == 'R'
            is_retrograde2 = planet_motions[1] == 'R'
            
            exact_jd = data['min_diff_jd']
            
            # Convert to calendar date
            y, m, d, h = swe.revjul(exact_jd, swe.GREG_CAL)
            exact_date = datetime.date(y, m, d)
            
            # Create group key for tracking related aspects
            group_key = f"{planet1_id}-{planet2_id}-{aspect}"
            
            # Initialize group if needed
            if group_key not in aspect_groups:
                aspect_groups[group_key] = []
            
            # Add to aspect group
            aspect_groups[group_key].append({
                'planet1': planet1_id,
                'planet2': planet2_id,
                'aspect': aspect,
                'date': exact_date,
                'is_heliocentric': is_heliocentric,
                'orb': data['min_diff'],
                'motion_key': motion_key,
                'is_retrograde1': is_retrograde1,
                'is_retrograde2': is_retrograde2
            })
        
        # Add all aspect groups to found_aspects
        for group_key, aspects in aspect_groups.items():
            # Sort aspects in the group by date
            aspects.sort(key=lambda x: x['date'])
            
            # If there are multiple aspects in a group, number them
            if len(aspects) > 1:
                for i, aspect in enumerate(aspects):
                    aspect['sequence_number'] = i + 1
                    aspect['total_in_sequence'] = len(aspects)
                    found_aspects.append(aspect)
            else:
                # Just add the single aspect
                aspects[0]['sequence_number'] = 1
                aspects[0]['total_in_sequence'] = 1
                found_aspects.append(aspects[0])
    
    # Sort all aspects by date
    found_aspects.sort(key=lambda x: x['date'])
    
    return found_aspects

def get_market_data(symbol, aspect_dates, window_days=30):
    """
    Get market data around aspect dates.
    
    Args:
        symbol: Market symbol to analyze
        aspect_dates: List of dates when aspects occur
        window_days: Number of days to analyze before/after each aspect
        
    Returns:
        Dictionary mapping aspect dates to market data
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
    elif symbol.upper() in ['WHEAT', 'WEAT']:
        yf_symbol = 'WEAT'  # Wheat Futures
    else:
        yf_symbol = symbol
    
    # Find the earliest and latest dates we need data for
    earliest_date = min(aspect_dates) - datetime.timedelta(days=window_days)
    latest_date = max(aspect_dates) + datetime.timedelta(days=window_days)
    
    # Get all market data for the full range
    try:
        all_data = yf.download(yf_symbol, start=earliest_date, end=latest_date, progress=False)
        
        if all_data.empty:
            print(f"No data available for {symbol} in the specified date range.")
            return {}
        
        # Prepare result dictionary
        result = {}
        
        # For each aspect date, extract the relevant window of data
        for aspect_date in aspect_dates:
            start_date = aspect_date - datetime.timedelta(days=window_days)
            end_date = aspect_date + datetime.timedelta(days=window_days)
            
            # Get data within this window
            mask = (all_data.index >= pd.Timestamp(start_date)) & (all_data.index <= pd.Timestamp(end_date))
            window_data = all_data.loc[mask].copy()
            
            # Calculate the daily percent change
            window_data['Pct_Change'] = window_data['Close'].pct_change() * 100
            
            # Calculate days relative to aspect date
            window_data['Days_From_Aspect'] = [(d.date() - aspect_date).days for d in window_data.index]
            
            # Store in result
            result[aspect_date] = window_data
        
        return result
        
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return {}

def analyze_market_behavior(market_data, aspect_date):
    """
    Analyze market behavior around an aspect date.
    
    Args:
        market_data: Market data around aspect date
        aspect_date: Date of the aspect
        
    Returns:
        Dictionary with analysis results
    """
    if aspect_date not in market_data or market_data[aspect_date].empty:
        return {
            'has_data': False,
            'price_at_aspect': None,
            'is_turning_point': False,
            'pattern': None,
            'before_change': None,
            'after_change': None
        }
    
    data = market_data[aspect_date]
    
    # Find the closest trading day to the aspect date
    closest_idx = data['Days_From_Aspect'].abs().idxmin()
    aspect_day_offset = data.loc[closest_idx, 'Days_From_Aspect']
    
    # Get price at aspect date
    price_at_aspect = float(data.loc[closest_idx, 'Close'])
    
    # Define window sizes for turning point analysis
    pre_window = 10  # Days before
    post_window = 10  # Days after
    
    # Extract pre and post windows using numeric indices
    pre_indices = data.index[data.index < closest_idx][-pre_window:] if len(data.index[data.index < closest_idx]) >= pre_window else data.index[data.index < closest_idx]
    post_indices = data.index[data.index > closest_idx][:post_window] if len(data.index[data.index > closest_idx]) >= post_window else data.index[data.index > closest_idx]
    
    pre_data = data.loc[pre_indices] if len(pre_indices) > 0 else pd.DataFrame()
    post_data = data.loc[post_indices] if len(post_indices) > 0 else pd.DataFrame()
    
    # Check if we have enough data for analysis
    if len(pre_data) < 5 or len(post_data) < 5:
        return {
            'has_data': True,
            'price_at_aspect': price_at_aspect,
            'is_turning_point': False,
            'pattern': 'Insufficient data',
            'before_change': None,
            'after_change': None
        }
    
    # Check if it's a local maximum (high)
    pre_max = float(pre_data['Close'].max())
    post_max = float(post_data['Close'].max())
    
    # Check if it's a local minimum (low)
    pre_min = float(pre_data['Close'].min())
    post_min = float(post_data['Close'].min())
    
    # Calculate percentage changes before and after
    if len(pre_data) > 0:
        pre_first = float(pre_data['Close'].iloc[0])
        pre_last = float(pre_data['Close'].iloc[-1])
        before_change = ((pre_last - pre_first) / pre_first) * 100
    else:
        before_change = None
    
    if len(post_data) > 0:
        post_first = float(post_data['Close'].iloc[0])
        post_last = float(post_data['Close'].iloc[-1])
        after_change = ((post_last - post_first) / post_first) * 100
    else:
        after_change = None
    
    # Determine if it's a turning point
    is_high = price_at_aspect >= pre_max and price_at_aspect >= post_max
    is_low = price_at_aspect <= pre_min and price_at_aspect <= post_min
    
    # Determine pattern
    if is_high:
        pattern = 'High'
    elif is_low:
        pattern = 'Low'
    else:
        # Calculate trend before and after
        pre_trend = (pre_last - pre_first) / pre_first if len(pre_data) > 0 else 0
        post_trend = (post_last - post_first) / post_first if len(post_data) > 0 else 0
        
        # Check for trend changes
        if pre_trend > 0 and post_trend < 0:
            pattern = 'Uptrend to Downtrend'
        elif pre_trend < 0 and post_trend > 0:
            pattern = 'Downtrend to Uptrend'
        elif pre_trend > 0 and post_trend > 0:
            pattern = 'Continued Uptrend'
        else:
            pattern = 'Continued Downtrend'
    
    return {
        'has_data': True,
        'price_at_aspect': price_at_aspect,
        'is_turning_point': is_high or is_low,
        'pattern': pattern,
        'before_change': before_change,
        'after_change': after_change
    }

def get_aspect_symbol(aspect):
    """Get the symbol for an aspect."""
    symbols = {
        0: "☌",   # Conjunction
        60: "⚹",  # Sextile
        90: "□",  # Square
        120: "△", # Trine
        180: "☍"  # Opposition
    }
    return symbols.get(aspect, str(aspect) + "°")

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
    return names.get(aspect, str(aspect) + "°")

def visualize_aspect_analysis(aspect_details, market_data, market_symbol, start_date, end_date, selected_planets, selected_aspects):
    """
    Create visualizations of market behavior around aspect dates.
    
    Args:
        aspect_details: List of dictionaries with aspect details
        market_data: Dictionary mapping aspect dates to market data
        market_symbol: Market symbol for title/labeling
        start_date: Start date of analysis period
        end_date: End date of analysis period
        selected_planets: List of selected planet names
        selected_aspects: List of selected aspect angles
    """
    # Group aspects by type for better visualization
    aspect_types = {}
    for aspect in aspect_details:
        aspect_date = aspect['date']
        
        if aspect_date not in market_data or market_data[aspect_date].empty:
            continue
        
        aspect_key = (aspect['aspect'], aspect['is_heliocentric'])
        if aspect_key not in aspect_types:
            aspect_types[aspect_key] = []
        
        aspect_types[aspect_key].append(aspect)
    
    # Create a directory name based on parameters
    date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    planets_str = "_".join(sorted(selected_planets))
    aspects_str = "_".join([str(int(a)) for a in sorted(selected_aspects)])
    
    folder_name = f"aspects_{market_symbol}_{date_range}_{planets_str}_{aspects_str}"
    # Ensure folder name is valid
    folder_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in folder_name)
    
    # Create the directory
    os.makedirs(folder_name, exist_ok=True)
    
    # Plot each aspect
    for aspect in aspect_details:
        aspect_date = aspect['date']
        
        if aspect_date not in market_data or market_data[aspect_date].empty:
            print(f"No market data available for {get_aspect_name(aspect['aspect'])} on {aspect_date}")
            continue
        
        data = market_data[aspect_date]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot price data
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'])
        
        # Mark aspect date
        closest_idx = data['Days_From_Aspect'].abs().idxmin()
        aspect_price = data.loc[closest_idx, 'Close']
        
        plt.axvline(x=closest_idx, color='r', linestyle='--')
        plt.plot(closest_idx, aspect_price, 'ro', markersize=8)
        
        # Add annotation
        p1_name = get_planet_name(aspect['planet1'])
        p2_name = get_planet_name(aspect['planet2'])
        aspect_type = get_aspect_name(aspect['aspect'])
        aspect_symbol = get_aspect_symbol(aspect['aspect'])
        system = "Heliocentric" if aspect['is_heliocentric'] else "Geocentric"

        # Add retrograde info to title
        motion_key = aspect.get('motion_key', 'D-D')
        retrograde_info = ""
        if 'R' in motion_key:
            retrograde_info = " ⟲"  # Add retrograde symbol
        
        # Add sequence info if part of a series
        sequence_info = ""
        if aspect.get('total_in_sequence', 1) > 1:
            sequence_info = f" ({aspect.get('sequence_number', 1)}/{aspect.get('total_in_sequence', 1)})"

        
        title = f"{p1_name} {aspect_symbol} {p2_name}{retrograde_info}{sequence_info} ({aspect_type}, {system})\n{aspect_date.strftime('%Y-%m-%d')}"
        plt.title(title)
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # Plot percent change
        plt.subplot(2, 1, 2)
        plt.bar(data.index, data['Pct_Change'])
        plt.axvline(x=closest_idx, color='r', linestyle='--')
        plt.ylabel('Daily % Change')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart
        chart_filename = f"{folder_name}/{p1_name}_{p2_name}_{aspect_type}_{aspect_date.strftime('%Y-%m-%d')}.png"
        chart_filename = chart_filename.replace(" ", "_").lower()
        plt.savefig(chart_filename)
        print(f"Chart saved as {chart_filename}")
        
        plt.close()
    
    # Create a summary chart by aspect type
    for (aspect_angle, is_helio), aspects in aspect_types.items():
        if not aspects:
            continue
            
        aspect_name = get_aspect_name(aspect_angle)
        system = "Heliocentric" if is_helio else "Geocentric"
        
        plt.figure(figsize=(14, 10))
        
        # Get all dates for this aspect type
        all_dates = []
        all_prices = []
        all_labels = []
        
        for aspect in aspects:
            aspect_date = aspect['date']
            if aspect_date in market_data and not market_data[aspect_date].empty:
                data = market_data[aspect_date]
                closest_idx = data['Days_From_Aspect'].abs().idxmin()
                
                if closest_idx in data.index:
                    all_dates.append(closest_idx)
                    all_prices.append(data.loc[closest_idx, 'Close'])
                    
                    p1_name = get_planet_name(aspect['planet1'])
                    p2_name = get_planet_name(aspect['planet2'])
                    all_labels.append(f"{p1_name}-{p2_name}")
        
        if all_dates:
            # Get full date range from the first to the last aspect
            earliest_date = min(all_dates) - pd.Timedelta(days=365)
            latest_date = max(all_dates) + pd.Timedelta(days=365)
            
            # Get market data for this entire period
            try:
                if market_symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
                    yf_symbol = '^GSPC'
                elif market_symbol.upper() in ['DJI', 'DJIA', 'DOW']:
                    yf_symbol = '^DJI'
                elif market_symbol.upper() in ['IXIC', 'NASDAQ', 'NDX']:
                    yf_symbol = '^IXIC'
                elif market_symbol.upper() in ['GOLD', 'XAU']:
                    yf_symbol = 'GC=F'  # Gold Futures
                elif market_symbol.upper() in ['SILVER', 'XAG']:
                    yf_symbol = 'SI=F'  # Silver Futures
                elif market_symbol.upper() in ['WHEAT', 'WEAT']:
                    yf_symbol = 'WEAT'  # Wheat Futures
                else:
                    yf_symbol = market_symbol
                    
                full_data = yf.download(yf_symbol, start=earliest_date, end=latest_date, progress=False)
                
                if not full_data.empty:
                    # Plot the market for the entire period
                    plt.plot(full_data.index, full_data['Close'], 'b-', alpha=0.7)
                    
                    # Mark each aspect date
                    for date, price, label in zip(all_dates, all_prices, all_labels):
                        plt.axvline(x=date, color='r', linestyle='--', alpha=0.5)
                        plt.plot(date, price, 'ro', markersize=8)
                        
                        # Add annotation
                        plt.annotate(label, 
                            xy=(closest_date, price), 
                            xytext=(0, 10),
                            textcoords="offset points",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))  
                          
                    plt.title(f"All {aspect_name} Aspects ({system}) - {market_symbol}")
                    plt.ylabel('Price')
                    plt.grid(True, alpha=0.3)
                    
                    # Format x-axis to show dates nicely
                    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                    plt.gcf().autofmt_xdate()
                    
                    # Save the summary chart
                    chart_filename = f"{folder_name}/Summary_{aspect_name}_{system}_{market_symbol}.png"
                    chart_filename = chart_filename.replace(" ", "_").lower()
                    plt.savefig(chart_filename)
                    print(f"Summary chart saved as {chart_filename}")
            except Exception as e:
                print(f"Error creating summary chart: {e}")
            
            plt.close()
    
    # Create an overall summary chart
    plt.figure(figsize=(16, 10))
    
    # Get unique aspect types for color coding
    unique_aspects = sorted(set(a['aspect'] for a in aspect_details))
    colors = plt.cm.tab10.colors  # Get a color cycle
    
    # Create a mapping from aspect type to color
    aspect_colors = {aspect: colors[i % len(colors)] for i, aspect in enumerate(unique_aspects)}
    
    # Get earliest and latest dates across all aspects
    all_dates = [a['date'] for a in aspect_details]
    if all_dates:
        earliest_date = min(all_dates) - datetime.timedelta(days=365)
        latest_date = max(all_dates) + datetime.timedelta(days=365)
        
        # Get market data for this entire period
        try:
            if market_symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
                yf_symbol = '^GSPC'
            elif market_symbol.upper() in ['DJI', 'DJIA', 'DOW']:
                yf_symbol = '^DJI'
            elif market_symbol.upper() in ['IXIC', 'NASDAQ', 'NDX']:
                yf_symbol = '^IXIC'
            else:
                yf_symbol = market_symbol
                
            full_data = yf.download(yf_symbol, start=earliest_date, end=latest_date, progress=False)
            
            if not full_data.empty:
                # Plot the market for the entire period
                plt.plot(full_data.index, full_data['Close'], 'k-', alpha=0.7)
                
                # Create legend patches
                legend_patches = []
                
                # Mark each aspect date
                for aspect in aspect_details:
                    aspect_date = aspect['date']
                    aspect_angle = aspect['aspect']
                    
                    # Get the pandas timestamp closest to this date
                    closest_date = None
                    min_diff = float('inf')
                    
                    for date_idx in full_data.index:
                        date_diff = abs((date_idx.date() - aspect_date).days)
                        if date_diff < min_diff:
                            min_diff = date_diff
                            closest_date = date_idx
                    
                    if closest_date is not None:
                        price = full_data.loc[closest_date, 'Close']
                        
                        # Get aspect info
                        p1_name = get_planet_name(aspect['planet1'])
                        p2_name = get_planet_name(aspect['planet2'])
                        aspect_type = get_aspect_name(aspect_angle)
                        aspect_symbol = get_aspect_symbol(aspect_angle)
                        
                        # Use color based on aspect type
                        color = aspect_colors[aspect_angle]
                        
                        # Add to legend if this aspect type hasn't been added yet
                        if aspect_type not in [p.get_label() for p in legend_patches]:
                            legend_patches.append(mpatches.Patch(color=color, label=aspect_type))
                        
                        # Mark on chart
                        plt.axvline(x=closest_date, color=color, linestyle='--', alpha=0.5)
                        plt.plot(closest_date, price, 'o', color=color, markersize=6)
                        
                        # Add annotation
                        # When marking aspect points on charts:
                        label = f"{p1_name}-{p2_name}"
                        motion_key = aspect.get('motion_key', 'D-D')
                        if 'R' in motion_key:
                            label += " ⟲"  # Add retrograde symbol
                        if aspect.get('total_in_sequence', 1) > 1:
                            label += f" ({aspect.get('sequence_number', 1)}/{aspect.get('total_in_sequence', 1)})"

                        plt.annotate(label, 
                                xy=(closest_date, price), 
                                xytext=(0, 10),
                                textcoords="offset points",
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                
                plt.title(f"All Major Aspects - {market_symbol}")
                plt.ylabel('Price')
                plt.grid(True, alpha=0.3)
                
                # Add legend
                plt.legend(handles=legend_patches, loc='best')
                
                # Format x-axis to show dates nicely
                plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.gcf().autofmt_xdate()
                
                # Save the overall summary chart
                chart_filename = f"{folder_name}/Overall_Summary.png"
                plt.savefig(chart_filename)
                print(f"Overall summary chart saved as {chart_filename}")
        except Exception as e:
            print(f"Error creating overall summary chart: {e}")
        
        plt.close('all')  # Explicitly close all figures

def display_results(aspect_details, market_analyses, market_symbol):
    """Display the analysis results in a formatted table."""
    print("\n" + "="*150)
    print(f"MAJOR PLANETARY ASPECTS ANALYSIS FOR {market_symbol}")
    print("="*150)
    
    if not aspect_details:
        print("No aspects found in the specified date range.")
        return
    
    # Print header
    print(f"{'Planets':20} | {'Aspect':12} | {'System':12} | {'Date':12} | {'Motion':10} | {'Seq':5} | {'Price':10} | {'Pattern':20} | {'% Before':10} | {'% After':10} | {'Turning Point':12}")
    print("-"*150)
    
    # Print each aspect
    for aspect in aspect_details:
        aspect_date = aspect['date']
        
        # Get analysis if available
        if aspect_date in market_analyses:
            analysis = market_analyses[aspect_date]
        else:
            analysis = {'has_data': False, 'price_at_aspect': None, 'is_turning_point': False, 'pattern': None, 'before_change': None, 'after_change': None}
        
        # Format fields
        p1_name = get_planet_name(aspect['planet1'])
        p2_name = get_planet_name(aspect['planet2'])
        planets = f"{p1_name}-{p2_name}"
        
        aspect_type = get_aspect_name(aspect['aspect'])
        date_str = aspect_date.strftime('%Y-%m-%d')
        system = "Heliocentric" if aspect['is_heliocentric'] else "Geocentric"
        
        # Format motion status
        motion = aspect.get('motion_key', 'D-D')
        
        # Format sequence information
        seq = f"{aspect.get('sequence_number', 1)}/{aspect.get('total_in_sequence', 1)}"
        
        if analysis['has_data']:
            price = f"${analysis['price_at_aspect']:.2f}" if analysis['price_at_aspect'] is not None else "N/A"
            pattern = analysis['pattern'] or "N/A"
            
            before_change = f"{analysis['before_change']:.2f}%" if analysis['before_change'] is not None else "N/A"
            after_change = f"{analysis['after_change']:.2f}%" if analysis['after_change'] is not None else "N/A"
            
            turning_point = "Yes" if analysis['is_turning_point'] else "No"
        else:
            price = "N/A"
            pattern = "No data"
            before_change = "N/A"
            after_change = "N/A"
            turning_point = "N/A"
        
        # Print row
        print(f"{planets:20} | {aspect_type:12} | {system:12} | {date_str:12} | {motion:10} | {seq:5} | {price:10} | {pattern:20} | {before_change:10} | {after_change:10} | {turning_point:12}")
    
    print("="*150)

def calculate_statistics(aspect_details, market_analyses):
    """
    Calculate statistics for different aspect types.
    
    Args:
        aspect_details: List of dictionaries with aspect details
        market_analyses: Dictionary mapping aspect dates to market analyses
        
    Returns:
        Dictionary with statistics by aspect type
    """
    # Initialize statistics
    stats = {}
    
    # Group by aspect type
    for aspect in aspect_details:
        aspect_date = aspect['date']
        aspect_type = aspect['aspect']
        
        # Create key if it doesn't exist
        if aspect_type not in stats:
            stats[aspect_type] = {
                'count': 0,
                'has_data_count': 0,
                'turning_points': 0,
                'patterns': {
                    'High': 0,
                    'Low': 0,
                    'Uptrend to Downtrend': 0,
                    'Downtrend to Uptrend': 0,
                    'Continued Uptrend': 0,
                    'Continued Downtrend': 0,
                    'Insufficient data': 0
                },
                'before_changes': [],
                'after_changes': []
            }
        
        # Get analysis if available
        if aspect_date in market_analyses:
            analysis = market_analyses[aspect_date]
            
            # Update statistics
            stats[aspect_type]['count'] += 1
            
            if analysis['has_data']:
                stats[aspect_type]['has_data_count'] += 1
                
                if analysis['is_turning_point']:
                    stats[aspect_type]['turning_points'] += 1
                
                pattern = analysis['pattern']
                if pattern in stats[aspect_type]['patterns']:
                    stats[aspect_type]['patterns'][pattern] += 1
                
                if analysis['before_change'] is not None:
                    stats[aspect_type]['before_changes'].append(analysis['before_change'])
                
                if analysis['after_change'] is not None:
                    stats[aspect_type]['after_changes'].append(analysis['after_change'])
    
    # Calculate percentages and averages
    for aspect_type, data in stats.items():
        if data['has_data_count'] > 0:
            data['turning_point_pct'] = (data['turning_points'] / data['has_data_count']) * 100
            
            data['avg_before_change'] = sum(data['before_changes']) / len(data['before_changes']) if data['before_changes'] else None
            data['avg_after_change'] = sum(data['after_changes']) / len(data['after_changes']) if data['after_changes'] else None
            
            # Calculate dominant pattern
            max_pattern = max(data['patterns'].items(), key=lambda x: x[1]) if data['patterns'] else ('None', 0)
            data['dominant_pattern'] = max_pattern[0] if max_pattern[1] > 0 else 'None'
    
    return stats

def display_statistics(stats, market_symbol):
    """Display statistical summary by aspect type."""
    print("\n" + "="*100)
    print(f"STATISTICAL SUMMARY FOR {market_symbol}")
    print("="*100)
    
    if not stats:
        print("No statistical data available.")
        return
    
    # Print header
    print(f"{'Aspect Type':12} | {'Count':5} | {'Turning Points %':15} | {'Dominant Pattern':20} | {'Avg % Before':12} | {'Avg % After':12}")
    print("-"*100)
    
    # Sort by aspect type
    for aspect_type in sorted(stats.keys()):
        data = stats[aspect_type]
        
        # Format fields
        aspect_name = get_aspect_name(aspect_type)
        count = str(data['count'])
        
        if data['has_data_count'] > 0:
            turning_point_pct = f"{data['turning_point_pct']:.1f}%" if 'turning_point_pct' in data else "N/A"
            dominant_pattern = data.get('dominant_pattern', 'None')
            
            avg_before = f"{data['avg_before_change']:.2f}%" if data.get('avg_before_change') is not None else "N/A"
            avg_after = f"{data['avg_after_change']:.2f}%" if data.get('avg_after_change') is not None else "N/A"
        else:
            turning_point_pct = "N/A"
            dominant_pattern = "N/A"
            avg_before = "N/A"
            avg_after = "N/A"
        
        # Print row
        print(f"{aspect_name:12} | {count:5} | {turning_point_pct:15} | {dominant_pattern:20} | {avg_before:12} | {avg_after:12}")
    
    print("="*100)

def save_to_csv(aspect_details, market_analyses, market_symbol, folder_name):
    """Save analysis results to CSV files."""
    # Prepare data for aspects CSV
    aspects_data = []
    
    for aspect in aspect_details:
        aspect_date = aspect['date']
        
        # Get analysis if available
        if aspect_date in market_analyses:
            analysis = market_analyses[aspect_date]
        else:
            analysis = {'has_data': False, 'price_at_aspect': None, 'is_turning_point': False, 'pattern': None, 'before_change': None, 'after_change': None}
        
        # Create row
        row = {
            'Planet1': get_planet_name(aspect['planet1']),
            'Planet2': get_planet_name(aspect['planet2']),
            'Aspect': get_aspect_name(aspect['aspect']),
            'Aspect_Angle': aspect['aspect'],
            'Date': aspect_date,
            'System': "Heliocentric" if aspect['is_heliocentric'] else "Geocentric",
            'Orb': aspect['orb'],
            'Motion': aspect.get('motion_key', 'D-D'),
            'Sequence_Number': aspect.get('sequence_number', 1),
            'Total_In_Sequence': aspect.get('total_in_sequence', 1),
            'Is_Retrograde1': aspect.get('is_retrograde1', False),
            'Is_Retrograde2': aspect.get('is_retrograde2', False),
            'Has_Price_Data': analysis['has_data'],
            'Price': analysis['price_at_aspect'],
            'Is_Turning_Point': analysis['is_turning_point'],
            'Pattern': analysis['pattern'],
            'Before_Change': analysis['before_change'],
            'After_Change': analysis['after_change']
        }
        
        aspects_data.append(row)
    
    # Create DataFrame and save
    if aspects_data:
        df = pd.DataFrame(aspects_data)
        
        # Format the market symbol for filename
        safe_symbol = "".join(c if c.isalnum() else "_" for c in market_symbol)
        filename = f"{folder_name}/{safe_symbol}_major_aspects_analysis.csv"
        
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    else:
        print("No data to save.")

def main():
    """Main function to run the analysis."""
    print("Major Planetary Aspects Analysis Tool")
    print("------------------------------------")
    
    # Get date range
    print("Enter start date (YYYY-MM-DD):")
    start_date_str = input()
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    
    print("Enter end date (YYYY-MM-DD):")
    end_date_str = input()
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    # Get market symbol
    print("Enter stock/index symbol (e.g., SPX, NASDAQ, DJI, GOLD, SILVER, WEAT):")
    market_symbol = input()
    
    # Get planets to analyze
    print("Select planets to analyze (comma-separated, e.g., Venus,Mars,Jupiter,Saturn,Uranus,Pluto):")
    print("Available: Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto")
    planet_input = input()
    planet_names = [p.strip() for p in planet_input.split(',')]
    
    # Map planet names to Swiss Ephemeris IDs
    planet_map = {
        'Mercury' : swe.MERCURY,
        'Venus' : swe.VENUS,
        'Mars': swe.MARS,
        'Jupiter': swe.JUPITER,
        'Saturn': swe.SATURN,
        'Uranus': swe.URANUS,
        'Neptune': swe.NEPTUNE,
        'Pluto': swe.PLUTO
    }
    
    # Convert selected planet names to IDs
    planet_ids = []
    for name in planet_names:
        if name in planet_map:
            planet_ids.append(planet_map[name])
        else:
            print(f"Warning: Planet {name} not recognized. Skipping.")
    
    if not planet_ids:
        print("No valid planets selected. Exiting.")
        return
    
    # Get aspects to analyze
    print("Enter aspects to analyze (comma-separated, e.g., 0,60,90,120,180):")
    print("Available: 0 (Conjunction), 60 (Sextile), 90 (Square), 120 (Trine), 180 (Opposition)")
    aspect_input = input()
    aspects_to_find = [float(a.strip()) for a in aspect_input.split(',')]
    
    # Get coordinate system preference
    print("Use heliocentric coordinates? (y/n):")
    is_heliocentric = input().lower() == 'y'
    
    # Create all possible planet pairs
    planet_pairs = []
    for i in range(len(planet_ids)):
        for j in range(i+1, len(planet_ids)):
            planet_pairs.append((planet_ids[i], planet_ids[j]))
    
    # Find aspects
    print("\nFinding aspects between selected planets...")
    aspects = find_aspects_between_planets(
        start_date, 
        end_date, 
        planet_pairs, 
        aspects_to_find, 
        orb=1.0, 
        is_heliocentric=is_heliocentric
    )
    
    if not aspects:
        print("No aspects found in the specified date range.")
        return
    
    print(f"Found {len(aspects)} aspects.")
    
    # Get market data around aspects
    print("\nRetrieving market data...")
    aspect_dates = [a['date'] for a in aspects]
    market_data = get_market_data(market_symbol, aspect_dates)
    
    # Analyze market behavior
    print("\nAnalyzing market behavior...")
    market_analyses = {}
    for aspect_date in aspect_dates:
        market_analyses[aspect_date] = analyze_market_behavior(market_data, aspect_date)
    
    # Calculate statistics
    stats = calculate_statistics(aspects, market_analyses)
    
    # Display results
    display_results(aspects, market_analyses, market_symbol)
    
    # Display statistics
    display_statistics(stats, market_symbol)

    # Create folder name for saving results
    date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    planets_str = "_".join(sorted(planet_names))
    aspects_str = "_".join([str(int(a)) for a in sorted(aspects_to_find)])
    folder_name = f"aspects_{market_symbol}_{date_range}_{planets_str}_{aspects_str}"
    folder_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in folder_name)
    os.makedirs(folder_name, exist_ok=True)
    
    # Option to save results
    print("\nWould you like to save the results to a CSV file? (y/n)")
    if input().lower() == 'y':
        save_to_csv(aspects, market_analyses, market_symbol, folder_name)
    
    # Option to visualize results
    print("\nWould you like to visualize the results? (y/n)")
    if input().lower() == 'y':
        print("\nCreating visualizations...")
        visualize_aspect_analysis(aspects, market_data, market_symbol, start_date, end_date, planet_names, aspects_to_find)

if __name__ == "__main__":
    main()