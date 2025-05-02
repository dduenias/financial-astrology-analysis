"""
Planetary Sign Ingress Analysis
-------------------------------
Analyzes market behavior when planets enter new zodiac signs.

This script:
1. Finds all occurrences when a selected planet enters a new zodiac sign within a date range
2. Analyzes market behavior around these ingress dates
3. Visualizes the results and identifies patterns
4. Exports data for further analysis

Uses Swiss Ephemeris for accurate planetary calculations
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

def initialize_ephemeris():
    """Initialize the Swiss Ephemeris with proper path."""
    # Let Swiss Ephemeris download files automatically:
    swe.set_ephe_path(None)

def get_zodiac_sign(longitude):
    """Convert longitude to zodiac sign."""
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer",
        "Leo", "Virgo", "Libra", "Scorpio",
        "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    
    # Normalize longitude to 0-360 range
    norm_longitude = fmod(longitude, 360.0)
    if norm_longitude < 0:
        norm_longitude += 360.0
    
    # Determine sign index (each sign is 30 degrees)
    sign_index = int(norm_longitude / 30)
    
    return signs[sign_index]

def is_retrograde(jd, planet_id):
    """
    Determine if a planet is retrograde at a given Julian day.
    
    Args:
        jd: Julian day number
        planet_id: Swiss Ephemeris ID for the planet
        
    Returns:
        Boolean: True if retrograde, False if direct
    """
    # Set up flag for calculation
    flag = swe.FLG_SWIEPH | swe.FLG_SPEED
    
    # Get position and speed
    pos, status = swe.calc_ut(jd, planet_id, flag)
    
    # The second element of the pos array contains the longitude speed
    # If it's negative, the planet is retrograde
    longitude_speed = pos[3]
    
    return longitude_speed < 0

def get_sign_degrees(longitude):
    """Get the degrees within the current sign (0-29.999...)."""
    # Normalize longitude to 0-360 range
    norm_longitude = fmod(longitude, 360.0)
    if norm_longitude < 0:
        norm_longitude += 360.0
    
    # Calculate degrees within sign
    return fmod(norm_longitude, 30)

def find_sign_ingresses(start_date, end_date, planet_id):
    """
    Find dates when a planet enters a new zodiac sign within a date range.
    
    Args:
        start_date: Beginning of search range (datetime.date)
        end_date: End of search range (datetime.date)
        planet_id: Swiss Ephemeris ID for the planet to track
        
    Returns:
        List of dictionaries with ingress dates and details
    """
    # Initialize ephemeris
    initialize_ephemeris()
    
    # Convert dates to Julian days
    jd_start = swe.julday(start_date.year, start_date.month, start_date.day, 0)
    jd_end = swe.julday(end_date.year, end_date.month, end_date.day, 0)
    
    # Step size for scanning (1 day for fast-moving planets, more for slow ones)
    if planet_id in [swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS]:
        step = 1  # 1 day for faster planets
    else:
        step = 7  # 7 days for slower planets
    
    # Store found ingresses
    ingresses = []
    
    # Set up flag for calculation
    flag = swe.FLG_SWIEPH
    
    # Get initial position
    pos_init, status = swe.calc_ut(jd_start, planet_id, flag)
    current_sign = get_zodiac_sign(pos_init[0])
    
    # Scan through the date range
    current_jd = jd_start + step
    while current_jd <= jd_end:
        # Get current position
        pos, status = swe.calc_ut(current_jd, planet_id, flag)
        longitude = pos[0]
        new_sign = get_zodiac_sign(longitude)
        
        # Check if sign has changed
        if new_sign != current_sign:
            # We've found a sign change, now refine to get the exact date
            # Do a binary search between current_jd - step and current_jd
            
            lower_jd = current_jd - step
            upper_jd = current_jd
            mid_jd = (lower_jd + upper_jd) / 2
            
            # Binary search to narrow down the exact moment of ingress
            for _ in range(10):  # Usually 10 iterations gives good precision
                pos_mid, _ = swe.calc_ut(mid_jd, planet_id, flag)
                mid_sign = get_zodiac_sign(pos_mid[0])
                
                if mid_sign == current_sign:
                    lower_jd = mid_jd
                else:
                    upper_jd = mid_jd
                
                mid_jd = (lower_jd + upper_jd) / 2
            
            # Convert Julian day to calendar date
            y, m, d, h = swe.revjul(mid_jd, swe.GREG_CAL)
            ingress_date = datetime.date(y, m, d)
            
            # Add to ingresses
            ingresses.append({
                'planet': planet_id,
                'date': ingress_date,
                'from_sign': current_sign,
                'to_sign': new_sign,
                'exact_jd': mid_jd,
                'is_retrograde': is_retrograde(mid_jd, planet_id)  
            })
            
            # Update current sign
            current_sign = new_sign
        
        # Move to next step
        current_jd += step
    
    return ingresses

def get_market_data(symbol, ingress_dates, window_days=30):
    """
    Get market data around ingress dates.
    
    Args:
        symbol: Market symbol to analyze
        ingress_dates: List of dates when ingresses occur
        window_days: Number of days to analyze before/after each ingress
        
    Returns:
        Dictionary mapping ingress dates to market data
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
    elif symbol.upper() in ['BITCOIN', 'BTC', 'BTC-USD']:
        yf_symbol = 'BTC-USD'  # Bitcoin
    elif symbol.upper() in ['ETH', 'ETHEREUM', 'ETH-USD']:
        yf_symbol = 'ETH-USD'  # Ethereum
    elif symbol.upper() in ['EUR/USD', 'EURUSD=X']:
        yf_symbol = 'EURUSD=X'  # EUR/USD
    else:
        yf_symbol = symbol
    
    # Find the earliest and latest dates we need data for
    earliest_date = min(ingress_dates) - datetime.timedelta(days=window_days)
    latest_date = max(ingress_dates) + datetime.timedelta(days=window_days)
    
    # Get all market data for the full range
    try:
        all_data = yf.download(yf_symbol, start=earliest_date, end=latest_date, progress=False)
        
        if all_data.empty:
            print(f"No data available for {symbol} in the specified date range.")
            return {}
        
        # Prepare result dictionary
        result = {}
        
        # For each ingress date, extract the relevant window of data
        for ingress_date in ingress_dates:
            start_date = ingress_date - datetime.timedelta(days=window_days)
            end_date = ingress_date + datetime.timedelta(days=window_days)
            
            # Get data within this window
            mask = (all_data.index >= pd.Timestamp(start_date)) & (all_data.index <= pd.Timestamp(end_date))
            window_data = all_data.loc[mask].copy()
            
            # Calculate the daily percent change
            window_data['Pct_Change'] = window_data['Close'].pct_change() * 100
            
            # Calculate days relative to ingress date
            window_data['Days_From_Ingress'] = [(d.date() - ingress_date).days for d in window_data.index]
            
            # Store in result
            result[ingress_date] = window_data
        
        return result
        
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return {}

def analyze_market_behavior(market_data, ingress_date):
    """
    Analyze market behavior around an ingress date.
    
    Args:
        market_data: Market data around ingress date
        ingress_date: Date of the ingress
        
    Returns:
        Dictionary with analysis results
    """
    if ingress_date not in market_data or market_data[ingress_date].empty:
        return {
            'has_data': False,
            'price_at_ingress': None,
            'is_turning_point': False,
            'pattern': None,
            'before_change': None,
            'after_change': None
        }
    
    data = market_data[ingress_date]
    
    # Find the closest trading day to the ingress date
    closest_idx = data['Days_From_Ingress'].abs().idxmin()
    ingress_day_offset = data.loc[closest_idx, 'Days_From_Ingress']
    
    # Get price at ingress date
    price_at_ingress = float(data.loc[closest_idx, 'Close'])
    
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
            'price_at_ingress': price_at_ingress,
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
    is_high = price_at_ingress >= pre_max and price_at_ingress >= post_max
    is_low = price_at_ingress <= pre_min and price_at_ingress <= post_min
    
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
        'price_at_ingress': price_at_ingress,
        'is_turning_point': is_high or is_low,
        'pattern': pattern,
        'before_change': before_change,
        'after_change': after_change
    }

def get_planet_name(planet_id):
    """Get the name of a planet from its Swiss Ephemeris ID."""
    names = {
        swe.SUN: "Sun",
        swe.MOON: "Moon",
        swe.MERCURY: "Mercury",
        swe.VENUS: "Venus",
        swe.MARS: "Mars",
        swe.JUPITER: "Jupiter",
        swe.SATURN: "Saturn",
        swe.URANUS: "Uranus",
        swe.NEPTUNE: "Neptune",
        swe.PLUTO: "Pluto",
        swe.TRUE_NODE: "North Node"
    }
    return names.get(planet_id, str(planet_id))

def get_planet_symbol(planet_id):
    """Get the astronomical symbol for a planet based on its Swiss Ephemeris ID."""
    symbols = {
        swe.SUN: "☉",
        swe.MOON: "☽",
        swe.MERCURY: "☿",
        swe.VENUS: "♀",
        swe.MARS: "♂",
        swe.JUPITER: "♃",
        swe.SATURN: "♄",
        swe.URANUS: "♅",
        swe.NEPTUNE: "♆",
        swe.PLUTO: "♇",
        swe.TRUE_NODE: "☊"
    }
    return symbols.get(planet_id, "")

def is_retrograde(jd, planet_id):
    """
    Determine if a planet is retrograde at a given Julian day.
    
    Args:
        jd: Julian day number
        planet_id: Swiss Ephemeris ID for the planet
        
    Returns:
        Boolean: True if retrograde, False if direct
    """
    # Set up flag for calculation
    flag = swe.FLG_SWIEPH | swe.FLG_SPEED
    
    # Get position and speed
    pos, status = swe.calc_ut(jd, planet_id, flag)
    
    # The second element of the pos array contains the longitude speed
    # If it's negative, the planet is retrograde
    longitude_speed = pos[3]
    
    return longitude_speed < 0

def visualize_ingress_analysis(ingress_details, market_data, market_symbol, start_date, end_date, planet_id):
    """
    Create visualizations of market behavior around ingress dates.
    
    Args:
        ingress_details: List of dictionaries with ingress details
        market_data: Dictionary mapping ingress dates to market data
        market_symbol: Market symbol for title/labeling
        start_date: Start date of analysis period
        end_date: End date of analysis period
        planet_id: Swiss Ephemeris ID of the planet analyzed
    """
    # Create a directory name based on parameters
    date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    planet_name = get_planet_name(planet_id)
    
    folder_name = f"ingress_{market_symbol}_{planet_name}_{date_range}"
    # Ensure folder name is valid
    folder_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in folder_name)
    
    # Create the directory
    os.makedirs(folder_name, exist_ok=True)
    
    # Plot each ingress
    for ingress in ingress_details:
        ingress_date = ingress['date']
        
        if ingress_date not in market_data or market_data[ingress_date].empty:
            print(f"No market data available for {ingress['to_sign']} ingress on {ingress_date}")
            continue
        
        data = market_data[ingress_date]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot price data
        plt.plot(data.index, data['Close'])
        
        # Mark ingress date
        closest_idx = data['Days_From_Ingress'].abs().idxmin()
        ingress_price = data.loc[closest_idx, 'Close']
        
        plt.axvline(x=closest_idx, color='r', linestyle='--')
        plt.plot(closest_idx, ingress_price, 'ro', markersize=8)
        
        # Add annotation
        from_sign = ingress['from_sign']
        to_sign = ingress['to_sign']
        
        title = f"{planet_name} Ingress: {from_sign} → {to_sign}\n{ingress_date.strftime('%Y-%m-%d')}"
        plt.title(title)
        
        # First, for the individual chart annotation (around line 325-345):
        # Check if ingress_price is a pandas Series
        if hasattr(ingress_price, 'iloc'):
            price_value = float(ingress_price.iloc[0])
        else:
            price_value = float(ingress_price)

        # Add retrograde symbol if applicable
        motion_symbol = "☿℞" if ingress['is_retrograde'] else "☿"  # Mercury symbol with retrograde symbol

       # Get planet symbol
        planet_symbol = get_planet_symbol(planet_id)

        # Add retrograde indicator if applicable
        motion_symbol = f"{planet_symbol}℞" if ingress['is_retrograde'] else planet_symbol

        # Add label with date, price, sign entered, and motion
        plt.annotate(
            f"{ingress_date.strftime('%Y-%m-%d')}\n${price_value:.2f}\n{to_sign} {motion_symbol}",xy=(closest_idx, ingress_price),
            xytext=(0, 30),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
        )

        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # Plot percent change
        plt.figure(figsize=(12, 8))
        plt.bar(data.index, data['Pct_Change'])
        plt.axvline(x=closest_idx, color='r', linestyle='--')
        plt.title(f"Daily % Change: {planet_name} Ingress {to_sign} ({ingress_date.strftime('%Y-%m-%d')})")
        plt.ylabel('Daily % Change')
        plt.grid(True, alpha=0.3)
        
        # Save the chart
        chart_filename = f"{folder_name}/{planet_name}_{to_sign}_{ingress_date.strftime('%Y-%m-%d')}.png"
        chart_filename = chart_filename.replace(" ", "_").lower()
        plt.savefig(chart_filename)
        plt.close()
        
        # Save the percent change chart
        pct_chart_filename = f"{folder_name}/{planet_name}_{to_sign}_{ingress_date.strftime('%Y-%m-%d')}_pct_change.png"
        pct_chart_filename = pct_chart_filename.replace(" ", "_").lower()
        plt.savefig(pct_chart_filename)
        plt.close()
        
        print(f"Charts saved as {chart_filename} and {pct_chart_filename}")
    
    # Create overall summary chart
    plt.figure(figsize=(16, 10))
    
    # Get unique signs for color coding
    all_signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", 
                "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    colors = plt.cm.tab10.colors + plt.cm.Set3.colors  # Get more colors for 12 signs
    
    # Create a mapping from sign to color
    sign_colors = {sign: colors[i % len(colors)] for i, sign in enumerate(all_signs)}
    
    # Get earliest and latest dates across all ingresses
    all_dates = [i['date'] for i in ingress_details]
    if all_dates:
        earliest_date = min(all_dates) - datetime.timedelta(days=30)
        latest_date = max(all_dates) + datetime.timedelta(days=30)
        
        # Get market data for this entire period
        try:
            if market_symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
                yf_symbol = '^GSPC'
            elif market_symbol.upper() in ['DJI', 'DJIA', 'DOW']:
                yf_symbol = '^DJI'
            elif market_symbol.upper() in ['IXIC', 'NASDAQ', 'NDX']:
                yf_symbol = '^IXIC'
            elif market_symbol.upper() in ['GOLD', 'XAU']:
                yf_symbol = 'GC=F'
            elif market_symbol.upper() in ['SILVER', 'XAG']:
                yf_symbol = 'SI=F'
            elif market_symbol.upper() in ['BITCOIN', 'BTC', 'BTC-USD']:
                yf_symbol = 'BTC-USD'
            elif market_symbol.upper() in ['ETH', 'ETHEREUM', 'ETH-USD']:
                yf_symbol = 'ETH-USD'
            elif market_symbol.upper() in ['EUR/USD', 'EURUSD=X']:
                yf_symbol = 'EURUSD=X'
            else:
                yf_symbol = market_symbol
                
            full_data = yf.download(yf_symbol, start=earliest_date, end=latest_date, progress=False)
            
            if not full_data.empty:
                # Plot the market for the entire period
                plt.plot(full_data.index, full_data['Close'], 'k-', alpha=0.7)
                
                # Create legend patches
                legend_patches = []
                
                # Mark each ingress date
                for ingress in ingress_details:
                    ingress_date = ingress['date']
                    to_sign = ingress['to_sign']
                    
                    # Get the pandas timestamp closest to this date
                    closest_date = None
                    min_diff = float('inf')
                    
                    for date_idx in full_data.index:
                        date_diff = abs((date_idx.date() - ingress_date).days)
                        if date_diff < min_diff:
                            min_diff = date_diff
                            closest_date = date_idx
                    
                    if closest_date is not None and min_diff <= 5:  # Only if we have a close match
                        price = full_data.loc[closest_date, 'Close']
                        
                        # Use color based on sign
                        color = sign_colors[to_sign]
                        
                        # Add to legend if this sign hasn't been added yet
                        if to_sign not in [p.get_label() for p in legend_patches]:
                            legend_patches.append(mpatches.Patch(color=color, label=to_sign))
                        
                        # Mark on chart
                        plt.axvline(x=closest_date, color=color, linestyle='--', alpha=0.5)
                        plt.plot(closest_date, price, 'o', color=color, markersize=6)
                        
                        # Add annotation with date, price, and sign
                        if hasattr(price, 'iloc'):
                            price_value = float(price.iloc[0])
                        else:
                            price_value = float(price)

                        # Add retrograde symbol if applicable
                        motion_symbol = "☿℞" if ingress['is_retrograde'] else "☿"  # Mercury symbol with retrograde symbol

                        # Add annotation with date, price, sign, and motion
                        plt.annotate(
                            f"{ingress_date.strftime('%Y-%m-%d')}\n${price_value:.2f}\n{to_sign} {motion_symbol}",
                            xy=(closest_date, price),
                            xytext=(0, 20),
                            textcoords="offset points",
                            fontsize=8,
                            color=color,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                        )
                
                plt.title(f"{planet_name} Sign Ingresses - {market_symbol}")
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
        
        plt.close()
    
    # Create a zodiac wheel visualization showing market behavior by sign
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
    
    # Define sign angles (each sign is 30°)
    sign_angles = np.linspace(0, 2*np.pi, 13)[:-1]  # 12 angles, remove the last (duplicate)
    
    # Group analyses by sign
    sign_data = {}
    for ingress in ingress_details:
        to_sign = ingress['to_sign']
        if to_sign not in sign_data:
            sign_data[to_sign] = []
        
        ingress_date = ingress['date']
        if ingress_date in market_data and not market_data[ingress_date].empty:
            analysis = analyze_market_behavior(market_data, ingress_date)
            if analysis['has_data']:
                sign_data[to_sign].append(analysis)
    
    # Calculate average performance by sign
    avg_performance = {}
    for sign, analyses in sign_data.items():
        if analyses:
            after_changes = [a['after_change'] for a in analyses if a['after_change'] is not None]
            if after_changes:
                avg_performance[sign] = sum(after_changes) / len(after_changes)
            else:
                avg_performance[sign] = 0
        else:
            avg_performance[sign] = 0
    
    # Plot the data
    bars = []
    for i, sign in enumerate(all_signs):
        angle = sign_angles[i]
        performance = avg_performance.get(sign, 0)
        
        # Use color based on performance
        if performance > 0:
            color = 'green'
        else:
            color = 'red'
        
        # Plot bar
        bar = ax.bar(angle, abs(performance), width=0.5, bottom=0, alpha=0.7, color=color)
        bars.append(bar)
        
        # Add sign label
        ax.text(angle, max(5, abs(performance) + 2), sign, horizontalalignment='center', verticalalignment='center')
        
        # Add performance value
        if sign in avg_performance:
            ax.text(angle, 2, f"{avg_performance[sign]:.2f}%", horizontalalignment='center', verticalalignment='center')
    
    # Set chart properties
    ax.set_theta_zero_location('N')  # Start from the top (Aries)
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_thetagrids(np.degrees(sign_angles), all_signs)
    ax.set_title(f"Average Market Performance After {planet_name} Sign Ingress")
    
    # Save chart
    chart_filename = f"{folder_name}/Sign_Performance_Wheel.png"
    plt.savefig(chart_filename)
    print(f"Zodiac wheel chart saved as {chart_filename}")
    plt.close()

def display_results(ingress_details, market_analyses, market_symbol, planet_id):
    """Display the analysis results in a formatted table."""
    planet_name = get_planet_name(planet_id)
    
    print("\n" + "="*120)
    print(f"{planet_name.upper()} SIGN INGRESS ANALYSIS FOR {market_symbol}")
    print("="*120)
    
    if not ingress_details:
        print("No ingresses found in the specified date range.")
        return
    
    # Print header
    print(f"{'Date':12} | {'From Sign':12} | {'To Sign':12} | {'Price':10} | {'Pattern':20} | {'% Before':10} | {'% After':10} | {'Turning Point':12}")
    print("-"*120)
    
    # Print each ingress
    for ingress in ingress_details:
        ingress_date = ingress['date']
        
        # Get analysis if available
        if ingress_date in market_analyses:
            analysis = market_analyses[ingress_date]
        else:
            analysis = {'has_data': False, 'price_at_ingress': None, 'is_turning_point': False, 'pattern': None, 'before_change': None, 'after_change': None}
        
        # Format fields
        date_str = ingress_date.strftime('%Y-%m-%d')
        from_sign = ingress['from_sign']
        to_sign = ingress['to_sign']
        
        if analysis['has_data']:
            price = f"${analysis['price_at_ingress']:.2f}" if analysis['price_at_ingress'] is not None else "N/A"
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


        # And in the results row:
        motion = "R" if ingress['is_retrograde'] else "D"
        print(f"{date_str:12} | {from_sign:12} | {to_sign:12} | {motion:4} | {price:10} | {pattern:20} | {before_change:10} | {after_change:10} | {turning_point:12}")
        # Print row
        print("="*132)

def display_statistics(stats, market_symbol, planet_name):
    """Display statistical summary by zodiac sign."""
    print("\n" + "="*100)
    print(f"STATISTICAL SUMMARY FOR {planet_name.upper()} INGRESSES - {market_symbol}")
    print("="*100)
    
    if not stats:
        print("No statistical data available.")
        return
    
    # Print header
    print(f"{'Zodiac Sign':12} | {'Count':5} | {'Turning Points %':15} | {'Dominant Pattern':20} | {'Avg % Before':12} | {'Avg % After':12}")
    print("-"*100)
    
    # Sort by zodiac sign in natural order
    zodiac_order = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", 
                    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    
    for sign in zodiac_order:
        if sign in stats:
            data = stats[sign]
            
            # Format fields
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
            print(f"{sign:12} | {count:5} | {turning_point_pct:15} | {dominant_pattern:20} | {avg_before:12} | {avg_after:12}")
    
    print("="*100)

def calculate_statistics(ingress_details, market_analyses):
    """
    Calculate statistics for different zodiac signs.
    
    Args:
        ingress_details: List of dictionaries with ingress details
        market_analyses: Dictionary mapping ingress dates to market analyses
        
    Returns:
        Dictionary with statistics by zodiac sign
    """
    # Initialize statistics
    stats = {}
    
    # Group by zodiac sign
    for ingress in ingress_details:
        ingress_date = ingress['date']
        to_sign = ingress['to_sign']
        
        # Create key if it doesn't exist
        if to_sign not in stats:
            stats[to_sign] = {
                'count': 0,
                'has_data_count': 0,
                'retrograde_count': 0,
                'direct_count': 0,
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
        
        # Then when updating counts:
        if ingress['is_retrograde']:
            stats[to_sign]['retrograde_count'] += 1
        else:
            stats[to_sign]['direct_count'] += 1

        # Get analysis if available
        if ingress_date in market_analyses:
            analysis = market_analyses[ingress_date]
            
            # Update statistics
            stats[to_sign]['count'] += 1
            
            if analysis['has_data']:
                stats[to_sign]['has_data_count'] += 1
                
                if analysis['is_turning_point']:
                    stats[to_sign]['turning_points'] += 1
                
                pattern = analysis['pattern']
                if pattern in stats[to_sign]['patterns']:
                    stats[to_sign]['patterns'][pattern] += 1
                
                if analysis['before_change'] is not None:
                    stats[to_sign]['before_changes'].append(analysis['before_change'])
                
                if analysis['after_change'] is not None:
                    stats[to_sign]['after_changes'].append(analysis['after_change'])
    
    # Calculate percentages and averages
    for sign, data in stats.items():
        if data['has_data_count'] > 0:
            data['turning_point_pct'] = (data['turning_points'] / data['has_data_count']) * 100
            # Calculate retrograde percentage:
            data['retrograde_pct'] = (data['retrograde_count'] / data['count']) * 100 if data['count'] > 0 else 0

            data['avg_before_change'] = sum(data['before_changes']) / len(data['before_changes']) if data['before_changes'] else None
            data['avg_after_change'] = sum(data['after_changes']) / len(data['after_changes']) if data['after_changes'] else None
            
            # Calculate dominant pattern
            max_pattern = max(data['patterns'].items(), key=lambda x: x[1]) if data['patterns'] else ('None', 0)
            data['dominant_pattern'] = max_pattern[0] if max_pattern[1] > 0 else 'None'
    
    return stats

def save_to_csv(ingress_details, market_analyses, market_symbol, planet_id, folder_name=None):
    """
    Save ingress analysis results to CSV files.
    
    Args:
        ingress_details: List of dictionaries with ingress details
        market_analyses: Dictionary mapping ingress dates to market analyses
        market_symbol: Market symbol (for filename)
        planet_id: Swiss Ephemeris ID of the planet analyzed
        folder_name: Folder to save the CSV in (optional)
    """
    # Get planet name
    planet_name = get_planet_name(planet_id)
    
    # Prepare data for CSV
    ingress_data = []
    
    for ingress in ingress_details:
        ingress_date = ingress['date']
        
        # Get analysis if available
        if ingress_date in market_analyses:
            analysis = market_analyses[ingress_date]
        else:
            analysis = {'has_data': False, 'price_at_ingress': None, 'is_turning_point': False, 'pattern': None, 'before_change': None, 'after_change': None}
        
        # Create row
        row = {
            'Date': ingress_date,
            'Planet': planet_name,
            'From_Sign': ingress['from_sign'],
            'To_Sign': ingress['to_sign'],
            'Motion': 'Retrograde' if ingress['is_retrograde'] else 'Direct',
            'Has_Price_Data': analysis['has_data'],
            'Price': analysis['price_at_ingress'],
            'Is_Turning_Point': analysis['is_turning_point'],
            'Pattern': analysis['pattern'],
            'Before_Change': analysis['before_change'],
            'After_Change': analysis['after_change']
        }
        
        ingress_data.append(row)
    
    # Create DataFrame and save
    if ingress_data:
        df = pd.DataFrame(ingress_data)
        
        # Format the market symbol and planet name for filename
        safe_symbol = "".join(c if c.isalnum() else "_" for c in market_symbol)
        safe_planet = "".join(c if c.isalnum() else "_" for c in planet_name)
        
        # If folder name is not provided, use current directory
        if folder_name is None:
            # Create default folder name
            date_range = f"{min(i['date'] for i in ingress_details).strftime('%Y%m%d')}-{max(i['date'] for i in ingress_details).strftime('%Y%m%d')}"
            folder_name = f"ingress_{safe_symbol}_{safe_planet}_{date_range}"
            folder_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in folder_name)
            os.makedirs(folder_name, exist_ok=True)
        
        filename = f"{folder_name}/{safe_symbol}_{safe_planet}_ingress_analysis.csv"
        
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        # Create a statistics summary CSV
        stats = calculate_statistics(ingress_details, market_analyses)
        
        if stats:
            stats_data = []
            
            for sign, data in stats.items():
                stats_row = {
                    'Sign': sign,
                    'Count': data['count'],
                    'Has_Data_Count': data['has_data_count'],
                    'Turning_Points': data['turning_points'],
                    'Turning_Points_Pct': data.get('turning_point_pct'),
                    'Dominant_Pattern': data.get('dominant_pattern'),
                    'Avg_Before_Change': data.get('avg_before_change'),
                    'Avg_After_Change': data.get('avg_after_change')
                }
                
                # Add pattern counts
                for pattern, count in data['patterns'].items():
                    stats_row[f'Pattern_{pattern.replace(" ", "_")}'] = count
                
                stats_data.append(stats_row)
            
            stats_df = pd.DataFrame(stats_data)
            stats_filename = f"{folder_name}/{safe_symbol}_{safe_planet}_sign_statistics.csv"
            stats_df.to_csv(stats_filename, index=False)
            print(f"Statistics saved to {stats_filename}")
    else:
        print("No data to save.")

def main():
    """Main function to run the analysis."""
    print("Planetary Sign Ingress Analysis Tool")
    print("-----------------------------------")
    
    # Get date range
    print("Enter start date (YYYY-MM-DD):")
    start_date_str = input()
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    
    print("Enter end date (YYYY-MM-DD):")
    end_date_str = input()
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    # Get planet selection
    print("Select planet to analyze:")
    print("Options: Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, North Node")
    planet_input = input().strip()
    
    # Map planet names to Swiss Ephemeris IDs
    planet_map = {
        'Sun': swe.SUN,
        'Moon': swe.MOON,
        'Mercury': swe.MERCURY,
        'Venus': swe.VENUS,
        'Mars': swe.MARS,
        'Jupiter': swe.JUPITER,
        'Saturn': swe.SATURN,
        'Uranus': swe.URANUS,
        'Neptune': swe.NEPTUNE,
        'Pluto': swe.PLUTO,
        'North Node': swe.TRUE_NODE
    }
    
    # Convert selected planet name to ID
    if planet_input in planet_map:
        planet_id = planet_map[planet_input]
    else:
        print(f"Warning: Planet {planet_input} not recognized. Using Jupiter as default.")
        planet_id = swe.JUPITER
        planet_input = "Jupiter"
    
    # Get market symbol
    print("Enter market symbol (e.g., SPX, NASDAQ, DJI, GOLD, SILVER, BTC-USD, EUR/USD):")
    market_symbol = input()
    
    # Find ingresses
    print(f"\nFinding {planet_input} sign ingresses between {start_date} and {end_date}...")
    ingresses = find_sign_ingresses(start_date, end_date, planet_id)
    
    if not ingresses:
        print("No ingresses found in the specified date range.")
        return
    
    print(f"Found {len(ingresses)} ingresses.")
    
    # Get market data around ingresses
    print("\nRetrieving market data...")
    ingress_dates = [i['date'] for i in ingresses]
    market_data = get_market_data(market_symbol, ingress_dates)
    
    # Analyze market behavior
    print("\nAnalyzing market behavior...")
    market_analyses = {}
    for ingress_date in ingress_dates:
        market_analyses[ingress_date] = analyze_market_behavior(market_data, ingress_date)
    
    # Calculate statistics
    stats = calculate_statistics(ingresses, market_analyses)
    
    # Display results
    display_results(ingresses, market_analyses, market_symbol, planet_id)
    
    # Display statistics
    display_statistics(stats, market_symbol, planet_input)
    
    # Option to save results
    print("\nWould you like to save the results to CSV files? (y/n)")
    if input().lower() == 'y':
        save_to_csv(ingresses, market_analyses, market_symbol, planet_id)
    
    # Option to visualize results
    print("\nWould you like to visualize the results? (y/n)")
    if input().lower() == 'y':
        print("\nCreating visualizations...")
        visualize_ingress_analysis(ingresses, market_data, market_symbol, start_date, end_date, planet_id)
        print("\nVisualization complete.")

if __name__ == "__main__":
    main()
