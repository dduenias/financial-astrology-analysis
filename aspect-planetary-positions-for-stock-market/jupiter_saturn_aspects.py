import swisseph as swe
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from math import fmod
import matplotlib.patches as mpatches

def find_aspects(start_date, end_date, planet1, planet2, aspects_to_find, orb=1.0, is_heliocentric=False):
    """
    Find dates when two planets form specific aspects within a date range.
    
    Args:
        start_date: Beginning of search range
        end_date: End of search range
        planet1: Swiss Ephemeris ID for first planet
        planet2: Swiss Ephemeris ID for second planet
        aspects_to_find: List of aspects to search for in degrees (e.g., [0, 90, 180])
        orb: Maximum allowed orb in degrees
        is_heliocentric: If True, use heliocentric positions, else geocentric
        
    Returns:
        List of dictionaries with aspect dates and details
    """
    # Initialize ephemeris
    swe.set_ephe_path(None)
    
    # Convert dates to Julian days
    jd_start = swe.julday(start_date.year, start_date.month, start_date.day, 0)
    jd_end = swe.julday(end_date.year, end_date.month, end_date.day, 0)
    
    # Step size for scanning (1 day)
    step = 1
    
    # Store found aspects
    found_aspects = []
    
    # Set up flags
    flag = swe.FLG_SWIEPH
    helio_flag = swe.FLG_SWIEPH | swe.FLG_SPEED
    
    # Track when we enter and exit orb ranges to find exact aspect dates
    active_aspects = {}
    
    # Scan through the date range
    current_jd = jd_start
    while current_jd <= jd_end:
        if is_heliocentric:
            # For heliocentric positions, we'll use a different approach
            # Get the ecliptic coordinates
            e1, status1 = swe.calc_ut(current_jd, planet1, flag)
            e2, status2 = swe.calc_ut(current_jd, planet2, flag)
            
            # Extract longitudes
            lon1 = e1[0]
            lon2 = e2[0]
        else:
            # For geocentric positions
            pos1, status1 = swe.calc_ut(current_jd, planet1, flag)
            pos2, status2 = swe.calc_ut(current_jd, planet2, flag)
            
            # Extract longitudes
            lon1 = pos1[0]
            lon2 = pos2[0]
        
        # Calculate angular difference, ensuring it's in the range 0-180
        diff = abs(lon1 - lon2) % 360
        if diff > 180:
            diff = 360 - diff
        
        # Check each aspect we're looking for
        for aspect in aspects_to_find:
            aspect_key = f"{aspect}"
            
            # Check if we're within orb of the aspect
            if abs(diff - aspect) <= orb:
                # If we're not tracking this aspect yet, add it
                if aspect_key not in active_aspects:
                    active_aspects[aspect_key] = {
                        'start_jd': current_jd,
                        'min_diff': abs(diff - aspect),
                        'min_diff_jd': current_jd
                    }
                # Otherwise update the minimum difference if needed
                elif abs(diff - aspect) < active_aspects[aspect_key]['min_diff']:
                    active_aspects[aspect_key]['min_diff'] = abs(diff - aspect)
                    active_aspects[aspect_key]['min_diff_jd'] = current_jd
            
            # If we're tracking this aspect but now out of orb, record it
            elif aspect_key in active_aspects:
                # Get the date when the aspect was most exact
                exact_jd = active_aspects[aspect_key]['min_diff_jd']
                
                # Convert to calendar date
                y, m, d, h = swe.revjul(exact_jd, swe.GREG_CAL)
                exact_date = datetime.date(y, m, d)
                
                # Add to found aspects
                found_aspects.append({
                    'planet1': planet1,
                    'planet2': planet2,
                    'aspect': aspect,
                    'date': exact_date,
                    'is_heliocentric': is_heliocentric,
                    'orb': active_aspects[aspect_key]['min_diff']
                })
                
                # Remove from active tracking
                del active_aspects[aspect_key]
        
        # Move to next day
        current_jd += step
    
    # Check if any aspects are still active at the end of the range
    for aspect_key, data in active_aspects.items():
        aspect = float(aspect_key)
        exact_jd = data['min_diff_jd']
        
        # Convert to calendar date
        y, m, d, h = swe.revjul(exact_jd, swe.GREG_CAL)
        exact_date = datetime.date(y, m, d)
        
        # Add to found aspects
        found_aspects.append({
            'planet1': planet1,
            'planet2': planet2,
            'aspect': aspect,
            'date': exact_date,
            'is_heliocentric': is_heliocentric,
            'orb': data['min_diff']
        })
    
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
            'pattern': None
        }
    
    data = market_data[aspect_date]
    
    # Find the closest trading day to the aspect date
    closest_idx = data['Days_From_Aspect'].abs().idxmin()
    aspect_day_offset = data.loc[closest_idx, 'Days_From_Aspect']
    
    # Get price at aspect date - ensure it's a scalar value
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
            'pattern': 'Insufficient data'
        }
    
    # Check if it's a local maximum (high) - convert to scalar values
    pre_max = float(pre_data['Close'].max())
    post_max = float(post_data['Close'].max())
    
    # Check if it's a local minimum (low) - convert to scalar values
    pre_min = float(pre_data['Close'].min())
    post_min = float(post_data['Close'].min())
    
    # Determine if it's a turning point
    is_high = price_at_aspect >= pre_max and price_at_aspect >= post_max
    is_low = price_at_aspect <= pre_min and price_at_aspect <= post_min
    
    # Determine pattern
    if is_high:
        pattern = 'High'
    elif is_low:
        pattern = 'Low'
    else:
        # Calculate trend before and after - convert to scalar values
        pre_first = float(pre_data['Close'].iloc[0])
        pre_last = float(pre_data['Close'].iloc[-1])
        post_first = float(post_data['Close'].iloc[0])
        post_last = float(post_data['Close'].iloc[-1])
        
        pre_trend = (pre_last - pre_first) / pre_first
        post_trend = (post_last - post_first) / post_first
        
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
        'pattern': pattern
    }

def visualize_aspect_analysis(aspect_details, market_data):
    """
    Create visualizations of market behavior around aspect dates.
    
    Args:
        aspect_details: List of dictionaries with aspect details
        market_data: Dictionary mapping aspect dates to market data
    """
    # Define aspect names
    aspect_names = {
        0: "Conjunction",
        60: "Sextile",
        90: "Square",
        120: "Trine",
        180: "Opposition"
    }
    
    # Plot each aspect
    for aspect in aspect_details:
        aspect_date = aspect['date']
        
        if aspect_date not in market_data or market_data[aspect_date].empty:
            print(f"No market data available for {aspect_names.get(aspect['aspect'], aspect['aspect'])} on {aspect_date}")
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
        planet_names = {
            swe.JUPITER: "Jupiter",
            swe.SATURN: "Saturn",
            swe.URANUS: "Uranus",
            swe.NEPTUNE: "Neptune",
            swe.PLUTO: "Pluto"
        }
        
        p1_name = planet_names.get(aspect['planet1'], str(aspect['planet1']))
        p2_name = planet_names.get(aspect['planet2'], str(aspect['planet2']))
        aspect_type = aspect_names.get(aspect['aspect'], f"{aspect['aspect']}°")
        system = "Heliocentric" if aspect['is_heliocentric'] else "Geocentric"
        
        title = f"{p1_name}-{p2_name} {aspect_type} ({system})\n{aspect_date.strftime('%Y-%m-%d')}"
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
        chart_filename = f"{p1_name}_{p2_name}_{aspect_type}_{aspect_date.strftime('%Y-%m-%d')}.png"
        chart_filename = chart_filename.replace(" ", "_")
        plt.savefig(chart_filename)
        print(f"Chart saved as {chart_filename}")
        
        plt.close()
    
    # Create a summary chart
    plt.figure(figsize=(14, 10))
    
    # Get unique aspect types
    aspect_types = sorted(set(a['aspect'] for a in aspect_details))
    markers = ['o', 's', '^', 'D', 'x']
    
    # Plot all data points
    for i, aspect_type in enumerate(aspect_types):
        # Filter aspects of this type
        filtered_aspects = [a for a in aspect_details if a['aspect'] == aspect_type]
        
        # Extract dates and corresponding prices
        dates = []
        prices = []
        patterns = []
        
        for aspect in filtered_aspects:
            aspect_date = aspect['date']
            
            if aspect_date in market_data and not market_data[aspect_date].empty:
                data = market_data[aspect_date]
                closest_idx = data['Days_From_Aspect'].abs().idxmin()
                aspect_price = data.loc[closest_idx, 'Close']
                
                dates.append(aspect_date)
                prices.append(aspect_price)
                
                # Get pattern
                analysis = analyze_market_behavior(market_data, aspect_date)
                patterns.append(analysis['pattern'])
        
        if dates:
            # Convert to pandas datetime for plotting
            pd_dates = pd.to_datetime(dates)
            
            # Plot this aspect type
            aspect_name = aspect_names.get(aspect_type, f"{aspect_type}°")
            marker = markers[i % len(markers)]
            
            plt.plot(pd_dates, prices, marker, label=aspect_name, markersize=10)
            
            # Add simple annotations for patterns
            for date, price, pattern in zip(pd_dates, prices, patterns):
                plt.annotate(pattern, 
                           xy=(date, price), 
                           xytext=(5, 5),
                           textcoords="offset points",
                           fontsize=8)
    
    plt.title("Jupiter-Saturn Aspects and Market Behavior")
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the summary chart
    plt.savefig("Aspect_Summary_Chart.png")
    print("Summary chart saved as Aspect_Summary_Chart.png")
    plt.close()

def display_results(aspect_details, market_analyses):
    """
    Display the analysis results in a formatted table.
    """
    print("\n" + "="*100)
    print("JUPITER-SATURN ASPECT ANALYSIS")
    print("="*100)
    
    # Define aspect names
    aspect_names = {
        0: "Conjunction",
        30: "Semi-Sextile",
        60: "Sextile",
        90: "Square",
        120: "Trine", 
        150: "Quincunx",
        180: "Opposition",
        210: "Quincunx",
        225: "Sesqui-Square",
        240: "Trine",
        270: "Square",
        300: "Sextile",
        330: "Semi-Sextile"
    }
    
    # Get planet names
    planet_names = {
        swe.JUPITER: "Jupiter",
        swe.SATURN: "Saturn",
        swe.URANUS: "Uranus",
        swe.NEPTUNE: "Neptune",
        swe.PLUTO: "Pluto"
    }
    
    # Print header
    print(f"{'Aspect Type':15} | {'Date':12} | {'System':12} | {'Price':10} | {'Pattern':20} | {'Turning Point':15}")
    print("-"*100)
    
    # Print each aspect
    for aspect in aspect_details:
        aspect_date = aspect['date']
        
        # Get analysis if available
        if aspect_date in market_analyses:
            analysis = market_analyses[aspect_date]
        else:
            analysis = {'has_data': False, 'price_at_aspect': None, 'is_turning_point': False, 'pattern': None}
        
        # Format fields
        aspect_type = aspect_names.get(aspect['aspect'], f"{aspect['aspect']}°")
        date_str = aspect_date.strftime('%Y-%m-%d')
        system = "Heliocentric" if aspect['is_heliocentric'] else "Geocentric"
        
        if analysis['has_data']:
            price = f"${analysis['price_at_aspect']:.2f}" if analysis['price_at_aspect'] is not None else "N/A"
            pattern = analysis['pattern'] or "N/A"
            turning_point = "Yes" if analysis['is_turning_point'] else "No"
        else:
            price = "N/A"
            pattern = "No data"
            turning_point = "N/A"
        
        # Print row
        print(f"{aspect_type:15} | {date_str:12} | {system:12} | {price:10} | {pattern:20} | {turning_point:15}")
    
    print("="*100)

def save_to_csv(aspect_details, market_analyses, filename="Jupiter_Saturn_Aspects.csv"):
    """
    Save analysis results to a CSV file.
    """
    # Define aspect names
    aspect_names = {
        0: "Conjunction",
        30: "Semi-Sextile",
        60: "Sextile",
        90: "Square",
        120: "Trine",
        150: "Quincunx",
        180: "Opposition",
        210: "Quincunx",
        225: "Sesqui-Square",
        240: "Trine",
        270: "Square",
        300: "Sextile",
        330: "Semi-Sextile"
    }
    
    # Get planet names
    planet_names = {
        swe.JUPITER: "Jupiter",
        swe.SATURN: "Saturn",
        swe.URANUS: "Uranus",
        swe.NEPTUNE: "Neptune",
        swe.PLUTO: "Pluto"
    }
    
    # Prepare data
    data = []
    
    for aspect in aspect_details:
        aspect_date = aspect['date']
        
        # Get analysis if available
        if aspect_date in market_analyses:
            analysis = market_analyses[aspect_date]
        else:
            analysis = {'has_data': False, 'price_at_aspect': None, 'is_turning_point': False, 'pattern': None}
        
        # Create row
        row = {
            'Planet1': planet_names.get(aspect['planet1'], str(aspect['planet1'])),
            'Planet2': planet_names.get(aspect['planet2'], str(aspect['planet2'])),
            'Aspect': aspect_names.get(aspect['aspect'], f"{aspect['aspect']}°"),
            'Date': aspect_date,
            'System': "Heliocentric" if aspect['is_heliocentric'] else "Geocentric",
            'Orb': aspect['orb'],
            'Has_Price_Data': analysis['has_data'],
            'Price': analysis['price_at_aspect'],
            'Is_Turning_Point': analysis['is_turning_point'],
            'Pattern': analysis['pattern']
        }
        
        data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main():
    """Main function to run the analysis."""
    print("Jupiter-Saturn Aspect Analysis Tool")
    print("----------------------------------")
    
    # Get date range
    print("Enter start date (YYYY-MM-DD):")
    start_date_str = input()
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    
    print("Enter end date (YYYY-MM-DD):")
    end_date_str = input()
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    # Get market symbol
    print("Enter stock/index symbol:")
    market_symbol = input()
    
    # Get aspects to analyze
    print("Enter aspects to analyze (comma-separated, e.g., 0,90,180):")
    aspect_input = input()
    aspects_to_find = [float(a.strip()) for a in aspect_input.split(',')]
    
    # Get coordinate system preference
    print("Use heliocentric coordinates? (y/n):")
    is_heliocentric = input().lower() == 'y'
    
    # Find Jupiter-Saturn aspects
    print("\nFinding Jupiter-Saturn aspects...")
    aspects = find_aspects(
        start_date, 
        end_date, 
        swe.JUPITER, 
        swe.SATURN, 
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
    
    # Display results
    display_results(aspects, market_analyses)
    
    # Option to save results
    print("\nWould you like to save the results to a CSV file? (y/n)")
    if input().lower() == 'y':
        save_to_csv(aspects, market_analyses)
    
    # Option to visualize results
    print("\nWould you like to visualize the results? (y/n)")
    if input().lower() == 'y':
        print("\nCreating visualizations...")
        visualize_aspect_analysis(aspects, market_data)

if __name__ == "__main__":
    main()