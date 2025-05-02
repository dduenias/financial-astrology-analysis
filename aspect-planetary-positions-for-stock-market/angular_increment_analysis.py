'''the scripts Take a major high/low date as input
Calculate planetary positions for that date
Project forward to find dates when planets reach specific angles from their original position
Fetch price data for those projected dates
Analyze if these dates coincide with market turns'''

import swisseph as swe
import datetime
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from math import fmod
import matplotlib.patches as mpatches

# Import from your existing script - you may need to adjust this import path
# If the scripts are in the same directory, this should work
from financial_astrology import (
    initialize_ephemeris, 
    get_julian_day, 
    get_human_design_gate,
    get_stock_price,
    get_zodiac_sign,
    connect_to_database,
    display_results,
    save_to_csv
)


def find_future_angular_positions(start_date, planet_id, angles_to_check, max_days=365, cycles=3):
    """
    Find future dates when a planet reaches specific angular increments from its position on start_date.
    
    Args:
        start_date: The reference date (major high/low)
        planet_id: The Swiss Ephemeris ID for the planet to track
        angles_to_check: List of angles to check (e.g., [15, 30, 45, 90, 180])
        max_days: Maximum number of days to look ahead
        cycles: Number of full 360° cycles to calculate
        
    Returns:
        Dictionary mapping angles to dates when the planet reaches those increments
    """
    # Initialize ephemeris
    initialize_ephemeris()
    
    # Calculate initial position
    jd_start = get_julian_day(start_date)
    initial_result, _ = swe.calc_ut(jd_start, planet_id, swe.FLG_SWIEPH)
    initial_longitude = initial_result[0]
    
    # Print initial position for debugging
    print(f"Initial longitude: {initial_longitude:.4f}°")
    
    result_dates = {}
    
    # Expand angles to include multiple cycles
    expanded_angles = []
    original_angles = sorted(angles_to_check)
    for cycle in range(cycles):
        for angle in original_angles:
            expanded_angle = angle + (cycle * 360.0)
            if expanded_angle > 0:  # Skip 0° on subsequent cycles to avoid duplication
                expanded_angles.append(expanded_angle)
    
    # Keep track of angles found
    found_angles = set()
    
    # Iterate through future dates
    for days_ahead in range(1, max_days + 1):
        current_date = start_date + datetime.timedelta(days=days_ahead)
        jd_current = get_julian_day(current_date)
        
        current_result, _ = swe.calc_ut(jd_current, planet_id, swe.FLG_SWIEPH)
        current_longitude = current_result[0]
        
        # Calculate total angular movement
        # For simplicity, just count one cycle every time we cross 0°
        # This works well enough for most outer planets
        total_days = days_ahead
        
        # For Jupiter (~12 year cycle), estimate cycle count
        if planet_id == swe.JUPITER:
            cycles_passed = total_days // (365 * 12)
        # For Saturn (~29.5 year cycle), estimate cycle count
        elif planet_id == swe.SATURN:
            cycles_passed = total_days // (365 * 29.5)
        # For Uranus (~84 year cycle), estimate cycle count
        elif planet_id == swe.URANUS:
            cycles_passed = total_days // (365 * 84)
        else:
            cycles_passed = 0
            
        # If current longitude is less than initial, we might have crossed 0°
        raw_diff = current_longitude - initial_longitude
        if raw_diff < 0:
            raw_diff += 360.0  # Add one cycle
            
        # Total movement includes estimated full cycles
        total_movement = raw_diff + (cycles_passed * 360)
        
        # Debug output for certain days
        if days_ahead % 30 == 0:  # Print every 30 days
            print(f"Day {days_ahead}: Current long: {current_longitude:.4f}°, Movement: {total_movement:.4f}°")
        
        # Check if we've reached any of our target angles
        for angle in expanded_angles:
            if angle not in found_angles and abs(total_movement - angle) < 0.5:  # Within half a degree
                print(f"Found {angle}° at day {days_ahead} ({current_date})")
                
                # Store the actual date (not a dictionary)
                result_dates[angle] = current_date
                found_angles.add(angle)  # Mark this angle as found
        
        # Exit if we've found all expanded angles
        if len(found_angles) == len(expanded_angles):
            break
    
    # Report missing angles
    missing_angles = set(expanded_angles) - found_angles
    if missing_angles:
        print(f"Warning: Could not find these angles within {max_days} days: {sorted(missing_angles)}")
    
    return result_dates

def analyze_market_at_angular_increments(market_symbol, ref_date, planet_names, angles, max_days=365):
    """
    Analyze market behavior when selected planets reach specific angular increments.
    
    Args:
        market_symbol: Stock/index to analyze
        ref_date: Reference date (major high/low)
        planet_names: List of planets to track
        angles: List of angular increments to check
        
    Returns:
        List of results for each planet and angle
    """
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
        'North Node': swe.TRUE_NODE,
    }
    
    results = []
    
    # Get reference price
    ref_price = get_stock_price(market_symbol, ref_date)
    
    # For each planet, find dates at angular increments
    for planet_name in planet_names:
        if planet_name not in planet_map:
            print(f"Warning: Planet {planet_name} not recognized. Skipping.")
            continue
        
        planet_id = planet_map[planet_name]
        
        # Find dates for angular increments (with cycles)
        angle_dates = find_future_angular_positions(ref_date, planet_id, angles, max_days, cycles=3)
        
        # For each angle, analyze market
        for angle, angle_date in angle_dates.items():
            # Get price at angular increment date
            angle_price = get_stock_price(market_symbol, angle_date)
            
            # Calculate price change
            if ref_price is not None and angle_price is not None:
                price_change = ((angle_price - ref_price) / ref_price) * 100
            else:
                price_change = None
            
            # Calculate which cycle this angle belongs to
            cycle = int(angle // 360) + 1
            display_angle = angle % 360
            if display_angle == 0:
                display_angle = 360  # Show 360 instead of 0
                
            # Check if this is a turning point
            turning_point_type = check_if_turning_point(market_symbol, angle_date)
            
            # Add to results
            results.append({
                'Planet': planet_name,
                'Angle': angle,  # Store the full angle including cycles
                'Display_Angle': display_angle,  # For display purposes
                'Cycle': cycle,
                'Reference_Date': ref_date,
                'Angular_Date': angle_date,  # Now a date object, not a dictionary
                'Days_Between': (angle_date - ref_date).days,
                'Reference_Price': ref_price,
                'Angular_Price': angle_price,
                'Percent_Change': price_change,
                'Turning_Point_Type': turning_point_type
            })
    
    return results

def check_if_turning_point(symbol, date, window=5):
    """
    Check if a date is a local turning point in the market.
    Uses a simple algorithm checking if it's a high/low within a window of days.
    
    Args:
        symbol: Stock/index symbol
        date: Date to check
        window: Number of days to check before and after the date
        
    Returns:
        'High', 'Low', or None
    """
    try:
        # Format symbol for Yahoo Finance
        if symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
            yf_symbol = '^GSPC'
        elif symbol.upper() in ['DJI', 'DJIA', 'DOW']:
            yf_symbol = '^DJI'
        elif symbol.upper() in ['IXIC', 'NASDAQ', 'NDX']:
            yf_symbol = '^IXIC'
        else:
            yf_symbol = symbol
        
        # Get data around the date
        start_date = date - datetime.timedelta(days=window)
        end_date = date + datetime.timedelta(days=window)
        data = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty or len(data) < 3:
            return None
        
        # Find the index closest to our target date
        target_idx = None
        min_days_diff = float('inf')
        
        for i, idx in enumerate(data.index):
            idx_date = idx.date()
            days_diff = abs((idx_date - date).days)
            if days_diff < min_days_diff:
                min_days_diff = days_diff
                target_idx = i
        
        if target_idx is None or min_days_diff > window:
            return None
        
        # Check if it's a local high
        if target_idx > 0 and target_idx < len(data) - 1:
            price = float(data['Close'].iloc[target_idx])
            prices_before = data['Close'].iloc[:target_idx].values
            prices_after = data['Close'].iloc[target_idx+1:].values
            
            # Check if it's a local maximum
            if price > max(prices_before) and price > max(prices_after):
                return 'High'
            
            # Check if it's a local minimum
            if price < min(prices_before) and price < min(prices_after):
                return 'Low'
        
        return None
    
    except Exception as e:
        print(f"Error checking for turning point: {e}")
        return None

def analyze_market_at_angular_increments(market_symbol, ref_date, planet_names, angles, max_days=365):
    """
    Analyze market behavior when selected planets reach specific angular increments.
    
    Args:
        market_symbol: Stock/index to analyze
        ref_date: Reference date (major high/low)
        planet_names: List of planets to track
        angles: List of angular increments to check
        
    Returns:
        List of results for each planet and angle
    """
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
        'North Node': swe.TRUE_NODE,
    }
    
    results = []
    
    # Get reference price
    ref_price = get_stock_price(market_symbol, ref_date)
    
    # For each planet, find dates at angular increments
    for planet_name in planet_names:
        if planet_name not in planet_map:
            print(f"Warning: Planet {planet_name} not recognized. Skipping.")
            continue
        
        planet_id = planet_map[planet_name]
        
        # Find dates for angular increments
        angle_dates = find_future_angular_positions(ref_date, planet_id, angles, max_days)
        
        # For each angle, analyze market
        for angle, angle_date in angle_dates.items():
            # Get price at angular increment date
            angle_price = get_stock_price(market_symbol, angle_date)
            
            # Calculate price change
            if ref_price is not None and angle_price is not None:
                price_change = ((angle_price - ref_price) / ref_price) * 100
            else:
                price_change = None
            
            # Determine if this is a local high/low
            turning_point_type = check_if_turning_point(market_symbol, angle_date)
            
            # Add to results
            results.append({
                'Planet': planet_name,
                'Angle': angle,
                'Reference_Date': ref_date,
                'Angular_Date': angle_date,
                'Days_Between': (angle_date - ref_date).days,
                'Reference_Price': ref_price,
                'Angular_Price': angle_price,
                'Percent_Change': price_change,
                'Turning_Point_Type': turning_point_type
            })
    
    return results

def visualize_results(results, market_symbol, ref_date, ref_position):
    """Create a visualization of the market with angular increments marked."""
    if not results:
        print("No results to visualize.")
        return
    
    # Format symbol for Yahoo Finance
    if market_symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
        yf_symbol = '^GSPC'
    elif market_symbol.upper() in ['DJI', 'DJIA', 'DOW']:
        yf_symbol = '^DJI'
    elif market_symbol.upper() in ['IXIC', 'NASDAQ', 'NDX']:
        yf_symbol = '^IXIC'
    else:
        yf_symbol = market_symbol
    
    # Determine date range
    min_date = ref_date
    max_date = max(r['Angular_Date'] for r in results)
    buffer_days = (max_date - min_date).days * 0.1  # Add 10% buffer
    
    start_date = min_date - datetime.timedelta(days=int(buffer_days))
    end_date = max_date + datetime.timedelta(days=int(buffer_days))
    
    # Get market data
    data = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        print("No market data available for the specified date range.")
        return
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data['Close'], label=market_symbol)
    
    # Mark reference date
    ref_idx = data.index[data.index >= pd.Timestamp(ref_date)][0]
    ref_price = float(data.loc[ref_idx, 'Close'])
    plt.axvline(x=ref_idx, color='r', linestyle='--', label=f"Reference ({ref_position})")
    plt.plot(ref_idx, ref_price, 'ro', markersize=8)
    
    # Group results by planet
    planets = set(r['Planet'] for r in results)
    colors = plt.cm.tab10.colors  # Get a color cycle
    
    # Create legend handlers for planets with proper colors
    planet_patches = [mpatches.Patch(color=colors[i % len(colors)], label=planet) 
                     for i, planet in enumerate(planets)]
    
    for i, planet in enumerate(planets):
        planet_results = [r for r in results if r['Planet'] == planet]
        color = colors[i % len(colors)]
        
        for result in planet_results:
            angle_date = result['Angular_Date']
            # Use display angle (normalized to 0-360) with cycle information
            display_angle = result.get('Display_Angle', result['Angle'] % 360)
            cycle = result.get('Cycle', result['Angle'] // 360 + 1)
            
            # Skip if no price data is available for this date
            try:
                closest_idx = data.index[data.index >= pd.Timestamp(angle_date)][0]
                price = float(data.loc[closest_idx, 'Close'])
                
                # Mark on chart
                plt.axvline(x=closest_idx, color=color, alpha=0.5, linestyle=':')
                plt.plot(closest_idx, price, 'o', color=color, markersize=6)
                
                # Format date and price for the label
                date_str = angle_date.strftime('%Y-%m-%d')
                price_str = f"${price:.2f}"
                
                # Add annotation with price information
                cycle_info = f" (Cycle {cycle})" if cycle > 1 else ""
                plt.annotate(
                    f"{planet} +{display_angle}°{cycle_info}\n{date_str}\n{price_str}",
                    xy=(closest_idx, price),
                    xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=8,
                    color=color
                )
            except (IndexError, KeyError):
                print(f"Warning: No price data available for {angle_date}. Skipping annotation.")
    
    plt.title(f"Angular Increment Analysis for {market_symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add a proper legend with planet colors + reference line
    plt.legend(handles=[*planet_patches, plt.Line2D([0], [0], color='r', linestyle='--', label=f"Reference ({ref_position})")],
              loc='best')
    
    # Save the chart
    chart_filename = f"{market_symbol}_angular_analysis.png"
    plt.savefig(chart_filename)
    print(f"Chart saved as {chart_filename}")
    
    # Show the chart
    plt.show()

def display_results(results, market_symbol, ref_date, market_position):
    """Display the analysis results in a formatted table."""
    print("\n" + "="*110)
    print(f"ANGULAR INCREMENT ANALYSIS FOR {market_symbol}")
    print(f"Reference Date: {ref_date} - Market Position: {'High' if market_position == 'H' else 'Low'}")
    print("="*110)
    
    if not results:
        print("No results found. Try increasing the maximum days to look ahead.")
        return
    
    print(f"{'Planet':10} | {'Angle':6} | {'Cycle':5} | {'Date':12} | {'Days':6} | {'Price':10} | {'Change %':10} | {'Turning Point':12}")
    print("-"*110)
    
    for result in results:
        try:
            # Basic values with safe defaults
            planet = str(result.get('Planet', 'Unknown'))
            angle = result.get('Angle', 0)
            
            # Handle potentially problematic values
            if angle is None:
                angle_str = "N/A"
            else:
                try:
                    display_angle = angle % 360
                    angle_str = f"{display_angle:.1f}°"
                except:
                    angle_str = "Error°"
            
            # Handle cycle 
            try:
                cycle = int(result.get('Cycle', 1))
                cycle_str = f"{cycle:d}"
            except:
                cycle_str = "1"
            
            # Handle date
            date = result.get('Angular_Date')
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)
            
            # Handle days
            try:
                days = int(result.get('Days_Between', 0))
                days_str = f"{days:d}"
            except:
                days_str = "0"
            
            # Handle price
            price = result.get('Angular_Price')
            if price is not None:
                try:
                    price_str = f"${float(price):.2f}"
                except:
                    price_str = "N/A"
            else:
                price_str = "N/A"
            
            # Handle change percentage
            change = result.get('Percent_Change')
            if change is not None:
                try:
                    change_str = f"{float(change):.2f}%"
                except:
                    change_str = "N/A"
            else:
                change_str = "N/A"
            
            # Handle turning point
            turning = str(result.get('Turning_Point_Type', 'No'))
            
            # Print the row with fixed-width formatting
            print(f"{planet:10} | {angle_str:6} | {cycle_str:5} | {date_str:12} | {days_str:6} | {price_str:10} | {change_str:10} | {turning:12}")
            
        except Exception as e:
            # If anything fails, print a simpler version of the row
            print(f"Error displaying result: {str(e)}")
            try:
                print(f"{str(result.get('Planet', 'Unknown')):10} | ERROR - Check data format")
            except:
                print("DISPLAY ERROR")
    
    print("="*110)

def save_to_csv(results, market_symbol, ref_date):
    """Save analysis results to a CSV file."""
    if not results:
        print("No results to save.")
        return
    
    # Create a DataFrame
    df = pd.DataFrame(results)
    
    # Generate filename
    safe_symbol = "".join(c if c.isalnum() else "_" for c in market_symbol)
    filename = f"{safe_symbol}_angular_analysis_{ref_date}.csv"
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main():
    # Get user input
    print("Angular Increment Analysis Tool")
    print("-------------------------------")
    
    print("Enter reference date (major high/low) (YYYY-MM-DD):")
    ref_date_str = input()
    ref_date = datetime.datetime.strptime(ref_date_str, "%Y-%m-%d").date()
    
    print("Enter stock/index symbol:")
    market_symbol = input()
    
    print("Is this a high (H) or low (L)?")
    market_position = input().upper()

    # Define future date cutoff (e.g., 2 years from now)
    future_cutoff = datetime.datetime.now().date() + datetime.timedelta(days=730)  # 2 years
    
    # Custom angles for each planet based on their speed
    planet_angles = {
        'Jupiter': [15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0, 195.0, 210.0, 225.0, 240.0, 255.0, 270.0, 285.0, 300.0, 315.0, 330.0, 345.0, 360.0],
        'Saturn': [15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0],
        'Uranus': [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]  # Smaller increments for slow-moving Uranus
    }
    
    # Get planet selection
    print("Select planets to analyze (comma-separated, e.g., Jupiter,Saturn,Uranus):")
    planet_input = input()
    planet_names = [p.strip() for p in planet_input.split(',')]
    
    # Get the appropriate angles for selected planets
    angles_to_check = []
    for planet in planet_names:
        if planet in planet_angles:
            angles_to_check.extend(planet_angles[planet])
        else:
            # Use default angles for planets not in the dictionary
            print(f"Using default angles for {planet}")
            angles_to_check.extend([15.0, 30.0, 45.0, 60.0, 90.0, 180.0])
    
    # Remove duplicates
    angles_to_check = sorted(list(set(angles_to_check)))
    
    print(f"Analyzing the following angles: {angles_to_check}")
    
    # Maximum days to look ahead
    print("Maximum days to look ahead (default 1800):")
    days_str = input()
    max_days = int(days_str) if days_str.strip() else 1800  # ~5 years default
    
    # Run analysis with customized angles per planet
    all_results = []
    today = datetime.datetime.now().date()
    
    for planet in planet_names:
        # Get the planet-specific angles
        planet_specific_angles = planet_angles.get(planet, [15.0, 30.0, 45.0, 60.0, 90.0, 180.0])
        
        # Run analysis for this planet with its specific angles
        planet_results = analyze_market_at_angular_increments(
            market_symbol, 
            ref_date, 
            [planet],  # Just this one planet
            planet_specific_angles, 
            max_days
        )
        
        # Process each result before adding to all_results
        for result in planet_results:
            # Handle future dates properly
            if result['Angular_Date'] > today:
                # Keep the date but mark the financial data as unavailable
                result['Angular_Price'] = None
                result['Percent_Change'] = None
                result['Turning_Point_Type'] = 'Future'
            
            # Only include dates within our cutoff
            if result['Angular_Date'] <= future_cutoff:
                all_results.append(result)
    
    # Sort results by date
    all_results.sort(key=lambda x: x['Angular_Date'])
    
    # Display results
    display_results(all_results, market_symbol, ref_date, market_position)
    
    # Option to visualize results
    print("\nWould you like to visualize the results? (y/n)")
    if input().lower() == 'y':
        visualize_results(all_results, market_symbol, ref_date, market_position)
    
    # Option to save results
    print("\nWould you like to save the results to a CSV file? (y/n)")
    if input().lower() == 'y':
        save_to_csv(all_results, market_symbol, ref_date)

if __name__ == "__main__":
    main()