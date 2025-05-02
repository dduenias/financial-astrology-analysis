'''Calculate planetary positions and aspects for any given date, 
with a focus on finding positions during major market highs/lows.

**Key Features**:
- Calculates precise planetary positions using Swiss Ephemeris
- Converts positions to zodiac signs and Human Design gates/lines
- Identifies aspects between planets (conjunctions, squares, etc.)
- Fetches closing prices from Yahoo Finance
- Saves results to CSV files for further analysis'''


import swisseph as swe
import datetime
import pandas as pd
import os
from math import fmod
import psycopg2
import yfinance as yf

def get_user_input():
    """Get date and stock/index name from user."""
    print("Enter date of major high/low (YYYY-MM-DD):")
    date_str = input()
    try:
        input_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return get_user_input()
    
    print("Enter stock/index/currency pair name:")
    stock_name = input()
    
    print("Is this a high (H) or low (L)?")
    market_position = input().upper()
    if market_position not in ["H", "L"]:
        print("Please enter H for high or L for low.")
        return get_user_input()
    
    return input_date, stock_name, market_position

def get_stock_price(symbol, date):
    """Get stock/index price for the given date using yfinance."""
    try:
        # Format stock symbol for Yahoo Finance
        if symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
            yf_symbol = '^GSPC'  # S&P 500
        elif symbol.upper() in ['DJI', 'DJIA', 'DOW']:
            yf_symbol = '^DJI'   # Dow Jones
        elif symbol.upper() in ['IXIC', 'NASDAQ', 'NDX']:
            yf_symbol = '^IXIC'  # NASDAQ
        elif symbol.upper() in ['RUT', 'RUSSELL', 'RUSSELL2000']:
            yf_symbol = '^RUT'   # Russell 2000
        else:
            yf_symbol = symbol
        
        # Get data for the specified date
        # Adding 5 days to handle weekends/holidays
        end_date = date + datetime.timedelta(days=5)
        data = yf.download(yf_symbol, start=date, end=end_date, progress=False)
        
        if data.empty:
            print(f"No price data found for {symbol} on or after {date}")
            return None
        
        # Get the first available date on or after the specified date
        first_available_date = data.index[0].date()
        if first_available_date != date:
            print(f"Price data not available for exact date. Using closest date: {first_available_date}")
        
        # Return the closing price - get the first value as a float
        close_price = float(data['Close'].iloc[0])
        return close_price
        
    except Exception as e:
        print(f"Error fetching stock price: {e}")
        return None

def initialize_ephemeris():
    """Initialize the Swiss Ephemeris with proper path."""
    # Let Swiss Ephemeris download files automatically:
    swe.set_ephe_path(None)

def get_julian_day(date):
    """Convert date to Julian day."""
    return swe.julday(date.year, date.month, date.day, 12.0)  # Noon UTC

def connect_to_database():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname='ephemeris_db',
            user='postgres',
            password='AnthonyH0pkins',
            host='localhost',
            port='5432'
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def get_human_design_gate(longitude):
    """Convert longitude to Human Design Gate and Line."""
    try:
        # Connect to the database
        conn = connect_to_database()
        if conn is None:
            return "N/A"
        
        cursor = conn.cursor()
        
        # Normalize longitude to 0-360 range
        norm_longitude = fmod(longitude, 360.0)
        if norm_longitude < 0:
            norm_longitude += 360.0
        
        # Query to find the correct gate and line
        query = """
        SELECT gate, line 
        FROM human_design.human_design_converter 
        WHERE longitude <= %s 
        ORDER BY longitude DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (norm_longitude,))
        result = cursor.fetchone()
        
        if result:
            gate, line = result
            human_design_position = f"{gate}.{line}"
        else:
            # If no result, try with 0 degrees as fallback
            cursor.execute(query, (360.0,))
            result = cursor.fetchone()
            if result:
                gate, line = result
                human_design_position = f"{gate}.{line}"
            else:
                human_design_position = "N/A"
        
        cursor.close()
        conn.close()
        
        return human_design_position
    except Exception as e:
        print(f"Error converting to Human Design: {e}")
        return "N/A"

def calculate_planetary_positions(date):
    """Calculate planetary positions for given date using Swiss Ephemeris."""
    # Initialize ephemeris
    initialize_ephemeris()
    
    # Get Julian day for the date
    jd = get_julian_day(date)
    
    # Define planets to track (Swiss Ephemeris planet constants)
    bodies = {
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
        'North Node': swe.TRUE_NODE  # Adding North Node (Mean Node)
    }
    
    # Calculate positions
    positions = {}
    for name, body_id in bodies.items():
        # Calculate geocentric tropical position (flag 0)
        flags = swe.FLG_SWIEPH
        result, status = swe.calc_ut(jd, body_id, flags)
        
        # Extract longitude from the result tuple - it's the first element
        longitude = result[0]
        
        # Get zodiac sign and position
        zodiac_sign = get_zodiac_sign(longitude)
        
        # Get Human Design gate and line
        human_design = get_human_design_gate(longitude)
        
        positions[name] = {
            'longitude_decimal': longitude,
            'zodiac_position': zodiac_sign,
            'human_design': human_design
        }
    
    # Calculate South Node (opposite to North Node)
    north_node_longitude = positions['North Node']['longitude_decimal']
    south_node_longitude = north_node_longitude + 180.0
    if south_node_longitude >= 360.0:
        south_node_longitude -= 360.0
    
    # Get zodiac sign and HD for South Node
    south_node_zodiac = get_zodiac_sign(south_node_longitude)
    south_node_hd = get_human_design_gate(south_node_longitude)
    
    # Add South Node to positions
    positions['South Node'] = {
        'longitude_decimal': south_node_longitude,
        'zodiac_position': south_node_zodiac,
        'human_design': south_node_hd
    }
    
    return positions

def get_zodiac_sign(longitude):
    """Convert longitude to zodiac sign and degrees."""
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
    
    # Calculate degrees within sign
    degrees = fmod(norm_longitude, 30)
    
    return f"{signs[sign_index]} {degrees:.2f}°"

def get_aspect_symbol(aspect_name):
    """Return the symbol for a given aspect."""
    symbols = {
        "Conjunction": "☌",
        "Sextile": "⚹",
        "Square": "□",
        "Trine": "△",
        "Opposition": "☍"
    }
    return symbols.get(aspect_name, "")

def calculate_aspects(positions, orb=2.0):
    """Calculate aspects between planets."""
    aspects = []
    aspect_types = {
        0: "Conjunction",
        60: "Sextile",
        90: "Square",
        120: "Trine",
        180: "Opposition"
    }
    
    # Get planet pairs
    planets = list(positions.keys())
    
    for i in range(len(planets)):
        for j in range(i+1, len(planets)):
            planet1 = planets[i]
            planet2 = planets[j]

            # Skip the North Node - South Node aspect since it's always opposition
            if (planet1 == 'North Node' and planet2 == 'South Node') or \
               (planet1 == 'South Node' and planet2 == 'North Node'):
                continue
            
            long1 = positions[planet1]['longitude_decimal']
            long2 = positions[planet2]['longitude_decimal']
            
            # Calculate angular difference
            diff = abs(long1 - long2)
            if diff > 180:
                diff = 360 - diff
            
            # Check for aspects
            for angle, aspect_name in aspect_types.items():
                if abs(diff - angle) <= orb:
                    # Get Human Design positions for each planet
                    hd1 = positions[planet1]['human_design']
                    hd2 = positions[planet2]['human_design']
                    aspect_symbol = get_aspect_symbol(aspect_name)
                    
                    # Create HD aspect string
                    hd_aspect = f"{hd1} {aspect_symbol} {hd2}"
                    
                    aspects.append({
                        'planet1': planet1,
                        'planet2': planet2,
                        'aspect': aspect_name,
                        'orb': abs(diff - angle),
                        'hd_aspect': hd_aspect
                    })
    
    return aspects

def display_results(date, stock_name, market_position, positions, aspects, price):
    """Display planetary positions and aspects."""
    print("\n" + "="*75)
    print(f"PLANETARY ANALYSIS FOR {stock_name}")
    print(f"Date: {date} - Market Position: {'High' if market_position == 'H' else 'Low'}")
    if price is not None:
        print(f"Closing Price: ${price:.2f}")
    else:
        print("Closing Price: Not available")
    print("="*75)
    
    print("\nPLANETARY POSITIONS:")
    print("-"*75)
    print(f"{'Planet':10} | {'Zodiac Position':20} | {'Longitude':15} | {'HD Gate.Line':10}")
    print("-"*75)
    
    # Define the order of planets for display
    display_order = [
        'Sun',
        'Moon',
        'North Node',
        'South Node',
        'Mercury',
        'Venus',
        'Mars',
        'Jupiter',
        'Saturn',
        'Uranus',
        'Neptune',
        'Pluto'
    ]
    
    # Display planets in the specified order
    for planet in display_order:
        if planet in positions:
            data = positions[planet]
            print(f"{planet:10} | {data['zodiac_position']:20} | {data['longitude_decimal']:15.4f}° | {data['human_design']:10}")
    
    print("\nPLANETARY ASPECTS:")
    print("-"*75)
    if not aspects:
        print("No significant aspects found within the defined orb.")
    else:
        print(f"{'Planet 1':10} | {'Aspect':12} | {'Planet 2':10} | {'Orb':8} | {'HD Aspect':25}")
        print("-"*75)
        for aspect in aspects:
            print(f"{aspect['planet1']:10} | {aspect['aspect']:12} | {aspect['planet2']:10} | {aspect['orb']:8.2f}° | {aspect['hd_aspect']:25}")
    
    print("="*75)

def save_to_csv(date, stock_name, market_position, positions, aspects, price):
    """Save results to CSV files with append functionality by stock/index."""
    # Create positions dataframe
    positions_data = []
    for planet, data in positions.items():
        positions_data.append({
            'Date': date,
            'Market_Position': 'High' if market_position == 'H' else 'Low',
            'Price': price if price is not None else 'N/A',
            'Planet': planet,
            'Zodiac_Position': data['zodiac_position'],
            'Longitude_Decimal': data['longitude_decimal'],
            'Human_Design_Gate_Line': data['human_design']
        })
    
    positions_df = pd.DataFrame(positions_data)
    
    # Create filenames based on stock name
    # Replace spaces and special characters to make valid filenames
    safe_stock_name = "".join(c if c.isalnum() else "_" for c in stock_name)
    
    # File names for positions and aspects
    positions_file = f"{safe_stock_name}_planetary_positions.csv"
    aspects_file = f"{safe_stock_name}_planetary_aspects.csv"
    
    # Check if files exist to determine whether to write headers
    positions_exist = os.path.isfile(positions_file)
    
    # Append to positions file
    positions_df.to_csv(positions_file, mode='a', header=not positions_exist, index=False)
    
    # Create and append aspects dataframe if aspects exist
    if aspects:
        aspects_data = []
        for aspect in aspects:
            aspects_data.append({
                'Date': date,
                'Market_Position': 'High' if market_position == 'H' else 'Low',
                'Price': price if price is not None else 'N/A',
                'Planet1': aspect['planet1'],
                'Aspect': aspect['aspect'],
                'Planet2': aspect['planet2'],
                'Orb': aspect['orb'],
                'HD_Aspect': aspect['hd_aspect']
            })
        
        aspects_df = pd.DataFrame(aspects_data)
        
        # Check if aspects file exists
        aspects_exist = os.path.isfile(aspects_file)
        
        # Append to aspects file
        aspects_df.to_csv(aspects_file, mode='a', header=not aspects_exist, index=False)
    
    print(f"\nResults appended to CSV files for {stock_name}:")
    print(f"- Positions: {positions_file}")
    if aspects:
        print(f"- Aspects: {aspects_file}")

def main():
    """Main function to run the program."""
    print("Financial Astrology Analysis Tool")
    print("--------------------------------")
    
    date, stock_name, market_position = get_user_input()
    
    # Get stock price for the date
    price = get_stock_price(stock_name, date)
    
    # Calculate positions
    positions = calculate_planetary_positions(date)
    
    # Calculate aspects
    aspects = calculate_aspects(positions)
    
    # Display results
    display_results(date, stock_name, market_position, positions, aspects, price)
    
    # Save results to CSV
    print("\nWould you like to save the results to CSV files? (y/n)")
    if input().lower() == 'y':
        save_to_csv(date, stock_name, market_position, positions, aspects, price)
    
    # Ask if user wants to analyze another date
    print("\nWould you like to analyze another date? (y/n)")
    if input().lower() == 'y':
        main()

if __name__ == "__main__":
    main()