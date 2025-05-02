import yfinance as yf
import datetime
import time
import sys
import requests

def print_separator():
    print("\n" + "="*50 + "\n")

# 1. Print version information
print("System Information:")
print(f"Python version: {sys.version}")
try:
    print(f"yfinance version: {yf.__version__}")
except:
    print("Unable to determine yfinance version")
print_separator()

# 2. Test basic internet connectivity
print("Testing internet connectivity...")
try:
    response = requests.get("https://www.google.com", timeout=5)
    print(f"Internet connection test: Success (status code {response.status_code})")
except Exception as e:
    print(f"Internet connection test failed: {e}")
print_separator()

# 3. Test simple ticker download with no parameters
print("Test #1: Basic download with period parameter")
try:
    print("Downloading AAPL data for the past month...")
    data = yf.download('AAPL', period='1mo')
    print(f"Success! Downloaded {len(data)} rows of data")
    if not data.empty:
        print("\nSample data (first 3 rows):")
        print(data.head(3))
except Exception as e:
    print(f"Failed: {e}")
print_separator()

# 4. Test with explicit past date range
print("Test #2: Download with explicit past date range")
try:
    start = datetime.datetime(2023, 1, 1)
    end = datetime.datetime(2023, 12, 31)
    print(f"Downloading AAPL data from {start.date()} to {end.date()}...")
    data = yf.download('AAPL', start=start, end=end)
    print(f"Success! Downloaded {len(data)} rows of data")
    if not data.empty:
        print("\nDate range in downloaded data:")
        print(f"First date: {data.index.min().date()}")
        print(f"Last date: {data.index.max().date()}")
except Exception as e:
    print(f"Failed: {e}")
print_separator()

# 5. Test Ticker method instead of download
print("Test #3: Using Ticker.history() method")
try:
    print("Getting AAPL data using Ticker.history()...")
    ticker = yf.Ticker('AAPL')
    data = ticker.history(period="1mo")
    print(f"Success! Downloaded {len(data)} rows of data")
    if not data.empty:
        print("\nSample data (first 3 rows):")
        print(data.head(3))
except Exception as e:
    print(f"Failed: {e}")
print_separator()

# 6. Test index download (S&P 500)
print("Test #4: Downloading market index (S&P 500)")
try:
    print("Downloading ^GSPC data (S&P 500)...")
    data = yf.download('^GSPC', period='1mo')
    print(f"Success! Downloaded {len(data)} rows of data")
    if not data.empty:
        print("\nSample data (first 3 rows):")
        print(data.head(3))
except Exception as e:
    print(f"Failed: {e}")
print_separator()

# 7. Test with alternative GOLD symbol formats
print("Test #5: Testing different GOLD symbol formats")
gold_symbols = ['GOLD', 'GC=F', 'GLD']
for symbol in gold_symbols:
    try:
        print(f"Trying to download {symbol} data...")
        data = yf.download(symbol, period='1mo')
        print(f"Success for {symbol}! Downloaded {len(data)} rows of data")
        if not data.empty:
            print(f"First date: {data.index.min().date()}, Last date: {data.index.max().date()}")
    except Exception as e:
        print(f"Failed for {symbol}: {e}")
    time.sleep(1)  # Brief pause to avoid rate limiting
print_separator()

# 8. Test with different NASDAQ (tech stocks) index symbols
print("Test #6: Testing NASDAQ index symbols")
nasdaq_symbols = ['^IXIC', 'QQQ', 'NDAQ']
for symbol in nasdaq_symbols:
    try:
        print(f"Trying to download {symbol} data...")
        data = yf.download(symbol, period='1mo')
        print(f"Success for {symbol}! Downloaded {len(data)} rows of data")
        if not data.empty:
            print(f"First date: {data.index.min().date()}, Last date: {data.index.max().date()}")
    except Exception as e:
        print(f"Failed for {symbol}: {e}")
    time.sleep(1)  # Brief pause to avoid rate limiting
print_separator()

print("All tests completed. Check the results above to diagnose any issues with yfinance.")