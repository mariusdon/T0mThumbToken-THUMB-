import os
from dotenv import load_dotenv
import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime, timedelta
from fredapi import Fred

# Load environment variables
load_dotenv()

# Check if FRED API key is set
fred_api_key = os.getenv('FRED_API_KEY')
print(f"FRED API Key found: {bool(fred_api_key)}")
if fred_api_key:
    print(f"Key starts with: {fred_api_key[:10]}...")

# Test FRED API
try:
    # Test with a simple series (10Y Treasury yield)
    end = datetime.today()
    start = end - timedelta(days=7)
    
    print("\nTesting FRED API with DGS10 (10Y Treasury yield)...")
    df = pdr.DataReader('DGS10', 'fred', start, end)
    print(f"Data shape: {df.shape}")
    print(f"Latest value: {df['DGS10'].dropna().iloc[-1]}")
    print("✅ FRED API is working!")
    
except Exception as e:
    print(f"❌ FRED API error: {e}")
    
    # Try alternative approach
    try:
        print("\nTrying alternative FRED approach...")
        fred = Fred(api_key=fred_api_key)
        value = fred.get_series_latest_release('DGS10')
        print(f"Alternative method - Latest 10Y yield: {value}")
        print("✅ Alternative FRED method is working!")
    except Exception as e2:
        print(f"❌ Alternative FRED method also failed: {e2}")

# Fetch latest unemployment rate
fred = Fred(api_key=os.getenv('FRED_API_KEY'))
print('FRED_API_KEY:', os.getenv('FRED_API_KEY'))
try:
    unrate_series = fred.get_series('UNRATE')
    latest_unrate = unrate_series.dropna().iloc[-1]
    print('Latest UNRATE:', latest_unrate)
except Exception as e:
    print('Error fetching UNRATE:', e) 