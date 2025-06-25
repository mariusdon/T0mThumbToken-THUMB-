import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
from fredapi import Fred
import yfinance as yf
import os
from dotenv import load_dotenv

load_dotenv()

def generate_ml_training_csv():
    """Generate comprehensive ML training data CSV with all features"""
    print("Generating ML training data CSV...")
    
    # Initialize FRED API
    FRED_API_KEY = os.getenv('FRED_API_KEY') or 'REPLACE_WITH_YOUR_FRED_API_KEY'
    fred = Fred(api_key=FRED_API_KEY)
    
    # Define date range (last 5 years of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # FRED series IDs for Treasury yields
    yield_series = {
        '3M': 'DGS3MO',
        '6M': 'DGS6MO', 
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',
        '30Y': 'DGS30',
    }
    
    # Macro series
    macro_series = {
        'CPI': 'CPIAUCSL',
        'UNEMP': 'UNRATE'
    }
    
    # Collect yield data
    yield_data = {}
    for maturity, series_id in yield_series.items():
        try:
            print(f"Fetching {maturity} yield data...")
            df = pdr.DataReader(series_id, 'fred', start_date, end_date)
            yield_data[maturity] = df[series_id]
        except Exception as e:
            print(f"Error fetching {maturity}: {e}")
            # Create synthetic data if FRED fails
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            base_yield = 4.0 + (maturity in ['3M', '6M']) * 0.5 + (maturity in ['1Y', '2Y']) * 0.3
            yield_data[maturity] = pd.Series([base_yield + np.random.normal(0, 0.1) for _ in dates], index=dates)
    
    # Collect macro data
    macro_data = {}
    for indicator, series_id in macro_series.items():
        try:
            print(f"Fetching {indicator} data...")
            df = pdr.DataReader(series_id, 'fred', start_date, end_date)
            macro_data[indicator] = df[series_id]
        except Exception as e:
            print(f"Error fetching {indicator}: {e}")
            # Create synthetic data if FRED fails
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            if indicator == 'CPI':
                base_value = 300.0
            else:  # UNEMP
                base_value = 4.0
            macro_data[indicator] = pd.Series([base_value + np.random.normal(0, 0.5) for _ in dates], index=dates)
    
    # Collect MOVE index data
    print("Fetching MOVE index data...")
    move = yf.Ticker("^MOVE")
    move_hist = move.history(start=start_date, end=end_date)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    if not move_hist.empty:
        move_data = move_hist['Close']
        print(f"Successfully fetched MOVE data with {len(move_data)} points")
        move_data = move_data.reindex(dates, method='ffill').fillna(method='bfill')
    else:
        print("MOVE data unavailable from Yahoo Finance. Trying to load from 'move_yahoo.csv'...")
        if not os.path.exists('move_yahoo.csv'):
            raise RuntimeError("MOVE data unavailable from Yahoo Finance. Please download it manually from Yahoo Finance (as 'move_yahoo.csv') and place it in the backend directory.")
        move_csv = pd.read_csv('move_yahoo.csv')
        # Try to find the right columns
        if 'Date' not in move_csv.columns or 'Close' not in move_csv.columns:
            raise RuntimeError("'move_yahoo.csv' must have 'Date' and 'Close' columns from Yahoo Finance export.")
        move_csv['Date'] = pd.to_datetime(move_csv['Date'])
        move_csv = move_csv.sort_values('Date')
        move_csv = move_csv.set_index('Date')
        # Reindex to match our date range, forward-fill missing values
        move_data = move_csv['Close'].reindex(dates, method='ffill').fillna(method='bfill')
        print(f"Loaded MOVE data from 'move_yahoo.csv' with {len(move_data)} points.")
    
    # Combine all data into a single DataFrame
    all_data = {}
    
    # Add yield data
    for maturity, series in yield_data.items():
        all_data[f'yield_{maturity}'] = series
    
    # Add macro data
    for indicator, series in macro_data.items():
        all_data[indicator] = series
    
    # Add MOVE data
    all_data['MOVE'] = move_data
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Forward fill missing values (common in financial data)
    df = df.fillna(method='ffill')
    
    # Backward fill any remaining NaN values
    df = df.fillna(method='bfill')
    
    # Add date column
    df['date'] = df.index
    
    # Reorder columns to put date first
    columns = ['date'] + [col for col in df.columns if col != 'date']
    df = df[columns]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Save to CSV
    csv_path = 'ml_training_data.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"ML training data saved to {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Show sample data
    print("\nSample data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    generate_ml_training_csv() 