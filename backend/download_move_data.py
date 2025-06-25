import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_move_data():
    print("Downloading MOVE data from Yahoo Finance...")
    
    # Define date range (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    try:
        # Download MOVE data
        move = yf.Ticker("^MOVE")
        hist = move.history(start=start_date, end=end_date)
        
        if not hist.empty:
            print(f"Successfully downloaded {len(hist)} rows of MOVE data")
            print("Sample data:")
            print(hist.head())
            
            # Save to CSV
            hist.to_csv('move_yahoo.csv')
            print("Saved to move_yahoo.csv")
            
            # Show the columns
            print(f"Columns: {list(hist.columns)}")
            
            return True
        else:
            print("No data received from Yahoo Finance")
            return False
            
    except Exception as e:
        print(f"Error downloading MOVE data: {e}")
        return False

if __name__ == "__main__":
    download_move_data() 