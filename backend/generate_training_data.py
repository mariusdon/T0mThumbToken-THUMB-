import pandas as pd
import numpy as np
from allocation_rules import compute_regime_allocation
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf

# Download historical data from FRED
def download_fred_series(series_id, start, end):
    try:
        df = pdr.DataReader(series_id, 'fred', start, end)
        df = df.resample('M').last()
        return df
    except Exception as e:
        print(f"Error downloading {series_id}: {e}")
        return pd.DataFrame()

def load_real_historical_data():
    start = '2010-01-01'
    end = datetime.today().strftime('%Y-%m-%d')
    # Treasury yields
    series = {
        '3M': 'DGS3MO',
        '6M': 'DGS6MO',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',
        '30Y': 'DGS30',
    }
    yield_dfs = [download_fred_series(code, start, end).rename(columns={code: mat}) for mat, code in series.items()]
    yields_df = pd.concat(yield_dfs, axis=1)
    # Macro data
    cpi = download_fred_series('CPIAUCSL', start, end).rename(columns={'CPIAUCSL': 'CPI'})
    unemp = download_fred_series('UNRATE', start, end).rename(columns={'UNRATE': 'UNEMP'})
    # MOVE index from yfinance
    move = yf.Ticker("^MOVE")
    move_hist = move.history(start=start, end=end, interval="1mo")
    if not move_hist.empty and 'Close' in move_hist.columns:
        if hasattr(move_hist.index, 'tz') and move_hist.index.tz is not None:
            move_hist.index = move_hist.index.tz_localize(None)
        move_df = move_hist[['Close']].rename(columns={'Close': 'MOVE'})
        move_df['MOVE'] = move_df['MOVE'].interpolate(method='linear', limit_direction='both')
    else:
        # Fill MOVE with a constant (e.g., 100)
        move_df = pd.DataFrame(index=yields_df.index, columns=['MOVE'])
        move_df['MOVE'] = 100.0
    # Merge all (do not drop rows if MOVE is missing)
    df = yields_df.join([cpi, unemp], how='inner')
    df = df.join(move_df, how='left')
    df['MOVE'] = df['MOVE'].fillna(100.0)
    df = df.dropna()
    df = df.reset_index().rename(columns={'index': 'date'})
    return df

def generate_training_data():
    df = load_real_historical_data()
    features = []
    targets = []
    for _, row in df.iterrows():
        yields = [row['3M'], row['6M'], row['1Y'], row['2Y'], row['5Y'], row['10Y'], row['30Y']]
        macro = [row['CPI'], row['UNEMP'], row['MOVE']]
        alloc, regime = compute_regime_allocation(yields)
        features.append(yields + macro)
        targets.append(alloc)
    features = np.array(features)
    targets = np.array(targets)
    # Save to disk
    np.savez('ml_training_data.npz', features=features, targets=targets)
    print(f"Saved training data: features shape {features.shape}, targets shape {targets.shape}")

if __name__ == "__main__":
    generate_training_data() 