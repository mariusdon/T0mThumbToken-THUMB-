import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TreasuryDataLoader:
    """
    Loads and processes historical U.S. Treasury yield data from FRED.
    Calculates total returns for each maturity.
    """
    
    def __init__(self):
        # FRED series IDs for Treasury yields
        self.yield_series = {
            '3M': 'DGS3MO',
            '6M': 'DGS6MO', 
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '5Y': 'DGS5',
            '10Y': 'DGS10',
            '30Y': 'DGS30'
        }
        
        # Additional economic indicators
        self.economic_series = {
            'CPI_YOY': 'CPIAUCSL_PC1',  # CPI YoY change
            'UNRATE': 'UNRATE',         # Unemployment rate
            'FEDFUNDS': 'FEDFUNDS'      # Federal funds rate
        }
        
        self.maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        
    def fetch_yield_data(self, start_date='2015-01-01', end_date=None):
        """
        Fetch historical Treasury yield data from FRED.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            pd.DataFrame: Daily yield data with columns for each maturity
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching Treasury yield data from {start_date} to {end_date}...")
        
        # Fetch yield data
        yield_data = {}
        for maturity, series_id in self.yield_series.items():
            try:
                df = pdr.DataReader(series_id, 'fred', start_date, end_date)
                yield_data[maturity] = df[series_id]
                print(f"✓ Loaded {maturity} yield data: {len(df)} observations")
            except Exception as e:
                print(f"✗ Error loading {maturity} data: {e}")
                yield_data[maturity] = pd.Series(dtype=float)
        
        # Combine into DataFrame
        yield_df = pd.DataFrame(yield_data)
        yield_df = yield_df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Total yield data shape: {yield_df.shape}")
        return yield_df
    
    def fetch_economic_data(self, start_date='2015-01-01', end_date=None):
        """
        Fetch additional economic indicators from FRED.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            pd.DataFrame: Economic data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching economic data from {start_date} to {end_date}...")
        
        economic_data = {}
        for indicator, series_id in self.economic_series.items():
            try:
                df = pdr.DataReader(series_id, 'fred', start_date, end_date)
                economic_data[indicator] = df[series_id]
                print(f"✓ Loaded {indicator} data: {len(df)} observations")
            except Exception as e:
                print(f"✗ Error loading {indicator} data: {e}")
                economic_data[indicator] = pd.Series(dtype=float)
        
        economic_df = pd.DataFrame(economic_data)
        economic_df = economic_df.fillna(method='ffill').fillna(method='bfill')
        
        return economic_df
    
    def calculate_total_returns(self, yield_df, frequency='M'):
        """
        Calculate realistic rolling total returns for each maturity, including price change and coupon.
        Args:
            yield_df (pd.DataFrame): Daily yield data
            frequency (str): Rebalancing frequency ('D', 'W', 'M')
        Returns:
            pd.DataFrame: Total return data for each maturity
        """
        print(f"Calculating realistic rolling total returns with {frequency} frequency...")
        freq_map = {'M': 12, 'W': 52, 'D': 252}
        periods_per_year = freq_map.get(frequency, 12)
        
        # Resample to desired frequency
        yield_resampled = yield_df.resample(frequency).last()
        
        total_returns = pd.DataFrame(index=yield_resampled.index, columns=yield_resampled.columns)
        
        # Maturity in years for each bond
        maturity_map = {'3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}
        
        for maturity in yield_resampled.columns:
            y = yield_resampled[maturity].values / 100  # Convert to decimal
            n = len(y)
            mat = maturity_map.get(maturity, 5)
            returns = np.zeros(n)
            for i in range(n-1):
                y0 = y[i]
                y1 = y[i+1]
                # Price at start (par)
                p0 = 100
                # Price at end (approximate, using bond price formula for bullet bond, 1 period shorter)
                # For simplicity, assume coupon = y0, price at end = price if sold after 1 period
                # Price at end: present value of remaining cash flows at new yield
                periods_left = int(mat * periods_per_year) - 1
                if periods_left < 1:
                    periods_left = 1
                # Coupon payment for the period
                coupon = 100 * y0 / periods_per_year
                # Price at end (dirty price, ignoring accrued)
                if periods_left == 1:
                    p1 = 100  # At maturity, price = par
                else:
                    # Present value of remaining coupons + principal
                    cfs = np.ones(periods_left-1) * 100 * y0 / periods_per_year
                    cfs = np.append(cfs, 100 + 100 * y0 / periods_per_year)  # Last payment includes principal
                    discount_factors = 1 / (1 + y1 / periods_per_year) ** np.arange(1, periods_left+1)
                    p1 = np.sum(cfs * discount_factors)
                # Total return for the period
                total_return = (p1 - p0 + coupon) / p0
                returns[i] = total_return
            # Last period: no return (or repeat previous)
            returns[-1] = 0
            total_returns[maturity] = returns
        
        print(f"Total return data shape: {total_returns.shape}")
        print(f"Sample total returns:")
        print(total_returns.head())
        print(f"Average annual returns by maturity:")
        for maturity in total_returns.columns:
            annual_return = total_returns[maturity].mean() * periods_per_year
            print(f"  {maturity}: {annual_return:.2%}")
        
        return total_returns
    
    def create_features(self, yield_df, economic_df=None):
        """
        Create feature set for ML model.
        
        Args:
            yield_df (pd.DataFrame): Daily yield data
            economic_df (pd.DataFrame): Economic data (optional)
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        print("Creating feature set...")
        
        features = pd.DataFrame(index=yield_df.index)
        
        # Yield curve slopes
        features['slope_10y_2y'] = yield_df['10Y'] - yield_df['2Y']
        features['slope_30y_3m'] = yield_df['30Y'] - yield_df['3M']
        features['slope_5y_1y'] = yield_df['5Y'] - yield_df['1Y']
        features['slope_10y_3m'] = yield_df['10Y'] - yield_df['3M']
        
        # Level of short rates
        features['short_rate_3m'] = yield_df['3M']
        features['short_rate_6m'] = yield_df['6M']
        
        # Yield curve curvature
        features['curvature'] = (yield_df['2Y'] + yield_df['10Y']) / 2 - yield_df['5Y']
        
        # Lagged slopes (1-week, 4-week)
        features['slope_10y_2y_lag1w'] = features['slope_10y_2y'].shift(7)
        features['slope_10y_2y_lag4w'] = features['slope_10y_2y'].shift(28)
        
        # Time-based features
        features['month'] = yield_df.index.month
        features['quarter'] = yield_df.index.quarter
        features['year'] = yield_df.index.year
        
        # Volatility features (rolling standard deviation)
        for maturity in self.maturities:
            features[f'{maturity}_volatility'] = yield_df[maturity].rolling(30).std()
        
        # Add economic indicators if available
        if economic_df is not None:
            # Align economic data with yield data
            economic_aligned = economic_df.reindex(yield_df.index, method='ffill')
            features = pd.concat([features, economic_aligned], axis=1)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        print(f"Feature matrix shape: {features.shape}")
        return features
    
    def prepare_training_data(self, start_date='2015-01-01', end_date=None, frequency='M'):
        """
        Prepare complete training dataset.
        
        Args:
            start_date (str): Start date for data
            end_date (str): End date for data
            frequency (str): Rebalancing frequency
            
        Returns:
            tuple: (features_df, returns_df, yield_df)
        """
        print("Preparing complete training dataset...")
        
        # Fetch data
        yield_df = self.fetch_yield_data(start_date, end_date)
        economic_df = self.fetch_economic_data(start_date, end_date)
        
        # Calculate returns
        returns_df = self.calculate_total_returns(yield_df, frequency)
        
        # Create enhanced features using the model's feature creation method
        from model import TreasuryAllocationModel
        temp_model = TreasuryAllocationModel()
        features_df = temp_model.create_features(yield_df, economic_df)
        
        # Align all data
        common_index = features_df.index.intersection(returns_df.index)
        features_df = features_df.loc[common_index]
        returns_df = returns_df.loc[common_index]
        yield_df = yield_df.loc[common_index]
        
        print(f"Final aligned dataset shapes:")
        print(f"  Features: {features_df.shape}")
        print(f"  Returns: {returns_df.shape}")
        print(f"  Yields: {yield_df.shape}")
        
        return features_df, returns_df, yield_df

if __name__ == "__main__":
    # Test the data loader
    loader = TreasuryDataLoader()
    features, returns, yields = loader.prepare_training_data()
    
    print("\nSample features:")
    print(features.head())
    
    print("\nSample returns:")
    print(returns.head())
    
    print("\nSample yields:")
    print(yields.head()) 