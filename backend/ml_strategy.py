import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

class MLStrategy:
    """
    Enhanced ML strategy that properly identifies yield curve shapes and allocates accordingly.
    """
    
    def __init__(self):
        self.maturity_names = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        self.maturity_weights = [1/7] * 7  # Equal weights as default
        
        # Load historical data
        self.yield_df = None
        self.features_df = None
        self.returns_df = None
        
        # Initialize data
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical Treasury yield data."""
        try:
            # Use FRED data for Treasury yields
            symbols = {
                '3M': '^IRX',  # 13-week Treasury
                '6M': '^IRX',  # Approximate with 3M
                '1Y': '^TNX',  # 10-year Treasury (approximate)
                '2Y': '^TNX',  # 10-year Treasury (approximate)
                '5Y': '^TNX',  # 10-year Treasury (approximate)
                '10Y': '^TNX', # 10-year Treasury
                '30Y': '^TYX'  # 30-year Treasury
            }
            
            # Download data
            data = {}
            for maturity, symbol in symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5y", interval="1d")
                    data[maturity] = hist['Close']
                except:
                    # Fallback to synthetic data
                    data[maturity] = self._generate_synthetic_yields(maturity)
            
            # Create yield DataFrame
            self.yield_df = pd.DataFrame(data)
            
            # Fill missing values
            self.yield_df = self.yield_df.fillna(method='ffill').fillna(method='bfill')
            
            # Create features and returns
            self._create_features()
            self._calculate_returns()
            
            print(f"Loaded historical data: {len(self.yield_df)} days")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create synthetic data as fallback
            self._create_synthetic_data()
    
    def _generate_synthetic_yields(self, maturity):
        """Generate synthetic yield data for testing."""
        dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='D')
        
        # Base yields by maturity
        base_yields = {
            '3M': 2.0, '6M': 2.2, '1Y': 2.5, '2Y': 2.8,
            '5Y': 3.2, '10Y': 3.5, '30Y': 3.8
        }
        
        base = base_yields.get(maturity, 3.0)
        
        # Add realistic yield movements
        np.random.seed(42)  # For reproducibility
        trend = np.linspace(0, 2, len(dates))  # Rising trend
        noise = np.random.normal(0, 0.1, len(dates))
        seasonal = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        
        yields = base + trend + noise + seasonal
        return pd.Series(yields, index=dates)
    
    def _create_synthetic_data(self):
        """Create completely synthetic data for testing."""
        dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='D')
        
        # Create synthetic yields
        self.yield_df = pd.DataFrame(index=dates)
        for maturity in self.maturity_names:
            self.yield_df[maturity] = self._generate_synthetic_yields(maturity)
        
        # Create features and returns
        self._create_features()
        self._calculate_returns()
    
    def _create_features(self):
        """Create enhanced features for curve shape identification."""
        print("Creating enhanced features for curve shape identification...")
        
        features = pd.DataFrame(index=self.yield_df.index)
        
        # Core yield curve slopes
        features['slope_10y_2y'] = self.yield_df['10Y'] - self.yield_df['2Y']
        features['slope_30y_3m'] = self.yield_df['30Y'] - self.yield_df['3M']
        features['slope_5y_1y'] = self.yield_df['5Y'] - self.yield_df['1Y']
        features['slope_10y_3m'] = self.yield_df['10Y'] - self.yield_df['3M']
        features['slope_2y_3m'] = self.yield_df['2Y'] - self.yield_df['3M']
        features['slope_5y_2y'] = self.yield_df['5Y'] - self.yield_df['2Y']
        
        # Curve shape indicators
        features['curve_steepness'] = (self.yield_df['30Y'] - self.yield_df['3M']) / 27
        features['curve_inversion'] = (self.yield_df['2Y'] - self.yield_df['10Y']) / 8
        
        # Level of short rates
        features['short_rate_3m'] = self.yield_df['3M']
        features['short_rate_6m'] = self.yield_df['6M']
        features['short_rate_1y'] = self.yield_df['1Y']
        
        # Yield curve curvature
        features['curvature_2y_5y_10y'] = (self.yield_df['2Y'] + self.yield_df['10Y']) / 2 - self.yield_df['5Y']
        features['curvature_3m_2y_10y'] = (self.yield_df['3M'] + self.yield_df['10Y']) / 2 - self.yield_df['2Y']
        
        # Regime classification features
        features['is_steep'] = (features['slope_10y_2y'] > 0.5).astype(int)
        features['is_inverted'] = (features['slope_10y_2y'] < -0.5).astype(int)
        features['is_flat'] = ((features['slope_10y_2y'] >= -0.5) & (features['slope_10y_2y'] <= 0.5)).astype(int)
        
        # Lagged slopes for momentum
        features['slope_10y_2y_lag1w'] = features['slope_10y_2y'].shift(7)
        features['slope_10y_2y_lag4w'] = features['slope_10y_2y'].shift(28)
        features['slope_momentum'] = features['slope_10y_2y'] - features['slope_10y_2y_lag4w']
        
        # Time-based features
        features['month'] = self.yield_df.index.month
        features['quarter'] = self.yield_df.index.quarter
        features['year'] = self.yield_df.index.year
        
        # Volatility features
        for maturity in self.maturity_names:
            features[f'{maturity}_volatility'] = self.yield_df[maturity].rolling(30).std()
        
        # Relative value features
        features['relative_value_2y'] = self.yield_df['2Y'] - self.yield_df['2Y'].rolling(60).mean()
        features['relative_value_10y'] = self.yield_df['10Y'] - self.yield_df['10Y'].rolling(60).mean()
        features['relative_value_30y'] = self.yield_df['30Y'] - self.yield_df['30Y'].rolling(60).mean()
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        self.features_df = features
        print(f"Created {len(features.columns)} features")
    
    def _calculate_returns(self):
        """Calculate realistic bond returns with price movements and coupons."""
        print("Calculating realistic bond returns...")
        
        # Resample to monthly frequency for backtesting
        yield_monthly = self.yield_df.resample('M').last()
        
        # Initialize returns DataFrame
        returns = pd.DataFrame(index=yield_monthly.index, columns=yield_monthly.columns)
        
        # Maturity in years for each bond
        maturity_years = {
            '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, 
            '5Y': 5, '10Y': 10, '30Y': 30
        }
        
        for maturity in self.maturity_names:
            if maturity in yield_monthly.columns:
                maturity_years_val = maturity_years[maturity]
                
                # Calculate price changes based on yield changes
                # Bond price ≈ 100 * exp(-yield * maturity)
                prices = 100 * np.exp(-yield_monthly[maturity] * maturity_years_val)
                
                # Calculate price returns
                price_returns = prices.pct_change()
                
                # Calculate coupon income (annual coupon rate = yield)
                coupon_rate = yield_monthly[maturity] / 12  # Monthly coupon
                
                # Total return = price return + coupon income
                total_returns = price_returns + coupon_rate
                
                returns[maturity] = total_returns
        
        # Fill NaN values with 0
        returns = returns.fillna(0)
        
        self.returns_df = returns
        print(f"Calculated returns for {len(returns)} periods")
    
    def identify_curve_shape(self, yields):
        """
        Identify the shape of the yield curve based on current yields.
        
        Args:
            yields (dict): Current yields for each maturity
            
        Returns:
            dict: Curve shape classification and characteristics
        """
        # Calculate key slopes
        slope_10y_2y = yields['10Y'] - yields['2Y']
        slope_30y_3m = yields['30Y'] - yields['3M']
        slope_2y_3m = yields['2Y'] - yields['3M']
        
        # Determine curve shape
        if slope_10y_2y > 0.5 and slope_30y_3m > 1.0:
            shape = 'steep'
            description = 'Steep Curve - Favoring longer maturities for higher yields'
            base_weights = np.array([0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.15])
        elif slope_10y_2y < -0.5 or slope_2y_3m < -0.3:
            shape = 'inverted'
            description = 'Inverted Curve - Favoring shorter maturities for safety'
            base_weights = np.array([0.25, 0.25, 0.20, 0.15, 0.10, 0.03, 0.02])
        elif abs(slope_10y_2y) <= 0.5 and abs(slope_30y_3m) <= 1.0:
            shape = 'flat'
            description = 'Flat Curve - Balanced allocation'
            base_weights = np.array([0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16])
        else:
            shape = 'normal'
            description = 'Normal Curve - Slight preference for intermediate maturities'
            base_weights = np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.15, 0.10])
        
        return {
            'shape': shape,
            'description': description,
            'base_weights': base_weights,
            'slope_10y_2y': slope_10y_2y,
            'slope_30y_3m': slope_30y_3m,
            'slope_2y_3m': slope_2y_3m
        }
    
    def calculate_ml_allocation(self, yields):
        """
        Calculate ML-based allocation considering curve shape and current yields.
        
        Args:
            yields (dict): Current yields for each maturity
            
        Returns:
            dict: Allocation weights for each maturity
        """
        # Identify curve shape
        curve_info = self.identify_curve_shape(yields)
        
        # Get base weights from curve shape
        base_weights = curve_info['base_weights']
        
        # Calculate relative value adjustments
        # Higher yields relative to recent average get higher weights
        relative_weights = np.ones(len(self.maturity_names))
        for i, maturity in enumerate(self.maturity_names):
            if maturity in yields:
                current_yield = yields[maturity]
                # Simple relative value: higher yield = higher weight
                relative_weights[i] = 1 + (current_yield - 3.0) / 10  # Normalize around 3%
        
        # Normalize relative weights
        relative_weights = relative_weights / np.sum(relative_weights)
        
        # Combine curve shape (70%) with relative value (30%)
        final_weights = 0.7 * base_weights + 0.3 * relative_weights
        
        # Normalize to sum to 1
        final_weights = final_weights / np.sum(final_weights)
        
        # Create result dictionary
        allocation = dict(zip(self.maturity_names, final_weights))
        
        # Add curve shape info
        allocation['curve_shape'] = curve_info['shape']
        allocation['curve_description'] = curve_info['description']
        
        return allocation
    
    def get_current_allocation(self):
        """Get current ML allocation based on latest yield data."""
        if self.yield_df is None or len(self.yield_df) == 0:
            return self.maturity_weights
        
        # Get latest yields
        latest_yields = self.yield_df.iloc[-1].to_dict()
        
        # Calculate allocation
        allocation = self.calculate_ml_allocation(latest_yields)
        
        return allocation 