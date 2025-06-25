import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class SophisticatedTreasuryAllocator:
    """
    Sophisticated Treasury allocation model that incorporates:
    - Coupon rates and yield-to-maturity dynamics
    - Yield curve shape analysis (steepness, curvature, inversion)
    - Duration and convexity considerations
    - Economic regime detection
    - Risk-adjusted positioning
    """
    
    def __init__(self, lookback_periods=12, risk_free_rate=0.02):
        """
        Initialize the sophisticated allocator.
        
        Args:
            lookback_periods (int): Number of periods for rolling calculations
            risk_free_rate (float): Risk-free rate for Sharpe calculations
        """
        self.lookback_periods = lookback_periods
        self.risk_free_rate = risk_free_rate
        self.scaler = StandardScaler()
        self.model = None
        
        # Treasury maturities and their characteristics
        self.maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        self.maturity_years = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        
        # Performance tracking
        self.performance_metrics = {}
        
    def calculate_coupon_equivalent_yields(self, yields_df):
        """
        Calculate coupon-equivalent yields for Treasury bills and notes.
        This is the standard way Treasury yields are quoted.
        """
        coupon_yields = yields_df.copy()
        
        for i, maturity in enumerate(self.maturities):
            years = self.maturity_years[i]
            
            if years <= 1.0:  # Treasury bills (zero-coupon)
                # Convert discount yield to coupon-equivalent yield
                # Formula: CEY = (365 * discount_rate) / (360 - discount_rate * days_to_maturity)
                days = years * 365
                discount_rate = yields_df[maturity] / 100  # Convert to decimal
                coupon_yields[maturity] = (365 * discount_rate) / (360 - discount_rate * days) * 100
            else:  # Treasury notes/bonds (semi-annual coupons)
                # For notes/bonds, the yield is already coupon-equivalent
                pass
                
        return coupon_yields
    
    def calculate_duration_and_convexity(self, yields_df, coupon_rates=None):
        """
        Calculate modified duration and convexity for each maturity.
        These are key risk metrics for bond portfolios.
        """
        if coupon_rates is None:
            # Assume coupon rates equal to yields (par bonds)
            coupon_rates = yields_df.copy()
        
        duration = pd.DataFrame(index=yields_df.index, columns=self.maturities)
        convexity = pd.DataFrame(index=yields_df.index, columns=self.maturities)
        
        for i, maturity in enumerate(self.maturities):
            years = self.maturity_years[i]
            yield_rate = yields_df[maturity] / 100
            coupon_rate = coupon_rates[maturity] / 100
            
            if years <= 1.0:  # Zero-coupon (T-bills)
                # Duration = time to maturity
                duration[maturity] = years
                # Convexity = time to maturity squared
                convexity[maturity] = years ** 2
            else:  # Coupon-paying bonds
                # Semi-annual payments
                n = years * 2
                y = yield_rate / 2
                c = coupon_rate / 2
                
                # Macaulay Duration
                mac_duration = 0
                for t in range(1, int(n) + 1):
                    if t == n:  # Final payment includes principal
                        payment = 100 * (1 + c)
                    else:
                        payment = 100 * c
                    pv = payment / ((1 + y) ** t)
                    mac_duration += (t / 2) * pv
                
                # Price calculation
                price = 0
                for t in range(1, int(n) + 1):
                    if t == n:
                        payment = 100 * (1 + c)
                    else:
                        payment = 100 * c
                    price += payment / ((1 + y) ** t)
                
                mac_duration = mac_duration / price
                
                # Modified Duration
                duration[maturity] = mac_duration / (1 + y)
                
                # Convexity (simplified)
                convexity[maturity] = mac_duration ** 2
        
        return duration, convexity
    
    def analyze_yield_curve_shape(self, yields_df):
        """
        Analyze yield curve shape and extract key features.
        """
        features = pd.DataFrame(index=yields_df.index)
        
        # Key spreads
        features['spread_2y10y'] = yields_df['10Y'] - yields_df['2Y']
        features['spread_3m10y'] = yields_df['10Y'] - yields_df['3M']
        features['spread_2y5y'] = yields_df['5Y'] - yields_df['2Y']
        features['spread_5y30y'] = yields_df['30Y'] - yields_df['5Y']
        
        # Curve steepness (overall slope)
        features['steepness'] = (yields_df['30Y'] - yields_df['3M']) / 29.75
        
        # Curve curvature (hump shape)
        features['curvature'] = (yields_df['2Y'] + yields_df['10Y']) / 2 - yields_df['5Y']
        
        # Short-term slope
        features['short_slope'] = yields_df['2Y'] - yields_df['3M']
        
        # Long-term slope
        features['long_slope'] = yields_df['30Y'] - yields_df['10Y']
        
        # Regime indicators
        features['is_steep'] = (features['spread_2y10y'] > 0.5).astype(int)
        features['is_inverted'] = (features['spread_2y10y'] < -0.2).astype(int)
        features['is_flat'] = ((features['spread_2y10y'] >= -0.2) & 
                              (features['spread_2y10y'] <= 0.5)).astype(int)
        
        # Volatility measures
        for maturity in self.maturities:
            features[f'{maturity}_volatility'] = yields_df[maturity].rolling(
                self.lookback_periods).std()
        
        # Momentum indicators
        for maturity in self.maturities:
            features[f'{maturity}_momentum'] = yields_df[maturity].diff(3)
            features[f'{maturity}_trend'] = yields_df[maturity].rolling(6).mean() - yields_df[maturity]
        
        return features
    
    def calculate_risk_adjusted_returns(self, returns_df, yields_df):
        """
        Calculate risk-adjusted returns and Sharpe ratios.
        """
        risk_metrics = pd.DataFrame(index=returns_df.index, columns=self.maturities)
        
        for maturity in self.maturities:
            # Rolling volatility
            volatility = returns_df[maturity].rolling(self.lookback_periods).std()
            
            # Rolling Sharpe ratio
            excess_returns = returns_df[maturity] - self.risk_free_rate / 12  # Monthly risk-free rate
            sharpe = excess_returns.rolling(self.lookback_periods).mean() / volatility
            
            risk_metrics[f'{maturity}_volatility'] = volatility
            risk_metrics[f'{maturity}_sharpe'] = sharpe
        
        return risk_metrics
    
    def create_sophisticated_features(self, yields_df, returns_df, economic_df=None):
        """
        Create sophisticated feature set incorporating all bond mechanics.
        """
        # Calculate coupon-equivalent yields
        coupon_yields = self.calculate_coupon_equivalent_yields(yields_df)
        
        # Calculate duration and convexity
        duration, convexity = self.calculate_duration_and_convexity(coupon_yields)
        
        # Analyze yield curve shape
        curve_features = self.analyze_yield_curve_shape(coupon_yields)
        
        # Calculate risk-adjusted returns
        risk_features = self.calculate_risk_adjusted_returns(returns_df, coupon_yields)
        
        # Combine all features
        features = pd.concat([
            curve_features,
            risk_features,
            duration.add_suffix('_duration'),
            convexity.add_suffix('_convexity')
        ], axis=1)
        
        # Add economic indicators if available
        if economic_df is not None:
            features = pd.concat([features, economic_df], axis=1)
        
        # Add time features
        features['month'] = yields_df.index.month
        features['quarter'] = yields_df.index.quarter
        features['year'] = yields_df.index.year
        
        # Cyclical encoding
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Remove any infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()
        
        return features
    
    def train_model(self, features_df, returns_df, test_size=0.2):
        """
        Train the sophisticated allocation model.
        """
        print("Training sophisticated Treasury allocation model...")
        
        # Prepare targets (next period returns)
        targets = returns_df.shift(-1).dropna()
        
        # Align features with targets
        common_index = features_df.index.intersection(targets.index)
        features_aligned = features_df.loc[common_index]
        targets_aligned = targets.loc[common_index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_aligned, targets_aligned, 
            test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        self.performance_metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred)
        }
        
        # Per-maturity R² scores
        maturity_r2 = {}
        for i, maturity in enumerate(self.maturities):
            maturity_r2[maturity] = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
        
        self.performance_metrics['maturity_r2'] = maturity_r2
        
        print("Model training completed!")
        print(f"Overall R² Score (Test): {self.performance_metrics['test_r2']:.4f}")
        
        return self.performance_metrics
    
    def predict_allocation(self, features, method='sophisticated'):
        """
        Predict optimal allocation using sophisticated logic.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get model predictions
        features_scaled = self.scaler.transform(features)
        predicted_returns = self.model.predict(features_scaled)[0]
        
        if method == 'sophisticated':
            return self._sophisticated_allocation(features, predicted_returns)
        elif method == 'risk_parity':
            return self._risk_parity_allocation(features)
        elif method == 'momentum':
            return self._momentum_allocation(features, predicted_returns)
        else:
            return self._sophisticated_allocation(features, predicted_returns)
    
    def _sophisticated_allocation(self, features, predicted_returns):
        """
        Sophisticated allocation incorporating curve shape, duration, and risk.
        """
        # Extract key features
        spread_2y10y = features['spread_2y10y'].iloc[0] if hasattr(features, 'iloc') else features['spread_2y10y']
        steepness = features['steepness'].iloc[0] if hasattr(features, 'iloc') else features['steepness']
        is_steep = features['is_steep'].iloc[0] if hasattr(features, 'iloc') else features['is_steep']
        is_inverted = features['is_inverted'].iloc[0] if hasattr(features, 'iloc') else features['is_inverted']
        
        # Base allocation on predicted returns
        base_weights = self._softmax_allocation(predicted_returns, temperature=0.3)
        
        # Adjust based on curve shape
        if is_steep:
            # Steep curve: favor intermediate maturities (2Y-10Y)
            curve_adjustment = np.array([0.05, 0.05, 0.10, 0.20, 0.25, 0.20, 0.15])
        elif is_inverted:
            # Inverted curve: favor short maturities for safety
            curve_adjustment = np.array([0.25, 0.25, 0.20, 0.15, 0.10, 0.03, 0.02])
        else:
            # Flat curve: balanced allocation
            curve_adjustment = np.array([0.10, 0.15, 0.20, 0.20, 0.15, 0.15, 0.05])
        
        # Combine base weights with curve adjustment
        final_weights = 0.7 * np.array(list(base_weights.values())) + 0.3 * curve_adjustment
        
        # Ensure minimum allocations and normalize
        final_weights = np.maximum(final_weights, 0.02)  # Minimum 2% allocation
        final_weights = final_weights / final_weights.sum()
        
        return dict(zip(self.maturities, final_weights))
    
    def _risk_parity_allocation(self, features):
        """
        Risk parity allocation based on duration and volatility.
        """
        # Extract duration and volatility features
        durations = []
        volatilities = []
        
        for maturity in self.maturities:
            duration_key = f'{maturity}_duration'
            vol_key = f'{maturity}_volatility'
            
            if duration_key in features.columns:
                durations.append(features[duration_key].iloc[0] if hasattr(features, 'iloc') else features[duration_key])
            else:
                durations.append(self.maturity_years[self.maturities.index(maturity)])
            
            if vol_key in features.columns:
                volatilities.append(features[vol_key].iloc[0] if hasattr(features, 'iloc') else features[vol_key])
            else:
                volatilities.append(0.02)  # Default volatility
        
        # Risk parity: equal risk contribution
        risk_contributions = np.array(durations) * np.array(volatilities)
        weights = 1 / risk_contributions
        weights = weights / weights.sum()
        
        return dict(zip(self.maturities, weights))
    
    def _momentum_allocation(self, features, predicted_returns):
        """
        Momentum-based allocation using trend indicators.
        """
        # Extract momentum features
        momentum_scores = []
        
        for maturity in self.maturities:
            momentum_key = f'{maturity}_momentum'
            trend_key = f'{maturity}_trend'
            
            momentum = features[momentum_key].iloc[0] if hasattr(features, 'iloc') else features[momentum_key]
            trend = features[trend_key].iloc[0] if hasattr(features, 'iloc') else features[trend_key]
            
            # Combine momentum and trend
            momentum_score = 0.7 * momentum + 0.3 * trend
            momentum_scores.append(momentum_score)
        
        # Convert to weights
        momentum_scores = np.array(momentum_scores)
        momentum_scores = np.maximum(momentum_scores, 0)  # Only positive momentum
        weights = momentum_scores / momentum_scores.sum() if momentum_scores.sum() > 0 else np.ones(7) / 7
        
        return dict(zip(self.maturities, weights))
    
    def _softmax_allocation(self, returns, temperature=1.0):
        """
        Softmax-based allocation.
        """
        exp_returns = np.exp(returns / temperature)
        weights = exp_returns / exp_returns.sum()
        return dict(zip(self.maturities, weights))
    
    def get_feature_importance(self):
        """Get feature importance from the model."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        feature_names = self.scaler.get_feature_names_out()
        
        return dict(zip(feature_names, importance))
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'performance_metrics': self.performance_metrics,
            'maturities': self.maturities,
            'maturity_years': self.maturity_years
        }
        
        joblib.dump(model_data, f"{filepath}.pkl")
        print(f"Model saved to {filepath}.pkl")
    
    def load_model(self, filepath):
        """Load a trained model."""
        model_data = joblib.load(f"{filepath}.pkl")
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.performance_metrics = model_data['performance_metrics']
        self.maturities = model_data['maturities']
        self.maturity_years = model_data['maturity_years']
        
        print(f"Model loaded from {filepath}.pkl") 