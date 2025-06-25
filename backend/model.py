import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class TreasuryAllocationModel:
    """
    Random Forest model for predicting optimal Treasury bond allocations.
    Uses MultiOutputRegressor to predict returns for all 7 maturities simultaneously.
    """
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize the Random Forest allocation model.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees
            random_state (int): Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Initialize the model
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            oob_score=True
        )
        
        # Wrap in MultiOutputRegressor for multiple target variables
        self.model = MultiOutputRegressor(self.rf_model)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Store feature names and maturity names
        self.feature_names = None
        self.maturity_names = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        
        # Model performance metrics
        self.performance_metrics = {}
        
    def prepare_targets(self, returns_df, forecast_horizon=1):
        """
        Prepare target variables for training.
        
        Args:
            returns_df (pd.DataFrame): Historical returns data
            forecast_horizon (int): Number of periods ahead to predict
            
        Returns:
            pd.DataFrame: Target variables (future returns)
        """
        # Shift returns forward to create targets
        targets = returns_df.shift(-forecast_horizon)
        
        # Remove the last few rows where we don't have targets
        targets = targets.dropna()
        
        return targets
    
    def train(self, features_df, returns_df, forecast_horizon=1, test_size=0.2):
        """
        Train the Random Forest model.
        
        Args:
            features_df (pd.DataFrame): Feature matrix
            returns_df (pd.DataFrame): Historical returns data
            forecast_horizon (int): Number of periods ahead to predict
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Training performance metrics
        """
        print(f"Training Random Forest model with {self.n_estimators} trees...")
        
        # Prepare targets
        targets = self.prepare_targets(returns_df, forecast_horizon)
        
        # Align features with targets
        common_index = features_df.index.intersection(targets.index)
        features_aligned = features_df.loc[common_index]
        targets_aligned = targets.loc[common_index]
        
        # Store feature names
        self.feature_names = features_aligned.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_aligned, targets_aligned, 
            test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training MultiOutput Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_metrics(
            y_train, y_train_pred, y_test, y_test_pred
        )
        
        # Print performance summary
        self._print_performance_summary()
        
        return self.performance_metrics
    
    def _calculate_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate model performance metrics."""
        metrics = {}
        
        # Overall R² scores
        metrics['train_r2'] = r2_score(y_train, y_train_pred)
        metrics['test_r2'] = r2_score(y_test, y_test_pred)
        
        # MSE scores
        metrics['train_mse'] = mean_squared_error(y_train, y_train_pred)
        metrics['test_mse'] = mean_squared_error(y_test, y_test_pred)
        
        # Per-maturity R² scores
        maturity_r2_train = []
        maturity_r2_test = []
        
        for i, maturity in enumerate(self.maturity_names):
            train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
            test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
            maturity_r2_train.append(train_r2)
            maturity_r2_test.append(test_r2)
            
        metrics['maturity_r2_train'] = dict(zip(self.maturity_names, maturity_r2_train))
        metrics['maturity_r2_test'] = dict(zip(self.maturity_names, maturity_r2_test))
        
        return metrics
    
    def _print_performance_summary(self):
        """Print model performance summary."""
        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Overall R² Score (Train): {self.performance_metrics['train_r2']:.4f}")
        print(f"Overall R² Score (Test):  {self.performance_metrics['test_r2']:.4f}")
        print(f"Overall MSE (Train):      {self.performance_metrics['train_mse']:.6f}")
        print(f"Overall MSE (Test):       {self.performance_metrics['test_mse']:.6f}")
        
        print("\nPer-Maturity R² Scores (Test):")
        for maturity, r2 in self.performance_metrics['maturity_r2_test'].items():
            print(f"  {maturity}: {r2:.4f}")
        print("="*50)
    
    def predict_returns(self, features):
        """
        Predict returns for all maturities.
        
        Args:
            features (pd.DataFrame or dict): Feature values
            
        Returns:
            np.array: Predicted returns for each maturity
        """
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Ensure features are in the correct order
        if self.feature_names:
            features = features[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        
        return predictions[0]  # Return first (and only) prediction
    
    def predict_allocation(self, features, method='curve_aware', risk_free_rate=0.02):
        """
        Predict optimal allocation weights based on predicted returns and curve shape.
        
        Args:
            features (pd.DataFrame or dict): Feature values
            method (str): Allocation method ('softmax', 'inverse_vol', 'rank', 'curve_aware')
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
            
        Returns:
            dict: Allocation weights for each maturity
        """
        # Predict returns
        predicted_returns = self.predict_returns(features)
        
        # Convert to allocation weights based on method
        if method == 'curve_aware':
            weights = self._curve_aware_allocation(features, predicted_returns)
        elif method == 'softmax':
            weights = self._softmax_allocation(predicted_returns)
        elif method == 'inverse_vol':
            weights = self._inverse_volatility_allocation(predicted_returns)
        elif method == 'rank':
            weights = self._rank_based_allocation(predicted_returns)
        else:
            raise ValueError(f"Unknown allocation method: {method}")
        
        # Create result dictionary
        allocation = dict(zip(self.maturity_names, weights))
        
        return allocation
    
    def _curve_aware_allocation(self, features, predicted_returns):
        """
        Allocate based on yield curve shape and predicted returns.
        This is the most sophisticated allocation method.
        """
        # Determine curve shape
        if isinstance(features, dict):
            slope_10y_2y = features.get('slope_10y_2y', 0)
            is_steep = features.get('is_steep', 0)
            is_inverted = features.get('is_inverted', 0)
            is_flat = features.get('is_flat', 1)
        else:
            slope_10y_2y = features.get('slope_10y_2y', 0) if hasattr(features, 'get') else 0
            is_steep = 1 if slope_10y_2y > 0.5 else 0
            is_inverted = 1 if slope_10y_2y < -0.5 else 0
            is_flat = 1 if not (is_steep or is_inverted) else 0
        
        # Base allocation weights by curve shape
        if is_steep:
            # Steep curve: favor longer maturities for higher yields
            base_weights = np.array([0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.15])
            print("Curve shape: STEEP - Favoring longer maturities")
        elif is_inverted:
            # Inverted curve: favor shorter maturities for safety
            base_weights = np.array([0.25, 0.25, 0.20, 0.15, 0.10, 0.03, 0.02])
            print("Curve shape: INVERTED - Favoring shorter maturities")
        else:
            # Flat curve: balanced allocation
            base_weights = np.array([0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16])
            print("Curve shape: FLAT - Balanced allocation")
        
        # Adjust based on predicted returns
        # Normalize predicted returns to 0-1 range
        returns_normalized = (predicted_returns - predicted_returns.min()) / (predicted_returns.max() - predicted_returns.min() + 1e-8)
        
        # Combine base weights with return predictions
        # 70% curve shape, 30% return prediction
        final_weights = 0.7 * base_weights + 0.3 * returns_normalized
        
        # Normalize to sum to 1
        final_weights = final_weights / np.sum(final_weights)
        
        return final_weights
    
    def _softmax_allocation(self, returns, temperature=1.0):
        """Convert predicted returns to weights using softmax."""
        # Apply softmax function
        exp_returns = np.exp(returns / temperature)
        weights = exp_returns / np.sum(exp_returns)
        return weights
    
    def _inverse_volatility_allocation(self, returns, volatility_window=30):
        """Allocate based on inverse volatility (simplified)."""
        # For simplicity, use equal volatility assumption
        # In practice, you'd calculate historical volatility
        weights = np.ones(len(returns)) / len(returns)
        return weights
    
    def _rank_based_allocation(self, returns):
        """Allocate based on ranking of predicted returns."""
        # Rank returns (higher rank = higher return)
        ranks = np.argsort(np.argsort(returns))[::-1] + 1
        
        # Convert ranks to weights (higher rank = higher weight)
        weights = ranks / np.sum(ranks)
        return weights
    
    def get_feature_importance(self):
        """Get feature importance from the Random Forest model."""
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("Model must be trained before getting feature importance")
        
        # Average feature importance across all estimators
        importance_matrix = np.array([
            estimator.feature_importances_ 
            for estimator in self.model.estimators_
        ])
        
        avg_importance = np.mean(importance_matrix, axis=0)
        
        # Create feature importance dictionary
        feature_importance = dict(zip(self.feature_names, avg_importance))
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return feature_importance
    
    def save_model(self, filepath):
        """Save the trained model and scaler."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'maturity_names': self.maturity_names,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and scaler."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.maturity_names = model_data['maturity_names']
        self.performance_metrics = model_data['performance_metrics']
        print(f"Model loaded from {filepath}")

    def create_features(self, yield_df, economic_df=None):
        """
        Create feature set for ML model with enhanced curve shape recognition.
        
        Args:
            yield_df (pd.DataFrame): Daily yield data
            economic_df (pd.DataFrame): Economic data (optional)
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        print("Creating enhanced feature set with curve shape recognition...")
        
        features = pd.DataFrame(index=yield_df.index)
        
        # Core yield curve slopes
        features['slope_10y_2y'] = yield_df['10Y'] - yield_df['2Y']
        features['slope_30y_3m'] = yield_df['30Y'] - yield_df['3M']
        features['slope_5y_1y'] = yield_df['5Y'] - yield_df['1Y']
        features['slope_10y_3m'] = yield_df['10Y'] - yield_df['3M']
        features['slope_2y_3m'] = yield_df['2Y'] - yield_df['3M']
        features['slope_5y_2y'] = yield_df['5Y'] - yield_df['2Y']
        
        # Curve shape indicators
        features['curve_steepness'] = (yield_df['30Y'] - yield_df['3M']) / 27  # Normalized steepness
        features['curve_inversion'] = (yield_df['2Y'] - yield_df['10Y']) / 8  # Inversion indicator
        
        # Level of short rates
        features['short_rate_3m'] = yield_df['3M']
        features['short_rate_6m'] = yield_df['6M']
        features['short_rate_1y'] = yield_df['1Y']
        
        # Yield curve curvature (hump shape)
        features['curvature_2y_5y_10y'] = (yield_df['2Y'] + yield_df['10Y']) / 2 - yield_df['5Y']
        features['curvature_3m_2y_10y'] = (yield_df['3M'] + yield_df['10Y']) / 2 - yield_df['2Y']
        
        # Regime classification features
        features['is_steep'] = (features['slope_10y_2y'] > 0.5).astype(int)
        features['is_inverted'] = (features['slope_10y_2y'] < -0.5).astype(int)
        features['is_flat'] = ((features['slope_10y_2y'] >= -0.5) & (features['slope_10y_2y'] <= 0.5)).astype(int)
        
        # Lagged slopes for momentum
        features['slope_10y_2y_lag1w'] = features['slope_10y_2y'].shift(7)
        features['slope_10y_2y_lag4w'] = features['slope_10y_2y'].shift(28)
        features['slope_momentum'] = features['slope_10y_2y'] - features['slope_10y_2y_lag4w']
        
        # Time-based features
        features['month'] = yield_df.index.month
        features['quarter'] = yield_df.index.quarter
        features['year'] = yield_df.index.year
        
        # Volatility features (rolling standard deviation)
        for maturity in self.maturity_names:
            features[f'{maturity}_volatility'] = yield_df[maturity].rolling(30).std()
        
        # Relative value features
        features['relative_value_2y'] = yield_df['2Y'] - yield_df['2Y'].rolling(60).mean()
        features['relative_value_10y'] = yield_df['10Y'] - yield_df['10Y'].rolling(60).mean()
        features['relative_value_30y'] = yield_df['30Y'] - yield_df['30Y'].rolling(60).mean()
        
        # Add economic indicators if available
        if economic_df is not None:
            # Align economic data with yield data
            economic_aligned = economic_df.reindex(yield_df.index, method='ffill')
            features = pd.concat([features, economic_aligned], axis=1)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        print(f"Enhanced feature matrix shape: {features.shape}")
        return features

if __name__ == "__main__":
    # Test the model
    from data_loader import TreasuryDataLoader
    
    # Load data
    loader = TreasuryDataLoader()
    features, returns, yields = loader.prepare_training_data()
    
    # Initialize and train model
    model = TreasuryAllocationModel(n_estimators=50, max_depth=8)
    metrics = model.train(features, returns)
    
    # Test prediction
    sample_features = features.iloc[-1].to_dict()
    allocation = model.predict_allocation(sample_features)
    
    print("\nSample allocation prediction:")
    for maturity, weight in allocation.items():
        print(f"  {maturity}: {weight:.4f}")
    
    # Feature importance
    importance = model.get_feature_importance()
    print("\nTop 10 most important features:")
    for i, (feature, imp) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {feature}: {imp:.4f}") 