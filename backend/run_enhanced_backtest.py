import pandas as pd
import os
from enhanced_backtest import EnhancedTreasuryBacktester
from ml_model_utils import ml_allocation_model, ml_allocation_scaler
from generate_training_data import load_real_historical_data
import numpy as np

# Create a wrapper class for the ML model
class MLModelWrapper:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.maturity_names = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
    
    def predict(self, X):
        if self.model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        else:
            # Fallback to equal weights
            return np.ones((X.shape[0], len(self.maturity_names))) / len(self.maturity_names)

# 1. Check if ML model exists, if not train it
if not os.path.exists('ml_allocation_model.pkl') or not os.path.exists('ml_allocation_scaler.pkl'):
    print('ML model files not found. Training new model...')
    try:
        # Generate training data
        from generate_training_data import generate_training_data
        generate_training_data()
        
        # Train the model
        from train_ml_model import train_ml_model
        train_ml_model()
        print('ML model training completed!')
    except Exception as e:
        print(f'Warning: Could not train ML model: {e}')
        print('Will use regime-based allocation as fallback.')

# 2. Load historical data
print('Loading historical data...')
df = load_real_historical_data()

# 3. Convert DATE column to datetime index
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')

# 4. Prepare features and yields DataFrames
feature_cols = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y', 'CPI', 'UNEMP', 'MOVE']
yield_cols = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']

# Check if we have the required columns
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    print(f'Warning: Missing columns: {missing_cols}')
    # Use only available columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    yield_cols = [col for col in yield_cols if col in df.columns]

features_df = df[feature_cols]
yield_df = df[yield_cols]

print(f'Loaded data from {df.index.min()} to {df.index.max()}')
print(f'Features shape: {features_df.shape}')
print(f'Yields shape: {yield_df.shape}')

# 5. Create wrapped ML model
wrapped_model = MLModelWrapper(ml_allocation_model, ml_allocation_scaler)

# 6. Run enhanced backtest
print('Running enhanced backtest...')
backtester = EnhancedTreasuryBacktester(
    model=wrapped_model,
    features_df=features_df,
    returns_df=None,  # Not used in enhanced version
    yield_df=yield_df
)
backtester.run_enhanced_backtest()

print('Enhanced backtest complete. Results saved to backtest_results.csv') 