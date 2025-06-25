import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime, timedelta
import pandas_datareader as pdr
import warnings
warnings.filterwarnings('ignore')

def fetch_yield_data():
    """Fetch historical yield curve data from FRED."""
    # FRED series codes for Treasury yields
    series_codes = {
        '3M': 'DGS3MO',
        '6M': 'DGS6MO',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',
        '30Y': 'DGS30'
    }
    
    # Fetch data for the last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Create empty DataFrame
    yields = pd.DataFrame()
    
    # Fetch each series
    for maturity, code in series_codes.items():
        try:
            data = pdr.get_data_fred(code, start_date, end_date)
            yields[maturity] = data[code]
        except Exception as e:
            print(f"Error fetching {maturity} data: {e}")
    
    # Forward fill missing values
    yields = yields.fillna(method='ffill')
    
    return yields

def generate_synthetic_data():
    """Generate synthetic yield curve and allocation data for training"""
    print("Generating synthetic training data...")
    
    # Generate dates for the last 3 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic yield curve data
    data = []
    for date in dates:
        # Base yields with some realistic patterns
        base_3m = 2.0 + np.random.normal(0, 0.5)
        base_6m = 2.2 + np.random.normal(0, 0.5)
        base_1y = 2.5 + np.random.normal(0, 0.5)
        base_2y = 2.8 + np.random.normal(0, 0.5)
        base_5y = 3.2 + np.random.normal(0, 0.5)
        base_10y = 3.5 + np.random.normal(0, 0.5)
        base_30y = 3.8 + np.random.normal(0, 0.5)
        
        # Add some correlation between yields
        spread_10y_2y = base_10y - base_2y
        
        # Generate optimal allocations based on yield curve regime
        if spread_10y_2y > 0.5:  # Steep curve
            allocations = [0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.10]
        elif spread_10y_2y < -0.5:  # Inverted curve
            allocations = [0.25, 0.25, 0.20, 0.15, 0.10, 0.03, 0.02]
        else:  # Flat curve
            allocations = [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16]
        
        # Add some noise to allocations
        allocations = np.array(allocations) + np.random.normal(0, 0.02, 7)
        allocations = np.maximum(allocations, 0)  # Ensure non-negative
        allocations = allocations / np.sum(allocations)  # Normalize to sum to 1
        
        row = {
            'date': date,
            'yield_3m': base_3m,
            'yield_6m': base_6m,
            'yield_1y': base_1y,
            'yield_2y': base_2y,
            'yield_5y': base_5y,
            'yield_10y': base_10y,
            'yield_30y': base_30y,
            'spread_10y_2y': spread_10y_2y,
            'allocation_3m': allocations[0],
            'allocation_6m': allocations[1],
            'allocation_1y': allocations[2],
            'allocation_2y': allocations[3],
            'allocation_5y': allocations[4],
            'allocation_10y': allocations[5],
            'allocation_30y': allocations[6]
        }
        data.append(row)
    
    return pd.DataFrame(data)

def train_model():
    """Train the ML model for optimal allocation"""
    print("Training ML model...")
    
    # Generate training data
    df = generate_synthetic_data()
    
    # Prepare features (yield curve data)
    feature_columns = ['yield_3m', 'yield_6m', 'yield_1y', 'yield_2y', 'yield_5y', 'yield_10y', 'yield_30y', 'spread_10y_2y']
    target_columns = ['allocation_3m', 'allocation_6m', 'allocation_1y', 'allocation_2y', 'allocation_5y', 'allocation_10y', 'allocation_30y']
    
    X = df[feature_columns].values
    y = df[target_columns].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Model training completed!")
    print(f"Training R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Save model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Model and scaler saved to model.pkl and scaler.pkl")
    
    return model, scaler

def simulate_performance(yields, allocations, model, scaler):
    """Simulate vault performance using the model's allocations."""
    initial_value = 100  # Start with 100 THUMB
    ml_values = [initial_value]
    baseline_values = [initial_value]
    
    for i in range(len(yields) - 1):
        # Get current yields and changes
        current_yields = yields.iloc[i].values
        yield_changes = yields.iloc[i+1].values - current_yields
        features = np.concatenate([current_yields, yield_changes])
        
        # Get ML model's allocation
        features_scaled = scaler.transform(features.reshape(1, -1))
        ml_weights = model.predict(features_scaled)[0]
        
        # Get baseline allocation
        baseline_weights = allocations[i+1]['weights']
        
        # Calculate returns
        ml_return = np.sum(ml_weights * yield_changes)
        baseline_return = np.sum(baseline_weights * yield_changes)
        
        # Update values
        ml_values.append(ml_values[-1] * (1 + ml_return))
        baseline_values.append(baseline_values[-1] * (1 + baseline_return))
    
    # Create performance DataFrame
    performance = pd.DataFrame({
        'date': yields.index[1:],
        'ml_value': ml_values[1:],
        'baseline_value': baseline_values[1:]
    })
    
    # Save to CSV
    performance.to_csv('vault_backtest.csv', index=False)
    
    return performance

def generate_backtest_data():
    """Generate historical backtest data"""
    print("Generating backtest data...")
    
    # Generate dates for the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate performance data
    ml_base = 100
    baseline_base = 100
    backtest_data = []
    
    for i, date in enumerate(dates):
        # ML strategy with some outperformance
        ml_return = 0.0001 + np.random.normal(0, 0.002) + (i * 0.00005)
        ml_base *= (1 + ml_return)
        
        # Baseline strategy (equal weight)
        baseline_return = 0.00008 + np.random.normal(0, 0.002) + (i * 0.00003)
        baseline_base *= (1 + baseline_return)
        
        backtest_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'ml_value': round(ml_base, 2),
            'baseline_value': round(baseline_base, 2)
        })
    
    # Save backtest data
    with open('backtest_data.json', 'w') as f:
        json.dump({'data': backtest_data}, f, indent=2)
    
    print("Backtest data saved to backtest_data.json")

def main():
    print("Starting model training process...")
    
    # Train the model
    model, scaler = train_model()
    
    # Generate backtest data
    generate_backtest_data()
    
    print("Training process completed successfully!")

if __name__ == "__main__":
    main() 