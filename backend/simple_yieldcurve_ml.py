import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_loader import TreasuryDataLoader

def engineer_features(yield_df):
    """Create simple features: yields and key slopes."""
    features = pd.DataFrame(index=yield_df.index)
    mats = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
    for m in mats:
        features[f'yield_{m}'] = yield_df[m]
    # Key slopes
    features['slope_2y10y'] = yield_df['10Y'] - yield_df['2Y']
    features['slope_3m10y'] = yield_df['10Y'] - yield_df['3M']
    features['slope_5y30y'] = yield_df['30Y'] - yield_df['5Y']
    features['slope_2y5y'] = yield_df['5Y'] - yield_df['2Y']
    features['short_rate'] = yield_df['3M']
    features = features.dropna()
    return features

def softmax(x, t=1.0):
    x = np.array(x)
    e_x = np.exp(x / t)
    return e_x / e_x.sum()

def run_backtest(model, scaler, features, returns, method_name):
    """Backtest ML allocation vs. equal-weight."""
    mats = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
    dates = features.index.intersection(returns.index)
    ml_vals = [100]
    eq_vals = [100]
    for i in range(1, len(dates)):
        X = scaler.transform([features.loc[dates[i-1]]])
        pred = model.predict(X)[0]
        weights = softmax(pred, t=0.5)
        # Ensure min allocation
        weights = np.maximum(weights, 0.05)
        weights = weights / weights.sum()
        r = returns.loc[dates[i]]
        ml_ret = np.dot(weights, r[mats])
        eq_ret = np.mean(r[mats])
        ml_vals.append(ml_vals[-1] * (1 + ml_ret))
        eq_vals.append(eq_vals[-1] * (1 + eq_ret))
    ml_total = ml_vals[-1] / ml_vals[0] - 1
    eq_total = eq_vals[-1] / eq_vals[0] - 1
    print(f"{method_name} ML total return: {ml_total:.2%} | Equal-weight: {eq_total:.2%}")
    return ml_total, eq_total, ml_vals, eq_vals, dates

def main():
    print("\n=== Simple Yield Curve ML Model Comparison ===\n")
    loader = TreasuryDataLoader()
    features, returns, yields = loader.prepare_training_data(
        start_date='2015-01-01', end_date=None, frequency='M')
    # Use only yield curve features
    X = engineer_features(yields)
    y = returns.loc[X.index]
    # Align
    X, y = X.align(y, join='inner', axis=0)
    # Predict next-period returns
    y_target = y.shift(-1).dropna()
    X = X.loc[y_target.index]
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_mse = mean_squared_error(y_test, rf_pred)
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    print(f"Random Forest MSE: {rf_mse:.6f}")
    print(f"Ridge Regression MSE: {ridge_mse:.6f}")
    # Backtest both
    print("\nBacktesting Random Forest...")
    rf_ml, rf_eq, rf_vals, eq_vals, dates = run_backtest(rf, scaler, X, y, "Random Forest")
    print("\nBacktesting Ridge Regression...")
    ridge_ml, ridge_eq, ridge_vals, eq_vals2, dates2 = run_backtest(ridge, scaler, X, y, "Ridge Regression")
    # Pick best
    if rf_ml > ridge_ml and rf_ml > rf_eq:
        print("\nRandom Forest outperforms equal-weight and Ridge. Saving model...")
        joblib.dump({'model': rf, 'scaler': scaler, 'type': 'RandomForest'}, 'best_simple_yieldcurve_model.pkl')
    elif ridge_ml > rf_ml and ridge_ml > ridge_eq:
        print("\nRidge Regression outperforms equal-weight and RF. Saving model...")
        joblib.dump({'model': ridge, 'scaler': scaler, 'type': 'Ridge'}, 'best_simple_yieldcurve_model.pkl')
    else:
        print("\nNeither ML model outperformed equal-weight. Not saving model.")
    # Save results for review
    pd.DataFrame({'date': dates, 'rf_ml': rf_vals, 'ridge_ml': ridge_vals, 'eq': eq_vals}).to_csv('simple_yieldcurve_backtest.csv', index=False)
    print("\nResults saved to simple_yieldcurve_backtest.csv")

if __name__ == "__main__":
    main() 