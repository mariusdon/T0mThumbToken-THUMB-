import pandas as pd
import numpy as np
from datetime import datetime
from generate_training_data import load_real_historical_data
from allocation_rules import compute_regime_allocation
from ml_model_utils import ml_allocation_model, ml_allocation_scaler

def calculate_bond_price(yield_rate, maturity_years, coupon_rate=None):
    """
    Calculate bond price using simplified bond pricing formula.
    For simplicity, assume par value of 100 and annual coupon payments.
    """
    if coupon_rate is None:
        coupon_rate = yield_rate  # Assume coupon equals yield for simplicity
    
    # Simplified bond price formula: P = C * (1 - (1 + Y)^-n) / Y + F / (1 + Y)^n
    # Where C = coupon payment, Y = yield, n = years to maturity, F = face value
    
    if yield_rate == 0:
        return 100.0
    
    face_value = 100.0
    coupon_payment = face_value * coupon_rate
    
    # Calculate present value of coupon payments
    pv_coupons = coupon_payment * (1 - (1 + yield_rate) ** (-maturity_years)) / yield_rate
    
    # Calculate present value of face value
    pv_face = face_value / ((1 + yield_rate) ** maturity_years)
    
    return pv_coupons + pv_face

def calculate_bond_return(start_yield, end_yield, maturity_years, coupon_rate=None):
    """
    Calculate bond return based on yield changes.
    """
    start_price = calculate_bond_price(start_yield, maturity_years, coupon_rate)
    end_price = calculate_bond_price(end_yield, maturity_years, coupon_rate)
    
    # Calculate coupon income (simplified)
    coupon_income = (coupon_rate or start_yield) * maturity_years
    
    # Total return = (end_price - start_price + coupon_income) / start_price
    total_return = (end_price - start_price + coupon_income) / start_price
    
    return total_return

def run_realistic_backtest():
    """
    Run a realistic backtest with proper bond price calculations.
    """
    print("Loading historical data...")
    df = load_real_historical_data()
    
    # Convert DATE to datetime index
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    
    # Maturity mapping
    maturities = {
        '3M': 0.25, '6M': 0.5, '1Y': 1.0, '2Y': 2.0, 
        '5Y': 5.0, '10Y': 10.0, '30Y': 30.0
    }
    
    # Initialize portfolio values
    ml_portfolio = [100.0]  # Start at $100
    equal_weight_portfolio = [100.0]
    dates = [df.index[0]]
    
    # Equal weight allocation
    equal_weight = np.ones(7) / 7
    
    print("Running realistic backtest...")
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        prev_date = df.index[i-1]
        
        # Get current and previous yields
        current_yields = df.iloc[i][['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']].values / 100  # Convert to decimal
        prev_yields = df.iloc[i-1][['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']].values / 100
        
        # Calculate bond returns for each maturity
        bond_returns = []
        for j, (maturity, years) in enumerate(maturities.items()):
            if prev_yields[j] > 0 and current_yields[j] > 0:
                ret = calculate_bond_return(prev_yields[j], current_yields[j], years)
                bond_returns.append(ret)
            else:
                bond_returns.append(0.0)
        
        bond_returns = np.array(bond_returns)
        
        # Equal weight portfolio return
        equal_weight_return = np.sum(equal_weight * bond_returns)
        equal_weight_portfolio.append(equal_weight_portfolio[-1] * (1 + equal_weight_return))
        
        # ML portfolio return
        try:
            # Get ML allocation
            if ml_allocation_model is not None and ml_allocation_scaler is not None:
                # Prepare features for ML model
                features = df.iloc[i-1][['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y', 'CPI', 'UNEMP', 'MOVE']].values
                features_scaled = ml_allocation_scaler.transform(features.reshape(1, -1))
                ml_weights = ml_allocation_model.predict(features_scaled)[0]
                ml_weights = np.minimum(ml_weights, 0.4)  # Cap at 40%
                ml_weights = ml_weights / np.sum(ml_weights)
            else:
                # Fallback to regime rule
                yields_list = df.iloc[i-1][['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']].values / 100
                ml_weights, _ = compute_regime_allocation(yields_list)
                ml_weights = np.array(ml_weights)
        except Exception as e:
            print(f"Warning: Using equal weight for ML at {current_date}: {e}")
            ml_weights = equal_weight
        
        ml_return = np.sum(ml_weights * bond_returns)
        ml_portfolio.append(ml_portfolio[-1] * (1 + ml_return))
        
        dates.append(current_date)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'date': dates,
        'ml_value': ml_portfolio,
        'equal_weight_value': equal_weight_portfolio
    })
    
    # Save results
    results_df.to_csv('backtest_results.csv', index=False)
    
    # Calculate and print performance metrics
    ml_total_return = (ml_portfolio[-1] / ml_portfolio[0] - 1) * 100
    equal_total_return = (equal_weight_portfolio[-1] / equal_weight_portfolio[0] - 1) * 100
    outperformance = ml_total_return - equal_total_return
    
    print(f"\nRealistic Backtest Results:")
    print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"ML Strategy Total Return: {ml_total_return:.2f}%")
    print(f"Equal Weight Total Return: {equal_total_return:.2f}%")
    print(f"ML Outperformance: {outperformance:.2f}%")
    print(f"Final ML Portfolio Value: ${ml_portfolio[-1]:.2f}")
    print(f"Final Equal Weight Portfolio Value: ${equal_weight_portfolio[-1]:.2f}")
    
    return results_df

if __name__ == "__main__":
    run_realistic_backtest() 