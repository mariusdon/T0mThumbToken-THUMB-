import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_loader import TreasuryDataLoader
from sophisticated_model import SophisticatedTreasuryAllocator
from sophisticated_backtest import SophisticatedTreasuryBacktester

def main():
    """Main training and backtesting pipeline for sophisticated model."""
    print("="*70)
    print("SOPHISTICATED TREASURY ALLOCATION MODEL TRAINING")
    print("="*70)
    
    # Step 1: Load and prepare data
    print("\n1. Loading historical Treasury data...")
    loader = TreasuryDataLoader()
    
    # Load data from 2015 to present
    features, returns, yields = loader.prepare_training_data(
        start_date='2015-01-01',
        end_date=None,
        frequency='M'  # Monthly rebalancing
    )
    
    print(f"✓ Data loaded successfully:")
    print(f"  - Features: {features.shape}")
    print(f"  - Returns: {returns.shape}")
    print(f"  - Yields: {yields.shape}")
    
    # Step 2: Create sophisticated features
    print("\n2. Creating sophisticated features...")
    sophisticated_model = SophisticatedTreasuryAllocator()
    
    # Create sophisticated features incorporating bond mechanics
    sophisticated_features = sophisticated_model.create_sophisticated_features(
        yields, returns
    )
    
    print(f"✓ Sophisticated features created: {sophisticated_features.shape}")
    
    # Step 3: Train the sophisticated model
    print("\n3. Training sophisticated model...")
    
    # Train the model
    metrics = sophisticated_model.train_model(sophisticated_features, returns, test_size=0.2)
    
    # Step 4: Run sophisticated backtest
    print("\n4. Running sophisticated backtest...")
    sophisticated_backtester = SophisticatedTreasuryBacktester(
        sophisticated_model, sophisticated_features, returns, yields
    )
    
    # Run backtest on recent data (2020 onwards)
    results = sophisticated_backtester.run_sophisticated_backtest(
        start_date='2020-01-01',
        end_date=None,
        rebalance_freq='M',
        allocation_method='sophisticated',
        transaction_cost=0.001,
        initial_capital=100000
    )
    
    # Step 5: Save model and results
    print("\n5. Saving model and results...")
    
    # Save the trained model
    sophisticated_model.save_model('sophisticated_treasury_model')
    
    # Save backtest results
    sophisticated_backtester.save_results('sophisticated_backtest_results.pkl')
    
    # Save current yield data for API
    current_yields = yields.iloc[-1].to_dict()
    current_features = sophisticated_features.iloc[-1].to_dict()
    
    # Get current allocation prediction
    current_allocation = sophisticated_model.predict_allocation(
        pd.DataFrame([current_features]), method='sophisticated'
    )
    
    # Save current state for API
    current_state = {
        'yields': current_yields,
        'features': current_features,
        'allocation': current_allocation,
        'model_performance': metrics,
        'backtest_results': {k: v for k, v in results.items() 
                           if not isinstance(v, (pd.DataFrame, pd.Series))}
    }
    
    joblib.dump(current_state, 'sophisticated_current_state.pkl')
    
    # Step 6: Generate performance report
    print("\n6. Generating performance report...")
    generate_sophisticated_performance_report(results, metrics)
    
    # Step 7: Plot results
    print("\n7. Generating sophisticated plots...")
    sophisticated_backtester.plot_sophisticated_results('sophisticated_backtest_plots.png')
    
    print("\n" + "="*70)
    print("SOPHISTICATED TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Files created:")
    print("  - sophisticated_treasury_model.pkl (trained sophisticated model)")
    print("  - sophisticated_backtest_results.pkl (sophisticated backtest results)")
    print("  - sophisticated_current_state.pkl (current market state)")
    print("  - sophisticated_backtest_plots.png (performance plots)")
    print("="*70)

def generate_sophisticated_performance_report(backtest_results, model_metrics):
    """Generate a comprehensive performance report for sophisticated model."""
    report = f"""
SOPHISTICATED TREASURY ALLOCATION MODEL - PERFORMANCE REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL FEATURES:
- Coupon-equivalent yield calculations
- Duration and convexity analysis
- Yield curve shape recognition (steepness, curvature, inversion)
- Risk-adjusted positioning
- Economic regime detection
- Sophisticated allocation logic

MODEL PERFORMANCE:
- Overall R² Score (Test): {model_metrics['test_r2']:.4f}
- Overall MSE (Test): {model_metrics['test_mse']:.6f}

Per-Maturity R² Scores (Test):
"""
    
    for maturity, r2 in model_metrics['maturity_r2'].items():
        report += f"  - {maturity}: {r2:.4f}\n"
    
    report += f"""
SOPHISTICATED BACKTEST RESULTS (2020-Present):
- ML Strategy Total Return: {backtest_results['ml_total_return']:.2%}
- Equal Weight Total Return: {backtest_results['equal_weight_total_return']:.2%}
- ML Strategy Annualized Return: {backtest_results['ml_annualized_return']:.2%}
- Equal Weight Annualized Return: {backtest_results['equal_weight_annualized_return']:.2%}
- ML Strategy Sharpe Ratio: {backtest_results['ml_sharpe']:.3f}
- Equal Weight Sharpe Ratio: {backtest_results['equal_weight_sharpe']:.3f}
- ML Strategy Max Drawdown: {backtest_results['ml_max_drawdown']:.2%}
- Equal Weight Max Drawdown: {backtest_results['equal_weight_max_drawdown']:.2%}
- Information Ratio: {backtest_results['information_ratio']:.3f}
- ML Strategy Win Rate: {backtest_results['ml_win_rate']:.2%}
- Equal Weight Win Rate: {backtest_results['equal_weight_win_rate']:.2%}

OUTPERFORMANCE ANALYSIS:
- Total Return Outperformance: {backtest_results['ml_total_return'] - backtest_results['equal_weight_total_return']:.2%}
- Sharpe Ratio Outperformance: {backtest_results['ml_sharpe'] - backtest_results['equal_weight_sharpe']:.3f}
- Risk-Adjusted Outperformance: {backtest_results['information_ratio']:.3f}

STRATEGY CHARACTERISTICS:
- Incorporates realistic bond mechanics (coupons, duration, convexity)
- Dynamic allocation based on yield curve shape
- Risk-adjusted positioning with transaction costs
- Regime-aware allocation (steep, flat, inverted curves)
- Sophisticated feature engineering for bond markets
"""
    
    # Save report
    with open('sophisticated_performance_report.txt', 'w') as f:
        f.write(report)
    
    print("Sophisticated performance report saved to sophisticated_performance_report.txt")
    print(report)

if __name__ == "__main__":
    main() 