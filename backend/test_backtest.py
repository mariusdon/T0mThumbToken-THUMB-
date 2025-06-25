#!/usr/bin/env python3

from enhanced_backtest import EnhancedTreasuryBacktester
import numpy as np

def test_backtest():
    print("Testing Enhanced Treasury Backtester...")
    
    try:
        # Initialize backtester
        bt = EnhancedTreasuryBacktester()
        
        # Test ML strategy
        print("\n--- ML Strategy ---")
        ml_results = bt.run_backtest('ml')
        ml_values = ml_results['portfolio_values']
        print(f"ML portfolio values: {len(ml_values)} points")
        print(f"First 5 values: {ml_values[:5]}")
        print(f"Last 5 values: {ml_values[-5:]}")
        
        # Test baseline strategy
        print("\n--- Baseline Strategy ---")
        baseline_results = bt.run_backtest('equal_weight')
        baseline_values = baseline_results['portfolio_values']
        print(f"Baseline portfolio values: {len(baseline_values)} points")
        print(f"First 5 values: {baseline_values[:5]}")
        print(f"Last 5 values: {baseline_values[-5:]}")
        
        # Calculate some basic metrics
        print("\n--- Basic Metrics ---")
        ml_total_return = ((ml_values[-1] / ml_values[0]) - 1) * 100
        baseline_total_return = ((baseline_values[-1] / baseline_values[0]) - 1) * 100
        
        print(f"ML Total Return: {ml_total_return:.2f}%")
        print(f"Baseline Total Return: {baseline_total_return:.2f}%")
        print(f"Difference: {ml_total_return - baseline_total_return:.2f}%")
        
        # Check for realistic values
        if ml_total_return > 100 or baseline_total_return > 100:
            print("WARNING: Returns seem unrealistic (>100%)")
        elif ml_total_return < 0 or baseline_total_return < 0:
            print("WARNING: Negative returns detected")
        else:
            print("Returns look realistic")
            
    except Exception as e:
        print(f"Error testing backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backtest() 