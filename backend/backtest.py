import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TreasuryBacktester:
    """
    Backtesting engine for Treasury allocation strategies.
    Compares ML strategy against equal-weight and other benchmarks.
    """
    
    def __init__(self, model, features_df, returns_df, yield_df):
        """
        Initialize the backtester.
        
        Args:
            model: Trained TreasuryAllocationModel
            features_df (pd.DataFrame): Historical features
            returns_df (pd.DataFrame): Historical returns
            yield_df (pd.DataFrame): Historical yields
        """
        self.model = model
        self.features_df = features_df
        self.returns_df = returns_df
        self.yield_df = yield_df
        
        # Store backtest results
        self.results = {}
        
    def run_backtest(self, start_date=None, end_date=None, rebalance_freq='M', 
                    allocation_method='curve_aware', transaction_cost=0.001):
        """
        Run backtest of ML strategy vs benchmarks.
        ML strategy: dynamic allocation from model
        Baseline: static equal weight
        """
        print(f"Running backtest from {start_date} to {end_date}")
        print(f"Rebalancing frequency: {rebalance_freq}")
        print(f"ML allocation: dynamic (model-based)")
        print(f"Baseline allocation: static (equal weight)")
        print(f"Transaction cost: {transaction_cost:.3f}")
        
        # Filter data to backtest period
        if start_date:
            self.features_df = self.features_df[self.features_df.index >= start_date]
            self.returns_df = self.returns_df[self.returns_df.index >= start_date]
            self.yield_df = self.yield_df[self.yield_df.index >= start_date]
        
        if end_date:
            self.features_df = self.features_df[self.features_df.index <= end_date]
            self.returns_df = self.returns_df[self.returns_df.index <= end_date]
            self.yield_df = self.yield_df[self.yield_df.index <= end_date]
        
        # Align all data
        common_index = self.features_df.index.intersection(self.returns_df.index)
        features = self.features_df.loc[common_index]
        returns = self.returns_df.loc[common_index]
        
        print(f"Backtest period: {len(returns)} periods")
        print(f"Average returns by maturity:")
        for maturity in returns.columns:
            avg_return = returns[maturity].mean()
            print(f"  {maturity}: {avg_return:.4f} per period ({avg_return*12:.2%} annualized)")
        
        # Initialize portfolio values
        initial_value = 100
        ml_portfolio = pd.Series(index=returns.index, dtype=float)
        equal_weight_portfolio = pd.Series(index=returns.index, dtype=float)
        
        # Track allocations over time
        ml_allocations = pd.DataFrame(index=returns.index, columns=self.model.maturity_names)
        equal_weight_allocations = pd.DataFrame(index=returns.index, columns=self.model.maturity_names)
        
        # Equal weight allocation (constant)
        equal_weight = np.ones(len(self.model.maturity_names)) / len(self.model.maturity_names)
        
        # Initialize portfolios
        ml_portfolio.iloc[0] = initial_value
        equal_weight_portfolio.iloc[0] = initial_value
        
        # Store initial allocations
        ml_allocations.iloc[0] = equal_weight  # Start with equal weight
        equal_weight_allocations.iloc[0] = equal_weight
        
        print("Running backtest simulation...")
        
        # Run backtest
        for i in range(1, len(returns)):
            current_date = returns.index[i]
            prev_date = returns.index[i-1]
            
            # Get current returns
            current_returns = returns.iloc[i]
            
            # Baseline: always equal weight
            equal_weight_portfolio.iloc[i] = equal_weight_portfolio.iloc[i-1] * (1 + np.sum(equal_weight * current_returns))
            equal_weight_allocations.iloc[i] = equal_weight
            
            # ML: dynamic allocation from model
            # Use previous period's features to predict this period's allocation
            current_features = features.iloc[i-1]  # Use previous period's features
            try:
                ml_allocation = self.model.predict_allocation(
                    current_features, method=allocation_method
                )
                ml_weights = np.array(list(ml_allocation.values()))
            except Exception as e:
                print(f"Warning: Using equal weight for {current_date} due to prediction error: {e}")
                ml_weights = equal_weight
            # Apply transaction costs
            prev_weights = ml_allocations.iloc[i-1].values
            weight_change = np.abs(ml_weights - prev_weights)
            transaction_cost_total = np.sum(weight_change) * transaction_cost
            # Calculate portfolio return
            portfolio_return = np.sum(ml_weights * current_returns) - transaction_cost_total
            ml_portfolio.iloc[i] = ml_portfolio.iloc[i-1] * (1 + portfolio_return)
            # Store allocation
            ml_allocations.iloc[i] = ml_weights
        
        # Calculate performance metrics
        self.results = self._calculate_performance_metrics(
            ml_portfolio, equal_weight_portfolio, returns
        )
        
        # Store detailed results
        self.results['ml_portfolio'] = ml_portfolio
        self.results['equal_weight_portfolio'] = equal_weight_portfolio
        self.results['ml_allocations'] = ml_allocations
        self.results['equal_weight_allocations'] = equal_weight_allocations
        self.results['returns'] = returns
        
        print("Backtest completed!")
        self._print_backtest_summary()
        
        return self.results
    
    def _calculate_performance_metrics(self, ml_portfolio, equal_weight_portfolio, returns):
        """Calculate comprehensive performance metrics."""
        # Calculate returns
        ml_returns = ml_portfolio.pct_change().dropna()
        equal_weight_returns = equal_weight_portfolio.pct_change().dropna()
        
        # Risk-free rate (approximate)
        risk_free_rate = 0.02 / 12  # 2% annual, monthly
        
        metrics = {}
        
        # Total return
        metrics['ml_total_return'] = (ml_portfolio.iloc[-1] / ml_portfolio.iloc[0]) - 1
        metrics['equal_weight_total_return'] = (equal_weight_portfolio.iloc[-1] / equal_weight_portfolio.iloc[0]) - 1
        
        # Annualized return
        years = len(returns) / 12  # Assuming monthly data
        metrics['ml_annualized_return'] = (1 + metrics['ml_total_return']) ** (1/years) - 1
        metrics['equal_weight_annualized_return'] = (1 + metrics['equal_weight_total_return']) ** (1/years) - 1
        
        # Volatility
        metrics['ml_volatility'] = ml_returns.std() * np.sqrt(12)  # Annualized
        metrics['equal_weight_volatility'] = equal_weight_returns.std() * np.sqrt(12)
        
        # Sharpe ratio
        metrics['ml_sharpe'] = (metrics['ml_annualized_return'] - 0.02) / metrics['ml_volatility']
        metrics['equal_weight_sharpe'] = (metrics['equal_weight_annualized_return'] - 0.02) / metrics['equal_weight_volatility']
        
        # Maximum drawdown
        metrics['ml_max_drawdown'] = self._calculate_max_drawdown(ml_portfolio)
        metrics['equal_weight_max_drawdown'] = self._calculate_max_drawdown(equal_weight_portfolio)
        
        # Information ratio
        excess_returns = ml_returns - equal_weight_returns
        metrics['information_ratio'] = excess_returns.mean() / excess_returns.std()
        
        # Win rate
        metrics['ml_win_rate'] = (ml_returns > 0).mean()
        metrics['equal_weight_win_rate'] = (equal_weight_returns > 0).mean()
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio):
        """Calculate maximum drawdown."""
        peak = portfolio.expanding().max()
        drawdown = (portfolio - peak) / peak
        return drawdown.min()
    
    def _print_backtest_summary(self):
        """Print backtest performance summary."""
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"{'Metric':<25} {'ML Strategy':<15} {'Equal Weight':<15}")
        print("-" * 60)
        
        metrics = [
            ('Total Return', 'ml_total_return', 'equal_weight_total_return'),
            ('Annualized Return', 'ml_annualized_return', 'equal_weight_annualized_return'),
            ('Volatility', 'ml_volatility', 'equal_weight_volatility'),
            ('Sharpe Ratio', 'ml_sharpe', 'equal_weight_sharpe'),
            ('Max Drawdown', 'ml_max_drawdown', 'equal_weight_max_drawdown'),
            ('Win Rate', 'ml_win_rate', 'equal_weight_win_rate')
        ]
        
        for name, ml_key, eq_key in metrics:
            ml_val = self.results[ml_key]
            eq_val = self.results[eq_key]
            
            if 'return' in name.lower():
                print(f"{name:<25} {ml_val:>14.2%} {eq_val:>14.2%}")
            elif 'drawdown' in name.lower():
                print(f"{name:<25} {ml_val:>14.2%} {eq_val:>14.2%}")
            elif 'rate' in name.lower():
                print(f"{name:<25} {ml_val:>14.2%} {eq_val:>14.2%}")
            else:
                print(f"{name:<25} {ml_val:>14.3f} {eq_val:>14.3f}")
        
        print(f"{'Information Ratio':<25} {self.results['information_ratio']:>14.3f}")
        print("="*60)
    
    def plot_results(self, save_path=None):
        """Plot backtest results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Treasury Allocation Strategy Backtest Results', fontsize=16)
        
        # Portfolio value comparison
        axes[0, 0].plot(self.results['ml_portfolio'].index, self.results['ml_portfolio'].values, 
                       label='ML Strategy', linewidth=2)
        axes[0, 0].plot(self.results['equal_weight_portfolio'].index, self.results['equal_weight_portfolio'].values, 
                       label='Equal Weight', linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        ml_cumulative = (self.results['ml_portfolio'] / self.results['ml_portfolio'].iloc[0]) - 1
        eq_cumulative = (self.results['equal_weight_portfolio'] / self.results['equal_weight_portfolio'].iloc[0]) - 1
        
        axes[0, 1].plot(ml_cumulative.index, ml_cumulative.values * 100, 
                       label='ML Strategy', linewidth=2)
        axes[0, 1].plot(eq_cumulative.index, eq_cumulative.values * 100, 
                       label='Equal Weight', linewidth=2)
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_ylabel('Cumulative Return (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # ML allocation weights over time
        allocation_data = self.results['ml_allocations']
        for maturity in self.model.maturity_names:
            axes[1, 0].plot(allocation_data.index, allocation_data[maturity] * 100, 
                           label=maturity, linewidth=1.5)
        axes[1, 0].set_title('ML Strategy Allocation Weights')
        axes[1, 0].set_ylabel('Allocation Weight (%)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance comparison bar chart
        metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
        ml_values = [
            self.results['ml_total_return'],
            self.results['ml_sharpe'],
            self.results['ml_max_drawdown']
        ]
        eq_values = [
            self.results['equal_weight_total_return'],
            self.results['equal_weight_sharpe'],
            self.results['equal_weight_max_drawdown']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, ml_values, width, label='ML Strategy', alpha=0.8)
        axes[1, 1].bar(x + width/2, eq_values, width, label='Equal Weight', alpha=0.8)
        axes[1, 1].set_title('Performance Metrics Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Backtest plots saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath):
        """Save backtest results to file."""
        # Convert DataFrames to CSV
        self.results['ml_portfolio'].to_csv(filepath.replace('.pkl', '_ml_portfolio.csv'))
        self.results['equal_weight_portfolio'].to_csv(filepath.replace('.pkl', '_equal_weight_portfolio.csv'))
        self.results['ml_allocations'].to_csv(filepath.replace('.pkl', '_ml_allocations.csv'))
        
        # Save metrics
        import joblib
        metrics_only = {k: v for k, v in self.results.items() 
                       if not isinstance(v, (pd.DataFrame, pd.Series))}
        joblib.dump(metrics_only, filepath)
        print(f"Backtest results saved to {filepath}")

if __name__ == "__main__":
    # Test the backtester
    from data_loader import TreasuryDataLoader
    from model import TreasuryAllocationModel
    
    # Load data and train model
    loader = TreasuryDataLoader()
    features, returns, yields = loader.prepare_training_data()
    
    model = TreasuryAllocationModel(n_estimators=50, max_depth=8)
    model.train(features, returns)
    
    # Run backtest
    backtester = TreasuryBacktester(model, features, returns, yields)
    results = backtester.run_backtest(start_date='2020-01-01')
    
    # Plot results
    backtester.plot_results() 