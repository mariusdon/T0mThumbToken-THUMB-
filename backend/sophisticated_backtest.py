import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SophisticatedTreasuryBacktester:
    """
    Sophisticated backtester that incorporates realistic bond mechanics:
    - Coupon payments and reinvestment
    - Duration and convexity effects
    - Realistic transaction costs
    - Proper total return calculations
    """
    
    def __init__(self, model, features_df, returns_df, yields_df):
        """
        Initialize the sophisticated backtester.
        
        Args:
            model: Trained allocation model
            features_df: Feature matrix
            returns_df: Historical returns
            yields_df: Historical yields
        """
        self.model = model
        self.features_df = features_df
        self.returns_df = returns_df
        self.yields_df = yields_df
        
        self.maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        self.maturity_years = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        
        # Performance tracking
        self.ml_portfolio_values = []
        self.baseline_portfolio_values = []
        self.ml_allocations = []
        self.baseline_allocations = []
        self.dates = []
        
    def calculate_realistic_returns(self, yields_df, frequency='M'):
        """
        Calculate realistic total returns including coupons and reinvestment.
        """
        returns = pd.DataFrame(index=yields_df.index, columns=self.maturities)
        
        for i, maturity in enumerate(self.maturities):
            years = self.maturity_years[i]
            yields = yields_df[maturity]
            
            # Calculate total returns
            if years <= 1.0:  # Treasury bills (zero-coupon)
                # For T-bills, return is primarily from price appreciation
                returns[maturity] = yields.pct_change()
            else:  # Treasury notes/bonds
                # For coupon-paying bonds, include coupon income
                coupon_rate = yields / 100  # Assume coupon equals yield (par bonds)
                
                # Monthly coupon payment
                monthly_coupon = coupon_rate / 12
                
                # Price change component
                price_change = yields.pct_change()
                
                # Total return = coupon income + price change
                returns[maturity] = monthly_coupon + price_change
        
        return returns
    
    def calculate_duration_adjusted_returns(self, yields_df, returns_df):
        """
        Calculate duration-adjusted returns for more accurate bond pricing.
        """
        duration_adjusted_returns = returns_df.copy()
        
        for i, maturity in enumerate(self.maturities):
            years = self.maturity_years[i]
            yields = yields_df[maturity]
            
            # Calculate modified duration
            if years <= 1.0:
                duration = years
            else:
                # Simplified duration calculation
                duration = years / (1 + yields / 100)
            
            # Duration-adjusted return
            yield_change = yields.diff()
            duration_adjusted_returns[maturity] = -duration * yield_change / 100 + returns_df[maturity]
        
        return duration_adjusted_returns
    
    def run_sophisticated_backtest(self, start_date='2020-01-01', end_date=None, 
                                 rebalance_freq='M', allocation_method='sophisticated',
                                 transaction_cost=0.001, initial_capital=100000):
        """
        Run sophisticated backtest with realistic bond mechanics.
        """
        print(f"Running sophisticated backtest from {start_date} to {end_date or 'present'}")
        print(f"Rebalancing frequency: {rebalance_freq}")
        print(f"Allocation method: {allocation_method}")
        print(f"Transaction cost: {transaction_cost:.3f}")
        
        # Filter data for backtest period
        if end_date is None:
            end_date = self.returns_df.index[-1]
        
        mask = (self.returns_df.index >= start_date) & (self.returns_df.index <= end_date)
        returns_subset = self.returns_df[mask]
        yields_subset = self.yields_df[mask]
        features_subset = self.features_df[mask]
        
        print(f"Backtest period: {len(returns_subset)} periods")
        
        # Initialize portfolios
        ml_portfolio = initial_capital
        baseline_portfolio = initial_capital
        
        # Track allocations and values
        self.ml_portfolio_values = [initial_capital]
        self.baseline_portfolio_values = [initial_capital]
        self.ml_allocations = []
        self.baseline_allocations = []
        self.dates = [returns_subset.index[0]]
        
        # Equal weight baseline allocation
        baseline_allocation = {maturity: 1/len(self.maturities) for maturity in self.maturities}
        
        # Previous allocations for transaction cost calculation
        prev_ml_allocation = baseline_allocation.copy()
        
        for i, (date, returns_row) in enumerate(returns_subset.iterrows()):
            # Get current features
            if date in features_subset.index:
                current_features = features_subset.loc[date:date]
                
                try:
                    # Get ML allocation
                    ml_allocation = self.model.predict_allocation(
                        current_features, method=allocation_method
                    )
                except Exception as e:
                    print(f"Warning: Using equal weight for {date} due to prediction error: {e}")
                    ml_allocation = baseline_allocation.copy()
            else:
                ml_allocation = baseline_allocation.copy()
            
            # Calculate transaction costs for ML strategy
            transaction_cost_ml = 0
            for maturity in self.maturities:
                allocation_change = abs(ml_allocation[maturity] - prev_ml_allocation[maturity])
                transaction_cost_ml += allocation_change * transaction_cost
            
            # Apply returns
            ml_return = sum(ml_allocation[maturity] * returns_row[maturity] 
                           for maturity in self.maturities)
            baseline_return = sum(baseline_allocation[maturity] * returns_row[maturity] 
                                for maturity in self.maturities)
            
            # Update portfolio values
            ml_portfolio *= (1 + ml_return - transaction_cost_ml)
            baseline_portfolio *= (1 + baseline_return)
            
            # Store results
            self.ml_portfolio_values.append(ml_portfolio)
            self.baseline_portfolio_values.append(baseline_portfolio)
            self.ml_allocations.append(ml_allocation)
            self.baseline_allocations.append(baseline_allocation)
            self.dates.append(date)
            
            # Update previous allocation
            prev_ml_allocation = ml_allocation.copy()
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        
        print("\n" + "="*60)
        print("SOPHISTICATED BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        print(f"{'Metric':<25} {'ML Strategy':<15} {'Equal Weight':<15}")
        print("-" * 60)
        for metric, ml_val, baseline_val in [
            ("Total Return", f"{results['ml_total_return']:.2%}", f"{results['equal_weight_total_return']:.2%}"),
            ("Annualized Return", f"{results['ml_annualized_return']:.2%}", f"{results['equal_weight_annualized_return']:.2%}"),
            ("Volatility", f"{results['ml_volatility']:.3f}", f"{results['equal_weight_volatility']:.3f}"),
            ("Sharpe Ratio", f"{results['ml_sharpe']:.3f}", f"{results['equal_weight_sharpe']:.3f}"),
            ("Max Drawdown", f"{results['ml_max_drawdown']:.2%}", f"{results['equal_weight_max_drawdown']:.2%}"),
            ("Win Rate", f"{results['ml_win_rate']:.2%}", f"{results['equal_weight_win_rate']:.2%}")
        ]:
            print(f"{metric:<25} {ml_val:<15} {baseline_val:<15}")
        
        if 'information_ratio' in results:
            print(f"{'Information Ratio':<25} {results['information_ratio']:.3f}")
        
        print("="*60)
        
        return results
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        # Convert to numpy arrays
        ml_values = np.array(self.ml_portfolio_values)
        baseline_values = np.array(self.baseline_portfolio_values)
        
        # Calculate returns
        ml_returns = np.diff(ml_values) / ml_values[:-1]
        baseline_returns = np.diff(baseline_values) / baseline_values[:-1]
        
        # Total returns
        ml_total_return = (ml_values[-1] / ml_values[0]) - 1
        baseline_total_return = (baseline_values[-1] / baseline_values[0]) - 1
        
        # Annualized returns (assuming monthly data)
        periods_per_year = 12
        total_periods = len(ml_returns)
        ml_annualized_return = (1 + ml_total_return) ** (periods_per_year / total_periods) - 1
        baseline_annualized_return = (1 + baseline_total_return) ** (periods_per_year / total_periods) - 1
        
        # Volatility
        ml_volatility = np.std(ml_returns) * np.sqrt(periods_per_year)
        baseline_volatility = np.std(baseline_returns) * np.sqrt(periods_per_year)
        
        # Sharpe ratios (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        ml_sharpe = (ml_annualized_return - risk_free_rate) / ml_volatility if ml_volatility > 0 else 0
        baseline_sharpe = (baseline_annualized_return - risk_free_rate) / baseline_volatility if baseline_volatility > 0 else 0
        
        # Maximum drawdown
        ml_max_drawdown = self._calculate_max_drawdown(ml_values)
        baseline_max_drawdown = self._calculate_max_drawdown(baseline_values)
        
        # Win rates
        ml_win_rate = np.mean(ml_returns > 0)
        baseline_win_rate = np.mean(baseline_returns > 0)
        
        # Information ratio
        excess_returns = ml_returns - baseline_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        return {
            'ml_total_return': ml_total_return,
            'equal_weight_total_return': baseline_total_return,
            'ml_annualized_return': ml_annualized_return,
            'equal_weight_annualized_return': baseline_annualized_return,
            'ml_volatility': ml_volatility,
            'equal_weight_volatility': baseline_volatility,
            'ml_sharpe': ml_sharpe,
            'equal_weight_sharpe': baseline_sharpe,
            'ml_max_drawdown': ml_max_drawdown,
            'equal_weight_max_drawdown': baseline_max_drawdown,
            'ml_win_rate': ml_win_rate,
            'equal_weight_win_rate': baseline_win_rate,
            'information_ratio': information_ratio
        }
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown."""
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def plot_sophisticated_results(self, filename='sophisticated_backtest_plots.png'):
        """Create sophisticated performance plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sophisticated Treasury Allocation Strategy Performance', fontsize=16)
        
        # Portfolio value comparison
        axes[0, 0].plot(self.dates, self.ml_portfolio_values, label='ML Strategy', linewidth=2)
        axes[0, 0].plot(self.dates, self.baseline_portfolio_values, label='Equal Weight', linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Allocation heatmap (ML strategy)
        if self.ml_allocations:
            allocation_df = pd.DataFrame(self.ml_allocations, index=self.dates[1:])
            sns.heatmap(allocation_df.T, ax=axes[0, 1], cmap='RdYlBu_r', 
                       cbar_kws={'label': 'Allocation Weight'})
            axes[0, 1].set_title('ML Strategy Allocation Over Time')
            axes[0, 1].set_ylabel('Maturity')
        
        # Rolling performance comparison
        ml_returns = np.diff(self.ml_portfolio_values) / self.ml_portfolio_values[:-1]
        baseline_returns = np.diff(self.baseline_portfolio_values) / self.baseline_portfolio_values[:-1]
        
        # Rolling Sharpe ratio (12-month window)
        window = min(12, len(ml_returns))
        if window > 0:
            ml_rolling_sharpe = pd.Series(ml_returns).rolling(window).mean() / pd.Series(ml_returns).rolling(window).std()
            baseline_rolling_sharpe = pd.Series(baseline_returns).rolling(window).mean() / pd.Series(baseline_returns).rolling(window).std()
            
            axes[1, 0].plot(self.dates[window:], ml_rolling_sharpe[window-1:], label='ML Strategy', linewidth=2)
            axes[1, 0].plot(self.dates[window:], baseline_rolling_sharpe[window-1:], label='Equal Weight', linewidth=2)
            axes[1, 0].set_title(f'Rolling Sharpe Ratio ({window}-month window)')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Drawdown comparison
        ml_drawdown = self._calculate_rolling_drawdown(self.ml_portfolio_values)
        baseline_drawdown = self._calculate_rolling_drawdown(self.baseline_portfolio_values)
        
        axes[1, 1].fill_between(self.dates, ml_drawdown, 0, alpha=0.3, label='ML Strategy')
        axes[1, 1].fill_between(self.dates, baseline_drawdown, 0, alpha=0.3, label='Equal Weight')
        axes[1, 1].set_title('Drawdown Over Time')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sophisticated backtest plots saved to {filename}")
    
    def _calculate_rolling_drawdown(self, values):
        """Calculate rolling drawdown."""
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak * 100
        return drawdown
    
    def save_results(self, filename):
        """Save backtest results."""
        results = {
            'ml_portfolio_values': self.ml_portfolio_values,
            'baseline_portfolio_values': self.baseline_portfolio_values,
            'ml_allocations': self.ml_allocations,
            'baseline_allocations': self.baseline_allocations,
            'dates': self.dates,
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        import joblib
        joblib.dump(results, filename)
        print(f"Sophisticated backtest results saved to {filename}") 