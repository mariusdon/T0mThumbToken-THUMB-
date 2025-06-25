import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from allocation_rules import compute_regime_allocation
from ml_model_utils import ml_allocation_model, ml_allocation_scaler
warnings.filterwarnings('ignore')

class EnhancedTreasuryBacktester:
    """
    Enhanced backtesting engine for Treasury allocation strategies.
    Properly identifies yield curve shapes and calculates realistic bond returns
    with price movements and coupon payments.
    """
    
    def __init__(self, model, features_df, returns_df, yield_df):
        """
        Initialize the enhanced backtester.
        
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
        
        # Maturity mapping for coupon calculations
        self.maturity_years = {
            '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, 
            '5Y': 5, '10Y': 10, '30Y': 30
        }
        
        # Store backtest results
        self.results = {}
        
    def identify_curve_shape(self, yields):
        """
        Identify the shape of the yield curve based on current yields.
        
        Args:
            yields (dict): Current yields for each maturity
            
        Returns:
            dict: Curve shape classification and characteristics
        """
        # Calculate key slopes
        slope_10y_2y = yields['10Y'] - yields['2Y']
        slope_30y_3m = yields['30Y'] - yields['3M']
        slope_2y_3m = yields['2Y'] - yields['3M']
        
        # Determine curve shape
        if slope_10y_2y > 0.5 and slope_30y_3m > 1.0:
            shape = 'steep'
            description = 'Steep Curve - Favoring longer maturities for higher yields'
            base_weights = np.array([0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.15])
        elif slope_10y_2y < -0.5 or slope_2y_3m < -0.3:
            shape = 'inverted'
            description = 'Inverted Curve - Favoring shorter maturities for safety'
            base_weights = np.array([0.25, 0.25, 0.20, 0.15, 0.10, 0.03, 0.02])
        elif abs(slope_10y_2y) <= 0.5 and abs(slope_30y_3m) <= 1.0:
            shape = 'flat'
            description = 'Flat Curve - Balanced allocation'
            base_weights = np.array([0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16])
        else:
            shape = 'normal'
            description = 'Normal Curve - Slight preference for intermediate maturities'
            base_weights = np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.15, 0.10])
        
        return {
            'shape': shape,
            'description': description,
            'base_weights': base_weights,
            'slope_10y_2y': slope_10y_2y,
            'slope_30y_3m': slope_30y_3m,
            'slope_2y_3m': slope_2y_3m
        }
    
    def calculate_realistic_bond_returns(self, yield_df, frequency='M'):
        """
        Calculate realistic bond returns including price movements and coupon payments.
        
        Args:
            yield_df (pd.DataFrame): Historical yield data
            frequency (str): Rebalancing frequency
            
        Returns:
            pd.DataFrame: Realistic bond returns
        """
        print("Calculating realistic bond returns with price movements and coupons...")
        
        # Resample to desired frequency
        yield_resampled = yield_df.resample(frequency).last()
        
        # Initialize returns DataFrame
        returns = pd.DataFrame(index=yield_resampled.index, columns=yield_resampled.columns)
        
        for maturity in self.maturity_years.keys():
            if maturity in yield_resampled.columns:
                maturity_years = self.maturity_years[maturity]
                
                # Calculate price changes based on yield changes
                # Bond price ≈ 100 * exp(-yield * maturity)
                prices = 100 * np.exp(-yield_resampled[maturity] * maturity_years)
                
                # Calculate price returns
                price_returns = prices.pct_change()
                
                # Calculate coupon income (annual coupon rate = yield)
                # For simplicity, assume coupon is paid at the frequency
                periods_per_year = {'M': 12, 'W': 52, 'D': 252}.get(frequency, 12)
                coupon_rate = yield_resampled[maturity] / periods_per_year
                
                # Total return = price return + coupon income
                total_returns = price_returns + coupon_rate
                
                returns[maturity] = total_returns
        
        # Fill NaN values with 0
        returns = returns.fillna(0)
        
        print(f"Realistic returns calculated for {len(returns)} periods")
        print("Average annualized returns by maturity:")
        for maturity in returns.columns:
            avg_return = returns[maturity].mean() * periods_per_year
            print(f"  {maturity}: {avg_return:.2%}")
        
        return returns
    
    def run_enhanced_backtest(self, start_date=None, end_date=None, rebalance_freq='M', 
                            allocation_method='curve_aware', transaction_cost=0.001):
        """
        Run enhanced backtest with proper curve shape identification and realistic returns.
        """
        print(f"Running ENHANCED backtest from {start_date} to {end_date}")
        print(f"Rebalancing frequency: {rebalance_freq}")
        print(f"Allocation method: {allocation_method}")
        print(f"Transaction cost: {transaction_cost:.3f}")
        
        # Filter data to backtest period
        if start_date:
            self.features_df = self.features_df[self.features_df.index >= start_date]
            self.yield_df = self.yield_df[self.yield_df.index >= start_date]
        
        if end_date:
            self.features_df = self.features_df[self.features_df.index <= end_date]
            self.yield_df = self.yield_df[self.yield_df.index <= end_date]
        
        # Calculate realistic returns
        realistic_returns = self.calculate_realistic_bond_returns(self.yield_df, rebalance_freq)
        
        # Align all data
        common_index = self.features_df.index.intersection(realistic_returns.index)
        features = self.features_df.loc[common_index]
        returns = realistic_returns.loc[common_index]
        yields = self.yield_df.loc[common_index]
        
        print(f"Backtest period: {len(returns)} periods")
        
        # Initialize portfolio values starting at $100
        initial_value = 100
        ml_portfolio = pd.Series(index=returns.index, dtype=float)
        equal_weight_portfolio = pd.Series(index=returns.index, dtype=float)
        
        # Track allocations and curve shapes over time
        ml_allocations = pd.DataFrame(index=returns.index, columns=self.model.maturity_names)
        equal_weight_allocations = pd.DataFrame(index=returns.index, columns=self.model.maturity_names)
        curve_shapes = pd.Series(index=returns.index, dtype=str)
        
        # Equal weight allocation (constant)
        equal_weight = np.ones(len(self.model.maturity_names)) / len(self.model.maturity_names)
        
        # Initialize portfolios
        ml_portfolio.iloc[0] = initial_value
        equal_weight_portfolio.iloc[0] = initial_value
        
        # Store initial allocations
        ml_allocations.iloc[0] = equal_weight
        equal_weight_allocations.iloc[0] = equal_weight
        curve_shapes.iloc[0] = 'initial'
        
        print("Running enhanced backtest simulation...")
        
        # Run backtest
        for i in range(1, len(returns)):
            current_date = returns.index[i]
            prev_date = returns.index[i-1]
            
            # Get current returns and yields
            current_returns = returns.iloc[i]
            current_yields = yields.iloc[i]
            
            # Identify current curve shape
            curve_info = self.identify_curve_shape(current_yields)
            curve_shapes.iloc[i] = curve_info['shape']
            
            # Baseline: always equal weight
            equal_weight_portfolio.iloc[i] = equal_weight_portfolio.iloc[i-1] * (1 + np.sum(equal_weight * current_returns))
            equal_weight_allocations.iloc[i] = equal_weight
            
            # ML: use ML model if available, else regime rule
            try:
                # Prepare features: [3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y, CPI, UNEMP, MOVE]
                # Assume macro features are available in features DataFrame
                feature_row = features.iloc[i-1] if i-1 < len(features) else None
                if feature_row is not None and ml_allocation_model is not None and ml_allocation_scaler is not None:
                    features_arr = np.array(feature_row).reshape(1, -1)
                    features_scaled = ml_allocation_scaler.transform(features_arr)
                    ml_weights = ml_allocation_model.predict(features_scaled)[0]
                    ml_weights = np.minimum(ml_weights, 0.4)
                    ml_weights = ml_weights / np.sum(ml_weights)
                else:
                    # Fallback to regime rule
                    yield_list = [current_yields['3M'], current_yields['6M'], current_yields['1Y'], current_yields['2Y'], current_yields['5Y'], current_yields['10Y'], current_yields['30Y']]
                    ml_weights, _ = compute_regime_allocation(yield_list)
                    ml_weights = np.array(ml_weights)
            except Exception as e:
                print(f"Warning: Using regime rule for ML allocation at {current_date} due to error: {e}")
                yield_list = [current_yields['3M'], current_yields['6M'], current_yields['1Y'], current_yields['2Y'], current_yields['5Y'], current_yields['10Y'], current_yields['30Y']]
                ml_weights, _ = compute_regime_allocation(yield_list)
                ml_weights = np.array(ml_weights)
            ml_weights = ml_weights / np.sum(ml_weights)
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
        self.results = self._calculate_enhanced_metrics(
            ml_portfolio, equal_weight_portfolio, returns, curve_shapes
        )
        
        # Store detailed results
        self.results['ml_portfolio'] = ml_portfolio
        self.results['equal_weight_portfolio'] = equal_weight_portfolio
        self.results['ml_allocations'] = ml_allocations
        self.results['equal_weight_allocations'] = equal_weight_allocations
        self.results['returns'] = returns
        self.results['curve_shapes'] = curve_shapes
        self.results['yields'] = yields
        
        print("Enhanced backtest completed!")
        self._print_enhanced_summary()
        
        # After backtest, save results to CSV for frontend
        results_df = pd.DataFrame({
            'date': ml_portfolio.index.strftime('%Y-%m-%d'),
            'ml_value': ml_portfolio.values,
            'equal_weight_value': equal_weight_portfolio.values
        })
        results_df.to_csv('backtest_results.csv', index=False)
        print('Saved backtest results to backtest_results.csv')
        
        return self.results
    
    def _calculate_enhanced_metrics(self, ml_portfolio, equal_weight_portfolio, returns, curve_shapes):
        """Calculate comprehensive performance metrics with curve shape analysis."""
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
        
        # Curve shape analysis
        curve_shape_counts = curve_shapes.value_counts()
        metrics['curve_shape_distribution'] = curve_shape_counts.to_dict()
        
        # Performance by curve shape
        shape_performance = {}
        for shape in ['steep', 'inverted', 'flat', 'normal']:
            shape_mask = curve_shapes == shape
            if shape_mask.sum() > 0:
                shape_returns = ml_returns[shape_mask]
                shape_performance[shape] = {
                    'periods': shape_mask.sum(),
                    'avg_return': shape_returns.mean(),
                    'volatility': shape_returns.std()
                }
        
        metrics['performance_by_curve_shape'] = shape_performance
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio):
        """Calculate maximum drawdown."""
        peak = portfolio.expanding().max()
        drawdown = (portfolio - peak) / peak
        return drawdown.min()
    
    def _print_enhanced_summary(self):
        """Print enhanced backtest performance summary."""
        print("\n" + "="*70)
        print("ENHANCED BACKTEST PERFORMANCE SUMMARY")
        print("="*70)
        
        print(f"{'Metric':<30} {'ML Strategy':<20} {'Equal Weight':<20}")
        print("-" * 70)
        
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
            
            if 'return' in name.lower() or 'drawdown' in name.lower() or 'rate' in name.lower():
                print(f"{name:<30} {ml_val:>18.2%} {eq_val:>18.2%}")
            else:
                print(f"{name:<30} {ml_val:>18.3f} {eq_val:>18.3f}")
        
        print(f"{'Information Ratio':<30} {self.results['information_ratio']:>18.3f}")
        
        print("\n" + "="*70)
        print("CURVE SHAPE ANALYSIS")
        print("="*70)
        
        shape_dist = self.results['curve_shape_distribution']
        for shape, count in shape_dist.items():
            percentage = count / sum(shape_dist.values()) * 100
            print(f"{shape.upper():<15}: {count:>5} periods ({percentage:>5.1f}%)")
        
        print("\nPerformance by Curve Shape:")
        shape_perf = self.results['performance_by_curve_shape']
        for shape, perf in shape_perf.items():
            print(f"{shape.upper():<15}: {perf['periods']:>3} periods, "
                  f"Avg Return: {perf['avg_return']*12:.2%}, "
                  f"Vol: {perf['volatility']*np.sqrt(12):.2%}")
        
        print("="*70)
    
    def plot_enhanced_results(self, save_path=None):
        """Plot enhanced backtest results with curve shape analysis."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Enhanced Treasury Allocation Strategy Backtest Results', fontsize=16)
        
        # Portfolio value comparison
        axes[0, 0].plot(self.results['ml_portfolio'].index, self.results['ml_portfolio'].values, 
                       label='ML Strategy', linewidth=2, color='blue')
        axes[0, 0].plot(self.results['equal_weight_portfolio'].index, self.results['equal_weight_portfolio'].values, 
                       label='Equal Weight', linewidth=2, color='red', alpha=0.7)
        axes[0, 0].set_title('Portfolio Value Over Time (Starting at $100)')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        ml_cumulative = (self.results['ml_portfolio'] / self.results['ml_portfolio'].iloc[0]) - 1
        eq_cumulative = (self.results['equal_weight_portfolio'] / self.results['equal_weight_portfolio'].iloc[0]) - 1
        
        axes[0, 1].plot(ml_cumulative.index, ml_cumulative.values * 100, 
                       label='ML Strategy', linewidth=2, color='blue')
        axes[0, 1].plot(eq_cumulative.index, eq_cumulative.values * 100, 
                       label='Equal Weight', linewidth=2, color='red', alpha=0.7)
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
        
        # Curve shape over time
        curve_shapes = self.results['curve_shapes']
        shape_colors = {'steep': 'green', 'inverted': 'red', 'flat': 'yellow', 'normal': 'blue'}
        for shape in shape_colors.keys():
            mask = curve_shapes == shape
            if mask.sum() > 0:
                axes[1, 1].scatter(curve_shapes[mask].index, [shape] * mask.sum(), 
                                 c=shape_colors[shape], label=shape, alpha=0.7)
        axes[1, 1].set_title('Yield Curve Shape Over Time')
        axes[1, 1].set_ylabel('Curve Shape')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
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
        
        axes[2, 0].bar(x - width/2, ml_values, width, label='ML Strategy', alpha=0.8, color='blue')
        axes[2, 0].bar(x + width/2, eq_values, width, label='Equal Weight', alpha=0.8, color='red')
        axes[2, 0].set_title('Performance Metrics Comparison')
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels(metrics)
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Yield curve evolution
        yields = self.results['yields']
        sample_dates = yields.index[::len(yields)//10]  # Sample 10 dates
        for date in sample_dates:
            if date in yields.index:
                curve = yields.loc[date]
                axes[2, 1].plot(list(self.maturity_years.values()), curve.values, 
                              marker='o', label=date.strftime('%Y-%m'), alpha=0.7)
        axes[2, 1].set_title('Yield Curve Evolution')
        axes[2, 1].set_xlabel('Maturity (Years)')
        axes[2, 1].set_ylabel('Yield (%)')
        axes[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced backtest plots saved to {save_path}")
        
        plt.show()
    
    def save_enhanced_results(self, filepath):
        """Save enhanced backtest results."""
        # Convert DataFrames to dict for JSON serialization
        results_to_save = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                results_to_save[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                results_to_save[key] = value.to_dict()
            else:
                results_to_save[key] = value
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        print(f"Enhanced backtest results saved to {filepath}") 