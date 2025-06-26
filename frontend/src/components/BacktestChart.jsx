import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const BacktestChart = () => {
  const [backtestData, setBacktestData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [showTotalReturn, setShowTotalReturn] = useState(true);
  const [includesCoupons, setIncludesCoupons] = useState(false);

  useEffect(() => {
    const fetchBacktestData = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch('http://127.0.0.1:8000/backtest-data-enhanced');
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const data = await response.json();
        // Defensive: check for expected structure
        if (!data || !Array.isArray(data.date) || !Array.isArray(data.ml_value) || !Array.isArray(data.baseline_value)) {
          setError('No backtest data available from backend.');
          setBacktestData(null);
          return;
        }
        // Build chartData from arrays
        const chartData = data.date.map((date, idx) => ({
          date,
          'ML Strategy': data.ml_value[idx],
          'Equal Weight Baseline': data.baseline_value[idx],
        }));
        setBacktestData({ ...data, chartData });
        // Set metrics
        setMetrics({
          ml_total_return: data.ml_total_return,
          equal_weight_total_return: data.baseline_total_return,
          ml_annualized_return: data.ml_annualized_return,
          equal_weight_annualized_return: data.baseline_annualized_return,
          ml_volatility: data.ml_volatility,
          equal_weight_volatility: data.baseline_volatility,
          ml_sharpe: data.ml_sharpe,
          equal_weight_sharpe: data.baseline_sharpe,
          alpha: data.alpha,
        });
        setIncludesCoupons(data.includes_coupons || false);
      } catch (err) {
        console.error('Error fetching backtest data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchBacktestData();
  }, []);

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-center h-[400px]">
          <div className="text-lg text-gray-600">Loading backtest data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex flex-col items-center justify-center h-[400px]">
          <div className="text-lg text-red-600 mb-2">Error loading backtest data</div>
          <div className="text-sm text-gray-600">{error}</div>
          <button 
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            onClick={() => window.location.reload()}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!backtestData || !Array.isArray(backtestData.chartData) || backtestData.chartData.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-center h-[400px]">
          <div className="text-lg text-gray-600">No backtest data available</div>
        </div>
      </div>
    );
  }

  // Use the new chartData
  const chartData = backtestData.chartData;

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Historical Backtest Performance</h2>
        
        {includesCoupons && (
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="totalReturn"
                checked={showTotalReturn}
                onChange={(e) => setShowTotalReturn(e.target.checked)}
                className="rounded border-gray-300 bg-white text-blue-600 focus:ring-blue-500"
              />
              <label htmlFor="totalReturn" className="text-sm text-gray-700">
                Show Total Return (with coupons)
              </label>
            </div>
            <div className="text-xs text-green-700 bg-green-100 px-2 py-1 rounded">
              ✓ Includes Real Historical Coupons (from FRED)
            </div>
          </div>
        )}
      </div>
      
      <div className="h-[400px] mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis 
              dataKey="date" 
              stroke="#6b7280"
              tick={{ fill: '#6b7280' }}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis 
              stroke="#6b7280"
              tick={{ fill: '#6b7280' }}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
              domain={[90, 120]}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#ffffff', 
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                color: '#374151'
              }}
              labelFormatter={(value) => new Date(value).toLocaleDateString()}
              formatter={(value, name) => [`$${value.toFixed(2)}`, name]}
            />
            <Legend 
              wrapperStyle={{ color: '#374151' }}
            />
            <Line 
              type="monotone" 
              dataKey="ML Strategy" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="Equal Weight Baseline" 
              stroke="#ef4444" 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-gray-50 p-4 rounded-lg border">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Total Return</h3>
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-sm text-blue-600">ML Strategy:</span>
                <span className="text-sm font-semibold">{metrics.ml_total_return !== undefined ? metrics.ml_total_return.toFixed(2) : 'N/A'}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-red-600">Baseline:</span>
                <span className="text-sm font-semibold">{metrics.equal_weight_total_return !== undefined ? metrics.equal_weight_total_return.toFixed(2) : 'N/A'}%</span>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {metrics.ml_total_return !== undefined && metrics.equal_weight_total_return !== undefined ? 
                  `Difference: ${(metrics.ml_total_return - metrics.equal_weight_total_return).toFixed(2)}%` : ''}
              </div>
            </div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg border">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Annualized Return (CAGR)</h3>
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-sm text-blue-600">ML Strategy:</span>
                <span className="text-sm font-semibold">{metrics.ml_annualized_return !== undefined ? metrics.ml_annualized_return.toFixed(2) : 'N/A'}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-red-600">Baseline:</span>
                <span className="text-sm font-semibold">{metrics.equal_weight_annualized_return !== undefined ? metrics.equal_weight_annualized_return.toFixed(2) : 'N/A'}%</span>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {metrics.ml_annualized_return !== undefined && metrics.equal_weight_annualized_return !== undefined ? 
                  `Difference: ${(metrics.ml_annualized_return - metrics.equal_weight_annualized_return).toFixed(2)}%` : ''}
              </div>
            </div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg border">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Volatility (Annualized)</h3>
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-sm text-blue-600">ML Strategy:</span>
                <span className="text-sm font-semibold">{metrics.ml_volatility !== undefined ? metrics.ml_volatility.toFixed(2) : 'N/A'}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-red-600">Baseline:</span>
                <span className="text-sm font-semibold">{metrics.equal_weight_volatility !== undefined ? metrics.equal_weight_volatility.toFixed(2) : 'N/A'}%</span>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {metrics.ml_volatility !== undefined && metrics.equal_weight_volatility !== undefined ? 
                  `Difference: ${(metrics.ml_volatility - metrics.equal_weight_volatility).toFixed(2)}%` : ''}
              </div>
            </div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg border">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Sharpe Ratio</h3>
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-sm text-blue-600">ML Strategy:</span>
                <span className="text-sm font-semibold">{metrics.ml_sharpe !== undefined ? metrics.ml_sharpe.toFixed(2) : 'N/A'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-red-600">Baseline:</span>
                <span className="text-sm font-semibold">{metrics.equal_weight_sharpe !== undefined ? metrics.equal_weight_sharpe.toFixed(2) : 'N/A'}</span>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {metrics.ml_sharpe !== undefined && metrics.equal_weight_sharpe !== undefined ? 
                  `Difference: ${(metrics.ml_sharpe - metrics.equal_weight_sharpe).toFixed(2)}` : ''}
              </div>
            </div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg border">
            <h3 className="text-sm font-medium text-gray-700 mb-1">CAPM Alpha (Annualized)</h3>
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-sm text-blue-600">ML Strategy:</span>
                <span className={`text-sm font-semibold ${metrics.alpha !== undefined && metrics.alpha >= 0 ? 'text-green-600' : 'text-red-600'}`}>{metrics.alpha !== undefined ? metrics.alpha.toFixed(2) : 'N/A'}%</span>
              </div>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Annualized excess return vs. CAPM expectation (backend)
            </div>
          </div>
        </div>
      )}
      
      {/* Coupon Income Metrics */}
      {includesCoupons && metrics.ml_coupon_income && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-gray-50 p-4 rounded-lg border">
            <h3 className="text-sm font-medium text-gray-700 mb-1">ML Strategy Coupon Income</h3>
            <div className="text-lg font-bold text-gray-900">{metrics.ml_coupon_income} THUMB</div>
            <div className="text-xs text-gray-500">Cumulative coupon payments</div>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg border">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Baseline Coupon Income</h3>
            <div className="text-lg font-bold text-gray-900">{metrics.equal_weight_coupon_income} THUMB</div>
            <div className="text-xs text-gray-500">Cumulative coupon payments</div>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg border">
            <h3 className="text-sm font-medium text-gray-700 mb-1">Coupon Outperformance</h3>
            <div className="flex justify-between items-center">
              <span className="text-lg font-bold text-gray-900">{metrics.coupon_outperformance ? (metrics.coupon_outperformance * 100).toFixed(2) : 'N/A'}%</span>
              <span className={`text-sm ${parseFloat(metrics.coupon_outperformance) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {parseFloat(metrics.coupon_outperformance) >= 0 ? 'Higher' : 'Lower'}
              </span>
            </div>
            <div className="text-xs text-gray-500">vs baseline coupon income</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BacktestChart; 