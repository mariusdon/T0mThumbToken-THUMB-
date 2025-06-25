import React from 'react';

const Methodology = () => {
  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h2 className="text-xl font-semibold mb-4 text-gray-900">Allocation Methodology</h2>
      
      {/* Purpose Blurb */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">Project Purpose</h3>
        <p className="text-blue-800 text-sm leading-relaxed">
          T0m Thumb is a DeFi experiment that shows how your cash can go to work on-chain instead of sitting in a bank. 
          In an age with increasing stablecoins, by tokenizing the U.S. Treasury yield curve and pairing it with an AI-driven 
          allocation engine, we're building a more liquid, transparent fixed-income market. T0m Thumb anticipates a future 
          where smart algorithms and machine learning are trusted internationally, democratizing institutional-grade bond 
          strategies and yield-bearing currency for everyday users.
        </p>
      </div>
      
      <div className="space-y-6">
        {/* Overview */}
        <div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Strategy Overview</h3>
          <p className="text-gray-600 text-sm leading-relaxed">
            The THUMB Vault employs a machine learning-driven approach to optimize U.S. Treasury bond allocations 
            based on real-time yield curve dynamics and macroeconomic indicators. Our strategy adapts to different 
            market regimes to maximize risk-adjusted returns while maintaining diversification across maturities.
          </p>
        </div>

        {/* Data Sources */}
        <div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Real-Time Data Sources</h3>
          <div className="bg-blue-50 p-4 rounded-lg">
            <ul className="text-blue-700 text-sm space-y-2">
              <li>• <strong>FRED API:</strong> Live Treasury yield curve data (3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y)</li>
              <li>• <strong>Economic Indicators:</strong> CPI (Consumer Price Index) and Unemployment rate</li>
              <li>• <strong>Market Volatility:</strong> MOVE index for Treasury volatility</li>
              <li>• <strong>Update Frequency:</strong> Daily market data updates</li>
            </ul>
          </div>
        </div>

        {/* Yield Curve Regimes */}
        <div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Yield Curve Regimes</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-medium text-blue-800 mb-2">Steep Curve</h4>
              <p className="text-blue-700 text-sm">
                10Y-2Y spread &gt; 0.5%. Favors longer maturities (5Y, 10Y, 30Y) to capture higher yields 
                and benefit from positive carry.
              </p>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg">
              <h4 className="font-medium text-yellow-800 mb-2">Flat Curve</h4>
              <p className="text-yellow-700 text-sm">
                10Y-2Y spread between -0.5% and 0.5%. Balanced allocation across all maturities 
                with slight overweight to intermediate bonds.
              </p>
            </div>
            <div className="bg-red-50 p-4 rounded-lg">
              <h4 className="font-medium text-red-800 mb-2">Inverted Curve</h4>
              <p className="text-red-700 text-sm">
                10Y-2Y spread &lt; -0.5%. Favors shorter maturities (3M, 6M, 1Y) for safety 
                and to avoid duration risk during economic stress.
              </p>
            </div>
          </div>
        </div>

        {/* ML Model */}
        <div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Machine Learning Model</h3>
          <div className="bg-gray-50 p-4 rounded-lg">
            <ul className="text-gray-700 text-sm space-y-2">
              <li>• <strong>Input Features:</strong> 7 Treasury yields + CPI + Unemployment + MOVE + regime indicators</li>
              <li>• <strong>Model Architecture:</strong> Neural Network with ReLU activation and dropout</li>
              <li>• <strong>Training Data:</strong> Historical Treasury data from 2020-2024 with regime-aware labels</li>
              <li>• <strong>Output:</strong> Optimal weight allocation across 7 maturity buckets (3M to 30Y)</li>
              <li>• <strong>Rebalancing:</strong> Dynamic allocation based on real-time market conditions</li>
              <li>• <strong>Fallback:</strong> Regime-based rules when ML model is unavailable</li>
            </ul>
          </div>
        </div>

        {/* Risk Management */}
        <div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Risk Management</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="font-medium text-green-800 mb-2">Diversification</h4>
              <p className="text-green-700 text-sm">
                Spreads risk across multiple maturities to reduce concentration risk. 
                Minimum 5% allocation per maturity, maximum 40% per maturity.
              </p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <h4 className="font-medium text-purple-800 mb-2">Regime Adaptation</h4>
              <p className="text-purple-700 text-sm">
                Dynamically adjusts allocation based on current market conditions and 
                yield curve shape to optimize risk-adjusted returns.
              </p>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg">
              <h4 className="font-medium text-orange-800 mb-2">Liquidity Management</h4>
              <p className="text-orange-700 text-sm">
                Maintains adequate allocation to short-term Treasuries for liquidity 
                and capital preservation during market stress.
              </p>
            </div>
            <div className="bg-indigo-50 p-4 rounded-lg">
              <h4 className="font-medium text-indigo-800 mb-2">Duration Control</h4>
              <p className="text-indigo-700 text-sm">
                Actively manages portfolio duration based on yield curve expectations 
                and interest rate outlook.
              </p>
            </div>
          </div>
        </div>

        {/* Smart Contract Integration */}
        <div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Smart Contract Integration</h3>
          <div className="bg-gray-50 p-4 rounded-lg">
            <ul className="text-gray-700 text-sm space-y-2">
              <li>• <strong>ERC-4626 Vault:</strong> Standardized vault interface for seamless DeFi integration</li>
              <li>• <strong>Automated Rebalancing:</strong> ML allocations can be applied directly to the vault</li>
              <li>• <strong>Transparent Allocation:</strong> All vault allocations are visible on-chain</li>
              <li>• <strong>Gas Optimization:</strong> Efficient rebalancing with minimal transaction costs</li>
            </ul>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h4 className="font-medium text-yellow-800 mb-2">Important Disclaimer</h4>
          <p className="text-yellow-700 text-sm">
            This is a simulation project for educational purposes. Past performance does not guarantee future results. 
            The ML model is trained on historical data and may not perform similarly in different market conditions. 
            Treasury bond investments carry interest rate risk and market risk. Always conduct your own research 
            before making investment decisions. This is not financial advice.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Methodology; 