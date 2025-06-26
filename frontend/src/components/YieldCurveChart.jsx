import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y'];

const YieldCurveChart = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchYieldCurveData = async () => {
      try {
        setLoading(true);
        console.log('Fetching yield curve data...');
        
        const response = await fetch('http://127.0.0.1:8000/optimal-allocation');
        if (!response.ok) {
          const errorText = await response.text();
          console.error('Server error:', response.status, errorText);
          throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Received data:', result);
        
        if (result.yields && Object.keys(result.yields).length > 0) {
          let chartData = [];
          if (Array.isArray(result.yields)) {
            // Map the list to the correct maturities
            const allMaturities = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'];
            const yieldMap = {};
            allMaturities.forEach((mat, idx) => {
              yieldMap[mat] = result.yields[idx];
            });
            chartData = maturities.map(maturity => ({
              maturity,
              yield: yieldMap[maturity] ?? 0
            }));
          } else if (typeof result.yields === 'object') {
            chartData = maturities.map(maturity => ({
              maturity,
              yield: result.yields[maturity] ?? 0
            }));
          }
          setData(chartData);
          setError(null);
        } else {
          console.error('No yield data in response:', result);
          setError('No yield data available');
        }
      } catch (e) {
        console.error('Error fetching yield curve:', e);
        setError(`Failed to fetch yield curve: ${e.message}`);
      } finally {
        setLoading(false);
      }
    };

    fetchYieldCurveData();
    // Refresh every 5 minutes
    const interval = setInterval(fetchYieldCurveData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return (
    <div className="bg-white rounded-lg shadow-sm border p-6 flex flex-col items-center">
      <div className="text-gray-600">Loading yield curve...</div>
    </div>
  );

  if (error) return (
    <div className="bg-white rounded-lg shadow-sm border p-6 flex flex-col items-center">
      <div className="text-red-500">{error}</div>
      <button 
        onClick={() => window.location.reload()}
        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
      >
        Retry
      </button>
    </div>
  );

  if (!data || data.length === 0) return (
    <div className="bg-white rounded-lg shadow-sm border p-6 flex flex-col items-center">
      <div className="text-yellow-600">No yield curve data available</div>
      <button 
        onClick={() => window.location.reload()}
        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
      >
        Retry
      </button>
    </div>
  );

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6 flex flex-col items-center">
      <h2 className="text-xl font-semibold mb-2 text-gray-900">Current Yield Curve</h2>
      <p className="mb-6 text-gray-600 text-sm">Latest U.S. Treasury yields by maturity</p>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
          <XAxis 
            dataKey="maturity" 
            stroke="#6b7280" 
            tick={{ fill: '#6b7280' }} 
          />
          <YAxis 
            domain={['auto', 'auto']} 
            label={{ 
              value: 'Yield (%)', 
              angle: -90, 
              position: 'insideLeft', 
              fill: '#6b7280' 
            }} 
            stroke="#6b7280" 
            tick={{ fill: '#6b7280' }}
            tickFormatter={(value) => value.toFixed(2)}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#ffffff', 
              border: '1px solid #e5e7eb', 
              borderRadius: '8px',
              color: '#374151' 
            }} 
            labelStyle={{ color: '#6b7280' }}
            formatter={(value) => [`${value.toFixed(2)}%`, 'Yield']}
          />
          <Legend wrapperStyle={{ color: '#374151' }} />
          <Line 
            type="monotone" 
            dataKey="yield" 
            stroke="#3b82f6" 
            name="Yield" 
            strokeWidth={3} 
            dot={{ r: 4, fill: '#3b82f6' }} 
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default YieldCurveChart; 