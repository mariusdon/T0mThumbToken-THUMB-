import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const BacktestChart = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    // Mock data for production deployment
    const mockData = [
      { date: '2023-01', returns: 0.02, benchmark: 0.015 },
      { date: '2023-02', returns: 0.015, benchmark: 0.012 },
      { date: '2023-03', returns: 0.025, benchmark: 0.018 },
      { date: '2023-04', returns: 0.018, benchmark: 0.020 },
      { date: '2023-05', returns: 0.022, benchmark: 0.016 },
      { date: '2023-06', returns: 0.019, benchmark: 0.014 },
      { date: '2023-07', returns: 0.024, benchmark: 0.017 },
      { date: '2023-08', returns: 0.021, benchmark: 0.019 },
      { date: '2023-09', returns: 0.016, benchmark: 0.013 },
      { date: '2023-10', returns: 0.023, benchmark: 0.020 },
      { date: '2023-11', returns: 0.020, benchmark: 0.018 },
      { date: '2023-12', returns: 0.025, benchmark: 0.022 }
    ];
    setData(mockData);
  }, []);

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Backtest Performance</h2>
      <p className="text-gray-600 mb-4">Historical performance comparison with benchmark</p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="date" stroke="#6b7280" />
          <YAxis stroke="#6b7280" tickFormatter={v => `${(v*100).toFixed(1)}%`} />
          <Tooltip formatter={v => [`${(v*100).toFixed(2)}%`, 'Return']} />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="returns" 
            stroke="#3b82f6" 
            strokeWidth={2}
            name="ML Strategy"
            dot={{ fill: '#3b82f6', strokeWidth: 2, r: 3 }}
          />
          <Line 
            type="monotone" 
            dataKey="benchmark" 
            stroke="#10b981" 
            strokeWidth={2}
            name="Benchmark"
            dot={{ fill: '#10b981', strokeWidth: 2, r: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BacktestChart; 