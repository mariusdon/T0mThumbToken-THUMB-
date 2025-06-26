import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const YieldCurveChart = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    // Mock data for production deployment
    const mockData = [
      { maturity: '3M', yield: 5.25 },
      { maturity: '6M', yield: 5.30 },
      { maturity: '1Y', yield: 5.35 },
      { maturity: '2Y', yield: 5.40 },
      { maturity: '5Y', yield: 5.45 },
      { maturity: '10Y', yield: 5.50 },
      { maturity: '30Y', yield: 5.55 }
    ];
    setData(mockData);
  }, []);

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">U.S. Treasury Yield Curve</h2>
      <p className="text-gray-600 mb-4">Current market yields across different maturities</p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="maturity" stroke="#6b7280" />
          <YAxis stroke="#6b7280" tickFormatter={v => `${v}%`} />
          <Tooltip formatter={v => [`${v}%`, 'Yield']} />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="yield" 
            stroke="#3b82f6" 
            strokeWidth={2}
            name="Yield %"
            dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default YieldCurveChart; 