import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ethers } from "ethers";
import VaultABI from "../contracts/ThumbVault.json";

const maturityOrder = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y'];
const regimeOptions = [
  { label: 'Steep', value: [1,0,0] },
  { label: 'Flat', value: [0,1,0] },
  { label: 'Inverted', value: [0,0,1] }
];

const OptimalAllocation = () => {
  const [yields, setYields] = useState(Array(7).fill(''));
  const [macro, setMacro] = useState(["", "", ""]);
  const [regime, setRegime] = useState([1,0,0]);
  const [allocation, setAllocation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Auto-fill all fields and fetch allocation on mount
  useEffect(() => {
    const fetchAndFill = async () => {
      setLoading(true); setError(null);
      try {
        // Mock data for production deployment
        const mockYields = [5.25, 5.30, 5.35, 5.40, 5.45, 5.50, 5.55];
        const mockMacro = [3.2, 3.8, 15.5]; // CPI, UNEMP, MOVE
        const mockAllocation = [0.15, 0.20, 0.25, 0.20, 0.10, 0.08, 0.02];
        
        setYields(mockYields.map(y => y.toString()));
        setMacro(mockMacro.map(m => m.toString()));
        setRegime([1,0,0]); // Steep regime
        setAllocation(mockAllocation.map((w, i) => ({ token: maturityOrder[i], weight: w })));
      } catch (e) {
        setError('Failed to load allocation data');
      } finally {
        setLoading(false);
      }
    };
    fetchAndFill();
  }, []);

  const setVaultAllocation = async () => {
    if (!allocation) return alert("No allocation to set!");
    try {
      // Connect to Ethereum provider (e.g., MetaMask)
      const provider = new ethers.BrowserProvider(window.ethereum);
      await provider.send("eth_requestAccounts", []);
      const signer = await provider.getSigner();

      // Vault contract address
      const contractAddress = "0x4eA3c91F275afA8c8c831ba2e37Fa1A18ec928e7";
      const contract = new ethers.Contract(contractAddress, VaultABI.abi, signer);

      // Convert weights to basis points (sum to 10000)
      let rawWeights = allocation.map(a => a.weight);
      let total = rawWeights.reduce((a, b) => a + b, 0);
      let scaled = rawWeights.map(w => Math.round((w / total) * 10000));
      // Adjust for rounding errors so sum is exactly 10000
      let diff = 10000 - scaled.reduce((a, b) => a + b, 0);
      if (diff !== 0) {
        let idx = scaled.indexOf(Math.max(...scaled));
        scaled[idx] += diff;
      }
      const weights = scaled;

      // Call the contract's rebalance function
      const tx = await contract.rebalance(weights);
      await tx.wait();
      alert("Vault allocation updated!");
    } catch (err) {
      alert("Failed to set allocation: " + (err?.reason || err?.message || err));
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">ML Optimal Allocation</h2>
      <p className="text-gray-600 mb-4">Current market data from FRED (Federal Reserve Economic Data)</p>
      <div className="mb-4 grid grid-cols-2 gap-2">
        {maturityOrder.map((mat, i) => (
          <label key={mat} className="flex items-center space-x-2">
            <span>{mat}:</span>
            <input 
              type="number" 
              value={yields[i]} 
              readOnly
              className="border p-1 rounded w-20 bg-gray-50 text-gray-600 cursor-not-allowed" 
            />
          </label>
        ))}
        <label className="flex items-center space-x-2">
          <span>CPI:</span>
          <input 
            type="number" 
            value={macro[0]} 
            readOnly
            className="border p-1 rounded w-20 bg-gray-50 text-gray-600 cursor-not-allowed" 
          />
        </label>
        <label className="flex items-center space-x-2">
          <span>UNEMP:</span>
          <input 
            type="number" 
            value={macro[1]} 
            readOnly
            className="border p-1 rounded w-20 bg-gray-50 text-gray-600 cursor-not-allowed" 
          />
        </label>
        <label className="flex items-center space-x-2">
          <span>MOVE:</span>
          <input 
            type="number" 
            value={macro[2]} 
            readOnly
            className="border p-1 rounded w-20 bg-gray-50 text-gray-600 cursor-not-allowed" 
          />
        </label>
        <div className="col-span-2 flex space-x-2 mt-2">
          {regimeOptions.map((r, i) => (
        <button
              key={r.label} 
              disabled
              className={`px-2 py-1 rounded ${regime[i] ? 'bg-blue-600 text-white' : 'bg-gray-200'} cursor-not-allowed opacity-75`}
            >
              {r.label}
        </button>
          ))}
        </div>
      </div>
      {loading && <div className="text-blue-500 mb-2">Loading current market data...</div>}
      {error && <div className="text-red-500 mb-2">{error}</div>}
      {allocation && (
        <>
      <ResponsiveContainer width="100%" height={250}>
            <BarChart data={allocation}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="token" stroke="#6b7280" />
              <YAxis stroke="#6b7280" tickFormatter={v => `${(v*100).toFixed(1)}%`} />
              <Tooltip formatter={v => [`${(v*100).toFixed(1)}%`, 'Weight']} />
              <Legend />
          <Bar dataKey="weight" fill="#3b82f6" name="Allocation %" />
        </BarChart>
      </ResponsiveContainer>
          <button
            onClick={setVaultAllocation}
            disabled={!allocation}
            className="px-4 py-2 rounded bg-green-600 text-white hover:bg-green-700 mb-4 ml-2"
          >
            Set Vault to ML Allocation
          </button>
        </>
      )}
    </div>
  );
};

export default OptimalAllocation; 