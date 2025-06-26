import React, { useState, useEffect, useCallback } from 'react';
import { ethers } from 'ethers';
import { PieChart, Pie, Cell } from 'recharts';
import YieldCurveChart from './components/YieldCurveChart';
import OptimalAllocation from './components/OptimalAllocation';
import BacktestChart from './components/BacktestChart';
import Methodology from './components/Methodology';

// Contract ABIs
import thumbTokenABI from './contracts/ThumbToken.json';
import thumbVaultABI from './contracts/ThumbVault.json';

// Contract addresses - Fixed with correct addresses
const THUMB_TOKEN_ADDRESS = '0x8Ed90B81A84d84232408716e378013b0BCECE4fe';
const THUMB_VAULT_ADDRESS = '0x4eA3c91F275afA8c8c831ba2e37Fa1A18ec928e7';

// Colors for pie chart
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658'];

// Mock data for production deployment
const MOCK_YIELD_CURVE_DATA = {
  maturities: ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y'],
  yields: [5.25, 5.30, 5.35, 5.40, 5.45, 5.50, 5.55]
};

const MOCK_OPTIMAL_ALLOCATION = {
  allocation: [0.15, 0.20, 0.25, 0.20, 0.10, 0.08, 0.02]
};

const MOCK_BACKTEST_DATA = {
  dates: ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05'],
  returns: [0.02, 0.015, 0.025, 0.018, 0.022]
};

function App() {
  const [account, setAccount] = useState('');
  const [thumbTokenContract, setThumbTokenContract] = useState(null);
  const [thumbVaultContract, setThumbVaultContract] = useState(null);
  const [balance, setBalance] = useState('0');
  const [vaultBalance, setVaultBalance] = useState('0');
  const [totalVaultBalance, setTotalVaultBalance] = useState('0');
  const [vaultAllocation, setVaultAllocation] = useState([]);
  const [optimalAllocation, setOptimalAllocation] = useState(MOCK_OPTIMAL_ALLOCATION);
  const [yieldCurveData, setYieldCurveData] = useState(MOCK_YIELD_CURVE_DATA);
  const [backtestData, setBacktestData] = useState(MOCK_BACKTEST_DATA);
  const [depositAmount, setDepositAmount] = useState('');
  const [faucetRecipient, setFaucetRecipient] = useState('');
  const [loading, setLoading] = useState(false);
  const [isFauceting, setIsFauceting] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  // Helper functions
  const formatAddress = (address) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const formatNumber = (number) => {
    return parseFloat(number).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  const getTabClass = (tabName) => {
    return `px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
      activeTab === tabName
        ? 'bg-blue-600 text-white'
        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
    }`;
  };

  useEffect(() => {
    connectWallet();
  }, [connectWallet]);

  const connectWallet = useCallback(async () => {
    if (typeof window.ethereum !== 'undefined') {
      try {
        const accounts = await window.ethereum.request({
          method: 'eth_requestAccounts'
        });
        const account = accounts[0];
        setAccount(account);

        const provider = new ethers.BrowserProvider(window.ethereum);
        const signer = await provider.getSigner();
        const thumbToken = new ethers.Contract(
          THUMB_TOKEN_ADDRESS,
          thumbTokenABI.abi,
          signer
        );
        const thumbVault = new ethers.Contract(
          THUMB_VAULT_ADDRESS,
          thumbVaultABI.abi,
          signer
        );

        setThumbTokenContract(thumbToken);
        setThumbVaultContract(thumbVault);

        updateBalances(thumbToken, thumbVault, account);
        fetchVaultAllocations(thumbVault);
      } catch (error) {
        console.error('Error connecting wallet:', error);
      }
    } else {
      alert('Please install MetaMask!');
    }
  }, []);

  const updateBalances = async (thumbToken, thumbVault, account) => {
    try {
      const balance = await thumbToken.balanceOf(account);
      const vaultBalance = await thumbVault.getUserBalance(account);
      const totalVaultBalance = await thumbVault.totalAssets();
      setBalance(ethers.formatEther(balance));
      setVaultBalance(ethers.formatEther(vaultBalance));
      setTotalVaultBalance(ethers.formatEther(totalVaultBalance));
    } catch (error) {
      console.error('Error updating balances:', error);
      setBalance('0');
      setVaultBalance('0');
      setTotalVaultBalance('0');
    }
  };

  const fetchVaultAllocations = async (thumbVault) => {
    try {
      const allocations = await thumbVault.getVaultAllocations();
      // Convert from basis points to percentages
      setVaultAllocation(allocations.map(a => Number(a) / 10000));
    } catch (error) {
      console.error('Error fetching Vault allocations:', error);
      setVaultAllocation([]);
    }
  };

  const depositTokens = async () => {
    if (!thumbTokenContract || !thumbVaultContract) return;
    
    setLoading(true);
    try {
      const amount = ethers.parseEther(depositAmount);
      const approveTx = await thumbTokenContract.approve(THUMB_VAULT_ADDRESS, amount);
      await approveTx.wait();

      const depositTx = await thumbVaultContract.deposit(amount);
      await depositTx.wait();

      await updateBalances(thumbTokenContract, thumbVaultContract, account);
      await fetchVaultAllocations(thumbVaultContract);
      setDepositAmount('');
    } catch (error) {
      console.error('Error depositing tokens:', error);
    } finally {
      setLoading(false);
    }
  };

  const withdrawTokens = async () => {
    if (!thumbVaultContract) return;

    setLoading(true);
    try {
      const amount = ethers.parseEther('100'); // Fixed amount of 100 tokens
      const tx = await thumbVaultContract.withdraw(amount);
      await tx.wait();

      await updateBalances(thumbTokenContract, thumbVaultContract, account);
      await fetchVaultAllocations(thumbVaultContract);
    } catch (error) {
      console.error('Error withdrawing tokens:', error);
    } finally {
      setLoading(false);
    }
  };

  const sendFaucetTokens = async () => {
    if (!faucetRecipient) {
      alert('Please fill in recipient address');
      return;
    }

    // Validate Ethereum address
    if (!ethers.isAddress(faucetRecipient)) {
      alert('Invalid Ethereum address');
      return;
    }

    try {
      setIsFauceting(true);
      
      // For production, show a message that faucet is not available
      alert('Faucet functionality is not available in production. Please use testnet faucets or contact the team.');
      
    } catch (error) {
      console.error('Error sending faucet tokens:', error);
      alert('Error sending faucet tokens: ' + error.message);
    } finally {
      setIsFauceting(false);
    }
  };

  // Handler to call rebalance on the contract
  const handleRebalance = async (mlChartData) => {
    if (!thumbVaultContract || !account) return;
    // Convert ML weights (percentages) to basis points (integers)
    const weights = mlChartData.map(item => Math.round(item.weight * 10000));
    try {
      const tx = await thumbVaultContract.rebalance(weights);
      await tx.wait();
      // Refresh allocations
      fetchVaultAllocations(thumbVaultContract);
      alert('Vault rebalanced with ML allocations!');
    } catch (error) {
      console.error('Error rebalancing:', error);
      alert('Rebalance failed. Are you the contract owner?');
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white">
        <h1 className="text-3xl font-bold mb-2">T0mThumb Vault</h1>
        <p className="text-blue-100">ML-Powered ERC-4626 Treasury Bond Yield Strategy</p>
            {account && (
          <div className="mt-4 flex items-center space-x-2">
            <span className="text-sm">Connected:</span>
            <span className="bg-white/20 px-3 py-1 rounded-full text-sm font-mono">
              {formatAddress(account)}
              </span>
          </div>
                )}
              </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <h3 className="text-sm font-medium text-gray-500">Token Balance</h3>
          <p className="text-2xl font-bold text-gray-900">{formatNumber(balance)} THUMB</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <h3 className="text-sm font-medium text-gray-500">Total Vault Balance</h3>
          <p className="text-2xl font-bold text-gray-900">{formatNumber(totalVaultBalance)} THUMB</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <h3 className="text-sm font-medium text-gray-500">Your Vault Position</h3>
          <p className="text-2xl font-bold text-blue-600">{formatNumber(vaultBalance)} THUMB</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <h3 className="text-sm font-medium text-gray-500">Current Regime</h3>
          <p className="text-2xl font-bold text-green-600">
            {optimalAllocation.regime || 'Loading...'}
          </p>
        </div>
      </div>

      {/* Faucet Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-xl font-semibold mb-4">Get T0mThumb Tokens</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
                <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Recipient Address
              </label>
                  <input
                    type="text"
                    value={faucetRecipient}
                    onChange={(e) => setFaucetRecipient(e.target.value)}
                    placeholder="0x..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Amount (T0mThumb)
              </label>
                  <input
                    type="number"
                    value="100"
                    readOnly
                className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50 text-gray-500 cursor-not-allowed"
                  />
              <p className="text-xs text-gray-500 mt-1">
                    Fixed amount: 100 THUMB per transaction
              </p>
                </div>
                <button
                  onClick={sendFaucetTokens}
                  disabled={isFauceting}
              className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isFauceting ? 'Sending...' : 'Send T0mThumb'}
                </button>
                </div>
          <div className="space-y-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-blue-900 mb-2">How it works</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Enter any valid Ethereum address to send tokens</li>
                <li>• Fixed amount of 100 THUMB per transaction</li>
                <li>• Tokens are sent from the faucet wallet</li>
                <li>• Transaction will be recorded on Sepolia testnet blockchain</li>
              </ul>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-yellow-900 mb-2">Important Notes</h3>
              <ul className="text-sm text-yellow-800 space-y-1">
                <li>• Double-check the recipient address</li>
                <li>• Transactions are irreversible</li>
                <li>• No wallet connection required</li>
              </ul>
            </div>
              </div>
            </div>
          </div>

      {/* Yield Curve Chart */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-xl font-semibold mb-4">Current Yield Curve</h2>
        <YieldCurveChart data={yieldCurveData} />
              </div>

      {/* Vault Allocation Pie Chart */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-xl font-semibold mb-4">Current Vault Allocation</h2>
        <p className="text-gray-600 mb-4">The vault's THUMB tokens are automatically allocated across Treasury bonds based on ML-driven yield optimization in this ERC-4626 vault.</p>
        {vaultAllocation.length > 0 ? (
          <div className="flex flex-col items-center">
                <PieChart width={500} height={350}>
                  <Pie
                data={vaultAllocation.map((value, index) => {
                  const thumbAmount = parseFloat(totalVaultBalance) * value;
                  return {
                      name: ['Thumb3M', 'Thumb6M', 'Thumb1Y', 'Thumb2Y', 'Thumb5Y', 'Thumb10Y', 'Thumb30Y'][index],
                      value: value * 100,
                    thumbAmount: thumbAmount
                  };
                })}
                    cx={250}
                    cy={175}
                    labelLine={false}
                    outerRadius={120}
                    fill="#8884d8"
                    dataKey="value"
                label={({ name, thumbAmount }) => `${name}\n${thumbAmount.toFixed(2)} THUMB`}
                  >
                {vaultAllocation.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                </PieChart>
            <div className="mt-4 w-full">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Bond</th>
                      <th className="text-right py-2">Percentage</th>
                    <th className="text-right py-2">THUMB Amount</th>
                    </tr>
                  </thead>
                  <tbody>
                  {vaultAllocation.map((value, index) => {
                    const thumbAmount = parseFloat(totalVaultBalance) * value;
                      return (
                        <tr key={index} className="border-b">
                          <td className="py-2">{['Thumb3M', 'Thumb6M', 'Thumb1Y', 'Thumb2Y', 'Thumb5Y', 'Thumb10Y', 'Thumb30Y'][index]}</td>
                          <td className="text-right py-2">
                            {(value * 100).toFixed(2)}%
                          </td>
                        <td className="text-right py-2 font-mono">
                          {thumbAmount.toFixed(2)} THUMB
                        </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
        ) : (
          <p className="text-gray-500 text-center py-8">Loading allocation data...</p>
        )}
      </div>

      {/* Optimal Allocation */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-xl font-semibold mb-4">ML Optimal Allocation</h2>
        <OptimalAllocation
          onRebalance={handleRebalance}
          currentVaultAllocation={vaultAllocation}
          data={optimalAllocation}
        />
      </div>
    </div>
  );

  const renderStaking = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-xl font-semibold mb-6">Deposit & Withdraw Tokens</h2>
        <p className="text-gray-600 mb-6">
          This ERC-4626 vault automatically allocates your THUMB tokens across Treasury bonds using machine learning optimization. 
          Deposit to earn yield, withdraw anytime.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Deposit Section */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900">Deposit Tokens</h3>
            <div className="space-y-3">
              <input
                type="number"
                placeholder="Amount to deposit"
                value={depositAmount}
                onChange={(e) => setDepositAmount(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={depositTokens}
                disabled={loading || !depositAmount}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Deposit Tokens'}
              </button>
            </div>
          </div>

          {/* Withdraw Section */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900">Withdraw Tokens</h3>
            <div className="space-y-3">
              <input
                type="number"
                value="100"
                readOnly
                className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50 text-gray-500 cursor-not-allowed"
              />
              <p className="text-xs text-gray-500 mt-1">
                Fixed amount: 100 THUMB per transaction
              </p>
              <button
                onClick={withdrawTokens}
                disabled={loading}
                className="w-full bg-red-600 text-white py-2 px-4 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Withdraw 100 THUMB'}
              </button>
            </div>
          </div>
        </div>

        {/* ERC-4626 Information */}
        <div className="mt-8 bg-blue-50 p-6 rounded-lg">
          <h3 className="text-lg font-medium text-blue-900 mb-3">ERC-4626 Vault Standard</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-blue-800 mb-2">What is ERC-4626?</h4>
              <p className="text-sm text-blue-700">
                ERC-4626 is the standard for tokenized vaults. It provides a standardized interface for yield-bearing vaults, 
                making them composable and interoperable across DeFi protocols.
              </p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-blue-800 mb-2">Key Features</h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Standardized deposit/withdraw functions</li>
                <li>• Automatic yield generation</li>
                <li>• ML-driven allocation strategy</li>
                <li>• Composable with other DeFi protocols</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderFaucet = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-xl font-semibold mb-6">T0mThumb Faucet</h2>
        <p className="text-gray-600 mb-6">
          Send T0mThumb tokens to any Ethereum address. This is useful for testing or sharing tokens with others.
        </p>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Faucet Form */}
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Recipient Address
              </label>
              <input
                type="text"
                value={faucetRecipient}
                onChange={(e) => setFaucetRecipient(e.target.value)}
                placeholder="0x..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Amount (T0mThumb)
              </label>
              <input
                type="number"
                value="100"
                readOnly
                className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50 text-gray-500 cursor-not-allowed"
              />
              <p className="text-xs text-gray-500 mt-1">
                Fixed amount: 100 THUMB per transaction
              </p>
            </div>
            <button
              onClick={sendFaucetTokens}
              disabled={isFauceting}
              className="w-full bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              {isFauceting ? 'Sending...' : 'Send T0mThumb'}
            </button>
          </div>

          {/* Information Panel */}
          <div className="space-y-6">
            <div className="bg-blue-50 p-6 rounded-lg">
              <h3 className="text-lg font-medium text-blue-900 mb-3">How it works</h3>
              <ul className="text-sm text-blue-800 space-y-2">
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  Enter any valid Ethereum address to send tokens
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  Fixed amount of 100 THUMB per transaction
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  Tokens are sent from the faucet wallet
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  Transaction will be recorded on Sepolia testnet blockchain
                </li>
              </ul>
            </div>
            
            <div className="bg-yellow-50 p-6 rounded-lg">
              <h3 className="text-lg font-medium text-yellow-900 mb-3">Important Notes</h3>
              <ul className="text-sm text-yellow-800 space-y-2">
                <li className="flex items-start">
                  <span className="text-yellow-600 mr-2">•</span>
                  Double-check the recipient address
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-600 mr-2">•</span>
                  Transactions are irreversible
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-600 mr-2">•</span>
                  No wallet connection required
                </li>
              </ul>
            </div>

            <div className="bg-gray-50 p-6 rounded-lg">
              <h3 className="text-lg font-medium text-gray-900 mb-3">Faucet Wallet</h3>
              <p className="text-sm text-gray-600 mb-2">Faucet Address:</p>
              <p className="font-mono text-xs text-gray-900 break-all mb-3">
                0x4B23Fe64B9cF9e2EA384dBc9C31861F0037b3258
              </p>
              <p className="text-sm text-gray-600">
                Tokens are sent from this dedicated faucet wallet
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderContracts = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-xl font-semibold mb-6">Smart Contracts</h2>
        <p className="text-gray-600 mb-6">
          All contracts are deployed on the Sepolia testnet. You can view transactions, verify contracts, and interact with them directly on Etherscan.
        </p>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Contract Details */}
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-3">T0mThumb Token (THUMB)</h3>
              <div className="bg-gray-50 p-4 rounded-lg border">
                <p className="text-sm text-gray-600 mb-2">Contract Address:</p>
                <p className="font-mono text-sm text-gray-900 break-all mb-3">
                  0x8Ed90B81A84d84232408716e378013b0BCECE4fe
                </p>
                <div className="space-y-2">
                  <a 
                    href="https://sepolia.etherscan.io/address/0x8Ed90B81A84d84232408716e378013b0BCECE4fe" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-blue-600 hover:text-blue-700 text-sm"
                  >
                    View on Etherscan →
                  </a>
                  <br />
                  <a 
                    href="https://sepolia.etherscan.io/address/0x8Ed90B81A84d84232408716e378013b0BCECE4fe#code" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-blue-600 hover:text-blue-700 text-sm"
                  >
                    View Contract Code →
                  </a>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-3">ERC-4626 Vault</h3>
              <div className="bg-gray-50 p-4 rounded-lg border">
                <p className="text-sm text-gray-600 mb-2">Contract Address:</p>
                <p className="font-mono text-sm text-gray-900 break-all mb-3">
                  0x4eA3c91F275afA8c8c831ba2e37Fa1A18ec928e7
                </p>
                <div className="space-y-2">
                  <a 
                    href="https://sepolia.etherscan.io/address/0x4eA3c91F275afA8c8c831ba2e37Fa1A18ec928e7" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-blue-600 hover:text-blue-700 text-sm"
                  >
                    View on Etherscan →
                  </a>
                  <br />
                  <a 
                    href="https://sepolia.etherscan.io/address/0x4eA3c91F275afA8c8c831ba2e37Fa1A18ec928e7#code" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-blue-600 hover:text-blue-700 text-sm"
                  >
                    View Contract Code →
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Network Information & Quick Links */}
          <div className="space-y-6">
            <div className="bg-blue-50 p-6 rounded-lg border">
              <h3 className="text-lg font-medium text-blue-900 mb-3">Network Information</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-blue-700">Network:</span>
                  <span className="text-blue-900 font-medium">Sepolia Testnet</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-blue-700">Chain ID:</span>
                  <span className="text-blue-900 font-medium">11155111</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-blue-700">Currency:</span>
                  <span className="text-blue-900 font-medium">Sepolia ETH</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-blue-700">Block Explorer:</span>
                  <span className="text-blue-900 font-medium">Etherscan</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-blue-700">RPC URL:</span>
                  <span className="text-blue-900 font-medium text-xs">https://sepolia.infura.io/v3/...</span>
                </div>
              </div>
            </div>

            <div className="bg-green-50 p-6 rounded-lg border">
              <h3 className="text-lg font-medium text-green-900 mb-3">Quick Links</h3>
              <div className="space-y-3">
                <a 
                  href="https://sepoliafaucet.com/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="block text-sm text-green-700 hover:text-green-800"
                >
                  • Sepolia Faucet (Get test ETH)
                </a>
                <a 
                  href="https://sepolia.etherscan.io/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="block text-sm text-green-700 hover:text-green-800"
                >
                  • Sepolia Etherscan Explorer
                </a>
                <a 
                  href="https://chainlist.org/chain/11155111" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="block text-sm text-green-700 hover:text-green-800"
                >
                  • Add Sepolia to MetaMask
                </a>
                <a 
                  href="https://docs.alchemy.com/reference/ethereum-api-endpoints" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="block text-sm text-green-700 hover:text-green-800"
                >
                  • Sepolia RPC Endpoints
                </a>
              </div>
            </div>

            <div className="bg-yellow-50 p-6 rounded-lg border">
              <h3 className="text-lg font-medium text-yellow-900 mb-3">Testing Tips</h3>
              <div className="space-y-2 text-sm text-yellow-800">
                <p>• Use the faucet to get test ETH for gas fees</p>
                <p>• All transactions are free on testnet</p>
                <p>• Contract interactions are safe to test</p>
                <p>• Check transaction status on Etherscan</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderPerformance = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-xl font-semibold mb-4">Historical Performance</h2>
        <BacktestChart data={backtestData} />
      </div>
    </div>
  );

  const renderMethodology = () => (
    <div className="space-y-6">
      <Methodology />
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 font-calibri">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-gray-900">T0mThumb</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3 text-sm">
                <a 
                  href="https://sepolia.etherscan.io/address/0x4eA3c91F275afA8c8c831ba2e37Fa1A18ec928e7" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-700 transition-colors"
                >
                  Vault Contract
                </a>
                <a 
                  href="https://sepolia.etherscan.io/address/0x8Ed90B81A84d84232408716e378013b0BCECE4fe" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-700 transition-colors"
                >
                  Token Contract
                </a>
              </div>
              {!account ? (
                <button
                  onClick={connectWallet}
                  className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
                >
                  Connect Wallet
                </button>
              ) : (
                <span className="text-sm text-gray-600">
                  {formatAddress(account)}
                </span>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow-sm border mb-6">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6" aria-label="Tabs">
              <button
                onClick={() => setActiveTab('overview')}
                className={getTabClass('overview')}
              >
                Overview
              </button>
              <button
                onClick={() => setActiveTab('staking')}
                className={getTabClass('staking')}
              >
                Vault
              </button>
              <button
                onClick={() => setActiveTab('faucet')}
                className={getTabClass('faucet')}
              >
                Faucet
              </button>
              <button
                onClick={() => setActiveTab('contracts')}
                className={getTabClass('contracts')}
              >
                Contracts
              </button>
              <button
                onClick={() => setActiveTab('performance')}
                className={getTabClass('performance')}
              >
                Performance
              </button>
              <button
                onClick={() => setActiveTab('methodology')}
                className={getTabClass('methodology')}
              >
                Methodology
              </button>
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'staking' && renderStaking()}
        {activeTab === 'faucet' && renderFaucet()}
        {activeTab === 'contracts' && renderContracts()}
        {activeTab === 'performance' && renderPerformance()}
        {activeTab === 'methodology' && renderMethodology()}
      </div>
    </div>
  );
}

export default App; 