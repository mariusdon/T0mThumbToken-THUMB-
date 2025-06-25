# TomThumbVault

A decentralized finance (DeFi) simulation that mimics an automated U.S. Treasury bonds investment strategy. The system allows users to deposit THUMB tokens into a vault which then allocates these funds across seven synthetic bond tokens representing various U.S. Treasury maturities.

## Features

- ERC-20 THUMB token for deposits
- Automated portfolio allocation using machine learning
- Real-time portfolio visualization with yield curve charts
- Historical performance tracking
- Dark/Light mode UI
- MetaMask integration
- ML-powered optimal allocation recommendations

## Prerequisites

- **Node.js** (v16 or higher)
- **Python 3.8+** with pip
- **MetaMask** browser extension
- **Sepolia testnet ETH** (for contract interactions)
- **Git**

## Quick Start Guide

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd "T0mThumbToken(TTT)"
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```bash
# Windows PowerShell
echo "SEPOLIA_RPC_URL=your_sepolia_rpc_url" > .env
echo "PRIVATE_KEY=your_wallet_private_key" >> .env
```

**Required values:**
- `SEPOLIA_RPC_URL`: Your Infura/Alchemy Sepolia RPC URL
- `PRIVATE_KEY`: Your wallet's private key (for contract deployment)

### 3. Install Dependencies

#### Root Dependencies (Hardhat)
```bash
npm install
```

#### Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

#### Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 4. Running the Application

**IMPORTANT:** You need to run both the backend and frontend simultaneously.

#### Step 1: Start the Backend (ML API Server)
Open a new terminal/PowerShell window and run:
```bash
cd "T0mThumbToken(TTT)/backend"
python run.py
```

**Expected output:**
```
Training ML model...
Fetching yield curve data...
Generating baseline allocations...
Preparing training data...
Training model...
Simulating performance...
Done! Model and simulation data saved.
Starting FastAPI server...
INFO: Uvicorn running on http://127.0.0.1:8000
```

**Keep this terminal running!** The backend provides the ML optimization and yield curve data.

#### Step 2: Start the Frontend (React App)
Open another terminal/PowerShell window and run:
```bash
cd "T0mThumbToken(TTT)/frontend"
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view tom-thumb-vault-frontend in the browser.

Local:            http://localhost:3000
```

#### Step 3: Access the Application
1. Open your browser and go to `http://localhost:3000`
2. Connect your MetaMask wallet (make sure you're on Sepolia testnet)
3. You should now see:
   - Yield curve visualization
   - Vault composition pie chart
   - ML optimal allocation recommendations
   - Performance charts
   - Deposit/withdraw functionality

## Troubleshooting

### Common Issues and Solutions

#### 1. Backend Won't Start
**Problem:** `ModuleNotFoundError` or missing dependencies
**Solution:**
```bash
cd backend
pip install -r requirements.txt
python run.py
```

#### 2. Frontend Won't Start
**Problem:** `npm start` fails or shows build errors
**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

#### 3. Yield Curve Not Loading
**Problem:** Charts show "No data" or loading errors
**Solution:**
- Ensure backend is running on `http://localhost:8000`
- Check browser console for CORS errors
- Verify the backend health endpoint: `http://localhost:8000/health`

#### 4. MetaMask Connection Issues
**Problem:** Can't connect wallet or interact with contracts
**Solution:**
- Ensure MetaMask is unlocked
- Switch to Sepolia testnet
- Get testnet ETH from [Sepolia Faucet](https://sepoliafaucet.com/)
- Clear browser cache and reload

#### 5. PowerShell Command Issues
**Problem:** `&&` operator not working
**Solution:** Use separate commands:
```powershell
cd "T0mThumbToken(TTT)/backend"
python run.py
```
Then in a new terminal:
```powershell
cd "T0mThumbToken(TTT)/frontend"
npm start
```

### Health Checks

#### Backend Health Check
```bash
# Test if backend is running
curl http://localhost:8000/health
# or in PowerShell:
Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
```

#### Frontend Health Check
- Open `http://localhost:3000` in browser
- Should see the TomThumbVault interface

## Development

### Project Structure
```
T0mThumbToken(TTT)/
├── backend/                 # Python FastAPI ML server
│   ├── app.py              # FastAPI application
│   ├── run.py              # Server startup script
│   ├── yield_curve_optimizer.py  # ML optimization logic
│   └── requirements.txt    # Python dependencies
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # React components
│   │   └── App.jsx         # Main application
│   └── package.json        # Node.js dependencies
├── contracts/              # Smart contracts
└── scripts/                # Deployment scripts
```

### Available Scripts

#### Backend
```bash
cd backend
python run.py              # Start ML server
python train_model.py      # Train ML model only
```

#### Frontend
```bash
cd frontend
npm start                  # Start development server
npm run build             # Build for production
npm test                  # Run tests
```

#### Smart Contracts
```bash
npx hardhat compile        # Compile contracts
npx hardhat test          # Run contract tests
npx hardhat run scripts/deploy.js --network sepolia  # Deploy to Sepolia
```

## Contract Addresses (Sepolia Testnet)

- **THUMB Token:** `0x8Ed90B81A84d84232408716e378013b0BCECE4fe`
- **Vault Contract:** `0x4eA3c91F275afA8c8c831ba2e37Fa1A18ec928e7`

## API Endpoints

When the backend is running, these endpoints are available:

- `GET /` - API root
- `GET /health` - Health check
- `GET /optimal-allocation` - Get ML-optimized allocation
- `GET /ytd-performance` - Get year-to-date performance data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (backend + frontend)
5. Submit a pull request

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify both backend and frontend are running
3. Check browser console for errors
4. Open an issue with detailed error messages

## License

MIT License 