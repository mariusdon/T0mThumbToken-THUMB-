#!/bin/bash

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "SEPOLIA_RPC_URL=your_sepolia_rpc_url" > .env
    echo "PRIVATE_KEY=your_wallet_private_key" >> .env
    echo "Please update the .env file with your actual values"
fi

# Install root dependencies
echo "Installing root dependencies..."
npm install

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

echo "Setup complete! Please update the .env file with your actual values before running the application." 