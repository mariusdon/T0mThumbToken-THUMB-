@echo off

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    echo SEPOLIA_RPC_URL=your_sepolia_rpc_url > .env
    echo PRIVATE_KEY=your_wallet_private_key >> .env
    echo Please update the .env file with your actual values
)

REM Install root dependencies
echo Installing root dependencies...
call npm install

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
call npm install
cd ..

REM Install backend dependencies
echo Installing backend dependencies...
cd backend
call pip install -r requirements.txt
cd ..

echo Setup complete! Please update the .env file with your actual values before running the application.
pause 