#!/usr/bin/env python3
"""
Script to update vault allocations based on ML strategy
This script fetches current ML allocations and updates the vault contract
"""

import asyncio
import aiohttp
import json
from web3 import Web3
from eth_account import Account
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Contract addresses
THUMB_VAULT_ADDRESS = '0x4eA3c91F275afA8c8c831ba2e37Fa1A18ec928e7'

# Vault ABI (simplified for allocation updates)
VAULT_ABI = [
    {
        "inputs": [
            {
                "internalType": "uint256[7]",
                "name": "newAllocations",
                "type": "uint256[7]"
            }
        ],
        "name": "rebalance",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getVaultAllocations",
        "outputs": [
            {
                "internalType": "uint256[7]",
                "name": "",
                "type": "uint256[7]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

class VaultAllocationUpdater:
    def __init__(self):
        self.w3 = None
        self.vault_contract = None
        self.account = None
        
    async def initialize_web3(self):
        """Initialize Web3 connection"""
        try:
            # Connect to Sepolia testnet
            rpc_url = os.getenv('SEPOLIA_RPC_URL')
            if not rpc_url:
                print("Error: SEPOLIA_RPC_URL not found in environment variables")
                return False
                
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Check connection
            if not self.w3.is_connected():
                print("Error: Could not connect to Sepolia network")
                return False
                
            print(f"Connected to Sepolia network: {self.w3.is_connected()}")
            
            # Initialize contract
            self.vault_contract = self.w3.eth.contract(
                address=THUMB_VAULT_ADDRESS,
                abi=VAULT_ABI
            )
            
            # Set up account
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key:
                print("Error: PRIVATE_KEY not found in environment variables")
                return False
                
            self.account = Account.from_key(private_key)
            print(f"Account: {self.account.address}")
            
            return True
            
        except Exception as e:
            print(f"Error initializing Web3: {e}")
            return False
    
    async def get_ml_allocation(self):
        """Fetch current ML allocation from backend"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://127.0.0.1:8000/ml-allocation') as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['weights']
                    else:
                        print(f"Error fetching ML allocation: {response.status}")
                        return None
        except Exception as e:
            print(f"Error fetching ML allocation: {e}")
            return None
    
    def convert_weights_to_basis_points(self, weights):
        """Convert decimal weights to basis points (multiply by 10000)"""
        return [int(w * 10000) for w in weights]
    
    async def update_vault_allocations(self, allocations):
        """Update vault contract with new allocations"""
        try:
            # Convert to basis points
            basis_points = self.convert_weights_to_basis_points(allocations)
            
            print(f"Current ML allocations: {allocations}")
            print(f"Basis points: {basis_points}")
            
            # Build transaction
            transaction = self.vault_contract.functions.rebalance(basis_points).build_transaction({
                'from': self.account.address,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            print(f"Transaction sent: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"Transaction confirmed: {receipt['status']}")
            
            return True
            
        except Exception as e:
            print(f"Error updating vault allocations: {e}")
            return False
    
    async def get_current_allocations(self):
        """Get current vault allocations"""
        try:
            allocations = self.vault_contract.functions.getVaultAllocations().call()
            return [a / 10000 for a in allocations]  # Convert from basis points
        except Exception as e:
            print(f"Error getting current allocations: {e}")
            return None
    
    async def run_allocation_update(self):
        """Main function to update allocations"""
        print("Starting vault allocation update...")
        
        # Initialize Web3
        if not await self.initialize_web3():
            return False
        
        # Get current ML allocation
        ml_allocations = await self.get_ml_allocation()
        if not ml_allocations:
            print("Failed to get ML allocation")
            return False
        
        # Get current vault allocations
        current_allocations = await self.get_current_allocations()
        if current_allocations:
            print(f"Current vault allocations: {current_allocations}")
        
        # Update vault with ML allocations
        success = await self.update_vault_allocations(ml_allocations)
        
        if success:
            print("✅ Vault allocations updated successfully!")
            
            # Verify the update
            new_allocations = await self.get_current_allocations()
            if new_allocations:
                print(f"New vault allocations: {new_allocations}")
        else:
            print("❌ Failed to update vault allocations")
        
        return success

async def main():
    """Main entry point"""
    updater = VaultAllocationUpdater()
    await updater.run_allocation_update()

if __name__ == "__main__":
    asyncio.run(main()) 