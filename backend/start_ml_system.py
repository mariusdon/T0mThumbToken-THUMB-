#!/usr/bin/env python3
"""
Comprehensive startup script for the ML-powered TomThumbVault system
This script starts the backend API, scheduler, and ensures everything is working
"""

import asyncio
import subprocess
import sys
import time
import logging
import aiohttp
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class MLSystemManager:
    def __init__(self):
        self.backend_process = None
        self.scheduler_process = None
        
    async def check_backend_health(self):
        """Check if backend is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://127.0.0.1:8000/health') as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Backend healthy: {data}")
                        return True
                    else:
                        logger.error(f"❌ Backend unhealthy: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Backend health check failed: {e}")
            return False
    
    async def test_ml_endpoints(self):
        """Test ML endpoints"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test ML allocation endpoint
                async with session.get('http://127.0.0.1:8000/ml-allocation') as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ ML allocation endpoint working: {data['weights']}")
                    else:
                        logger.error(f"❌ ML allocation endpoint failed: {response.status}")
                
                # Test backtest endpoint
                async with session.get('http://127.0.0.1:8000/backtest-data-enhanced') as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Backtest endpoint working: {len(data['date'])} data points")
                    else:
                        logger.error(f"❌ Backtest endpoint failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ Endpoint testing failed: {e}")
    
    def start_backend(self):
        """Start the FastAPI backend"""
        try:
            logger.info("🚀 Starting FastAPI backend...")
            self.backend_process = subprocess.Popen([
                sys.executable, 'app.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"✅ Backend started with PID: {self.backend_process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def start_scheduler(self):
        """Start the allocation scheduler"""
        try:
            logger.info("📅 Starting allocation scheduler...")
            self.scheduler_process = subprocess.Popen([
                sys.executable, 'scheduler.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"✅ Scheduler started with PID: {self.scheduler_process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {e}")
            return False
    
    async def wait_for_backend(self, timeout=30):
        """Wait for backend to be ready"""
        logger.info("⏳ Waiting for backend to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.check_backend_health():
                logger.info("✅ Backend is ready!")
                return True
            await asyncio.sleep(2)
        
        logger.error("❌ Backend failed to start within timeout")
        return False
    
    async def run_initial_tests(self):
        """Run initial system tests"""
        logger.info("🧪 Running initial system tests...")
        
        # Test ML endpoints
        await self.test_ml_endpoints()
        
        # Test yield curve endpoint
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://127.0.0.1:8000/yield-curve') as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Yield curve endpoint working: {len(data['yields'])} yields")
                    else:
                        logger.error(f"❌ Yield curve endpoint failed: {response.status}")
        except Exception as e:
            logger.error(f"❌ Yield curve test failed: {e}")
    
    async def start_system(self):
        """Start the complete ML system"""
        logger.info("🎯 Starting TomThumbVault ML System...")
        
        # Start backend
        if not self.start_backend():
            logger.error("❌ Failed to start backend")
            return False
        
        # Wait for backend to be ready
        if not await self.wait_for_backend():
            logger.error("❌ Backend failed to start")
            return False
        
        # Run initial tests
        await self.run_initial_tests()
        
        # Start scheduler
        if not self.start_scheduler():
            logger.error("❌ Failed to start scheduler")
            return False
        
        logger.info("🎉 TomThumbVault ML System is running!")
        logger.info("📊 Backend API: http://127.0.0.1:8000")
        logger.info("📈 Frontend: http://localhost:3000")
        logger.info("🤖 ML allocations are being updated automatically")
        
        return True
    
    def stop_system(self):
        """Stop the complete system"""
        logger.info("🛑 Stopping TomThumbVault ML System...")
        
        if self.backend_process:
            self.backend_process.terminate()
            logger.info("✅ Backend stopped")
        
        if self.scheduler_process:
            self.scheduler_process.terminate()
            logger.info("✅ Scheduler stopped")
    
    async def monitor_system(self):
        """Monitor system health"""
        logger.info("👁️ Starting system monitoring...")
        
        while True:
            try:
                # Check backend health
                if not await self.check_backend_health():
                    logger.warning("⚠️ Backend health check failed")
                
                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    logger.error("❌ Backend process died")
                    break
                
                if self.scheduler_process and self.scheduler_process.poll() is not None:
                    logger.error("❌ Scheduler process died")
                    break
                
                await asyncio.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(60)

async def main():
    """Main entry point"""
    manager = MLSystemManager()
    
    try:
        # Start the system
        if await manager.start_system():
            # Monitor the system
            await manager.monitor_system()
        else:
            logger.error("❌ Failed to start system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    finally:
        manager.stop_system()
        logger.info("👋 System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 