#!/usr/bin/env python3
"""
Scheduler to automatically update vault allocations based on ML strategy
Runs every hour to keep the vault optimized
"""

import asyncio
import schedule
import time
import logging
from datetime import datetime
from update_vault_allocations import VaultAllocationUpdater

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vault_scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class VaultScheduler:
    def __init__(self):
        self.updater = VaultAllocationUpdater()
        self.is_running = False
        
    async def update_allocations_job(self):
        """Job to update vault allocations"""
        logger.info("🔄 Starting scheduled allocation update...")
        
        try:
            success = await self.updater.run_allocation_update()
            
            if success:
                logger.info("✅ Scheduled allocation update completed successfully")
            else:
                logger.error("❌ Scheduled allocation update failed")
                
        except Exception as e:
            logger.error(f"❌ Error in scheduled allocation update: {e}")
    
    async def health_check_job(self):
        """Job to check system health"""
        logger.info("🏥 Performing health check...")
        
        try:
            # Check if backend is running
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://127.0.0.1:8000/health') as response:
                    if response.status == 200:
                        logger.info("✅ Backend health check passed")
                    else:
                        logger.warning("⚠️ Backend health check failed")
                        
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
    
    def run_scheduler(self):
        """Run the scheduler"""
        logger.info("🚀 Starting vault allocation scheduler...")
        
        # Schedule jobs
        schedule.every().hour.do(lambda: asyncio.create_task(self.update_allocations_job()))
        schedule.every(30).minutes.do(lambda: asyncio.create_task(self.health_check_job()))
        
        # Run initial update
        asyncio.create_task(self.update_allocations_job())
        
        self.is_running = True
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        logger.info("🛑 Stopping vault allocation scheduler...")
        self.is_running = False

async def main():
    """Main entry point"""
    scheduler = VaultScheduler()
    
    try:
        # Run scheduler in a separate thread
        import threading
        scheduler_thread = threading.Thread(target=scheduler.run_scheduler)
        scheduler_thread.start()
        
        logger.info("📅 Scheduler started. Press Ctrl+C to stop.")
        
        # Keep the main thread alive
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
        scheduler.stop_scheduler()
        logger.info("👋 Scheduler stopped")

if __name__ == "__main__":
    asyncio.run(main()) 