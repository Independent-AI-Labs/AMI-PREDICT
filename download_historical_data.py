#!/usr/bin/env python
"""
Download 1+ year of historical data for comprehensive training
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import ConfigManager, Database
from src.data_providers.data_sync import DataSyncManager

async def download_long_term_data():
    """Download 1+ year of historical data"""
    print("=" * 80)
    print("DOWNLOADING 1+ YEAR HISTORICAL DATA")
    print("=" * 80)
    
    # Initialize components
    config = ConfigManager()
    database = Database(config.get('database'))
    sync_manager = DataSyncManager(config, database)
    
    # Configuration
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']  # Multiple timeframes
    days_back = 400  # ~13 months of data
    
    print(f"\nConfiguration:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Period: {days_back} days (~{days_back/30:.1f} months)")
    print("-" * 80)
    
    try:
        # Download data for each symbol and timeframe
        for symbol in symbols:
            print(f"\nDownloading {symbol}...")
            
            for timeframe in timeframes:
                print(f"  {timeframe}: ", end="", flush=True)
                
                try:
                    # Download in chunks to avoid rate limits
                    if timeframe in ['1m', '5m']:
                        # For minute data, download in smaller chunks (30 days at a time)
                        chunks = []
                        for i in range(0, days_back, 30):
                            start_date = datetime.now() - timedelta(days=min(i+30, days_back))
                            end_date = datetime.now() - timedelta(days=i)
                            
                            await sync_manager.sync_historical_data(
                                symbols=[symbol],
                                timeframes=[timeframe],
                                start_date=start_date,
                                end_date=end_date
                            )
                            print(".", end="", flush=True)
                            await asyncio.sleep(0.5)  # Rate limiting
                    else:
                        # For larger timeframes, can download all at once
                        await sync_manager.sync_historical_data(
                            symbols=[symbol],
                            timeframes=[timeframe],
                            days_back=days_back
                        )
                    
                    # Verify downloaded data
                    session = database.get_session()
                    try:
                        from src.core.database import MarketData
                        count = session.query(MarketData).filter(
                            MarketData.symbol == symbol,
                            MarketData.timeframe == timeframe
                        ).count()
                        print(f" OK {count} candles")
                    finally:
                        session.close()
                        
                except Exception as e:
                    print(f" ERROR: {e}")
                    continue
                
                # Rate limiting between requests
                await asyncio.sleep(1)
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("DOWNLOAD SUMMARY")
        print("=" * 80)
        
        session = database.get_session()
        try:
            from src.core.database import MarketData
            
            print("\nData by Symbol and Timeframe:")
            print("-" * 40)
            
            total_records = 0
            for symbol in symbols:
                print(f"\n{symbol}:")
                for timeframe in timeframes:
                    count = session.query(MarketData).filter(
                        MarketData.symbol == symbol,
                        MarketData.timeframe == timeframe
                    ).count()
                    if count > 0:
                        # Get date range
                        oldest = session.query(MarketData.timestamp).filter(
                            MarketData.symbol == symbol,
                            MarketData.timeframe == timeframe
                        ).order_by(MarketData.timestamp).first()
                        
                        newest = session.query(MarketData.timestamp).filter(
                            MarketData.symbol == symbol,
                            MarketData.timeframe == timeframe
                        ).order_by(MarketData.timestamp.desc()).first()
                        
                        if oldest and newest:
                            days = (newest[0] - oldest[0]).days
                            print(f"  {timeframe:3}: {count:6} candles ({days} days)")
                        else:
                            print(f"  {timeframe:3}: {count:6} candles")
                    total_records += count
            
            print(f"\nTotal records in database: {total_records:,}")
            
        finally:
            session.close()
        
        print("\n[SUCCESS] Historical data download completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Error downloading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        await sync_manager.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(download_long_term_data()))