#!/usr/bin/env python
"""Sync Binance data to local database"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import ConfigManager, Database
from src.data_providers.data_sync import DataSyncManager


async def main():
    """Main sync function"""
    print("=" * 60)
    print("BINANCE DATA SYNC TO DATABASE")
    print("=" * 60)
    
    # Initialize components
    config = ConfigManager()
    database = Database(config.get('database'))
    sync_manager = DataSyncManager(config, database)
    
    # Get trading pairs from config
    symbols = config.get('trading.pairs', ['BTC/USDT', 'ETH/USDT'])
    timeframes = ['1h', '1d']  # Start with hourly and daily data
    
    print(f"\nSymbols to sync: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    
    try:
        # 1. Sync historical data (last 7 days)
        print("\n1. Syncing Historical Data (7 days)...")
        print("-" * 40)
        
        await sync_manager.sync_historical_data(
            symbols=symbols,
            timeframes=timeframes,
            days_back=7
        )
        
        # 2. Verify data in database
        print("\n2. Verifying Database Content...")
        print("-" * 40)
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Check what we have in the database
                start_date = datetime.now() - timedelta(days=7)
                df = database.get_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_date
                )
                
                if not df.empty:
                    print(f"   {symbol:10} {timeframe:3} - {len(df):3} candles")
                    print(f"      Latest: {df.index[-1]} Close: ${df['close'].iloc[-1]:,.2f}")
                else:
                    print(f"   {symbol:10} {timeframe:3} - No data")
        
        # 3. Update with latest data
        print("\n3. Fetching Latest Updates...")
        print("-" * 40)
        
        await sync_manager.update_latest_data(
            symbols=symbols,
            timeframe='1m',
            limit=10
        )
        
        print("   Latest 1-minute data updated")
        
        # 4. Show summary statistics
        print("\n4. Database Summary:")
        print("-" * 40)
        
        session = database.get_session()
        try:
            from src.core.database import MarketData
            
            # Count total records
            total_records = session.query(MarketData).count()
            
            # Count by symbol
            for symbol in symbols:
                count = session.query(MarketData).filter(
                    MarketData.symbol == symbol
                ).count()
                print(f"   {symbol}: {count} total records")
            
            print(f"\n   Total database records: {total_records}")
            
        finally:
            session.close()
        
        print("\n" + "=" * 60)
        print("DATA SYNC COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Show next steps
        print("\nNext Steps:")
        print("-" * 40)
        print("1. Data is now available in the local database")
        print("2. The trading engine can use this data for:")
        print("   - Backtesting strategies")
        print("   - Training ML models")
        print("   - Generating technical indicators")
        print("3. To sync more data, modify days_back parameter")
        print("4. To add real-time streaming, use start_real_time_sync()")
        
    except Exception as e:
        print(f"\n[ERROR] Sync failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        await sync_manager.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))