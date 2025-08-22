#!/usr/bin/env python
"""
Download minimal data for testing - 30 days of BTC data.
"""

import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

def download_data():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    symbol = 'BTC/USDT'
    timeframe = '1m'
    
    # Get last 30 days
    end = datetime.now()
    start = end - timedelta(days=30)
    
    print(f"Downloading {symbol} from {start} to {end}")
    
    since = int(start.timestamp() * 1000)
    all_data = []
    
    while since < int(end.timestamp() * 1000):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 60000  # Next minute
            
            print(f"Downloaded {len(all_data)} candles...", end='\r')
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp')
    df = df.sort_values('timestamp')
    
    # Save
    output_dir = Path('data/btc_30d')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'btc_usdt_30d.parquet'
    df.to_parquet(output_file)
    
    print(f"\nSaved {len(df)} records to {output_file}")
    return df

if __name__ == "__main__":
    df = download_data()
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")