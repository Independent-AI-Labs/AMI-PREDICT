#!/usr/bin/env python
"""
Download 5 years of cryptocurrency data for all symbols.
Optimized for parallel downloading with checkpoint support.
"""

import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'MATIC/USDT']
YEARS = 5
CHECKPOINT_FILE = 'download_checkpoint.json'


def load_checkpoint():
    """Load download checkpoint."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint):
    """Save download checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def download_symbol_data(symbol, start_date, end_date, checkpoint):
    """Download data for a single symbol."""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # Check if already downloaded
    checkpoint_key = f"{symbol}_{start_date}_{end_date}"
    if checkpoint.get(checkpoint_key, {}).get('completed', False):
        logger.info(f"Skipping {symbol} - already downloaded")
        return None
    
    logger.info(f"Downloading {symbol} from {start_date} to {end_date}")
    
    all_data = []
    timeframe = '1m'
    
    # Resume from checkpoint if exists
    since = checkpoint.get(checkpoint_key, {}).get('last_timestamp', 
                          int(start_date.timestamp() * 1000))
    
    end_timestamp = int(end_date.timestamp() * 1000)
    
    while since < end_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 60000  # Next minute
            
            # Update checkpoint every 10000 records
            if len(all_data) % 10000 == 0:
                checkpoint[checkpoint_key] = {
                    'last_timestamp': since,
                    'records': len(all_data),
                    'completed': False
                }
                save_checkpoint(checkpoint)
                logger.info(f"{symbol}: Downloaded {len(all_data):,} records...")
            
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            time.sleep(5)
            continue
    
    if not all_data:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['symbol'] = symbol.replace('/', '_')
    df = df.drop_duplicates(subset='timestamp')
    df = df.sort_values('timestamp')
    
    # Save to parquet
    output_dir = Path(f'data/5year/{symbol.replace("/", "_")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'{symbol.replace("/", "_")}_{start_date.year}_{end_date.year}.parquet'
    output_file = output_dir / filename
    df.to_parquet(output_file)
    
    # Mark as completed
    checkpoint[checkpoint_key] = {
        'last_timestamp': since,
        'records': len(df),
        'completed': True,
        'file': str(output_file)
    }
    save_checkpoint(checkpoint)
    
    logger.info(f"Saved {len(df):,} records for {symbol} to {output_file}")
    return len(df)


def download_all_data():
    """Download all data in parallel."""
    checkpoint = load_checkpoint()
    
    # Calculate date ranges
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS)
    
    logger.info(f"Downloading {YEARS} years of data for {len(SYMBOLS)} symbols")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Download each symbol in parallel
    total_records = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for symbol in SYMBOLS:
            future = executor.submit(download_symbol_data, symbol, start_date, end_date, checkpoint)
            futures.append((symbol, future))
        
        for symbol, future in futures:
            try:
                records = future.result(timeout=3600)  # 1 hour timeout
                if records:
                    total_records += records
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"DOWNLOAD COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total records: {total_records:,}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Download speed: {total_records/elapsed:.0f} records/second")
    
    # Combine all data into single file for easy loading
    combine_all_data()


def combine_all_data():
    """Combine all downloaded data into single files per symbol."""
    logger.info("\nCombining data files...")
    
    for symbol in SYMBOLS:
        symbol_clean = symbol.replace('/', '_')
        data_dir = Path(f'data/5year/{symbol_clean}')
        
        if not data_dir.exists():
            continue
        
        files = list(data_dir.glob('*.parquet'))
        if not files:
            continue
        
        dfs = []
        for f in files:
            dfs.append(pd.read_parquet(f))
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset='timestamp')
            combined = combined.sort_values('timestamp')
            
            output_file = Path('data/5year') / f'{symbol_clean}_combined.parquet'
            combined.to_parquet(output_file)
            
            logger.info(f"Combined {len(combined):,} records for {symbol} -> {output_file}")


if __name__ == "__main__":
    download_all_data()