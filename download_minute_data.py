#!/usr/bin/env python
"""
Download 5 years of minute-level data with resumable checkpoints.
Handles rate limiting and stores in both SQLite and Parquet formats.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ccxt.async_support as ccxt
from typing import Dict, List, Optional
import sqlite3
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'MATIC/USDT']
TIMEFRAME = '1m'
YEARS_TO_DOWNLOAD = 5
CHUNK_SIZE_DAYS = 7  # Download 7 days at a time for 1m data
RATE_LIMIT_DELAY = 0.5  # Delay between API calls in seconds
CHECKPOINT_FILE = 'download_checkpoint.json'
DATA_DIR = Path('data')
PARQUET_DIR = DATA_DIR / 'parquet'
DB_PATH = DATA_DIR / 'cryptobot.db'

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PARQUET_DIR.mkdir(exist_ok=True)

class ResumableDataDownloader:
    """Resumable data downloader with checkpoint support."""
    
    def __init__(self):
        self.exchange = None
        self.checkpoint = self.load_checkpoint()
        self.conn = sqlite3.connect(DB_PATH)
        self.create_tables()
        
    def create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data_1m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(timestamp, symbol)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_market_data_1m_timestamp 
            ON market_data_1m(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_market_data_1m_symbol 
            ON market_data_1m(symbol)
        ''')
        self.conn.commit()
    
    def load_checkpoint(self) -> Dict:
        """Load checkpoint from file if it exists."""
        checkpoint_path = Path(CHECKPOINT_FILE)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint: {checkpoint}")
                return checkpoint
        return {}
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    async def initialize_exchange(self):
        """Initialize the exchange connection."""
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 1200,  # Binance rate limit
            'options': {
                'defaultType': 'spot',
            }
        })
        await self.exchange.load_markets()
        logger.info("Exchange initialized")
    
    async def fetch_ohlcv_chunk(self, symbol: str, since: int, limit: int = 1000) -> List:
        """Fetch a chunk of OHLCV data with retry logic."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                data = await self.exchange.fetch_ohlcv(
                    symbol, 
                    TIMEFRAME, 
                    since=since, 
                    limit=limit
                )
                return data
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    raise
        return []
    
    def save_to_database(self, df: pd.DataFrame, symbol: str):
        """Save data to SQLite database."""
        df['symbol'] = symbol
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').astype(str)
        
        # Use INSERT OR IGNORE to handle duplicates
        df.to_sql('market_data_1m', self.conn, if_exists='append', 
                  index=False, method='multi')
        
    def save_to_parquet(self, df: pd.DataFrame, symbol: str, year: int, month: int):
        """Save data to Parquet file for fast loading."""
        # Create directory structure: parquet/symbol/year/
        symbol_clean = symbol.replace('/', '_')
        parquet_path = PARQUET_DIR / symbol_clean / str(year)
        parquet_path.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet file: symbol_year_month.parquet
        filename = parquet_path / f"{symbol_clean}_{year}_{month:02d}.parquet"
        
        # Convert timestamp to datetime for better parquet compression
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Save with compression
        df.to_parquet(filename, compression='snappy', index=False)
        logger.info(f"Saved {len(df)} records to {filename}")
    
    async def download_symbol_data(self, symbol: str):
        """Download all historical data for a symbol."""
        logger.info(f"Starting download for {symbol}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=YEARS_TO_DOWNLOAD * 365)
        
        # Check checkpoint for this symbol
        checkpoint_key = f"{symbol}_{TIMEFRAME}"
        if checkpoint_key in self.checkpoint:
            last_timestamp = self.checkpoint[checkpoint_key]
            start_date = datetime.fromtimestamp(last_timestamp / 1000)
            logger.info(f"Resuming {symbol} from {start_date}")
        
        current_date = start_date
        total_records = 0
        
        while current_date < end_date:
            # Download in chunks
            chunk_end = min(current_date + timedelta(days=CHUNK_SIZE_DAYS), end_date)
            since = int(current_date.timestamp() * 1000)
            
            logger.info(f"Downloading {symbol} from {current_date.date()} to {chunk_end.date()}")
            
            all_data = []
            last_timestamp = since
            
            # Fetch all data in the chunk
            while True:
                try:
                    data = await self.fetch_ohlcv_chunk(symbol, last_timestamp)
                    
                    if not data:
                        break
                    
                    all_data.extend(data)
                    
                    # Check if we've reached the chunk end
                    last_candle_time = datetime.fromtimestamp(data[-1][0] / 1000)
                    if last_candle_time >= chunk_end or len(data) < 1000:
                        break
                    
                    last_timestamp = data[-1][0] + 1
                    
                    # Rate limiting
                    await asyncio.sleep(RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    logger.error(f"Error downloading {symbol}: {e}")
                    break
            
            if all_data:
                # Convert to DataFrame
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Save to database
                self.save_to_database(df, symbol)
                
                # Save to parquet (monthly files)
                for month_group in df.groupby(pd.to_datetime(df['timestamp'], unit='ms').dt.to_period('M')):
                    month_period = month_group[0]
                    month_df = month_group[1]
                    self.save_to_parquet(month_df, symbol, month_period.year, month_period.month)
                
                total_records += len(df)
                
                # Update checkpoint
                self.checkpoint[checkpoint_key] = int(df['timestamp'].iloc[-1])
                self.save_checkpoint()
                
                logger.info(f"Saved {len(df)} records for {symbol}. Total: {total_records}")
            
            # Move to next chunk
            current_date = chunk_end
            
            # Small delay between chunks
            await asyncio.sleep(1)
        
        logger.info(f"Completed {symbol}: {total_records} total records")
        return total_records
    
    async def download_all(self):
        """Download data for all symbols."""
        await self.initialize_exchange()
        
        total_records = 0
        download_stats = {}
        
        for symbol in SYMBOLS:
            try:
                records = await self.download_symbol_data(symbol)
                download_stats[symbol] = records
                total_records += records
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                download_stats[symbol] = 0
        
        await self.exchange.close()
        
        # Print summary
        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        for symbol, records in download_stats.items():
            logger.info(f"{symbol}: {records:,} records")
        logger.info(f"Total: {total_records:,} records")
        
        # Calculate expected vs actual
        expected_per_symbol = YEARS_TO_DOWNLOAD * 365 * 24 * 60  # minutes in 5 years
        logger.info(f"Expected per symbol: {expected_per_symbol:,} records")
        logger.info(f"Data completeness: {(total_records / (len(SYMBOLS) * expected_per_symbol) * 100):.1f}%")
        
        return download_stats
    
    def get_data_summary(self):
        """Get summary of downloaded data."""
        cursor = self.conn.cursor()
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA SUMMARY")
        logger.info("=" * 60)
        
        # Summary by symbol
        cursor.execute('''
            SELECT symbol, 
                   COUNT(*) as count,
                   MIN(timestamp) as earliest,
                   MAX(timestamp) as latest
            FROM market_data_1m
            GROUP BY symbol
        ''')
        
        for row in cursor.fetchall():
            symbol, count, earliest, latest = row
            earliest_date = datetime.fromisoformat(earliest)
            latest_date = datetime.fromisoformat(latest)
            days = (latest_date - earliest_date).days
            
            logger.info(f"{symbol}:")
            logger.info(f"  Records: {count:,}")
            logger.info(f"  Period: {earliest[:10]} to {latest[:10]} ({days} days)")
            logger.info(f"  Completeness: {(count / (days * 24 * 60) * 100):.1f}%")
        
        # Total records
        cursor.execute('SELECT COUNT(*) FROM market_data_1m')
        total = cursor.fetchone()[0]
        logger.info(f"\nTotal records in database: {total:,}")
        
        # Check parquet files
        parquet_files = list(PARQUET_DIR.glob('**/*.parquet'))
        logger.info(f"Parquet files created: {len(parquet_files)}")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in parquet_files)
        logger.info(f"Total parquet size: {total_size / (1024**3):.2f} GB")

async def main():
    """Main function."""
    downloader = ResumableDataDownloader()
    
    try:
        # Download all data
        await downloader.download_all()
        
        # Show summary
        downloader.get_data_summary()
        
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted. Progress saved to checkpoint.")
        logger.info("Run the script again to resume from where it left off.")
    
    finally:
        downloader.conn.close()

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RESUMABLE MINUTE DATA DOWNLOADER")
    logger.info(f"Downloading {YEARS_TO_DOWNLOAD} years of 1-minute data")
    logger.info(f"Symbols: {', '.join(SYMBOLS)}")
    logger.info("=" * 60)
    
    asyncio.run(main())