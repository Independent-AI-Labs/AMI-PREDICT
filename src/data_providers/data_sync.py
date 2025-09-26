"""Data synchronization module for fetching and storing market data"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from ..core import ConfigManager, Database
from .binance_provider import BinanceDataProvider


class DataSyncManager:
    """Manages data synchronization from providers to database"""

    def __init__(self, config: ConfigManager, database: Database):
        """Initialize data sync manager

        Args:
            config: Configuration manager
            database: Database instance
        """
        self.config = config
        self.database = database
        self.log = logger.bind(name=__name__)

        # Initialize data providers
        self.providers = {"binance": BinanceDataProvider(config.config)}

        # Get default provider
        self.default_provider = config.get("data_providers.default", "binance")

    async def sync_historical_data(self, symbols: list[str], timeframes: list[str], days_back: int = 30, provider: Optional[str] = None):
        """Sync historical data for multiple symbols and timeframes

        Args:
            symbols: List of trading pairs
            timeframes: List of timeframes
            days_back: Number of days to fetch
            provider: Data provider to use
        """
        provider_name = provider or self.default_provider
        data_provider = self.providers.get(provider_name)

        if not data_provider:
            raise ValueError(f"Provider {provider_name} not found")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        self.log.info(f"Starting historical data sync for {len(symbols)} symbols, {days_back} days")

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    self.log.info(f"Syncing {symbol} {timeframe}")

                    # Fetch data
                    df = await data_provider.fetch_ohlcv(symbol=symbol, timeframe=timeframe, start=start_date, end=end_date)

                    if not df.empty:
                        # Add timestamp column from index
                        df["timestamp"] = df.index

                        # Save to database
                        self.database.save_market_data(df, symbol, timeframe)

                        self.log.info(f"Saved {len(df)} candles for {symbol} {timeframe}")
                    else:
                        self.log.warning(f"No data received for {symbol} {timeframe}")

                    # Rate limiting
                    await asyncio.sleep(0.2)

                except Exception as e:
                    self.log.error(f"Error syncing {symbol} {timeframe}: {e}")
                    continue

        self.log.info("Historical data sync completed")

    async def start_real_time_sync(self, symbols: list[str], timeframes: list[str], provider: Optional[str] = None):
        """Start real-time data synchronization

        Args:
            symbols: List of trading pairs
            timeframes: List of timeframes
            provider: Data provider to use
        """
        provider_name = provider or self.default_provider
        data_provider = self.providers.get(provider_name)

        if not data_provider:
            raise ValueError(f"Provider {provider_name} not found")

        self.log.info(f"Starting real-time sync for {len(symbols)} symbols")

        # Create callback for handling real-time data
        async def handle_candle(candle_data):
            """Handle incoming candle data"""
            try:
                # Only save closed candles
                if candle_data.get("is_closed"):
                    df = pd.DataFrame(
                        [
                            {
                                "timestamp": candle_data["timestamp"],
                                "open": candle_data["open"],
                                "high": candle_data["high"],
                                "low": candle_data["low"],
                                "close": candle_data["close"],
                                "volume": candle_data["volume"],
                            }
                        ]
                    )
                    df.set_index("timestamp", inplace=True)

                    # Save to database
                    self.database.save_market_data(
                        df,
                        candle_data["symbol"],
                        "1m",  # Assuming 1m for real-time
                    )

                    self.log.debug(f"Saved real-time candle for {candle_data['symbol']}")

            except Exception as e:
                self.log.error(f"Error handling real-time data: {e}")

        # Start streaming for each symbol
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                task = asyncio.create_task(data_provider.stream_ohlcv(symbol, timeframe, handle_candle))
                tasks.append(task)

        self.log.info(f"Started {len(tasks)} streaming tasks")

        # Keep running until cancelled
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.log.info("Real-time sync stopped")
            for task in tasks:
                task.cancel()

    async def update_latest_data(self, symbols: list[str], timeframe: str = "1m", limit: int = 100):
        """Update latest data for symbols

        Args:
            symbols: List of trading pairs
            timeframe: Timeframe to update
            limit: Number of recent candles
        """
        provider = self.providers[self.default_provider]

        for symbol in symbols:
            try:
                # Fetch latest data
                df = await provider.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)

                if not df.empty:
                    # Add timestamp column
                    df["timestamp"] = df.index

                    # Save to database
                    self.database.save_market_data(df, symbol, timeframe)

                    self.log.debug(f"Updated {len(df)} candles for {symbol}")

            except Exception as e:
                self.log.error(f"Error updating {symbol}: {e}")

    async def get_data_gaps(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[tuple]:
        """Find gaps in historical data

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start: Start date
            end: End date

        Returns:
            List of (gap_start, gap_end) tuples
        """
        # Get existing data from database
        existing_data = self.database.get_market_data(symbol=symbol, timeframe=timeframe, start=start, end=end)

        if existing_data.empty:
            return [(start, end)]

        # Find gaps
        gaps = []
        freq_map = {"1m": "T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "H", "4h": "4H", "1d": "D"}

        freq = freq_map.get(timeframe, "T")
        expected_index = pd.date_range(start=start, end=end, freq=freq)
        missing_timestamps = expected_index.difference(existing_data.index)

        if len(missing_timestamps) > 0:
            # Group consecutive missing timestamps into gaps
            # This is simplified - you might want more sophisticated gap detection
            gaps.append((missing_timestamps[0], missing_timestamps[-1]))

        return gaps

    async def fill_data_gaps(self, symbol: str, timeframe: str, start: datetime, end: datetime):
        """Fill gaps in historical data

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start: Start date
            end: End date
        """
        gaps = await self.get_data_gaps(symbol, timeframe, start, end)

        if not gaps:
            self.log.info(f"No gaps found for {symbol} {timeframe}")
            return

        provider = self.providers[self.default_provider]

        for gap_start, gap_end in gaps:
            try:
                self.log.info(f"Filling gap for {symbol} from {gap_start} to {gap_end}")

                df = await provider.fetch_ohlcv(symbol=symbol, timeframe=timeframe, start=gap_start, end=gap_end)

                if not df.empty:
                    df["timestamp"] = df.index
                    self.database.save_market_data(df, symbol, timeframe)
                    self.log.info(f"Filled {len(df)} candles")

            except Exception as e:
                self.log.error(f"Error filling gap: {e}")

    async def close(self):
        """Close all connections"""
        for provider in self.providers.values():
            if hasattr(provider, "close"):
                await provider.close()
