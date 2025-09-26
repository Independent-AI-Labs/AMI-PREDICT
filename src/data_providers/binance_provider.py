"""Binance data provider implementation"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import ccxt.async_support as ccxt
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from binance.client import Client
from loguru import logger

from .base_provider import BaseDataProvider


class BinanceDataProvider(BaseDataProvider):
    """Binance data provider for historical and real-time data"""

    # Timeframe mappings
    TIMEFRAME_MAP = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "3m": Client.KLINE_INTERVAL_3MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "2h": Client.KLINE_INTERVAL_2HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "6h": Client.KLINE_INTERVAL_6HOUR,
        "8h": Client.KLINE_INTERVAL_8HOUR,
        "12h": Client.KLINE_INTERVAL_12HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
        "3d": Client.KLINE_INTERVAL_3DAY,
        "1w": Client.KLINE_INTERVAL_1WEEK,
        "1M": Client.KLINE_INTERVAL_1MONTH,
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize Binance data provider

        Args:
            config: Configuration with optional api_key and api_secret
        """
        super().__init__(config)

        # API credentials (optional for public data)
        self.api_key = config.get("binance", {}).get("api_key", "")
        self.api_secret = config.get("binance", {}).get("api_secret", "")

        # Initialize sync client (public data doesn't require keys)
        self.client = Client(self.api_key, self.api_secret)

        # Initialize async client for streaming
        self.async_client = None
        self.socket_manager = None

        # Initialize CCXT for additional features
        self.ccxt_exchange = ccxt.binance({"apiKey": self.api_key, "secret": self.api_secret, "enableRateLimit": True, "options": {"defaultType": "spot"}})

        # Cache for symbols and exchange info
        self._symbols_cache = None
        self._exchange_info = None

        logger.info("Binance data provider initialized")

    async def initialize_async(self):
        """Initialize async components"""
        if not self.async_client:
            self.async_client = await AsyncClient.create(self.api_key, self.api_secret)
            self.socket_manager = BinanceSocketManager(self.async_client)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, start: Optional[datetime] = None, end: Optional[datetime] = None, limit: Optional[int] = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT' or 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '1d')
            start: Start datetime
            end: End datetime
            limit: Maximum number of candles (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert symbol format (BTC/USDT -> BTCUSDT)
            binance_symbol = symbol.replace("/", "")

            # Convert timeframe
            if timeframe not in self.TIMEFRAME_MAP:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            interval = self.TIMEFRAME_MAP[timeframe]

            # Prepare time parameters
            start_str = None
            end_str = None

            if start:
                start_str = str(int(start.timestamp() * 1000))
            if end:
                end_str = str(int(end.timestamp() * 1000))

            # Fetch klines
            if start_str and end_str:
                klines = self.client.get_historical_klines(binance_symbol, interval, start_str=start_str, end_str=end_str, limit=limit)
            elif start_str:
                klines = self.client.get_historical_klines(binance_symbol, interval, start_str=start_str, limit=limit)
            else:
                # Get recent klines
                klines = self.client.get_klines(symbol=binance_symbol, interval=interval, limit=min(limit, 1000))

            # Convert to DataFrame
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades_count",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = df[col].astype(float)

            for col in ["trades_count"]:
                df[col] = df[col].astype(int)

            # Set timestamp as index
            df.set_index("timestamp", inplace=True)

            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker data

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data dictionary
        """
        try:
            # Convert symbol format
            binance_symbol = symbol.replace("/", "")

            # Get 24hr ticker
            ticker = self.client.get_ticker(symbol=binance_symbol)

            return {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "bid": float(ticker["bidPrice"]),
                "ask": float(ticker["askPrice"]),
                "last": float(ticker["lastPrice"]),
                "open": float(ticker["openPrice"]),
                "high": float(ticker["highPrice"]),
                "low": float(ticker["lowPrice"]),
                "close": float(ticker["lastPrice"]),
                "volume": float(ticker["volume"]),
                "quote_volume": float(ticker["quoteVolume"]),
                "change": float(ticker["priceChange"]),
                "percentage": float(ticker["priceChangePercent"]),
            }

        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            raise

    async def fetch_order_book(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Fetch order book

        Args:
            symbol: Trading pair symbol
            limit: Depth of order book (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Order book data
        """
        try:
            # Convert symbol format
            binance_symbol = symbol.replace("/", "")

            # Get order book
            depth = self.client.get_order_book(symbol=binance_symbol, limit=limit)

            return {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "bids": [[float(price), float(qty)] for price, qty in depth["bids"]],
                "asks": [[float(price), float(qty)] for price, qty in depth["asks"]],
            }

        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            raise

    async def stream_trades(self, symbol: str, callback: callable):
        """Stream real-time trades

        Args:
            symbol: Trading pair symbol
            callback: Function to call with trade data
        """
        await self.initialize_async()

        # Convert symbol format
        binance_symbol = symbol.replace("/", "").lower()

        # Start trade socket
        trade_socket = self.socket_manager.trade_socket(binance_symbol)

        async with trade_socket as stream:
            while True:
                msg = await stream.recv()

                # Process trade data
                trade = {
                    "symbol": symbol,
                    "timestamp": datetime.fromtimestamp(msg["T"] / 1000),
                    "trade_id": msg["t"],
                    "price": float(msg["p"]),
                    "amount": float(msg["q"]),
                    "side": "buy" if msg["m"] else "sell",
                }

                await callback(trade)

    async def stream_ohlcv(self, symbol: str, timeframe: str, callback: callable):
        """Stream real-time OHLCV data

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            callback: Function to call with OHLCV data
        """
        await self.initialize_async()

        # Convert symbol format
        binance_symbol = symbol.replace("/", "").lower()

        # Convert timeframe
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        interval = self.TIMEFRAME_MAP[timeframe]

        # Start kline socket
        kline_socket = self.socket_manager.kline_socket(binance_symbol, interval)

        async with kline_socket as stream:
            while True:
                msg = await stream.recv()

                # Process kline data
                k = msg["k"]
                candle = {
                    "symbol": symbol,
                    "timestamp": datetime.fromtimestamp(k["t"] / 1000),
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"]),
                    "is_closed": k["x"],  # Whether this kline is closed
                }

                await callback(candle)

    def get_supported_symbols(self) -> list[str]:
        """Get list of supported trading pairs

        Returns:
            List of symbol strings
        """
        if not self._symbols_cache:
            # Get exchange info
            if not self._exchange_info:
                self._exchange_info = self.client.get_exchange_info()

            # Extract active trading pairs
            self._symbols_cache = [s["symbol"] for s in self._exchange_info["symbols"] if s["status"] == "TRADING"]

        return self._symbols_cache

    def get_supported_timeframes(self) -> list[str]:
        """Get list of supported timeframes

        Returns:
            List of timeframe strings
        """
        return list(self.TIMEFRAME_MAP.keys())

    async def download_historical_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime, save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Download large historical dataset

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            save_path: Optional path to save CSV

        Returns:
            Complete DataFrame
        """
        logger.info(f"Downloading historical data for {symbol} from {start_date} to {end_date}")

        all_data = []
        current_start = start_date

        while current_start < end_date:
            # Fetch batch (max 1000 candles per request)
            batch = await self.fetch_ohlcv(symbol=symbol, timeframe=timeframe, start=current_start, limit=1000)

            if batch.empty:
                break

            all_data.append(batch)

            # Move to next batch
            current_start = batch.index[-1] + timedelta(minutes=1)

            # Rate limiting
            await asyncio.sleep(0.1)

            logger.debug(f"Downloaded up to {current_start}")

        # Combine all batches
        if all_data:
            df = pd.concat(all_data)
            df = df[~df.index.duplicated(keep="first")]
            df.sort_index(inplace=True)

            # Save if path provided
            if save_path:
                df.to_csv(save_path)
                logger.info(f"Saved {len(df)} candles to {save_path}")

            return df

        return pd.DataFrame()

    async def close(self):
        """Close connections"""
        if self.async_client:
            await self.async_client.close_connection()
            self.async_client = None
            self.socket_manager = None

        if self.ccxt_exchange:
            await self.ccxt_exchange.close()

    def get_exchange_info(self) -> dict[str, Any]:
        """Get exchange information

        Returns:
            Exchange info including limits, fees, etc.
        """
        if not self._exchange_info:
            self._exchange_info = self.client.get_exchange_info()
        return self._exchange_info

    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get specific symbol information

        Args:
            symbol: Trading pair

        Returns:
            Symbol info including filters, limits
        """
        binance_symbol = symbol.replace("/", "")
        info = self.get_exchange_info()

        for s in info["symbols"]:
            if s["symbol"] == binance_symbol:
                return s

        raise ValueError(f"Symbol {symbol} not found")
