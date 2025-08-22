"""
Market data simulator
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta
from loguru import logger

from ..core import ConfigManager


class MarketSimulator:
    """Simulates realistic market data"""
    
    def __init__(self, config: ConfigManager):
        """Initialize market simulator
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.log = logger.bind(name=__name__)
        
        self.pairs = config.get('trading.pairs', [])
        
        # Base prices for simulation
        self.base_prices = {
            'BTC/USDT': 43000,
            'ETH/USDT': 2300,
            'BNB/USDT': 315,
            'ADA/USDT': 0.58,
            'SOL/USDT': 98
        }
        
        # Current prices
        self.current_prices = self.base_prices.copy()
        
        # Market parameters
        self.volatility = {
            'BTC/USDT': 0.02,
            'ETH/USDT': 0.025,
            'BNB/USDT': 0.03,
            'ADA/USDT': 0.035,
            'SOL/USDT': 0.04
        }
        
        # Trend parameters
        self.trends = {pair: 0 for pair in self.pairs}
        self.trend_strength = 0.0001
        
        # Volume parameters
        self.base_volumes = {
            'BTC/USDT': 1000000000,
            'ETH/USDT': 500000000,
            'BNB/USDT': 100000000,
            'ADA/USDT': 50000000,
            'SOL/USDT': 75000000
        }
        
        self.timestamp = datetime.now()
    
    async def generate_tick(self) -> Dict[str, pd.DataFrame]:
        """Generate next market tick
        
        Returns:
            Market data for all pairs
        """
        market_data = {}
        
        for pair in self.pairs:
            if pair not in self.current_prices:
                continue
            
            # Generate price movement
            price_data = self._generate_price_movement(pair)
            
            # Create DataFrame
            df = pd.DataFrame([{
                'timestamp': self.timestamp,
                'open': price_data['open'],
                'high': price_data['high'],
                'low': price_data['low'],
                'close': price_data['close'],
                'volume': price_data['volume']
            }])
            
            market_data[pair] = df
        
        # Advance timestamp
        self.timestamp += timedelta(minutes=1)
        
        return market_data
    
    def _generate_price_movement(self, pair: str) -> Dict[str, float]:
        """Generate realistic price movement
        
        Args:
            pair: Trading pair
            
        Returns:
            Price data
        """
        current_price = self.current_prices[pair]
        volatility = self.volatility.get(pair, 0.02)
        
        # Generate random walk with trend
        returns = np.random.normal(self.trends[pair], volatility)
        
        # Apply momentum
        if random.random() > 0.7:
            returns *= 1.5  # Occasional larger moves
        
        # Calculate new price
        new_price = current_price * (1 + returns)
        
        # Generate OHLC
        open_price = current_price
        close_price = new_price
        
        # High and low with some randomness
        price_range = abs(new_price - current_price) * random.uniform(1.2, 1.8)
        
        if new_price > current_price:
            high_price = new_price + price_range * random.uniform(0, 0.3)
            low_price = min(current_price, new_price) - price_range * random.uniform(0, 0.1)
        else:
            high_price = max(current_price, new_price) + price_range * random.uniform(0, 0.1)
            low_price = new_price - price_range * random.uniform(0, 0.3)
        
        # Generate volume
        base_volume = self.base_volumes.get(pair, 1000000)
        volume = base_volume * random.uniform(0.5, 1.5)
        
        # Occasionally generate spikes
        if random.random() > 0.95:
            volume *= random.uniform(2, 4)
        
        # Update current price
        self.current_prices[pair] = new_price
        
        # Update trend (mean reversion)
        self.trends[pair] = self.trends[pair] * 0.95 + random.uniform(-self.trend_strength, self.trend_strength)
        
        return {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
    
    def get_current_price(self, pair: str) -> float:
        """Get current price for a pair
        
        Args:
            pair: Trading pair
            
        Returns:
            Current price
        """
        return self.current_prices.get(pair, 0)
    
    def set_market_regime(self, regime: str):
        """Set market regime for simulation
        
        Args:
            regime: Market regime (trending_bull, trending_bear, sideways, high_volatility)
        """
        self.log.info(f"Setting market regime to: {regime}")
        
        if regime == 'trending_bull':
            for pair in self.trends:
                self.trends[pair] = random.uniform(0.0001, 0.0003)
        elif regime == 'trending_bear':
            for pair in self.trends:
                self.trends[pair] = random.uniform(-0.0003, -0.0001)
        elif regime == 'sideways':
            for pair in self.trends:
                self.trends[pair] = random.uniform(-0.00005, 0.00005)
        elif regime == 'high_volatility':
            for pair in self.volatility:
                self.volatility[pair] *= 1.5
    
    def create_flash_crash(self, pair: str, magnitude: float = 0.1):
        """Simulate a flash crash
        
        Args:
            pair: Trading pair
            magnitude: Crash magnitude (percentage)
        """
        if pair in self.current_prices:
            self.log.warning(f"Flash crash initiated for {pair}: -{magnitude*100:.1f}%")
            self.current_prices[pair] *= (1 - magnitude)