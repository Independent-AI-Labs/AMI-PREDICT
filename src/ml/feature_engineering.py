"""
Feature engineering for trading models - focused single responsibility
"""
import numpy as np
import pandas as pd
from typing import List

class PriceFeatures:
    """Extract price-based features"""
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['close'].shift(1)
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        return df

class VolumeFeatures:
    """Extract volume-based features"""
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_change'] = df['volume'].pct_change()
        return df

class MomentumFeatures:
    """Extract momentum indicators"""
    
    def compute(self, df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Rate of change
        df['roc'] = df['close'].pct_change(periods=10) * 100
        
        return df

class MovingAverages:
    """Calculate moving averages"""
    
    def __init__(self, periods: List[int] = [5, 10, 20, 50]):
        self.periods = periods
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        return df