"""
Investigate why our trading system is failing
"""
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality():
    """Check for data quality issues"""
    print("="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    conn = sqlite3.connect('data/crypto_5years.db')
    
    # Sample data
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM market_data_1m
    WHERE symbol = 'BTC/USDT'
    AND timestamp >= '2023-01-01'
    AND timestamp <= '2023-01-31'
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
    conn.close()
    
    df = df.set_index('timestamp')
    
    # Check for missing data
    print(f"\n1. Data Shape: {df.shape}")
    print(f"   Expected minutes in January: {31*24*60} = {31*24*60}")
    print(f"   Data completeness: {len(df)/(31*24*60)*100:.1f}%")
    
    # Check for gaps
    time_diffs = df.index.to_series().diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=1)]
    print(f"\n2. Time Gaps Found: {len(gaps)}")
    if len(gaps) > 0:
        print(f"   Largest gap: {gaps.max()}")
        print(f"   Average gap: {gaps.mean()}")
    
    # Check for data anomalies
    print("\n3. Price Anomalies:")
    
    # Check if high >= low
    invalid_hl = df[df['high'] < df['low']]
    print(f"   High < Low violations: {len(invalid_hl)}")
    
    # Check if close is within high-low range
    invalid_close = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
    print(f"   Close outside H-L range: {len(invalid_close)}")
    
    # Check for zero or negative prices
    zero_prices = df[(df['close'] <= 0) | (df['open'] <= 0)]
    print(f"   Zero/negative prices: {len(zero_prices)}")
    
    # Check for extreme price movements
    df['returns'] = df['close'].pct_change()
    extreme_moves = df[abs(df['returns']) > 0.10]  # 10% moves in 1 minute
    print(f"   Extreme moves (>10% in 1min): {len(extreme_moves)}")
    
    # Check volume
    print("\n4. Volume Analysis:")
    zero_volume = df[df['volume'] == 0]
    print(f"   Zero volume bars: {len(zero_volume)}")
    print(f"   Average volume: {df['volume'].mean():.2f}")
    print(f"   Volume std dev: {df['volume'].std():.2f}")
    
    # Statistical analysis
    print("\n5. Price Statistics:")
    print(f"   Mean price: ${df['close'].mean():.2f}")
    print(f"   Std dev: ${df['close'].std():.2f}")
    print(f"   Min price: ${df['close'].min():.2f}")
    print(f"   Max price: ${df['close'].max():.2f}")
    print(f"   Price range: ${df['close'].max() - df['close'].min():.2f}")
    
    return df

def analyze_prediction_distribution():
    """Analyze what our model is actually predicting"""
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    # Load a backtest result to analyze predictions
    try:
        with open('experiments/backtest_AttentionTCN_20250822_153701.json', 'r') as f:
            data = json.load(f)
        
        print("\n1. Backtest Results Summary:")
        print(f"   Total Return: {data['metrics']['total_return']:.2f}%")
        print(f"   Win Rate: {data['metrics']['win_rate']:.2f}%")
        print(f"   Total Trades: {data['metrics']['total_trades']}")
        print(f"   Sharpe Ratio: {data['metrics']['sharpe_ratio']:.2f}")
        
        # Analyze equity curve
        if 'equity_curve_samples' in data:
            equity = data['equity_curve_samples']
            print(f"\n2. Equity Curve Analysis:")
            print(f"   Starting: ${equity[0]:,.2f}")
            print(f"   Ending: ${equity[-1]:,.2f}")
            print(f"   Samples: {len(equity)}")
            
            # Calculate drawdowns
            equity_arr = np.array(equity)
            running_max = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - running_max) / running_max * 100
            print(f"   Max Drawdown: {drawdown.min():.2f}%")
            print(f"   Current Drawdown: {drawdown[-1]:.2f}%")
    
    except Exception as e:
        print(f"Could not load backtest results: {e}")

def test_buy_and_hold():
    """Test simple buy and hold strategy as baseline"""
    print("\n" + "="*60)
    print("BUY AND HOLD BASELINE")
    print("="*60)
    
    conn = sqlite3.connect('data/crypto_5years.db')
    
    # Test each quarter of 2023
    quarters = [
        ('2023-01-01', '2023-03-31', 'Q1'),
        ('2023-04-01', '2023-06-30', 'Q2'),
        ('2023-07-01', '2023-09-30', 'Q3'),
        ('2023-10-01', '2023-12-31', 'Q4'),
    ]
    
    results = []
    
    for start, end, name in quarters:
        query = f"""
        SELECT MIN(close) as first_price, MAX(close) as last_price
        FROM (
            SELECT close, ROW_NUMBER() OVER (ORDER BY timestamp) as rn,
                   COUNT(*) OVER () as total
            FROM market_data_1m
            WHERE symbol = 'BTC/USDT'
            AND timestamp >= '{start}'
            AND timestamp <= '{end}'
        )
        WHERE rn = 1 OR rn = total
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) >= 1:
            # Get first and last prices more reliably
            query2 = f"""
            SELECT close FROM market_data_1m
            WHERE symbol = 'BTC/USDT'
            AND timestamp >= '{start}'
            AND timestamp <= '{end}'
            ORDER BY timestamp
            LIMIT 1
            """
            first_price = pd.read_sql_query(query2, conn)['close'].iloc[0]
            
            query3 = f"""
            SELECT close FROM market_data_1m
            WHERE symbol = 'BTC/USDT'
            AND timestamp >= '{start}'
            AND timestamp <= '{end}'
            ORDER BY timestamp DESC
            LIMIT 1
            """
            last_price = pd.read_sql_query(query3, conn)['close'].iloc[0]
            
            returns = (last_price - first_price) / first_price * 100
            
            print(f"\n{name} 2023:")
            print(f"   Start Price: ${first_price:,.2f}")
            print(f"   End Price: ${last_price:,.2f}")
            print(f"   Return: {returns:.2f}%")
            
            results.append(returns)
    
    conn.close()
    
    print(f"\n2023 Full Year:")
    print(f"   Average Quarterly Return: {np.mean(results):.2f}%")
    print(f"   Total Return (compound): {np.prod([1 + r/100 for r in results])*100 - 100:.2f}%")
    
    print("\nComparison with our system:")
    print("   Buy & Hold: Better than -60% loss")
    print("   Our System: Significantly underperformed")

def analyze_feature_statistics():
    """Analyze the features we're using"""
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    conn = sqlite3.connect('data/crypto_5years.db')
    
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM market_data_1m
    WHERE symbol = 'BTC/USDT'
    AND timestamp >= '2023-06-01'
    AND timestamp <= '2023-06-02'
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
    conn.close()
    
    df = df.set_index('timestamp')
    
    # Calculate our features
    df['returns'] = df['close'].pct_change()
    df['rsi'] = calculate_rsi(df['close'])
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['volatility'] = df['returns'].rolling(20).std()
    
    print("\n1. Feature Correlations with Future Returns:")
    df['future_return'] = df['close'].shift(-60) / df['close'] - 1  # 1 hour future
    
    correlations = {}
    for col in ['returns', 'rsi', 'volatility']:
        if col in df.columns:
            corr = df[col].corr(df['future_return'])
            correlations[col] = corr
            print(f"   {col}: {corr:.4f}")
    
    print("\n2. Feature Statistics:")
    for col in ['returns', 'rsi', 'volatility']:
        if col in df.columns:
            print(f"\n   {col}:")
            print(f"      Mean: {df[col].mean():.6f}")
            print(f"      Std: {df[col].std():.6f}")
            print(f"      Min: {df[col].min():.6f}")
            print(f"      Max: {df[col].max():.6f}")

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def analyze_profitable_periods():
    """Find when the market was actually profitable to trade"""
    print("\n" + "="*60)
    print("PROFITABLE PERIODS ANALYSIS")
    print("="*60)
    
    conn = sqlite3.connect('data/crypto_5years.db')
    
    query = """
    SELECT timestamp, close
    FROM market_data_1m
    WHERE symbol = 'BTC/USDT'
    AND timestamp >= '2023-01-01'
    AND timestamp <= '2023-12-31'
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
    conn.close()
    
    df = df.set_index('timestamp')
    
    # Calculate rolling returns for different periods
    df['return_1h'] = df['close'].pct_change(60)  # 1 hour
    df['return_4h'] = df['close'].pct_change(240)  # 4 hours
    df['return_1d'] = df['close'].pct_change(1440)  # 1 day
    
    print("\n1. Profitable Trade Opportunities (2% gain within timeframe):")
    
    for period, col in [('1 hour', 'return_1h'), ('4 hours', 'return_4h'), ('1 day', 'return_1d')]:
        profitable = df[df[col] > 0.02]  # 2% gain
        total = len(df[col].dropna())
        pct = len(profitable) / total * 100 if total > 0 else 0
        print(f"   {period}: {len(profitable)}/{total} ({pct:.2f}%)")
    
    print("\n2. Risk/Reward Analysis (2% gain vs 2% loss):")
    
    for period, col in [('1 hour', 'return_1h'), ('4 hours', 'return_4h'), ('1 day', 'return_1d')]:
        wins = df[df[col] > 0.02]
        losses = df[df[col] < -0.02]
        total = len(df[col].dropna())
        
        if total > 0:
            win_rate = len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) > 0 else 0
            print(f"   {period}: {win_rate:.2f}% win rate")

def main():
    """Run all investigations"""
    print("COMPREHENSIVE FAILURE INVESTIGATION")
    print("="*80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Check data quality
    df = analyze_data_quality()
    
    # 2. Analyze predictions
    analyze_prediction_distribution()
    
    # 3. Test buy and hold baseline
    test_buy_and_hold()
    
    # 4. Analyze features
    analyze_feature_statistics()
    
    # 5. Find profitable periods
    analyze_profitable_periods()
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print("""
    1. DATA QUALITY: Check for gaps and anomalies
    2. BASELINE: Buy & hold performance vs our system
    3. FEATURES: Correlation with future returns
    4. TIMEFRAME: Minute-level may be too noisy
    5. OPPORTUNITIES: Limited profitable windows
    """)

if __name__ == "__main__":
    main()