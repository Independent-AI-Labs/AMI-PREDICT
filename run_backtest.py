"""
Unified backtesting script for all trained models
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.ml.backtester import Backtester
from advanced_models import AttentionTCN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path: str, device: str = 'cpu') -> torch.nn.Module:
    """Load a trained model from disk."""
    model = AttentionTCN(input_size=15, hidden_size=128, num_layers=4)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model

def prepare_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load and prepare data for backtesting."""
    import sqlite3
    
    conn = sqlite3.connect('data/crypto_5years.db')
    
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM market_data_1m
    WHERE symbol = 'BTC/USDT'
    AND timestamp >= ?
    AND timestamp <= ?
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(
        query, 
        conn, 
        params=(start_date, end_date),
        parse_dates=['timestamp']
    )
    conn.close()
    
    df = df.set_index('timestamp')
    logger.info(f"Loaded {len(df)} records from {start_date} to {end_date}")
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for the model - exactly 15 features to match the model."""
    features = pd.DataFrame(index=df.index)
    
    # Price features (4)
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features['high_low_ratio'] = df['high'] / df['low']
    features['close_open_ratio'] = df['close'] / df['open']
    
    # Volume features (2)
    volume_sma = df['volume'].rolling(20).mean()
    features['volume_ratio'] = df['volume'] / volume_sma
    features['volume_std'] = df['volume'].rolling(20).std() / volume_sma
    
    # Technical indicators (3)
    features['rsi'] = calculate_rsi(df['close'], 14)
    features['volatility_5'] = features['returns'].rolling(5).std()
    features['volatility_20'] = features['returns'].rolling(20).std()
    
    # Momentum (3)
    features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    features['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # Moving average positions (3)
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    features['close_to_sma20'] = df['close'] / sma_20
    features['close_to_sma50'] = df['close'] / sma_50
    features['sma20_to_sma50'] = sma_20 / sma_50
    
    # Ensure exactly 15 features
    assert len(features.columns) == 15, f"Expected 15 features, got {len(features.columns)}"
    
    return features

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(model: torch.nn.Module, df: pd.DataFrame, device: str = 'cpu') -> pd.Series:
    """Generate trading signals using the model."""
    
    # Feature engineering
    features_df = create_features(df)
    features_df = features_df.dropna()
    
    # Prepare sequences
    sequence_length = 50
    X = []
    indices = []
    
    for i in range(sequence_length, len(features_df)):
        X.append(features_df.iloc[i-sequence_length:i].values)
        indices.append(features_df.index[i])
    
    if not X:
        logger.error("Not enough data for sequences")
        return pd.Series(dtype=float)
    
    X = np.array(X, dtype=np.float32)
    
    # Generate predictions in batches
    batch_size = 1024
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            outputs = model(batch)
            # Model outputs single value (regression), apply sigmoid for probability
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.squeeze().cpu().numpy())
    
    # Convert predictions to signals
    predictions = np.array(predictions)
    
    # Dynamic thresholding
    buy_threshold = np.percentile(predictions, 70)
    sell_threshold = np.percentile(predictions, 30)
    
    signals = np.zeros(len(predictions))
    signals[predictions > buy_threshold] = 1
    signals[predictions < sell_threshold] = -1
    
    # Create series with proper index
    signals_series = pd.Series(signals, index=indices)
    
    # Add position management rules
    positions = []
    current_position = 0
    min_holding_period = 10  # bars
    bars_held = 0
    
    for i, (idx, signal) in enumerate(signals_series.items()):
        if current_position == 0:
            # No position - can open
            if signal == 1:
                current_position = 1
                bars_held = 0
            positions.append(current_position)
        else:
            # Have position
            bars_held += 1
            if signal == -1 and bars_held >= min_holding_period:
                # Close position
                current_position = 0
                positions.append(-1)  # Sell signal
            else:
                positions.append(0)  # Hold
    
    final_signals = pd.Series(positions, index=signals_series.index)
    
    # Log signal statistics
    buy_signals = (final_signals == 1).sum()
    sell_signals = (final_signals == -1).sum()
    logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
    
    return final_signals

def run_backtest(
    model_name: str = "AttentionTCN",
    model_path: str = "models/AttentionTCN_best.pth",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    initial_capital: float = 100000,
    position_size: float = 0.2,
    commission: float = 0.001,
    slippage: float = 0.0005
) -> Dict[str, Any]:
    """Run a complete backtest."""
    
    logger.info("="*60)
    logger.info(f"BACKTESTING {model_name}")
    logger.info("="*60)
    
    # Check for XPU availability
    if torch.xpu.is_available():
        device = 'xpu'
        logger.info(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    
    # Load model
    model = load_model(model_path, device)
    
    # Load data
    df = prepare_data(start_date, end_date)
    
    # Generate signals
    signals = generate_signals(model, df, device)
    
    # Align data with signals
    aligned_df = df.loc[signals.index]
    
    # Run backtest
    backtester = Backtester(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )
    
    results = backtester.run(
        data=aligned_df,
        signals=signals,
        position_size=position_size
    )
    
    # Generate detailed report
    print_backtest_report(results, model_name)
    
    # Save results
    save_results(results, model_name, start_date, end_date)
    
    return results

def print_backtest_report(results: Dict[str, Any], model_name: str):
    """Print a detailed backtest report."""
    
    print("\n" + "="*60)
    print(f"BACKTEST RESULTS - {model_name}")
    print("="*60)
    
    stats = results['portfolio_stats']
    
    print("\nPERFORMANCE METRICS")
    print("-"*40)
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    print("\nTRADING STATISTICS")
    print("-"*40)
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Winning Trades: {stats['winning_trades']}")
    print(f"Losing Trades: {stats['losing_trades']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Average Win: ${stats['avg_win']:.2f}")
    print(f"Average Loss: ${stats['avg_loss']:.2f}")
    
    # Calculate additional metrics
    if results['returns'] is not None and len(results['returns']) > 0:
        returns = results['returns']
        print("\nRISK METRICS")
        print("-"*40)
        print(f"Daily Volatility: {np.std(returns)*100:.2f}%")
        print(f"Annual Volatility: {np.std(returns)*np.sqrt(252)*100:.2f}%")
        
        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        print(f"Sortino Ratio: {sortino:.2f}")
        
        # Calmar ratio
        calmar = results['total_return'] / abs(results['max_drawdown']) if results['max_drawdown'] != 0 else 0
        print(f"Calmar Ratio: {calmar:.2f}")
    
    print("\n" + "="*60)

def save_results(results: Dict[str, Any], model_name: str, start_date: str, end_date: str):
    """Save backtest results to JSON."""
    
    # Prepare results for JSON serialization
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'backtest_period': f"{start_date} to {end_date}",
        'metrics': {
            'total_return': results['total_return'],
            'final_equity': results['final_equity'],
            'max_drawdown': results['max_drawdown'],
            'sharpe_ratio': results['sharpe_ratio'],
            'total_trades': results['portfolio_stats']['total_trades'],
            'win_rate': results['portfolio_stats']['win_rate'],
            'profit_factor': results['portfolio_stats']['profit_factor']
        },
        'equity_curve_samples': [float(x) for x in results['equity_curve'][::100]]  # Sample every 100th point
    }
    
    # Save to experiments folder
    output_file = Path(f'experiments/backtest_{model_name}_{datetime.now():%Y%m%d_%H%M%S}.json')
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Also save equity curve as CSV for analysis
    equity_df = pd.DataFrame({
        'equity': results['equity_curve']
    })
    csv_file = Path(f'experiments/equity_curve_{model_name}.csv')
    equity_df.to_csv(csv_file, index=False)
    logger.info(f"Equity curve saved to {csv_file}")

def main():
    """Run backtests for all models."""
    
    # Test different time periods (use available data from 2020-2025)
    test_periods = [
        ("2023-01-01", "2023-03-31", "Q1 2023"),
        ("2023-04-01", "2023-06-30", "Q2 2023"),
        ("2023-07-01", "2023-09-30", "Q3 2023"),
        ("2023-10-01", "2023-12-31", "Q4 2023"),
    ]
    
    all_results = []
    
    for start_date, end_date, period_name in test_periods:
        logger.info(f"\nBacktesting {period_name}")
        
        results = run_backtest(
            model_name="AttentionTCN",
            model_path="models/AttentionTCN_best.pth",
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            position_size=0.2,
            commission=0.001,
            slippage=0.0005
        )
        
        all_results.append({
            'period': period_name,
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['portfolio_stats']['win_rate']
        })
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST SUMMARY - ALL PERIODS")
    print("="*60)
    
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    
    # Calculate average metrics
    print("\nAVERAGE METRICS")
    print("-"*40)
    print(f"Avg Return: {summary_df['total_return'].mean():.2f}%")
    print(f"Avg Sharpe: {summary_df['sharpe_ratio'].mean():.2f}")
    print(f"Avg Max DD: {summary_df['max_drawdown'].mean():.2f}%")
    print(f"Avg Win Rate: {summary_df['win_rate'].mean():.2f}%")

if __name__ == "__main__":
    main()