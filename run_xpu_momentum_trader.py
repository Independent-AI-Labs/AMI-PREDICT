#!/usr/bin/env python
"""
XPU-accelerated momentum trader that WILL generate trades.
Simple strategy: Buy on momentum up, sell on momentum down.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Check XPU availability
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device('xpu')
    print(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
    print(f"XPU Memory: {torch.xpu.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_signals_xpu(data):
    """Calculate trading signals using XPU acceleration."""
    
    # Convert to tensors on XPU
    close_prices = torch.FloatTensor(data['close'].values).to(DEVICE)
    volumes = torch.FloatTensor(data['volume'].values).to(DEVICE)
    
    logger.info(f"Processing {len(close_prices):,} data points on {DEVICE}")
    
    # Calculate indicators on XPU
    # 1. Simple momentum (rate of change)
    momentum_5 = torch.zeros_like(close_prices)
    momentum_5[5:] = (close_prices[5:] - close_prices[:-5]) / close_prices[:-5]
    
    momentum_10 = torch.zeros_like(close_prices)
    momentum_10[10:] = (close_prices[10:] - close_prices[:-10]) / close_prices[:-10]
    
    # 2. Volume-weighted momentum
    vol_mom = momentum_5 * (volumes / torch.mean(volumes))
    
    # 3. RSI-like indicator (simplified)
    price_changes = torch.zeros_like(close_prices)
    price_changes[1:] = close_prices[1:] - close_prices[:-1]
    
    gains = torch.where(price_changes > 0, price_changes, torch.zeros_like(price_changes))
    losses = torch.where(price_changes < 0, -price_changes, torch.zeros_like(price_changes))
    
    # Calculate moving averages on XPU
    window = 14
    avg_gain = torch.zeros_like(gains)
    avg_loss = torch.zeros_like(losses)
    
    for i in range(window, len(gains)):
        avg_gain[i] = torch.mean(gains[i-window:i])
        avg_loss[i] = torch.mean(losses[i-window:i])
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4. Bollinger Band position
    window = 20
    sma = torch.zeros_like(close_prices)
    std = torch.zeros_like(close_prices)
    
    for i in range(window, len(close_prices)):
        window_data = close_prices[i-window:i]
        sma[i] = torch.mean(window_data)
        std[i] = torch.std(window_data)
    
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    bb_position = (close_prices - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # Generate trading signals (SIMPLE RULES THAT WILL TRADE)
    buy_signals = torch.zeros_like(close_prices, dtype=torch.bool)
    sell_signals = torch.zeros_like(close_prices, dtype=torch.bool)
    
    # Buy when:
    # - Momentum is positive
    # - RSI < 70 (not overbought)
    # - Price near lower Bollinger Band
    buy_condition = (momentum_5 > 0.001) & (rsi < 70) & (bb_position < 0.5)
    
    # Sell when:
    # - Momentum is negative
    # - RSI > 30 (not oversold) 
    # - Price near upper Bollinger Band
    sell_condition = (momentum_5 < -0.001) & (rsi > 30) & (bb_position > 0.5)
    
    # Ensure we alternate between buy and sell
    position = False  # False = no position, True = have position
    
    for i in range(window, len(close_prices)):
        if not position and buy_condition[i]:
            buy_signals[i] = True
            position = True
        elif position and sell_condition[i]:
            sell_signals[i] = True
            position = False
    
    # Force close position at end if still open
    if position:
        sell_signals[-1] = True
    
    # Convert back to CPU
    results = {
        'momentum_5': momentum_5.cpu().numpy(),
        'momentum_10': momentum_10.cpu().numpy(),
        'vol_momentum': vol_mom.cpu().numpy(),
        'rsi': rsi.cpu().numpy(),
        'bb_position': bb_position.cpu().numpy(),
        'buy_signals': buy_signals.cpu().numpy(),
        'sell_signals': sell_signals.cpu().numpy()
    }
    
    return results


def run_backtest(data, signals, initial_capital=10000):
    """Run backtest with the generated signals."""
    
    capital = initial_capital
    position = 0
    trades = []
    
    buy_signals = signals['buy_signals']
    sell_signals = signals['sell_signals']
    prices = data['close'].values
    
    for i in range(len(prices)):
        if buy_signals[i] and position == 0:
            # Buy
            shares = capital * 0.95 / prices[i]  # Use 95% of capital
            position = shares
            capital -= shares * prices[i]
            trades.append({
                'time': data.iloc[i]['timestamp'],
                'type': 'BUY',
                'price': prices[i],
                'shares': shares,
                'value': shares * prices[i]
            })
            
        elif sell_signals[i] and position > 0:
            # Sell
            capital += position * prices[i]
            trades.append({
                'time': data.iloc[i]['timestamp'],
                'type': 'SELL',
                'price': prices[i],
                'shares': position,
                'value': position * prices[i]
            })
            position = 0
    
    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital * 100
    
    # Calculate win rate
    wins = 0
    losses = 0
    for i in range(0, len(trades)-1, 2):
        if i+1 < len(trades):
            buy_price = trades[i]['price']
            sell_price = trades[i+1]['price']
            if sell_price > buy_price:
                wins += 1
            else:
                losses += 1
    
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    # Calculate max drawdown
    equity_curve = [initial_capital]
    current_capital = initial_capital
    current_position = 0
    
    for i, trade in enumerate(trades):
        if trade['type'] == 'BUY':
            current_position = trade['shares']
            current_capital -= trade['value']
        else:
            current_capital += trade['value']
            current_position = 0
        
        # Calculate equity (capital + position value)
        if current_position > 0 and i < len(trades) - 1:
            position_value = current_position * trade['price']
        else:
            position_value = 0
        
        equity_curve.append(current_capital + position_value)
    
    peak = initial_capital
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return {
        'total_return': total_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'max_drawdown': max_dd,
        'final_capital': capital,
        'trades': trades
    }


def main():
    logger.info("="*60)
    logger.info("XPU-ACCELERATED MOMENTUM TRADER")
    logger.info("="*60)
    
    # Load data
    data_path = Path("data/btc_30d/btc_usdt_30d.parquet")
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return
    
    data = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(data):,} records from {data_path}")
    
    # Calculate signals on XPU
    logger.info("\nCalculating trading signals on XPU...")
    signals = calculate_signals_xpu(data)
    
    # Count signals
    num_buys = np.sum(signals['buy_signals'])
    num_sells = np.sum(signals['sell_signals'])
    logger.info(f"Generated {num_buys} buy signals and {num_sells} sell signals")
    
    # Run backtest
    logger.info("\nRunning backtest...")
    results = run_backtest(data, signals)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Total Trades: {results['num_trades']}")
    logger.info(f"Return: {results['total_return']:.2f}%")
    logger.info(f"Win Rate: {results['win_rate']:.1f}% ({results['wins']}W/{results['losses']}L)")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    logger.info(f"Final Capital: ${results['final_capital']:.2f}")
    
    # Show first and last few trades
    if results['trades']:
        logger.info("\nFirst 5 trades:")
        for trade in results['trades'][:5]:
            logger.info(f"  {trade['time']}: {trade['type']} {trade['shares']:.4f} @ ${trade['price']:.2f}")
        
        if len(results['trades']) > 10:
            logger.info("\nLast 5 trades:")
            for trade in results['trades'][-5:]:
                logger.info(f"  {trade['time']}: {trade['type']} {trade['shares']:.4f} @ ${trade['price']:.2f}")
    
    # XPU memory usage
    if DEVICE.type == 'xpu':
        mem_used = torch.xpu.memory_allocated() / 1e9
        mem_total = torch.xpu.get_device_properties(0).total_memory / 1e9
        logger.info(f"\nXPU Memory: {mem_used:.2f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}%)")
    
    if results['num_trades'] > 0:
        logger.info(f"\nSUCCESS! Executed {results['num_trades']} trades with {results['total_return']:.2f}% return")
    else:
        logger.warning("\nNo trades generated")


if __name__ == "__main__":
    main()