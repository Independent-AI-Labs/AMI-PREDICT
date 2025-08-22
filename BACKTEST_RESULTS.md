# Backtest Results - AttentionTCN Model

## Executive Summary
Comprehensive backtesting of the AttentionTCN model on 2023 market data reveals significant issues with the current trading strategy implementation. While the model achieved 75.47% validation accuracy during training, the live trading simulation shows consistent losses across all quarters.

## Backtest Configuration
- **Model**: AttentionTCN (75.47% validation accuracy)
- **Initial Capital**: $100,000
- **Position Size**: 20% of capital
- **Commission**: 0.1%
- **Slippage**: 0.05%
- **Test Period**: Full year 2023 (Q1-Q4)

## Quarterly Performance

### Q1 2023 (Jan-Mar)
- **Total Return**: -64.47%
- **Final Equity**: $35,533
- **Max Drawdown**: -64.51%
- **Sharpe Ratio**: -1.15
- **Win Rate**: 18.73%
- **Total Trades**: 1,858

### Q2 2023 (Apr-Jun)
- **Total Return**: -62.71%
- **Final Equity**: $37,287
- **Max Drawdown**: -62.72%
- **Sharpe Ratio**: -1.30
- **Win Rate**: 16.74%
- **Total Trades**: 1,667

### Q3 2023 (Jul-Sep)
- **Total Return**: -56.81%
- **Final Equity**: $43,187
- **Max Drawdown**: -56.82%
- **Sharpe Ratio**: -1.27
- **Win Rate**: 11.70%
- **Total Trades**: 1,402

### Q4 2023 (Oct-Dec)
- **Total Return**: -59.37%
- **Final Equity**: $40,634
- **Max Drawdown**: -59.38%
- **Sharpe Ratio**: -1.03
- **Win Rate**: 21.00%
- **Total Trades**: 1,657

## Average Performance Metrics
- **Average Return**: -60.84%
- **Average Sharpe Ratio**: -1.19
- **Average Max Drawdown**: -60.86%
- **Average Win Rate**: 17.04%

## Key Issues Identified

### 1. Model-Market Mismatch
The model was trained on classification labels (up/down movement) but is being used for regression-based trading signals. This fundamental mismatch leads to poor signal quality.

### 2. Overtrading
With 1,400-1,800 trades per quarter on minute-level data, the strategy is overtrading, resulting in excessive commission costs.

### 3. Poor Win Rate
Win rates below 20% across all quarters indicate the model's predictions are worse than random.

### 4. No Risk Management
The current implementation lacks:
- Stop-loss orders
- Take-profit targets
- Position sizing based on volatility
- Maximum drawdown limits

## Root Cause Analysis

### Training vs Trading Disconnect
1. **Training**: Model trained to predict price direction (classification)
2. **Trading**: Using raw probability outputs as trading signals
3. **Result**: Signals don't align with profitable trading opportunities

### Signal Generation Issues
- Dynamic thresholding (70th/30th percentile) creates arbitrary entry/exit points
- No consideration of market regime or volatility
- Minimum holding period (10 bars) may be too short for minute data

## Recommendations

### Immediate Actions
1. **Retrain Model**: Train specifically for trading signals, not just price direction
2. **Add Risk Management**: Implement stop-loss (2%) and take-profit (3%) orders
3. **Reduce Trade Frequency**: Increase minimum holding period to 60 minutes
4. **Filter Signals**: Only trade high-confidence predictions (>80% probability)

### Strategic Improvements
1. **Feature Engineering**: Add market microstructure features
2. **Label Engineering**: Use risk-adjusted returns as labels
3. **Ensemble Methods**: Combine multiple models for robust signals
4. **Walk-Forward Analysis**: Test on rolling windows to avoid overfitting

### Alternative Approaches
1. **Regime Detection**: Trade only in favorable market conditions
2. **Mean Reversion**: Focus on oversold/overbought conditions
3. **Trend Following**: Use model to identify trend strength
4. **Options Strategy**: Use predictions for options trading instead

## Conclusion
While the AttentionTCN model shows high validation accuracy, it fails to generate profitable trading signals in live market conditions. The disconnect between training objectives and trading requirements is the primary issue. A complete redesign of the training pipeline with trading-specific objectives is necessary before deployment.

## Next Steps
1. ✅ Completed comprehensive backtesting
2. ⏳ Retrain model with proper trading labels
3. ⏳ Implement risk management framework
4. ⏳ Develop market regime detection
5. ⏳ Create ensemble trading system
6. ⏳ Perform walk-forward validation

---

*Generated: 2025-08-22*
*Framework: PyTorch 2.8.0+xpu*
*Hardware: Intel Arc A770*