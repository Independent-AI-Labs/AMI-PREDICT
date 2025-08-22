'use client'

import { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown, DollarSign, Percent, Activity, Target, BarChart3, AlertCircle, Trophy, Clock } from 'lucide-react'
import { MetricTooltip } from './Tooltip'

interface Trade {
  timestamp: string
  pair: string
  type: 'buy' | 'sell'
  amount: number
  price: number
  pnl?: number
  status: 'open' | 'closed'
}

interface PerformanceData {
  totalPnL: number
  totalPnLPercent: number
  winRate: number
  totalTrades: number
  winningTrades: number
  losingTrades: number
  avgWin: number
  avgLoss: number
  sharpeRatio: number
  calmarRatio: number
  sortinoRatio: number
  maxDrawdown: number
  maxDrawdownPercent: number
  currentDrawdown: number
  expectancy: number
  profitFactor: number
  recoveryFactor: number
  avgHoldTime: number
  bestTrade: number
  worstTrade: number
  consecutiveWins: number
  consecutiveLosses: number
  dailyReturns: number[]
  equity: number[]
}

export default function PerformanceMetrics() {
  const [performance, setPerformance] = useState<PerformanceData>({
    totalPnL: 0,
    totalPnLPercent: 0,
    winRate: 0,
    totalTrades: 0,
    winningTrades: 0,
    losingTrades: 0,
    avgWin: 0,
    avgLoss: 0,
    sharpeRatio: 0,
    calmarRatio: 0,
    sortinoRatio: 0,
    maxDrawdown: 0,
    maxDrawdownPercent: 0,
    currentDrawdown: 0,
    expectancy: 0,
    profitFactor: 0,
    recoveryFactor: 0,
    avgHoldTime: 0,
    bestTrade: 0,
    worstTrade: 0,
    consecutiveWins: 0,
    consecutiveLosses: 0,
    dailyReturns: [],
    equity: []
  })
  
  const [timeframe, setTimeframe] = useState<'24h' | '7d' | '30d' | 'all'>('7d')
  const [refreshing, setRefreshing] = useState(false)

  const fetchPerformanceData = async () => {
    setRefreshing(true)
    try {
      const response = await fetch(`/api/performance?timeframe=${timeframe}`)
      if (response.ok) {
        const data = await response.json()
        setPerformance(data)
      }
    } catch (error) {
      console.error('Failed to fetch performance data:', error)
    }
    setRefreshing(false)
  }

  useEffect(() => {
    fetchPerformanceData()
    const interval = setInterval(fetchPerformanceData, 30000)
    return () => clearInterval(interval)
  }, [timeframe])

  const calculateRiskRewardRatio = () => {
    if (performance.avgLoss === 0) return 0
    return Math.abs(performance.avgWin / performance.avgLoss)
  }

  return (
    <div className="space-y-6">
      {/* Header with Timeframe Selector */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold">Performance Metrics</h2>
          <div className="flex items-center gap-2">
            <div className="flex bg-gray-700 rounded-lg p-1">
              {(['24h', '7d', '30d', 'all'] as const).map(tf => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  className={`px-3 py-1 rounded text-sm transition-colors ${
                    timeframe === tf ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  {tf === 'all' ? 'All Time' : tf}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Primary Metrics */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center">
                <span className="text-sm text-gray-400">Total P&L</span>
                <MetricTooltip 
                  metric="Profit & Loss"
                  formula="Σ(Exit Price - Entry Price) × Position Size"
                  description="Total profit or loss from all closed positions. Includes realized gains/losses only."
                />
              </div>
              <DollarSign className="w-4 h-4 text-blue-400" />
            </div>
            <div className={`text-2xl font-bold ${performance.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {performance.totalPnL >= 0 ? '+' : ''}{performance.totalPnL.toFixed(2)} USDT
            </div>
            <div className={`text-sm ${performance.totalPnLPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {performance.totalPnLPercent >= 0 ? '+' : ''}{performance.totalPnLPercent.toFixed(2)}%
            </div>
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center">
                <span className="text-sm text-gray-400">Win Rate</span>
                <MetricTooltip 
                  metric="Win Rate"
                  formula="(Winning Trades / Total Trades) × 100"
                  description="Percentage of trades that resulted in profit. Higher win rate doesn't always mean higher profitability."
                />
              </div>
              <Trophy className="w-4 h-4 text-yellow-400" />
            </div>
            <div className="text-2xl font-bold">{performance.winRate.toFixed(1)}%</div>
            <div className="text-sm text-gray-400">
              {performance.winningTrades}W / {performance.losingTrades}L
            </div>
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center">
                <span className="text-sm text-gray-400">Sharpe Ratio</span>
                <MetricTooltip 
                  metric="Sharpe Ratio"
                  formula="(Return - Risk Free Rate) / Standard Deviation"
                  description="Risk-adjusted returns. Values > 1 are good, > 2 are very good, > 3 are excellent."
                />
              </div>
              <Activity className="w-4 h-4 text-purple-400" />
            </div>
            <div className="text-2xl font-bold">{performance.sharpeRatio.toFixed(2)}</div>
            <div className="text-sm text-gray-400">Risk-adjusted returns</div>
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center">
                <span className="text-sm text-gray-400">Max Drawdown</span>
                <MetricTooltip 
                  metric="Maximum Drawdown"
                  formula="(Peak Value - Trough Value) / Peak Value × 100"
                  description="Largest peak-to-trough decline in portfolio value. Measures downside risk."
                />
              </div>
              <TrendingDown className="w-4 h-4 text-red-400" />
            </div>
            <div className="text-2xl font-bold text-red-400">
              {performance.maxDrawdownPercent.toFixed(2)}%
            </div>
            <div className="text-sm text-gray-400">
              ${Math.abs(performance.maxDrawdown).toFixed(2)}
            </div>
          </div>
        </div>

        {/* Advanced Metrics */}
        <div className="grid grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="text-center">
            <div className="text-xs text-gray-400 mb-1 flex items-center justify-center">
              Expectancy
              <MetricTooltip 
                metric="Expectancy"
                formula="(Win% × Avg Win) - (Loss% × Avg Loss)"
                description="Expected profit per trade. Positive values indicate profitable system."
              />
            </div>
            <div className="text-lg font-mono">${performance.expectancy.toFixed(2)}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-400 mb-1 flex items-center justify-center">
              Profit Factor
              <MetricTooltip 
                metric="Profit Factor"
                formula="Total Profits / Total Losses"
                description="Ratio of gross profits to gross losses. Values > 1.5 are good."
              />
            </div>
            <div className="text-lg font-mono">{performance.profitFactor.toFixed(2)}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-400 mb-1 flex items-center justify-center">
              Calmar Ratio
              <MetricTooltip 
                metric="Calmar Ratio"
                formula="Annual Return / Max Drawdown"
                description="Risk-adjusted return considering worst drawdown. Higher is better."
              />
            </div>
            <div className="text-lg font-mono">{performance.calmarRatio.toFixed(2)}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-400 mb-1 flex items-center justify-center">
              Sortino Ratio
              <MetricTooltip 
                metric="Sortino Ratio"
                formula="(Return - Target) / Downside Deviation"
                description="Like Sharpe but only penalizes downside volatility. Higher is better."
              />
            </div>
            <div className="text-lg font-mono">{performance.sortinoRatio.toFixed(2)}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-400 mb-1 flex items-center justify-center">
              Avg Hold Time
              <MetricTooltip 
                metric="Average Hold Time"
                description="Average duration positions are held before closing."
              />
            </div>
            <div className="text-lg font-mono">{performance.avgHoldTime.toFixed(1)}h</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-400 mb-1 flex items-center justify-center">
              Risk/Reward
              <MetricTooltip 
                metric="Risk/Reward Ratio"
                formula="Average Win / Average Loss"
                description="Ratio of average winning trade to average losing trade."
              />
            </div>
            <div className="text-lg font-mono">1:{calculateRiskRewardRatio().toFixed(2)}</div>
          </div>
        </div>
      </div>

      {/* Trade Statistics */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Trade Statistics</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <div className="text-sm text-gray-400 mb-1">Total Trades</div>
            <div className="text-xl font-bold">{performance.totalTrades}</div>
          </div>
          <div>
            <div className="text-sm text-gray-400 mb-1">Average Win</div>
            <div className="text-xl font-bold text-green-400">+${performance.avgWin.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-sm text-gray-400 mb-1">Average Loss</div>
            <div className="text-xl font-bold text-red-400">${performance.avgLoss.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-sm text-gray-400 mb-1">Best Trade</div>
            <div className="text-xl font-bold text-green-400">+${performance.bestTrade.toFixed(2)}</div>
          </div>
        </div>

        <div className="mt-6 pt-6 border-t border-gray-700">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Win/Loss Distribution</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                  <div 
                    className="bg-green-500 h-full"
                    style={{ width: `${performance.winRate}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400">{performance.winRate.toFixed(0)}%</span>
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Current Streak</span>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-green-400">
                  <span className="text-lg font-bold">{performance.consecutiveWins}</span>
                  <span className="text-xs ml-1">wins</span>
                </div>
                <div className="text-red-400">
                  <span className="text-lg font-bold">{performance.consecutiveLosses}</span>
                  <span className="text-xs ml-1">losses</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Risk Analysis</h3>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-6">
          <div>
            <div className="text-sm text-gray-400 mb-2">Current Drawdown</div>
            <div className={`text-xl font-bold ${performance.currentDrawdown < 0 ? 'text-yellow-400' : 'text-gray-400'}`}>
              {performance.currentDrawdown < 0 ? `${performance.currentDrawdown.toFixed(2)} USDT` : 'None'}
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-400 mb-2">Recovery Factor</div>
            <div className="text-xl font-bold">{performance.recoveryFactor.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-sm text-gray-400 mb-2">Worst Trade</div>
            <div className="text-xl font-bold text-red-400">${performance.worstTrade.toFixed(2)}</div>
          </div>
        </div>
      </div>
    </div>
  )
}