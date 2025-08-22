'use client'

import { useState, useEffect } from 'react'
import { Shield, AlertTriangle, TrendingDown, DollarSign } from 'lucide-react'

export default function RiskMonitor() {
  const [riskMetrics, setRiskMetrics] = useState({
    currentDrawdown: 3.2,
    maxDrawdown: 10,
    portfolioHeat: 45,
    correlation: 0.65,
    leverage: 1.8,
    marginUsed: 28
  })

  useEffect(() => {
    const interval = setInterval(() => {
      setRiskMetrics(prev => ({
        ...prev,
        currentDrawdown: Math.max(0, prev.currentDrawdown + (Math.random() - 0.5) * 0.5),
        portfolioHeat: Math.max(20, Math.min(80, prev.portfolioHeat + (Math.random() - 0.5) * 5)),
        marginUsed: Math.max(10, Math.min(50, prev.marginUsed + (Math.random() - 0.5) * 3))
      }))
    }, 4000)

    return () => clearInterval(interval)
  }, [])

  const getRiskLevel = (value: number, threshold: number) => {
    const ratio = value / threshold
    if (ratio < 0.5) return { color: 'text-crypto-green', bg: 'bg-crypto-green', label: 'LOW' }
    if (ratio < 0.8) return { color: 'text-yellow-500', bg: 'bg-yellow-500', label: 'MEDIUM' }
    return { color: 'text-crypto-red', bg: 'bg-crypto-red', label: 'HIGH' }
  }

  return (
    <div className="glass-panel p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Risk Monitor</h2>
        <Shield className="w-5 h-5 text-crypto-purple" />
      </div>
      
      <div className="space-y-4">
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <TrendingDown className="w-4 h-4 text-crypto-red" />
              <span className="text-sm">Current Drawdown</span>
            </div>
            <span className={`text-sm font-semibold ${getRiskLevel(riskMetrics.currentDrawdown, riskMetrics.maxDrawdown).color}`}>
              -{riskMetrics.currentDrawdown.toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2">
            <div 
              className={`${getRiskLevel(riskMetrics.currentDrawdown, riskMetrics.maxDrawdown).bg} h-2 rounded-full transition-all duration-500`}
              style={{ width: `${(riskMetrics.currentDrawdown / riskMetrics.maxDrawdown) * 100}%` }}
            />
          </div>
          <div className="text-xs text-gray-400 mt-1">Max allowed: {riskMetrics.maxDrawdown}%</div>
        </div>

        <div className="p-3 bg-white/5 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm">Portfolio Heat</span>
            <span className={`text-sm font-semibold ${getRiskLevel(riskMetrics.portfolioHeat, 100).color}`}>
              {riskMetrics.portfolioHeat.toFixed(0)}%
            </span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2">
            <div 
              className={`${getRiskLevel(riskMetrics.portfolioHeat, 100).bg} h-2 rounded-full transition-all duration-500`}
              style={{ width: `${riskMetrics.portfolioHeat}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 bg-white/5 rounded-lg">
            <div className="text-xs text-gray-400 mb-1">Correlation</div>
            <div className={`text-lg font-semibold ${getRiskLevel(riskMetrics.correlation, 1).color}`}>
              {riskMetrics.correlation.toFixed(2)}
            </div>
          </div>
          <div className="p-3 bg-white/5 rounded-lg">
            <div className="text-xs text-gray-400 mb-1">Leverage</div>
            <div className="text-lg font-semibold">
              {riskMetrics.leverage.toFixed(1)}x
            </div>
          </div>
        </div>

        <div className="p-3 bg-white/5 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <DollarSign className="w-4 h-4 text-crypto-gold" />
              <span className="text-sm">Margin Used</span>
            </div>
            <span className="text-sm font-semibold">{riskMetrics.marginUsed}%</span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2">
            <div 
              className="bg-crypto-gold h-2 rounded-full transition-all duration-500"
              style={{ width: `${riskMetrics.marginUsed}%` }}
            />
          </div>
        </div>

        <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-yellow-500" />
            <span className="text-sm text-yellow-500">Risk Level: MODERATE</span>
          </div>
          <div className="text-xs text-gray-400 mt-1">All limits within acceptable range</div>
        </div>
      </div>
    </div>
  )
}