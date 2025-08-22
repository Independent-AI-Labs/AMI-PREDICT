'use client'

import { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'

interface MarketData {
  symbol: string
  price: number
  change24h: number
  volume: number
}

export default function MarketOverview() {
  const [markets, setMarkets] = useState<MarketData[]>([
    { symbol: 'BTC/USDT', price: 43250.50, change24h: 2.34, volume: 1234567890 },
    { symbol: 'ETH/USDT', price: 2280.30, change24h: -1.12, volume: 987654321 },
    { symbol: 'BNB/USDT', price: 315.75, change24h: 3.45, volume: 456789012 },
    { symbol: 'ADA/USDT', price: 0.58, change24h: -0.89, volume: 234567890 },
    { symbol: 'SOL/USDT', price: 98.45, change24h: 5.67, volume: 345678901 },
  ])

  useEffect(() => {
    // Simulate price updates
    const interval = setInterval(() => {
      setMarkets(prev => prev.map(market => ({
        ...market,
        price: market.price * (1 + (Math.random() - 0.5) * 0.002),
        change24h: market.change24h + (Math.random() - 0.5) * 0.1
      })))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="glass-panel p-6">
      <h2 className="text-xl font-semibold mb-4">Market Overview</h2>
      <div className="grid grid-cols-5 gap-4">
        {markets.map(market => (
          <div key={market.symbol} className="metric-card">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-400">{market.symbol}</span>
              {market.change24h > 0 ? (
                <TrendingUp className="w-4 h-4 text-crypto-green" />
              ) : (
                <TrendingDown className="w-4 h-4 text-crypto-red" />
              )}
            </div>
            <div className="text-lg font-bold">
              ${market.price.toFixed(2)}
            </div>
            <div className={`text-sm font-medium ${
              market.change24h > 0 ? 'text-crypto-green' : 'text-crypto-red'
            }`}>
              {market.change24h > 0 ? '+' : ''}{market.change24h.toFixed(2)}%
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Vol: ${(market.volume / 1000000).toFixed(1)}M
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}