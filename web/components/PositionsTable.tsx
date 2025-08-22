'use client'

import { useState, useEffect } from 'react'
import { ArrowUpRight, ArrowDownRight, X } from 'lucide-react'

interface Position {
  id: string
  pair: string
  side: 'LONG' | 'SHORT'
  entryPrice: number
  currentPrice: number
  size: number
  pnl: number
  pnlPercent: number
  duration: string
}

export default function PositionsTable() {
  const [positions, setPositions] = useState<Position[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    // Fetch positions from backend
    const fetchPositions = async () => {
      try {
        const response = await fetch('/api/positions')
        if (response.ok) {
          const data = await response.json()
          setPositions(data)
        }
      } catch (error) {
        console.error('Failed to fetch positions:', error)
      }
    }

    fetchPositions()
    const interval = setInterval(fetchPositions, 5000) // Update every 5 seconds
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="glass-panel p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Open Positions</h2>
        <span className="text-sm text-gray-400">{positions.length} active</span>
      </div>
      
      {positions.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <p>No open positions</p>
          <p className="text-sm mt-2">Positions will appear here when trades are executed</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-white/10">
                <th className="pb-2 text-sm text-gray-400 font-medium">Pair</th>
                <th className="pb-2 text-sm text-gray-400 font-medium">Side</th>
                <th className="pb-2 text-sm text-gray-400 font-medium text-right">Entry</th>
                <th className="pb-2 text-sm text-gray-400 font-medium text-right">Current</th>
                <th className="pb-2 text-sm text-gray-400 font-medium text-right">Size</th>
                <th className="pb-2 text-sm text-gray-400 font-medium text-right">P&L</th>
                <th className="pb-2 text-sm text-gray-400 font-medium">Duration</th>
                <th className="pb-2 text-sm text-gray-400 font-medium text-center">Actions</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position) => (
                <tr key={position.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                  <td className="py-3 font-medium">{position.pair}</td>
                  <td className="py-3">
                    <span className={`flex items-center gap-1 ${
                      position.side === 'LONG' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {position.side === 'LONG' ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                      {position.side}
                    </span>
                  </td>
                  <td className="py-3 text-right font-mono text-sm">${position.entryPrice.toFixed(2)}</td>
                  <td className="py-3 text-right font-mono text-sm">${position.currentPrice.toFixed(2)}</td>
                  <td className="py-3 text-right font-mono text-sm">{position.size}</td>
                  <td className="py-3 text-right">
                    <div>
                      <div className={`font-mono text-sm ${position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {position.pnl >= 0 ? '+' : ''}{position.pnl.toFixed(2)} USDT
                      </div>
                      <div className={`text-xs ${position.pnlPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%
                      </div>
                    </div>
                  </td>
                  <td className="py-3 text-sm text-gray-400">{position.duration}</td>
                  <td className="py-3">
                    <button className="w-full flex items-center justify-center p-1 text-red-400 hover:bg-red-400/10 rounded transition-colors">
                      <X className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}