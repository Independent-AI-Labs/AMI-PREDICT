'use client'

import { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown, Clock, DollarSign, Activity } from 'lucide-react'

export default function RunHistory() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchRuns()
  }, [])

  const fetchRuns = async () => {
    try {
      const response = await fetch('/api/runs')
      if (response.ok) {
        const data = await response.json()
        setRuns(data)
      }
    } catch (error) {
      console.error('Failed to fetch runs:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading run history...</div>
      </div>
    )
  }

  // No mock data - only real runs
  const displayRuns = runs

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-bold mb-4">Run History</h2>
        
        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
              <Activity className="w-4 h-4" />
              Total Runs
            </div>
            <div className="text-2xl font-bold">{displayRuns.length}</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
              <TrendingUp className="w-4 h-4" />
              Profitable Runs
            </div>
            <div className="text-2xl font-bold text-green-400">
              {displayRuns.filter(r => r.pnl > 0).length}
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
              <DollarSign className="w-4 h-4" />
              Total P&L
            </div>
            <div className={`text-2xl font-bold ${
              displayRuns.reduce((sum, r) => sum + r.pnl, 0) >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              ${displayRuns.reduce((sum, r) => sum + r.pnl, 0).toFixed(2)}
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
              <Clock className="w-4 h-4" />
              Avg Win Rate
            </div>
            <div className="text-2xl font-bold">
              {displayRuns.length > 0 
                ? `${(displayRuns.reduce((sum, r) => sum + r.winRate, 0) / displayRuns.length).toFixed(1)}%`
                : '0.0%'
              }
            </div>
          </div>
        </div>

        {/* Runs Table */}
        {displayRuns.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <p className="text-lg mb-2">No trading runs yet</p>
            <p className="text-sm">Start a simulation to see your trading history here</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Run ID</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Start Time</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Duration</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Mode</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">Initial</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">Final</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">P&L</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">Trades</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">Win Rate</th>
                  <th className="text-center py-3 px-4 text-gray-400 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {displayRuns.map((run) => (
                <tr key={run.id} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
                  <td className="py-3 px-4 font-mono text-sm">#{run.id}</td>
                  <td className="py-3 px-4 text-sm">{run.startTime}</td>
                  <td className="py-3 px-4 text-sm">{run.duration}</td>
                  <td className="py-3 px-4">
                    <span className={`text-xs px-2 py-1 rounded-lg ${
                      run.mode === 'simulation' ? 'bg-blue-900/50 text-blue-400' :
                      run.mode === 'paper' ? 'bg-yellow-900/50 text-yellow-400' :
                      'bg-green-900/50 text-green-400'
                    }`}>
                      {run.mode}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-sm">
                    ${run.initialBalance.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-sm">
                    ${run.finalBalance.toFixed(2)}
                  </td>
                  <td className={`py-3 px-4 text-right font-mono text-sm ${
                    run.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {run.pnl >= 0 ? '+' : ''}{run.pnl.toFixed(2)}
                    <span className="text-xs ml-1 text-gray-400">
                      ({run.pnlPercent >= 0 ? '+' : ''}{run.pnlPercent.toFixed(2)}%)
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-sm">{run.trades}</td>
                  <td className="py-3 px-4 text-right font-mono text-sm">{run.winRate.toFixed(1)}%</td>
                  <td className="py-3 px-4 text-center">
                    <button className="text-blue-400 hover:text-blue-300 text-sm">
                      View Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        )}
      </div>
    </div>
  )
}