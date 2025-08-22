'use client'

import { useState } from 'react'
import { Play, Square, Settings, RefreshCw } from 'lucide-react'

interface TradingControlsProps {
  currentRun: any
  onRunStart: (run: any) => void
  onRunStop: () => void
}

export default function TradingControls({ currentRun, onRunStart, onRunStop }: TradingControlsProps) {
  const [isStarting, setIsStarting] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [config, setConfig] = useState({
    mode: 'simulation',
    duration: 'unlimited',
    initialBalance: 0,
    maxPositions: 5,
    stakePerTrade: 2
  })

  const handleStart = async () => {
    setIsStarting(true)
    setError(null)
    try {
      const response = await fetch('/api/trading/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      
      if (response.ok) {
        const run = await response.json()
        onRunStart(run)
      } else {
        const errorData = await response.json()
        setError(errorData.message || 'Failed to start trading')
      }
    } catch (error) {
      console.error('Failed to start run:', error)
      setError('Connection failed. Please check if the backend is running.')
    } finally {
      setIsStarting(false)
    }
  }

  const handleStop = async () => {
    setIsStopping(true)
    try {
      const response = await fetch('/api/trading/stop', {
        method: 'POST'
      })
      
      if (response.ok) {
        onRunStop()
      }
    } catch (error) {
      console.error('Failed to stop run:', error)
    } finally {
      setIsStopping(false)
    }
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">Trading Controls</h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">Mode:</span>
          <select 
            value={config.mode}
            onChange={(e) => setConfig({...config, mode: e.target.value})}
            disabled={!!currentRun}
            className="bg-gray-700 text-white px-3 py-1 rounded-lg text-sm"
          >
            <option value="simulation">Simulation</option>
            <option value="paper">Paper Trading</option>
            <option value="live" disabled>Live (Requires API)</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="text-xs text-gray-400 block mb-1">Initial Balance</label>
          <input
            type="number"
            value={config.initialBalance}
            onChange={(e) => setConfig({...config, initialBalance: Number(e.target.value)})}
            disabled={!!currentRun}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Max Positions</label>
          <input
            type="number"
            value={config.maxPositions}
            onChange={(e) => setConfig({...config, maxPositions: Number(e.target.value)})}
            disabled={!!currentRun}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Stake Per Trade (%)</label>
          <input
            type="number"
            value={config.stakePerTrade}
            onChange={(e) => setConfig({...config, stakePerTrade: Number(e.target.value)})}
            disabled={!!currentRun}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Duration</label>
          <select
            value={config.duration}
            onChange={(e) => setConfig({...config, duration: e.target.value})}
            disabled={!!currentRun}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
          >
            <option value="unlimited">Unlimited</option>
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
            <option value="1w">1 Week</option>
          </select>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {!currentRun ? (
          <button
            onClick={handleStart}
            disabled={isStarting}
            className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-semibold transition-colors disabled:opacity-50"
          >
            {isStarting ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <Play className="w-5 h-5" />
            )}
            {isStarting ? 'Starting...' : 'Start Simulation'}
          </button>
        ) : (
          <button
            onClick={handleStop}
            disabled={isStopping}
            className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold transition-colors disabled:opacity-50"
          >
            {isStopping ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <Square className="w-5 h-5" />
            )}
            {isStopping ? 'Stopping...' : 'Stop Simulation'}
          </button>
        )}

        <div className="flex-1">
          {error && (
            <div className="px-4 py-2 bg-red-900/20 border border-red-700 rounded-lg text-red-400 text-sm">
              {error}
            </div>
          )}
        </div>

        {currentRun && (
          <div className="flex items-center gap-4 text-sm">
            <div>
              <span className="text-gray-400">Balance:</span>
              <span className="ml-2 font-mono">${currentRun.balance?.toFixed(2) || '0.00'}</span>
            </div>
            <div>
              <span className="text-gray-400">P&L:</span>
              <span className={`ml-2 font-mono ${currentRun.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${currentRun.pnl?.toFixed(2) || '0.00'}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Trades:</span>
              <span className="ml-2 font-mono">{currentRun.trades || 0}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}