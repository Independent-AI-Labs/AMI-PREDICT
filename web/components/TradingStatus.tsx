'use client'

import { useState, useEffect } from 'react'
import { Play, Pause, RefreshCw, Shield } from 'lucide-react'

export default function TradingStatus() {
  const [status, setStatus] = useState('RUNNING')
  const [tradesExecuted, setTradesExecuted] = useState(0)
  const [signalsGenerated, setSignalsGenerated] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setTradesExecuted(prev => prev + Math.floor(Math.random() * 3))
      setSignalsGenerated(prev => prev + Math.floor(Math.random() * 5))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="glass-panel p-6 h-full">
      <h2 className="text-xl font-semibold mb-4">Trading Status</h2>
      
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Status</span>
          <div className="flex items-center space-x-2">
            <div className="status-indicator bg-crypto-green"></div>
            <span className="text-crypto-green font-semibold">{status}</span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-gray-400">Mode</span>
          <span className="font-semibold">Simulation</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-gray-400">Trades</span>
          <span className="font-semibold">{tradesExecuted}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-gray-400">Signals</span>
          <span className="font-semibold">{signalsGenerated}</span>
        </div>

        <div className="pt-4 space-y-2">
          <button className="w-full px-4 py-2 bg-crypto-green/20 hover:bg-crypto-green/30 text-crypto-green rounded-lg transition-colors flex items-center justify-center space-x-2">
            <Play className="w-4 h-4" />
            <span>Resume Trading</span>
          </button>
          
          <button className="w-full px-4 py-2 bg-crypto-red/20 hover:bg-crypto-red/30 text-crypto-red rounded-lg transition-colors flex items-center justify-center space-x-2">
            <Shield className="w-4 h-4" />
            <span>Emergency Stop</span>
          </button>
        </div>
      </div>
    </div>
  )
}