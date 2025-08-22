'use client'

import { useState, useEffect } from 'react'
import Header from '@/components/Header'
import MarketOverview from '@/components/MarketOverview'
import TradingControls from '@/components/TradingControls'
import PerformanceMetrics from '@/components/PerformanceMetrics'
import PositionsTable from '@/components/PositionsTable'
import RunHistory from '@/components/RunHistory'
import SettingsPanel from '@/components/SettingsPanel'
import LiveChart from '@/components/LiveChart'
import ModelDashboard from '@/components/ModelDashboard'
import WalletManager from '@/components/WalletManager'
import PairAnalyzer from '@/components/PairAnalyzer'

export default function Dashboard() {
  const [isConnected, setIsConnected] = useState(false)
  const [activeTab, setActiveTab] = useState('trading')
  const [currentRun, setCurrentRun] = useState(null)
  const [marketData, setMarketData] = useState(null)
  const [performance, setPerformance] = useState(null)

  // Check connection and fetch data
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch('/api/status')
        if (response.ok) {
          setIsConnected(true)
          const data = await response.json()
          setPerformance(data.performance)
        }
      } catch (error) {
        setIsConnected(false)
      }
    }

    const fetchMarketData = async () => {
      try {
        const response = await fetch('/api/market')
        if (response.ok) {
          const data = await response.json()
          setMarketData(data)
        }
      } catch (error) {
        console.error('Failed to fetch market data:', error)
      }
    }

    checkConnection()
    fetchMarketData()
    
    // Poll for updates
    const interval = setInterval(() => {
      checkConnection()
      fetchMarketData()
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Header isConnected={isConnected} currentRun={currentRun} />
      
      <div className="flex">
        {/* Sidebar Navigation */}
        <div className="w-64 bg-gray-800 min-h-screen p-4">
          <nav className="space-y-2">
            <button
              onClick={() => setActiveTab('trading')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                activeTab === 'trading' ? 'bg-blue-600' : 'hover:bg-gray-700'
              }`}
            >
              <span className="text-xl">📊</span>
              <span>Trading Dashboard</span>
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                activeTab === 'history' ? 'bg-blue-600' : 'hover:bg-gray-700'
              }`}
            >
              <span className="text-xl">📈</span>
              <span>Run History</span>
            </button>
            <button
              onClick={() => setActiveTab('pairs')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                activeTab === 'pairs' ? 'bg-blue-600' : 'hover:bg-gray-700'
              }`}
            >
              <span className="text-xl">💱</span>
              <span>Pair Analysis</span>
            </button>
            <button
              onClick={() => setActiveTab('models')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                activeTab === 'models' ? 'bg-blue-600' : 'hover:bg-gray-700'
              }`}
            >
              <span className="text-xl">🤖</span>
              <span>ML Models</span>
            </button>
            <button
              onClick={() => setActiveTab('wallet')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                activeTab === 'wallet' ? 'bg-blue-600' : 'hover:bg-gray-700'
              }`}
            >
              <span className="text-xl">💰</span>
              <span>Wallets</span>
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`w-full text-left px-4 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                activeTab === 'settings' ? 'bg-blue-600' : 'hover:bg-gray-700'
              }`}
            >
              <span className="text-xl">⚙️</span>
              <span>Settings</span>
            </button>
          </nav>

          {/* Quick Stats */}
          <div className="mt-8 p-4 bg-gray-700 rounded-lg">
            <h3 className="text-sm font-semibold text-gray-400 mb-3">Quick Stats</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Total Runs:</span>
                <span className="font-mono">{performance?.totalRuns || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Win Rate:</span>
                <span className="font-mono text-green-400">
                  {performance?.winRate || 0}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Total P&L:</span>
                <span className={`font-mono ${(performance?.totalPnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${(performance?.totalPnl || 0).toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {activeTab === 'trading' && (
            <div className="space-y-6">
              {/* Trading Controls */}
              <TradingControls 
                currentRun={currentRun}
                onRunStart={(run) => setCurrentRun(run)}
                onRunStop={() => setCurrentRun(null)}
              />

              {/* Market Overview */}
              <MarketOverview marketData={marketData} />

              {/* Performance and Chart Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <PerformanceMetrics currentRun={currentRun} />
                <LiveChart symbol="BTC/USDT" />
              </div>

              {/* Positions Table */}
              <PositionsTable currentRun={currentRun} />
            </div>
          )}

          {activeTab === 'history' && (
            <RunHistory />
          )}

          {activeTab === 'pairs' && (
            <PairAnalyzer />
          )}

          {activeTab === 'models' && (
            <ModelDashboard />
          )}

          {activeTab === 'wallet' && (
            <WalletManager />
          )}

          {activeTab === 'settings' && (
            <SettingsPanel />
          )}
        </main>
      </div>
    </div>
  )
}