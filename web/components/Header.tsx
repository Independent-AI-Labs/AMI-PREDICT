'use client'

import { Activity, Zap, AlertCircle, CheckCircle, Database } from 'lucide-react'

interface HeaderProps {
  isConnected: boolean
  currentRun?: any
}

export default function Header({ isConnected, currentRun }: HeaderProps) {
  return (
    <header className="bg-gray-800 border-b border-gray-700">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <Zap className="w-8 h-8 text-blue-400" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                CryptoBot Pro
              </h1>
              <span className="text-xs text-gray-500 font-mono">v0.2.0</span>
            </div>
            
            {currentRun && (
              <div className="flex items-center space-x-2 px-3 py-1 bg-green-900/20 border border-green-700 rounded-lg">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-green-400">
                  Run #{currentRun.id} Active • {currentRun.duration || '00:00:00'}
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <Database className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Data:</span>
              <span className="text-sm text-blue-400 font-semibold">Binance Live</span>
            </div>
            
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <>
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span className="text-sm text-green-400">Connected</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-5 h-5 text-yellow-500 animate-pulse" />
                  <span className="text-sm text-yellow-500">Connecting...</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}