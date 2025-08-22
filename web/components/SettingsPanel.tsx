'use client'

import { useState } from 'react'
import { Save, RotateCcw, AlertTriangle } from 'lucide-react'

interface Settings {
  pairs: string[]
  timeframes: string[]
  maxOpenTrades: number
  stakePerTrade: number
  stopLoss: number
  takeProfit: number
  maxDrawdown: number
  dailyLossLimit: number
  trailingStop: boolean
  trailingStopDistance: number
  enableLightGBM: boolean
  enableCatBoost: boolean
  enableLSTM: boolean
  enableRandomForest: boolean
  ensembleMethod: string
  retrainFrequency: string
  regimeDetection: boolean
  regimeStates: number
  dataProvider: string
  dataUpdateInterval: number
  historicalDays: number
}

export default function SettingsPanel() {
  const [settings, setSettings] = useState<Settings>({
    pairs: [],
    timeframes: ['1h'],
    maxOpenTrades: 5,
    stakePerTrade: 2,
    stopLoss: 2,
    takeProfit: 5,
    maxDrawdown: 10,
    dailyLossLimit: 3,
    trailingStop: false,
    trailingStopDistance: 1,
    enableLightGBM: false,
    enableCatBoost: false,
    enableLSTM: false,
    enableRandomForest: false,
    ensembleMethod: 'stacking',
    retrainFrequency: 'daily',
    regimeDetection: false,
    regimeStates: 4,
    dataProvider: 'binance',
    dataUpdateInterval: 5,
    historicalDays: 30
  })

  const [saving, setSaving] = useState(false)

  const handleSave = async () => {
    setSaving(true)
    try {
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      })
      
      if (response.ok) {
        console.log('Settings saved successfully')
      }
    } catch (error) {
      console.error('Failed to save settings:', error)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    window.location.reload()
  }

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-bold mb-4">Trading Configuration</h2>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Max Open Trades
            </label>
            <input
              type="number"
              value={settings.maxOpenTrades}
              onChange={(e) => setSettings({...settings, maxOpenTrades: Number(e.target.value)})}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Stake Per Trade (%)
            </label>
            <input
              type="number"
              value={settings.stakePerTrade}
              onChange={(e) => setSettings({...settings, stakePerTrade: Number(e.target.value)})}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
            />
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-bold mb-4">Risk Management</h2>
        
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Stop Loss (%)
            </label>
            <input
              type="number"
              value={settings.stopLoss}
              onChange={(e) => setSettings({...settings, stopLoss: Number(e.target.value)})}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Take Profit (%)
            </label>
            <input
              type="number"
              value={settings.takeProfit}
              onChange={(e) => setSettings({...settings, takeProfit: Number(e.target.value)})}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Max Drawdown (%)
            </label>
            <input
              type="number"
              value={settings.maxDrawdown}
              onChange={(e) => setSettings({...settings, maxDrawdown: Number(e.target.value)})}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
            />
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-bold mb-4">Machine Learning Models</h2>
        
        <div className="space-y-2">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={settings.enableLightGBM}
              onChange={(e) => setSettings({...settings, enableLightGBM: e.target.checked})}
              className="rounded text-blue-600"
            />
            <span className="text-sm">LightGBM (Fast)</span>
          </label>
          
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={settings.enableCatBoost}
              onChange={(e) => setSettings({...settings, enableCatBoost: e.target.checked})}
              className="rounded text-blue-600"
            />
            <span className="text-sm">CatBoost (Robust)</span>
          </label>
          
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={settings.enableLSTM}
              onChange={(e) => setSettings({...settings, enableLSTM: e.target.checked})}
              className="rounded text-blue-600"
            />
            <span className="text-sm">LSTM (Sequence)</span>
          </label>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-yellow-500">
          <AlertTriangle className="w-5 h-5" />
          <span className="text-sm">Changes will take effect on next simulation run</span>
        </div>
        
        <div className="flex items-center gap-4">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
          
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors disabled:opacity-50"
          >
            <Save className="w-4 h-4" />
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}