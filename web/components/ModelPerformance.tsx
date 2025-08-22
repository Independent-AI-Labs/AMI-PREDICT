'use client'

import { useState, useEffect } from 'react'
import { Brain, Zap, TrendingUp, Layers } from 'lucide-react'

interface ModelData {
  name: string
  accuracy: number
  latency: number
  signals: number
  icon: any
  color: string
}

export default function ModelPerformance() {
  const [models, setModels] = useState<ModelData[]>([
    { name: 'LightGBM', accuracy: 62.5, latency: 8, signals: 45, icon: Zap, color: 'text-yellow-500' },
    { name: 'CatBoost', accuracy: 59.8, latency: 12, signals: 38, icon: Brain, color: 'text-blue-500' },
    { name: 'LSTM', accuracy: 57.3, latency: 45, signals: 32, icon: Layers, color: 'text-purple-500' },
    { name: 'Ensemble', accuracy: 64.2, latency: 15, signals: 52, icon: TrendingUp, color: 'text-crypto-green' },
  ])

  useEffect(() => {
    const interval = setInterval(() => {
      setModels(prev => prev.map(model => ({
        ...model,
        accuracy: Math.max(50, Math.min(70, model.accuracy + (Math.random() - 0.5) * 2)),
        signals: model.signals + Math.floor(Math.random() * 3)
      })))
    }, 4000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="glass-panel p-6">
      <h2 className="text-xl font-semibold mb-4">Model Performance</h2>
      
      <div className="space-y-4">
        {models.map((model, index) => {
          const Icon = model.icon
          return (
            <div key={index} className="p-3 bg-white/5 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <Icon className={`w-5 h-5 ${model.color}`} />
                  <span className="font-medium">{model.name}</span>
                </div>
                <span className="text-sm text-gray-400">{model.signals} signals</span>
              </div>
              
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">Accuracy</span>
                    <span className="font-semibold">{model.accuracy.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-crypto-purple to-crypto-green h-2 rounded-full transition-all duration-500"
                      style={{ width: `${model.accuracy}%` }}
                    />
                  </div>
                </div>
                
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Latency</span>
                  <span className={`font-semibold ${model.latency < 20 ? 'text-crypto-green' : 'text-yellow-500'}`}>
                    {model.latency}ms
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
      
      <div className="mt-4 p-3 bg-crypto-green/10 border border-crypto-green/30 rounded-lg">
        <div className="text-sm text-crypto-green font-semibold">Regime Detection: TRENDING BULL</div>
        <div className="text-xs text-gray-400 mt-1">Confidence: 78.5%</div>
      </div>
    </div>
  )
}