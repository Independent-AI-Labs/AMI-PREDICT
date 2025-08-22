'use client'

import { useState, useEffect } from 'react'
import { Brain, Play, Square, Download, Upload, AlertCircle, CheckCircle, Clock, TrendingUp, AlertTriangle } from 'lucide-react'
import { MetricTooltip } from './Tooltip'

interface ModelStatus {
  name: string
  trained: boolean
  accuracy: number
  lastTrained: string
  dataPoints: number
  features: number
}

export default function ModelDashboard() {
  const [models, setModels] = useState<ModelStatus[]>([
    { name: 'LightGBM', trained: false, accuracy: 0, lastTrained: 'Never', dataPoints: 0, features: 0 },
    { name: 'CatBoost', trained: false, accuracy: 0, lastTrained: 'Never', dataPoints: 0, features: 0 },
    { name: 'LSTM', trained: false, accuracy: 0, lastTrained: 'Never', dataPoints: 0, features: 0 },
    { name: 'Random Forest', trained: false, accuracy: 0, lastTrained: 'Never', dataPoints: 0, features: 0 }
  ])
  
  const [showPreTrainingDialog, setShowPreTrainingDialog] = useState(false)
  
  const [trainingConfig, setTrainingConfig] = useState({
    startDate: new Date().toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0],
    symbols: [],
    timeframe: '1h',
    splitRatio: 80,
    validationRatio: 10,
    features: 'all'
  })
  
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [currentModel, setCurrentModel] = useState('')
  const [trainingLogs, setTrainingLogs] = useState<string[]>([])
  const [preTrainingRange, setPreTrainingRange] = useState({
    months: 3,
    startDate: '',
    endDate: new Date().toISOString().split('T')[0]
  })
  
  // Check if any models need training on mount
  useEffect(() => {
    const untrainedModels = models.filter(m => !m.trained)
    if (untrainedModels.length === models.length) {
      // All models are untrained, show pre-training dialog
      setTimeout(() => setShowPreTrainingDialog(true), 1000)
    }
  }, [])

  const handleTrain = async () => {
    setIsTraining(true)
    setTrainingProgress(0)
    setTrainingLogs([])
    
    // Simulate training process
    const modelNames = ['LightGBM', 'CatBoost', 'LSTM', 'Random Forest']
    
    for (let i = 0; i < modelNames.length; i++) {
      setCurrentModel(modelNames[i])
      setTrainingLogs(prev => [...prev, `Starting ${modelNames[i]} training...`])
      
      // Simulate progress
      for (let p = 0; p <= 100; p += 20) {
        setTrainingProgress((i * 25) + (p * 0.25))
        await new Promise(resolve => setTimeout(resolve, 200))
      }
      
      // Update model status
      setModels(prev => prev.map(m => 
        m.name === modelNames[i] 
          ? {
              ...m,
              trained: true,
              accuracy: 60 + Math.random() * 20,
              lastTrained: new Date().toLocaleString(),
              dataPoints: 15000 + Math.floor(Math.random() * 5000),
              features: 45 + Math.floor(Math.random() * 15)
            }
          : m
      ))
      
      setTrainingLogs(prev => [...prev, `✓ ${modelNames[i]} training complete`])
    }
    
    setIsTraining(false)
    setCurrentModel('')
    setTrainingProgress(100)
  }

  const calculateDataSize = () => {
    const start = new Date(trainingConfig.startDate)
    const end = new Date(trainingConfig.endDate)
    const days = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24))
    const dataPointsPerDay = trainingConfig.timeframe === '1m' ? 1440 : 
                             trainingConfig.timeframe === '5m' ? 288 :
                             trainingConfig.timeframe === '15m' ? 96 :
                             trainingConfig.timeframe === '1h' ? 24 : 1
    return days * dataPointsPerDay * trainingConfig.symbols.length
  }

  return (
    <div className="space-y-6">
      {/* Training Configuration */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Brain className="w-6 h-6 text-blue-400" />
            Model Training Dashboard
          </h2>
          <div className="flex items-center gap-2">
            <button className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2 transition-colors">
              <Download className="w-4 h-4" />
              Export Models
            </button>
            <button className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2 transition-colors">
              <Upload className="w-4 h-4" />
              Import Models
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Start Date
            </label>
            <input
              type="date"
              value={trainingConfig.startDate}
              onChange={(e) => setTrainingConfig({...trainingConfig, startDate: e.target.value})}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
              disabled={isTraining}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              End Date
            </label>
            <input
              type="date"
              value={trainingConfig.endDate}
              onChange={(e) => setTrainingConfig({...trainingConfig, endDate: e.target.value})}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
              disabled={isTraining}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Timeframe
            </label>
            <select
              value={trainingConfig.timeframe}
              onChange={(e) => setTrainingConfig({...trainingConfig, timeframe: e.target.value})}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
              disabled={isTraining}
            >
              <option value="1m">1 Minute</option>
              <option value="5m">5 Minutes</option>
              <option value="15m">15 Minutes</option>
              <option value="1h">1 Hour</option>
              <option value="4h">4 Hours</option>
              <option value="1d">1 Day</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Train/Val Split (%)
            </label>
            <input
              type="number"
              value={trainingConfig.splitRatio}
              onChange={(e) => setTrainingConfig({...trainingConfig, splitRatio: Number(e.target.value)})}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
              disabled={isTraining}
              min={50}
              max={90}
            />
          </div>
        </div>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Trading Pairs
          </label>
          <div className="flex flex-wrap gap-2">
            {['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT'].map(symbol => (
              <label key={symbol} className="flex items-center gap-2 px-3 py-2 bg-gray-700 rounded-lg">
                <input
                  type="checkbox"
                  checked={trainingConfig.symbols.includes(symbol)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setTrainingConfig({...trainingConfig, symbols: [...trainingConfig.symbols, symbol]})
                    } else {
                      setTrainingConfig({...trainingConfig, symbols: trainingConfig.symbols.filter(s => s !== symbol)})
                    }
                  }}
                  className="rounded text-blue-600"
                  disabled={isTraining}
                />
                <span className="text-sm">{symbol}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg mb-6">
          <div className="flex items-center gap-6 text-sm">
            <div>
              <span className="text-gray-400">Estimated Data Points:</span>
              <span className="ml-2 font-mono text-blue-400">{calculateDataSize().toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-400">Features:</span>
              <span className="ml-2 font-mono text-blue-400">60</span>
            </div>
            <div>
              <span className="text-gray-400">Models:</span>
              <span className="ml-2 font-mono text-blue-400">4</span>
            </div>
          </div>
          
          <button
            onClick={handleTrain}
            disabled={isTraining || trainingConfig.symbols.length === 0}
            className={`px-6 py-3 rounded-lg font-semibold flex items-center gap-2 transition-colors ${
              isTraining 
                ? 'bg-yellow-600 text-white' 
                : 'bg-green-600 hover:bg-green-700 text-white'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {isTraining ? (
              <>
                <Square className="w-5 h-5" />
                Training... {trainingProgress.toFixed(0)}%
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Start Training
              </>
            )}
          </button>
        </div>

        {/* Training Progress */}
        {isTraining && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Training Progress</span>
              <span className="text-sm text-blue-400">{currentModel}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${trainingProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Training Logs */}
        {trainingLogs.length > 0 && (
          <div className="bg-gray-900 rounded-lg p-4 max-h-40 overflow-y-auto">
            <div className="space-y-1 text-xs font-mono">
              {trainingLogs.map((log, idx) => (
                <div key={idx} className={log.includes('✓') ? 'text-green-400' : 'text-gray-400'}>
                  {log}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Model Status Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {models.map((model) => (
          <div key={model.name} className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <h3 className="text-lg font-semibold">{model.name}</h3>
                <MetricTooltip
                  metric={model.name}
                  description={
                    model.name === 'LightGBM' ? 'Gradient boosting framework using tree-based learning. Fast and efficient for tabular data.' :
                    model.name === 'CatBoost' ? 'Gradient boosting with categorical feature support. Robust to overfitting.' :
                    model.name === 'LSTM' ? 'Long Short-Term Memory neural network. Excellent for time-series patterns.' :
                    'Ensemble of decision trees. Good for feature importance analysis.'
                  }
                />
              </div>
              {model.trained ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-xs text-yellow-500">Needs Training</span>
                </div>
              )}
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Status:</span>
                <span className={model.trained ? 'text-green-400' : 'text-gray-500'}>
                  {model.trained ? 'Trained' : 'Not Trained'}
                </span>
              </div>
              
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Accuracy:</span>
                <span className="font-mono">
                  {model.trained ? `${model.accuracy.toFixed(2)}%` : 'N/A'}
                </span>
              </div>
              
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Last Trained:</span>
                <span className="text-xs">{model.lastTrained}</span>
              </div>
              
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Data Points:</span>
                <span className="font-mono">
                  {model.dataPoints > 0 ? model.dataPoints.toLocaleString() : 'N/A'}
                </span>
              </div>
              
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Features:</span>
                <span className="font-mono">{model.features || 'N/A'}</span>
              </div>
              
              {model.trained && (
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <div className="flex gap-2">
                    <button className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors">
                      Test Model
                    </button>
                    <button className="flex-1 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors">
                      View Details
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Ensemble Configuration */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Ensemble Configuration</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {models.map((model) => (
            <div key={model.name} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
              <span className="text-sm">{model.name}</span>
              <input
                type="number"
                className="w-16 bg-gray-600 text-white px-2 py-1 rounded text-sm text-center"
                placeholder="25"
                defaultValue={25}
                min={0}
                max={100}
                disabled={!model.trained}
              />
              <span className="text-xs text-gray-400">%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Pre-Training Dialog */}
      {showPreTrainingDialog && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="w-8 h-8 text-yellow-500" />
              <div>
                <h3 className="text-lg font-semibold">Models Need Training</h3>
                <p className="text-sm text-gray-400">Your ML models are not trained yet.</p>
              </div>
            </div>
            
            <div className="mb-6">
              <p className="text-sm text-gray-300 mb-4">
                Would you like to pre-train all models using historical data? This will help the bot make better predictions from the start.
              </p>
              
              <div className="space-y-3">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Training Period</label>
                  <select
                    value={preTrainingRange.months}
                    onChange={(e) => setPreTrainingRange({...preTrainingRange, months: Number(e.target.value)})}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
                  >
                    <option value={1}>Last 1 Month</option>
                    <option value={3}>Last 3 Months</option>
                    <option value={6}>Last 6 Months</option>
                    <option value={12}>Last 12 Months</option>
                  </select>
                </div>
                
                <div className="p-3 bg-gray-700 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">Estimated Training Time</div>
                  <div className="text-sm">~{preTrainingRange.months * 2} minutes</div>
                  <div className="text-xs text-gray-400 mt-2">Data Points</div>
                  <div className="text-sm">~{(preTrainingRange.months * 720 * trainingConfig.symbols.length).toLocaleString()}</div>
                </div>
              </div>
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={() => {
                  const endDate = new Date()
                  const startDate = new Date()
                  startDate.setMonth(startDate.getMonth() - preTrainingRange.months)
                  setTrainingConfig({
                    ...trainingConfig,
                    startDate: startDate.toISOString().split('T')[0],
                    endDate: endDate.toISOString().split('T')[0]
                  })
                  setShowPreTrainingDialog(false)
                  handleTrain()
                }}
                className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors"
              >
                Start Pre-Training
              </button>
              <button
                onClick={() => setShowPreTrainingDialog(false)}
                className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                Skip for Now
              </button>
            </div>
            
            <p className="text-xs text-gray-500 mt-3 text-center">
              You can always train models later from this dashboard
            </p>
          </div>
        </div>
      )}
    </div>
  )
}