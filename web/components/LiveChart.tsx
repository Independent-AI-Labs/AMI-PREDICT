'use client'

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'

export default function LiveChart() {
  const [data, setData] = useState<any[]>([])
  const [selectedPair, setSelectedPair] = useState('BTC/USDT')

  useEffect(() => {
    // Initialize with some data
    const initialData = Array.from({ length: 20 }, (_, i) => ({
      time: new Date(Date.now() - (20 - i) * 60000).toLocaleTimeString(),
      price: 43000 + Math.random() * 500,
      volume: Math.random() * 1000000
    }))
    setData(initialData)

    // Add new data points
    const interval = setInterval(() => {
      setData(prev => {
        const newData = [...prev.slice(-19)]
        newData.push({
          time: new Date().toLocaleTimeString(),
          price: 43000 + Math.random() * 500,
          volume: Math.random() * 1000000
        })
        return newData
      })
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="glass-panel p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Live Price Chart</h2>
        <select 
          value={selectedPair}
          onChange={(e) => setSelectedPair(e.target.value)}
          className="bg-white/10 border border-white/20 rounded px-3 py-1 text-sm"
        >
          <option value="BTC/USDT">BTC/USDT</option>
          <option value="ETH/USDT">ETH/USDT</option>
          <option value="BNB/USDT">BNB/USDT</option>
        </select>
      </div>
      
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
          <XAxis 
            dataKey="time" 
            stroke="#ffffff30"
            tick={{ fill: '#ffffff50', fontSize: 12 }}
          />
          <YAxis 
            stroke="#ffffff30"
            tick={{ fill: '#ffffff50', fontSize: 12 }}
            domain={['dataMin - 100', 'dataMax + 100']}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#0B1426', 
              border: '1px solid #ffffff20',
              borderRadius: '8px'
            }}
          />
          <Area
            type="monotone"
            dataKey="price"
            stroke="#10B981"
            fillOpacity={1}
            fill="url(#colorPrice)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
      
      <div className="grid grid-cols-3 gap-4 mt-4">
        <div className="text-center">
          <div className="text-xs text-gray-400">24h High</div>
          <div className="text-sm font-semibold text-crypto-green">$43,850.00</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400">24h Low</div>
          <div className="text-sm font-semibold text-crypto-red">$42,100.00</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400">24h Volume</div>
          <div className="text-sm font-semibold">$1.23B</div>
        </div>
      </div>
    </div>
  )
}