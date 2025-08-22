'use client'

import { useState, useEffect } from 'react'
import { Cpu, Database, Wifi, Clock, Server, Activity } from 'lucide-react'

interface SystemMetric {
  name: string
  value: number | string
  unit: string
  icon: any
  status: 'good' | 'warning' | 'critical'
}

export default function SystemStatus() {
  const [metrics, setMetrics] = useState<SystemMetric[]>([
    { name: 'CPU Usage', value: 45, unit: '%', icon: Cpu, status: 'good' },
    { name: 'Memory', value: 2.8, unit: 'GB', icon: Server, status: 'good' },
    { name: 'Latency', value: 12, unit: 'ms', icon: Clock, status: 'good' },
    { name: 'Data Rate', value: 1250, unit: 'msg/s', icon: Activity, status: 'good' },
    { name: 'DB Size', value: 1.2, unit: 'GB', icon: Database, status: 'good' },
    { name: 'Network', value: 'Stable', unit: '', icon: Wifi, status: 'good' },
  ])

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => prev.map(metric => {
        if (typeof metric.value === 'number') {
          let newValue = metric.value
          if (metric.name === 'CPU Usage') {
            newValue = Math.max(20, Math.min(80, metric.value + (Math.random() - 0.5) * 10))
          } else if (metric.name === 'Latency') {
            newValue = Math.max(5, Math.min(50, metric.value + (Math.random() - 0.5) * 5))
          } else if (metric.name === 'Data Rate') {
            newValue = Math.max(800, Math.min(2000, metric.value + (Math.random() - 0.5) * 100))
          }
          
          let status: 'good' | 'warning' | 'critical' = 'good'
          if (metric.name === 'CPU Usage' && newValue > 70) status = 'warning'
          if (metric.name === 'CPU Usage' && newValue > 85) status = 'critical'
          if (metric.name === 'Latency' && newValue > 30) status = 'warning'
          if (metric.name === 'Latency' && newValue > 50) status = 'critical'
          
          return { ...metric, value: newValue, status }
        }
        return metric
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-crypto-green bg-crypto-green/20'
      case 'warning': return 'text-yellow-500 bg-yellow-500/20'
      case 'critical': return 'text-crypto-red bg-crypto-red/20'
      default: return 'text-gray-400 bg-gray-400/20'
    }
  }

  return (
    <div className="glass-panel p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">System Status</h2>
        <div className="flex items-center space-x-2">
          <div className="status-indicator bg-crypto-green"></div>
          <span className="text-sm text-crypto-green">All Systems Operational</span>
        </div>
      </div>
      
      <div className="grid grid-cols-6 gap-4">
        {metrics.map((metric, index) => {
          const Icon = metric.icon
          return (
            <div key={index} className="metric-card">
              <div className="flex items-center justify-between mb-2">
                <Icon className={`w-5 h-5 ${getStatusColor(metric.status).split(' ')[0]}`} />
                <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getStatusColor(metric.status)}`}>
                  {metric.status.toUpperCase()}
                </span>
              </div>
              <div className="text-sm text-gray-400 mb-1">{metric.name}</div>
              <div className="text-lg font-bold">
                {typeof metric.value === 'number' ? metric.value.toFixed(metric.unit === '%' ? 0 : 1) : metric.value}
                {metric.unit && <span className="text-sm text-gray-400 ml-1">{metric.unit}</span>}
              </div>
            </div>
          )
        })}
      </div>
      
      <div className="mt-4 grid grid-cols-3 gap-4">
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-xs text-gray-400 mb-1">Uptime</div>
          <div className="text-lg font-semibold">99.98%</div>
          <div className="text-xs text-gray-500">Last 24 hours</div>
        </div>
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-xs text-gray-400 mb-1">API Calls</div>
          <div className="text-lg font-semibold">142.3K</div>
          <div className="text-xs text-gray-500">Today</div>
        </div>
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-xs text-gray-400 mb-1">Error Rate</div>
          <div className="text-lg font-semibold text-crypto-green">0.02%</div>
          <div className="text-xs text-gray-500">Last hour</div>
        </div>
      </div>
    </div>
  )
}