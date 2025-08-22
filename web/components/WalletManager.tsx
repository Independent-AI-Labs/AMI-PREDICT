'use client'

import { useState, useEffect } from 'react'
import { Wallet, DollarSign, TrendingUp, TrendingDown, RefreshCw, Send, ArrowDownLeft, ArrowUpRight } from 'lucide-react'

interface WalletData {
  type: 'simulation' | 'paper' | 'live'
  balance: number
  initialBalance: number
  currency: string
  assets: Asset[]
  transactions: Transaction[]
  pnl: number
  pnlPercent: number
}

interface Asset {
  symbol: string
  amount: number
  value: number
  avgPrice: number
  currentPrice: number
  pnl: number
  pnlPercent: number
}

interface Transaction {
  id: string
  type: 'deposit' | 'withdraw' | 'buy' | 'sell'
  asset: string
  amount: number
  price: number
  value: number
  timestamp: string
  status: 'completed' | 'pending' | 'failed'
}

export default function WalletManager() {
  const [activeWallet, setActiveWallet] = useState<'simulation' | 'paper' | 'live'>('simulation')
  const [wallets, setWallets] = useState<Record<string, WalletData>>({
    simulation: {
      type: 'simulation',
      balance: 0,
      initialBalance: 0,
      currency: 'USDT',
      assets: [],
      transactions: [],
      pnl: 0,
      pnlPercent: 0
    },
    paper: {
      type: 'paper',
      balance: 0,
      initialBalance: 0,
      currency: 'USDT',
      assets: [],
      transactions: [],
      pnl: 0,
      pnlPercent: 0
    },
    live: {
      type: 'live',
      balance: 0,
      initialBalance: 0,
      currency: 'USDT',
      assets: [],
      transactions: [],
      pnl: 0,
      pnlPercent: 0
    }
  })

  const [showDepositModal, setShowDepositModal] = useState(false)
  const [depositAmount, setDepositAmount] = useState('')
  const [refreshing, setRefreshing] = useState(false)

  const currentWallet = wallets[activeWallet]
  const totalValue = currentWallet.balance + currentWallet.assets.reduce((sum, asset) => sum + asset.value, 0)

  const handleRefresh = async () => {
    setRefreshing(true)
    // Simulate fetching latest prices
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // Update asset values with latest prices
    setWallets(prev => ({
      ...prev,
      [activeWallet]: {
        ...prev[activeWallet],
        assets: prev[activeWallet].assets.map(asset => ({
          ...asset,
          // Simulate price changes
          currentPrice: asset.currentPrice * (1 + (Math.random() - 0.5) * 0.02),
          value: asset.amount * asset.currentPrice * (1 + (Math.random() - 0.5) * 0.02)
        }))
      }
    }))
    
    setRefreshing(false)
  }

  const handleDeposit = () => {
    const amount = parseFloat(depositAmount)
    if (amount > 0) {
      setWallets(prev => ({
        ...prev,
        [activeWallet]: {
          ...prev[activeWallet],
          balance: prev[activeWallet].balance + amount,
          transactions: [
            {
              id: Date.now().toString(),
              type: 'deposit',
              asset: 'USDT',
              amount: amount,
              price: 1,
              value: amount,
              timestamp: new Date().toISOString(),
              status: 'completed'
            },
            ...prev[activeWallet].transactions
          ]
        }
      }))
      setDepositAmount('')
      setShowDepositModal(false)
    }
  }

  const getWalletStatusColor = (type: string) => {
    switch (type) {
      case 'simulation': return 'bg-blue-900/20 border-blue-700 text-blue-400'
      case 'paper': return 'bg-yellow-900/20 border-yellow-700 text-yellow-400'
      case 'live': return 'bg-green-900/20 border-green-700 text-green-400'
      default: return 'bg-gray-700'
    }
  }

  return (
    <div className="space-y-6">
      {/* Wallet Selector */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Wallet className="w-6 h-6 text-blue-400" />
            Wallet Manager
          </h2>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh Prices
          </button>
        </div>

        {/* Wallet Tabs */}
        <div className="flex gap-2 mb-6">
          {(['simulation', 'paper', 'live'] as const).map(type => (
            <button
              key={type}
              onClick={() => setActiveWallet(type)}
              className={`flex-1 px-4 py-3 rounded-lg font-medium transition-colors ${
                activeWallet === type 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <Wallet className="w-4 h-4" />
                <span className="capitalize">{type}</span>
                {type === 'live' && (
                  <span className="text-xs bg-red-600 px-2 py-0.5 rounded">Requires API</span>
                )}
              </div>
            </button>
          ))}
        </div>

        {/* Wallet Overview */}
        <div className={`p-4 rounded-lg border ${getWalletStatusColor(activeWallet)}`}>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <div className="text-xs text-gray-400 mb-1">Available Balance</div>
              <div className="text-2xl font-bold font-mono">
                ${currentWallet.balance.toFixed(2)}
              </div>
            </div>
            
            <div>
              <div className="text-xs text-gray-400 mb-1">In Positions</div>
              <div className="text-2xl font-bold font-mono">
                ${currentWallet.assets.reduce((sum, a) => sum + a.value, 0).toFixed(2)}
              </div>
            </div>
            
            <div>
              <div className="text-xs text-gray-400 mb-1">Total Value</div>
              <div className="text-2xl font-bold font-mono">
                ${totalValue.toFixed(2)}
              </div>
            </div>
            
            <div>
              <div className="text-xs text-gray-400 mb-1">Total P&L</div>
              <div className={`text-2xl font-bold font-mono flex items-center gap-2 ${
                currentWallet.pnl >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {currentWallet.pnl >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                ${Math.abs(currentWallet.pnl).toFixed(2)}
                <span className="text-sm">({currentWallet.pnlPercent.toFixed(2)}%)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 mt-4">
          <button
            onClick={() => setShowDepositModal(true)}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center gap-2 transition-colors"
          >
            <ArrowDownLeft className="w-4 h-4" />
            Deposit
          </button>
          <button className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2 transition-colors">
            <ArrowUpRight className="w-4 h-4" />
            Withdraw
          </button>
          <button className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2 transition-colors">
            <Send className="w-4 h-4" />
            Transfer
          </button>
        </div>
      </div>

      {/* Assets Table */}
      {currentWallet.assets.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Assets</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 text-gray-400 font-medium">Asset</th>
                  <th className="text-right py-3 text-gray-400 font-medium">Amount</th>
                  <th className="text-right py-3 text-gray-400 font-medium">Avg Price</th>
                  <th className="text-right py-3 text-gray-400 font-medium">Current Price</th>
                  <th className="text-right py-3 text-gray-400 font-medium">Value</th>
                  <th className="text-right py-3 text-gray-400 font-medium">P&L</th>
                  <th className="text-center py-3 text-gray-400 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {currentWallet.assets.map((asset) => (
                  <tr key={asset.symbol} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="py-3 font-medium">{asset.symbol}</td>
                    <td className="py-3 text-right font-mono">{asset.amount.toFixed(6)}</td>
                    <td className="py-3 text-right font-mono">${asset.avgPrice.toFixed(2)}</td>
                    <td className="py-3 text-right font-mono">${asset.currentPrice.toFixed(2)}</td>
                    <td className="py-3 text-right font-mono">${asset.value.toFixed(2)}</td>
                    <td className={`py-3 text-right font-mono ${asset.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {asset.pnl >= 0 ? '+' : ''}{asset.pnl.toFixed(2)}
                      <span className="text-xs ml-1">({asset.pnlPercent.toFixed(2)}%)</span>
                    </td>
                    <td className="py-3 text-center">
                      <button className="text-blue-400 hover:text-blue-300 text-sm">Sell</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent Transactions */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Recent Transactions</h3>
        {currentWallet.transactions.length > 0 ? (
          <div className="space-y-2">
            {currentWallet.transactions.slice(0, 5).map((tx) => (
              <div key={tx.id} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                <div className="flex items-center gap-3">
                  {tx.type === 'deposit' && <ArrowDownLeft className="w-4 h-4 text-green-400" />}
                  {tx.type === 'withdraw' && <ArrowUpRight className="w-4 h-4 text-red-400" />}
                  {tx.type === 'buy' && <TrendingUp className="w-4 h-4 text-blue-400" />}
                  {tx.type === 'sell' && <TrendingDown className="w-4 h-4 text-yellow-400" />}
                  <div>
                    <div className="font-medium capitalize">{tx.type} {tx.asset}</div>
                    <div className="text-xs text-gray-400">
                      {new Date(tx.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-mono">{tx.amount} {tx.asset}</div>
                  <div className="text-sm text-gray-400">${tx.value.toFixed(2)}</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-500 py-8">No transactions yet</div>
        )}
      </div>

      {/* Deposit Modal */}
      {showDepositModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-96">
            <h3 className="text-lg font-semibold mb-4">Deposit Funds</h3>
            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-2">Amount (USDT)</label>
              <input
                type="number"
                value={depositAmount}
                onChange={(e) => setDepositAmount(e.target.value)}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded-lg"
                placeholder="Enter amount"
              />
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleDeposit}
                className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
              >
                Confirm
              </button>
              <button
                onClick={() => setShowDepositModal(false)}
                className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}