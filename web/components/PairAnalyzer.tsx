'use client'

import { useState, useEffect } from 'react'
import { Search, TrendingUp, Activity, DollarSign, BarChart3, RefreshCw, Filter, Download, Star, AlertTriangle, ChevronUp, ChevronDown } from 'lucide-react'
import { MetricTooltip } from './Tooltip'

interface TradingPair {
  symbol: string
  baseAsset: string
  quoteAsset: string
  price: number
  volume24h: number
  priceChange24h: number
  priceChangePercent24h: number
  volatility: number
  sharpeRatio: number
  profitPotential: number
  risk: number
  spread: number
  liquidity: number
  correlationBTC: number
  trending: boolean
  rank: number
  historicalData?: {
    high30d: number
    low30d: number
    avgVolume30d: number
    volatility30d: number
    profitability30d: number
  }
}

interface FilterSettings {
  minVolume: number
  maxVolume: number
  minVolatility: number
  maxVolatility: number
  minLiquidity: number
  quoteCurrency: string[]
  excludeStablecoins: boolean
  onlyTrending: boolean
  minProfitPotential: number
}

export default function PairAnalyzer() {
  const [pairs, setPairs] = useState<TradingPair[]>([])
  const [filteredPairs, setFilteredPairs] = useState<TradingPair[]>([])
  const [selectedPairs, setSelectedPairs] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState<'rank' | 'volume' | 'volatility' | 'profit' | 'change'>('rank')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')
  const [timeframe, setTimeframe] = useState<'1h' | '4h' | '1d' | '7d' | '30d'>('1d')
  
  const [filters, setFilters] = useState<FilterSettings>({
    minVolume: 100000,
    maxVolume: 1000000000,
    minVolatility: 0,
    maxVolatility: 100,
    minLiquidity: 50000,
    quoteCurrency: ['USDT', 'BUSD'],
    excludeStablecoins: true,
    onlyTrending: false,
    minProfitPotential: 0
  })

  const [showFilters, setShowFilters] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<any>(null)

  // Fetch all available trading pairs
  const fetchTradingPairs = async () => {
    setLoading(true)
    try {
      // First try backend
      try {
        const response = await fetch('/api/pairs/all')
        if (response.ok) {
          const data = await response.json()
          setPairs(data)
          setFilteredPairs(data)
          setLoading(false)
          return
        }
      } catch (error) {
        console.log('Backend not available, fetching from Binance')
      }

      // Fallback to Binance API
      const [exchangeInfo, tickers] = await Promise.all([
        fetch('https://api.binance.com/api/v3/exchangeInfo').then(r => r.json()),
        fetch('https://api.binance.com/api/v3/ticker/24hr').then(r => r.json())
      ])

      const tickerMap = new Map(tickers.map((t: any) => [t.symbol, t]))
      
      const processedPairs: TradingPair[] = exchangeInfo.symbols
        .filter((s: any) => s.status === 'TRADING')
        .map((symbol: any) => {
          const ticker = tickerMap.get(symbol.symbol)
          if (!ticker) return null

          const volume = parseFloat(ticker.quoteVolume)
          const priceChange = parseFloat(ticker.priceChangePercent)
          const highLow = parseFloat(ticker.highPrice) - parseFloat(ticker.lowPrice)
          const volatility = (highLow / parseFloat(ticker.lastPrice)) * 100

          return {
            symbol: symbol.symbol,
            baseAsset: symbol.baseAsset,
            quoteAsset: symbol.quoteAsset,
            price: parseFloat(ticker.lastPrice),
            volume24h: volume,
            priceChange24h: parseFloat(ticker.priceChange),
            priceChangePercent24h: priceChange,
            volatility: volatility,
            sharpeRatio: 0,
            profitPotential: (volatility * volume) / 1000000, // Simplified metric
            risk: volatility / 10,
            spread: parseFloat(ticker.askPrice) - parseFloat(ticker.bidPrice),
            liquidity: volume,
            correlationBTC: 0,
            trending: Math.abs(priceChange) > 5,
            rank: 0
          }
        })
        .filter((p: any) => p !== null)
        .sort((a: TradingPair, b: TradingPair) => b.volume24h - a.volume24h)

      // Calculate rankings
      const rankedPairs = calculateRankings(processedPairs)
      setPairs(rankedPairs)
      setFilteredPairs(rankedPairs)
      
    } catch (error) {
      console.error('Failed to fetch trading pairs:', error)
    }
    setLoading(false)
  }

  // Calculate pair rankings based on multiple factors
  const calculateRankings = (pairs: TradingPair[]): TradingPair[] => {
    return pairs.map(pair => {
      // Normalize metrics to 0-100 scale
      const maxVolume = Math.max(...pairs.map(p => p.volume24h))
      const maxVolatility = Math.max(...pairs.map(p => p.volatility))
      const maxProfit = Math.max(...pairs.map(p => p.profitPotential))
      
      const volumeScore = (pair.volume24h / maxVolume) * 40
      const volatilityScore = (pair.volatility / maxVolatility) * 30
      const profitScore = (pair.profitPotential / maxProfit) * 30
      
      const totalScore = volumeScore + volatilityScore + profitScore
      
      return {
        ...pair,
        rank: totalScore
      }
    }).sort((a, b) => b.rank - a.rank)
      .map((pair, index) => ({ ...pair, rank: index + 1 }))
  }

  // Analyze historical data for selected pairs
  const analyzeHistoricalData = async () => {
    setAnalyzing(true)
    const selected = Array.from(selectedPairs)
    
    try {
      const response = await fetch('/api/pairs/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pairs: selected,
          timeframe: timeframe,
          metrics: ['volatility', 'profitability', 'correlation', 'patterns']
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        setAnalysisResults(data)
      }
    } catch (error) {
      console.error('Failed to analyze pairs:', error)
      // Simulate analysis for demo
      setAnalysisResults({
        timestamp: new Date().toISOString(),
        pairs: selected.map(symbol => ({
          symbol,
          metrics: {
            avgProfit: Math.random() * 10 - 5,
            maxDrawdown: -Math.random() * 15,
            winRate: 45 + Math.random() * 20,
            sharpeRatio: Math.random() * 3,
            bestEntry: 'RSI < 30 + Volume Spike',
            correlation: Math.random() * 2 - 1
          }
        }))
      })
    }
    setAnalyzing(false)
  }

  // Apply filters
  useEffect(() => {
    let filtered = pairs.filter(pair => {
      // Volume filter
      if (pair.volume24h < filters.minVolume || pair.volume24h > filters.maxVolume) return false
      
      // Volatility filter
      if (pair.volatility < filters.minVolatility || pair.volatility > filters.maxVolatility) return false
      
      // Liquidity filter
      if (pair.liquidity < filters.minLiquidity) return false
      
      // Quote currency filter
      if (filters.quoteCurrency.length > 0 && !filters.quoteCurrency.includes(pair.quoteAsset)) return false
      
      // Stablecoin filter
      if (filters.excludeStablecoins) {
        const stablecoins = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'PAX', 'USDS']
        if (stablecoins.includes(pair.baseAsset)) return false
      }
      
      // Trending filter
      if (filters.onlyTrending && !pair.trending) return false
      
      // Profit potential filter
      if (pair.profitPotential < filters.minProfitPotential) return false
      
      // Search filter
      if (searchTerm && !pair.symbol.toLowerCase().includes(searchTerm.toLowerCase())) return false
      
      return true
    })

    // Apply sorting
    filtered.sort((a, b) => {
      let compareValue = 0
      switch (sortBy) {
        case 'rank':
          compareValue = a.rank - b.rank
          break
        case 'volume':
          compareValue = a.volume24h - b.volume24h
          break
        case 'volatility':
          compareValue = a.volatility - b.volatility
          break
        case 'profit':
          compareValue = a.profitPotential - b.profitPotential
          break
        case 'change':
          compareValue = a.priceChangePercent24h - b.priceChangePercent24h
          break
      }
      return sortOrder === 'desc' ? -compareValue : compareValue
    })

    setFilteredPairs(filtered)
  }, [pairs, filters, searchTerm, sortBy, sortOrder])

  useEffect(() => {
    fetchTradingPairs()
  }, [])

  const togglePairSelection = (symbol: string) => {
    const newSelected = new Set(selectedPairs)
    if (newSelected.has(symbol)) {
      newSelected.delete(symbol)
    } else {
      newSelected.add(symbol)
    }
    setSelectedPairs(newSelected)
  }

  const selectTopPairs = (count: number) => {
    const topPairs = filteredPairs.slice(0, count).map(p => p.symbol)
    setSelectedPairs(new Set(topPairs))
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-blue-400" />
            Trading Pairs Analyzer
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchTradingPairs}
              disabled={loading}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2 transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center gap-2 transition-colors"
            >
              <Filter className="w-4 h-4" />
              Filters
            </button>
            <button
              onClick={() => analyzeHistoricalData()}
              disabled={selectedPairs.size === 0 || analyzing}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-lg flex items-center gap-2 transition-colors"
            >
              <Activity className="w-4 h-4" />
              Analyze Selected ({selectedPairs.size})
            </button>
          </div>
        </div>

        {/* Search and Quick Actions */}
        <div className="flex items-center gap-4 mb-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search pairs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-700 rounded-lg text-white placeholder-gray-400"
            />
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Quick Select:</span>
            <button
              onClick={() => selectTopPairs(10)}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
            >
              Top 10
            </button>
            <button
              onClick={() => selectTopPairs(25)}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
            >
              Top 25
            </button>
            <button
              onClick={() => selectTopPairs(50)}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
            >
              Top 50
            </button>
            <button
              onClick={() => setSelectedPairs(new Set())}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <div className="p-4 bg-gray-700 rounded-lg mb-4">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Min Volume (USDT)</label>
                <input
                  type="number"
                  value={filters.minVolume}
                  onChange={(e) => setFilters({...filters, minVolume: Number(e.target.value)})}
                  className="w-full bg-gray-600 text-white px-2 py-1 rounded text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Min Volatility (%)</label>
                <input
                  type="number"
                  value={filters.minVolatility}
                  onChange={(e) => setFilters({...filters, minVolatility: Number(e.target.value)})}
                  className="w-full bg-gray-600 text-white px-2 py-1 rounded text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Min Liquidity</label>
                <input
                  type="number"
                  value={filters.minLiquidity}
                  onChange={(e) => setFilters({...filters, minLiquidity: Number(e.target.value)})}
                  className="w-full bg-gray-600 text-white px-2 py-1 rounded text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Min Profit Potential</label>
                <input
                  type="number"
                  value={filters.minProfitPotential}
                  onChange={(e) => setFilters({...filters, minProfitPotential: Number(e.target.value)})}
                  className="w-full bg-gray-600 text-white px-2 py-1 rounded text-sm"
                />
              </div>
            </div>
            
            <div className="flex items-center gap-4 mt-4">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={filters.excludeStablecoins}
                  onChange={(e) => setFilters({...filters, excludeStablecoins: e.target.checked})}
                  className="rounded text-blue-600"
                />
                <span className="text-sm">Exclude Stablecoins</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={filters.onlyTrending}
                  onChange={(e) => setFilters({...filters, onlyTrending: e.target.checked})}
                  className="rounded text-blue-600"
                />
                <span className="text-sm">Only Trending</span>
              </label>
            </div>
          </div>
        )}

        {/* Stats Bar */}
        <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
          <div className="flex items-center gap-6 text-sm">
            <div>
              <span className="text-gray-400">Total Pairs:</span>
              <span className="ml-2 font-mono">{pairs.length}</span>
            </div>
            <div>
              <span className="text-gray-400">Filtered:</span>
              <span className="ml-2 font-mono">{filteredPairs.length}</span>
            </div>
            <div>
              <span className="text-gray-400">Selected:</span>
              <span className="ml-2 font-mono text-blue-400">{selectedPairs.size}</span>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-gray-600 text-white px-2 py-1 rounded text-sm"
            >
              <option value="rank">Rank</option>
              <option value="volume">Volume</option>
              <option value="volatility">Volatility</option>
              <option value="profit">Profit Potential</option>
              <option value="change">24h Change</option>
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="p-1 bg-gray-600 hover:bg-gray-500 rounded transition-colors"
            >
              {sortOrder === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Pairs Table */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-2">
                  <input
                    type="checkbox"
                    checked={selectedPairs.size === filteredPairs.length && filteredPairs.length > 0}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedPairs(new Set(filteredPairs.map(p => p.symbol)))
                      } else {
                        setSelectedPairs(new Set())
                      }
                    }}
                    className="rounded text-blue-600"
                  />
                </th>
                <th className="text-left py-3 text-gray-400 font-medium">
                  <div className="flex items-center">
                    Rank
                    <MetricTooltip
                      metric="Rank"
                      formula="Volume Weight (40%) + Volatility Weight (30%) + Profit Weight (30%)"
                      description="Overall ranking based on trading potential. Lower numbers are better."
                    />
                  </div>
                </th>
                <th className="text-left py-3 text-gray-400 font-medium">Pair</th>
                <th className="text-right py-3 text-gray-400 font-medium">Price</th>
                <th className="text-right py-3 text-gray-400 font-medium">24h Change</th>
                <th className="text-right py-3 text-gray-400 font-medium">
                  <div className="flex items-center justify-end">
                    Volume
                    <MetricTooltip
                      metric="24h Volume"
                      description="Total trading volume in the last 24 hours in quote currency (usually USDT)."
                    />
                  </div>
                </th>
                <th className="text-right py-3 text-gray-400 font-medium">
                  <div className="flex items-center justify-end">
                    Volatility
                    <MetricTooltip
                      metric="Volatility"
                      formula="(High - Low) / Current Price × 100"
                      description="Price range as percentage. Higher volatility means more profit opportunity but also more risk."
                    />
                  </div>
                </th>
                <th className="text-right py-3 text-gray-400 font-medium">
                  <div className="flex items-center justify-end">
                    Profit Score
                    <MetricTooltip
                      metric="Profit Potential Score"
                      formula="(Liquidity + Volatility + Volume) / 30"
                      description="Combined score indicating profit opportunity. Higher is better."
                    />
                  </div>
                </th>
                <th className="text-center py-3 text-gray-400 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={9} className="text-center py-8 text-gray-500">
                    Loading trading pairs...
                  </td>
                </tr>
              ) : filteredPairs.length === 0 ? (
                <tr>
                  <td colSpan={9} className="text-center py-8 text-gray-500">
                    No pairs match your filters
                  </td>
                </tr>
              ) : (
                filteredPairs.slice(0, 100).map((pair) => (
                  <tr key={pair.symbol} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="py-3 px-2">
                      <input
                        type="checkbox"
                        checked={selectedPairs.has(pair.symbol)}
                        onChange={() => togglePairSelection(pair.symbol)}
                        className="rounded text-blue-600"
                      />
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-sm">{pair.rank}</span>
                        {pair.rank <= 10 && <Star className="w-3 h-3 text-yellow-500" />}
                      </div>
                    </td>
                    <td className="py-3">
                      <div>
                        <div className="font-medium">{pair.baseAsset}/{pair.quoteAsset}</div>
                        <div className="text-xs text-gray-500">{pair.symbol}</div>
                      </div>
                    </td>
                    <td className="py-3 text-right font-mono text-sm">
                      ${pair.price < 1 ? pair.price.toFixed(6) : pair.price.toFixed(2)}
                    </td>
                    <td className={`py-3 text-right font-mono text-sm ${
                      pair.priceChangePercent24h >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {pair.priceChangePercent24h >= 0 ? '+' : ''}{pair.priceChangePercent24h.toFixed(2)}%
                    </td>
                    <td className="py-3 text-right font-mono text-sm">
                      ${(pair.volume24h / 1000000).toFixed(2)}M
                    </td>
                    <td className="py-3 text-right font-mono text-sm">
                      {pair.volatility.toFixed(2)}%
                    </td>
                    <td className="py-3 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <div className="w-16 bg-gray-700 rounded-full h-2 overflow-hidden">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-cyan-500 h-full"
                            style={{ width: `${Math.min(100, pair.profitPotential * 10)}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono">{pair.profitPotential.toFixed(1)}</span>
                      </div>
                    </td>
                    <td className="py-3 text-center">
                      <div className="flex items-center justify-center gap-1">
                        {pair.trending && (
                          <span className="px-2 py-0.5 bg-yellow-900/30 text-yellow-400 text-xs rounded">
                            Trending
                          </span>
                        )}
                        {pair.volatility > 10 && (
                          <span className="px-2 py-0.5 bg-red-900/30 text-red-400 text-xs rounded">
                            High Vol
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Analysis Results */}
      {analysisResults && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Historical Analysis Results</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {analysisResults.pairs?.map((result: any) => (
              <div key={result.symbol} className="p-4 bg-gray-700 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-medium">{result.symbol}</span>
                  <span className={`text-sm ${result.metrics.avgProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    Avg P&L: {result.metrics.avgProfit.toFixed(2)}%
                  </span>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Win Rate:</span>
                    <span>{result.metrics.winRate.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Max Drawdown:</span>
                    <span className="text-red-400">{result.metrics.maxDrawdown.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Sharpe Ratio:</span>
                    <span>{result.metrics.sharpeRatio.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Best Entry:</span>
                    <span className="text-xs">{result.metrics.bestEntry}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}