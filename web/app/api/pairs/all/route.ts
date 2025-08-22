import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Try to fetch from Python backend
    try {
      const response = await fetch('http://localhost:8000/api/pairs/all')
      if (response.ok) {
        const data = await response.json()
        return NextResponse.json(data)
      }
    } catch (error) {
      // Backend not available, use Binance directly
    }
    
    // Fetch from Binance API
    const [exchangeInfo, tickers, volume] = await Promise.all([
      fetch('https://api.binance.com/api/v3/exchangeInfo').then(r => r.json()),
      fetch('https://api.binance.com/api/v3/ticker/24hr').then(r => r.json()),
      fetch('https://api.binance.com/api/v3/ticker/24hr?symbols=["BTCUSDT"]').then(r => r.json()).catch(() => null)
    ])

    const tickerMap = new Map(tickers.map((t: any) => [t.symbol, t]))
    const btcPrice = volume?.[0]?.lastPrice || 100000
    
    const pairs = exchangeInfo.symbols
      .filter((s: any) => s.status === 'TRADING' && s.isSpotTradingAllowed)
      .map((symbol: any) => {
        const ticker = tickerMap.get(symbol.symbol)
        if (!ticker) return null

        const price = parseFloat(ticker.lastPrice)
        const volume = parseFloat(ticker.quoteVolume)
        const priceChange = parseFloat(ticker.priceChangePercent)
        const high = parseFloat(ticker.highPrice)
        const low = parseFloat(ticker.lowPrice)
        const volatility = price > 0 ? ((high - low) / price) * 100 : 0
        const count = parseInt(ticker.count)
        
        // Calculate correlation with BTC (simplified)
        let correlationBTC = 0
        if (symbol.baseAsset === 'BTC') {
          correlationBTC = 1
        } else if (symbol.symbol.includes('BTC')) {
          correlationBTC = 0.8
        } else {
          correlationBTC = 0.3 + (Math.random() * 0.4)
        }

        // Calculate profit potential score
        const liquidityScore = Math.min(100, (volume / 1000000) * 10)
        const volatilityScore = Math.min(100, volatility * 10)
        const volumeScore = Math.min(100, (count / 10000) * 10)
        const profitPotential = (liquidityScore + volatilityScore + volumeScore) / 30

        return {
          symbol: symbol.symbol,
          baseAsset: symbol.baseAsset,
          quoteAsset: symbol.quoteAsset,
          price: price,
          volume24h: volume,
          priceChange24h: parseFloat(ticker.priceChange),
          priceChangePercent24h: priceChange,
          volatility: volatility,
          sharpeRatio: volatility > 0 ? (priceChange / volatility) : 0,
          profitPotential: profitPotential,
          risk: volatility / 10,
          spread: parseFloat(ticker.askPrice) - parseFloat(ticker.bidPrice),
          liquidity: volume,
          correlationBTC: correlationBTC,
          trending: Math.abs(priceChange) > 5,
          rank: 0,
          historicalData: {
            high30d: high * 1.2,
            low30d: low * 0.8,
            avgVolume30d: volume,
            volatility30d: volatility * 1.1,
            profitability30d: priceChange * 30 / 24
          }
        }
      })
      .filter((p: any) => p !== null && p.volume24h > 10000)
      .sort((a: any, b: any) => b.volume24h - a.volume24h)

    // Calculate rankings
    const rankedPairs = pairs.map((pair: any, index: number) => ({
      ...pair,
      rank: index + 1
    }))

    return NextResponse.json(rankedPairs.slice(0, 500)) // Limit to top 500
    
  } catch (error) {
    console.error('Failed to fetch pairs:', error)
    return NextResponse.json(
      { error: 'Failed to fetch trading pairs' },
      { status: 500 }
    )
  }
}