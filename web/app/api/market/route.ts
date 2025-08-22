import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Try to fetch from Python backend first
    const response = await fetch('http://localhost:8000/api/market')
    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
  } catch (error) {
    // Backend not available, fetch directly from Binance
  }
  
  try {
    // Fetch real-time data from Binance public API
    const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    const promises = symbols.map(symbol => 
      fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${symbol}`)
        .then(res => res.json())
    )
    
    const tickers = await Promise.all(promises)
    
    const marketData = {
      pairs: tickers.map(ticker => ({
        symbol: ticker.symbol.replace('USDT', '/USDT'),
        price: parseFloat(ticker.lastPrice),
        change24h: parseFloat(ticker.priceChangePercent),
        volume24h: parseFloat(ticker.quoteVolume),
        high24h: parseFloat(ticker.highPrice),
        low24h: parseFloat(ticker.lowPrice),
        bidPrice: parseFloat(ticker.bidPrice),
        askPrice: parseFloat(ticker.askPrice),
        trades24h: parseInt(ticker.count)
      })),
      timestamp: new Date().toISOString()
    }
    
    return NextResponse.json(marketData)
  } catch (error) {
    // Fallback to mock data if Binance is not available
    const marketData = {
    pairs: [
      {
        symbol: 'BTC/USDT',
        price: 43250.50,
        change24h: 2.34,
        volume24h: 1234567890,
        high24h: 43850.00,
        low24h: 42100.00
      },
      {
        symbol: 'ETH/USDT',
        price: 2280.30,
        change24h: -1.12,
        volume24h: 987654321,
        high24h: 2320.00,
        low24h: 2250.00
      },
      {
        symbol: 'BNB/USDT',
        price: 315.75,
        change24h: 3.45,
        volume24h: 456789012,
        high24h: 318.50,
        low24h: 305.20
      },
      {
        symbol: 'ADA/USDT',
        price: 0.58,
        change24h: -0.89,
        volume24h: 234567890,
        high24h: 0.59,
        low24h: 0.57
      },
      {
        symbol: 'SOL/USDT',
        price: 98.45,
        change24h: 5.67,
        volume24h: 345678901,
        high24h: 99.80,
        low24h: 93.20
      }
    ],
    timestamp: new Date().toISOString()
  }

  return NextResponse.json(marketData)
  }
}