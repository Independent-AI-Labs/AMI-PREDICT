import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const timeframe = searchParams.get('timeframe') || '7d'
  
  try {
    // Try to fetch from Python backend
    try {
      const response = await fetch(`http://localhost:8000/api/performance?timeframe=${timeframe}`)
      if (response.ok) {
        const data = await response.json()
        return NextResponse.json(data)
      }
    } catch (error) {
      // Backend not available
    }
    
    // Return empty data for clean slate
    const baseData = {
      totalPnL: 0,
      totalPnLPercent: 0,
      winRate: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      avgWin: 0,
      avgLoss: 0,
      sharpeRatio: 0,
      calmarRatio: 0,
      sortinoRatio: 0,
      maxDrawdown: 0,
      maxDrawdownPercent: 0,
      currentDrawdown: 0,
      expectancy: 0,
      profitFactor: 0,
      recoveryFactor: 0,
      avgHoldTime: 0,
      bestTrade: 0,
      worstTrade: 0,
      consecutiveWins: 0,
      consecutiveLosses: 0,
      dailyReturns: [],
      equity: []
    }
    
    return NextResponse.json(baseData)
    
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch performance data' },
      { status: 500 }
    )
  }
}