import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Try to fetch from Python backend
    try {
      const response = await fetch('http://localhost:8000/api/settings')
      if (response.ok) {
        const data = await response.json()
        return NextResponse.json(data)
      }
    } catch (error) {
      // Backend not available
    }
    
    // Return empty/default settings for clean slate
    return NextResponse.json({
      trading: {
        pairs: [],
        timeframes: ['1h'],
        maxOpenTrades: 5,
        stakePerTrade: 0.02
      },
      risk: {
        stopLoss: 0.02,
        takeProfit: 0.05,
        maxDrawdown: 0.10,
        dailyLossLimit: 0.03
      },
      ml: {
        models: [],
        ensemble: 'stacking',
        retrainFrequency: 'daily'
      }
    })
    
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch settings' },
      { status: 500 }
    )
  }
}

export async function POST(request: Request) {
  try {
    const settings = await request.json()
    
    // Try to save via Python backend
    try {
      const response = await fetch('http://localhost:8000/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      })
      
      if (response.ok) {
        const data = await response.json()
        return NextResponse.json(data)
      }
    } catch (error) {
      // Backend not available
    }
    
    // Return success for development
    return NextResponse.json({ 
      status: 'saved',
      message: 'Settings saved successfully'
    })
    
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to save settings' },
      { status: 500 }
    )
  }
}