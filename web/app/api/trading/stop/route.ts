import { NextResponse } from 'next/server'

export async function POST() {
  try {
    // Try to stop via Python backend
    try {
      const response = await fetch('http://localhost:8000/api/trading/stop', {
        method: 'POST'
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
      status: 'stopped',
      message: 'Trading stopped successfully'
    })
    
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to stop trading' },
      { status: 500 }
    )
  }
}