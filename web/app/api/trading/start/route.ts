import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const config = await request.json()
    
    // Try to start via Python backend
    try {
      const response = await fetch('http://localhost:8000/api/trading/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      
      if (response.ok) {
        const data = await response.json()
        return NextResponse.json(data)
      }
    } catch (error) {
      // Backend not available - return error
      return NextResponse.json(
        { 
          error: 'Backend service not available',
          message: 'Please ensure the Python backend is running on port 8000'
        },
        { status: 503 }
      )
    }
    
    // No mock data - require real backend
    return NextResponse.json(
      { 
        error: 'Backend required',
        message: 'Trading requires the Python backend service to be running'
      },
      { status: 503 }
    )
    
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to start trading' },
      { status: 500 }
    )
  }
}