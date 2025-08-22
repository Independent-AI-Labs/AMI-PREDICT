import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Try to fetch from Python backend
    try {
      const response = await fetch('http://localhost:8000/api/runs')
      if (response.ok) {
        const data = await response.json()
        return NextResponse.json(data)
      }
    } catch (error) {
      // Backend not available
    }
    
    // Return empty array for now
    return NextResponse.json([])
    
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch runs' },
      { status: 500 }
    )
  }
}