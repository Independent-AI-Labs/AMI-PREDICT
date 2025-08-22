'use client'

import { useState } from 'react'
import { HelpCircle, Info } from 'lucide-react'

interface TooltipProps {
  content: string
  children?: React.ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
  delay?: number
  icon?: 'help' | 'info'
}

export default function Tooltip({ 
  content, 
  children, 
  position = 'top',
  delay = 200,
  icon = 'help'
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null)

  const handleMouseEnter = () => {
    const id = setTimeout(() => setIsVisible(true), delay)
    setTimeoutId(id)
  }

  const handleMouseLeave = () => {
    if (timeoutId) {
      clearTimeout(timeoutId)
      setTimeoutId(null)
    }
    setIsVisible(false)
  }

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2'
  }

  const Icon = icon === 'help' ? HelpCircle : Info

  return (
    <div className="relative inline-flex items-center">
      {children}
      <div
        className="cursor-help ml-1"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <Icon className="w-3 h-3 text-gray-500 hover:text-gray-400 transition-colors" />
      </div>
      {isVisible && (
        <div className={`absolute z-50 ${positionClasses[position]} pointer-events-none`}>
          <div className="bg-gray-900 text-gray-200 text-xs rounded-lg px-3 py-2 shadow-lg border border-gray-700 max-w-xs">
            <div className="whitespace-normal">{content}</div>
            <div className={`absolute w-2 h-2 bg-gray-900 border-gray-700 transform rotate-45 ${
              position === 'top' ? 'bottom-[-5px] left-1/2 -translate-x-1/2 border-r border-b' :
              position === 'bottom' ? 'top-[-5px] left-1/2 -translate-x-1/2 border-l border-t' :
              position === 'left' ? 'right-[-5px] top-1/2 -translate-y-1/2 border-t border-r' :
              'left-[-5px] top-1/2 -translate-y-1/2 border-b border-l'
            }`} />
          </div>
        </div>
      )}
    </div>
  )
}

export function MetricTooltip({ metric, formula, description }: { metric: string; formula?: string; description: string }) {
  const content = (
    <div className="space-y-1">
      <div className="font-semibold text-blue-400">{metric}</div>
      {formula && (
        <div className="text-xs font-mono text-gray-400 border-t border-gray-700 pt-1 mt-1">
          {formula}
        </div>
      )}
      <div className="text-gray-300">{description}</div>
    </div>
  )

  return (
    <div className="relative inline-flex items-center group">
      <HelpCircle className="w-3 h-3 text-gray-500 hover:text-gray-400 transition-colors cursor-help ml-1" />
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
        <div className="bg-gray-900 text-gray-200 text-xs rounded-lg px-3 py-2 shadow-lg border border-gray-700 min-w-[200px] max-w-xs">
          {content}
        </div>
        <div className="absolute w-2 h-2 bg-gray-900 border-gray-700 transform rotate-45 bottom-[-5px] left-1/2 -translate-x-1/2 border-r border-b" />
      </div>
    </div>
  )
}