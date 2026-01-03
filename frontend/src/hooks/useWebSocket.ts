/**
 * WebSocket hook for real-time glucose updates
 */
import { useEffect, useRef, useCallback, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { queryKeys } from './useGlucose'

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/v1'

export interface GlucoseUpdate {
  type: 'glucose_update' | 'error' | 'pong'
  data?: {
    timestamp: string
    value: number
    trend: string
    trendArrow: string
    predictions: {
      linear: number[]
      lstm: number[] | null
      horizons: number[]
    }
    metrics: {
      iob: number
      cob: number
      isf: number
    }
    modelAvailable: boolean
  }
  message?: string
  serverTime?: string
}

interface UseWebSocketOptions {
  userId: string
  interval?: number
  onMessage?: (data: GlucoseUpdate) => void
  onError?: (error: Event) => void
  onConnect?: () => void
  onDisconnect?: () => void
  autoReconnect?: boolean
  reconnectDelay?: number
}

export function useGlucoseWebSocket({
  userId,
  interval = 60,
  onMessage,
  onError,
  onConnect,
  onDisconnect,
  autoReconnect = true,
  reconnectDelay = 3000,
}: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const queryClient = useQueryClient()

  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<GlucoseUpdate | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<
    'connecting' | 'connected' | 'disconnected' | 'error'
  >('disconnected')

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setConnectionStatus('connecting')

    const wsUrl = `${WS_BASE_URL}/ws/glucose/${userId}?interval=${interval}`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      setIsConnected(true)
      setConnectionStatus('connected')
      onConnect?.()
      console.log('[WebSocket] Connected to glucose stream')
    }

    ws.onmessage = (event) => {
      try {
        const data: GlucoseUpdate = JSON.parse(event.data)
        setLastUpdate(data)

        if (data.type === 'glucose_update' && data.data) {
          // Update React Query cache
          queryClient.setQueryData(
            queryKeys.glucoseCurrent(userId),
            (oldData: any) => {
              if (!oldData) return oldData
              return {
                ...oldData,
                glucose: {
                  ...oldData.glucose,
                  value: data.data!.value,
                  trend: data.data!.trend,
                  timestamp: data.data!.timestamp,
                  predictions: {
                    timestamp: data.data!.timestamp,
                    linear: data.data!.predictions.linear,
                    lstm: data.data!.predictions.lstm || [],
                  },
                },
                metrics: {
                  ...oldData.metrics,
                  iob: data.data!.metrics.iob,
                  cob: data.data!.metrics.cob,
                  isf: data.data!.metrics.isf,
                },
              }
            }
          )
        }

        onMessage?.(data)
      } catch (e) {
        console.error('[WebSocket] Failed to parse message:', e)
      }
    }

    ws.onerror = (error) => {
      setConnectionStatus('error')
      console.error('[WebSocket] Error:', error)
      onError?.(error)
    }

    ws.onclose = () => {
      setIsConnected(false)
      setConnectionStatus('disconnected')
      onDisconnect?.()
      console.log('[WebSocket] Disconnected')

      // Auto reconnect
      if (autoReconnect) {
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('[WebSocket] Attempting to reconnect...')
          connect()
        }, reconnectDelay)
      }
    }

    wsRef.current = ws
  }, [userId, interval, onMessage, onError, onConnect, onDisconnect, autoReconnect, reconnectDelay, queryClient])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setIsConnected(false)
    setConnectionStatus('disconnected')
  }, [])

  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  const ping = useCallback(() => {
    sendMessage({ type: 'ping' })
  }, [sendMessage])

  const requestRefresh = useCallback(() => {
    sendMessage({ type: 'refresh' })
  }, [sendMessage])

  // Connect on mount
  useEffect(() => {
    connect()

    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    isConnected,
    connectionStatus,
    lastUpdate,
    connect,
    disconnect,
    ping,
    requestRefresh,
  }
}

// Simple hook for components that just need the connection
export function useGlucoseStream(userId: string = 'demo_user') {
  const [latestData, setLatestData] = useState<GlucoseUpdate['data'] | null>(null)

  const { isConnected, connectionStatus, lastUpdate } = useGlucoseWebSocket({
    userId,
    interval: 60,
    onMessage: (data) => {
      if (data.type === 'glucose_update' && data.data) {
        setLatestData(data.data)
      }
    },
  })

  return {
    isConnected,
    connectionStatus,
    latestData,
    lastUpdate,
  }
}
