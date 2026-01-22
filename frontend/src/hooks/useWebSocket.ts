/**
 * WebSocket hook for real-time glucose updates
 */
import { useEffect, useRef, useCallback, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { queryKeys } from './useGlucose'

// WebSocket URL - construct from current page location for production
const getWsBaseUrl = () => {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL
  }
  // Use current host with appropriate protocol
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/api/v1`
}
const WS_BASE_URL = getWsBaseUrl()

export interface GlucoseUpdate {
  type: 'glucose_update' | 'error' | 'pong' | 'gluroo_sync'
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
      pob: number
      isf: number
      recommendedDose: number
      proteinDoseNow: number
      proteinDoseLater: number
      effectiveBg: number
    }
    modelAvailable: boolean
  }
  message?: string
  serverTime?: string
  // Gluroo sync notification fields
  status?: 'success' | 'error'
  treatment_type?: string
  value?: number
  carbs?: number
  insulin?: number
  notes?: string
  protein?: number
  fat?: number
  glycemicIndex?: number
  absorptionRate?: string
  isLiquid?: boolean
}

export interface GlurooSyncEvent {
  status: 'success' | 'error'
  message: string
  treatment_type: string
  value?: number
  carbs?: number
  insulin?: number
  notes?: string
  protein?: number
  fat?: number
  glycemicIndex?: number
  absorptionRate?: string
  isLiquid?: boolean
}

interface UseWebSocketOptions {
  userId: string
  interval?: number
  onMessage?: (data: GlucoseUpdate) => void
  onError?: (error: Event) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onGlurooSync?: (event: GlurooSyncEvent) => void
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
  onGlurooSync,
  autoReconnect = true,
  reconnectDelay = 3000,
}: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const lastPongRef = useRef<number>(Date.now())
  const reconnectAttemptRef = useRef(0)
  const maxBackoffDelay = 30000 // Cap backoff at 30 seconds (no max attempts - infinite reconnect)
  const heartbeatInterval = 30000 // Send ping every 30 seconds
  const heartbeatTimeout = 45000 // Consider connection dead if no pong in 45 seconds
  const queryClient = useQueryClient()

  // Use refs for callbacks to avoid dependency issues causing reconnects
  const onMessageRef = useRef(onMessage)
  const onErrorRef = useRef(onError)
  const onConnectRef = useRef(onConnect)
  const onDisconnectRef = useRef(onDisconnect)
  const onGlurooSyncRef = useRef(onGlurooSync)

  // Update refs when callbacks change
  useEffect(() => {
    onMessageRef.current = onMessage
    onErrorRef.current = onError
    onConnectRef.current = onConnect
    onDisconnectRef.current = onDisconnect
    onGlurooSyncRef.current = onGlurooSync
  }, [onMessage, onError, onConnect, onDisconnect, onGlurooSync])

  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<GlucoseUpdate | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<
    'connecting' | 'connected' | 'disconnected' | 'error'
  >('disconnected')

  // Start heartbeat to detect stale connections
  const startHeartbeat = useCallback(() => {
    // Clear any existing heartbeat
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current)
    }

    lastPongRef.current = Date.now()

    heartbeatIntervalRef.current = setInterval(() => {
      // Check if we've received a pong recently
      const timeSinceLastPong = Date.now() - lastPongRef.current
      if (timeSinceLastPong > heartbeatTimeout) {
        console.log('[WebSocket] Heartbeat timeout - connection appears dead, reconnecting...')
        // Force close and reconnect
        if (wsRef.current) {
          wsRef.current.close()
        }
        return
      }

      // Send ping if connected
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }))
      }
    }, heartbeatInterval)
  }, [])

  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current)
      heartbeatIntervalRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    // Don't connect with empty userId
    if (!userId) {
      console.log('[WebSocket] Skipping connection - no userId')
      return
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    // No max attempts - always try to reconnect with exponential backoff
    setConnectionStatus('connecting')

    const wsUrl = `${WS_BASE_URL}/ws/glucose/${userId}?interval=${interval}`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      reconnectAttemptRef.current = 0 // Reset on successful connect
      setIsConnected(true)
      setConnectionStatus('connected')
      onConnectRef.current?.()
      console.log('[WebSocket] Connected to glucose stream')
      // Start heartbeat monitoring
      startHeartbeat()
      // Request fresh data immediately on connect
      ws.send(JSON.stringify({ type: 'refresh' }))
    }

    ws.onmessage = (event) => {
      try {
        const data: GlucoseUpdate = JSON.parse(event.data)

        // Track pong responses for heartbeat
        if (data.type === 'pong') {
          lastPongRef.current = Date.now()
          return // Don't process pong as a regular message
        }

        // Any message means connection is alive
        lastPongRef.current = Date.now()
        setLastUpdate(data)

        // Handle Gluroo sync notifications
        if (data.type === 'gluroo_sync') {
          console.log('[WebSocket] Gluroo sync notification:', data)
          onGlurooSyncRef.current?.({
            status: data.status || 'error',
            message: data.message || 'Unknown sync status',
            treatment_type: data.treatment_type || 'unknown',
            value: data.value,
            carbs: data.carbs,
            insulin: data.insulin,
            notes: data.notes,
            protein: data.protein,
            fat: data.fat,
            glycemicIndex: data.glycemicIndex,
            absorptionRate: data.absorptionRate,
            isLiquid: data.isLiquid,
          })

          // Invalidate all data queries when Gluroo syncs new data
          // This ensures charts and metrics refresh with the new treatment data
          queryClient.invalidateQueries({ queryKey: ['glucose'] })
          queryClient.invalidateQueries({ queryKey: ['treatments'] })
          queryClient.invalidateQueries({ queryKey: ['calculations'] })
          queryClient.invalidateQueries({ queryKey: ['predictions'] })
        }

        if (data.type === 'glucose_update' && data.data) {
          // Update React Query cache for current glucose
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

          // Invalidate history queries so chart redraws with new data
          // This triggers a refetch of all glucose history data
          queryClient.invalidateQueries({ queryKey: ['glucose', 'history'] })
          queryClient.invalidateQueries({ queryKey: ['calculations'] })
        }

        onMessageRef.current?.(data)
      } catch (e) {
        console.error('[WebSocket] Failed to parse message:', e)
      }
    }

    ws.onerror = (error) => {
      setConnectionStatus('error')
      console.error('[WebSocket] Error:', error)
      onErrorRef.current?.(error)
    }

    ws.onclose = () => {
      setIsConnected(false)
      setConnectionStatus('disconnected')
      onDisconnectRef.current?.()
      stopHeartbeat()
      console.log('[WebSocket] Disconnected')

      // Auto reconnect with exponential backoff (no max attempts - always retry)
      if (autoReconnect) {
        reconnectAttemptRef.current++
        // Exponential backoff with jitter, capped at maxBackoffDelay
        const baseDelay = Math.min(reconnectDelay * Math.pow(1.5, reconnectAttemptRef.current - 1), maxBackoffDelay)
        const jitter = Math.random() * 1000 // Add 0-1 second jitter
        const delay = baseDelay + jitter

        console.log(`[WebSocket] Reconnecting in ${Math.round(delay / 1000)}s (attempt ${reconnectAttemptRef.current})...`)

        reconnectTimeoutRef.current = setTimeout(() => {
          connect()
        }, delay)
      }
    }

    wsRef.current = ws
  }, [userId, interval, autoReconnect, reconnectDelay, queryClient, startHeartbeat, stopHeartbeat])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    stopHeartbeat()

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setIsConnected(false)
    setConnectionStatus('disconnected')
  }, [stopHeartbeat])

  // Force reconnect - resets attempt counter and immediately reconnects
  const forceReconnect = useCallback(() => {
    console.log('[WebSocket] Force reconnecting...')
    reconnectAttemptRef.current = 0
    disconnect()
    // Small delay to ensure clean disconnect
    setTimeout(() => {
      connect()
    }, 100)
  }, [disconnect, connect])

  // Force refresh all data without page reload
  const forceRefreshAllData = useCallback(async () => {
    console.log('[WebSocket] Force refreshing all data...')

    // Clear all cached data first to ensure fresh fetch
    queryClient.removeQueries({ queryKey: ['glucose'] })
    queryClient.removeQueries({ queryKey: ['treatments'] })
    queryClient.removeQueries({ queryKey: ['calculations'] })
    queryClient.removeQueries({ queryKey: ['predictions'] })
    queryClient.removeQueries({ queryKey: ['insights'] })
    queryClient.removeQueries({ queryKey: ['training'] })

    // Force refetch all queries (this triggers immediate fetch, not just invalidation)
    await Promise.all([
      queryClient.refetchQueries({ queryKey: ['glucose'], type: 'all' }),
      queryClient.refetchQueries({ queryKey: ['treatments'], type: 'all' }),
      queryClient.refetchQueries({ queryKey: ['calculations'], type: 'all' }),
      queryClient.refetchQueries({ queryKey: ['predictions'], type: 'all' }),
      queryClient.refetchQueries({ queryKey: ['insights'], type: 'all' }),
      queryClient.refetchQueries({ queryKey: ['training'], type: 'all' }),
    ])

    // Always force reconnect WebSocket to ensure fresh connection
    forceReconnect()

    console.log('[WebSocket] Force refresh complete')
  }, [queryClient, forceReconnect])

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
      stopHeartbeat()
      disconnect()
    }
  }, [connect, disconnect, stopHeartbeat])

  return {
    isConnected,
    connectionStatus,
    lastUpdate,
    connect,
    disconnect,
    forceReconnect,
    forceRefreshAllData,
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
