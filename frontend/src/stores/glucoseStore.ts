/**
 * Zustand store for glucose state management
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

// ==================== Types ====================

export interface GlucoseReading {
  timestamp: string
  value: number
  trend: string | null
}

export interface Metrics {
  iob: number
  cob: number
  isf: number
  recommendedDose: number
}

export interface Predictions {
  linear: number[]
  lstm: number[] | null
  timestamp: string
}

export interface UserPreferences {
  targetBg: number
  highThreshold: number
  lowThreshold: number
  criticalHighThreshold: number
  criticalLowThreshold: number
  enableAlerts: boolean
  enablePredictiveAlerts: boolean
  darkMode: boolean
  chartTimeRange: '1hr' | '3hr' | '6hr' | '12hr' | '24hr'
}

// ==================== Store Interface ====================

interface GlucoseState {
  // Current glucose data
  currentGlucose: GlucoseReading | null
  metrics: Metrics | null
  predictions: Predictions | null

  // Historical data (for offline support)
  glucoseHistory: GlucoseReading[]

  // User preferences
  preferences: UserPreferences

  // UI state
  isConnected: boolean
  lastSyncTime: string | null
  isSyncing: boolean

  // Actions
  setCurrentGlucose: (reading: GlucoseReading) => void
  setMetrics: (metrics: Metrics) => void
  setPredictions: (predictions: Predictions) => void
  addGlucoseReading: (reading: GlucoseReading) => void
  setGlucoseHistory: (history: GlucoseReading[]) => void
  setConnectionStatus: (connected: boolean) => void
  setLastSyncTime: (time: string) => void
  setSyncing: (syncing: boolean) => void
  updatePreferences: (prefs: Partial<UserPreferences>) => void
  resetStore: () => void
}

// ==================== Initial State ====================

const defaultPreferences: UserPreferences = {
  targetBg: 100,
  highThreshold: 180,
  lowThreshold: 70,
  criticalHighThreshold: 250,
  criticalLowThreshold: 54,
  enableAlerts: true,
  enablePredictiveAlerts: true,
  darkMode: true,
  chartTimeRange: '3hr',
}

const initialState = {
  currentGlucose: null,
  metrics: null,
  predictions: null,
  glucoseHistory: [],
  preferences: defaultPreferences,
  isConnected: false,
  lastSyncTime: null,
  isSyncing: false,
}

// ==================== Store ====================

export const useGlucoseStore = create<GlucoseState>()(
  persist(
    (set, get) => ({
      ...initialState,

      setCurrentGlucose: (reading) =>
        set({ currentGlucose: reading }),

      setMetrics: (metrics) =>
        set({ metrics }),

      setPredictions: (predictions) =>
        set({ predictions }),

      addGlucoseReading: (reading) =>
        set((state) => {
          // Add to history, keeping only last 24 hours of data
          const newHistory = [...state.glucoseHistory, reading]
          const cutoff = Date.now() - 24 * 60 * 60 * 1000
          const filtered = newHistory.filter(
            (r) => new Date(r.timestamp).getTime() > cutoff
          )
          return { glucoseHistory: filtered }
        }),

      setGlucoseHistory: (history) =>
        set({ glucoseHistory: history }),

      setConnectionStatus: (connected) =>
        set({ isConnected: connected }),

      setLastSyncTime: (time) =>
        set({ lastSyncTime: time }),

      setSyncing: (syncing) =>
        set({ isSyncing: syncing }),

      updatePreferences: (prefs) =>
        set((state) => ({
          preferences: { ...state.preferences, ...prefs },
        })),

      resetStore: () =>
        set(initialState),
    }),
    {
      name: 't1d-ai-glucose-storage',
      partialize: (state) => ({
        preferences: state.preferences,
        glucoseHistory: state.glucoseHistory.slice(-100), // Only persist last 100 readings
      }),
    }
  )
)

// ==================== Selectors ====================

export const selectCurrentBg = (state: GlucoseState) =>
  state.currentGlucose?.value ?? null

export const selectTrend = (state: GlucoseState) =>
  state.currentGlucose?.trend ?? null

export const selectIOB = (state: GlucoseState) =>
  state.metrics?.iob ?? 0

export const selectCOB = (state: GlucoseState) =>
  state.metrics?.cob ?? 0

export const selectISF = (state: GlucoseState) =>
  state.metrics?.isf ?? 50

export const selectRecommendedDose = (state: GlucoseState) =>
  state.metrics?.recommendedDose ?? 0

export const selectLinearPredictions = (state: GlucoseState) =>
  state.predictions?.linear ?? []

export const selectLstmPredictions = (state: GlucoseState) =>
  state.predictions?.lstm ?? null

export const selectChartTimeRange = (state: GlucoseState) =>
  state.preferences.chartTimeRange

export const selectIsInRange = (state: GlucoseState) => {
  const bg = state.currentGlucose?.value
  if (!bg) return null
  const { lowThreshold, highThreshold } = state.preferences
  return bg >= lowThreshold && bg <= highThreshold
}

// ==================== Helper Hooks ====================

// Hook to get glucose range status
export function useGlucoseRange() {
  const currentBg = useGlucoseStore(selectCurrentBg)
  const preferences = useGlucoseStore((state) => state.preferences)

  if (!currentBg) return null

  const { lowThreshold, highThreshold, criticalLowThreshold, criticalHighThreshold } = preferences

  if (currentBg < criticalLowThreshold) return 'critical-low'
  if (currentBg < lowThreshold) return 'low'
  if (currentBg <= highThreshold) return 'in-range'
  if (currentBg <= criticalHighThreshold) return 'high'
  return 'critical-high'
}

// Hook to check if alert should be shown
export function useShouldShowAlert() {
  const range = useGlucoseRange()
  const preferences = useGlucoseStore((state) => state.preferences)

  if (!preferences.enableAlerts) return false
  if (!range) return false

  return range === 'critical-low' || range === 'critical-high'
}
