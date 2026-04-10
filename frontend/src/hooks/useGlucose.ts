/**
 * React Query hooks for glucose data
 * User authentication is handled via JWT token in request headers
 */
import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  glucoseApi,
  predictionsApi,
  calculationsApi,
  treatmentsApi,
  insightsApi,
  datasourcesApi,
  GlucoseCurrentResponse,
  GlucoseHistoryResponse,
  RangeStats,
  Treatment,
  TreatmentCreate,
  TreatmentResponse,
  ChatResponse,
  PumpStatus,
} from '@/lib/api'

// Query keys - user ID from JWT token, no longer needed in keys for most endpoints
export const queryKeys = {
  // Glucose endpoints still use user_id for parent/child access
  glucoseCurrent: (userId: string) => ['glucose', 'current', userId],
  glucoseHistory: (userId: string, hours: number) => ['glucose', 'history', userId, hours],
  rangeStats: (userId: string, hours: number) => ['glucose', 'rangeStats', userId, hours],
  // These endpoints use JWT auth, keys don't need userId
  treatments: (hours: number) => ['treatments', hours],
  iob: () => ['calculations', 'iob'],
  cob: () => ['calculations', 'cob'],
  calculationsSummary: () => ['calculations', 'summary'],
  predictions: () => ['predictions'],
  insights: () => ['insights'],
  anomalies: (hours: number) => ['anomalies', hours],
  pumpStatus: (userId: string) => ['glucose', 'pumpStatus', userId],
}

// ==================== Glucose Hooks ====================
// These still use user_id for parent/child access support

export function useCurrentGlucose(userId: string) {
  return useQuery<GlucoseCurrentResponse>({
    queryKey: queryKeys.glucoseCurrent(userId),
    queryFn: () => glucoseApi.getCurrent(userId),
    enabled: !!userId, // Only fetch when userId is available
    refetchInterval: 60000, // Refetch every minute
    staleTime: 30000, // Consider stale after 30 seconds
    retry: (failureCount, error: any) => {
      // Don't retry on auth errors - let the event listener handle redirect
      if (error?.message?.includes('Session expired') || error?.response?.status === 401) {
        console.log('[useCurrentGlucose] Auth error detected, not retrying')
        return false
      }
      return failureCount < 3
    },
  })
}

export function useGlucoseHistory(
  userId: string,
  hours: number = 24
) {
  return useQuery<GlucoseHistoryResponse>({
    queryKey: queryKeys.glucoseHistory(userId, hours),
    queryFn: () => glucoseApi.getHistory(userId, hours),
    enabled: !!userId, // Only fetch when userId is available
    staleTime: 60000,
    refetchInterval: 60000, // Auto-refresh every minute for real-time updates
    retry: (failureCount, error: any) => {
      // Don't retry on auth errors - let the event listener handle redirect
      if (error?.message?.includes('Session expired') || error?.response?.status === 401) {
        console.log('[useGlucoseHistory] Auth error detected, not retrying')
        return false
      }
      return failureCount < 3
    },
  })
}

// Prefetch all time ranges for instant switching
export function usePrefetchAllTimeRanges(userId: string) {
  const queryClient = useQueryClient()
  const timeRanges = [1, 3, 6, 12, 24]

  const prefetchAll = async () => {
    if (!userId) return

    // Prefetch all time ranges in parallel
    await Promise.all(
      timeRanges.map(hours =>
        queryClient.prefetchQuery({
          queryKey: queryKeys.glucoseHistory(userId, hours),
          queryFn: () => glucoseApi.getHistory(userId, hours),
          staleTime: 60000,
        })
      )
    )
  }

  return { prefetchAll, timeRanges }
}

export function useRangeStats(
  userId: string,
  hours: number = 24
) {
  return useQuery<RangeStats>({
    queryKey: queryKeys.rangeStats(userId, hours),
    queryFn: () => glucoseApi.getRangeStats(userId, hours),
    enabled: !!userId, // Only fetch when userId is available
    staleTime: 300000, // 5 minutes
  })
}

export function usePumpStatus(userId: string) {
  return useQuery<PumpStatus | null>({
    queryKey: queryKeys.pumpStatus(userId),
    queryFn: () => glucoseApi.getPumpStatus(userId),
    enabled: !!userId,
    refetchInterval: 300000, // Every 5 min (matches Tandem sync cadence)
    staleTime: 120000,
  })
}

// ==================== Treatment Hooks ====================
// User ID extracted from JWT token on backend

export function useRecentTreatments(hours: number = 24, userId?: string) {
  return useQuery<Treatment[]>({
    queryKey: [...queryKeys.treatments(hours), userId],
    queryFn: () => treatmentsApi.getRecent(hours, userId),
    staleTime: 60000,
    refetchInterval: 60000, // Auto-refresh every minute
    retry: (failureCount, error: any) => {
      // Don't retry on auth errors - let the event listener handle redirect
      if (error?.message?.includes('Session expired') || error?.response?.status === 401) {
        console.log('[useRecentTreatments] Auth error detected, not retrying')
        return false
      }
      return failureCount < 3
    },
  })
}

export function useLogTreatment() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (treatment: TreatmentCreate) =>
      treatmentsApi.create(treatment),
    onSuccess: (_response: TreatmentResponse) => {
      // ALWAYS invalidate ALL related queries after treatment
      // TFT predictions must recalculate with new IOB/COB
      queryClient.invalidateQueries({ queryKey: ['treatments'] })
      queryClient.invalidateQueries({ queryKey: ['calculations'] })
      // Always refresh glucose/predictions - TFT recalculates with new IOB/COB
      queryClient.invalidateQueries({ queryKey: ['glucose', 'current'] })
      queryClient.invalidateQueries({ queryKey: ['glucose', 'history'] })
      queryClient.invalidateQueries({ queryKey: ['predictions'] })
    },
  })
}

export function useUpdateTreatment() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ treatmentId, data }: { treatmentId: string; data: Partial<TreatmentCreate> }) => {
      console.log('[useUpdateTreatment] Updating treatment:', treatmentId, data)
      const result = await treatmentsApi.update(treatmentId, data)
      console.log('[useUpdateTreatment] Update successful:', result)
      return result
    },
    onSuccess: () => {
      console.log('[useUpdateTreatment] Invalidating queries')
      // Invalidate all related queries
      queryClient.invalidateQueries({ queryKey: ['treatments'] })
      queryClient.invalidateQueries({ queryKey: ['calculations'] })
      queryClient.invalidateQueries({ queryKey: ['glucose', 'current'] })
      queryClient.invalidateQueries({ queryKey: ['glucose', 'history'] })
      queryClient.invalidateQueries({ queryKey: ['predictions'] })
    },
    onError: (error) => {
      console.error('[useUpdateTreatment] Error updating treatment:', error)
      // Show error to user via alert for now
      alert(`Failed to update treatment: ${error instanceof Error ? error.message : 'Unknown error'}`)
    },
  })
}

export function useDeleteTreatment() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (treatmentId: string) =>
      treatmentsApi.delete(treatmentId),
    onSuccess: () => {
      // Invalidate all related queries
      queryClient.invalidateQueries({ queryKey: ['treatments'] })
      queryClient.invalidateQueries({ queryKey: ['calculations'] })
      queryClient.invalidateQueries({ queryKey: ['glucose', 'current'] })
      queryClient.invalidateQueries({ queryKey: ['glucose', 'history'] })
      queryClient.invalidateQueries({ queryKey: ['predictions'] })
    },
  })
}

// ==================== Gluroo Sync Hook ====================

/**
 * Hook to manually sync data from Gluroo.
 * Returns the number of glucose readings and treatments synced.
 */
export function useGlurooSync() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (fullSync: boolean = false) =>
      datasourcesApi.sync(fullSync),
    onSuccess: (data) => {
      console.log(`Gluroo sync completed: ${data.glucoseReadings} readings, ${data.treatments} treatments`)
      // Invalidate all data queries to show new data
      queryClient.invalidateQueries({ queryKey: ['glucose'] })
      queryClient.invalidateQueries({ queryKey: ['treatments'] })
      queryClient.invalidateQueries({ queryKey: ['calculations'] })
      queryClient.invalidateQueries({ queryKey: ['predictions'] })
    },
    onError: (error) => {
      console.error('Gluroo sync failed:', error)
    },
  })
}

// ==================== Calculation Hooks ====================
// User ID extracted from JWT token on backend

export function useIOB() {
  return useQuery({
    queryKey: queryKeys.iob(),
    queryFn: () => calculationsApi.getIob(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

export function useCOB() {
  return useQuery({
    queryKey: queryKeys.cob(),
    queryFn: () => calculationsApi.getCob(),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

export function useCalculationsSummary(userId?: string) {
  return useQuery({
    queryKey: ['calculations', 'summary', userId],
    queryFn: () => calculationsApi.getSummary(120, userId),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

export function useDoseCalculation() {
  return useMutation({
    mutationFn: ({
      currentBg,
      targetBg,
      isfOverride,
    }: {
      currentBg: number
      targetBg?: number
      isfOverride?: number
    }) => calculationsApi.calculateDose(currentBg, targetBg, isfOverride),
  })
}

// ==================== Prediction Hooks ====================

export function usePredictionStatus() {
  return useQuery({
    queryKey: ['predictions', 'status'],
    queryFn: () => predictionsApi.getStatus(),
    staleTime: 300000, // 5 minutes
  })
}

export function usePredictionAccuracy() {
  return useQuery({
    queryKey: ['predictions', 'accuracy'],
    queryFn: () => predictionsApi.getAccuracy(),
    staleTime: 300000,
  })
}

// ==================== Insight Hooks ====================
// User ID extracted from JWT token on backend

export function useInsights(limit: number = 10) {
  return useQuery({
    queryKey: queryKeys.insights(),
    queryFn: () => insightsApi.getAll(undefined, limit),
    staleTime: 300000,
  })
}

/**
 * Tracks whether the user has interacted with the page recently.
 * Returns false after `timeoutMs` of no mouse/touch/keyboard/scroll activity.
 * Used to gate AI polling so calls only happen when someone is actively using the app.
 */
function useUserActivity(timeoutMs = 300000): boolean {
  const [isActive, setIsActive] = useState(true)

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>

    const reset = () => {
      setIsActive(true)
      clearTimeout(timer)
      timer = setTimeout(() => setIsActive(false), timeoutMs)
    }

    const events = ['mousedown', 'mousemove', 'touchstart', 'keypress', 'scroll', 'click']
    events.forEach(e => window.addEventListener(e, reset, { passive: true }))
    reset() // start the timer immediately

    return () => {
      events.forEach(e => window.removeEventListener(e, reset))
      clearTimeout(timer)
    }
  }, [timeoutMs])

  return isActive
}

/**
 * Hook for real-time AI insights that update when data changes.
 * Provides actionable advice based on current BG, IOB, COB, ISF, ICR, PIR, dose, and TFT predictions.
 * Enhanced with BG Pressure, TFT predictions, and GI data for better advice.
 */
export function useRealtimeInsight(params: {
  currentBg?: number
  trend?: string
  iob?: number
  cob?: number
  enabled?: boolean
  // All diabetes metrics for comprehensive AI advice
  isf?: number
  icr?: number
  pir?: number
  dose?: number
  bgPressure?: number
  tftPredictions?: Array<{ horizon: number; value: number; lower: number; upper: number }>
  recentGI?: number
  absorptionRate?: string
}) {
  const {
    currentBg, trend, iob, cob, enabled = true,
    isf, icr, pir, dose, bgPressure, tftPredictions, recentGI, absorptionRate
  } = params

  const isUserActive = useUserActivity(300000) // 5 min idle timeout

  return useQuery({
    queryKey: ['insights', 'realtime'],
    queryFn: () => insightsApi.getRealtime({
      currentBg: currentBg!,
      trend,
      iob,
      cob,
      // All diabetes metrics for comprehensive AI advice
      isf,
      icr,
      pir,
      dose,
      bgPressure,
      tftPredictions,
      recentGI,
      absorptionRate,
    }),
    enabled: enabled && currentBg !== undefined && currentBg > 0 && isUserActive,
    staleTime: 300000, // 5 minutes - reuse cached response within window
    refetchInterval: isUserActive ? 300000 : false, // stop polling when idle
    refetchIntervalInBackground: false, // Never poll when tab is hidden
    retry: 1, // Only retry once on failure
  })
}

export function useAnomalies(hours: number = 24) {
  return useQuery({
    queryKey: queryKeys.anomalies(hours),
    queryFn: () => insightsApi.getAnomalies(hours),
    staleTime: 300000,
  })
}

export function usePatterns(days: number = 14) {
  return useQuery({
    queryKey: ['patterns', days],
    queryFn: () => insightsApi.getPatterns(days),
    staleTime: 3600000, // 1 hour
  })
}

// ==================== Combined Dashboard Hook ====================

export function useDashboardData(userId: string) {
  const currentGlucose = useCurrentGlucose(userId)
  const treatments = useRecentTreatments(6)
  const insights = useInsights(5)

  return {
    currentGlucose,
    treatments,
    insights,
    isLoading: currentGlucose.isLoading || treatments.isLoading,
    isError: currentGlucose.isError || treatments.isError,
    refetch: () => {
      currentGlucose.refetch()
      treatments.refetch()
      insights.refetch()
    },
  }
}

// ==================== AI Chat Hook ====================

/**
 * Hook for AI chat "what-if" questions.
 * Allows users to ask questions like "What will happen if I eat 37 carbs?"
 * Passes all diabetes metrics for accurate calculations.
 */
export function useAIChat() {
  return useMutation<ChatResponse, Error, {
    question: string
    currentBg?: number
    trend?: string
    iob?: number
    cob?: number
    isf?: number
    icr?: number
    pir?: number
    dose?: number
    bgPressure?: number
    tftPredictions?: Array<{ horizon: number; value: number; lower: number; upper: number }>
  }>({
    mutationFn: (params) => insightsApi.chat(params),
  })
}

// ==================== Training Status Hooks ====================

import { trainingApi, ISFStatus, ICRStatus, PIRStatus } from '@/lib/api'

/**
 * Hook to get current ISF learning status.
 * Returns learned ISF values and source info.
 * Supports viewing shared accounts when userId is provided.
 */
export function useISFStatus(userId?: string) {
  return useQuery<ISFStatus>({
    queryKey: ['training', 'isf', 'status', userId],
    queryFn: () => trainingApi.getISFStatus(userId),
    staleTime: 60000, // 1 minute
  })
}

/**
 * Hook to get current ICR learning status.
 * Returns learned ICR values and source info.
 * Supports viewing shared accounts when userId is provided.
 */
export function useICRStatus(userId?: string) {
  return useQuery<ICRStatus>({
    queryKey: ['training', 'icr', 'status', userId],
    queryFn: () => trainingApi.getICRStatus(userId),
    staleTime: 60000, // 1 minute
  })
}

/**
 * Hook to get current PIR learning status.
 * Returns learned PIR values and source info.
 * Supports viewing shared accounts when userId is provided.
 */
export function usePIRStatus(userId?: string) {
  return useQuery<PIRStatus>({
    queryKey: ['training', 'pir', 'status', userId],
    queryFn: () => trainingApi.getPIRStatus(userId),
    staleTime: 60000, // 1 minute
  })
}

/**
 * Combined hook for all learned ratios.
 * Used by Dashboard to display learned values.
 * Supports viewing shared accounts when userId is provided.
 */
export function useLearnedRatios(userId?: string) {
  const isfStatus = useISFStatus(userId)
  const icrStatus = useICRStatus(userId)
  const pirStatus = usePIRStatus(userId)

  // Calculate baseline ISF (long-term learned value)
  const baselineIsf = isfStatus.data?.meal_isf ?? isfStatus.data?.fasting_isf ?? isfStatus.data?.default_isf ?? 50

  return {
    isfStatus,
    icrStatus,
    pirStatus,
    isLoading: isfStatus.isLoading || icrStatus.isLoading || pirStatus.isLoading,
    isf: {
      value: baselineIsf,
      source: isfStatus.data?.has_learned_isf ? 'learned' : 'default',
      confidence: isfStatus.data?.meal_confidence ?? isfStatus.data?.fasting_confidence ?? 0,
      // Short-term ISF fields for detecting temporary changes (illness, exercise, etc.)
      currentIsf: isfStatus.data?.current_isf ?? null,
      currentIsfConfidence: isfStatus.data?.current_isf_confidence ?? null,
      currentIsfSamples: isfStatus.data?.current_isf_samples ?? 0,
      deviation: isfStatus.data?.isf_deviation ?? null,  // % deviation from baseline
      recentDataPoints: isfStatus.data?.recent_data_points ?? null,
    },
    icr: {
      value: icrStatus.data?.overall_icr ?? icrStatus.data?.default_icr ?? 10,
      source: icrStatus.data?.has_learned_icr ? 'learned' : 'default',
      confidence: icrStatus.data?.overall_confidence ?? 0,
      // Short-term ICR fields for detecting temporary changes
      currentIcr: icrStatus.data?.current_icr ?? null,
      currentIcrConfidence: icrStatus.data?.current_icr_confidence ?? null,
      currentIcrSamples: icrStatus.data?.current_icr_samples ?? 0,
      deviation: icrStatus.data?.icr_deviation ?? null,  // % deviation from baseline
    },
    pir: {
      value: pirStatus.data?.overall_pir ?? pirStatus.data?.default_pir ?? 25,
      source: pirStatus.data?.has_learned_pir ? 'learned' : 'default',
      confidence: pirStatus.data?.overall_confidence ?? 0,
      // Short-term PIR fields for detecting temporary changes (when backend supports it)
      currentPir: pirStatus.data?.current_pir ?? null,
      currentPirConfidence: pirStatus.data?.current_pir_confidence ?? null,
      currentPirSamples: pirStatus.data?.current_pir_samples ?? 0,
      deviation: pirStatus.data?.pir_deviation ?? null,  // % deviation from baseline
      // Protein timing info
      onsetMin: pirStatus.data?.avg_onset_minutes ?? null,
      peakMin: pirStatus.data?.avg_peak_minutes ?? null,
    },
  }
}
