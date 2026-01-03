/**
 * React Query hooks for glucose data
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  glucoseApi,
  predictionsApi,
  calculationsApi,
  treatmentsApi,
  insightsApi,
  GlucoseCurrentResponse,
  GlucoseHistoryResponse,
  RangeStats,
  Treatment,
  TreatmentCreate,
} from '@/lib/api'

// Default user ID (will be replaced with auth)
const DEFAULT_USER_ID = 'demo_user'

// Query keys
export const queryKeys = {
  glucoseCurrent: (userId: string) => ['glucose', 'current', userId],
  glucoseHistory: (userId: string, hours: number) => ['glucose', 'history', userId, hours],
  rangeStats: (userId: string, hours: number) => ['glucose', 'rangeStats', userId, hours],
  treatments: (userId: string, hours: number) => ['treatments', userId, hours],
  iob: (userId: string) => ['calculations', 'iob', userId],
  cob: (userId: string) => ['calculations', 'cob', userId],
  calculationsSummary: (userId: string) => ['calculations', 'summary', userId],
  predictions: (userId: string) => ['predictions', userId],
  insights: (userId: string) => ['insights', userId],
  anomalies: (userId: string, hours: number) => ['anomalies', userId, hours],
}

// ==================== Glucose Hooks ====================

export function useCurrentGlucose(userId: string = DEFAULT_USER_ID) {
  return useQuery<GlucoseCurrentResponse>({
    queryKey: queryKeys.glucoseCurrent(userId),
    queryFn: () => glucoseApi.getCurrent(userId),
    refetchInterval: 60000, // Refetch every minute
    staleTime: 30000, // Consider stale after 30 seconds
  })
}

export function useGlucoseHistory(
  userId: string = DEFAULT_USER_ID,
  hours: number = 24
) {
  return useQuery<GlucoseHistoryResponse>({
    queryKey: queryKeys.glucoseHistory(userId, hours),
    queryFn: () => glucoseApi.getHistory(userId, hours),
    staleTime: 60000,
  })
}

export function useRangeStats(
  userId: string = DEFAULT_USER_ID,
  hours: number = 24
) {
  return useQuery<RangeStats>({
    queryKey: queryKeys.rangeStats(userId, hours),
    queryFn: () => glucoseApi.getRangeStats(userId, hours),
    staleTime: 300000, // 5 minutes
  })
}

// ==================== Treatment Hooks ====================

export function useRecentTreatments(
  userId: string = DEFAULT_USER_ID,
  hours: number = 24
) {
  return useQuery<Treatment[]>({
    queryKey: queryKeys.treatments(userId, hours),
    queryFn: () => treatmentsApi.getRecent(userId, hours),
    staleTime: 60000,
  })
}

export function useLogTreatment(userId: string = DEFAULT_USER_ID) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (treatment: TreatmentCreate) =>
      treatmentsApi.create(userId, treatment),
    onSuccess: () => {
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['treatments'] })
      queryClient.invalidateQueries({ queryKey: ['calculations'] })
      queryClient.invalidateQueries({ queryKey: ['glucose', 'current'] })
    },
  })
}

// ==================== Calculation Hooks ====================

export function useIOB(userId: string = DEFAULT_USER_ID) {
  return useQuery({
    queryKey: queryKeys.iob(userId),
    queryFn: () => calculationsApi.getIob(userId),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

export function useCOB(userId: string = DEFAULT_USER_ID) {
  return useQuery({
    queryKey: queryKeys.cob(userId),
    queryFn: () => calculationsApi.getCob(userId),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

export function useCalculationsSummary(userId: string = DEFAULT_USER_ID) {
  return useQuery({
    queryKey: queryKeys.calculationsSummary(userId),
    queryFn: () => calculationsApi.getSummary(userId),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

export function useDoseCalculation(userId: string = DEFAULT_USER_ID) {
  return useMutation({
    mutationFn: ({
      currentBg,
      targetBg,
      isfOverride,
    }: {
      currentBg: number
      targetBg?: number
      isfOverride?: number
    }) => calculationsApi.calculateDose(userId, currentBg, targetBg, isfOverride),
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

export function useInsights(userId: string = DEFAULT_USER_ID, limit: number = 10) {
  return useQuery({
    queryKey: queryKeys.insights(userId),
    queryFn: () => insightsApi.getAll(userId, undefined, limit),
    staleTime: 300000,
  })
}

export function useAnomalies(
  userId: string = DEFAULT_USER_ID,
  hours: number = 24
) {
  return useQuery({
    queryKey: queryKeys.anomalies(userId, hours),
    queryFn: () => insightsApi.getAnomalies(userId, hours),
    staleTime: 300000,
  })
}

export function usePatterns(userId: string = DEFAULT_USER_ID, days: number = 14) {
  return useQuery({
    queryKey: ['patterns', userId, days],
    queryFn: () => insightsApi.getPatterns(userId, days),
    staleTime: 3600000, // 1 hour
  })
}

// ==================== Combined Dashboard Hook ====================

export function useDashboardData(userId: string = DEFAULT_USER_ID) {
  const currentGlucose = useCurrentGlucose(userId)
  const treatments = useRecentTreatments(userId, 6)
  const insights = useInsights(userId, 5)

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
