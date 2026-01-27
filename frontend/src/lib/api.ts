/**
 * API Client for T1D-AI Backend
 * Includes automatic token refresh on expiry
 */
import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios'
import { isTokenExpired, getTokenTimeRemaining } from '@/stores/authStore'

// Base domain - use env var for local dev, empty for production (same domain)
const BASE_DOMAIN = import.meta.env.VITE_API_URL || ''
// API Base URL - append /api/v1 to the domain
const API_BASE_URL = `${BASE_DOMAIN}/api/v1`

// Token refresh state management
let isRefreshing = false
let refreshSubscribers: ((token: string) => void)[] = []

// Subscribe to token refresh completion
function subscribeTokenRefresh(callback: (token: string) => void) {
  refreshSubscribers.push(callback)
}

// Notify all subscribers when token is refreshed
function onTokenRefreshed(newToken: string) {
  refreshSubscribers.forEach(callback => callback(newToken))
  refreshSubscribers = []
}

// Notify all subscribers when refresh fails
function onRefreshFailed() {
  refreshSubscribers = []
}

/**
 * Attempt to refresh the access token using the refresh token.
 * Returns the new access token on success, null on failure.
 */
async function refreshAccessToken(): Promise<string | null> {
  const stored = localStorage.getItem('t1d-ai-auth')
  if (!stored) return null

  try {
    const data = JSON.parse(stored)
    const refreshToken = data.state?.tokens?.refreshToken

    if (!refreshToken) {
      console.warn('[API] No refresh token available')
      return null
    }

    // Check if refresh token itself is expired
    if (isTokenExpired(refreshToken, 0)) {
      console.warn('[API] Refresh token is expired')
      return null
    }

    console.log('[API] Attempting token refresh...')

    // Use the auth endpoint directly (not through the interceptor)
    const response = await axios.post(`${BASE_DOMAIN}/api/auth/refresh`, {
      refresh_token: refreshToken
    }, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 10000
    })

    const newAccessToken = response.data.access_token
    const newRefreshToken = response.data.refresh_token
    const expiresIn = response.data.expires_in

    // Update stored tokens
    data.state.tokens = {
      accessToken: newAccessToken,
      refreshToken: newRefreshToken,
      expiresIn: expiresIn
    }
    localStorage.setItem('t1d-ai-auth', JSON.stringify(data))

    const remaining = getTokenTimeRemaining(newAccessToken)
    console.log(`[API] Token refreshed successfully, expires in ${Math.round(remaining / 60)} minutes`)

    return newAccessToken

  } catch (error) {
    console.error('[API] Token refresh failed:', error)
    return null
  }
}

/**
 * Clear auth state and trigger logout event
 */
function clearAuthAndLogout() {
  console.warn('[API] Clearing auth state and triggering logout')
  localStorage.removeItem('t1d-ai-auth')
  window.dispatchEvent(new CustomEvent('auth:unauthorized'))
}

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for auth token with automatic refresh
api.interceptors.request.use(async (config: InternalAxiosRequestConfig) => {
  // Get token from Zustand persist storage
  const stored = localStorage.getItem('t1d-ai-auth')
  if (!stored) {
    return config
  }

  try {
    const data = JSON.parse(stored)
    let token = data.state?.tokens?.accessToken
    const refreshToken = data.state?.tokens?.refreshToken

    if (!token) {
      return config
    }

    // Check if token is expired or expiring soon (within 60 seconds)
    if (isTokenExpired(token, 60)) {
      // Token is expired or expiring soon - try to refresh

      if (!refreshToken || isTokenExpired(refreshToken, 0)) {
        // No valid refresh token - must log out
        console.warn('[API] Token expired and no valid refresh token')
        clearAuthAndLogout()
        return Promise.reject(new Error('Session expired - please log in again'))
      }

      // If already refreshing, wait for the refresh to complete
      if (isRefreshing) {
        console.log('[API] Already refreshing, waiting...')
        return new Promise((resolve, reject) => {
          subscribeTokenRefresh((newToken: string) => {
            config.headers.Authorization = `Bearer ${newToken}`
            resolve(config)
          })
          // Set a timeout to avoid hanging forever
          setTimeout(() => {
            reject(new Error('Token refresh timeout'))
          }, 15000)
        })
      }

      // Start refresh
      isRefreshing = true

      try {
        const newToken = await refreshAccessToken()

        if (newToken) {
          token = newToken
          onTokenRefreshed(newToken)
        } else {
          onRefreshFailed()
          clearAuthAndLogout()
          return Promise.reject(new Error('Token refresh failed - please log in again'))
        }
      } finally {
        isRefreshing = false
      }
    }

    config.headers.Authorization = `Bearer ${token}`

  } catch {
    // Invalid JSON, ignore
  }

  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean }

    // Handle 401 Unauthorized
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true

      // Try to refresh the token
      if (!isRefreshing) {
        isRefreshing = true

        try {
          const newToken = await refreshAccessToken()

          if (newToken) {
            isRefreshing = false
            onTokenRefreshed(newToken)

            // Retry the original request with new token
            originalRequest.headers.Authorization = `Bearer ${newToken}`
            return api(originalRequest)
          }
        } catch (refreshError) {
          console.error('[API] Refresh failed in response interceptor:', refreshError)
        } finally {
          isRefreshing = false
        }
      } else {
        // Wait for ongoing refresh
        return new Promise((resolve, reject) => {
          subscribeTokenRefresh((newToken: string) => {
            originalRequest.headers.Authorization = `Bearer ${newToken}`
            resolve(api(originalRequest))
          })
          setTimeout(() => reject(error), 15000)
        })
      }

      // Refresh failed - clear auth and logout
      console.warn('[API] 401 Unauthorized - token refresh failed, logging out')
      clearAuthAndLogout()

      // Return a rejected promise with a clear message for the UI
      return Promise.reject(new Error('Session expired - redirecting to login'))
    }

    return Promise.reject(error)
  }
)

// ==================== Types ====================

export interface GlucoseReading {
  id: string
  userId: string
  timestamp: string
  value: number
  trend: string | null
  source: string
  iob?: number
  cob?: number
  isf?: number
}

export interface GlucosePrediction {
  timestamp: string
  linear: number[]
  lstm: number[]
}

export interface GlucoseWithPredictions extends GlucoseReading {
  predictions: GlucosePrediction | null
}

// Food suggestion based on user's historical eating patterns
export interface FoodSuggestion {
  name: string
  carbs: number
  typical_portion: string
  glycemic_index?: number | null
  times_eaten: number
}

export interface CurrentMetrics {
  iob: number
  cob: number
  pob: number  // Protein on Board (grams)
  isf: number
  recommendedDose: number
  effectiveBg: number
  proteinDoseNow: number   // Protein insulin to give NOW (with decay)
  proteinDoseLater: number // Protein insulin to give LATER (remaining)
  // Food recommendation fields (when BG predicted below target)
  actionType: 'insulin' | 'food' | 'none'
  recommendedCarbs: number
  foodSuggestions: FoodSuggestion[]
  predictedBgWithoutAction: number
  predictedBgWithAction: number
  recommendationReasoning: string
}

export interface PredictionAccuracy {
  linearWins: number
  lstmWins: number
  totalComparisons: number
}

// TFT prediction with uncertainty bands
export interface TFTPrediction {
  timestamp: string
  horizon: number  // 30, 45, 60 minutes
  value: number    // Median (50th percentile)
  lower: number    // 10th percentile
  upper: number    // 90th percentile
  tftDelta?: number  // TFT modifier delta (mg/dL) - how much TFT adjusted physics baseline
}

// IOB/COB/POB effect curve point
export interface EffectPoint {
  minutesAhead: number
  iobEffect: number
  cobEffect: number
  pobEffect?: number  // BG raising from POB (mg/dL, delayed)
  netEffect: number
  remainingIOB?: number
  remainingCOB?: number
  remainingPOB?: number  // POB remaining (grams)
  proteinActivity?: number  // Protein absorption activity level (0-1)
  expectedBg?: number
  bgWithIobOnly?: number
  bgWithCobOnly?: number
}

// Historical IOB/COB/POB point with BG pressure (for continuous plotting)
export interface HistoricalIobCobPoint {
  timestamp: string
  iob: number
  cob: number
  pob: number  // Protein on Board (grams)
  bgPressure?: number  // Where BG is heading based on IOB (down) + COB (up) + POB (up, delayed)
  actualBg?: number    // Actual BG reading at this time (for comparison)
}

export interface GlucoseCurrentResponse {
  glucose: GlucoseWithPredictions
  metrics: CurrentMetrics
  accuracy: PredictionAccuracy
  tftPredictions?: TFTPrediction[]
  effectCurve?: EffectPoint[]
  historicalIobCob?: HistoricalIobCobPoint[]
}

export interface GlucoseHistoryResponse {
  readings: GlucoseReading[]
  totalCount: number
  startTime: string
  endTime: string
}

export interface RangeStats {
  totalReadings: number
  criticalLow: number
  low: number
  inRange: number
  high: number
  criticalHigh: number
  averageBg: number | null
  estimatedA1c: number | null
}

export interface Treatment {
  id: string
  userId: string
  timestamp: string
  type: string
  insulin?: number
  carbs?: number
  protein?: number
  fat?: number
  notes?: string
  source?: string             // 'manual', 'gluroo', etc.
  // AI glycemic prediction fields
  glycemicIndex?: number      // 0-100 precise value from GI database or GPT
  glycemicLoad?: number       // carbs * GI / 100
  absorptionRate?: 'very_fast' | 'fast' | 'medium' | 'slow' | 'very_slow'
  fatContent?: 'none' | 'low' | 'medium' | 'high'
  isLiquid?: boolean          // True for drinks - absorb 40% faster
  enrichedAt?: string         // When food was analyzed by AI
}

export interface TreatmentCreate {
  type: 'insulin' | 'carbs'
  insulin?: number
  carbs?: number
  protein?: number
  fat?: number
  notes?: string  // Food description for AI glycemic prediction
  timestamp?: string  // ISO timestamp, defaults to now if not provided
}

export interface TreatmentResponse {
  treatment: Treatment
  predictionsStale: boolean  // True if predictions should be refreshed
}

export interface DoseCalculation {
  currentBg: number
  targetBg: number
  effectiveBg: number
  iob: number
  cob: number
  isf: number
  iobEffectMgdl: number
  cobEffectMgdl: number
  rawCorrectionUnits: number
  recommendedDoseUnits: number
  formula: string
  warning?: string
  timestamp: string
}

export interface UserSettings {
  timezone: string
  targetBg: number
  insulinSensitivity: number
  carbRatio: number
  insulinDuration: number
  carbAbsorptionDuration: number
  highThreshold: number
  lowThreshold: number
  criticalHighThreshold: number
  criticalLowThreshold: number
  enableAlerts: boolean
  enablePredictiveAlerts: boolean
  showInsights: boolean
  // Prediction settings
  useTFTModifiers?: boolean
  trackPredictionAccuracy?: boolean
}

export interface User {
  id: string
  email: string
  displayName?: string
  createdAt: string
  settings: UserSettings
}

export interface Insight {
  id: string
  content: string
  category: string
  createdAt: string
  expiresAt?: string
}

export interface Pattern {
  type: string
  description: string
  timeOfDay?: string
  dayOfWeek?: string
  frequency: string
  confidence: number
  recommendation?: string
}

export interface Anomaly {
  type: string
  severity: 'info' | 'warning' | 'critical'
  value: number
  expectedRange: [number, number]
  timestamp: string
  context?: string
}

// ==================== API Functions ====================

// Glucose endpoints
export const glucoseApi = {
  getCurrent: async (userId: string = 'demo_user'): Promise<GlucoseCurrentResponse> => {
    const response = await api.get('/glucose/current', { params: { user_id: userId } })
    return response.data
  },

  getHistory: async (
    userId: string = 'demo_user',
    hours: number = 24,
    limit: number = 1000
  ): Promise<GlucoseHistoryResponse> => {
    const response = await api.get('/glucose/history', {
      params: { user_id: userId, hours, limit }
    })
    return response.data
  },

  getRangeStats: async (userId: string = 'demo_user', hours: number = 24): Promise<RangeStats> => {
    const response = await api.get('/glucose/range-stats', {
      params: { user_id: userId, hours }
    })
    return response.data
  },
}

// Predictions endpoints
export const predictionsApi = {
  getBgPrediction: async (
    userId: string,
    currentBg: number,
    trend: number = 0
  ) => {
    const response = await api.post('/predictions/bg', {
      current_bg: currentBg,
      trend,
      include_history: true,
      history_minutes: 120
    }, { params: { user_id: userId } })
    return response.data
  },

  getIsf: async (userId: string = 'demo_user') => {
    const response = await api.get('/predictions/isf', { params: { user_id: userId } })
    return response.data
  },

  getAccuracy: async () => {
    const response = await api.get('/predictions/accuracy')
    return response.data
  },

  getStatus: async () => {
    const response = await api.get('/predictions/status')
    return response.data
  },

  /**
   * Get dual BG predictions comparing learned model vs standard formula.
   * Model-based: Uses LEARNED absorption curves from this person's BG data
   * Hardcoded: Uses standard textbook parameters (adult average)
   */
  getDualPredictions: async (userId: string = 'demo_user') => {
    const response = await api.get('/predictions/dual', { params: { user_id: userId } })
    return response.data as {
      model_predictions: Array<{
        minutesAhead: number
        predictedBg: number
      }>
      hardcoded_predictions: Array<{
        minutesAhead: number
        predictedBg: number
      }>
      learned_parameters: {
        iob_onset_min: number
        iob_half_life_min: number
        cob_onset_min: number
        cob_half_life_min: number
      }
      current_bg: number
      isf: number
    }
  },
}

// Calculations endpoints (supports shared access via user_id)
export const calculationsApi = {
  getIob: async (hours: number = 6, userId?: string) => {
    const response = await api.get('/calculations/iob', {
      params: { hours, user_id: userId }
    })
    return response.data
  },

  getCob: async (hours: number = 6, userId?: string) => {
    const response = await api.get('/calculations/cob', {
      params: { hours, user_id: userId }
    })
    return response.data
  },

  getPob: async (hours: number = 6, userId?: string): Promise<{
    pob: number
    total_protein_24h: number
    absorption_duration_min: number
    half_life_min: number
    onset_min: number
    bg_impact: number
    timestamp: string
  }> => {
    const response = await api.get('/calculations/pob', {
      params: { hours, user_id: userId }
    })
    return response.data
  },

  getProteinDoseDecay: async (
    treatmentId: string,
    upfrontPercent: number = 40
  ): Promise<{
    time_since_meal_min: number
    original_dose_now: number
    original_dose_later: number
    current_dose_now: number
    current_dose_later: number
    decayed_amount: number
    decay_percent: number
    all_now: boolean
    timestamp: string
  }> => {
    const response = await api.get('/calculations/protein-dose-decay', {
      params: { treatment_id: treatmentId, upfront_percent: upfrontPercent }
    })
    return response.data
  },

  calculateDose: async (
    currentBg: number,
    targetBg: number = 100,
    isfOverride?: number
  ): Promise<DoseCalculation> => {
    const response = await api.post('/calculations/dose', {
      current_bg: currentBg,
      target_bg: targetBg,
      isf_override: isfOverride,
      include_cob: true
    })
    return response.data
  },

  getSummary: async (currentBg: number = 120, userId?: string) => {
    const response = await api.get('/calculations/summary', {
      params: { current_bg: currentBg, user_id: userId }
    })
    return response.data
  },

  getActiveInsulin: async () => {
    const response = await api.get('/calculations/active-insulin')
    return response.data
  },
}

// Treatments endpoints (user ID from JWT token, or provided for shared access)
export const treatmentsApi = {
  getRecent: async (hours: number = 24, userId?: string): Promise<Treatment[]> => {
    const response = await api.get('/treatments/recent', {
      params: { hours, user_id: userId }
    })
    return response.data
  },

  /**
   * Log a new treatment with AI glycemic prediction for carbs.
   * Returns predictionsStale=true to signal that predictions should be refreshed.
   */
  create: async (treatment: TreatmentCreate): Promise<TreatmentResponse> => {
    const response = await api.post('/treatments/log', treatment)
    return response.data
  },

  getIob: async () => {
    const response = await api.get('/treatments/iob')
    return response.data
  },

  getCob: async () => {
    const response = await api.get('/treatments/cob')
    return response.data
  },

  /**
   * Update an existing treatment.
   */
  update: async (treatmentId: string, data: Partial<TreatmentCreate>): Promise<TreatmentResponse> => {
    const response = await api.put(`/treatments/${treatmentId}`, data)
    return response.data
  },

  /**
   * Delete a treatment.
   */
  delete: async (treatmentId: string): Promise<{ message: string; predictionsStale: boolean }> => {
    const response = await api.delete(`/treatments/${treatmentId}`)
    return response.data
  },
}

// Users endpoints
export const usersApi = {
  getMe: async (userId: string): Promise<User> => {
    const response = await api.get('/users/me', { params: { user_id: userId } })
    return response.data
  },

  getSettings: async (userId: string): Promise<{ settings: UserSettings; updatedAt: string }> => {
    const response = await api.get('/users/settings', { params: { user_id: userId } })
    return response.data
  },

  updateSettings: async (userId: string, settings: Partial<UserSettings>) => {
    const response = await api.put('/users/settings', settings, {
      params: { user_id: userId }
    })
    return response.data
  },

  getDefaultSettings: async () => {
    const response = await api.get('/users/settings/defaults')
    return response.data
  },
}

// Realtime insight response
export interface RealtimeInsight {
  insight: string
  urgency: 'low' | 'normal' | 'high' | 'critical'
  action?: string
  reasoning: string
  generatedAt: string
}

// AI Chat response for what-if questions
export interface ChatResponse {
  response: string
  prediction?: {
    bg30min?: number
    bg60min?: number
    bg90min?: number
    peakBg?: number
    timeOfPeak?: string
  }
  recommendation?: {
    insulin?: number
    timing?: string
    prebolus?: string
  }
  calculation?: string
  confidence: 'low' | 'medium' | 'high'
  generatedAt: string
}

// Insights endpoints (user ID from JWT token)
export const insightsApi = {
  /**
   * Get real-time AI insight based on current diabetes state.
   * Returns immediate, actionable advice with specific recommendations.
   * Enhanced with ISF, BG Pressure, TFT predictions, and GI data.
   */
  getRealtime: async (params: {
    currentBg: number
    trend?: string
    iob?: number
    cob?: number
    predictions?: { linear?: number[]; lstm?: number[] }
    recentFood?: string
    recentInsulin?: number
    // Enhanced context - all metrics for comprehensive AI advice
    isf?: number
    icr?: number
    pir?: number
    dose?: number
    bgPressure?: number
    tftPredictions?: Array<{ horizon: number; value: number; lower: number; upper: number }>
    recentGI?: number
    absorptionRate?: string
  }): Promise<RealtimeInsight> => {
    const response = await api.post('/insights/realtime', {
      currentBg: params.currentBg,
      trend: params.trend || 'Flat',
      iob: params.iob || 0,
      cob: params.cob || 0,
      predictions: params.predictions,
      recentFood: params.recentFood,
      recentInsulin: params.recentInsulin,
      // Enhanced context for better AI advice
      isf: params.isf,
      icr: params.icr,
      pir: params.pir,
      dose: params.dose,
      bgPressure: params.bgPressure,
      tftPredictions: params.tftPredictions,
      recentGI: params.recentGI,
      absorptionRate: params.absorptionRate
    })
    return response.data
  },

  /**
   * Chat with AI for "what-if" scenario questions.
   * Returns specific, calculated answers based on current diabetes state.
   */
  chat: async (params: {
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
  }): Promise<ChatResponse> => {
    const response = await api.post('/insights/chat', {
      question: params.question,
      currentBg: params.currentBg,
      trend: params.trend,
      iob: params.iob,
      cob: params.cob,
      isf: params.isf,
      icr: params.icr,
      pir: params.pir,
      dose: params.dose,
      bgPressure: params.bgPressure,
      tftPredictions: params.tftPredictions,
    })
    return response.data
  },

  getAll: async (
    category?: string,
    limit: number = 10
  ): Promise<{ insights: Insight[]; totalCount: number; hasMore: boolean }> => {
    const response = await api.get('/insights/', {
      params: { category, limit }
    })
    return response.data
  },

  getPatterns: async (days: number = 14) => {
    const response = await api.get('/insights/patterns', {
      params: { days }
    })
    return response.data
  },

  getMealImpact: async (days: number = 14) => {
    const response = await api.get('/insights/meal-impact', {
      params: { days }
    })
    return response.data
  },

  getAnomalies: async (hours: number = 24): Promise<{
    anomalies: Anomaly[]
    count: number
    periodHours: number
    analyzedReadings: number
  }> => {
    const response = await api.get('/insights/anomalies', {
      params: { hours }
    })
    return response.data
  },

  generate: async (force: boolean = false) => {
    const response = await api.post('/insights/generate', null, {
      params: { force }
    })
    return response.data
  },

  getWeeklySummary: async (): Promise<{
    stats: {
      avgBg: number
      stdBg: number
      timeInRange: number
      totalReadings: number
      lows: number
      highs: number
    }
    comparison: {
      avgBgChange?: number
      tirChange?: number
    }
    summary: {
      summary: string
      highlight: string
      focus: string
      motivation: string
    }
    generatedAt: string
  }> => {
    const response = await api.get('/insights/weekly-summary')
    return response.data
  },

  getCategories: async () => {
    const response = await api.get('/insights/categories')
    return response.data
  },
}

// ==================== Training Types ====================

export interface TrainingEligibility {
  eligible: boolean
  reason: string
  stats: {
    totalReadings: number
    totalTreatments: number
    dataSpanDays: number
    oldestData?: string
    newestData?: string
  }
  requirements: {
    min_readings: number
    min_treatments: number
    min_days: number
  }
}

export interface TrainingStatus {
  hasPersonalizedModel: boolean
  modelStatus: string | null
  modelVersion: number | null
  modelMetrics: {
    mae_30min?: number
    mae_60min?: number
    rmse_30min?: number
    rmse_60min?: number
  } | null
  lastTrainedAt: string | null
  activeJob: TrainingJob | null
  recentJobs: TrainingJob[]
}

export interface TrainingJob {
  id: string
  userId: string
  modelType: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  createdAt: string
  completedAt?: string
  metrics?: Record<string, number>
  error?: string
}

export interface StartTrainingResponse {
  jobId: string
  status: string
  message: string
}

export interface ISFStatus {
  has_learned_isf: boolean
  fasting_isf: number | null
  fasting_confidence: number | null
  fasting_samples: number
  meal_isf: number | null
  meal_confidence: number | null
  meal_samples: number
  default_isf: number
  last_updated: string | null
  time_of_day_pattern: {
    morning?: number | null
    afternoon?: number | null
    evening?: number | null
    night?: number | null
  } | null
  // Short-term ISF fields for detecting illness/resistance changes
  current_isf: number | null
  current_isf_confidence: number | null
  current_isf_samples: number
  isf_deviation: number | null  // Percentage deviation from baseline
  recent_data_points: Array<{
    timestamp: string
    isf: number
    bg_before: number
    bg_after: number
    insulin: number
    confidence: number
    hours_ago: number
  }> | null
}

export interface ISFLearningResult {
  success: boolean
  message: string
  imported?: number
  validated_clean?: number
  rejected?: number
  learned_isf?: number
  confidence?: number
  sample_count: number
  time_of_day_pattern?: Record<string, number | null>
  isf_range?: {
    min: number
    max: number
    mean: number
    std?: number
  }
}

export interface DataStats {
  totalReadings: number
  totalTreatments: number
  totalInsulin: number
  totalCarbs: number
  dataSpanDays: number
  readingsPerDay: number
  treatmentsPerDay: number
  oldestData?: string
  newestData?: string
}

// ICR (Insulin-to-Carb Ratio) Types
export interface ICRStatus {
  has_learned_icr: boolean
  overall_icr: number | null
  overall_confidence: number | null
  overall_samples: number
  breakfast_icr: number | null
  lunch_icr: number | null
  dinner_icr: number | null
  default_icr: number
  last_updated: string | null
  meal_type_pattern: {
    breakfast?: number | null
    lunch?: number | null
    dinner?: number | null
  } | null
  icr_range?: {
    min: number
    max: number
    mean: number
    std?: number
  }
  // Short-term ICR (last 3 days) for detecting temporary changes
  current_icr: number | null
  current_icr_confidence: number | null
  current_icr_samples: number
  icr_deviation: number | null  // Percentage deviation from baseline
}

export interface ICRLearningResult {
  success: boolean
  message: string
  overall_icr?: number
  meal_type_pattern?: Record<string, number | null>
  sample_count: number
  confidence?: number
  icr_range?: {
    min: number
    max: number
    mean: number
    std?: number
  }
}

// PIR (Protein-to-Insulin Ratio) Types
export interface PIRStatus {
  has_learned_pir: boolean
  overall_pir: number | null
  overall_confidence: number | null
  overall_samples: number
  protein_onset_min: number | null
  protein_peak_min: number | null
  protein_duration_min: number | null
  default_pir: number
  last_updated: string | null
  pir_range?: {
    min: number
    max: number
    mean: number
    std?: number
  }
  // Short-term PIR fields for detecting temporary changes (when backend supports it)
  current_pir?: number | null
  current_pir_confidence?: number | null
  current_pir_samples?: number
  pir_deviation?: number | null  // % deviation from baseline
  // Timing info from backend
  avg_onset_minutes?: number | null
  avg_peak_minutes?: number | null
}

export interface PIRLearningResult {
  success: boolean
  message: string
  overall_pir?: number
  protein_timing?: {
    onset_min: number
    peak_min: number
    duration_min: number
  }
  sample_count: number
  confidence?: number
}

export interface PIRTimingInfo {
  onset_min: number
  peak_min: number
  duration_min: number
  recommendation: string
}

// Training/ML endpoints (user ID from JWT token)
export const trainingApi = {
  /**
   * Check if user has enough data for model training.
   */
  checkEligibility: async (modelType: string = 'tft'): Promise<TrainingEligibility> => {
    const response = await api.get('/training/eligibility', {
      params: { model_type: modelType }
    })
    return response.data
  },

  /**
   * Get current training/model status.
   */
  getStatus: async (modelType: string = 'tft'): Promise<TrainingStatus> => {
    const response = await api.get('/training/status', {
      params: { model_type: modelType }
    })
    return response.data
  },

  /**
   * Start personalized model training.
   */
  startTraining: async (modelType: string = 'tft', force: boolean = false): Promise<StartTrainingResponse> => {
    const response = await api.post('/training/start', null, {
      params: { model_type: modelType, force }
    })
    return response.data
  },

  /**
   * Get details of a specific training job.
   */
  getJob: async (jobId: string): Promise<TrainingJob> => {
    const response = await api.get(`/training/jobs/${jobId}`)
    return response.data
  },

  /**
   * List all personalized models for the user.
   */
  listModels: async (): Promise<{ models: Record<string, unknown>[]; count: number }> => {
    const response = await api.get('/training/models')
    return response.data
  },

  /**
   * Delete a personalized model.
   */
  deleteModel: async (modelType: string): Promise<{ message: string }> => {
    const response = await api.delete(`/training/models/${modelType}`)
    return response.data
  },

  /**
   * Get detailed training data statistics.
   */
  getDataStats: async (): Promise<DataStats> => {
    const response = await api.get('/training/data-stats')
    return response.data
  },

  // ==================== ISF Learning ====================

  /**
   * Get current learned ISF status.
   * Supports viewing shared accounts when userId is provided.
   */
  getISFStatus: async (userId?: string): Promise<ISFStatus> => {
    const response = await api.get('/training/isf/status', {
      params: userId ? { user_id: userId } : {}
    })
    return response.data
  },

  /**
   * Learn ISF from recent data (basic learner).
   */
  learnISF: async (days: number = 30): Promise<ISFLearningResult> => {
    const response = await api.post('/training/isf/learn', null, {
      params: { days }
    })
    return response.data
  },

  /**
   * Import and learn ISF from historic bolus_moments.jsonl data.
   * Uses enhanced clean bolus detection.
   */
  importHistoricISF: async (): Promise<ISFLearningResult> => {
    const response = await api.post('/training/isf/import-historic')
    return response.data
  },

  /**
   * Learn ISF using enhanced clean bolus detection.
   * Validates each bolus and detects undocumented carbs.
   */
  learnISFEnhanced: async (days: number = 30, importHistoric: boolean = true): Promise<ISFLearningResult> => {
    const response = await api.post('/training/isf/learn-enhanced', null, {
      params: { days, import_historic: importHistoric }
    })
    return response.data
  },

  /**
   * Get ISF for a specific context (time of day, fasting state).
   */
  getISFForContext: async (timestamp?: string, isFasting: boolean = true): Promise<{
    isf: number
    source: string
    confidence: number
    sample_count: number
    timestamp: string
    is_fasting: boolean
  }> => {
    const response = await api.get('/training/isf/context', {
      params: { timestamp, is_fasting: isFasting }
    })
    return response.data
  },

  /**
   * Reset learned ISF data.
   */
  resetISF: async (): Promise<{ message: string; reset_types: string[] }> => {
    const response = await api.delete('/training/isf/reset')
    return response.data
  },

  // ==================== ICR Learning ====================

  /**
   * Get current learned ICR status.
   * Supports viewing shared accounts when userId is provided.
   */
  getICRStatus: async (userId?: string): Promise<ICRStatus> => {
    const response = await api.get('/training/icr/status', {
      params: userId ? { user_id: userId } : {}
    })
    return response.data
  },

  /**
   * Learn ICR from meal bolus data.
   */
  learnICR: async (days: number = 30): Promise<ICRLearningResult> => {
    const response = await api.post('/training/icr/learn', null, {
      params: { days }
    })
    return response.data
  },

  /**
   * Reset learned ICR data.
   */
  resetICR: async (): Promise<{ message: string }> => {
    const response = await api.delete('/training/icr/reset')
    return response.data
  },

  // ==================== PIR Learning ====================

  /**
   * Get current learned PIR status.
   * Supports viewing shared accounts when userId is provided.
   */
  getPIRStatus: async (userId?: string): Promise<PIRStatus> => {
    const response = await api.get('/training/pir/status', {
      params: userId ? { user_id: userId } : {}
    })
    return response.data
  },

  /**
   * Learn PIR from high-protein meal data.
   */
  learnPIR: async (days: number = 30): Promise<PIRLearningResult> => {
    const response = await api.post('/training/pir/learn', null, {
      params: { days }
    })
    return response.data
  },

  /**
   * Get protein timing details.
   */
  getPIRTiming: async (): Promise<PIRTimingInfo> => {
    const response = await api.get('/training/pir/timing')
    return response.data
  },

  /**
   * Reset learned PIR data.
   */
  resetPIR: async (): Promise<{ message: string }> => {
    const response = await api.delete('/training/pir/reset')
    return response.data
  },

  /**
   * Get current metabolic state with all effective parameters.
   * Includes ISF, ICR, PIR deviations and absorption state.
   * Supports viewing shared accounts when userId is provided.
   */
  getMetabolicState: async (isFasting: boolean = false, mealType?: string, userId?: string): Promise<MetabolicStateResponse> => {
    const params: Record<string, string | boolean> = { is_fasting: isFasting }
    if (mealType) params.meal_type = mealType
    if (userId) params.user_id = userId
    const response = await api.get('/training/metabolic-state', { params })
    return response.data
  },
}

// ==================== Metabolic State Types ====================

export interface MetabolicStateResponse {
  state: 'sick' | 'resistant' | 'normal' | 'sensitive' | 'very_sensitive'
  state_description: string

  // ISF (Insulin Sensitivity Factor)
  isf_value: number
  isf_baseline: number
  isf_deviation_percent: number
  isf_source: string
  is_resistant: boolean
  is_sick: boolean

  // ICR (Insulin-to-Carb Ratio)
  icr_value: number
  icr_baseline: number
  icr_deviation_percent: number
  icr_source: string

  // PIR (Protein-to-Insulin Ratio)
  pir_value: number
  pir_baseline: number

  // Absorption (optional)
  absorption_time_to_peak?: number
  absorption_baseline_time_to_peak?: number
  absorption_deviation_percent?: number
  absorption_state?: 'very_slow' | 'slow' | 'normal' | 'fast' | 'very_fast'
  absorption_is_slow?: boolean

  confidence: number
}

// ==================== Sharing Types ====================

export interface ShareInvitation {
  id: string
  ownerEmail: string
  ownerName?: string
  profileId?: string  // Specific profile being shared (null = all profiles)
  profileName?: string  // Profile display name
  role: string
  permissions: string[]
  expiresAt: string
}

export interface ShareInfo {
  id: string
  ownerId: string
  ownerEmail?: string
  ownerName?: string
  profileId?: string  // Specific profile being shared (null = all profiles)
  profileName?: string  // Profile display name
  sharedWithId: string
  sharedWithEmail: string
  sharedWithName?: string
  role: string
  permissions: string[]
  createdAt: string
  isActive: boolean
}

export interface InviteResponse {
  invitationId: string
  inviteeEmail: string
  expiresAt: string
  message: string
}

export interface AcceptInviteResponse {
  shareId: string
  ownerId: string
  ownerEmail: string
  message: string
}

// Sharing endpoints
export const sharingApi = {
  /**
   * Invite a user by email to view your data.
   * @param profileId - Optional specific profile to share (null = share all profiles)
   * @param profileName - Display name of the profile being shared
   */
  invite: async (
    email: string,
    role: string,
    permissions?: string[],
    profileId?: string,
    profileName?: string
  ): Promise<InviteResponse> => {
    const response = await api.post('/sharing/invite', {
      email,
      role,
      permissions: permissions || [],
      profileId: profileId || null,
      profileName: profileName || null,
    })
    return response.data
  },

  /**
   * Get invitation details by token (no auth required).
   */
  getInvitationByToken: async (token: string): Promise<ShareInvitation> => {
    const response = await api.get(`/sharing/invitation/${token}`)
    return response.data
  },

  /**
   * Accept a sharing invitation using the token.
   */
  acceptInvitation: async (token: string): Promise<AcceptInviteResponse> => {
    const response = await api.post(`/sharing/accept/${token}`)
    return response.data
  },

  /**
   * Get list of users I'm sharing my data with.
   */
  getMyShares: async (): Promise<ShareInfo[]> => {
    const response = await api.get('/sharing/my-shares')
    return response.data
  },

  /**
   * Get list of users who have shared their data with me.
   */
  getSharedWithMe: async (): Promise<ShareInfo[]> => {
    const response = await api.get('/sharing/shared-with-me')
    return response.data
  },

  /**
   * Get pending invitations for the current user.
   */
  getPendingInvitations: async (): Promise<ShareInvitation[]> => {
    const response = await api.get('/sharing/pending-invitations')
    return response.data
  },

  /**
   * Revoke a share (stop sharing with a user).
   */
  revokeShare: async (shareId: string): Promise<{ message: string }> => {
    const response = await api.delete(`/sharing/revoke/${shareId}`)
    return response.data
  },

  /**
   * Update permissions for an existing share.
   */
  updatePermissions: async (shareId: string, permissions: string[]): Promise<ShareInfo> => {
    const response = await api.put(`/sharing/permissions/${shareId}`, { permissions })
    return response.data
  },

  /**
   * Check if current user can view another user's data.
   */
  canViewUser: async (userId: string): Promise<{
    canView: boolean
    isOwner: boolean
    role?: string
    permissions: string[]
  }> => {
    const response = await api.get(`/sharing/can-view/${userId}`)
    return response.data
  },
}

// Data sources endpoints (user ID from JWT token)
export const datasourcesApi = {
  connectGluroo: async (url: string, apiSecret: string) => {
    const response = await api.post('/datasources/gluroo/connect', {
      url,
      apiSecret
    })
    return response.data
  },

  testGluroo: async (url: string, apiSecret: string) => {
    const response = await api.post('/datasources/gluroo/test', {
      url,
      apiSecret
    })
    return response.data
  },

  sync: async (fullSync: boolean = false) => {
    const response = await api.post('/datasources/gluroo/sync', null, {
      params: { full_sync: fullSync }
    })
    return response.data
  },

  disconnect: async () => {
    const response = await api.delete('/datasources/gluroo')
    return response.data
  },

  getStatus: async () => {
    const response = await api.get('/datasources/gluroo/status')
    return response.data
  },

  getDefaults: async (): Promise<{
    url: string
    apiSecret: string
    syncInterval: number
    isOwner: boolean
  }> => {
    const response = await api.get('/datasources/gluroo/defaults')
    return response.data
  },
}

// ==================== Profiles API ====================

export type ProfileRelationship = 'self' | 'child' | 'spouse' | 'parent' | 'other'
export type DiabetesType = 'T1D' | 'T2D' | 'LADA' | 'gestational' | 'other'

export interface ManagedProfile {
  id: string
  accountId: string
  displayName: string
  relationship: ProfileRelationship
  diabetesType: DiabetesType
  dateOfBirth?: string
  diagnosisDate?: string
  isActive: boolean
  createdAt: string
  updatedAt: string
}

export interface ProfileCreateData {
  displayName: string
  relationship: ProfileRelationship
  diabetesType?: DiabetesType
  dateOfBirth?: string
  diagnosisDate?: string
}

export interface ProfileUpdateData {
  displayName?: string
  relationship?: ProfileRelationship
  diabetesType?: DiabetesType
  dateOfBirth?: string
  diagnosisDate?: string
}

export const profilesApi = {
  /**
   * List all managed profiles for the current account.
   */
  list: async (): Promise<{ profiles: ManagedProfile[]; total: number }> => {
    const response = await api.get('/profiles')
    return response.data
  },

  /**
   * Get a specific profile by ID.
   */
  get: async (profileId: string): Promise<ManagedProfile> => {
    const response = await api.get(`/profiles/${profileId}`)
    return response.data
  },

  /**
   * Create a new managed profile.
   */
  create: async (data: ProfileCreateData): Promise<ManagedProfile> => {
    const response = await api.post('/profiles', data)
    return response.data
  },

  /**
   * Update a profile.
   */
  update: async (profileId: string, data: ProfileUpdateData): Promise<ManagedProfile> => {
    const response = await api.put(`/profiles/${profileId}`, data)
    return response.data
  },

  /**
   * Delete a profile (soft delete by default).
   */
  delete: async (profileId: string, hardDelete: boolean = false): Promise<void> => {
    await api.delete(`/profiles/${profileId}`, { params: { hard_delete: hardDelete } })
  },
}

export default api
