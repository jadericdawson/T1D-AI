/**
 * API Client for T1D-AI Backend
 */
import axios, { AxiosInstance, AxiosError } from 'axios'

// API Base URL - will use environment variable in production
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Handle unauthorized - redirect to login
      localStorage.removeItem('auth_token')
      window.location.href = '/'
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

export interface CurrentMetrics {
  iob: number
  cob: number
  isf: number
  recommendedDose: number
  effectiveBg: number
}

export interface PredictionAccuracy {
  linearWins: number
  lstmWins: number
  totalComparisons: number
}

export interface GlucoseCurrentResponse {
  glucose: GlucoseWithPredictions
  metrics: CurrentMetrics
  accuracy: PredictionAccuracy
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
}

export interface TreatmentCreate {
  type: 'insulin' | 'carbs'
  insulin?: number
  carbs?: number
  protein?: number
  fat?: number
  notes?: string
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
}

// Calculations endpoints
export const calculationsApi = {
  getIob: async (userId: string = 'demo_user', hours: number = 6) => {
    const response = await api.get('/calculations/iob', {
      params: { user_id: userId, hours }
    })
    return response.data
  },

  getCob: async (userId: string = 'demo_user', hours: number = 6) => {
    const response = await api.get('/calculations/cob', {
      params: { user_id: userId, hours }
    })
    return response.data
  },

  calculateDose: async (
    userId: string,
    currentBg: number,
    targetBg: number = 100,
    isfOverride?: number
  ): Promise<DoseCalculation> => {
    const response = await api.post('/calculations/dose', {
      current_bg: currentBg,
      target_bg: targetBg,
      isf_override: isfOverride,
      include_cob: true
    }, { params: { user_id: userId } })
    return response.data
  },

  getSummary: async (userId: string = 'demo_user', currentBg: number = 120) => {
    const response = await api.get('/calculations/summary', {
      params: { user_id: userId, current_bg: currentBg }
    })
    return response.data
  },

  getActiveInsulin: async (userId: string = 'demo_user') => {
    const response = await api.get('/calculations/active-insulin', {
      params: { user_id: userId }
    })
    return response.data
  },
}

// Treatments endpoints
export const treatmentsApi = {
  getRecent: async (userId: string = 'demo_user', hours: number = 24): Promise<Treatment[]> => {
    const response = await api.get('/treatments/', {
      params: { user_id: userId, hours }
    })
    return response.data
  },

  create: async (userId: string, treatment: TreatmentCreate): Promise<Treatment> => {
    const response = await api.post('/treatments/', treatment, {
      params: { user_id: userId }
    })
    return response.data
  },

  getIob: async (userId: string = 'demo_user') => {
    const response = await api.get('/treatments/iob', { params: { user_id: userId } })
    return response.data
  },

  getCob: async (userId: string = 'demo_user') => {
    const response = await api.get('/treatments/cob', { params: { user_id: userId } })
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

// Insights endpoints
export const insightsApi = {
  getAll: async (
    userId: string = 'demo_user',
    category?: string,
    limit: number = 10
  ): Promise<{ insights: Insight[]; totalCount: number; hasMore: boolean }> => {
    const response = await api.get('/insights/', {
      params: { user_id: userId, category, limit }
    })
    return response.data
  },

  getPatterns: async (userId: string = 'demo_user', days: number = 14) => {
    const response = await api.get('/insights/patterns', {
      params: { user_id: userId, days }
    })
    return response.data
  },

  getMealImpact: async (userId: string = 'demo_user', days: number = 14) => {
    const response = await api.get('/insights/meal-impact', {
      params: { user_id: userId, days }
    })
    return response.data
  },

  getAnomalies: async (userId: string = 'demo_user', hours: number = 24): Promise<{
    anomalies: Anomaly[]
    count: number
    periodHours: number
    analyzedReadings: number
  }> => {
    const response = await api.get('/insights/anomalies', {
      params: { user_id: userId, hours }
    })
    return response.data
  },

  generate: async (userId: string = 'demo_user', force: boolean = false) => {
    const response = await api.post('/insights/generate', null, {
      params: { user_id: userId, force }
    })
    return response.data
  },

  getWeeklySummary: async (userId: string = 'demo_user'): Promise<{
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
    const response = await api.get('/insights/weekly-summary', {
      params: { user_id: userId }
    })
    return response.data
  },

  getCategories: async () => {
    const response = await api.get('/insights/categories')
    return response.data
  },
}

// Data sources endpoints
export const datasourcesApi = {
  connectGluroo: async (userId: string, url: string, apiSecret: string) => {
    const response = await api.post('/datasources/gluroo', {
      url,
      api_secret: apiSecret
    }, { params: { user_id: userId } })
    return response.data
  },

  testGluroo: async (userId: string) => {
    const response = await api.post('/datasources/gluroo/test', null, {
      params: { user_id: userId }
    })
    return response.data
  },

  sync: async (userId: string) => {
    const response = await api.post('/datasources/sync', null, {
      params: { user_id: userId }
    })
    return response.data
  },

  disconnect: async (userId: string) => {
    const response = await api.delete('/datasources/gluroo', {
      params: { user_id: userId }
    })
    return response.data
  },
}

export default api
