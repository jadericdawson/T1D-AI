/**
 * T1D-AI Dashboard
 * Main glucose monitoring dashboard with real-time updates
 */
import { useState, useEffect, useRef, lazy, Suspense } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Activity, Clock, RefreshCw, Bell, Sparkles,
  Wifi, WifiOff, Loader2, TrendingUp, Layers,
  MoreVertical, Pencil, Trash2
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { cn, formatTime } from '@/lib/utils'
import { formatDateTimeLocal as formatDateTimeLocalTz } from '@/utils/timezone'

// Helper: Convert UTC ISO timestamp to datetime-local format using user's timezone
const formatDateTimeLocal = (isoTimestamp: string): string => {
  return formatDateTimeLocalTz(isoTimestamp)
}

// Helper: Parse datetime-local (local time) to UTC ISO string
const parseDateTimeLocalToISO = (localDateTimeString: string): string => {
  const [datePart, timePart] = localDateTimeString.split('T')
  const [year, month, day] = datePart.split('-').map(Number)
  const [hours, minutes] = timePart.split(':').map(Number)
  const localDate = new Date(year, month - 1, day, hours, minutes, 0, 0)
  return localDate.toISOString()
}

// Components
import { CurrentGlucose } from '@/components/glucose/CurrentGlucose'
// Lazy load Plotly chart to prevent bundle crashes from Node.js dependencies
const PlotlyGlucoseChart = lazy(() => import('@/components/glucose/PlotlyGlucoseChart').then(m => ({ default: m.PlotlyGlucoseChart })))
import { ChartLegend } from '@/components/glucose/ChartLegend'
import { PredictionsCard } from '@/components/glucose/Predictions'
import { IOBCard, COBCard, POBCard, ISFGaugeCard, ICRGaugeCard, PIRGaugeCard, ProteinInsulinCard, WarningCard, RecommendationCard, FoodSuggestion } from '@/components/metrics/MetricCard'
import MetabolicStatePanel from '@/components/metrics/MetabolicStatePanel'
import { LogCarbsModal, LogInsulinModal } from '@/components/treatments/TreatmentModal'
import { UserMenu } from '@/components/layout/UserMenu'
import { ProfileSelector } from '@/components/layout/ProfileSelector'
import { AddProfileWizard } from '@/components/profiles/AddProfileWizard'

// Hooks
import {
  useCurrentGlucose,
  useGlucoseHistory,
  useRecentTreatments,
  useInsights,
  useRealtimeInsight,
  usePredictionAccuracy,
  usePrefetchAllTimeRanges,
  useUpdateTreatment,
  useDeleteTreatment,
  useAIChat,
  useLearnedRatios,
  useGlurooSync,
} from '@/hooks/useGlucose'
import { useGlucoseWebSocket, GlurooSyncEvent } from '@/hooks/useWebSocket'
import { toast } from '@/hooks/useToast'

// Store
import { useGlucoseStore } from '@/stores/glucoseStore'
import { useAuthStore } from '@/stores/authStore'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, staggerChildren: 0.1 } }
}

const staggerItem = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0 }
}

export default function Dashboard() {
  // Local state
  const [timeRange, setTimeRange] = useState<'1hr' | '3hr' | '6hr' | '12hr' | '24hr'>('3hr')
  const [carbsModalOpen, setCarbsModalOpen] = useState(false)
  const [insulinModalOpen, setInsulinModalOpen] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())

  // Visualization toggles for IOB/COB effects and TFT predictions
  const [showIobCobLines, setShowIobCobLines] = useState(true)  // IOB/COB projection lines on secondary Y-axes
  const [showEffectiveBg, setShowEffectiveBg] = useState(true)  // BG Pressure line (IOB↓ + COB↑)

  // Auto-sync toggles (off by default per user request)
  const [autoSyncInsulin, setAutoSyncInsulin] = useState(false)
  const [autoSyncCarbs, setAutoSyncCarbs] = useState(false)

  // Add profile wizard state
  const [addProfileOpen, setAddProfileOpen] = useState(false)

  // Edit/Delete treatment state
  const [editingTreatment, setEditingTreatment] = useState<{
    id: string
    type: string
    value: number | undefined
    notes: string | undefined
    timestamp: string  // ISO string
  } | null>(null)
  const [deletingTreatmentId, setDeletingTreatmentId] = useState<string | null>(null)

  // WebSocket data state - needs to be declared early for cache clearing effect
  const [latestWsData, setLatestWsData] = useState<any>(null)

  // AI Chat state with conversation history (persists for browser session)
  const [chatQuestion, setChatQuestion] = useState('')
  const [chatHistory, setChatHistory] = useState<Array<{
    question: string
    response: string
    prediction?: { bg30min?: number; bg60min?: number; peakBg?: number; timeOfPeak?: string }
    recommendation?: { insulin?: number; timing?: string; prebolus?: string }
    calculation?: string
    confidence: string
    timestamp: string
  }>>([])
  const aiChat = useAIChat()

  // Update current time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  // Scroll to top on mount to prevent auto-scroll to bottom on small screens
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  // Auth store - get effective userId (for profile-based access)
  // IMPORTANT: Select viewingUserId directly so Zustand detects changes when profile switches
  const viewingUserId = useAuthStore(state => state.viewingUserId)
  const user = useAuthStore(state => state.user)
  const activeProfileId = useAuthStore(state => state.activeProfileId)
  const managedProfiles = useAuthStore(state => state.managedProfiles)
  const userTimezone = useAuthStore(state => state.timezone)

  // Compute effective userId - MUST match getEffectiveProfileId logic for managed profiles
  // Priority: viewingUserId (shared) > activeProfileId (managed child) > user?.id (self)
  const userId = viewingUserId || activeProfileId || user?.id || ''

  // Compute active profile (same logic as getActiveProfile)
  const activeProfile = managedProfiles.find((p: { id: string }) => p.id === activeProfileId) || null

  // NOTE: For legacy profiles (profile_{userId}), data exists without formal data sources
  // We allow dashboard access regardless of data source configuration - empty states handle missing data
  void activeProfile?.id?.startsWith('profile_')

  // Query client for cache management
  const queryClient = useQueryClient()
  const prevUserIdRef = useRef<string | null>(null)

  // Track if profile switch needs full refresh (set by cache clearing effect)
  const needsRefreshRef = useRef(false)
  // Track if we're in the middle of a profile switch - ignore WebSocket messages during this time
  const isProfileSwitchingRef = useRef(false)

  // Clear cache when userId changes to prevent data leaks between profiles
  useEffect(() => {
    if (prevUserIdRef.current && prevUserIdRef.current !== userId) {
      // User switched profiles - clear all cached data and force refetch
      console.log(`[Dashboard] Profile switching from ${prevUserIdRef.current} to ${userId}`)

      // IMMEDIATELY set profile switching flag to ignore incoming WebSocket messages
      isProfileSwitchingRef.current = true

      // Clear WebSocket state first to prevent stale data
      setLatestWsData(null)

      // Remove all queries for the OLD userId to prevent data leaks
      queryClient.removeQueries({ queryKey: ['glucose'] })
      queryClient.removeQueries({ queryKey: ['treatments'] })
      queryClient.removeQueries({ queryKey: ['predictions'] })
      queryClient.removeQueries({ queryKey: ['insights'] })
      queryClient.removeQueries({ queryKey: ['training'] })
      queryClient.removeQueries({ queryKey: ['calculations'] })

      // Mark that we need to force refresh the WebSocket connection
      needsRefreshRef.current = true

      console.log(`[Dashboard] Cache cleared for profile switch to ${userId}`)
    }
    prevUserIdRef.current = userId
  }, [userId, queryClient])

  // Prefetch all time ranges on mount for instant switching
  const { prefetchAll } = usePrefetchAllTimeRanges(userId)

  // Trigger prefetch when userId is available
  useEffect(() => {
    if (userId) {
      prefetchAll()
    }
  }, [userId, prefetchAll])

  // Zustand store
  const { preferences, updatePreferences } = useGlucoseStore()

  // React Query hooks - glucose endpoints need userId for parent/child access
  const {
    data: currentData,
    isLoading: isLoadingCurrent,
    refetch: refetchCurrent
  } = useCurrentGlucose(userId)

  // Always fetch 24hr of data - tabs just control the visible zoom window
  const {
    data: historyData,
    isLoading: isLoadingHistory
  } = useGlucoseHistory(userId, 24)

  // Treatments need userId for shared access
  const { data: treatmentsData } = useRecentTreatments(24, userId)
  const { data: insightsData } = useInsights(5)
  const { data: accuracyData } = usePredictionAccuracy()

  // Learned ratios (ISF, ICR, PIR) from training - pass userId for shared access
  const learnedRatios = useLearnedRatios(userId)

  // Treatment mutations
  const { mutate: updateTreatment, isPending: isUpdating } = useUpdateTreatment()
  const { mutate: deleteTreatment, isPending: isDeleting } = useDeleteTreatment()

  // Gluroo sync mutation
  const { mutate: syncGluroo, isPending: isSyncing } = useGlurooSync()

  // Handle Gluroo sync notifications
  const handleGlurooSync = (event: GlurooSyncEvent) => {
    if (event.status === 'success') {
      // Build detailed message with enrichment data
      let details = event.message
      if (event.notes) details += ` - ${event.notes}`
      if (event.protein) details += ` (P:${event.protein}g`
      if (event.fat) details += event.protein ? `, F:${event.fat}g)` : ` (F:${event.fat}g)`
      else if (event.protein) details += ')'

      toast({
        title: '✓ Synced to Gluroo',
        description: details,
        variant: 'success',
      })
    } else {
      toast({
        title: '✗ Gluroo Sync Failed',
        description: event.message,
        variant: 'destructive',
      })
    }
  }

  // WebSocket for real-time updates with Gluroo sync notifications
  const { isConnected, connectionStatus, forceRefreshAllData, forceReconnect } = useGlucoseWebSocket({
    userId,
    interval: 60,
    onGlurooSync: handleGlurooSync,
    onMessage: (data) => {
      // IMPORTANT: Ignore WebSocket messages during profile switch to prevent data leaks
      if (isProfileSwitchingRef.current) {
        console.log('[Dashboard] Ignoring WebSocket message during profile switch')
        return
      }
      if (data.type === 'glucose_update' && data.data) {
        setLatestWsData(data.data)
      }
    },
  })
  const latestData = latestWsData

  // Force reconnect WebSocket after profile switch to ensure fresh connection with new userId
  useEffect(() => {
    if (needsRefreshRef.current && userId) {
      console.log(`[Dashboard] Forcing WebSocket reconnect for new userId: ${userId}`)
      needsRefreshRef.current = false
      // Small delay to ensure cache clearing is complete
      const timer = setTimeout(() => {
        forceReconnect()
        // Clear profile switching flag after WebSocket has time to reconnect
        // This allows the new connection's messages to be processed
        setTimeout(() => {
          isProfileSwitchingRef.current = false
          console.log(`[Dashboard] Profile switch complete, accepting WebSocket messages for ${userId}`)
        }, 500)
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [userId, forceReconnect])

  // Handle force refresh - reconnects WebSocket and refetches all data
  const handleForceRefresh = async () => {
    // Clear local WebSocket state to force UI update
    setLatestWsData(null)

    toast({
      title: 'Refreshing...',
      description: 'Reconnecting and fetching latest data',
    })

    await forceRefreshAllData()

    toast({
      title: 'Refreshed',
      description: 'Data updated successfully',
      variant: 'success',
    })
  }

  // Update preferences when time range changes
  useEffect(() => {
    if (timeRange !== preferences.chartTimeRange) {
      updatePreferences({ chartTimeRange: timeRange })
    }
  }, [timeRange, preferences.chartTimeRange, updatePreferences])

  // Extract current glucose data - use FRESHER of WebSocket or API data
  // Compare timestamps to pick the most recent reading
  const wsTimestamp = latestData?.timestamp ? new Date(latestData.timestamp).getTime() : 0
  const apiTimestamp = currentData?.glucose?.timestamp ? new Date(currentData.glucose.timestamp).getTime() : 0
  const useWebSocketData = wsTimestamp > apiTimestamp && latestData?.value !== undefined

  const currentBg = useWebSocketData ? latestData?.value : (currentData?.glucose?.value ?? latestData?.value)
  const currentTrend = useWebSocketData ? latestData?.trend : (currentData?.glucose?.trend ?? latestData?.trend)
  const currentTimestamp = useWebSocketData ? latestData?.timestamp : (currentData?.glucose?.timestamp ?? latestData?.timestamp)
  const currentSource = useWebSocketData ? latestData?.source : (currentData?.glucose?.source ?? latestData?.source)

  // Extract predictions
  const linearPredictions = latestData?.predictions?.linear ?? currentData?.glucose?.predictions?.linear ?? []
  const lstmPredictions = latestData?.predictions?.lstm ?? currentData?.glucose?.predictions?.lstm ?? null
  const modelAvailable = latestData?.modelAvailable ?? (lstmPredictions !== null && lstmPredictions.length > 0)

  // Extract metrics
  const iob = latestData?.metrics?.iob ?? currentData?.metrics?.iob ?? 0
  const cob = latestData?.metrics?.cob ?? currentData?.metrics?.cob ?? 0
  const pob = latestData?.metrics?.pob ?? currentData?.metrics?.pob ?? 0
  const isf = latestData?.metrics?.isf ?? currentData?.metrics?.isf ?? 50
  const recommendedDose = latestData?.metrics?.recommendedDose ?? currentData?.metrics?.recommendedDose ?? 0

  // Food recommendation data (when BG predicted low)
  const actionType = (latestData?.metrics?.actionType ?? currentData?.metrics?.actionType ?? 'none') as 'insulin' | 'food' | 'none'
  const recommendedCarbs = latestData?.metrics?.recommendedCarbs ?? currentData?.metrics?.recommendedCarbs ?? 0
  const foodSuggestions: FoodSuggestion[] = (latestData?.metrics?.foodSuggestions ?? currentData?.metrics?.foodSuggestions ?? []) as FoodSuggestion[]
  const predictedBgWithoutAction = latestData?.metrics?.predictedBgWithoutAction ?? currentData?.metrics?.predictedBgWithoutAction ?? 0
  const predictedBgWithAction = latestData?.metrics?.predictedBgWithAction ?? currentData?.metrics?.predictedBgWithAction ?? 0
  const recommendationReasoning = latestData?.metrics?.recommendationReasoning ?? currentData?.metrics?.recommendationReasoning ?? ''

  // Protein insulin with time-based decay (LATER → NOW as time passes)
  // Backend calculates this with decay factor based on time since protein meal
  // Also clears protein NOW if no correction is needed (to prevent lows)
  const proteinInsulinNow = latestData?.metrics?.proteinDoseNow ?? currentData?.metrics?.proteinDoseNow ?? 0
  const proteinInsulinLater = latestData?.metrics?.proteinDoseLater ?? currentData?.metrics?.proteinDoseLater ?? 0

  // Accuracy stats
  const accuracy = accuracyData ?? currentData?.accuracy ?? null

  // TFT predictions and effect curves from API (new ML features)
  const tftPredictions = currentData?.tftPredictions ?? []
  const effectCurve = currentData?.effectCurve ?? []
  const historicalIobCob = currentData?.historicalIobCob ?? []

  // Calculate BG Pressure from effect curve (where BG is heading based on IOB/COB)
  const bgPressure = effectCurve.length > 0 ? effectCurve[0]?.netEffect ?? 0 : 0

  // Format TFT predictions for AI context
  const formattedTft = tftPredictions.map((p: { horizon: number; value: number; lower: number; upper: number }) => ({
    horizon: p.horizon,
    value: p.value,
    lower: p.lower,
    upper: p.upper,
  }))

  // Track if we have real glucose data - used to show dashes in gauges when no data
  const hasGlucoseData = currentBg !== undefined && currentBg !== null
  // Loading flag for metric cards - show dashes when loading OR no data
  const metricsLoading = (isLoadingCurrent && !latestData?.metrics) || !hasGlucoseData

  // Real-time AI insight - updates when BG/IOB/COB/ISF/ICR/PIR/Dose/BgPressure changes
  const { data: realtimeInsight, isLoading: isLoadingInsight } = useRealtimeInsight({
    currentBg,
    trend: currentTrend ?? undefined,
    iob,
    cob,
    enabled: !!currentBg && currentBg > 0,
    // All diabetes metrics for comprehensive AI advice
    isf,
    icr: learnedRatios.icr.value,
    pir: learnedRatios.pir.value,
    dose: recommendedDose,
    bgPressure,
    tftPredictions: formattedTft.length > 0 ? formattedTft : undefined,
  })

  // Format treatments for display (include all enrichment data)
  // Deduplicate: when a treatment is edited, prefer the one with more complete notes
  const safeTreatmentsData = Array.isArray(treatmentsData) ? treatmentsData : []
  const deduplicatedTreatments = safeTreatmentsData.reduce((acc, t) => {
    // Create a key based on timestamp (to minute) + normalized type + value
    const timeKey = new Date(t.timestamp).toISOString().slice(0, 16) // YYYY-MM-DDTHH:MM
    const isInsulinType = t.type === 'insulin' || t.type === 'Correction Bolus'
      || t.type === 'auto_correction' || t.type === 'basal'
    const normalType = isInsulinType ? 'insulin' : 'carbs'
    const value = isInsulinType ? t.insulin : t.carbs
    const key = `${timeKey}_${normalType}_${value}`

    // If we already have this treatment, keep the one with longer notes (edited version)
    const existing = acc.get(key)
    if (!existing || (t.notes?.length ?? 0) > (existing.notes?.length ?? 0)) {
      acc.set(key, t)
    }
    return acc
  }, new Map())

  const recentTreatments = Array.from(deduplicatedTreatments.values())
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
    .map((t) => {
      // Normalize type: 'Correction Bolus', 'auto_correction', 'basal' → 'insulin'
      //                 'Carb Correction' → 'carbs'
      const isInsulin = t.type === 'insulin' || t.type === 'Correction Bolus'
        || t.type === 'auto_correction' || t.type === 'basal'
      const normalizedType = isInsulin ? 'insulin' : 'carbs'
      return {
        id: t.id,
        type: normalizedType,
        value: isInsulin ? t.insulin : t.carbs,
        time: formatTime(t.timestamp, userTimezone),
        timestamp: t.timestamp,
        notes: t.notes,
        // Full macro data from Gluroo
        protein: t.protein,
        fat: t.fat,
        // AI enrichment data
        glycemicIndex: t.glycemicIndex,
        glycemicLoad: t.glycemicLoad,
        absorptionRate: t.absorptionRate,
        fatContent: t.fatContent,
        isLiquid: t.isLiquid,
        enrichedAt: t.enrichedAt,
        source: t.source,
        // Pump fields
        bolusType: (t as any).bolusType,
        deliveryMethod: (t as any).deliveryMethod,
        basalRate: (t as any).basalRate,
      }
    })

  // Format history for chart
  const chartReadings = (historyData?.readings ?? []).map((r) => ({
    timestamp: r.timestamp,
    value: r.value,
    trend: r.trend
  }))

  // Generate chart predictions
  const chartPredictions = linearPredictions.map((linear: number, i: number) => ({
    timestamp: new Date(Date.now() + (i + 1) * 5 * 60 * 1000).toISOString(),
    linear,
    lstm: lstmPredictions?.[i]
  }))

  // Format treatments for chart - handle all treatment types
  // Include notes and isLiquid for food emoji selection
  const chartTreatments = safeTreatmentsData.map((t) => {
    // Normalize type: 'insulin', 'Correction Bolus', 'auto_correction', 'basal' → 'insulin'
    //                 'carbs' and 'Carb Correction' → 'carbs'
    const isInsulin = t.type === 'insulin' || t.type === 'Correction Bolus'
      || t.type === 'auto_correction' || t.type === 'basal'
    const normalizedType = isInsulin ? 'insulin' : 'carbs'
    return {
      timestamp: t.timestamp,
      type: normalizedType as 'insulin' | 'carbs',
      value: isInsulin ? (t.insulin ?? 0) : (t.carbs ?? 0),
      notes: t.notes,
      isLiquid: t.isLiquid
    }
  })

  // Handle refresh - syncs from Gluroo first, then refetches local data
  const handleRefresh = () => {
    // Sync from Gluroo to get latest data
    syncGluroo(false)  // false = incremental sync
    // Also refetch current data
    refetchCurrent()
  }

  // Loading state - show while auth is loading (no userId) or query is loading
  if (!userId || (isLoadingCurrent && !latestData)) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-cyan mx-auto mb-4" />
          <p className="text-gray-400">{!userId ? 'Loading user...' : 'Loading glucose data...'}</p>
        </div>
      </div>
    )
  }


  return (
    <div className="min-h-screen p-6 pb-24">
      {/* Header */}
      <motion.header
        initial="hidden"
        animate="visible"
        variants={fadeIn}
        className="flex items-center justify-between mb-8"
      >
        <div className="flex items-center gap-4">
          <Link to="/" className="flex items-center hover:opacity-80 transition">
            <img src="/logo.svg" alt="T1D-AI" className="h-10" />
          </Link>

          {/* Profile Selector - switch between managed profiles and shared users */}
          <ProfileSelector />

          {/* Connection Status - Clickable to force refresh when not live */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge
                  variant="outline"
                  className={cn(
                    'flex items-center gap-1.5 cursor-pointer transition-colors',
                    isConnected
                      ? 'border-cyan/30 text-cyan hover:bg-cyan/10'
                      : currentBg !== undefined
                        ? 'border-yellow-500/30 text-yellow-500 hover:bg-yellow-500/10'
                        : 'border-red-500/30 text-red-500 hover:bg-red-500/10'
                  )}
                  onClick={handleForceRefresh}
                >
                  {isConnected ? (
                    <>
                      <Wifi className="w-3 h-3" />
                      Live
                    </>
                  ) : connectionStatus === 'connecting' ? (
                    <>
                      <RefreshCw className="w-3 h-3 animate-spin" />
                      Connecting...
                    </>
                  ) : currentBg !== undefined ? (
                    <>
                      <WifiOff className="w-3 h-3" />
                      Reconnecting...
                    </>
                  ) : (
                    <>
                      <WifiOff className="w-3 h-3" />
                      Offline
                    </>
                  )}
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <p>{isConnected ? 'Connected - Click to force refresh' : 'Click to reconnect and refresh all data'}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        {/* Current Time (EST) */}
        <div className="flex items-center gap-2 text-gray-300">
          <Clock className="w-4 h-4 text-cyan" />
          <span className="font-mono text-lg">
            {currentTime.toLocaleTimeString('en-US', {
              timeZone: userTimezone,
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
              hour12: true
            })}
          </span>
          <span className="text-xs text-gray-500">
            {new Intl.DateTimeFormat('en-US', { timeZone: userTimezone, timeZoneName: 'short' }).formatToParts(currentTime).find(p => p.type === 'timeZoneName')?.value || ''}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-gray-400 hover:text-white"
                  onClick={handleRefresh}
                  disabled={isSyncing}
                >
                  <RefreshCw className={cn('w-5 h-5', (isLoadingCurrent || isSyncing) && 'animate-spin')} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Sync from Gluroo</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white relative">
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-orange-500" />
          </Button>
          <UserMenu />
        </div>
      </motion.header>

      {/* Main Grid */}
      <motion.div
        initial="hidden"
        animate="visible"
        variants={fadeIn}
        className="grid grid-cols-1 lg:grid-cols-12 gap-6"
      >
        {/* Left Column - Glucose & Metrics */}
        <div className="lg:col-span-4 space-y-6">
          {/* Current Glucose */}
          <motion.div variants={staggerItem}>
            {currentBg !== undefined && currentTimestamp ? (
              <CurrentGlucose
                value={currentBg}
                trend={currentTrend ?? null}
                timestamp={currentTimestamp}
                source={currentSource}
              />
            ) : (
              <div className="glass-card text-center py-8">
                <Activity className="w-10 h-10 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-400 text-lg font-medium">No glucose data yet</p>
                <p className="text-gray-500 text-sm mt-1">
                  <Link to="/settings" className="text-cyan hover:underline">Connect a data source</Link> to see your readings
                </p>
              </div>
            )}
          </motion.div>

          {/* Metrics Grid - IOB, COB, POB, ISF */}
          <motion.div variants={staggerItem} className="grid grid-cols-2 gap-4">
            <IOBCard value={iob} isLoading={metricsLoading} />
            <COBCard value={cob} isLoading={metricsLoading} />
            <POBCard value={pob} isLoading={metricsLoading} />
            <ISFGaugeCard
              baselineIsf={learnedRatios.isf.value}
              currentIsf={learnedRatios.isf.currentIsf}
              deviation={learnedRatios.isf.deviation}
              source={learnedRatios.isf.source as 'learned' | 'default'}
              isLoading={metricsLoading || learnedRatios.isLoading}
              confidence={learnedRatios.isf.currentIsfConfidence ?? learnedRatios.isf.confidence}
            />
          </motion.div>

          {/* Action Card - Shows Dose OR Food recommendation */}
          <motion.div variants={staggerItem} className="grid grid-cols-2 gap-4">
            <RecommendationCard
              actionType={actionType}
              recommendedDose={recommendedDose}
              recommendedCarbs={recommendedCarbs}
              foodSuggestions={foodSuggestions}
              predictedBgWithoutAction={predictedBgWithoutAction}
              predictedBgWithAction={predictedBgWithAction}
              reasoning={recommendationReasoning}
              proteinDoseNow={proteinInsulinNow}
              proteinDoseLater={proteinInsulinLater}
              isLoading={metricsLoading}
            />
            <ProteinInsulinCard
              now={proteinInsulinNow}
              later={proteinInsulinLater}
              pob={pob}
              isLoading={metricsLoading}
            />
          </motion.div>

          {/* Learned Ratios - ICR with gauge and PIR */}
          <motion.div variants={staggerItem} className="grid grid-cols-2 gap-4">
            <ICRGaugeCard
              baselineIcr={learnedRatios.icr.value}
              currentIcr={learnedRatios.icr.currentIcr}
              deviation={learnedRatios.icr.deviation}
              source={learnedRatios.icr.source as 'learned' | 'default'}
              confidence={learnedRatios.icr.currentIcrConfidence ?? learnedRatios.icr.confidence}
              isLoading={metricsLoading || learnedRatios.isLoading}
            />
            <PIRGaugeCard
              baselinePir={learnedRatios.pir.value}
              currentPir={learnedRatios.pir.currentPir}
              deviation={learnedRatios.pir.deviation}
              source={learnedRatios.pir.source as 'learned' | 'default'}
              confidence={learnedRatios.pir.currentPirConfidence ?? learnedRatios.pir.confidence}
              onsetMin={learnedRatios.pir.onsetMin ?? undefined}
              peakMin={learnedRatios.pir.peakMin ?? undefined}
              isLoading={metricsLoading || learnedRatios.isLoading}
            />
          </motion.div>

          {/* Metabolic State Panel - 4 gauges showing overall metabolic status */}
          <motion.div variants={staggerItem}>
            <MetabolicStatePanel userId={userId} />
          </motion.div>

          {/* Warnings */}
          {currentBg !== undefined && currentBg < 70 && (
            <motion.div variants={staggerItem}>
              <WarningCard
                type="Low Glucose"
                message={`Current BG is ${currentBg} mg/dL. Consider treating with fast-acting carbs.`}
                severity={currentBg < 54 ? 'critical' : 'warning'}
              />
            </motion.div>
          )}

          {iob > 4 && (
            <motion.div variants={staggerItem}>
              <WarningCard
                type="High IOB"
                message={`${iob.toFixed(1)}U of insulin on board. Be cautious with additional doses.`}
                severity="warning"
              />
            </motion.div>
          )}
        </div>

        {/* Middle Column - Chart */}
        <motion.div variants={staggerItem} className="lg:col-span-5">
          <div className="glass-card h-full">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Glucose Trend</h3>
              <Tabs value={timeRange} onValueChange={(v) => setTimeRange(v as any)}>
                <TabsList className="bg-slate-800/50">
                  {(['1hr', '3hr', '6hr', '12hr', '24hr'] as const).map((range) => (
                    <TabsTrigger
                      key={range}
                      value={range}
                      className="data-[state=active]:bg-cyan/20 data-[state=active]:text-cyan"
                    >
                      {range}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>
            </div>

            {/* Glucose Chart */}
            {isLoadingHistory && chartReadings.length === 0 ? (
              <div className="h-80 flex items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-cyan" />
              </div>
            ) : chartReadings.length === 0 ? (
              <div className="h-80 flex items-center justify-center border border-dashed border-gray-700 rounded-lg">
                <div className="text-center">
                  <TrendingUp className="w-10 h-10 text-gray-600 mx-auto mb-3" />
                  <p className="text-gray-400">No glucose history</p>
                  <p className="text-gray-500 text-sm mt-1">Data will appear here once synced</p>
                </div>
              </div>
            ) : (
              <Suspense fallback={
                <div className="h-80 flex items-center justify-center">
                  <Loader2 className="w-8 h-8 animate-spin text-cyan" />
                </div>
              }>
                <PlotlyGlucoseChart
                  readings={chartReadings}
                  predictions={chartPredictions}
                  treatments={chartTreatments}
                  timeRange={timeRange}
                  targetLow={preferences.lowThreshold}
                  targetHigh={preferences.highThreshold}
                  criticalLow={preferences.criticalLowThreshold}
                  criticalHigh={preferences.criticalHighThreshold}
                  showPredictions={true}
                  showTreatments={true}
                  showTargetRange={true}
                  // ISF from API metrics for BG pressure calculation
                  isf={isf}
                  // New ML visualization props
                  tftPredictions={tftPredictions}
                  effectCurve={effectCurve}
                  historicalIobCob={historicalIobCob}
                  showEffectCurve={false}
                  showEffectAreas={false}
                  showEffectiveBg={showEffectiveBg}
                  showIobCobLines={showIobCobLines}
                />
              </Suspense>
            )}

            {/* Legend */}
            <div className="mt-4">
              <ChartLegend
                showTftPredictions={tftPredictions.length > 0}
                showEffectiveBg={showEffectiveBg}
                showEffectAreas={false}
                showHistoricalIobCob={showIobCobLines && historicalIobCob.length > 0}
              />
            </div>

            {/* Visualization Toggles */}
            <div className="mt-4 pt-4 border-t border-slate-700">
              <div className="flex items-center gap-2 mb-3">
                <Layers className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-400">Visualization Options</span>
              </div>
              <TooltipProvider>
                <div className="flex flex-wrap gap-4">
                  {/* IOB/COB/POB Effect Lines */}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-2">
                        <Switch
                          id="iobCobLines"
                          checked={showIobCobLines}
                          onCheckedChange={setShowIobCobLines}
                          className="data-[state=checked]:bg-cyan"
                        />
                        <Label htmlFor="iobCobLines" className="text-xs text-gray-300 cursor-pointer">
                          IOB/COB/POB
                        </Label>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Show IOB (blue), COB (green), and POB (orange) curves</p>
                    </TooltipContent>
                  </Tooltip>

                  {/* BG Pressure Line */}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-2">
                        <Switch
                          id="effectiveBg"
                          checked={showEffectiveBg}
                          onCheckedChange={setShowEffectiveBg}
                          className="data-[state=checked]:bg-cyan"
                        />
                        <Label htmlFor="effectiveBg" className="text-xs text-gray-300 cursor-pointer">
                          BG Pressure
                        </Label>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Net effect of IOB (↓), COB (↑), and POB (↑) on BG</p>
                    </TooltipContent>
                  </Tooltip>

                  {/* Auto-sync toggles separator */}
                  <div className="h-4 border-l border-slate-600 mx-2" />

                  {/* Auto Insulin Sync */}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-2">
                        <Switch
                          id="autoSyncInsulin"
                          checked={autoSyncInsulin}
                          onCheckedChange={setAutoSyncInsulin}
                          className="data-[state=checked]:bg-blue-500"
                        />
                        <Label htmlFor="autoSyncInsulin" className="text-xs text-gray-300 cursor-pointer">
                          Auto Insulin
                        </Label>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Automatically infer insulin from BG data (disabled)</p>
                    </TooltipContent>
                  </Tooltip>

                  {/* Auto Carbs Sync */}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-2">
                        <Switch
                          id="autoSyncCarbs"
                          checked={autoSyncCarbs}
                          onCheckedChange={setAutoSyncCarbs}
                          className="data-[state=checked]:bg-green-500"
                        />
                        <Label htmlFor="autoSyncCarbs" className="text-xs text-gray-300 cursor-pointer">
                          Auto Carbs
                        </Label>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Automatically infer carbs from BG data (disabled)</p>
                    </TooltipContent>
                  </Tooltip>

                  {/* TFT indicator with average delta */}
                  {tftPredictions.length > 0 && (() => {
                    // Get average TFT delta from predictions that have it
                    const deltas = tftPredictions
                      .filter(p => p.tftDelta !== undefined && p.tftDelta !== null)
                      .map(p => p.tftDelta as number)
                    const avgDelta = deltas.length > 0
                      ? deltas.reduce((a, b) => a + b, 0) / deltas.length
                      : null

                    return (
                      <div className="flex items-center gap-2 ml-auto" title={avgDelta !== null ? `Average TFT adjustment across all horizons: ${avgDelta >= 0 ? '+' : ''}${avgDelta.toFixed(1)} mg/dL` : 'TFT model active'}>
                        <TrendingUp className="w-4 h-4 text-purple-400" />
                        <span className="text-xs text-purple-400">
                          TFT Active
                          {avgDelta !== null && (
                            <span className={avgDelta >= 0 ? 'text-yellow-400' : 'text-green-400'}>
                              {' '}(avg: {avgDelta >= 0 ? '+' : ''}{avgDelta.toFixed(1)})
                            </span>
                          )}
                        </span>
                      </div>
                    )
                  })()}
                </div>
              </TooltipProvider>
            </div>

            {/* Predicted BG */}
            <div className="mt-4 pt-4 border-t border-slate-700">
              <PredictionsCard
                linear={linearPredictions}
                lstm={lstmPredictions}
                tft={tftPredictions}
                effectCurve={effectCurve}
                accuracy={accuracy ? {
                  linearWins: accuracy.linearWins,
                  lstmWins: accuracy.lstmWins,
                  totalComparisons: accuracy.totalComparisons
                } : undefined}
                modelAvailable={modelAvailable}
              />
            </div>
          </div>
        </motion.div>

        {/* Right Column - Treatments & Insights */}
        <div className="lg:col-span-3 space-y-6">
          {/* Activity Log */}
          <motion.div variants={staggerItem} className="glass-card">
            <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan" />
              Activity Log
            </h3>

            {/* Today's Totals Summary - Only data from midnight today */}
            {recentTreatments.length > 0 && (() => {
              // Get midnight today in local timezone
              const today = new Date()
              today.setHours(0, 0, 0, 0)

              // Filter to only today's treatments
              const todaysTreatments = recentTreatments.filter(t => {
                const treatmentDate = new Date(t.timestamp)
                return treatmentDate >= today
              })

              const totals = todaysTreatments.reduce((acc, t) => {
                if (t.type === 'carbs') {
                  acc.carbs += t.value || 0
                  acc.protein += t.protein || 0
                  acc.fat += t.fat || 0
                } else if (t.type === 'insulin') {
                  acc.insulin += t.value || 0
                }
                return acc
              }, { carbs: 0, insulin: 0, protein: 0, fat: 0 })

              return (
                <div className="mb-4 p-3 rounded-lg bg-slate-800/30 border border-slate-700/50">
                  <div className="text-xs text-gray-400 mb-2 text-center">Today's Totals</div>
                  <div className="grid grid-cols-4 gap-2">
                    <div className="text-center">
                      <div className="text-lg font-bold text-teal-400">{Math.round(totals.carbs)}</div>
                      <div className="text-xs text-gray-500">Carbs (g)</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-blue-400">{totals.insulin.toFixed(1)}</div>
                      <div className="text-xs text-gray-500">Insulin (U)</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-orange-400">{Math.round(totals.protein)}</div>
                      <div className="text-xs text-gray-500">Protein (g)</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-amber-400">{Math.round(totals.fat)}</div>
                      <div className="text-xs text-gray-500">Fat (g)</div>
                    </div>
                  </div>
                </div>
              )
            })()}

            {recentTreatments.length > 0 ? (
              <div className="space-y-3 max-h-[400px] overflow-y-auto pr-1">
                {recentTreatments.map((treatment, i) => (
                  <div
                    key={treatment.id || i}
                    className="p-3 rounded-lg bg-slate-800/50 group hover:bg-slate-700/50 transition-colors"
                  >
                    {/* Header row */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {treatment.type === 'carbs' ? (
                          <span className="text-2xl">{treatment.isLiquid ? '🥤' : '🍎'}</span>
                        ) : (treatment as any).deliveryMethod === 'pump_auto_correction' ? (
                          <span className="text-2xl" title="Auto correction (Control-IQ)">🤖</span>
                        ) : (treatment as any).deliveryMethod === 'pump_basal' ? (
                          <span className="text-2xl" title="Basal delivery">⏳</span>
                        ) : (treatment as any).deliveryMethod === 'pump_bolus' ? (
                          <span className="text-2xl" title="Pump bolus">💊</span>
                        ) : (
                          <span className="text-2xl">💉</span>
                        )}
                        <div className="flex flex-col flex-1 min-w-0">
                          <span className="text-gray-300 font-medium">
                            {treatment.value}{treatment.type === 'carbs' ? 'g' : 'U'}
                          </span>
                          {treatment.notes && (
                            <span className="text-gray-400 text-xs break-words whitespace-normal">
                              {treatment.notes}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-gray-500 text-sm">{treatment.time}</span>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              <MoreVertical className="h-4 w-4 text-gray-400" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="bg-slate-800 border-gray-700">
                            <DropdownMenuItem
                              onClick={() => setEditingTreatment({
                                id: treatment.id,
                                type: treatment.type,
                                value: treatment.value,
                                notes: treatment.notes,
                                timestamp: treatment.timestamp
                              })}
                              className="text-gray-300 hover:text-white hover:bg-slate-700 cursor-pointer"
                            >
                              <Pencil className="h-4 w-4 mr-2" />
                              Edit
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={() => setDeletingTreatmentId(treatment.id)}
                              className="text-red-400 hover:text-red-300 hover:bg-red-900/30 cursor-pointer"
                            >
                              <Trash2 className="h-4 w-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>

                    {/* Detail row - show macros and enrichment data for carbs */}
                    {treatment.type === 'carbs' && (
                      <div className="mt-2 ml-11 flex flex-wrap gap-x-3 gap-y-1 text-xs">
                        {/* Macros from Gluroo */}
                        {(treatment.protein !== undefined || treatment.fat !== undefined) && (
                          <div className="flex gap-2 text-gray-400">
                            {treatment.protein !== undefined && treatment.protein > 0 && (
                              <span>P: {treatment.protein}g</span>
                            )}
                            {treatment.fat !== undefined && treatment.fat > 0 && (
                              <span>F: {treatment.fat}g</span>
                            )}
                          </div>
                        )}
                        {/* GI enrichment data */}
                        {treatment.glycemicIndex != null && treatment.glycemicIndex > 0 && (
                          <span className={cn(
                            "px-1.5 py-0.5 rounded",
                            treatment.glycemicIndex >= 70 ? "bg-red-900/30 text-red-400" :
                            treatment.glycemicIndex >= 55 ? "bg-yellow-900/30 text-yellow-400" :
                            "bg-green-900/30 text-green-400"
                          )}>
                            GI: {treatment.glycemicIndex}
                          </span>
                        )}
                        {treatment.absorptionRate && (
                          <span className={cn(
                            "px-1.5 py-0.5 rounded",
                            treatment.absorptionRate === 'very_fast' ? "bg-red-900/30 text-red-400" :
                            treatment.absorptionRate === 'fast' ? "bg-orange-900/30 text-orange-400" :
                            treatment.absorptionRate === 'medium' ? "bg-yellow-900/30 text-yellow-400" :
                            treatment.absorptionRate === 'slow' ? "bg-green-900/30 text-green-400" :
                            "bg-cyan-900/30 text-cyan-400"
                          )}>
                            {treatment.absorptionRate.replace('_', ' ')}
                          </span>
                        )}
                        {treatment.isLiquid && (
                          <span className="px-1.5 py-0.5 rounded bg-blue-900/30 text-blue-400">
                            liquid
                          </span>
                        )}
                        {treatment.fatContent && treatment.fatContent !== 'none' && treatment.fatContent !== 'low' && (
                          <span className="px-1.5 py-0.5 rounded bg-amber-900/30 text-amber-400">
                            {treatment.fatContent} fat
                          </span>
                        )}
                        {/* AI enriched indicator */}
                        {treatment.enrichedAt && (
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger>
                                <Sparkles className="w-3 h-3 text-purple-400" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>AI-enriched prediction</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                      </div>
                    )}

                    {/* Insulin notes */}
                    {treatment.type === 'insulin' && treatment.notes && (
                      <div className="mt-1 ml-11 text-xs text-gray-500">
                        {treatment.notes}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm text-center py-4">
                No recent activity
              </p>
            )}
          </motion.div>

          {/* AI Insights - Real-time actionable advice */}
          <motion.div variants={staggerItem} className="glass-card border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-500" />
              AI Insights
              {isLoadingInsight && (
                <span className="ml-2 animate-pulse text-xs text-purple-400">updating...</span>
              )}
            </h3>

            {/* Real-time insight - Primary display */}
            {realtimeInsight ? (
              <div className="space-y-3">
                {/* Urgency indicator */}
                {realtimeInsight.urgency === 'critical' && (
                  <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-2 mb-2">
                    <span className="text-red-400 text-xs font-semibold">⚠️ URGENT</span>
                  </div>
                )}
                {realtimeInsight.urgency === 'high' && (
                  <div className="bg-orange-500/20 border border-orange-500/50 rounded-lg p-2 mb-2">
                    <span className="text-orange-400 text-xs font-semibold">⚡ Action Needed</span>
                  </div>
                )}

                {/* Main insight */}
                <p className="text-gray-200 text-sm leading-relaxed font-medium">
                  {realtimeInsight.insight}
                </p>

                {/* Action recommendation */}
                {realtimeInsight.action && (
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3 mt-2">
                    <p className="text-purple-300 text-sm font-semibold">
                      💡 {realtimeInsight.action}
                    </p>
                  </div>
                )}

                {/* Reasoning (collapsible) */}
                <details className="mt-2">
                  <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                    Why this recommendation?
                  </summary>
                  <p className="text-xs text-gray-400 mt-1 pl-2 border-l border-gray-700">
                    {realtimeInsight.reasoning}
                  </p>
                </details>
              </div>
            ) : insightsData?.insights && insightsData.insights.length > 0 ? (
              <div className="space-y-3">
                {insightsData.insights.slice(0, 2).map((insight, i) => (
                  <p key={i} className="text-gray-300 text-sm leading-relaxed">
                    {insight.content}
                  </p>
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-gray-400 text-sm italic">
                  {isLoadingInsight
                    ? 'Analyzing your current data...'
                    : 'AI insights will appear here based on your glucose data.'
                  }
                </p>
              </div>
            )}

            {/* AI Chat - Ask "what-if" questions */}
            <div className="mt-4 pt-4 border-t border-gray-700">
              {/* Chat History - displayed ABOVE input */}
              {chatHistory.length > 0 && (
                <div className="mb-3 space-y-2 max-h-64 overflow-y-auto">
                  {chatHistory.map((chat, idx) => (
                    <div key={idx} className="bg-slate-800/50 rounded-lg p-3 space-y-2">
                      <p className="text-purple-400 text-xs font-medium">Q: {chat.question}</p>
                      <p className="text-gray-200 text-sm">{chat.response}</p>

                      {/* BG Predictions */}
                      {chat.prediction && (chat.prediction.bg30min || chat.prediction.peakBg) && (
                        <div className="flex gap-2 flex-wrap text-xs">
                          {chat.prediction.bg30min && (
                            <span className="bg-blue-500/20 px-2 py-0.5 rounded text-blue-300">
                              +30m: {chat.prediction.bg30min}
                            </span>
                          )}
                          {chat.prediction.bg60min && (
                            <span className="bg-blue-500/20 px-2 py-0.5 rounded text-blue-300">
                              +60m: {chat.prediction.bg60min}
                            </span>
                          )}
                          {chat.prediction.peakBg && (
                            <span className="bg-orange-500/20 px-2 py-0.5 rounded text-orange-300">
                              Peak: {chat.prediction.peakBg} ({chat.prediction.timeOfPeak})
                            </span>
                          )}
                        </div>
                      )}

                      {/* Recommendation */}
                      {chat.recommendation && chat.recommendation.insulin && (
                        <div className="bg-purple-500/10 border border-purple-500/30 rounded p-2">
                          <p className="text-purple-300 text-xs font-semibold">
                            💉 Recommended: {chat.recommendation.insulin}U
                            {chat.recommendation.timing && ` - ${chat.recommendation.timing}`}
                          </p>
                        </div>
                      )}

                      {/* Calculation (collapsible) */}
                      {chat.calculation && (
                        <details className="mt-1">
                          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                            Show calculation
                          </summary>
                          <p className="text-xs text-gray-400 mt-1 pl-2 border-l border-gray-700 whitespace-pre-wrap">
                            {chat.calculation}
                          </p>
                        </details>
                      )}

                      <div className="text-xs text-gray-500">
                        Confidence: {chat.confidence}
                      </div>
                    </div>
                  ))}
                  <button
                    onClick={() => setChatHistory([])}
                    className="text-xs text-gray-500 hover:text-gray-400"
                  >
                    Clear history
                  </button>
                </div>
              )}

              {/* Chat Input */}
              <div className="flex gap-2">
                <Input
                  placeholder="Ask: What if I eat 37g carbs?"
                  value={chatQuestion}
                  onChange={(e) => setChatQuestion(e.target.value)}
                  autoFocus={false}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && chatQuestion.trim() && !aiChat.isPending) {
                      const q = chatQuestion
                      aiChat.mutate({
                        question: chatQuestion,
                        currentBg,
                        trend: currentTrend ?? undefined,
                        iob,
                        cob,
                        isf,
                        icr: learnedRatios.icr.value,
                        pir: learnedRatios.pir.value,
                        dose: recommendedDose,
                        bgPressure,
                        tftPredictions: formattedTft.length > 0 ? formattedTft : undefined,
                      }, {
                        onSuccess: (data) => {
                          setChatHistory(prev => [...prev, {
                            question: q,
                            response: data.response,
                            prediction: data.prediction,
                            recommendation: data.recommendation,
                            calculation: data.calculation,
                            confidence: data.confidence,
                            timestamp: new Date().toISOString(),
                          }])
                          setChatQuestion('')
                        }
                      })
                    }
                  }}
                  className="flex-1 bg-slate-800 border-gray-700 text-sm"
                  disabled={aiChat.isPending}
                />
                <Button
                  size="sm"
                  onClick={() => {
                    if (chatQuestion.trim() && !aiChat.isPending) {
                      const q = chatQuestion
                      aiChat.mutate({
                        question: chatQuestion,
                        currentBg,
                        trend: currentTrend ?? undefined,
                        iob,
                        cob,
                        isf,
                        icr: learnedRatios.icr.value,
                        pir: learnedRatios.pir.value,
                        dose: recommendedDose,
                        bgPressure,
                        tftPredictions: formattedTft.length > 0 ? formattedTft : undefined,
                      }, {
                        onSuccess: (data) => {
                          setChatHistory(prev => [...prev, {
                            question: q,
                            response: data.response,
                            prediction: data.prediction,
                            recommendation: data.recommendation,
                            calculation: data.calculation,
                            confidence: data.confidence,
                            timestamp: new Date().toISOString(),
                          }])
                          setChatQuestion('')
                        }
                      })
                    }
                  }}
                  disabled={!chatQuestion.trim() || aiChat.isPending}
                  className="bg-purple-600 hover:bg-purple-700"
                >
                  {aiChat.isPending ? '...' : 'Ask'}
                </Button>
              </div>
            </div>
          </motion.div>

          {/* Quick Actions - No animation to prevent button movement on re-render */}
          <div className="flex gap-3">
            <Button
              className="flex-1 bg-orange-600 hover:bg-orange-700"
              onClick={() => setCarbsModalOpen(true)}
            >
              🍎 Log Carbs
            </Button>
            <Button
              className="flex-1 bg-green-600 hover:bg-green-700"
              onClick={() => setInsulinModalOpen(true)}
            >
              💉 Log Insulin
            </Button>
          </div>

          {/* Last Sync Info - No animation */}
          {currentTimestamp && (
            <div className="text-center text-sm text-gray-500">
              Last updated: {formatTime(currentTimestamp, userTimezone)}
            </div>
          )}
        </div>
      </motion.div>

      {/* Treatment Modals */}
      <LogCarbsModal
        open={carbsModalOpen}
        onOpenChange={setCarbsModalOpen}
      />

      {/* Add Profile Wizard */}
      <AddProfileWizard
        open={addProfileOpen}
        onClose={() => setAddProfileOpen(false)}
        onSuccess={() => setAddProfileOpen(false)}
      />
      <LogInsulinModal
        open={insulinModalOpen}
        onOpenChange={setInsulinModalOpen}
        currentBg={currentBg}
        recommendedDose={recommendedDose}
      />

      {/* Edit Treatment Dialog */}
      <Dialog open={!!editingTreatment} onOpenChange={() => setEditingTreatment(null)}>
        <DialogContent className="sm:max-w-md bg-slate-900 border-gray-700">
          <DialogHeader>
            <DialogTitle className="text-white flex items-center gap-2">
              <Pencil className="h-5 w-5" />
              Edit {editingTreatment?.type === 'carbs' ? 'Carbs' : 'Insulin'}
            </DialogTitle>
            <DialogDescription className="text-gray-400">
              Update the treatment details
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm text-gray-300">
                {editingTreatment?.type === 'carbs' ? 'Carbs (g)' : 'Insulin (U)'}
              </label>
              <Input
                type="number"
                value={editingTreatment?.value || ''}
                onChange={(e) => setEditingTreatment(prev => prev ? {
                  ...prev,
                  value: e.target.value ? parseFloat(e.target.value) : undefined
                } : null)}
                className="bg-slate-800 border-gray-700 text-white"
                step={editingTreatment?.type === 'insulin' ? 0.1 : 1}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-gray-300">Notes</label>
              <Textarea
                value={editingTreatment?.notes || ''}
                onChange={(e) => setEditingTreatment(prev => prev ? {
                  ...prev,
                  notes: e.target.value || undefined
                } : null)}
                className="bg-slate-800 border-gray-700 text-white resize-none"
                rows={2}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-gray-300 flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Time
              </label>
              <Input
                type="datetime-local"
                value={editingTreatment?.timestamp ? formatDateTimeLocal(editingTreatment.timestamp) : ''}
                onChange={(e) => setEditingTreatment(prev => prev ? {
                  ...prev,
                  timestamp: parseDateTimeLocalToISO(e.target.value)
                } : null)}
                className="bg-slate-800 border-gray-700 text-white"
                step="60"
              />
            </div>
          </div>
          <DialogFooter className="gap-2">
            <Button
              variant="ghost"
              onClick={() => setEditingTreatment(null)}
              className="text-gray-400"
            >
              Cancel
            </Button>
            <Button
              onClick={() => {
                if (editingTreatment) {
                  const data = editingTreatment.type === 'carbs'
                    ? { carbs: editingTreatment.value, notes: editingTreatment.notes, timestamp: editingTreatment.timestamp }
                    : { insulin: editingTreatment.value, notes: editingTreatment.notes, timestamp: editingTreatment.timestamp }
                  updateTreatment({ treatmentId: editingTreatment.id, data }, {
                    onSuccess: () => setEditingTreatment(null)
                  })
                }
              }}
              disabled={isUpdating}
              className={editingTreatment?.type === 'carbs' ? 'bg-green-600 hover:bg-green-700' : 'bg-orange-600 hover:bg-orange-700'}
            >
              {isUpdating ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Treatment Confirmation */}
      <AlertDialog open={!!deletingTreatmentId} onOpenChange={() => setDeletingTreatmentId(null)}>
        <AlertDialogContent className="bg-slate-900 border-gray-700">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-white">Delete Treatment?</AlertDialogTitle>
            <AlertDialogDescription className="text-gray-400">
              This action cannot be undone. The treatment will be permanently removed.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-slate-800 border-gray-700 text-gray-300 hover:bg-slate-700 hover:text-white">
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deletingTreatmentId) {
                  deleteTreatment(deletingTreatmentId, {
                    onSuccess: () => setDeletingTreatmentId(null)
                  })
                }
              }}
              disabled={isDeleting}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              {isDeleting ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Delete'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
