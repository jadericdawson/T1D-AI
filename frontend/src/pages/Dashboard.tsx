/**
 * T1D-AI Dashboard
 * Main glucose monitoring dashboard with real-time updates
 */
import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity, Clock, Settings, RefreshCw, Bell, Sparkles,
  Wifi, WifiOff, AlertCircle, Loader2
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { cn, formatTime } from '@/lib/utils'

// Components
import { CurrentGlucose } from '@/components/glucose/CurrentGlucose'
import { GlucoseChart, ChartLegend } from '@/components/glucose/GlucoseChart'
import { PredictionsCard } from '@/components/glucose/Predictions'
import { IOBCard, COBCard, ISFCard, DoseCard, WarningCard } from '@/components/metrics/MetricCard'
import { LogCarbsModal, LogInsulinModal } from '@/components/treatments/TreatmentModal'

// Hooks
import {
  useCurrentGlucose,
  useGlucoseHistory,
  useRecentTreatments,
  useInsights,
  usePredictionAccuracy,
} from '@/hooks/useGlucose'
import { useGlucoseStream } from '@/hooks/useWebSocket'

// Store
import { useGlucoseStore } from '@/stores/glucoseStore'

const USER_ID = 'demo_user'

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

  // Zustand store
  const { preferences, updatePreferences } = useGlucoseStore()

  // React Query hooks
  const {
    data: currentData,
    isLoading: isLoadingCurrent,
    isError: isCurrentError,
    refetch: refetchCurrent
  } = useCurrentGlucose(USER_ID)

  const timeRangeHours: Record<string, number> = {
    '1hr': 1,
    '3hr': 3,
    '6hr': 6,
    '12hr': 12,
    '24hr': 24,
  }

  const {
    data: historyData,
    isLoading: isLoadingHistory
  } = useGlucoseHistory(USER_ID, timeRangeHours[timeRange])

  const { data: treatmentsData } = useRecentTreatments(USER_ID, 12)
  const { data: insightsData } = useInsights(USER_ID, 5)
  const { data: accuracyData } = usePredictionAccuracy()

  // WebSocket for real-time updates
  const { isConnected, connectionStatus, latestData } = useGlucoseStream(USER_ID)

  // Update preferences when time range changes
  useEffect(() => {
    if (timeRange !== preferences.chartTimeRange) {
      updatePreferences({ chartTimeRange: timeRange })
    }
  }, [timeRange, preferences.chartTimeRange, updatePreferences])

  // Extract current glucose data (prefer WebSocket, fallback to query)
  const currentBg = latestData?.value ?? currentData?.glucose?.value
  const currentTrend = latestData?.trend ?? currentData?.glucose?.trend
  const currentTimestamp = latestData?.timestamp ?? currentData?.glucose?.timestamp

  // Extract predictions
  const linearPredictions = latestData?.predictions?.linear ?? currentData?.glucose?.predictions?.linear ?? []
  const lstmPredictions = latestData?.predictions?.lstm ?? currentData?.glucose?.predictions?.lstm ?? null
  const modelAvailable = latestData?.modelAvailable ?? (lstmPredictions !== null && lstmPredictions.length > 0)

  // Extract metrics
  const iob = latestData?.metrics?.iob ?? currentData?.metrics?.iob ?? 0
  const cob = latestData?.metrics?.cob ?? currentData?.metrics?.cob ?? 0
  const isf = latestData?.metrics?.isf ?? currentData?.metrics?.isf ?? 50
  const recommendedDose = currentData?.metrics?.recommendedDose ?? 0

  // Accuracy stats
  const accuracy = accuracyData ?? currentData?.accuracy ?? null

  // Format treatments for display
  const recentTreatments = (treatmentsData ?? []).slice(0, 5).map((t) => ({
    type: t.type,
    value: t.type === 'insulin' ? t.insulin : t.carbs,
    time: formatTime(t.timestamp),
    timestamp: t.timestamp
  }))

  // Format history for chart
  const chartReadings = (historyData?.readings ?? []).map((r) => ({
    timestamp: r.timestamp,
    value: r.value,
    trend: r.trend
  }))

  // Generate chart predictions
  const chartPredictions = linearPredictions.map((linear, i) => ({
    timestamp: new Date(Date.now() + (i + 1) * 5 * 60 * 1000).toISOString(),
    linear,
    lstm: lstmPredictions?.[i]
  }))

  // Format treatments for chart
  const chartTreatments = (treatmentsData ?? []).map((t) => ({
    timestamp: t.timestamp,
    type: t.type as 'insulin' | 'carbs',
    value: t.type === 'insulin' ? (t.insulin ?? 0) : (t.carbs ?? 0)
  }))

  // Handle refresh
  const handleRefresh = () => {
    refetchCurrent()
  }

  // Loading state
  if (isLoadingCurrent && !latestData) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-cyan mx-auto mb-4" />
          <p className="text-gray-400">Loading glucose data...</p>
        </div>
      </div>
    )
  }

  // Error state
  if (isCurrentError && !latestData) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-gray-300 mb-4">Failed to load glucose data</p>
          <Button onClick={handleRefresh} className="bg-cyan hover:bg-cyan/80">
            Try Again
          </Button>
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
          <Link to="/" className="flex items-center gap-2 text-cyan hover:opacity-80 transition">
            <Activity className="w-8 h-8" />
            <span className="font-bold text-xl">T1D-AI</span>
          </Link>

          {/* Connection Status */}
          <Badge
            variant="outline"
            className={cn(
              'flex items-center gap-1.5',
              isConnected
                ? 'border-cyan/30 text-cyan'
                : 'border-yellow-500/30 text-yellow-500'
            )}
          >
            {isConnected ? (
              <>
                <Wifi className="w-3 h-3" />
                Live
              </>
            ) : (
              <>
                <WifiOff className="w-3 h-3" />
                {connectionStatus === 'connecting' ? 'Connecting...' : 'Offline'}
              </>
            )}
          </Badge>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            className="text-gray-400 hover:text-white"
            onClick={handleRefresh}
          >
            <RefreshCw className={cn('w-5 h-5', isLoadingCurrent && 'animate-spin')} />
          </Button>
          <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white">
            <Bell className="w-5 h-5" />
          </Button>
          <Link to="/settings">
            <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white">
              <Settings className="w-5 h-5" />
            </Button>
          </Link>
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
              />
            ) : (
              <div className="glass-card text-center py-8">
                <p className="text-gray-500">No glucose data available</p>
              </div>
            )}
          </motion.div>

          {/* Metrics Grid */}
          <motion.div variants={staggerItem} className="grid grid-cols-2 gap-4">
            <IOBCard value={iob} />
            <COBCard value={cob} />
            <ISFCard value={isf} />
            <DoseCard value={recommendedDose} />
          </motion.div>

          {/* Predictions */}
          <motion.div variants={staggerItem}>
            <PredictionsCard
              linear={linearPredictions}
              lstm={lstmPredictions}
              accuracy={accuracy ? {
                linearWins: accuracy.linearWins,
                lstmWins: accuracy.lstmWins,
                totalComparisons: accuracy.totalComparisons
              } : undefined}
              modelAvailable={modelAvailable}
            />
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
            ) : (
              <GlucoseChart
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
              />
            )}

            {/* Legend */}
            <div className="mt-4">
              <ChartLegend />
            </div>
          </div>
        </motion.div>

        {/* Right Column - Treatments & Insights */}
        <div className="lg:col-span-3 space-y-6">
          {/* Recent Treatments */}
          <motion.div variants={staggerItem} className="glass-card">
            <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
              <Clock className="w-5 h-5 text-gray-400" />
              Recent Treatments
            </h3>

            {recentTreatments.length > 0 ? (
              <div className="space-y-3">
                {recentTreatments.map((treatment, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50"
                  >
                    <div className="flex items-center gap-3">
                      {treatment.type === 'carbs' ? (
                        <span className="text-2xl">🍎</span>
                      ) : (
                        <span className="text-2xl">💉</span>
                      )}
                      <span className="text-gray-300">
                        {treatment.value}{treatment.type === 'carbs' ? 'g' : 'U'}
                      </span>
                    </div>
                    <span className="text-gray-500 text-sm">{treatment.time}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm text-center py-4">
                No recent treatments
              </p>
            )}
          </motion.div>

          {/* AI Insights */}
          <motion.div variants={staggerItem} className="glass-card border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-500" />
              AI Insights
            </h3>

            {insightsData?.insights && insightsData.insights.length > 0 ? (
              <div className="space-y-3">
                {insightsData.insights.map((insight, i) => (
                  <p key={i} className="text-gray-300 text-sm leading-relaxed">
                    {insight.content}
                  </p>
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-gray-400 text-sm italic">
                  AI insights will appear here based on your glucose patterns.
                </p>
              </div>
            )}
          </motion.div>

          {/* Quick Actions */}
          <motion.div variants={staggerItem} className="flex gap-3">
            <Button
              className="flex-1 bg-green-600 hover:bg-green-700"
              onClick={() => setCarbsModalOpen(true)}
            >
              🍎 Log Carbs
            </Button>
            <Button
              className="flex-1 bg-orange-600 hover:bg-orange-700"
              onClick={() => setInsulinModalOpen(true)}
            >
              💉 Log Insulin
            </Button>
          </motion.div>

          {/* Last Sync Info */}
          {currentTimestamp && (
            <motion.div variants={staggerItem} className="text-center text-sm text-gray-500">
              Last updated: {formatTime(currentTimestamp)}
            </motion.div>
          )}
        </div>
      </motion.div>

      {/* Treatment Modals */}
      <LogCarbsModal
        open={carbsModalOpen}
        onOpenChange={setCarbsModalOpen}
      />
      <LogInsulinModal
        open={insulinModalOpen}
        onOpenChange={setInsulinModalOpen}
        currentBg={currentBg}
        recommendedDose={recommendedDose}
      />
    </div>
  )
}
