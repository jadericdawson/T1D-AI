/**
 * Glucose Chart Component
 * Displays glucose readings over time with predictions and target range
 */
import { useMemo, useEffect, Component, ErrorInfo, ReactNode } from 'react'
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  // Area, // Temporarily disabled
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  // ReferenceDot, // Temporarily disabled
} from 'recharts'
import { format, subHours, isAfter } from 'date-fns'
import { AlertCircle, RefreshCw } from 'lucide-react'
import { getGlucoseColor, cn } from '@/lib/utils'

// Error boundary specifically for the chart component
interface ChartErrorBoundaryProps {
  children: ReactNode
  fallbackHeight?: string
}

interface ChartErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

class ChartErrorBoundary extends Component<ChartErrorBoundaryProps, ChartErrorBoundaryState> {
  constructor(props: ChartErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ChartErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Chart Error:', error, errorInfo)
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className={cn('flex flex-col items-center justify-center bg-slate-800/50 rounded-lg', this.props.fallbackHeight || 'h-80')}>
          <AlertCircle className="w-8 h-8 text-yellow-500 mb-2" />
          <p className="text-gray-400 text-sm mb-2">Chart failed to render</p>
          <p className="text-gray-500 text-xs mb-3 max-w-xs text-center">
            {this.state.error?.message || 'An error occurred'}
          </p>
          <button
            onClick={this.handleRetry}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-gray-300 text-sm rounded transition"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

// Types
interface GlucoseReading {
  timestamp: string
  value: number
  trend?: string | null
}

interface Treatment {
  timestamp: string
  type: 'insulin' | 'carbs'
  value: number
}

interface Prediction {
  timestamp: string
  linear?: number
  lstm?: number
}

// IOB/COB effect curve point
interface EffectPoint {
  minutesAhead: number
  iobEffect: number
  cobEffect: number
  netEffect: number
  remainingIOB?: number
  remainingCOB?: number
}

// TFT prediction with uncertainty
interface TFTPrediction {
  timestamp: string
  horizon: number  // 30, 45, 60 minutes
  value: number    // Median (50th percentile)
  lower: number    // 10th percentile
  upper: number    // 90th percentile
}

interface GlucoseChartProps {
  readings: GlucoseReading[]
  predictions?: Prediction[]
  tftPredictions?: TFTPrediction[]
  treatments?: Treatment[]
  effectCurve?: EffectPoint[]
  currentBg?: number
  iob?: number
  cob?: number
  timeRange: '1hr' | '3hr' | '6hr' | '12hr' | '24hr'
  targetLow?: number
  targetHigh?: number
  criticalLow?: number
  criticalHigh?: number
  showPredictions?: boolean
  showTreatments?: boolean
  showTargetRange?: boolean
  showEffectCurve?: boolean
  showEffectAreas?: boolean
  showEffectiveBg?: boolean
  className?: string
}

// Time range in hours
const timeRangeHours: Record<string, number> = {
  '1hr': 1,
  '3hr': 3,
  '6hr': 6,
  '12hr': 12,
  '24hr': 24,
}

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null

  const time = format(new Date(label), 'h:mm a')
  const glucoseData = payload.find((p: any) => p.dataKey === 'value')
  const linearPred = payload.find((p: any) => p.dataKey === 'linear')
  const lstmPred = payload.find((p: any) => p.dataKey === 'lstm')
  const tftPred = payload.find((p: any) => p.dataKey === 'tftValue')
  const tftLower = payload.find((p: any) => p.dataKey === 'tftLower')
  const tftUpper = payload.find((p: any) => p.dataKey === 'tftUpper')

  return (
    <div className="glass-card p-3 text-sm border border-gray-700/50">
      <p className="text-gray-400 mb-2">{time}</p>
      {glucoseData && glucoseData.value !== undefined && (
        <p className="text-white">
          <span className="text-cyan">Glucose:</span>{' '}
          <span style={{ color: getGlucoseColor(glucoseData.value) }}>
            {glucoseData.value} mg/dL
          </span>
        </p>
      )}
      {linearPred && linearPred.value !== undefined && (
        <p className="text-gray-400">
          Linear: {linearPred.value?.toFixed(0)} mg/dL
        </p>
      )}
      {lstmPred && lstmPred.value !== undefined && (
        <p className="text-purple-400">
          LSTM: {lstmPred.value?.toFixed(0)} mg/dL
        </p>
      )}
      {tftPred && tftPred.value !== undefined && tftLower && tftUpper && (
        <p className="text-orange-400">
          TFT: {tftPred.value?.toFixed(0)} mg/dL
          <span className="text-gray-500 text-xs ml-1">
            ({tftLower.value?.toFixed(0)}-{tftUpper.value?.toFixed(0)})
          </span>
        </p>
      )}
    </div>
  )
}

// Custom shape for treatment ReferenceDots - temporarily disabled
// const TreatmentShape = (props: any) => { ... }

export function GlucoseChart({
  readings,
  predictions = [],
  tftPredictions = [],
  treatments = [],
  effectCurve = [],
  currentBg,
  iob: _iob = 0,
  cob: _cob = 0,
  timeRange = '3hr',
  targetLow = 70,
  targetHigh = 180,
  criticalLow = 54,
  criticalHigh = 250,
  showPredictions = true,
  showTreatments: _showTreatments = true,
  showTargetRange = true,
  showEffectCurve = false,
  showEffectAreas = false,
  showEffectiveBg = false,
  className,
}: GlucoseChartProps) {
  // Process data for chart
  const chartData = useMemo(() => {
    const hours = timeRangeHours[timeRange]
    const cutoff = subHours(new Date(), hours)

    // Filter readings within time range and sort by timestamp (oldest first)
    // Also validate that values are finite numbers
    const filteredReadings = readings
      .filter((r) => {
        const ts = new Date(r.timestamp).getTime()
        return isAfter(new Date(r.timestamp), cutoff) &&
               Number.isFinite(ts) &&
               Number.isFinite(r.value)
      })
      .map((r) => ({
        timestamp: new Date(r.timestamp).getTime(),
        value: r.value,
        trend: r.trend,
      }))
      .sort((a, b) => a.timestamp - b.timestamp)

    // Add treatments as markers
    const treatmentMap = new Map<number, Treatment>()
    treatments.forEach((t) => {
      if (isAfter(new Date(t.timestamp), cutoff)) {
        const ts = new Date(t.timestamp).getTime()
        treatmentMap.set(ts, t)
      }
    })

    // Chart data point type - allow null for Recharts compatibility
    type ChartDataPoint = {
      timestamp: number
      value: number | null
      trend: string | null | undefined
      treatment: Treatment | null
      linear?: number | null
      lstm?: number | null
      // TFT predictions
      tftValue?: number | null
      tftLower?: number | null
      tftUpper?: number | null
      tftHorizon?: number
      // Effect curve data
      iobEffect?: number | null
      cobEffect?: number | null
      netEffect?: number | null
      effectiveBg?: number | null
      // Remaining IOB/COB for decay visualization
      remainingIOB?: number | null
      remainingCOB?: number | null
    }

    // Merge with treatment data
    const dataWithTreatments: ChartDataPoint[] = filteredReadings.map((r) => ({
      ...r,
      treatment: treatmentMap.get(r.timestamp) || null,
    }))

    // Add predictions (extend beyond current time)
    if (showPredictions && predictions.length > 0) {
      const latestReading = filteredReadings[filteredReadings.length - 1]
      if (latestReading) {
        // Add bridge point - last actual reading with first prediction values
        // This connects the prediction lines to the actual glucose line
        const lastIdx = dataWithTreatments.findIndex(d => d.timestamp === latestReading.timestamp)
        if (lastIdx >= 0) {
          dataWithTreatments[lastIdx].linear = latestReading.value
          dataWithTreatments[lastIdx].lstm = latestReading.value
        }

        // Add prediction points (use null for missing glucose - Recharts handles null better than undefined)
        predictions.forEach((pred, i) => {
          const predTime = latestReading.timestamp + (i + 1) * 5 * 60 * 1000 // 5 min intervals
          dataWithTreatments.push({
            timestamp: predTime,
            value: null,
            trend: null,
            treatment: null,
            linear: Number.isFinite(pred.linear) ? pred.linear : null,
            lstm: pred.lstm != null && Number.isFinite(pred.lstm) ? pred.lstm : null,
          })
        })
      }
    }

    // Add TFT predictions with uncertainty bands
    if (showPredictions && tftPredictions.length > 0) {
      const latestReading = filteredReadings[filteredReadings.length - 1]
      if (latestReading) {
        // Add bridge point - connect TFT line to the last actual reading
        const lastIdx = dataWithTreatments.findIndex(d => d.timestamp === latestReading.timestamp)
        if (lastIdx >= 0) {
          dataWithTreatments[lastIdx].tftValue = latestReading.value
          dataWithTreatments[lastIdx].tftLower = latestReading.value
          dataWithTreatments[lastIdx].tftUpper = latestReading.value
        }

        // Add TFT prediction points (validate all values are finite)
        tftPredictions.forEach((pred) => {
          const predTime = new Date(pred.timestamp).getTime()
          if (!Number.isFinite(predTime)) return // Skip invalid timestamps
          dataWithTreatments.push({
            timestamp: predTime,
            value: null,
            trend: null,
            treatment: null,
            tftValue: Number.isFinite(pred.value) ? pred.value : null,
            tftLower: Number.isFinite(pred.lower) ? pred.lower : null,
            tftUpper: Number.isFinite(pred.upper) ? pred.upper : null,
            tftHorizon: pred.horizon,
          })
        })
      }
    }

    // Add effect curve data (IOB/COB push/pull)
    if ((showEffectCurve || showEffectAreas || showEffectiveBg) && effectCurve.length > 0) {
      const latestReading = filteredReadings[filteredReadings.length - 1]
      if (latestReading) {
        const baseBg = currentBg ?? latestReading.value
        effectCurve.forEach((effect) => {
          const effectTime = latestReading.timestamp + effect.minutesAhead * 60 * 1000
          const existingPoint = dataWithTreatments.find(d => d.timestamp === effectTime)

          if (existingPoint) {
            existingPoint.iobEffect = effect.iobEffect
            existingPoint.cobEffect = effect.cobEffect
            existingPoint.netEffect = effect.netEffect
            existingPoint.effectiveBg = baseBg + effect.netEffect
            existingPoint.remainingIOB = effect.remainingIOB
            existingPoint.remainingCOB = effect.remainingCOB
          } else {
            const effectiveBgValue = baseBg + effect.netEffect
            dataWithTreatments.push({
              timestamp: effectTime,
              value: null,
              trend: null,
              treatment: null,
              iobEffect: Number.isFinite(effect.iobEffect) ? effect.iobEffect : null,
              cobEffect: Number.isFinite(effect.cobEffect) ? effect.cobEffect : null,
              netEffect: Number.isFinite(effect.netEffect) ? effect.netEffect : null,
              effectiveBg: Number.isFinite(effectiveBgValue) ? effectiveBgValue : null,
              remainingIOB: effect.remainingIOB != null && Number.isFinite(effect.remainingIOB) ? effect.remainingIOB : null,
              remainingCOB: effect.remainingCOB != null && Number.isFinite(effect.remainingCOB) ? effect.remainingCOB : null,
            })
          }
        })
      }
    }

    return dataWithTreatments.sort((a, b) => a.timestamp - b.timestamp)
  }, [readings, predictions, tftPredictions, treatments, effectCurve, currentBg, timeRange, showPredictions, showEffectCurve, showEffectAreas, showEffectiveBg])

  // Separate dataset for treatment markers - only points with valid glucose values AND treatments
  // This prevents Recharts invariant errors when calculating coordinates for undefined values
  const treatmentMarkerData = useMemo(() => {
    return chartData.filter(d =>
      d.treatment &&
      d.value !== undefined &&
      d.value !== null &&
      Number.isFinite(d.value)
    )
  }, [chartData])

  // Filtered glucose data - ONLY points with valid glucose values
  // This is critical for the Line component to avoid invariant errors
  const glucoseOnlyData = useMemo(() => {
    return chartData.filter(d =>
      d.value !== undefined &&
      d.value !== null &&
      Number.isFinite(d.value)
    )
  }, [chartData])

  // Calculate Y axis domain (include all prediction types)
  const yDomain = useMemo(() => {
    const values = chartData
      .flatMap((d: any) => [d.value, d.linear, d.lstm, d.tftValue, d.tftLower, d.tftUpper])
      .filter((v): v is number => v !== undefined && v !== null && Number.isFinite(v))

    if (values.length === 0) return [40, 300]

    const min = Math.min(...values, criticalLow)
    const max = Math.max(...values, criticalHigh)

    // Ensure valid bounds
    const lowerBound = Math.max(40, Math.floor(min / 10) * 10 - 20)
    const upperBound = Math.min(400, Math.ceil(max / 10) * 10 + 20)

    // Safety check - ensure valid domain
    if (!Number.isFinite(lowerBound) || !Number.isFinite(upperBound) || lowerBound >= upperBound) {
      return [40, 300]
    }

    return [lowerBound, upperBound]
  }, [chartData, criticalLow, criticalHigh])

  // IOB/COB domains and TFT validation temporarily removed for debugging

  // Diagnostic logging (temporary - helps identify data issues)
  useEffect(() => {
    console.log('[GlucoseChart] Data Summary:', {
      chartDataLength: chartData.length,
      glucoseOnlyLength: glucoseOnlyData.length,
      treatmentMarkers: treatmentMarkerData.length,
      yDomain,
    })

    // Log sample glucose data
    if (glucoseOnlyData.length > 0) {
      console.log('[GlucoseChart] Sample glucose data:', {
        first: glucoseOnlyData[0],
        last: glucoseOnlyData[glucoseOnlyData.length - 1],
      })
    }
  }, [chartData, glucoseOnlyData, treatmentMarkerData, yDomain])

  // Format X axis
  const formatXAxis = (timestamp: number) => {
    return format(new Date(timestamp), timeRange === '24hr' ? 'HH:mm' : 'h:mm a')
  }

  if (chartData.length === 0) {
    return (
      <div className={cn('flex items-center justify-center h-80 text-gray-500', className)}>
        No glucose data available
      </div>
    )
  }

  return (
    <div className={cn('w-full h-80', className)}>
      <ChartErrorBoundary fallbackHeight="h-80">
      <ResponsiveContainer width="100%" height="100%">
        {/* Use glucoseOnlyData instead of chartData to avoid null value coordinate calculation errors */}
        <ComposedChart data={glucoseOnlyData} margin={{ top: 10, right: showEffectAreas ? 80 : 10, left: 0, bottom: 0 }}>
          {/* Grid */}
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#374151"
            opacity={0.5}
          />

          {/* Target range background */}
          {showTargetRange && (
            <ReferenceArea
              y1={targetLow}
              y2={targetHigh}
              fill="#00c6ff"
              fillOpacity={0.1}
            />
          )}

          {/* Critical lines */}
          <ReferenceLine
            y={criticalLow}
            stroke="#dc2626"
            strokeDasharray="5 5"
            strokeWidth={1}
          />
          <ReferenceLine
            y={criticalHigh}
            stroke="#dc2626"
            strokeDasharray="5 5"
            strokeWidth={1}
          />

          {/* Target lines */}
          <ReferenceLine
            y={targetLow}
            stroke="#00c6ff"
            strokeDasharray="3 3"
            strokeWidth={1}
            opacity={0.5}
          />
          <ReferenceLine
            y={targetHigh}
            stroke="#00c6ff"
            strokeDasharray="3 3"
            strokeWidth={1}
            opacity={0.5}
          />

          {/* Axes */}
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatXAxis}
            stroke="#6b7280"
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            yAxisId="glucose"
            domain={yDomain}
            stroke="#6b7280"
            fontSize={11}
            tickLine={false}
            axisLine={false}
            width={40}
          />

          {/* Secondary Y axes temporarily disabled for debugging */}

          {/* Tooltip */}
          <Tooltip content={<CustomTooltip />} />

          {/* Glucose line */}
          <Line
            type="monotone"
            dataKey="value"
            yAxisId="glucose"
            stroke="#00c6ff"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6, fill: '#00c6ff', stroke: '#fff', strokeWidth: 2 }}
            connectNulls={false}
            isAnimationActive={false}
          />

          {/* All predictions and effect lines temporarily disabled for debugging */}

          {/* Treatment markers temporarily disabled for debugging */}
        </ComposedChart>
      </ResponsiveContainer>
      </ChartErrorBoundary>
    </div>
  )
}

// Chart legend component
interface ChartLegendProps {
  showEffectiveBg?: boolean
  showEffectAreas?: boolean
  showTftPredictions?: boolean
}

export function ChartLegend({
  showEffectiveBg = false,
  showEffectAreas = false,
  showTftPredictions = false,
}: ChartLegendProps = {}) {
  return (
    <div className="flex flex-wrap items-center gap-4 text-sm">
      {/* Core glucose */}
      <div className="flex items-center gap-2">
        <div className="w-4 h-0.5 bg-cyan" />
        <span className="text-gray-400">Glucose</span>
      </div>

      {/* Predictions */}
      <div className="flex items-center gap-2">
        <div className="w-4 h-0.5 bg-purple-500" style={{ borderTopWidth: 2, borderTopStyle: 'dashed' }} />
        <span className="text-gray-400">LSTM (5-15m)</span>
      </div>

      {showTftPredictions && (
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-orange-500" style={{ borderTopWidth: 2, borderTopStyle: 'dashed' }} />
          <span className="text-gray-400">TFT (30-60m)</span>
        </div>
      )}

      <div className="flex items-center gap-2">
        <div className="w-4 h-0.5 bg-gray-400" style={{ borderTopWidth: 2, borderTopStyle: 'dashed' }} />
        <span className="text-gray-400">Linear</span>
      </div>

      {/* IOB/COB Effects */}
      {showEffectiveBg && (
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-amber-500" style={{ borderTopWidth: 2, borderTopStyle: 'dashed' }} />
          <span className="text-gray-400">Effective BG</span>
        </div>
      )}

      {showEffectAreas && (
        <>
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-blue-500" />
            <span className="text-gray-400">IOB (U)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-green-500" />
            <span className="text-gray-400">COB (g)</span>
          </div>
        </>
      )}

      {/* Treatments */}
      <div className="flex items-center gap-2">
        <div className="w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-b-[8px] border-b-green-500" />
        <span className="text-gray-400">Carbs</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-orange-500" />
        <span className="text-gray-400">Insulin</span>
      </div>
    </div>
  )
}

export default GlucoseChart
