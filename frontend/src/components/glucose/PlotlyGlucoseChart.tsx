/**
 * Plotly-based Glucose Chart Component
 * Displays glucose readings with ML predictions, IOB/COB decay curves, and treatments
 */
import { useMemo, useRef, useCallback, useState, useEffect, Component, ErrorInfo, ReactNode } from 'react'
import Plot from 'react-plotly.js'
import type { Data, Layout, Config } from 'plotly.js'
// date-fns import removed - using native Date math for golden ratio calculation
import { AlertCircle, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  transformGlucoseData,
  transformEffectCurve,
  transformTreatments,
  transformHistoricalIobCob,
  calculateGlucoseDomain,
  calculateIobDomain,
  calculateCobDomain,
  calculatePobDomain,
} from './chartDataTransform'

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
    console.error('PlotlyGlucoseChart Error:', error, errorInfo)
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
  notes?: string
  isLiquid?: boolean
}

interface Prediction {
  timestamp: string
  linear?: number
  lstm?: number
}

interface EffectPoint {
  minutesAhead: number
  iobEffect: number
  cobEffect: number
  netEffect: number
  remainingIOB?: number
  remainingCOB?: number
  remainingPOB?: number     // Future projected POB (grams)
  insulinActivity?: number  // 0-1 bell-shaped activity curve
  carbActivity?: number     // 0-1 bell-shaped activity curve
  expectedBg?: number       // Combined BG trajectory
  bgWithIobOnly?: number    // BG trajectory with only insulin effect (pull-down floor)
  bgWithCobOnly?: number    // BG trajectory with only carb effect (push-up ceiling)
}

interface TFTPrediction {
  timestamp: string
  horizon: number
  value: number
  lower: number
  upper: number
}

// Dual physics-based prediction (model vs hardcoded)
interface PhysicsPrediction {
  minutesAhead: number
  predictedBg: number
  bgTrendComponent?: number
  insulinComponent?: number
  carbComponent?: number
  remainingIOB?: number
  remainingCOB?: number
  timestamp?: string
}

interface HistoricalIobCobPoint {
  timestamp: string
  iob: number
  cob: number
  pob: number  // Protein on Board (grams)
}

interface PlotlyGlucoseChartProps {
  readings: GlucoseReading[]
  predictions?: Prediction[]
  tftPredictions?: TFTPrediction[]
  treatments?: Treatment[]
  effectCurve?: EffectPoint[]
  historicalIobCob?: HistoricalIobCobPoint[]
  currentBg?: number
  iob?: number
  cob?: number
  isf?: number  // Insulin Sensitivity Factor (mg/dL per unit)
  icr?: number  // Insulin to Carb Ratio (grams per unit)
  // Dual physics-based predictions for comparison
  modelPredictions?: PhysicsPrediction[]  // Learned from BG data (prominent)
  hardcodedPredictions?: PhysicsPrediction[]  // Standard textbook (faded)
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
  showIobCobLines?: boolean  // IOB/COB decay lines on secondary Y-axes
  showDualPredictions?: boolean  // Show model vs hardcoded comparison
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

export function PlotlyGlucoseChart({
  readings,
  predictions: _predictions = [],  // Legacy: kept for backward compatibility
  tftPredictions: _tftPredictions = [],  // Legacy: replaced by physics-based prediction
  treatments = [],
  effectCurve = [],
  historicalIobCob = [],
  currentBg,
  iob: _iob = 0,
  cob: _cob = 0,
  isf: _isf = 55,  // Legacy: now using effect curves for prediction
  icr: _icr = 10,  // Legacy: now using effect curves for prediction
  modelPredictions = [],  // Learned from BG data (prominent)
  hardcodedPredictions = [],  // Standard textbook (faded)
  timeRange = '3hr',
  targetLow = 70,
  targetHigh = 180,
  criticalLow = 54,
  criticalHigh = 250,
  showPredictions = true,
  showTreatments = true,
  showTargetRange = true,
  showEffectCurve = false,
  showEffectAreas = false,
  showEffectiveBg = false,
  showIobCobLines = false,  // IOB/COB decay lines on secondary Y-axes
  showDualPredictions = false,  // Show model vs hardcoded comparison
  className,
}: PlotlyGlucoseChartProps) {
  // Ref for fullscreen container
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Listen for fullscreen changes (native API)
  useEffect(() => {
    const handleFullscreenChange = () => {
      // Only update if we're using native fullscreen (not CSS fallback)
      if (document.fullscreenElement) {
        setIsFullscreen(true)
      } else if (!isFullscreen) {
        // Don't reset if we're using CSS fullscreen mode
      }
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [isFullscreen])

  // Handle escape key to exit CSS fullscreen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false)
      }
    }
    if (isFullscreen) {
      document.addEventListener('keydown', handleKeyDown)
      return () => document.removeEventListener('keydown', handleKeyDown)
    }
  }, [isFullscreen])

  // Toggle fullscreen - try native API first, fall back to CSS
  const toggleFullscreen = useCallback(() => {
    if (isFullscreen) {
      // Exit fullscreen
      if (document.fullscreenElement) {
        document.exitFullscreen().catch(() => {})
      }
      setIsFullscreen(false)
    } else {
      // Enter fullscreen - try native API first
      if (chartContainerRef.current && document.fullscreenEnabled) {
        chartContainerRef.current.requestFullscreen().catch(() => {
          // Native fullscreen failed (mobile/iOS), use CSS fallback
          setIsFullscreen(true)
        })
      } else {
        // No native fullscreen support, use CSS fallback
        setIsFullscreen(true)
      }
    }
  }, [isFullscreen])

  // Process all readings - DON'T filter by time range
  // Time range controls x-axis zoom, not data filtering (user can pan to see all data)
  const allReadings = useMemo(() => {
    return readings
      .filter(r => Number.isFinite(r.value))
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
  }, [readings])

  // Transform all data to Plotly format
  const glucoseData = useMemo(() =>
    transformGlucoseData(allReadings),
    [allReadings]
  )

  const lastReading = allReadings[allReadings.length - 1]
  const lastTimestamp = lastReading ? new Date(lastReading.timestamp) : new Date()
  const lastGlucose = currentBg ?? lastReading?.value ?? 100

  // Effect data is needed for predictions (Predicted BG Pressure) as well as effect curves
  const effectData = useMemo(() =>
    (showPredictions || showEffectCurve || showEffectAreas || showEffectiveBg) && effectCurve.length > 0
      ? transformEffectCurve(effectCurve, lastTimestamp, lastGlucose)
      : { timestamps: [], iob: [], cob: [], pob: [], iobEffect: [], cobEffect: [], insulinActivity: [], carbActivity: [], expectedBg: [], bgWithIobOnly: [], bgWithCobOnly: [] },
    [effectCurve, lastTimestamp, lastGlucose, showPredictions, showEffectCurve, showEffectAreas, showEffectiveBg]
  )

  const treatmentData = useMemo(() =>
    showTreatments && treatments.length > 0
      ? transformTreatments(treatments, allReadings)
      : { insulin: { x: [], y: [], emojis: [] }, carbs: { x: [], y: [], emojis: [] } },
    [treatments, allReadings, showTreatments]
  )

  // Transform historical IOB/COB/POB for continuous plotting over time
  const historicalIobCobData = useMemo(() =>
    (showEffectCurve || showEffectAreas || showEffectiveBg || showIobCobLines) && historicalIobCob.length > 0
      ? transformHistoricalIobCob(historicalIobCob)
      : { timestamps: [], iob: [], cob: [], pob: [], bgPressure: [], actualBg: [] },
    [historicalIobCob, showEffectCurve, showEffectAreas, showEffectiveBg, showIobCobLines]
  )

  // DEBUG: Log data to find BG Pressure discrepancy
  useEffect(() => {
    if (historicalIobCobData.timestamps.length > 0 && glucoseData.x.length > 0) {
      console.log('=== BG PRESSURE DEBUG ===')
      console.log('glucoseData points:', glucoseData.x.length)
      console.log('historicalIobCobData points:', historicalIobCobData.timestamps.length)

      // Find mismatches
      const glucoseMap = new Map(glucoseData.x.map((t, i) => [t.getTime(), glucoseData.y[i]]))
      const histMap = new Map(historicalIobCobData.timestamps.map((t, i) => [t.getTime(), {
        actualBg: historicalIobCobData.actualBg[i],
        bgPressure: historicalIobCobData.bgPressure[i],
        iob: historicalIobCobData.iob[i],
        cob: historicalIobCobData.cob[i]
      }]))

      // Check for points in glucose but not in historical
      let missingInHist = 0
      glucoseMap.forEach((_, time) => {
        if (!histMap.has(time)) missingInHist++
      })
      console.log('Points in glucose but NOT in historicalIobCob:', missingInHist)

      // Log first few points with big pressure differences
      console.log('Sample data (first 5 with large pressure diff):')
      let count = 0
      historicalIobCobData.timestamps.forEach((t, i) => {
        const diff = Math.abs(historicalIobCobData.bgPressure[i] - historicalIobCobData.actualBg[i])
        if (diff > 50 && count < 5) {
          console.log(`  ${t.toLocaleTimeString()}: BG=${historicalIobCobData.actualBg[i]}, Pressure=${historicalIobCobData.bgPressure[i]}, IOB=${historicalIobCobData.iob[i].toFixed(2)}, COB=${historicalIobCobData.cob[i]}`)
          count++
        }
      })

      // POB Debug logging
      const maxPob = Math.max(...historicalIobCobData.pob)
      const hasPobData = historicalIobCobData.pob.some(p => p > 0)
      console.log(`[POB Debug] POB values: max=${maxPob.toFixed(1)}g, hasPobData=${hasPobData}, count=${historicalIobCobData.pob.length}`)
      if (hasPobData) {
        const nonZeroPob = historicalIobCobData.pob.filter(p => p > 0)
        console.log(`[POB Debug] Non-zero POB values: ${nonZeroPob.length} points, sample: ${nonZeroPob.slice(0, 5).map(p => p.toFixed(1)).join(', ')}`)
      }
    }
  }, [historicalIobCobData, glucoseData])

  // Calculate domains - include historical bgPressure to ensure shaded area is fully visible
  const glucoseDomain = useMemo(() =>
    calculateGlucoseDomain(
      glucoseData.y,
      effectData.expectedBg,  // Predicted BG Pressure values
      [
        ...effectData.bgWithIobOnly,
        ...effectData.bgWithCobOnly,
        ...historicalIobCobData.bgPressure  // Include historical BG pressure for full shading visibility
      ],
      criticalLow,
      criticalHigh
    ),
    [glucoseData.y, effectData.expectedBg, effectData.bgWithIobOnly, effectData.bgWithCobOnly, historicalIobCobData.bgPressure, criticalLow, criticalHigh]
  )

  const iobDomain = useMemo(() =>
    calculateIobDomain([...effectData.iob, ...historicalIobCobData.iob]),
    [effectData.iob, historicalIobCobData.iob]
  )

  const cobDomain = useMemo(() =>
    calculateCobDomain([...effectData.cob, ...historicalIobCobData.cob]),
    [effectData.cob, historicalIobCobData.cob]
  )

  const pobDomain = useMemo(() =>
    calculatePobDomain(historicalIobCobData.pob),
    [historicalIobCobData.pob]
  )

  // Build Plotly traces
  const traces: Data[] = useMemo(() => {
    const data: Data[] = []

    // 1. Glucose readings - main line
    if (glucoseData.x.length > 0) {
      data.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Glucose',
        x: glucoseData.x,
        y: glucoseData.y,
        line: { color: '#00c6ff', width: 2 },
        yaxis: 'y',
        connectgaps: false,
        hovertemplate: '%{y:.0f} mg/dL<extra>Glucose</extra>',
      })
    }

    // 2. Predicted BG (orange line) - future glucose based on IOB/COB physics
    // This is the MAIN prediction line - where BG is heading based on current trend + IOB/COB
    if (showPredictions && effectData.timestamps.length > 0 && effectData.expectedBg.length > 0) {
      // Skip the t=0 point from effectData since it duplicates lastTimestamp
      // effectData.timestamps[0] is at lastTimestamp + 0 minutes = lastTimestamp
      // We want: [lastTimestamp with lastGlucose] -> [future points with expectedBg]
      const futureTimestamps = effectData.timestamps.filter((_, i) => i > 0 || effectData.timestamps.length === 1)
      const futureExpectedBg = effectData.expectedBg.filter((_, i) => i > 0 || effectData.expectedBg.length === 1)

      // Start from current glucose for smooth connection
      const predictionX: Date[] = [lastTimestamp, ...futureTimestamps]
      const predictionY: number[] = [lastGlucose, ...futureExpectedBg]

      data.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Predicted BG',
        x: predictionX,
        y: predictionY,
        line: { color: '#f97316', width: 2.5, dash: 'solid' },  // Orange, prominent
        yaxis: 'y',
        hovertemplate: '%{y:.0f} mg/dL<extra>Predicted BG</extra>',
      })
    }

    // 3. Model-based physics prediction (LEARNED from this person's BG data) - PROMINENT
    if (showDualPredictions && modelPredictions.length > 0) {
      const modelX = modelPredictions.map(p =>
        new Date(lastTimestamp.getTime() + p.minutesAhead * 60000)
      )
      const modelY = modelPredictions.map(p => p.predictedBg)

      data.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Learned Model',
        x: modelX,
        y: modelY,
        line: { color: '#10b981', width: 2.5, dash: 'solid' },  // Green, solid, prominent
        yaxis: 'y',
        hovertemplate: '%{y:.0f} mg/dL<extra>Learned Model (IOB/COB)</extra>',
      })
    }

    // 4. Hardcoded physics prediction (standard textbook) - FADED for comparison
    if (showDualPredictions && hardcodedPredictions.length > 0) {
      const hardcodedX = hardcodedPredictions.map(p =>
        new Date(lastTimestamp.getTime() + p.minutesAhead * 60000)
      )
      const hardcodedY = hardcodedPredictions.map(p => p.predictedBg)

      data.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Standard Formula',
        x: hardcodedX,
        y: hardcodedY,
        line: { color: 'rgba(156, 163, 175, 0.4)', width: 1.5, dash: 'dot' },  // Gray, faded, dotted
        yaxis: 'y',
        hovertemplate: '%{y:.0f} mg/dL<extra>Standard Formula (Textbook)</extra>',
      })
    }

    // NOTE: Sections 5-6c removed - they duplicated section 9's IOB/COB lines
    // Section 9 (showIobCobLines) is the single source of truth for IOB/COB decay lines

    // 7. BG Pressure gradient area - spans BOTH past and future as ONE CONTINUOUS line
    // BLUE for normal (pressure below glucose), TAN/ORANGE when pressure is ABOVE glucose (rising BG)
    // Uses smoothing to soften edges while preserving detail
    // CRITICAL: Future BG Pressure CONTINUES from past BG Pressure (same accumulation)
    if (showEffectiveBg) {
      // Color constants - use subtle cyan gradient for all BG Pressure shading
      // Single color looks cleaner than trying to switch between colors
      const PRESSURE_FILL = 'rgba(34, 211, 238, 0.15)'  // Subtle cyan to match the app theme

      // Get the last past BG Pressure to continue from
      let lastPastBgPressure = lastGlucose  // Default to current BG

      // === PAST GRADIENT (historical) ===
      // BG Pressure comes from backend via historicalIobCobData.bgPressure
      // CRITICAL: Use glucoseData (visible BG line) for the BG side of shading
      // This ensures the shading aligns EXACTLY with the visible BG line
      const pastTimestamps: Date[] = []
      const pastGlucose: number[] = []
      const pastPressure: number[] = []

      if (historicalIobCobData.timestamps.length > 0 && historicalIobCobData.bgPressure.length > 0) {
        // Create a lookup map from timestamp to bgPressure
        // Use a 5-minute tolerance for timestamp matching (CGM readings are every 5 min)
        const pressureMap = new Map<number, number>()
        historicalIobCobData.timestamps.forEach((ts, i) => {
          const bgPressure = historicalIobCobData.bgPressure[i]
          if (Number.isFinite(bgPressure)) {
            // Round to nearest minute for matching
            pressureMap.set(Math.round(ts.getTime() / 60000), bgPressure)
          }
        })

        // Use glucoseData (the visible BG line) as the source of truth
        // Match each glucose point to its corresponding pressure value
        if (glucoseData.x && glucoseData.x.length > 0) {
          glucoseData.x.forEach((ts, i) => {
            const bg = glucoseData.y[i]
            if (!Number.isFinite(bg) || bg <= 0) return

            // Look up pressure at this timestamp (with 1-minute tolerance)
            const tsKey = Math.round(ts.getTime() / 60000)
            const pressure = pressureMap.get(tsKey)

            if (pressure !== undefined) {
              pastTimestamps.push(ts)
              pastGlucose.push(bg)
              pastPressure.push(pressure)
            }
          })
        }

        // Get the last past BG Pressure for continuity with future
        if (pastPressure.length > 0) {
          lastPastBgPressure = pastPressure[pastPressure.length - 1]
        }

        // ALWAYS add past BG Pressure traces (even if empty) to ensure future traces
        // have the correct previous trace for fill:tonexty
        // First trace: Pressure line (one edge of shaded area)
        data.push({
          type: 'scatter',
          mode: 'lines',
          name: 'BG Pressure (past)',
          x: pastTimestamps.length > 0 ? pastTimestamps : [lastTimestamp],
          y: pastTimestamps.length > 0 ? pastPressure : [lastPastBgPressure],
          line: { color: 'rgba(0,0,0,0)', width: 0, shape: 'linear' },
          yaxis: 'y',
          showlegend: false,
          hoverinfo: 'skip',
        })

        // Second trace: Glucose - must match BG line exactly
        data.push({
          type: 'scatter',
          mode: 'lines',
          name: 'BG Pressure Area (past)',
          x: pastTimestamps.length > 0 ? pastTimestamps : [lastTimestamp],
          y: pastTimestamps.length > 0 ? pastGlucose : [lastGlucose],
          line: { color: 'rgba(0,0,0,0)', width: 0, shape: 'linear' },
          fill: 'tonexty',
          fillcolor: PRESSURE_FILL,
          yaxis: 'y',
          showlegend: false,
          hoverinfo: 'skip',
        })
      }

      // === FUTURE GRADIENT ===
      // Fill between Predicted BG (orange line) and future BG Pressure (invisible)
      // IMPORTANT: Future BG Pressure continues from lastPastBgPressure for smooth transition
      if (effectData.timestamps.length > 0 && effectData.expectedBg.length > 0) {
        // Calculate future BG Pressure that continues from the past pressure
        // The pressure decays toward the expected BG over time (IOB/COB effects diminish)
        const futurePressureY: number[] = effectData.timestamps.map((_, i) => {
          // Get effects at this time point
          const iobEffect = effectData.iobEffect[i] || 0
          const cobEffect = effectData.cobEffect[i] || 0
          const expectedBg = effectData.expectedBg[i]

          // Calculate decay factor - pressure converges toward expectedBg over time
          // At t=0, use lastPastBgPressure; at t→∞, converge to expectedBg
          const minutesAhead = (effectData.timestamps[i].getTime() - lastTimestamp.getTime()) / 60000
          const decayFactor = Math.exp(-minutesAhead / 60)  // ~37% at 60 min

          // BG Pressure is a blend between past pressure and expected BG
          // Plus an offset based on the active IOB/COB forces
          const netEffect = iobEffect + cobEffect
          const pressureOffset = netEffect * 0.5 * decayFactor  // Pressure shows the force direction

          // Blend: start from past pressure, decay toward expected + offset
          const blendedPressure = lastPastBgPressure * decayFactor + expectedBg * (1 - decayFactor) + pressureOffset

          return blendedPressure
        })

        // Include connection point at lastTimestamp for smooth transition from past
        // Skip the t=0 point from effectData since it duplicates lastTimestamp
        const futureTimestamps = effectData.timestamps.filter((_, i) => i > 0 || effectData.timestamps.length === 1)
        const futureExpectedBgFiltered = effectData.expectedBg.filter((_, i) => i > 0 || effectData.expectedBg.length === 1)
        const futurePressureFiltered = futurePressureY.filter((_, i) => i > 0 || futurePressureY.length === 1)

        const futureX: Date[] = [lastTimestamp, ...futureTimestamps]
        const futurePredictedY: number[] = [lastGlucose, ...futureExpectedBgFiltered]
        // Start future pressure at lastPastBgPressure for continuity
        const futurePressureFull: number[] = [lastPastBgPressure, ...futurePressureFiltered]

        // First trace: Pressure line (one edge of shaded area)
        // Use LINEAR shape - no smoothing to ensure fill aligns correctly
        data.push({
          type: 'scatter',
          mode: 'lines',
          name: 'BG Pressure (future)',
          x: futureX,
          y: futurePressureFull,  // No smoothing - use raw values
          line: { color: 'rgba(0,0,0,0)', width: 0, shape: 'linear' },
          yaxis: 'y',
          showlegend: false,
          hoverinfo: 'skip',
        })

        // Second trace: Predicted BG
        data.push({
          type: 'scatter',
          mode: 'lines',
          name: 'BG Pressure Area (future)',
          x: futureX,
          y: futurePredictedY,
          line: { color: 'rgba(0,0,0,0)', width: 0, shape: 'linear' },
          fill: 'tonexty',
          fillcolor: PRESSURE_FILL,
          yaxis: 'y',
          showlegend: false,
          hoverinfo: 'skip',
        })
      }
    }

    // 8. BG Pressure boundary lines - REMOVED
    // The IOB Floor and COB Ceiling are now integrated into the main BG Pressure calculation
    // No separate dashed lines needed - BG Pressure shows the combined effect

    // 9. IOB/COB decay lines on secondary Y-axes (conditional addition like BG Pressure)
    // Historical: SOLID, smooth spline - FADED to not overpower BG line
    // Future: DOTTED, smooth spline - FADED
    // Only add traces when toggle is ON (same pattern as BG Pressure which works)
    if (showIobCobLines && historicalIobCobData.timestamps.length > 0) {
      // Historical IOB line (solid, smooth, translucent)
      data.push({
        type: 'scatter',
        mode: 'lines',
        name: 'IOB',
        x: historicalIobCobData.timestamps,
        y: historicalIobCobData.iob,
        line: { color: 'rgba(59, 130, 246, 0.35)', width: 1.5, shape: 'spline', smoothing: 1.0 },
        yaxis: 'y2',
        hovertemplate: '%{y:.2f}U<extra>IOB</extra>',
      })

      // Historical COB line (solid, smooth, translucent)
      data.push({
        type: 'scatter',
        mode: 'lines',
        name: 'COB',
        x: historicalIobCobData.timestamps,
        y: historicalIobCobData.cob,
        line: { color: 'rgba(34, 197, 94, 0.35)', width: 1.5, shape: 'spline', smoothing: 1.0 },
        yaxis: 'y3',
        hovertemplate: '%{y:.0f}g<extra>COB</extra>',
      })

      // Historical POB line (solid, smooth, faint purple - delayed protein effect)
      // POB has a much longer duration (5h vs 3h for COB) and delayed onset (2h vs 5-15min for COB)
      const hasPobData = historicalIobCobData.pob.some(p => p > 0)
      console.log(`[POB Trace] showIobCobLines=${showIobCobLines}, hasPobData=${hasPobData}, pobCount=${historicalIobCobData.pob.length}`)
      if (hasPobData) {
        console.log(`[POB Trace] Adding POB trace with ${historicalIobCobData.pob.filter(p => p > 0).length} non-zero points`)
        data.push({
          type: 'scatter',
          mode: 'lines',
          name: 'POB',
          x: historicalIobCobData.timestamps,
          y: historicalIobCobData.pob,
          line: { color: 'rgba(168, 85, 247, 0.35)', width: 1.5, shape: 'spline', smoothing: 1.0 },  // Faint purple, matches IOB/COB opacity
          yaxis: 'y4',  // New axis for POB
          hovertemplate: '%{y:.0f}g<extra>POB (Protein)</extra>',
        })
      } else {
        console.log('[POB Trace] Not adding POB trace - no protein data > 0')
      }
    }

    // Future IOB/COB lines (dotted, smooth, translucent)
    // Include connection point at lastTimestamp for smooth transition from historical
    if (showIobCobLines && effectData.timestamps.length > 0) {
      // Get the last historical IOB/COB values for continuity
      const lastHistIob = historicalIobCobData.iob.length > 0
        ? historicalIobCobData.iob[historicalIobCobData.iob.length - 1]
        : effectData.iob[0]
      const lastHistCob = historicalIobCobData.cob.length > 0
        ? historicalIobCobData.cob[historicalIobCobData.cob.length - 1]
        : effectData.cob[0]

      // Add connection point at lastTimestamp for smooth transition
      const futureIobX = [lastTimestamp, ...effectData.timestamps]
      const futureIobY = [lastHistIob, ...effectData.iob]
      const futureCobX = [lastTimestamp, ...effectData.timestamps]
      const futureCobY = [lastHistCob, ...effectData.cob]

      data.push({
        type: 'scatter',
        mode: 'lines',
        name: 'IOB (projected)',
        x: futureIobX,
        y: futureIobY,
        line: { color: 'rgba(59, 130, 246, 0.25)', width: 1.5, dash: 'dot', shape: 'spline', smoothing: 1.0 },
        yaxis: 'y2',
        hovertemplate: '%{y:.2f}U<extra>IOB (projected)</extra>',
      })

      // Future COB line (dotted, smooth, translucent)
      data.push({
        type: 'scatter',
        mode: 'lines',
        name: 'COB (projected)',
        x: futureCobX,
        y: futureCobY,
        line: { color: 'rgba(34, 197, 94, 0.25)', width: 1.5, dash: 'dot', shape: 'spline', smoothing: 1.0 },
        yaxis: 'y3',
        hovertemplate: '%{y:.0f}g<extra>COB (projected)</extra>',
      })

      // Future POB line (dotted, smooth, translucent purple)
      // Calculate decay from last historical POB to ensure smooth transition
      const lastHistPob = historicalIobCobData.pob.length > 0
        ? historicalIobCobData.pob[historicalIobCobData.pob.length - 1]
        : 0

      if (lastHistPob > 0) {
        // Calculate POB decay from lastHistPob using same formula as backend
        // POB half-life is 90 minutes (slower than COB due to delayed protein digestion)
        const pobHalfLife = 90  // minutes
        const futurePobX = [lastTimestamp, ...effectData.timestamps]
        const futurePobY = futurePobX.map((ts) => {
          const minutesAhead = (ts.getTime() - lastTimestamp.getTime()) / 60000
          // Exponential decay: POB(t) = POB(0) * e^(-0.693 * t / halfLife)
          return lastHistPob * Math.exp(-0.693 * minutesAhead / pobHalfLife)
        })

        data.push({
          type: 'scatter',
          mode: 'lines',
          name: 'POB (projected)',
          x: futurePobX,
          y: futurePobY,
          line: { color: 'rgba(168, 85, 247, 0.25)', width: 1.5, dash: 'dot', shape: 'spline', smoothing: 1.0 },
          yaxis: 'y4',
          hovertemplate: '%{y:.0f}g<extra>POB (projected)</extra>',
        })
      }
    }

    // 10. Treatment markers - Insulin (syringe emoji)
    if (showTreatments && treatmentData.insulin.x.length > 0) {
      data.push({
        type: 'scatter',
        mode: 'text',
        name: 'Insulin',
        x: treatmentData.insulin.x,
        y: treatmentData.insulin.y,
        text: treatmentData.insulin.emojis,
        textfont: { size: 16 },
        textposition: 'middle center',
        yaxis: 'y',
        hovertemplate: '💉 Insulin at %{x|%H:%M}<extra></extra>',
        showlegend: false,
      })
    }

    // 10. Treatment markers - Carbs (food emoji based on description)
    if (showTreatments && treatmentData.carbs.x.length > 0) {
      data.push({
        type: 'scatter',
        mode: 'text',
        name: 'Carbs',
        x: treatmentData.carbs.x,
        y: treatmentData.carbs.y,
        text: treatmentData.carbs.emojis,
        textfont: { size: 16 },
        textposition: 'middle center',
        yaxis: 'y',
        hovertemplate: '%{text} Carbs at %{x|%H:%M}<extra></extra>',
        showlegend: false,
      })
    }

    return data
  }, [
    glucoseData,
    effectData,
    treatmentData,
    historicalIobCobData,
    showPredictions,
    showTreatments,
    showEffectCurve,
    showEffectAreas,
    showEffectiveBg,
    showIobCobLines,
    showDualPredictions,
    modelPredictions,
    hardcodedPredictions,
    lastTimestamp,
    lastGlucose,
  ])

  // Calculate initial x-axis range based on timeRange using GOLDEN RATIO
  // Golden ratio (φ ≈ 1.618): Past data 61.8%, Future predictions 38.2%
  // CENTER ON LAST READING (not browser time) to ensure current BG is visible
  const xAxisRange = useMemo(() => {
    const hours = timeRangeHours[timeRange]

    // Use the last reading timestamp as the anchor point
    // This ensures the current BG is always visible in the chart
    const anchorTime = lastTimestamp.getTime()

    // Golden ratio: φ = 1.618, or 61.8% past / 38.2% future
    const GOLDEN_RATIO = 1.618
    const PAST_RATIO = 1 / GOLDEN_RATIO  // ≈ 0.618
    const FUTURE_RATIO = 1 - PAST_RATIO  // ≈ 0.382

    // Calculate total time window based on selected hours
    const totalMinutes = hours * 60
    const pastMinutes = totalMinutes * PAST_RATIO
    const futureMinutes = totalMinutes * FUTURE_RATIO

    // Ensure at least 60 min future for predictions
    const actualFutureMinutes = Math.max(futureMinutes, 60)

    // Start: go back by past portion from last reading
    const startTime = new Date(anchorTime - pastMinutes * 60 * 1000)

    // End: extend into future for predictions/effects
    const endTime = new Date(anchorTime + actualFutureMinutes * 60 * 1000)

    // Return millisecond timestamps for Plotly - displays in LOCAL timezone
    // (ISO strings cause UTC display issues)
    return [startTime.getTime(), endTime.getTime()]
  }, [timeRange, lastTimestamp]) // Recalculate when timeRange or lastTimestamp changes

  // Build layout
  const layout: Partial<Layout> = useMemo(() => {
    const shapes: Layout['shapes'] = []

    // Target range shading
    if (showTargetRange) {
      shapes.push({
        type: 'rect',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: targetLow,
        y1: targetHigh,
        yref: 'y',
        fillcolor: 'rgba(0, 198, 255, 0.1)',
        line: { width: 0 },
        layer: 'below',
      })
    }

    // Critical threshold lines
    shapes.push(
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: criticalLow,
        y1: criticalLow,
        yref: 'y',
        line: { color: '#dc2626', width: 1, dash: 'dash' },
        layer: 'below',
      },
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: criticalHigh,
        y1: criticalHigh,
        yref: 'y',
        line: { color: '#dc2626', width: 1, dash: 'dash' },
        layer: 'below',
      }
    )

    // Target threshold lines
    shapes.push(
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: targetLow,
        y1: targetLow,
        yref: 'y',
        line: { color: 'rgba(0, 198, 255, 0.5)', width: 1, dash: 'dash' },
        layer: 'below',
      },
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: targetHigh,
        y1: targetHigh,
        yref: 'y',
        line: { color: 'rgba(0, 198, 255, 0.5)', width: 1, dash: 'dash' },
        layer: 'below',
      }
    )

    const hasSecondaryAxes = showEffectCurve || showEffectAreas || showIobCobLines

    return {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'rgba(30, 41, 59, 0.5)',
      font: { color: '#9ca3af', size: 11 },
      margin: { l: 45, r: hasSecondaryAxes ? 80 : 10, t: 10, b: 40 },
      showlegend: false,
      hovermode: 'x unified',
      dragmode: 'pan', // Enable drag-to-pan for horizontal scrolling
      // Preserve user zoom/pan state across re-renders, reset on timeRange or toggle change
      // Include toggle states so Plotly resets trace visibility when toggles change
      uirevision: `${timeRange}-${showIobCobLines}-${showEffectiveBg}`,
      shapes,
      xaxis: {
        type: 'date',
        autorange: false,
        range: xAxisRange,
        tickformat: timeRange === '24hr' ? '%H:%M' : '%-I:%M %p',
        gridcolor: 'rgba(55, 65, 81, 0.5)',
        tickfont: { color: '#6b7280' },
        showline: false,
        zeroline: false,
        fixedrange: false, // Allow zooming/panning on x-axis
        // Note: uirevision should preserve user zoom state
        // Range is only applied when uirevision changes (i.e., timeRange changes)
      },
      yaxis: {
        title: { text: '' },
        range: glucoseDomain,
        gridcolor: 'rgba(55, 65, 81, 0.5)',
        tickfont: { color: '#6b7280' },
        showline: false,
        zeroline: false,
        side: 'left',
        fixedrange: true, // Lock y-axis (only pan horizontally)
      },
      yaxis2: hasSecondaryAxes ? {
        title: { text: 'IOB (U)', font: { color: '#3b82f6', size: 10 } },
        tickfont: { color: '#3b82f6', size: 10 },
        overlaying: 'y',
        side: 'right',
        range: iobDomain,
        showgrid: false,
        showline: false,
        zeroline: false,
        fixedrange: true,
      } : undefined,
      yaxis3: hasSecondaryAxes ? {
        title: { text: 'COB (g)', font: { color: '#22c55e', size: 10 } },
        tickfont: { color: '#22c55e', size: 10 },
        overlaying: 'y',
        side: 'right',
        position: 0.95,
        range: cobDomain,
        showgrid: false,
        showline: false,
        zeroline: false,
        anchor: 'free',
        fixedrange: true,
      } : undefined,
      yaxis4: hasSecondaryAxes ? {
        title: { text: 'POB (g)', font: { color: '#a855f7', size: 10 } },
        tickfont: { color: '#a855f7', size: 10 },
        overlaying: 'y',
        side: 'right',
        position: 0.90,  // Position between COB axis and chart edge
        range: pobDomain,
        showgrid: false,
        showline: false,
        zeroline: false,
        anchor: 'free',
        fixedrange: true,
      } : undefined,
    }
  }, [
    showTargetRange,
    targetLow,
    targetHigh,
    criticalLow,
    criticalHigh,
    glucoseDomain,
    iobDomain,
    cobDomain,
    pobDomain,
    showEffectCurve,
    showEffectAreas,
    showEffectiveBg,  // Added for uirevision dependency
    showIobCobLines,
    timeRange,
    xAxisRange,
  ])

  // Custom fullscreen button for modebar
  const fullscreenButton = useMemo(() => ({
    name: 'fullscreen',
    title: 'Toggle Fullscreen',
    icon: {
      width: 1024,
      height: 1024,
      path: 'M128 32H32C14.3 32 0 46.3 0 64v96c0 17.7 14.3 32 32 32s32-14.3 32-32V96h64c17.7 0 32-14.3 32-32s-14.3-32-32-32zM64 864v-64c0-17.7-14.3-32-32-32s-32 14.3-32 32v96c0 17.7 14.3 32 32 32h96c17.7 0 32-14.3 32-32s-14.3-32-32-32H64zM992 32h-96c-17.7 0-32 14.3-32 32s14.3 32 32 32h64v64c0 17.7 14.3 32 32 32s32-14.3 32-32V64c0-17.7-14.3-32-32-32zM992 800c-17.7 0-32 14.3-32 32v64h-64c-17.7 0-32 14.3-32 32s14.3 32 32 32h96c17.7 0 32-14.3 32-32v-96c0-17.7-14.3-32-32-32z',
      transform: 'matrix(1 0 0 1 0 0)'
    },
    click: toggleFullscreen
  }), [toggleFullscreen])

  // Plotly config
  const config: Partial<Config> = useMemo(() => ({
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'zoomIn2d', 'zoomOut2d'],
    modeBarButtonsToAdd: [fullscreenButton as any],
    scrollZoom: 'x', // Enable horizontal scroll zoom only
    displaylogo: false,
  }), [fullscreenButton])

  // Empty state
  if (allReadings.length === 0) {
    return (
      <div className={cn('flex items-center justify-center h-80 text-gray-500', className)}>
        No glucose data available
      </div>
    )
  }

  return (
    <div
      ref={chartContainerRef}
      className={cn(
        'w-full transition-all relative',
        isFullscreen
          ? 'fixed inset-0 z-50 h-screen bg-slate-900 p-4'
          : 'h-80',
        className
      )}
    >
      {/* Close button for CSS fullscreen mode */}
      {isFullscreen && (
        <button
          onClick={toggleFullscreen}
          className="absolute top-2 right-2 z-10 p-2 bg-slate-700 hover:bg-slate-600 rounded-full text-white transition-colors"
          aria-label="Exit fullscreen"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      )}
      <ChartErrorBoundary fallbackHeight={isFullscreen ? "h-full" : "h-80"}>
        <Plot
          key={`plot-${showIobCobLines}-${showEffectiveBg}-${timeRange}-${isFullscreen}`}
          data={traces}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
          revision={showIobCobLines ? 1 : 0}
        />
      </ChartErrorBoundary>
    </div>
  )
}

export default PlotlyGlucoseChart
