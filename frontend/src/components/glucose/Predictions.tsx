/**
 * Predictions Display Component
 * Shows predicted BG values from the orange line (IOB/COB physics model)
 */
import { motion } from 'framer-motion'
import { TrendingUp, Clock } from 'lucide-react'
import { cn, getGlucoseColor } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'

// Effect curve point with expectedBg (the orange line)
interface EffectPoint {
  minutesAhead: number
  iobEffect: number
  cobEffect: number
  netEffect: number
  remainingIOB?: number
  remainingCOB?: number
  expectedBg?: number
  bgWithIobOnly?: number
  bgWithCobOnly?: number
}

// TFT prediction with uncertainty bands
interface TFTPrediction {
  timestamp: string
  horizon: number  // 30, 45, 60 minutes
  value: number    // Median (50th percentile)
  lower: number    // 10th percentile
  upper: number    // 90th percentile
  tftDelta?: number  // TFT modifier delta (mg/dL)
}

interface PredictionsProps {
  linear?: number[]
  lstm?: number[] | null
  tft?: TFTPrediction[]  // TFT predictions with uncertainty
  effectCurve?: EffectPoint[]  // Orange line (expectedBg) values
  horizons?: number[] // [5, 10, 15] minutes
  accuracy?: {
    linearWins: number
    lstmWins: number
    totalComparisons: number
  }
  modelAvailable?: boolean
  tftAvailable?: boolean  // Whether TFT model is loaded
  dataReadings?: number   // Current number of glucose readings
  dataRequired?: number   // Required readings for ML predictions (default 24)
  className?: string
}

const fadeIn = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
}

export function PredictionsCard({
  effectCurve,
  tft,
  tftAvailable = false,
  dataReadings,
  dataRequired = 24,
  className,
}: PredictionsProps) {
  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className={cn('glass-card', className)}
    >
      <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
        <TrendingUp className="w-5 h-5 text-orange-500" />
        Predicted BG
      </h3>

      <div className="space-y-4">
        {/* Predicted BG from effect curve (orange line values) */}
        {effectCurve && effectCurve.length > 0 ? (
          <PredictedBGRow effectCurve={effectCurve} tftPredictions={tft} />
        ) : tft && tft.length > 0 ? (
          // Fallback to TFT if no effect curve
          <TFTPredictionRow predictions={tft} />
        ) : (
          <div className="flex items-center justify-between text-gray-500">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              <span>Predicted BG</span>
            </div>
            <span className="text-sm italic">
              {tftAvailable ? 'Insufficient data' : 'Loading...'}
            </span>
          </div>
        )}

        {/* Data Requirements Note */}
        {dataReadings !== undefined && dataRequired !== undefined && dataReadings < dataRequired && (
          <DataRequirementsNote current={dataReadings} required={dataRequired} />
        )}
      </div>
    </motion.div>
  )
}

// Predicted BG row showing orange line values at key horizons
function PredictedBGRow({ effectCurve, tftPredictions }: { effectCurve: EffectPoint[], tftPredictions?: TFTPrediction[] }) {
  // Find values at key horizons: 15, 30, 60, 120 min
  const displayHorizons = [15, 30, 60, 120]

  // Create a map of TFT deltas by horizon
  const tftDeltaMap = new Map<number, number>()
  if (tftPredictions) {
    tftPredictions.forEach(p => {
      if (p.tftDelta !== undefined && p.tftDelta !== null) {
        tftDeltaMap.set(p.horizon, p.tftDelta)
      }
    })
  }

  // Guard against empty effectCurve
  if (!effectCurve || effectCurve.length === 0) {
    return (
      <div className="flex items-center justify-between text-gray-500">
        <span>Predicted BG</span>
        <span className="text-sm italic">No predictions available</span>
      </div>
    )
  }

  const predictions = displayHorizons.map(horizon => {
    // Find the closest effect point to this horizon
    const point = effectCurve.find(p => p.minutesAhead === horizon) ||
      (effectCurve.length > 0
        ? effectCurve.reduce((prev, curr) =>
            Math.abs(curr.minutesAhead - horizon) < Math.abs(prev.minutesAhead - horizon) ? curr : prev
          )
        : null)
    // Get TFT delta for this horizon (or closest)
    let tftDelta = tftDeltaMap.get(horizon)
    if (tftDelta === undefined && tftPredictions && tftPredictions.length > 0) {
      // Try to find closest TFT horizon
      const closestTft = tftPredictions.reduce((prev, curr) =>
        Math.abs(curr.horizon - horizon) < Math.abs(prev.horizon - horizon) ? curr : prev
      )
      if (closestTft && Math.abs(closestTft.horizon - horizon) <= 15) {
        tftDelta = closestTft.tftDelta
      }
    }
    return {
      horizon,
      value: point?.expectedBg ?? 0,
      actualHorizon: point?.minutesAhead ?? horizon,
      tftDelta
    }
  }).filter(p => p.value > 0)

  if (predictions.length === 0) {
    return (
      <div className="flex items-center justify-between text-gray-500">
        <span>Predicted BG</span>
        <span className="text-sm italic">No predictions available</span>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-orange-400">
          <TrendingUp className="w-4 h-4" />
          <span className="font-medium">Predicted BG</span>
          <Badge variant="outline" className="text-[10px] px-1 py-0 border-orange-500/30 text-orange-400">
            IOB+COB
          </Badge>
        </div>
        <div className="flex items-center gap-1 flex-wrap justify-end">
          {predictions.map((pred, i) => (
            <span key={pred.horizon} className="flex items-center">
              <span
                className="font-semibold text-sm"
                style={{ color: getGlucoseColor(pred.value) }}
                title={`+${pred.actualHorizon}min: ${Math.round(pred.value)} mg/dL${pred.tftDelta != null ? ` (TFT: ${pred.tftDelta >= 0 ? '+' : ''}${pred.tftDelta.toFixed(1)})` : ''}`}
              >
                {Math.round(pred.value)}
              </span>
              {/* Show TFT delta next to prediction */}
              {pred.tftDelta != null && (
                <span
                  className={cn(
                    "text-[10px] ml-0.5",
                    pred.tftDelta >= 0 ? "text-yellow-400" : "text-green-400"
                  )}
                  title={`TFT adjusted physics by ${pred.tftDelta >= 0 ? '+' : ''}${pred.tftDelta.toFixed(1)} mg/dL`}
                >
                  ({pred.tftDelta >= 0 ? '+' : ''}{pred.tftDelta.toFixed(0)})
                </span>
              )}
              {i < predictions.length - 1 && (
                <span className="text-gray-600 mx-1">/</span>
              )}
            </span>
          ))}
        </div>
      </div>
      {/* Time labels */}
      <div className="flex justify-end gap-2 text-[10px] text-gray-500">
        {predictions.map((pred, i) => (
          <span key={pred.horizon}>
            +{pred.horizon}m
            {i < predictions.length - 1 && <span className="mx-1">/</span>}
          </span>
        ))}
      </div>
    </div>
  )
}

// Data requirements note component
function DataRequirementsNote({ current, required }: { current: number; required: number }) {
  const progress = Math.min((current / required) * 100, 100)
  const minutesNeeded = (required - current) * 5 // 5 min per reading

  return (
    <div className="pt-3 border-t border-gray-700/50">
      <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Clock className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <p className="text-sm text-yellow-400 font-medium">
              ML Predictions Require More Data
            </p>
            <p className="text-xs text-gray-400 mt-1">
              LSTM/TFT models need {required} readings ({required * 5} min of data).
              Currently have {current} readings.
            </p>
            <div className="mt-2">
              <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-yellow-500 transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                ~{minutesNeeded} min until ML predictions available
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Legacy TFT prediction row (fallback)
function TFTPredictionRow({ predictions }: { predictions: TFTPrediction[] }) {
  // Sort by horizon
  const sorted = [...predictions].sort((a, b) => a.horizon - b.horizon)

  // Key horizons to display (show 4-5 to keep compact)
  const displayHorizons = [15, 30, 60, 120]
  const displayPreds = sorted.filter(p => displayHorizons.includes(p.horizon))

  // If no matching horizons, show first 4
  const toShow = displayPreds.length > 0 ? displayPreds : sorted.slice(0, 4)

  // Get 60-min prediction for uncertainty display
  const pred60 = sorted.find(p => p.horizon === 60) || sorted[Math.min(sorted.length - 1, 5)]

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-orange-400">
          <TrendingUp className="w-4 h-4" />
          <span className="font-medium">Predicted BG</span>
          <Badge variant="outline" className="text-[10px] px-1 py-0 border-orange-500/30 text-orange-400">
            TFT
          </Badge>
        </div>
        <div className="flex items-center gap-1 flex-wrap justify-end">
          {toShow.map((pred, i) => (
            <span key={pred.horizon} className="flex items-center">
              <span
                className="font-semibold text-sm"
                style={{ color: getGlucoseColor(pred.value) }}
                title={`+${pred.horizon}min: ${Math.round(pred.value)} (${Math.round(pred.lower)}-${Math.round(pred.upper)})`}
              >
                {Math.round(pred.value)}
              </span>
              {i < toShow.length - 1 && (
                <span className="text-gray-600 mx-1">/</span>
              )}
            </span>
          ))}
        </div>
      </div>
      {/* Time labels */}
      <div className="flex justify-end gap-2 text-[10px] text-gray-500">
        {toShow.map((pred, i) => (
          <span key={pred.horizon}>
            +{pred.horizon}m
            {i < toShow.length - 1 && <span className="mx-1">/</span>}
          </span>
        ))}
      </div>
      {/* Uncertainty range at 60 min */}
      {pred60 && (
        <div className="flex justify-between items-center">
          <span className="text-xs text-gray-500">
            1hr uncertainty:
          </span>
          <span className="text-xs text-gray-400">
            {Math.round(pred60.lower)}-{Math.round(pred60.upper)} mg/dL
          </span>
        </div>
      )}
    </div>
  )
}

export default PredictionsCard
