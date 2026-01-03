/**
 * Predictions Display Component
 * Shows linear and LSTM predictions with accuracy comparison
 */
import { motion } from 'framer-motion'
import { TrendingUp, Cpu, LineChart, Trophy, Clock } from 'lucide-react'
import { cn, getGlucoseColor, formatDecimal } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'

interface PredictionsProps {
  linear: number[]
  lstm: number[] | null
  horizons?: number[] // [5, 10, 15] minutes
  accuracy?: {
    linearWins: number
    lstmWins: number
    totalComparisons: number
  }
  modelAvailable?: boolean
  className?: string
}

const fadeIn = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
}

export function PredictionsCard({
  linear,
  lstm,
  horizons = [5, 10, 15],
  accuracy,
  modelAvailable = true,
  className,
}: PredictionsProps) {
  const winner = accuracy
    ? accuracy.lstmWins > accuracy.linearWins
      ? 'lstm'
      : 'linear'
    : null

  const totalComparisons = accuracy?.totalComparisons || 0
  const linearPct = totalComparisons > 0
    ? (accuracy!.linearWins / totalComparisons) * 100
    : 50
  const lstmPct = totalComparisons > 0
    ? (accuracy!.lstmWins / totalComparisons) * 100
    : 50

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className={cn('glass-card', className)}
    >
      <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
        <TrendingUp className="w-5 h-5 text-purple-500" />
        Predictions
        {!modelAvailable && (
          <Badge variant="outline" className="ml-2 text-xs border-yellow-500/30 text-yellow-500">
            Linear Only
          </Badge>
        )}
      </h3>

      <div className="space-y-4">
        {/* Linear Predictions */}
        <PredictionRow
          icon={<LineChart className="w-4 h-4" />}
          label="Linear"
          values={linear}
          horizons={horizons}
          color="text-gray-400"
          isWinner={winner === 'linear'}
        />

        {/* LSTM Predictions */}
        {lstm && lstm.length > 0 ? (
          <PredictionRow
            icon={<Cpu className="w-4 h-4" />}
            label="LSTM"
            values={lstm}
            horizons={horizons}
            color="text-purple-400"
            isWinner={winner === 'lstm'}
          />
        ) : (
          <div className="flex items-center justify-between text-gray-500">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              <span>LSTM</span>
            </div>
            <span className="text-sm italic">Model not available</span>
          </div>
        )}

        {/* Accuracy Comparison */}
        {accuracy && totalComparisons > 0 && (
          <div className="pt-3 border-t border-gray-700/50">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-gray-500 flex items-center gap-1">
                <Trophy className="w-4 h-4 text-yellow-500" />
                Accuracy (wins)
              </span>
              <span className="text-gray-400">
                Lin: {accuracy.linearWins} / LSTM: {accuracy.lstmWins}
              </span>
            </div>

            {/* Visual progress bar */}
            <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="absolute left-0 top-0 h-full bg-gray-500 transition-all"
                style={{ width: `${linearPct}%` }}
              />
              <div
                className="absolute right-0 top-0 h-full bg-purple-500 transition-all"
                style={{ width: `${lstmPct}%` }}
              />
            </div>

            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Linear {linearPct.toFixed(0)}%</span>
              <span>LSTM {lstmPct.toFixed(0)}%</span>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  )
}

// Individual prediction row
function PredictionRow({
  icon,
  label,
  values,
  horizons,
  color,
  isWinner,
}: {
  icon: React.ReactNode
  label: string
  values: number[]
  horizons: number[]
  color: string
  isWinner?: boolean
}) {
  return (
    <div className="flex items-center justify-between">
      <div className={cn('flex items-center gap-2', color)}>
        {icon}
        <span>{label}</span>
        {isWinner && (
          <Trophy className="w-3 h-3 text-yellow-500" />
        )}
      </div>
      <div className="flex items-center gap-1">
        {values.map((val, i) => (
          <span key={i} className="flex items-center">
            <span
              className={cn(
                'font-medium',
                color,
                label === 'LSTM' && 'font-semibold'
              )}
              style={{ color: getGlucoseColor(val) }}
            >
              {Math.round(val)}
            </span>
            {i < values.length - 1 && (
              <span className="text-gray-600 mx-1">/</span>
            )}
          </span>
        ))}
      </div>
    </div>
  )
}

// Compact predictions for inline display
export function PredictionsCompact({
  linear,
  lstm,
  className,
}: {
  linear: number[]
  lstm: number[] | null
  className?: string
}) {
  const values = lstm || linear
  const label = lstm ? 'LSTM' : 'Linear'

  return (
    <div className={cn('flex items-center gap-2 text-sm', className)}>
      <Clock className="w-4 h-4 text-gray-500" />
      <span className="text-gray-400">+15m:</span>
      <span
        className={cn(
          'font-medium',
          lstm ? 'text-purple-400' : 'text-gray-400'
        )}
        style={{ color: getGlucoseColor(values[2]) }}
      >
        {Math.round(values[2])} mg/dL
      </span>
      <span className="text-gray-600">({label})</span>
    </div>
  )
}

// Prediction timeline view
export function PredictionTimeline({
  currentBg,
  linear,
  lstm,
  horizons = [5, 10, 15],
  className,
}: {
  currentBg: number
  linear: number[]
  lstm: number[] | null
  horizons?: number[]
  className?: string
}) {
  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className={cn('glass-card p-4', className)}
    >
      <h4 className="text-sm font-medium text-gray-400 mb-4 flex items-center gap-2">
        <Clock className="w-4 h-4" />
        Prediction Timeline
      </h4>

      <div className="flex items-center justify-between">
        {/* Current */}
        <div className="text-center">
          <div
            className="text-2xl font-bold font-orbitron"
            style={{ color: getGlucoseColor(currentBg) }}
          >
            {currentBg}
          </div>
          <div className="text-xs text-gray-500 mt-1">Now</div>
        </div>

        {/* Arrow */}
        <div className="flex-1 mx-4 border-t border-dashed border-gray-600 relative">
          <div className="absolute right-0 top-1/2 -translate-y-1/2 w-0 h-0 border-t-4 border-t-transparent border-b-4 border-b-transparent border-l-4 border-l-gray-600" />
        </div>

        {/* Predictions */}
        {horizons.map((horizon, i) => {
          const linearVal = linear[i]
          const lstmVal = lstm?.[i]
          const displayVal = lstmVal ?? linearVal

          return (
            <div key={horizon} className="text-center">
              <div
                className={cn(
                  'text-xl font-bold font-orbitron',
                  lstmVal ? 'text-purple-400' : 'text-gray-400'
                )}
                style={{ color: getGlucoseColor(displayVal) }}
              >
                {Math.round(displayVal)}
              </div>
              <div className="text-xs text-gray-500 mt-1">+{horizon}m</div>
            </div>
          )
        })}
      </div>
    </motion.div>
  )
}

export default PredictionsCard
