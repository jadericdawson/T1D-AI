/**
 * Metabolic State Panel Component
 * Displays 2 gauges showing metabolic status:
 * 1. Absorption Gauge (Speed) - carb absorption timing
 * 2. Combined State (Overall status) - sick/resistant/normal/sensitive
 *
 * Note: ISF and ICR have their own dedicated gauge cards on the dashboard
 */
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Clock, Activity, AlertTriangle, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { trainingApi, MetabolicStateResponse } from '@/lib/api'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

const fadeIn = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
}

interface MetabolicStatePanelProps {
  className?: string
  onRefresh?: () => void
  userId?: string  // For viewing shared accounts
}

export function MetabolicStatePanel({ className, onRefresh, userId }: MetabolicStatePanelProps) {
  const [data, setData] = useState<MetabolicStateResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchMetabolicState = async () => {
    try {
      setIsLoading(true)
      setError(null)
      const response = await trainingApi.getMetabolicState(false, undefined, userId)
      setData(response)
    } catch (err) {
      console.error('Failed to fetch metabolic state:', err)
      setError('Failed to load metabolic state')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchMetabolicState()
  }, [userId])

  const handleRefresh = () => {
    fetchMetabolicState()
    onRefresh?.()
  }

  if (error) {
    return (
      <div className={cn('glass-card p-4', className)}>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan" />
            Metabolic State
          </h3>
          <button
            onClick={handleRefresh}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
        <p className="text-sm text-red-400">{error}</p>
      </div>
    )
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className={cn('glass-card p-4', className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan" />
          Metabolic State
        </h3>
        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className={cn(
            'text-gray-400 hover:text-white transition-colors',
            isLoading && 'animate-spin'
          )}
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* 2-Gauge Grid - Absorption and Combined State */}
      <div className="grid grid-cols-2 gap-3">
        {/* Absorption Gauge */}
        <AbsorptionMiniGauge
          timeToTPeak={data?.absorption_time_to_peak}
          baseline={data?.absorption_baseline_time_to_peak}
          deviation={data?.absorption_deviation_percent}
          state={data?.absorption_state}
          isLoading={isLoading}
        />

        {/* Combined State */}
        <CombinedStateGauge
          state={data?.state ?? 'normal'}
          description={data?.state_description ?? ''}
          confidence={data?.confidence ?? 0}
          isLoading={isLoading}
        />
      </div>

      {/* State Description */}
      {data && data.state !== 'normal' && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className={cn(
            'mt-3 p-2 rounded text-xs',
            data.state === 'sick' && 'bg-red-500/10 text-red-400 border border-red-500/20',
            data.state === 'resistant' && 'bg-orange-500/10 text-orange-400 border border-orange-500/20',
            data.state === 'sensitive' && 'bg-green-500/10 text-green-400 border border-green-500/20',
            data.state === 'very_sensitive' && 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
          )}
        >
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            <p>{data.state_description}</p>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}

// ==================== Mini Gauge Components ====================

interface MiniGaugeProps {
  label: string
  icon: React.ReactNode
  value: number | string
  deviation: number
  color: string
  tooltip: string
  isLoading?: boolean
}

function MiniGauge({ label, icon, value, deviation, color, tooltip, isLoading }: MiniGaugeProps) {
  // Clamp deviation to ±50% for display
  const clampedDeviation = Math.max(-50, Math.min(50, deviation))
  const needleRotation = (clampedDeviation / 50) * 45

  const getDeviationColor = () => {
    if (Math.abs(deviation) < 5) return 'text-gray-400'
    if (deviation < -20) return 'text-red-500'
    if (deviation < -10) return 'text-orange-500'
    if (deviation > 20) return 'text-green-500'
    if (deviation > 10) return 'text-teal-400'
    return 'text-yellow-400'
  }

  if (isLoading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-2 text-center">
        <div className={cn('mx-auto mb-1 w-4 h-4', color)}>{icon}</div>
        <div className="h-8 flex items-center justify-center">
          <span className="text-gray-500">—</span>
        </div>
        <p className="text-gray-600 text-[10px] uppercase tracking-wider">{label}</p>
      </div>
    )
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="bg-slate-800/50 rounded-lg p-2 text-center cursor-help">
            <div className={cn('mx-auto mb-1 w-4 h-4', color)}>{icon}</div>

            {/* Mini Gauge */}
            <div className="relative h-8 mb-1">
              <svg viewBox="0 0 60 30" className="w-full h-full">
                {/* Background arc */}
                <path
                  d="M 5 28 A 25 25 0 0 1 55 28"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  className="text-slate-700"
                />
                {/* Gradient arc */}
                <defs>
                  <linearGradient id={`miniGauge-${label}`} x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#ef4444" />
                    <stop offset="50%" stopColor="#a855f7" />
                    <stop offset="100%" stopColor="#22c55e" />
                  </linearGradient>
                </defs>
                <path
                  d="M 5 28 A 25 25 0 0 1 55 28"
                  fill="none"
                  stroke={`url(#miniGauge-${label})`}
                  strokeWidth="2"
                  opacity="0.5"
                />
              </svg>

              {/* Needle */}
              <div
                className="absolute left-1/2 bottom-0.5 w-0.5 h-5 origin-bottom transition-transform duration-500"
                style={{
                  transform: `translateX(-50%) rotate(${needleRotation}deg)`,
                  background: `linear-gradient(to top, ${
                    deviation < -10 ? '#ef4444' : deviation > 10 ? '#22c55e' : '#a855f7'
                  }, transparent)`
                }}
              />

              {/* Center pivot */}
              <div className="absolute left-1/2 bottom-0 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-slate-600 border border-purple-500" />
            </div>

            {/* Value */}
            <div className="text-sm font-bold font-orbitron">
              <span className={getDeviationColor()}>{value}</span>
            </div>

            {/* Label */}
            <p className="text-gray-500 text-[10px] uppercase tracking-wider">{label}</p>

            {/* Deviation indicator */}
            {Math.abs(deviation) >= 5 && (
              <p className={cn('text-[10px]', getDeviationColor())}>
                {deviation > 0 ? '+' : ''}{deviation.toFixed(0)}%
              </p>
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent className="glass-card border-gray-700 max-w-xs">
          <p className="text-sm">{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

// Absorption Mini Gauge
function AbsorptionMiniGauge({
  timeToTPeak,
  baseline,
  deviation,
  state,
  isLoading,
}: {
  timeToTPeak?: number
  baseline?: number
  deviation?: number
  state?: string
  isLoading?: boolean
}) {
  const hasData = timeToTPeak !== undefined && timeToTPeak !== null

  if (isLoading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-2 text-center">
        <div className="text-amber-500 mx-auto mb-1 w-4 h-4"><Clock className="w-4 h-4" /></div>
        <div className="h-8 flex items-center justify-center">
          <span className="text-gray-500">—</span>
        </div>
        <p className="text-gray-600 text-[10px] uppercase tracking-wider">Absorb</p>
      </div>
    )
  }

  if (!hasData) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="bg-slate-800/50 rounded-lg p-2 text-center cursor-help">
              <div className="text-amber-500 mx-auto mb-1 w-4 h-4"><Clock className="w-4 h-4" /></div>
              <div className="h-8 flex items-center justify-center">
                <span className="text-gray-500 text-sm">—</span>
              </div>
              <p className="text-gray-500 text-[10px] uppercase tracking-wider">Absorb</p>
              <p className="text-gray-600 text-[10px]">No data</p>
            </div>
          </TooltipTrigger>
          <TooltipContent className="glass-card border-gray-700">
            <p className="text-sm">Insufficient meal data to calculate absorption rate.</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  return (
    <MiniGauge
      label="Absorb"
      icon={<Clock className="w-4 h-4" />}
      value={`${Math.round(timeToTPeak)}m`}
      deviation={-(deviation ?? 0)} // Invert: slow = positive deviation = negative on gauge
      color="text-amber-500"
      tooltip={`Time-to-peak: ${Math.round(timeToTPeak ?? 60)} min (baseline: ${Math.round(baseline ?? 60)} min). ${
        state === 'very_slow' ? 'Very slow absorption - may indicate gastroparesis or illness.' :
        state === 'slow' ? 'Slower absorption than usual.' :
        state === 'fast' ? 'Faster absorption than usual.' :
        state === 'very_fast' ? 'Very fast absorption - likely liquids or high-GI foods.' :
        'Normal absorption rate.'
      }`}
      isLoading={isLoading}
    />
  )
}

// Combined State Gauge
function CombinedStateGauge({
  state,
  description,
  confidence,
  isLoading,
}: {
  state: string
  description: string
  confidence: number
  isLoading?: boolean
}) {
  const getStateConfig = () => {
    switch (state) {
      case 'sick':
        return { color: 'text-red-500', bgColor: 'bg-red-500/20', label: 'SICK', icon: '🤒' }
      case 'resistant':
        return { color: 'text-orange-500', bgColor: 'bg-orange-500/20', label: 'RESIST', icon: '⚠️' }
      case 'sensitive':
        return { color: 'text-green-500', bgColor: 'bg-green-500/20', label: 'SENS', icon: '✨' }
      case 'very_sensitive':
        return { color: 'text-emerald-400', bgColor: 'bg-emerald-500/20', label: 'V.SENS', icon: '⚡' }
      default:
        return { color: 'text-gray-400', bgColor: 'bg-slate-700/50', label: 'OK', icon: '✓' }
    }
  }

  const config = getStateConfig()

  if (isLoading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-2 text-center">
        <div className="text-cyan mx-auto mb-1 w-4 h-4"><Activity className="w-4 h-4" /></div>
        <div className="h-8 flex items-center justify-center">
          <span className="text-gray-500">—</span>
        </div>
        <p className="text-gray-600 text-[10px] uppercase tracking-wider">State</p>
      </div>
    )
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="bg-slate-800/50 rounded-lg p-2 text-center cursor-help">
            <div className="text-cyan mx-auto mb-1 w-4 h-4"><Activity className="w-4 h-4" /></div>

            {/* State Indicator */}
            <div className={cn(
              'h-8 flex items-center justify-center rounded-full mx-auto w-8',
              config.bgColor
            )}>
              <span className="text-lg">{config.icon}</span>
            </div>

            {/* Label */}
            <p className={cn('text-sm font-bold', config.color)}>{config.label}</p>
            <p className="text-gray-500 text-[10px] uppercase tracking-wider">State</p>

            {/* Confidence */}
            {confidence < 0.5 && (
              <p className="text-gray-600 text-[10px]">Low conf.</p>
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent className="glass-card border-gray-700 max-w-xs">
          <p className="text-sm">{description || 'Metabolic state is normal.'}</p>
          <p className="text-xs text-gray-500 mt-1">Confidence: {Math.round(confidence * 100)}%</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export default MetabolicStatePanel
