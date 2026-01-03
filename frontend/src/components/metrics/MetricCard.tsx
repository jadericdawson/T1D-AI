/**
 * Metric Card Components
 * Displays IOB, COB, ISF, and Dose metrics
 */
import { motion } from 'framer-motion'
import { Droplet, Pill, Brain, TrendingUp, Syringe, Activity, AlertTriangle } from 'lucide-react'
import { cn, formatDecimal } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

interface MetricCardProps {
  icon: React.ReactNode
  label: string
  value: string
  unit?: string
  color: string
  highlighted?: boolean
  tooltip?: string
  subValue?: string
  onClick?: () => void
}

const fadeIn = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
}

export function MetricCard({
  icon,
  label,
  value,
  unit,
  color,
  highlighted = false,
  tooltip,
  subValue,
  onClick,
}: MetricCardProps) {
  const content = (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      whileHover={{ scale: 1.02 }}
      whileTap={onClick ? { scale: 0.98 } : undefined}
      onClick={onClick}
      className={cn(
        'glass-card text-center py-4 cursor-default',
        highlighted && 'border-cyan/50 animate-pulse-glow',
        onClick && 'cursor-pointer hover:border-cyan/30'
      )}
    >
      <div className={cn('mx-auto mb-2', color)}>{icon}</div>
      <div className="flex items-baseline justify-center gap-1">
        <span className={cn('text-2xl font-orbitron font-bold', color)}>
          {value}
        </span>
        {unit && <span className="text-gray-500 text-sm">{unit}</span>}
      </div>
      <p className="text-gray-500 text-xs uppercase tracking-wider mt-1">
        {label}
      </p>
      {subValue && (
        <p className="text-gray-600 text-xs mt-1">{subValue}</p>
      )}
    </motion.div>
  )

  if (tooltip) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{content}</TooltipTrigger>
          <TooltipContent className="glass-card border-gray-700">
            <p className="text-sm">{tooltip}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  return content
}

// Pre-configured metric cards
interface MetricsData {
  iob: number
  cob: number
  isf: number
  recommendedDose: number
  effectiveBg?: number
}

export function IOBCard({ value, tooltip }: { value: number; tooltip?: string }) {
  return (
    <MetricCard
      icon={<Droplet className="w-5 h-5" />}
      label="IOB"
      value={formatDecimal(value)}
      unit="U"
      color="text-blue-500"
      tooltip={tooltip || `Insulin on Board: ${formatDecimal(value)} units of active insulin`}
    />
  )
}

export function COBCard({ value, tooltip }: { value: number; tooltip?: string }) {
  return (
    <MetricCard
      icon={<Pill className="w-5 h-5" />}
      label="COB"
      value={formatDecimal(value, 0)}
      unit="g"
      color="text-teal-500"
      tooltip={tooltip || `Carbs on Board: ${formatDecimal(value, 0)}g of active carbs`}
    />
  )
}

export function ISFCard({ value, tooltip }: { value: number; tooltip?: string }) {
  return (
    <MetricCard
      icon={<Brain className="w-5 h-5" />}
      label="ISF"
      value={formatDecimal(value)}
      color="text-purple-500"
      tooltip={tooltip || `Insulin Sensitivity: 1U lowers BG by ${formatDecimal(value)} mg/dL`}
    />
  )
}

export function DoseCard({
  value,
  targetBg = 100,
  tooltip,
  highlighted = true,
}: {
  value: number
  targetBg?: number
  tooltip?: string
  highlighted?: boolean
}) {
  return (
    <MetricCard
      icon={<TrendingUp className="w-5 h-5" />}
      label="Dose"
      value={formatDecimal(value, 2)}
      unit="U"
      color="text-cyan"
      highlighted={highlighted && value > 0}
      tooltip={tooltip || `Recommended correction: ${formatDecimal(value, 2)}U to reach ${targetBg} mg/dL`}
      subValue={value > 0 ? `to reach ${targetBg}` : 'No correction needed'}
    />
  )
}

// Metrics grid component
export function MetricsGrid({
  iob,
  cob,
  isf,
  recommendedDose,
  className,
}: MetricsData & { className?: string }) {
  return (
    <div className={cn('grid grid-cols-2 gap-4', className)}>
      <IOBCard value={iob} />
      <COBCard value={cob} />
      <ISFCard value={isf} />
      <DoseCard value={recommendedDose} />
    </div>
  )
}

// Effective BG card (shows BG adjusted for IOB/COB)
export function EffectiveBGCard({
  currentBg,
  effectiveBg,
  iobEffect,
  cobEffect,
}: {
  currentBg: number
  effectiveBg: number
  iobEffect: number
  cobEffect: number
}) {
  const difference = effectiveBg - currentBg
  const isRising = cobEffect > iobEffect

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className="glass-card p-4"
    >
      <div className="flex items-center gap-2 mb-3">
        <Activity className="w-5 h-5 text-cyan" />
        <h4 className="text-sm font-medium text-gray-300">Effective BG</h4>
      </div>

      <div className="flex items-baseline gap-2 mb-2">
        <span className="text-3xl font-orbitron font-bold text-white">
          {Math.round(effectiveBg)}
        </span>
        <span className="text-gray-500">mg/dL</span>
        {difference !== 0 && (
          <span className={cn(
            'text-sm',
            isRising ? 'text-yellow-500' : 'text-green-500'
          )}>
            ({isRising ? '+' : ''}{Math.round(difference)})
          </span>
        )}
      </div>

      <div className="text-xs text-gray-500 space-y-1">
        <p>Current: {currentBg} + COB effect: +{Math.round(cobEffect)} - IOB effect: -{Math.round(iobEffect)}</p>
      </div>
    </motion.div>
  )
}

// Warning card for high IOB or predicted lows
export function WarningCard({
  type,
  message,
  severity = 'warning',
}: {
  type: string
  message: string
  severity?: 'info' | 'warning' | 'critical'
}) {
  const severityColors = {
    info: 'border-blue-500/30 bg-blue-500/10 text-blue-400',
    warning: 'border-yellow-500/30 bg-yellow-500/10 text-yellow-400',
    critical: 'border-red-500/30 bg-red-500/10 text-red-400',
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className={cn(
        'glass-card p-3 flex items-start gap-3 border-l-4',
        severityColors[severity]
      )}
    >
      <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
      <div>
        <p className="font-medium text-sm">{type}</p>
        <p className="text-xs opacity-80">{message}</p>
      </div>
    </motion.div>
  )
}

export default MetricCard
