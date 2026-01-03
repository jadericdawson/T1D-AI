/**
 * Current Glucose Display Component
 * Shows current BG value with trend and range badge
 */
import { motion } from 'framer-motion'
import { Badge } from '@/components/ui/badge'
import { cn, getGlucoseColor, getGlucoseRange, getTrendArrow, formatTime } from '@/lib/utils'

interface CurrentGlucoseProps {
  value: number
  trend: string | null
  timestamp: string
  className?: string
}

const fadeIn = {
  hidden: { opacity: 0, scale: 0.95 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.3 } },
}

export function CurrentGlucose({
  value,
  trend,
  timestamp,
  className,
}: CurrentGlucoseProps) {
  const glucoseColor = getGlucoseColor(value)
  const glucoseRange = getGlucoseRange(value)
  const trendArrow = getTrendArrow(trend || 'Flat')

  const rangeLabels: Record<string, string> = {
    'critical-low': 'CRITICAL LOW',
    'low': 'LOW',
    'in-range': 'In Range',
    'high': 'HIGH',
    'critical-high': 'CRITICAL HIGH',
  }

  const rangeBadgeStyles: Record<string, string> = {
    'critical-low': 'bg-red-500/20 text-red-500 border-red-500/30',
    'low': 'bg-orange-500/20 text-orange-500 border-orange-500/30',
    'in-range': 'bg-cyan/20 text-cyan border-cyan/30',
    'high': 'bg-yellow-500/20 text-yellow-500 border-yellow-500/30',
    'critical-high': 'bg-red-500/20 text-red-500 border-red-500/30',
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className={cn('glass-card text-center py-8', className)}
    >
      <p className="text-gray-400 text-sm mb-2">Current Glucose</p>

      <div className="flex items-center justify-center gap-2">
        <motion.span
          key={value}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="glucose-value"
          style={{ color: glucoseColor }}
        >
          {value}
        </motion.span>
        <motion.span
          key={trend}
          initial={{ x: -10, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="trend-arrow"
          style={{ color: glucoseColor }}
        >
          {trendArrow}
        </motion.span>
      </div>

      <p className="text-gray-500 text-sm mt-2">
        mg/dL &bull; {formatTime(timestamp)}
      </p>

      <Badge
        className={cn(
          'mt-4 border',
          rangeBadgeStyles[glucoseRange]
        )}
      >
        {rangeLabels[glucoseRange]}
      </Badge>

      {/* Urgent indicator for critical values */}
      {(glucoseRange === 'critical-low' || glucoseRange === 'critical-high') && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          className="mt-3 text-red-500 text-sm font-medium"
        >
          {glucoseRange === 'critical-low' ? 'Treat Hypoglycemia Now!' : 'Check Ketones'}
        </motion.div>
      )}
    </motion.div>
  )
}

// Compact version for header/navbar
export function CurrentGlucoseCompact({
  value,
  trend,
  className,
}: {
  value: number
  trend: string | null
  className?: string
}) {
  const glucoseColor = getGlucoseColor(value)
  const trendArrow = getTrendArrow(trend || 'Flat')

  return (
    <div className={cn('flex items-center gap-1', className)}>
      <span
        className="text-2xl font-bold font-orbitron"
        style={{ color: glucoseColor }}
      >
        {value}
      </span>
      <span
        className="text-xl"
        style={{ color: glucoseColor }}
      >
        {trendArrow}
      </span>
    </div>
  )
}

export default CurrentGlucose
