/**
 * Metric Card Components
 * Displays IOB, COB, ISF, and Dose metrics
 */
import { motion } from 'framer-motion'
import { Droplet, Pill, Brain, TrendingUp, Activity, AlertTriangle, Beef, Utensils } from 'lucide-react'
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
  isLoading?: boolean
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
  isLoading = false,
}: MetricCardProps) {
  // Show dash when loading
  const displayValue = isLoading ? '—' : value
  const displaySubValue = isLoading ? undefined : subValue

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
        highlighted && !isLoading && 'border-cyan/50 animate-pulse-glow',
        onClick && 'cursor-pointer hover:border-cyan/30'
      )}
    >
      <div className={cn('mx-auto mb-2', isLoading ? 'text-gray-500' : color)}>{icon}</div>
      <div className="flex items-baseline justify-center gap-1">
        <span className={cn('text-2xl font-orbitron font-bold', isLoading ? 'text-gray-500' : color)}>
          {displayValue}
        </span>
        {unit && !isLoading && <span className="text-gray-500 text-sm">{unit}</span>}
      </div>
      <p className="text-gray-500 text-xs uppercase tracking-wider mt-1">
        {label}
      </p>
      {displaySubValue && (
        <p className="text-gray-600 text-xs mt-1">{displaySubValue}</p>
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
  pob: number  // Protein on Board
  isf: number
  recommendedDose: number
  effectiveBg?: number
}

export function IOBCard({ value, tooltip, isLoading }: { value: number; tooltip?: string; isLoading?: boolean }) {
  return (
    <MetricCard
      icon={<Droplet className="w-5 h-5" />}
      label="IOB"
      value={formatDecimal(value)}
      unit="U"
      color="text-blue-500"
      tooltip={tooltip || `Insulin on Board: ${formatDecimal(value)} units of active insulin`}
      isLoading={isLoading}
    />
  )
}

export function COBCard({ value, tooltip, isLoading }: { value: number; tooltip?: string; isLoading?: boolean }) {
  return (
    <MetricCard
      icon={<Pill className="w-5 h-5" />}
      label="COB"
      value={formatDecimal(value, 0)}
      unit="g"
      color="text-teal-500"
      tooltip={tooltip || `Carbs on Board: ${formatDecimal(value, 0)}g of active carbs`}
      isLoading={isLoading}
    />
  )
}

export function POBCard({ value, tooltip, isLoading }: { value: number; tooltip?: string; isLoading?: boolean }) {
  return (
    <MetricCard
      icon={<Beef className="w-5 h-5" />}
      label="POB"
      value={formatDecimal(value, 0)}
      unit="g"
      color="text-orange-500"
      tooltip={tooltip || `Protein on Board: ${formatDecimal(value, 0)}g of active protein (2-5h delayed effect)`}
      isLoading={isLoading}
    />
  )
}

export function ISFCard({
  value,
  tooltip,
  source,
  isLoading,
}: {
  value: number
  tooltip?: string
  source?: 'learned' | 'default' | 'manual'
  isLoading?: boolean
}) {
  const sourceLabel = source === 'learned' ? '🤖' : source === 'manual' ? '✏️' : ''
  return (
    <MetricCard
      icon={<Brain className="w-5 h-5" />}
      label="ISF"
      value={formatDecimal(value)}
      color="text-purple-500"
      tooltip={tooltip || `Insulin Sensitivity: 1U lowers BG by ${formatDecimal(value)} mg/dL${source === 'learned' ? ' (AI-learned)' : ''}`}
      subValue={sourceLabel ? `${sourceLabel} ${source}` : undefined}
      isLoading={isLoading}
    />
  )
}

// ISF Gauge Card - Shows baseline vs current ISF with needle indicator
export function ISFGaugeCard({
  baselineIsf,
  currentIsf,
  deviation,
  source,
  isLoading,
  confidence,
}: {
  baselineIsf: number
  currentIsf?: number | null
  deviation?: number | null  // Percentage: positive = more sensitive, negative = more resistant
  source?: 'learned' | 'default' | 'manual'
  isLoading?: boolean
  confidence?: number
}) {
  // Clamp deviation to ±50% for display
  const clampedDeviation = deviation ? Math.max(-50, Math.min(50, deviation)) : 0

  // Calculate needle rotation: -45deg (resistant) to +45deg (sensitive)
  // Map -50% to +50% deviation to -45 to +45 degrees
  const needleRotation = (clampedDeviation / 50) * 45

  // Color based on deviation
  const getDeviationColor = () => {
    if (!deviation || Math.abs(deviation) < 5) return 'text-gray-400'
    if (deviation < -20) return 'text-red-500'  // Very resistant (sick)
    if (deviation < -10) return 'text-orange-500'  // Moderately resistant
    if (deviation > 20) return 'text-green-500'  // Very sensitive
    if (deviation > 10) return 'text-teal-400'  // Moderately sensitive
    return 'text-yellow-400'  // Slightly off
  }

  const getStatusLabel = () => {
    if (!deviation || Math.abs(deviation) < 5) return 'Normal'
    if (deviation < -20) return 'Resistant'
    if (deviation < -10) return 'Slightly Resistant'
    if (deviation > 20) return 'Very Sensitive'
    if (deviation > 10) return 'Sensitive'
    return deviation < 0 ? 'Slightly Resistant' : 'Slightly Sensitive'
  }

  const sourceLabel = source === 'learned' ? '🤖' : source === 'manual' ? '✏️' : ''

  // Show dash when loading OR when no learned data (source is 'default')
  const showDash = isLoading || source === 'default'

  if (showDash) {
    return (
      <MetricCard
        icon={<Brain className="w-5 h-5" />}
        label="ISF"
        value="—"
        color="text-purple-500"
        tooltip={isLoading ? "Loading..." : "No learned ISF yet - needs more data"}
        isLoading={isLoading}
      />
    )
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className="glass-card text-center py-4"
    >
      {/* Icon */}
      <div className="text-purple-500 mx-auto mb-2">
        <Brain className="w-5 h-5 mx-auto" />
      </div>

      {/* Gauge Container */}
      <div className="relative h-16 mb-2">
        {/* Gauge Background Arc */}
        <svg viewBox="0 0 100 50" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 10 45 A 40 40 0 0 1 90 45"
            fill="none"
            stroke="currentColor"
            strokeWidth="6"
            className="text-slate-700"
          />
          {/* Gradient sections - resistant (red) to normal (purple) to sensitive (green) */}
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="30%" stopColor="#f97316" />
              <stop offset="50%" stopColor="#a855f7" />
              <stop offset="70%" stopColor="#14b8a6" />
              <stop offset="100%" stopColor="#22c55e" />
            </linearGradient>
          </defs>
          <path
            d="M 10 45 A 40 40 0 0 1 90 45"
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="4"
            opacity="0.6"
          />
          {/* Center tick mark */}
          <line x1="50" y1="10" x2="50" y2="16" stroke="#a855f7" strokeWidth="2" />
          {/* Side labels */}
          <text x="8" y="48" fontSize="6" fill="#ef4444" className="font-mono">−</text>
          <text x="89" y="48" fontSize="6" fill="#22c55e" className="font-mono">+</text>
        </svg>

        {/* Needle */}
        <div
          className="absolute left-1/2 bottom-1 w-0.5 h-10 origin-bottom transition-transform duration-500 ease-out"
          style={{
            transform: `translateX(-50%) rotate(${needleRotation}deg)`,
            background: `linear-gradient(to top, ${
              deviation && deviation < -10 ? '#ef4444' :
              deviation && deviation > 10 ? '#22c55e' : '#a855f7'
            }, transparent)`
          }}
        >
          {/* Needle tip */}
          <div
            className={cn(
              "absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full",
              deviation && deviation < -10 ? "bg-red-500" :
              deviation && deviation > 10 ? "bg-green-500" : "bg-purple-500"
            )}
          />
        </div>

        {/* Center pivot */}
        <div className="absolute left-1/2 bottom-0 -translate-x-1/2 w-3 h-3 rounded-full bg-slate-600 border-2 border-purple-500" />
      </div>

      {/* Values */}
      <div className="flex items-baseline justify-center gap-1">
        <span className={cn('text-2xl font-orbitron font-bold', getDeviationColor())}>
          {currentIsf ? formatDecimal(currentIsf) : formatDecimal(baselineIsf)}
        </span>
        {currentIsf && currentIsf !== baselineIsf && (
          <span className="text-gray-500 text-xs">
            (base: {formatDecimal(baselineIsf)})
          </span>
        )}
      </div>

      {/* Label */}
      <p className="text-gray-500 text-xs uppercase tracking-wider mt-1">
        ISF
      </p>

      {/* Status & Deviation */}
      <div className="mt-1 space-y-0.5">
        <p className={cn("text-xs font-medium", getDeviationColor())}>
          {getStatusLabel()}
          {deviation && Math.abs(deviation) >= 5 && (
            <span className="ml-1">
              ({deviation > 0 ? '+' : ''}{deviation.toFixed(0)}%)
            </span>
          )}
        </p>
        {sourceLabel && (
          <p className="text-gray-600 text-xs">{sourceLabel} {source}</p>
        )}
        {confidence !== undefined && confidence < 0.5 && (
          <p className="text-gray-600 text-xs italic">Low confidence</p>
        )}
      </div>
    </motion.div>
  )
}

export function DoseCard({
  value,
  targetBg = 100,
  tooltip,
  highlighted = true,
  carbDose,
  proteinDoseNow,
  proteinDoseLater,
  icrSource: _icrSource,
  pirSource: _pirSource,
  onClick,
  isLoading,
}: {
  value: number
  targetBg?: number
  tooltip?: string
  highlighted?: boolean
  carbDose?: number
  proteinDoseNow?: number     // Immediate protein dose
  proteinDoseLater?: number   // Delayed protein dose (decays to NOW over time)
  icrSource?: 'learned' | 'default' | 'manual'
  pirSource?: 'learned' | 'default' | 'manual'
  onClick?: () => void        // Opens DoseBreakdownModal
  isLoading?: boolean
}) {
  // Calculate protein total for backward compatibility
  const proteinDose = (proteinDoseNow ?? 0) + (proteinDoseLater ?? 0)

  // Backend calculates protein NOW/LATER based on insulin-protein timing overlap:
  // NOW = portion of remaining protein BG effect that insulin given now can cover
  // LATER = protein effect that occurs after insulin wears off (~5hr DIA)
  // No additional scaling needed - backend does the physics-based calculation

  // If we have breakdown data, show it
  const hasBreakdown = carbDose !== undefined || proteinDose > 0
  const totalDoseNow = value + (carbDose ?? 0) + (proteinDoseNow ?? 0)
  const totalDoseLater = proteinDoseLater ?? 0
  const totalDose = totalDoseNow + totalDoseLater
  const doseNowValue = hasBreakdown ? totalDoseNow : value

  // Build subtext - show what's included in the dose NOW
  let subText = ''
  if (doseNowValue > 0) {
    subText = `to 100`
    // Show protein NOW included note
    if (proteinDoseNow && proteinDoseNow > 0) {
      subText += ` (+${formatDecimal(proteinDoseNow, 1)}U P)`
    }
  } else {
    subText = 'No correction needed'
  }

  // Build detailed tooltip showing what's included
  const tooltipParts = [`Dose NOW: ${formatDecimal(doseNowValue, 2)}U`]
  if (hasBreakdown) {
    const breakdown = []
    if (value > 0) breakdown.push(`correction: ${formatDecimal(value, 1)}U`)
    if (carbDose && carbDose > 0) breakdown.push(`carbs: ${formatDecimal(carbDose, 1)}U`)
    if (proteinDoseNow && proteinDoseNow > 0) breakdown.push(`protein: ${formatDecimal(proteinDoseNow, 1)}U`)
    if (breakdown.length > 0) tooltipParts.push(`(${breakdown.join(' + ')})`)
  }
  if (totalDoseLater > 0) tooltipParts.push(`LATER: ${formatDecimal(totalDoseLater, 2)}U protein`)
  tooltipParts.push(`→ target ${targetBg} mg/dL`)

  return (
    <MetricCard
      icon={<TrendingUp className="w-5 h-5" />}
      label="Dose"
      value={formatDecimal(doseNowValue, 2)}
      unit="U"
      color="text-cyan"
      highlighted={highlighted && (hasBreakdown ? totalDose : value) > 0}
      tooltip={tooltip || tooltipParts.join(' ')}
      subValue={subText}
      onClick={onClick}
      isLoading={isLoading}
    />
  )
}

// Food Suggestion interface
export interface FoodSuggestion {
  name: string
  carbs: number
  typical_portion: string
  glycemic_index?: number | null
  times_eaten: number
}

// Unified Recommendation Card - Shows either insulin dose OR food recommendation
export function RecommendationCard({
  actionType,
  recommendedDose,
  recommendedCarbs,
  foodSuggestions,
  predictedBgWithAction,
  reasoning,
  proteinDoseNow,
  onClick,
  isLoading,
}: {
  actionType: 'insulin' | 'food' | 'none'
  recommendedDose: number
  recommendedCarbs: number
  foodSuggestions: FoodSuggestion[]
  predictedBgWithoutAction?: number  // Optional - kept for future use
  predictedBgWithAction: number
  reasoning: string
  proteinDoseNow?: number
  proteinDoseLater?: number  // Optional - kept for future use
  onClick?: () => void
  isLoading?: boolean
}) {
  if (isLoading) {
    return (
      <MetricCard
        icon={<TrendingUp className="w-5 h-5" />}
        label="Action"
        value="—"
        color="text-cyan"
        isLoading={true}
      />
    )
  }

  // Case 1: No action needed
  if (actionType === 'none') {
    return (
      <MetricCard
        icon={<TrendingUp className="w-5 h-5" />}
        label="Action"
        value="✓"
        unit=""
        color="text-green-500"
        tooltip={reasoning || `BG predicted ${predictedBgWithAction} - no action needed`}
        subValue="On target"
      />
    )
  }

  // Case 2: Insulin dose recommended
  if (actionType === 'insulin') {
    const totalDoseNow = recommendedDose + (proteinDoseNow ?? 0)
    let subText = `to 100`
    if (proteinDoseNow && proteinDoseNow > 0) {
      subText += ` (+${formatDecimal(proteinDoseNow, 1)}U P)`
    }

    return (
      <MetricCard
        icon={<TrendingUp className="w-5 h-5" />}
        label="Dose"
        value={formatDecimal(totalDoseNow, 2)}
        unit="U"
        color="text-cyan"
        highlighted={totalDoseNow > 0}
        tooltip={reasoning || `${recommendedDose.toFixed(2)}U insulin to reach target 100`}
        subValue={subText}
        onClick={onClick}
      />
    )
  }

  // Case 3: Food recommended (BG predicted low)
  const topFood = foodSuggestions[0]
  const subText = topFood
    ? `Try: ${topFood.name}`
    : `~${recommendedCarbs}g needed`

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={{
        hidden: { opacity: 0, y: 10 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
      }}
      whileHover={{ scale: 1.02 }}
      onClick={onClick}
      className={cn(
        'glass-card text-center py-4 cursor-pointer border-yellow-500/50 animate-pulse-glow',
        onClick && 'hover:border-yellow-400/50'
      )}
    >
      <div className="mx-auto mb-2 text-yellow-500">
        <Utensils className="w-5 h-5 mx-auto" />
      </div>
      <div className="flex items-baseline justify-center gap-1">
        <span className="text-2xl font-orbitron font-bold text-yellow-500">
          {formatDecimal(recommendedCarbs, 0)}
        </span>
        <span className="text-gray-500 text-sm">g carbs</span>
      </div>
      <p className="text-gray-500 text-xs uppercase tracking-wider mt-1">
        Eat Food
      </p>
      <p className="text-yellow-400 text-xs mt-1 truncate px-2">
        {subText}
      </p>
      {/* Food suggestions dropdown */}
      {foodSuggestions.length > 0 && (
        <div className="mt-2 px-2 text-left">
          <div className="text-xs text-gray-500 mb-1">Your foods:</div>
          <div className="space-y-1">
            {foodSuggestions.slice(0, 3).map((food, idx) => (
              <div key={idx} className="text-xs flex justify-between text-gray-400">
                <span className="truncate">{food.name}</span>
                <span className="text-yellow-400 ml-1">{food.carbs}g</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}

// ICR Card - Carb-to-Insulin Ratio (simple version)
export function ICRCard({
  value,
  source,
  isLoading,
}: {
  value: number
  source?: 'learned' | 'default' | 'manual'
  isLoading?: boolean
}) {
  const sourceLabel = source === 'learned' ? '🤖' : source === 'manual' ? '✏️' : ''
  return (
    <MetricCard
      icon={<Pill className="w-5 h-5" />}
      label="ICR"
      value={formatDecimal(value, 1)}
      unit="g/U"
      color="text-green-500"
      tooltip={`Carb Ratio: ${formatDecimal(value, 1)}g carbs per 1U insulin${source === 'learned' ? ' (AI-learned)' : ''}`}
      subValue={sourceLabel ? `${sourceLabel} ${source}` : undefined}
      isLoading={isLoading}
    />
  )
}

// ICR Gauge Card - Shows baseline vs current ICR with needle indicator (like ISFGaugeCard)
export function ICRGaugeCard({
  baselineIcr,
  currentIcr,
  deviation,
  source,
  isLoading,
  confidence,
}: {
  baselineIcr: number
  currentIcr?: number | null
  deviation?: number | null  // Percentage: positive = less carb effect, negative = more carb effect
  source?: 'learned' | 'default' | 'manual'
  isLoading?: boolean
  confidence?: number
}) {
  // Clamp deviation to ±50% for display
  const clampedDeviation = deviation ? Math.max(-50, Math.min(50, deviation)) : 0

  // Calculate needle rotation: -45deg to +45deg
  const needleRotation = (clampedDeviation / 50) * 45

  // Color based on deviation
  const getDeviationColor = () => {
    if (!deviation || Math.abs(deviation) < 5) return 'text-gray-400'
    if (deviation < -20) return 'text-red-500'  // Carbs hitting much harder
    if (deviation < -10) return 'text-orange-500'  // Carbs hitting harder
    if (deviation > 20) return 'text-green-500'  // Carbs having less effect
    if (deviation > 10) return 'text-teal-400'
    return 'text-yellow-400'
  }

  const getStatusLabel = () => {
    if (!deviation || Math.abs(deviation) < 5) return 'Normal'
    if (deviation < -20) return 'Carbs Hit Hard'
    if (deviation < -10) return 'Carbs Stronger'
    if (deviation > 20) return 'Carbs Weaker'
    if (deviation > 10) return 'Carbs Lighter'
    return deviation < 0 ? 'Slightly Stronger' : 'Slightly Lighter'
  }

  const sourceLabel = source === 'learned' ? '🤖' : source === 'manual' ? '✏️' : ''

  // Show dash when loading OR when no learned data (source is 'default')
  const showDash = isLoading || source === 'default'

  if (showDash) {
    return (
      <MetricCard
        icon={<Pill className="w-5 h-5" />}
        label="ICR"
        value="—"
        color="text-green-500"
        tooltip={isLoading ? "Loading..." : "No learned ICR yet - needs more data"}
        isLoading={isLoading}
      />
    )
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className="glass-card text-center py-4"
    >
      {/* Icon */}
      <div className="text-green-500 mx-auto mb-2">
        <Pill className="w-5 h-5 mx-auto" />
      </div>

      {/* Gauge Container */}
      <div className="relative h-16 mb-2">
        {/* Gauge Background Arc */}
        <svg viewBox="0 0 100 50" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 10 45 A 40 40 0 0 1 90 45"
            fill="none"
            stroke="currentColor"
            strokeWidth="6"
            className="text-slate-700"
          />
          {/* Gradient sections - stronger carb effect (red) to normal (green) to weaker effect (teal) */}
          <defs>
            <linearGradient id="icrGaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="30%" stopColor="#f97316" />
              <stop offset="50%" stopColor="#22c55e" />
              <stop offset="70%" stopColor="#14b8a6" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
          </defs>
          <path
            d="M 10 45 A 40 40 0 0 1 90 45"
            fill="none"
            stroke="url(#icrGaugeGradient)"
            strokeWidth="4"
            opacity="0.6"
          />
          {/* Center tick mark */}
          <line x1="50" y1="10" x2="50" y2="16" stroke="#22c55e" strokeWidth="2" />
          {/* Side labels */}
          <text x="8" y="48" fontSize="6" fill="#ef4444" className="font-mono">−</text>
          <text x="89" y="48" fontSize="6" fill="#06b6d4" className="font-mono">+</text>
        </svg>

        {/* Needle */}
        <div
          className="absolute left-1/2 bottom-1 w-0.5 h-10 origin-bottom transition-transform duration-500 ease-out"
          style={{
            transform: `translateX(-50%) rotate(${needleRotation}deg)`,
            background: `linear-gradient(to top, ${
              deviation && deviation < -10 ? '#ef4444' :
              deviation && deviation > 10 ? '#14b8a6' : '#22c55e'
            }, transparent)`
          }}
        >
          {/* Needle tip */}
          <div
            className={cn(
              "absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full",
              deviation && deviation < -10 ? "bg-red-500" :
              deviation && deviation > 10 ? "bg-teal-500" : "bg-green-500"
            )}
          />
        </div>

        {/* Center pivot */}
        <div className="absolute left-1/2 bottom-0 -translate-x-1/2 w-3 h-3 rounded-full bg-slate-600 border-2 border-green-500" />
      </div>

      {/* Values */}
      <div className="flex items-baseline justify-center gap-1">
        <span className={cn('text-2xl font-orbitron font-bold', getDeviationColor())}>
          {currentIcr ? formatDecimal(currentIcr, 1) : formatDecimal(baselineIcr, 1)}
        </span>
        <span className="text-gray-500 text-sm">g/U</span>
        {currentIcr && currentIcr !== baselineIcr && (
          <span className="text-gray-500 text-xs ml-1">
            (base: {formatDecimal(baselineIcr, 1)})
          </span>
        )}
      </div>

      {/* Label */}
      <p className="text-gray-500 text-xs uppercase tracking-wider mt-1">
        ICR
      </p>

      {/* Status & Deviation */}
      <div className="mt-1 space-y-0.5">
        <p className={cn("text-xs font-medium", getDeviationColor())}>
          {getStatusLabel()}
          {deviation && Math.abs(deviation) >= 5 && (
            <span className="ml-1">
              ({deviation > 0 ? '+' : ''}{deviation.toFixed(0)}%)
            </span>
          )}
        </p>
        {sourceLabel && (
          <p className="text-gray-600 text-xs">{sourceLabel} {source}</p>
        )}
        {confidence !== undefined && confidence < 0.5 && (
          <p className="text-gray-600 text-xs italic">Low confidence</p>
        )}
      </div>
    </motion.div>
  )
}

// PIR Card - Protein-to-Insulin Ratio (simple version)
export function PIRCard({
  value,
  source,
  onsetMin,
  peakMin,
  isLoading,
}: {
  value: number
  source?: 'learned' | 'default' | 'manual'
  onsetMin?: number
  peakMin?: number
  isLoading?: boolean
}) {
  const sourceLabel = source === 'learned' ? '🤖' : source === 'manual' ? '✏️' : ''
  const timingText = onsetMin && peakMin ? `${onsetMin}-${peakMin}min` : undefined
  return (
    <MetricCard
      icon={<Activity className="w-5 h-5" />}
      label="PIR"
      value={formatDecimal(value, 1)}
      unit="g/U"
      color="text-orange-500"
      tooltip={`Protein Ratio: ${formatDecimal(value, 1)}g protein per 1U insulin${source === 'learned' ? ' (AI-learned)' : ''}${timingText ? `. Timing: ${timingText}` : ''}`}
      subValue={sourceLabel ? `${sourceLabel} ${source}${timingText ? ` · ${timingText}` : ''}` : timingText}
      isLoading={isLoading}
    />
  )
}

// PIR Gauge Card - Shows baseline vs current PIR with needle indicator (like ISFGaugeCard and ICRGaugeCard)
export function PIRGaugeCard({
  baselinePir,
  currentPir,
  deviation,
  source,
  isLoading,
  confidence,
  onsetMin,
  peakMin,
}: {
  baselinePir: number
  currentPir?: number | null
  deviation?: number | null  // Percentage: positive = protein having less effect, negative = more effect
  source?: 'learned' | 'default' | 'manual'
  isLoading?: boolean
  confidence?: number
  onsetMin?: number
  peakMin?: number
}) {
  // Clamp deviation to ±50% for display
  const clampedDeviation = deviation ? Math.max(-50, Math.min(50, deviation)) : 0

  // Calculate needle rotation: -45deg to +45deg
  const needleRotation = (clampedDeviation / 50) * 45

  // Color based on deviation
  const getDeviationColor = () => {
    if (!deviation || Math.abs(deviation) < 5) return 'text-gray-400'
    if (deviation < -20) return 'text-red-500'  // Protein hitting much harder
    if (deviation < -10) return 'text-orange-400'  // Protein hitting harder
    if (deviation > 20) return 'text-green-500'  // Protein having less effect
    if (deviation > 10) return 'text-teal-400'
    return 'text-yellow-400'
  }

  const getStatusLabel = () => {
    if (!deviation || Math.abs(deviation) < 5) return 'Normal'
    if (deviation < -20) return 'Protein Hit Hard'
    if (deviation < -10) return 'Protein Stronger'
    if (deviation > 20) return 'Protein Weaker'
    if (deviation > 10) return 'Protein Lighter'
    return deviation < 0 ? 'Slightly Stronger' : 'Slightly Lighter'
  }

  const sourceLabel = source === 'learned' ? '🤖' : source === 'manual' ? '✏️' : ''
  const timingText = onsetMin && peakMin ? `${onsetMin}-${peakMin}min` : undefined

  // Show dash when loading OR when no learned data (source is 'default')
  const showDash = isLoading || source === 'default'

  if (showDash) {
    return (
      <MetricCard
        icon={<Activity className="w-5 h-5" />}
        label="PIR"
        value="—"
        color="text-orange-500"
        tooltip={isLoading ? "Loading..." : "No learned PIR yet - needs more data"}
        isLoading={isLoading}
      />
    )
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className="glass-card text-center py-4"
    >
      {/* Icon */}
      <div className="text-orange-500 mx-auto mb-2">
        <Activity className="w-5 h-5 mx-auto" />
      </div>

      {/* Gauge Container */}
      <div className="relative h-16 mb-2">
        {/* Gauge Background Arc */}
        <svg viewBox="0 0 100 50" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 10 45 A 40 40 0 0 1 90 45"
            fill="none"
            stroke="currentColor"
            strokeWidth="6"
            className="text-slate-700"
          />
          {/* Gradient sections - stronger protein effect (red) to normal (orange) to weaker effect (teal) */}
          <defs>
            <linearGradient id="pirGaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="30%" stopColor="#f97316" />
              <stop offset="50%" stopColor="#f97316" />
              <stop offset="70%" stopColor="#fb923c" />
              <stop offset="100%" stopColor="#14b8a6" />
            </linearGradient>
          </defs>
          <path
            d="M 10 45 A 40 40 0 0 1 90 45"
            fill="none"
            stroke="url(#pirGaugeGradient)"
            strokeWidth="4"
            opacity="0.6"
          />
          {/* Center tick mark */}
          <line x1="50" y1="10" x2="50" y2="16" stroke="#f97316" strokeWidth="2" />
          {/* Side labels */}
          <text x="8" y="48" fontSize="6" fill="#ef4444" className="font-mono">−</text>
          <text x="89" y="48" fontSize="6" fill="#14b8a6" className="font-mono">+</text>
        </svg>

        {/* Needle */}
        <div
          className="absolute left-1/2 bottom-1 w-0.5 h-10 origin-bottom transition-transform duration-500 ease-out"
          style={{
            transform: `translateX(-50%) rotate(${needleRotation}deg)`,
            background: `linear-gradient(to top, ${
              deviation && deviation < -10 ? '#ef4444' :
              deviation && deviation > 10 ? '#14b8a6' : '#f97316'
            }, transparent)`
          }}
        >
          {/* Needle tip */}
          <div
            className={cn(
              "absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full",
              deviation && deviation < -10 ? "bg-red-500" :
              deviation && deviation > 10 ? "bg-teal-500" : "bg-orange-500"
            )}
          />
        </div>

        {/* Center pivot */}
        <div className="absolute left-1/2 bottom-0 -translate-x-1/2 w-3 h-3 rounded-full bg-slate-600 border-2 border-orange-500" />
      </div>

      {/* Values */}
      <div className="flex items-baseline justify-center gap-1">
        <span className={cn('text-2xl font-orbitron font-bold', getDeviationColor())}>
          {currentPir ? formatDecimal(currentPir, 1) : formatDecimal(baselinePir, 1)}
        </span>
        <span className="text-gray-500 text-sm">g/U</span>
        {currentPir && currentPir !== baselinePir && (
          <span className="text-gray-500 text-xs ml-1">
            (base: {formatDecimal(baselinePir, 1)})
          </span>
        )}
      </div>

      {/* Label */}
      <p className="text-gray-500 text-xs uppercase tracking-wider mt-1">
        PIR
      </p>

      {/* Status & Deviation */}
      <div className="mt-1 space-y-0.5">
        <p className={cn("text-xs font-medium", getDeviationColor())}>
          {getStatusLabel()}
          {deviation && Math.abs(deviation) >= 5 && (
            <span className="ml-1">
              ({deviation > 0 ? '+' : ''}{deviation.toFixed(0)}%)
            </span>
          )}
        </p>
        {sourceLabel && (
          <p className="text-gray-600 text-xs">{sourceLabel} {source}</p>
        )}
        {timingText && (
          <p className="text-gray-600 text-xs">Timing: {timingText}</p>
        )}
        {confidence !== undefined && confidence < 0.5 && (
          <p className="text-gray-600 text-xs italic">Low confidence</p>
        )}
      </div>
    </motion.div>
  )
}

// Protein Insulin Card - Shows NOW/LATER split for protein dosing
export function ProteinInsulinCard({
  now,
  later,
  pob,
  isLoading,
}: {
  now: number
  later: number
  pob: number
  isLoading?: boolean
}) {
  // Show loading state
  if (isLoading) {
    return (
      <MetricCard
        icon={<Beef className="w-5 h-5" />}
        label="Protein Dose"
        value="—"
        unit=""
        color="text-orange-500"
        tooltip="Loading..."
        isLoading={true}
      />
    )
  }

  const total = now + later
  if (total <= 0) {
    return (
      <MetricCard
        icon={<Beef className="w-5 h-5" />}
        label="Protein Dose"
        value="—"
        unit=""
        color="text-orange-500"
        tooltip="No protein insulin needed"
        subValue="No active protein"
      />
    )
  }

  return (
    <MetricCard
      icon={<Beef className="w-5 h-5" />}
      label="Protein Dose"
      value={formatDecimal(total, 1)}
      unit="U"
      color="text-orange-500"
      tooltip={`Protein insulin: ${formatDecimal(now, 1)}U now + ${formatDecimal(later, 1)}U later (from ${formatDecimal(pob, 0)}g POB)`}
      subValue={`NOW ${formatDecimal(now, 1)} · LATER ${formatDecimal(later, 1)}`}
    />
  )
}

// Metrics grid component
export function MetricsGrid({
  iob,
  cob,
  pob,
  isf,
  recommendedDose,
  className,
}: MetricsData & { className?: string }) {
  return (
    <div className={cn('grid grid-cols-2 md:grid-cols-3 gap-4', className)}>
      <IOBCard value={iob} />
      <COBCard value={cob} />
      <POBCard value={pob} />
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

// Dose breakdown card for meal bolus calculations
interface DoseBreakdownData {
  correction: number
  carbs: number
  proteinImmediate: number
  proteinDelayed: number
  totalNow: number
  totalLater: number
  grandTotal: number
  icrSource?: 'learned' | 'manual'
  pirSource?: 'learned' | 'manual'
  proteinOnsetMin?: number
  proteinPeakMin?: number
}

export function DoseBreakdownCard({
  correction,
  carbs,
  proteinImmediate,
  proteinDelayed,
  totalNow,
  totalLater,
  grandTotal,
  icrSource = 'manual',
  pirSource = 'manual',
  proteinOnsetMin,
  proteinPeakMin,
}: DoseBreakdownData) {
  const hasProtein = proteinImmediate > 0 || proteinDelayed > 0
  const aiIcon = '🤖'

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className="glass-card p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-cyan" />
          <h4 className="text-sm font-medium text-gray-300">Insulin Dose Breakdown</h4>
        </div>
        <span className="text-xl font-orbitron font-bold text-cyan">
          {formatDecimal(grandTotal, 1)}U
        </span>
      </div>

      <div className="space-y-2 text-sm">
        {/* Correction */}
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Correction</span>
          <span className="text-white font-medium">{formatDecimal(correction, 2)}U</span>
        </div>

        {/* Carbs */}
        <div className="flex justify-between items-center">
          <span className="text-gray-400">
            Carbs {icrSource === 'learned' && <span title="AI-learned">{aiIcon}</span>}
          </span>
          <span className="text-white font-medium">{formatDecimal(carbs, 2)}U</span>
        </div>

        {/* Protein (if applicable) */}
        {hasProtein && (
          <>
            <div className="flex justify-between items-center text-purple-400">
              <span>
                Protein Now {pirSource === 'learned' && <span title="AI-learned">{aiIcon}</span>}
              </span>
              <span className="font-medium">{formatDecimal(proteinImmediate, 2)}U</span>
            </div>
            <div className="flex justify-between items-center text-purple-300">
              <span>Protein Later</span>
              <span className="font-medium">{formatDecimal(proteinDelayed, 2)}U</span>
            </div>
          </>
        )}

        {/* Divider */}
        <div className="border-t border-gray-700 my-2" />

        {/* Totals */}
        <div className="flex justify-between items-center">
          <span className="text-cyan font-medium">Give NOW</span>
          <span className="text-cyan font-bold">{formatDecimal(totalNow, 1)}U</span>
        </div>

        {totalLater > 0 && (
          <div className="flex justify-between items-center">
            <span className="text-purple-400 font-medium">Give LATER</span>
            <span className="text-purple-400 font-bold">{formatDecimal(totalLater, 1)}U</span>
          </div>
        )}
      </div>

      {/* Protein timing advice */}
      {totalLater > 0 && proteinOnsetMin && proteinPeakMin && (
        <div className="mt-3 p-2 bg-purple-500/10 rounded text-xs text-purple-300">
          💡 Extended bolus: {Math.round(proteinOnsetMin)}-{Math.round(proteinPeakMin)} min
        </div>
      )}
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
