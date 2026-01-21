/**
 * Dose Breakdown Modal Component
 * Shows detailed dose calculation breakdown with protein NOW/LATER split
 * and decay progress as time passes since the meal
 */
import { motion } from 'framer-motion'
import { X, TrendingUp, Clock, Beef } from 'lucide-react'
import { formatDecimal } from '@/lib/utils'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'

export interface DoseBreakdownData {
  // Input values
  currentBg: number
  targetBg: number
  carbs: number
  protein: number
  fat: number

  // Ratios used
  isf: number
  isfSource: 'learned' | 'default' | 'manual'
  icr: number
  icrSource: 'learned' | 'default' | 'manual'
  pir?: number
  pirSource?: 'learned' | 'default' | 'manual'

  // IOB adjustment
  iob: number
  iobEffectMgdl: number

  // Dose breakdown
  correctionDose: number
  carbDose: number
  proteinDoseNow: number
  proteinDoseLater: number
  proteinDoseTotal: number

  // Protein decay tracking (when viewing after meal was logged)
  timeSinceMealMin?: number
  decayedAmount?: number
  decayPercent?: number

  // Totals
  totalNow: number
  totalLater: number
  grandTotal: number

  // Timing
  proteinOnsetMin?: number
  proteinPeakMin?: number
}

interface DoseBreakdownModalProps {
  isOpen: boolean
  onClose: () => void
  data: DoseBreakdownData
}

function Row({
  label,
  value,
  unit = 'U',
  color,
  formula,
  bold,
  highlight,
  icon,
}: {
  label: string
  value: number
  unit?: string
  color?: string
  formula?: string
  bold?: boolean
  highlight?: boolean
  icon?: React.ReactNode
}) {
  return (
    <div
      className={`flex justify-between items-center py-1.5 ${highlight ? 'bg-cyan/10 -mx-4 px-4 rounded' : ''}`}
    >
      <div className="flex items-center gap-2">
        {icon}
        <span className={`${bold ? 'font-medium' : ''} ${color || 'text-gray-400'}`}>
          {label}
        </span>
        {formula && (
          <span className="text-xs text-gray-600">({formula})</span>
        )}
      </div>
      <span className={`font-orbitron ${bold ? 'font-bold' : 'font-medium'} ${color || 'text-white'}`}>
        {formatDecimal(value, 2)}{unit}
      </span>
    </div>
  )
}

export function DoseBreakdownModal({
  isOpen,
  onClose,
  data,
}: DoseBreakdownModalProps) {
  const aiIcon = '🤖'
  const hasProtein = data.proteinDoseTotal > 0
  const hasDecay = data.timeSinceMealMin !== undefined && data.timeSinceMealMin > 0

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="glass-card border-gray-700 max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-lg">
            <TrendingUp className="w-5 h-5 text-cyan" />
            Dose Breakdown
          </DialogTitle>
          <button
            onClick={onClose}
            className="absolute right-4 top-4 text-gray-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </DialogHeader>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4 mt-4"
        >
          {/* Input Summary */}
          <div className="text-sm text-gray-500 flex justify-between">
            <span>BG: {data.currentBg} mg/dL</span>
            <span>Target: {data.targetBg}</span>
            {data.carbs > 0 && <span>Carbs: {data.carbs}g</span>}
            {data.protein > 0 && <span>Protein: {data.protein}g</span>}
          </div>

          {/* Dose Components */}
          <div className="space-y-2 text-sm">
            {/* Correction */}
            <Row
              label="Correction"
              value={data.correctionDose}
              formula={`(${Math.round(data.currentBg - data.iobEffectMgdl)} - ${data.targetBg}) / ${formatDecimal(data.isf, 0)}`}
            />

            {/* Carbs */}
            {data.carbDose > 0 && (
              <Row
                label={`Carbs ${data.icrSource === 'learned' ? aiIcon : ''}`}
                value={data.carbDose}
                formula={`${data.carbs}g / ${formatDecimal(data.icr, 1)}`}
              />
            )}

            {/* Protein NOW */}
            {hasProtein && (
              <Row
                label={`Protein NOW ${data.pirSource === 'learned' ? aiIcon : ''}`}
                value={data.proteinDoseNow}
                color="text-orange-400"
                icon={<Beef className="w-4 h-4 text-orange-400" />}
              />
            )}

            {/* Protein LATER */}
            {data.proteinDoseLater > 0 && (
              <Row
                label="Protein LATER"
                value={data.proteinDoseLater}
                color="text-orange-300"
                icon={<Clock className="w-4 h-4 text-orange-300" />}
              />
            )}

            {/* Protein Decay Progress (if viewing after meal) */}
            {hasDecay && data.decayPercent !== undefined && (
              <div className="mt-3 p-3 bg-orange-500/10 rounded-lg">
                <div className="flex justify-between text-xs text-orange-400 mb-1">
                  <span>Protein Decay Progress</span>
                  <span>{formatDecimal(data.decayPercent, 0)}% shifted to NOW</span>
                </div>
                <Progress
                  value={data.decayPercent}
                  className="h-2"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>{formatDecimal(data.timeSinceMealMin ?? 0, 0)} min since meal</span>
                  <span>+{formatDecimal(data.decayedAmount ?? 0, 2)}U decayed</span>
                </div>
              </div>
            )}

            {/* Divider */}
            <div className="border-t border-gray-700 my-3" />

            {/* Give NOW Total */}
            <Row
              label="Give NOW"
              value={data.totalNow}
              color="text-cyan"
              bold
              highlight
            />

            {/* Give LATER Total */}
            {data.totalLater > 0 && (
              <Row
                label="Give LATER"
                value={data.totalLater}
                color="text-orange-400"
                bold
              />
            )}

            {/* Grand Total */}
            <Row
              label="TOTAL"
              value={data.grandTotal}
              bold
              highlight
            />
          </div>

          {/* Protein Timing Advice */}
          {data.totalLater > 0 && data.proteinOnsetMin && data.proteinPeakMin && (
            <div className="mt-4 p-3 bg-purple-500/10 rounded-lg text-sm">
              <p className="text-purple-300 font-medium mb-1">Protein Timing</p>
              <p className="text-gray-400 text-xs">
                Give {formatDecimal(data.totalLater, 1)}U as extended bolus over{' '}
                {Math.round(data.proteinOnsetMin)}-{Math.round(data.proteinPeakMin)} minutes,
                or give manually ~{Math.round(data.proteinOnsetMin / 60)}h after meal.
              </p>
            </div>
          )}

          {/* Ratios Used */}
          <div className="mt-4 pt-3 border-t border-gray-700">
            <p className="text-xs text-gray-500 mb-2">Ratios Used:</p>
            <div className="flex flex-wrap gap-3 text-xs">
              <span className="text-purple-400">
                ISF: {formatDecimal(data.isf, 1)} {data.isfSource === 'learned' && aiIcon}
              </span>
              <span className="text-green-400">
                ICR: {formatDecimal(data.icr, 1)} {data.icrSource === 'learned' && aiIcon}
              </span>
              {data.pir && (
                <span className="text-orange-400">
                  PIR: {formatDecimal(data.pir, 1)} {data.pirSource === 'learned' && aiIcon}
                </span>
              )}
            </div>
          </div>
        </motion.div>
      </DialogContent>
    </Dialog>
  )
}

export default DoseBreakdownModal
