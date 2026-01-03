/**
 * Treatment Logging Modal
 * Modal for logging insulin and carb treatments
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Pill, Syringe, X, Loader2, Check, AlertCircle } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { useLogTreatment } from '@/hooks/useGlucose'

type TreatmentType = 'carbs' | 'insulin'

interface TreatmentModalProps {
  type: TreatmentType
  open: boolean
  onOpenChange: (open: boolean) => void
  currentBg?: number
  recommendedDose?: number
}

const config: Record<TreatmentType, {
  title: string
  description: string
  icon: React.ReactNode
  color: string
  unit: string
  placeholder: string
  min: number
  max: number
  step: number
}> = {
  carbs: {
    title: 'Log Carbs',
    description: 'Record carbohydrate intake',
    icon: <Pill className="w-6 h-6" />,
    color: 'text-green-500',
    unit: 'g',
    placeholder: '0',
    min: 1,
    max: 200,
    step: 1,
  },
  insulin: {
    title: 'Log Insulin',
    description: 'Record insulin dose',
    icon: <Syringe className="w-6 h-6" />,
    color: 'text-orange-500',
    unit: 'U',
    placeholder: '0.0',
    min: 0.1,
    max: 50,
    step: 0.1,
  },
}

// Quick amount buttons
const quickAmounts: Record<TreatmentType, number[]> = {
  carbs: [15, 30, 45, 60],
  insulin: [1, 2, 3, 5],
}

export function TreatmentModal({
  type,
  open,
  onOpenChange,
  currentBg,
  recommendedDose,
}: TreatmentModalProps) {
  const [value, setValue] = useState('')
  const [notes, setNotes] = useState('')
  const [showSuccess, setShowSuccess] = useState(false)

  const { mutate: logTreatment, isPending, isError, error } = useLogTreatment()
  const cfg = config[type]

  const handleSubmit = () => {
    const numValue = parseFloat(value)
    if (isNaN(numValue) || numValue < cfg.min || numValue > cfg.max) {
      return
    }

    const treatment = type === 'carbs'
      ? { type: 'carbs' as const, carbs: numValue, notes: notes || undefined }
      : { type: 'insulin' as const, insulin: numValue, notes: notes || undefined }

    logTreatment(treatment, {
      onSuccess: () => {
        setShowSuccess(true)
        setTimeout(() => {
          setShowSuccess(false)
          onOpenChange(false)
          setValue('')
          setNotes('')
        }, 1500)
      },
    })
  }

  const handleQuickAmount = (amount: number) => {
    setValue(amount.toString())
  }

  const handleClose = () => {
    if (!isPending) {
      onOpenChange(false)
      setValue('')
      setNotes('')
    }
  }

  const numValue = parseFloat(value)
  const isValid = !isNaN(numValue) && numValue >= cfg.min && numValue <= cfg.max

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="glass-card border-gray-700 sm:max-w-md">
        <AnimatePresence mode="wait">
          {showSuccess ? (
            <motion.div
              key="success"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center justify-center py-12"
            >
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200, damping: 15 }}
                className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mb-4"
              >
                <Check className="w-8 h-8 text-green-500" />
              </motion.div>
              <p className="text-lg font-medium text-white">
                {type === 'carbs' ? 'Carbs logged!' : 'Insulin logged!'}
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="form"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <DialogHeader>
                <div className="flex items-center gap-3">
                  <div className={cn('p-2 rounded-lg bg-slate-800', cfg.color)}>
                    {cfg.icon}
                  </div>
                  <div>
                    <DialogTitle className="text-white">{cfg.title}</DialogTitle>
                    <DialogDescription className="text-gray-400">
                      {cfg.description}
                    </DialogDescription>
                  </div>
                </div>
              </DialogHeader>

              <div className="space-y-6 py-6">
                {/* Value Input */}
                <div className="space-y-2">
                  <Label htmlFor="value" className="text-gray-300">
                    {type === 'carbs' ? 'Carbohydrates' : 'Insulin dose'}
                  </Label>
                  <div className="relative">
                    <Input
                      id="value"
                      type="number"
                      value={value}
                      onChange={(e) => setValue(e.target.value)}
                      placeholder={cfg.placeholder}
                      min={cfg.min}
                      max={cfg.max}
                      step={cfg.step}
                      className="text-2xl font-orbitron h-14 pr-12 bg-slate-800 border-gray-700 text-white"
                      autoFocus
                    />
                    <span className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 text-lg">
                      {cfg.unit}
                    </span>
                  </div>
                </div>

                {/* Quick amounts */}
                <div className="space-y-2">
                  <Label className="text-gray-400 text-sm">Quick select</Label>
                  <div className="flex gap-2">
                    {quickAmounts[type].map((amount) => (
                      <Button
                        key={amount}
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => handleQuickAmount(amount)}
                        className={cn(
                          'flex-1 border-gray-700 bg-slate-800/50 hover:bg-slate-700',
                          parseFloat(value) === amount && 'border-cyan bg-cyan/10'
                        )}
                      >
                        {amount}{cfg.unit}
                      </Button>
                    ))}
                  </div>
                </div>

                {/* Recommended dose hint for insulin */}
                {type === 'insulin' && recommendedDose !== undefined && recommendedDose > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-3 rounded-lg bg-cyan/10 border border-cyan/30"
                  >
                    <p className="text-sm text-cyan">
                      Recommended dose: <span className="font-bold">{recommendedDose.toFixed(2)}U</span>
                      {currentBg && <span className="text-gray-400"> (current BG: {currentBg})</span>}
                    </p>
                    <Button
                      type="button"
                      variant="link"
                      size="sm"
                      onClick={() => setValue(recommendedDose.toFixed(1))}
                      className="text-cyan p-0 h-auto mt-1"
                    >
                      Use recommended dose
                    </Button>
                  </motion.div>
                )}

                {/* Notes */}
                <div className="space-y-2">
                  <Label htmlFor="notes" className="text-gray-400">
                    Notes (optional)
                  </Label>
                  <Textarea
                    id="notes"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder={type === 'carbs' ? 'e.g., Lunch - pasta' : 'e.g., Correction for high BG'}
                    className="bg-slate-800 border-gray-700 text-white resize-none"
                    rows={2}
                  />
                </div>

                {/* Error */}
                {isError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center gap-2"
                  >
                    <AlertCircle className="w-4 h-4 text-red-500" />
                    <p className="text-sm text-red-400">
                      Failed to log treatment. Please try again.
                    </p>
                  </motion.div>
                )}
              </div>

              <DialogFooter className="gap-3">
                <Button
                  type="button"
                  variant="ghost"
                  onClick={handleClose}
                  disabled={isPending}
                  className="text-gray-400 hover:text-white"
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  onClick={handleSubmit}
                  disabled={!isValid || isPending}
                  className={cn(
                    type === 'carbs' ? 'bg-green-600 hover:bg-green-700' : 'bg-orange-600 hover:bg-orange-700',
                    'min-w-24'
                  )}
                >
                  {isPending ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <>Log {type === 'carbs' ? 'Carbs' : 'Insulin'}</>
                  )}
                </Button>
              </DialogFooter>
            </motion.div>
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  )
}

// Convenience components
export function LogCarbsModal({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return <TreatmentModal type="carbs" open={open} onOpenChange={onOpenChange} />
}

export function LogInsulinModal({
  open,
  onOpenChange,
  currentBg,
  recommendedDose,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  currentBg?: number
  recommendedDose?: number
}) {
  return (
    <TreatmentModal
      type="insulin"
      open={open}
      onOpenChange={onOpenChange}
      currentBg={currentBg}
      recommendedDose={recommendedDose}
    />
  )
}

export default TreatmentModal
