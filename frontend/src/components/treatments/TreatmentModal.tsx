/**
 * Treatment Logging Modal
 * Modal for logging insulin and carb treatments with custom timestamps
 */
import { useState, useMemo } from 'react'
import { Pill, Syringe, Loader2, Check, AlertCircle, Clock } from 'lucide-react'
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
  notesHint?: string
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
    notesHint: 'Describe your food for AI glycemic prediction (e.g., "pizza", "apple juice", "oatmeal")',
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
    notesHint: 'Optional notes (e.g., "Correction for high BG")',
  },
}

// Quick amount buttons
const quickAmounts: Record<TreatmentType, number[]> = {
  carbs: [15, 30, 45, 60],
  insulin: [1, 2, 3, 5],
}

// Quick time offsets (in minutes, negative = past, positive = future)
const quickTimeOffsets = [
  { label: '-30m', offset: -30 },
  { label: '-15m', offset: -15 },
  { label: 'Now', offset: 0 },
  { label: '+15m', offset: 15 },
  { label: '+30m', offset: 30 },
]

// Format date for datetime-local input (YYYY-MM-DDTHH:mm)
// Must use local time methods (not UTC) to match datetime-local input behavior
const formatDateTimeLocal = (date: Date): string => {
  const pad = (n: number) => n.toString().padStart(2, '0')
  const year = date.getFullYear()
  const month = pad(date.getMonth() + 1)
  const day = pad(date.getDate())
  const hours = pad(date.getHours())
  const minutes = pad(date.getMinutes())
  return `${year}-${month}-${day}T${hours}:${minutes}`
}

// Format time for display
const formatTimeDisplay = (date: Date): string => {
  const now = new Date()
  const diffMs = date.getTime() - now.getTime()
  const diffMins = Math.round(diffMs / 60000)

  if (Math.abs(diffMins) < 2) return 'Now'
  if (diffMins < 0) return `${Math.abs(diffMins)} min ago`
  return `In ${diffMins} min`
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
  const [timestamp, setTimestamp] = useState<Date>(new Date())
  const [showSuccess, setShowSuccess] = useState(false)

  const { mutate: logTreatment, isPending, isError } = useLogTreatment()
  const cfg = config[type]

  // Formatted timestamp for display
  const timeDisplay = useMemo(() => formatTimeDisplay(timestamp), [timestamp])
  const dateTimeLocalValue = useMemo(() => formatDateTimeLocal(timestamp), [timestamp])

  const handleSubmit = () => {
    const numValue = parseFloat(value)
    if (isNaN(numValue) || numValue < cfg.min || numValue > cfg.max) {
      return
    }

    const treatment = type === 'carbs'
      ? { type: 'carbs' as const, carbs: numValue, notes: notes || undefined, timestamp: timestamp.toISOString() }
      : { type: 'insulin' as const, insulin: numValue, notes: notes || undefined, timestamp: timestamp.toISOString() }

    logTreatment(treatment, {
      onSuccess: () => {
        setShowSuccess(true)
        setTimeout(() => {
          setShowSuccess(false)
          onOpenChange(false)
          setValue('')
          setNotes('')
          setTimestamp(new Date())
        }, 1500)
      },
    })
  }

  const handleTimeOffset = (offsetMinutes: number) => {
    const newDate = new Date()
    newDate.setMinutes(newDate.getMinutes() + offsetMinutes)
    setTimestamp(newDate)
  }

  const handleDateTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // datetime-local input value is in format: "2026-01-27T15:30"
    // We need to parse it as local time, not UTC
    const inputValue = e.target.value
    if (!inputValue) return

    // Parse as local time by appending current timezone offset
    const newDate = new Date(inputValue)
    if (!isNaN(newDate.getTime())) {
      setTimestamp(newDate)
    }
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
        {showSuccess ? (
          <div className="flex flex-col items-center justify-center py-12">
            <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mb-4">
              <Check className="w-8 h-8 text-green-500" />
            </div>
            <p className="text-lg font-medium text-white">
              {type === 'carbs' ? 'Carbs logged!' : 'Insulin logged!'}
            </p>
          </div>
        ) : (
          <div>
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

                {/* Time Selection */}
                <div className="space-y-2">
                  <Label className="text-gray-400 text-sm flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    Time ({timeDisplay})
                  </Label>
                  <div className="flex gap-2 flex-wrap">
                    {quickTimeOffsets.map((item) => (
                      <Button
                        key={item.offset}
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => handleTimeOffset(item.offset)}
                        className={cn(
                          'border-gray-700 bg-slate-800/50 hover:bg-slate-700',
                          Math.abs(timestamp.getTime() - (Date.now() + item.offset * 60000)) < 60000 && 'border-cyan bg-cyan/10'
                        )}
                      >
                        {item.label}
                      </Button>
                    ))}
                  </div>
                  <Input
                    type="datetime-local"
                    value={dateTimeLocalValue}
                    onChange={handleDateTimeChange}
                    className="bg-slate-800 border-gray-700 text-white"
                    step="60"
                  />
                </div>

                {/* Recommended dose hint for insulin */}
                {type === 'insulin' && recommendedDose !== undefined && recommendedDose > 0 && (
                  <div className="p-3 rounded-lg bg-cyan/10 border border-cyan/30">
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
                  </div>
                )}

                {/* Notes */}
                <div className="space-y-2">
                  <Label htmlFor="notes" className="text-gray-400">
                    {type === 'carbs' ? 'Food description' : 'Notes'} (optional)
                  </Label>
                  <Textarea
                    id="notes"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder={cfg.notesHint || 'Optional notes'}
                    className="bg-slate-800 border-gray-700 text-white resize-none"
                    rows={2}
                  />
                  {type === 'carbs' && (
                    <p className="text-xs text-cyan/70">
                      AI analyzes your food to predict glycemic impact and improve BG predictions
                    </p>
                  )}
                </div>

                {/* Error */}
                {isError && (
                  <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-red-500" />
                    <p className="text-sm text-red-400">
                      Failed to log treatment. Please try again.
                    </p>
                  </div>
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
            </div>
          )}
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
