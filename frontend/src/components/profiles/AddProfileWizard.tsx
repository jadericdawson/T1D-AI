/**
 * Add Profile Wizard Component
 * Multi-step wizard for creating a new managed profile
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  User, ArrowRight, ArrowLeft, Check, Database, Droplet,
  Loader2, Baby, Heart, UserCircle, Users, AlertCircle, X
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useAuthStore, ProfileRelationship, DiabetesType } from '@/stores/authStore'
import api from '@/lib/api'
import { cn } from '@/lib/utils'

interface AddProfileWizardProps {
  open: boolean
  onClose: () => void
  onSuccess?: () => void
}

interface ProfileFormData {
  displayName: string
  relationship: ProfileRelationship
  diabetesType: DiabetesType
  dateOfBirth: string
  diagnosisDate: string
  // Settings
  targetBg: number
  isf: number
  icr: number
  insulinDuration: number
  // Thresholds
  highThreshold: number
  lowThreshold: number
  criticalHigh: number
  criticalLow: number
  // Data source (Gluroo)
  glurooUrl: string
  glurooApiSecret: string
}

const steps = [
  { id: 'basics', title: 'Basic Info', icon: User },
  { id: 'insulin', title: 'Settings', icon: Droplet },
  { id: 'datasource', title: 'Data Source', icon: Database },
  { id: 'complete', title: 'Complete', icon: Check },
]

const slideVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? 200 : -200,
    opacity: 0,
  }),
  center: {
    x: 0,
    opacity: 1,
  },
  exit: (direction: number) => ({
    x: direction < 0 ? 200 : -200,
    opacity: 0,
  }),
}

const relationshipOptions: { value: ProfileRelationship; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { value: 'self', label: 'Myself', icon: User },
  { value: 'child', label: 'My Child', icon: Baby },
  { value: 'spouse', label: 'Spouse/Partner', icon: Heart },
  { value: 'parent', label: 'Parent', icon: UserCircle },
  { value: 'other', label: 'Other', icon: Users },
]

const diabetesTypeOptions: { value: DiabetesType; label: string }[] = [
  { value: 'T1D', label: 'Type 1 Diabetes' },
  { value: 'T2D', label: 'Type 2 Diabetes' },
  { value: 'LADA', label: 'LADA' },
  { value: 'gestational', label: 'Gestational' },
  { value: 'other', label: 'Other' },
]

export function AddProfileWizard({ open, onClose, onSuccess }: AddProfileWizardProps) {
  const { loadProfiles } = useAuthStore()
  const [currentStep, setCurrentStep] = useState(0)
  const [direction, setDirection] = useState(0)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle')

  const [formData, setFormData] = useState<ProfileFormData>({
    displayName: '',
    relationship: 'child',
    diabetesType: 'T1D',
    dateOfBirth: '',
    diagnosisDate: '',
    targetBg: 100,
    isf: 50,
    icr: 10,
    insulinDuration: 180,
    highThreshold: 180,
    lowThreshold: 70,
    criticalHigh: 250,
    criticalLow: 54,
    glurooUrl: '',
    glurooApiSecret: '',
  })

  const updateField = <K extends keyof ProfileFormData>(field: K, value: ProfileFormData[K]) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    setError(null)
  }

  const goNext = () => {
    if (currentStep < steps.length - 1) {
      setDirection(1)
      setCurrentStep(currentStep + 1)
    }
  }

  const goBack = () => {
    if (currentStep > 0) {
      setDirection(-1)
      setCurrentStep(currentStep - 1)
    }
  }

  const canProceed = (): boolean => {
    switch (currentStep) {
      case 0: // Basics
        return formData.displayName.trim().length > 0
      case 1: // Insulin settings
        return formData.targetBg > 0 && formData.isf > 0 && formData.icr > 0
      case 2: // Data source - optional
        return true
      default:
        return true
    }
  }

  const testConnection = async () => {
    if (!formData.glurooUrl || !formData.glurooApiSecret) {
      return
    }

    setIsTestingConnection(true)
    setConnectionStatus('idle')

    try {
      const response = await api.post('/api/datasources/test', {
        nightscoutUrl: formData.glurooUrl,
        apiSecret: formData.glurooApiSecret,
      })

      if (response.data.success) {
        setConnectionStatus('success')
      } else {
        setConnectionStatus('error')
      }
    } catch {
      setConnectionStatus('error')
    } finally {
      setIsTestingConnection(false)
    }
  }

  const handleCreate = async () => {
    setIsSaving(true)
    setError(null)

    try {
      // Create the profile
      const profileResponse = await api.post('/api/v1/profiles', {
        displayName: formData.displayName,
        relationship: formData.relationship,
        diabetesType: formData.diabetesType,
        dateOfBirth: formData.dateOfBirth || null,
        diagnosisDate: formData.diagnosisDate || null,
        settings: {
          targetBg: formData.targetBg,
          isf: formData.isf,
          icr: formData.icr,
          insulinDuration: formData.insulinDuration,
          highThreshold: formData.highThreshold,
          lowThreshold: formData.lowThreshold,
          criticalHigh: formData.criticalHigh,
          criticalLow: formData.criticalLow,
        },
      })

      const profileId = profileResponse.data.id

      // Add Gluroo data source if provided
      if (formData.glurooUrl && formData.glurooApiSecret) {
        await api.post(`/api/v1/profiles/${profileId}/sources`, {
          sourceType: 'gluroo',
          credentials: {
            url: formData.glurooUrl,
            apiSecret: formData.glurooApiSecret,
          },
          priority: 1,
          providesGlucose: true,
          providesTreatments: true,
        })
      }

      // Reload profiles in the store
      await loadProfiles()

      // Move to complete step
      setDirection(1)
      setCurrentStep(steps.length - 1)

      // Call success callback after a delay
      setTimeout(() => {
        onSuccess?.()
      }, 2000)
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create profile'
      setError(errorMessage)
    } finally {
      setIsSaving(false)
    }
  }

  const handleClose = () => {
    // Reset state
    setCurrentStep(0)
    setFormData({
      displayName: '',
      relationship: 'child',
      diabetesType: 'T1D',
      dateOfBirth: '',
      diagnosisDate: '',
      targetBg: 100,
      isf: 50,
      icr: 10,
      insulinDuration: 180,
      highThreshold: 180,
      lowThreshold: 70,
      criticalHigh: 250,
      criticalLow: 54,
      glurooUrl: '',
      glurooApiSecret: '',
    })
    setError(null)
    setConnectionStatus('idle')
    onClose()
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[600px] p-0 gap-0 overflow-hidden">
        <DialogHeader className="p-6 pb-0">
          <DialogTitle className="flex items-center gap-3 text-xl">
            <div className="p-2 rounded-lg bg-primary/10">
              <Users className="h-5 w-5 text-primary" />
            </div>
            Add New Profile
          </DialogTitle>
        </DialogHeader>

        {/* Progress */}
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            {steps.map((step, i) => {
              const Icon = step.icon
              const isComplete = i < currentStep
              const isCurrent = i === currentStep

              return (
                <div key={step.id} className="flex items-center">
                  <div
                    className={cn(
                      'w-8 h-8 rounded-full flex items-center justify-center transition-all text-sm',
                      isComplete && 'bg-primary text-primary-foreground',
                      isCurrent && 'bg-primary/20 text-primary border-2 border-primary',
                      !isComplete && !isCurrent && 'bg-muted text-muted-foreground'
                    )}
                  >
                    {isComplete ? <Check className="w-4 h-4" /> : <Icon className="w-4 h-4" />}
                  </div>
                  {i < steps.length - 1 && (
                    <div
                      className={cn(
                        'w-12 h-0.5 mx-1',
                        i < currentStep ? 'bg-primary' : 'bg-muted'
                      )}
                    />
                  )}
                </div>
              )
            })}
          </div>
          <p className="text-center text-sm text-muted-foreground mt-2">
            {steps[currentStep].title}
          </p>
        </div>

        {/* Error display */}
        {error && (
          <div className="mx-6 p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Content */}
        <div className="px-6 min-h-[320px]">
          <AnimatePresence mode="wait" custom={direction}>
            <motion.div
              key={currentStep}
              custom={direction}
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              {/* Step 0: Basic Info */}
              {currentStep === 0 && (
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="displayName">Name *</Label>
                    <Input
                      id="displayName"
                      value={formData.displayName}
                      onChange={(e) => updateField('displayName', e.target.value)}
                      placeholder="e.g., Emrys, Dad, Mom"
                      className="mt-1"
                      autoFocus
                    />
                  </div>

                  <div>
                    <Label>Relationship *</Label>
                    <div className="grid grid-cols-5 gap-2 mt-2">
                      {relationshipOptions.map((option) => {
                        const Icon = option.icon
                        return (
                          <button
                            key={option.value}
                            type="button"
                            onClick={() => updateField('relationship', option.value)}
                            className={cn(
                              'p-3 rounded-lg border text-center transition-all',
                              formData.relationship === option.value
                                ? 'border-primary bg-primary/10'
                                : 'border-muted hover:border-muted-foreground/50'
                            )}
                          >
                            <Icon className="h-5 w-5 mx-auto mb-1" />
                            <span className="text-xs">{option.label}</span>
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  <div>
                    <Label>Diabetes Type *</Label>
                    <div className="grid grid-cols-3 gap-2 mt-2">
                      {diabetesTypeOptions.slice(0, 3).map((option) => (
                        <button
                          key={option.value}
                          type="button"
                          onClick={() => updateField('diabetesType', option.value)}
                          className={cn(
                            'p-2 rounded-lg border text-sm transition-all',
                            formData.diabetesType === option.value
                              ? 'border-primary bg-primary/10'
                              : 'border-muted hover:border-muted-foreground/50'
                          )}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="dateOfBirth">Date of Birth</Label>
                      <Input
                        id="dateOfBirth"
                        type="date"
                        value={formData.dateOfBirth}
                        onChange={(e) => updateField('dateOfBirth', e.target.value)}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="diagnosisDate">Diagnosis Date</Label>
                      <Input
                        id="diagnosisDate"
                        type="date"
                        value={formData.diagnosisDate}
                        onChange={(e) => updateField('diagnosisDate', e.target.value)}
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Step 1: Insulin Settings */}
              {currentStep === 1 && (
                <div className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    Enter {formData.displayName || 'their'} typical insulin settings.
                    The AI will learn and refine these over time.
                  </p>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="targetBg">Target BG (mg/dL)</Label>
                      <Input
                        id="targetBg"
                        type="number"
                        value={formData.targetBg}
                        onChange={(e) => updateField('targetBg', parseInt(e.target.value) || 0)}
                        className="mt-1"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Desired glucose level</p>
                    </div>
                    <div>
                      <Label htmlFor="isf">ISF (mg/dL per unit)</Label>
                      <Input
                        id="isf"
                        type="number"
                        value={formData.isf}
                        onChange={(e) => updateField('isf', parseInt(e.target.value) || 0)}
                        className="mt-1"
                      />
                      <p className="text-xs text-muted-foreground mt-1">How much 1U lowers BG</p>
                    </div>
                    <div>
                      <Label htmlFor="icr">Carb Ratio (g per unit)</Label>
                      <Input
                        id="icr"
                        type="number"
                        value={formData.icr}
                        onChange={(e) => updateField('icr', parseInt(e.target.value) || 0)}
                        className="mt-1"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Grams covered by 1U</p>
                    </div>
                    <div>
                      <Label htmlFor="insulinDuration">Insulin Duration (min)</Label>
                      <Input
                        id="insulinDuration"
                        type="number"
                        value={formData.insulinDuration}
                        onChange={(e) => updateField('insulinDuration', parseInt(e.target.value) || 0)}
                        className="mt-1"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Active insulin time</p>
                    </div>
                  </div>

                  <div className="border-t pt-4 mt-4">
                    <Label className="text-sm font-medium">Alert Thresholds</Label>
                    <div className="grid grid-cols-4 gap-3 mt-2">
                      <div>
                        <Label htmlFor="criticalLow" className="text-xs text-red-500">Critical Low</Label>
                        <Input
                          id="criticalLow"
                          type="number"
                          value={formData.criticalLow}
                          onChange={(e) => updateField('criticalLow', parseInt(e.target.value) || 0)}
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                      <div>
                        <Label htmlFor="lowThreshold" className="text-xs text-yellow-500">Low</Label>
                        <Input
                          id="lowThreshold"
                          type="number"
                          value={formData.lowThreshold}
                          onChange={(e) => updateField('lowThreshold', parseInt(e.target.value) || 0)}
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                      <div>
                        <Label htmlFor="highThreshold" className="text-xs text-yellow-500">High</Label>
                        <Input
                          id="highThreshold"
                          type="number"
                          value={formData.highThreshold}
                          onChange={(e) => updateField('highThreshold', parseInt(e.target.value) || 0)}
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                      <div>
                        <Label htmlFor="criticalHigh" className="text-xs text-red-500">Critical High</Label>
                        <Input
                          id="criticalHigh"
                          type="number"
                          value={formData.criticalHigh}
                          onChange={(e) => updateField('criticalHigh', parseInt(e.target.value) || 0)}
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Step 2: Data Source */}
              {currentStep === 2 && (
                <div className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    Connect {formData.displayName || 'their'} CGM data via Gluroo.
                    You can also add this later in settings.
                  </p>

                  <div className="p-4 border rounded-lg space-y-3">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">🔗</span>
                      <div>
                        <h4 className="font-medium">Gluroo (Nightscout Global Connect)</h4>
                        <p className="text-xs text-muted-foreground">
                          Sync glucose and treatments via Nightscout-compatible API
                        </p>
                      </div>
                    </div>

                    <div>
                      <Label htmlFor="glurooUrl">Gluroo/Nightscout URL</Label>
                      <Input
                        id="glurooUrl"
                        value={formData.glurooUrl}
                        onChange={(e) => {
                          updateField('glurooUrl', e.target.value)
                          setConnectionStatus('idle')
                        }}
                        placeholder="https://your-gluroo-url.gluroo.io"
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label htmlFor="glurooApiSecret">API Secret</Label>
                      <Input
                        id="glurooApiSecret"
                        type="password"
                        value={formData.glurooApiSecret}
                        onChange={(e) => {
                          updateField('glurooApiSecret', e.target.value)
                          setConnectionStatus('idle')
                        }}
                        placeholder="Enter your API secret"
                        className="mt-1"
                      />
                    </div>

                    {formData.glurooUrl && formData.glurooApiSecret && (
                      <div className="flex items-center gap-3">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={testConnection}
                          disabled={isTestingConnection}
                        >
                          {isTestingConnection ? (
                            <>
                              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                              Testing...
                            </>
                          ) : (
                            'Test Connection'
                          )}
                        </Button>
                        {connectionStatus === 'success' && (
                          <span className="text-sm text-green-500 flex items-center gap-1">
                            <Check className="h-4 w-4" />
                            Connected!
                          </span>
                        )}
                        {connectionStatus === 'error' && (
                          <span className="text-sm text-red-500 flex items-center gap-1">
                            <X className="h-4 w-4" />
                            Connection failed
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  <p className="text-xs text-muted-foreground text-center">
                    More data sources (Dexcom, Nightscout, etc.) coming soon!
                  </p>
                </div>
              )}

              {/* Step 3: Complete */}
              {currentStep === 3 && (
                <div className="text-center py-6">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-green-500/20 flex items-center justify-center">
                    <Check className="w-8 h-8 text-green-500" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">
                    {formData.displayName}'s Profile Created!
                  </h3>
                  <p className="text-muted-foreground mb-4">
                    You can now switch to this profile to view their data.
                    {formData.glurooUrl && ' Data will start syncing automatically.'}
                  </p>
                  <Button onClick={handleClose} className="mt-2">
                    Done
                  </Button>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Navigation */}
        {currentStep < 3 && (
          <div className="flex items-center justify-between p-6 border-t">
            <Button
              variant="ghost"
              onClick={goBack}
              disabled={currentStep === 0 || isSaving}
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>

            {currentStep < 2 ? (
              <Button onClick={goNext} disabled={!canProceed()}>
                Continue
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            ) : (
              <Button onClick={handleCreate} disabled={isSaving || !canProceed()}>
                {isSaving ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    Create Profile
                    <Check className="w-4 h-4 ml-2" />
                  </>
                )}
              </Button>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

export default AddProfileWizard
