/**
 * Onboarding Wizard
 * 8-step onboarding flow for new users
 */
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  ArrowRight,
  ArrowLeft,
  Check,
  Database,
  User,
  Droplet,
  Bell,
  Shield,
  Brain,
  Loader2,
  Sparkles,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useAuthStore } from '@/stores/authStore'
import DataSourceSelector from '@/components/onboarding/DataSourceSelector'
import GlurooInstructions from '@/components/onboarding/GlurooInstructions'
import AITrainingPath from '@/components/onboarding/AITrainingPath'

const API_URL = import.meta.env.VITE_API_URL || ''

const steps = [
  { id: 'welcome', title: 'Welcome', icon: Sparkles },
  { id: 'datasources', title: 'Data Sources', icon: Database },
  { id: 'connect', title: 'Connect CGM', icon: Activity },
  { id: 'profile', title: 'Your Profile', icon: User },
  { id: 'insulin', title: 'Insulin', icon: Droplet },
  { id: 'alerts', title: 'Alerts', icon: Bell },
  { id: 'aitraining', title: 'AI Training', icon: Brain },
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

export default function Onboarding() {
  const navigate = useNavigate()
  const { user, tokens, sharedWithMe, managedProfiles, setOnboardingCompleted } = useAuthStore()
  const [currentStep, setCurrentStep] = useState(0)
  const [direction, setDirection] = useState(0)
  const [isSaving, setIsSaving] = useState(false)

  // Skip onboarding for follower-only users (have shared profiles, no own data sources)
  useEffect(() => {
    const hasOwnDataSources = managedProfiles.some(p => p.dataSourceCount > 0)
    if (!hasOwnDataSources && sharedWithMe.length > 0) {
      console.log('[Onboarding] Follower-only user detected, skipping onboarding')
      setOnboardingCompleted(true)
      navigate('/dashboard', { replace: true })
    }
  }, [sharedWithMe, managedProfiles, navigate, setOnboardingCompleted])

  // Data sources
  const [currentDataSources, setCurrentDataSources] = useState<string[]>([])
  const [desiredDataSources, setDesiredDataSources] = useState<string[]>([])

  // Gluroo connection
  const [glurooUrl, setGlurooUrl] = useState('')
  const [glurooApiSecret, setGlurooApiSecret] = useState('')
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [connectionError, setConnectionError] = useState('')

  // Profile
  const [formData, setFormData] = useState({
    name: user?.displayName || '',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    diabetesType: 'type1',
    diagnosisYear: '',
    // Insulin Settings
    targetBg: 100,
    insulinSensitivity: 50,
    carbRatio: 10,
    insulinDuration: 180,
    // Alerts
    highThreshold: 180,
    lowThreshold: 70,
    criticalHigh: 250,
    criticalLow: 54,
  })

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

  const testGlurooConnection = async (): Promise<boolean> => {
    setIsTestingConnection(true)
    setConnectionStatus('idle')
    setConnectionError('')

    try {
      const response = await fetch(`${API_URL}/api/datasources/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.accessToken}`,
        },
        body: JSON.stringify({
          nightscoutUrl: glurooUrl,
          apiSecret: glurooApiSecret,
        }),
      })

      const data = await response.json()

      if (response.ok && data.success) {
        setConnectionStatus('success')
        return true
      } else {
        setConnectionStatus('error')
        setConnectionError(data.error || 'Connection test failed')
        return false
      }
    } catch (error) {
      setConnectionStatus('error')
      setConnectionError('Failed to test connection. Please check your network.')
      return false
    } finally {
      setIsTestingConnection(false)
    }
  }

  const handleComplete = async () => {
    // Check if we have valid tokens
    if (!tokens?.accessToken) {
      console.error('No access token available')
      // Mark onboarding as complete locally to prevent loop
      useAuthStore.getState().setOnboardingCompleted(true)
      // Redirect to dashboard - user can configure settings later
      navigate('/dashboard')
      return
    }

    setIsSaving(true)

    try {
      // Save user settings
      const settingsResponse = await fetch(`${API_URL}/api/v1/users/${user?.id}/settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens.accessToken}`,
        },
        body: JSON.stringify({
          displayName: formData.name,
          timezone: formData.timezone,
          targetBg: formData.targetBg,
          isf: formData.insulinSensitivity,
          icr: formData.carbRatio,
          insulinDuration: formData.insulinDuration,
          highThreshold: formData.highThreshold,
          lowThreshold: formData.lowThreshold,
          criticalHigh: formData.criticalHigh,
          criticalLow: formData.criticalLow,
        }),
      })

      if (!settingsResponse.ok && settingsResponse.status === 401) {
        throw new Error('Authentication error - please try logging in again')
      }

      // Save Gluroo connection if provided
      if (glurooUrl && glurooApiSecret) {
        await fetch(`${API_URL}/api/datasources`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${tokens.accessToken}`,
          },
          body: JSON.stringify({
            nightscoutUrl: glurooUrl,
            apiSecret: glurooApiSecret,
          }),
        })
      }

      // Save data source preferences
      if (desiredDataSources.length > 0 || currentDataSources.length > 0) {
        await fetch(`${API_URL}/api/v1/users/${user?.id}/preferences`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${tokens.accessToken}`,
          },
          body: JSON.stringify({
            preferredDataSources: [...currentDataSources, ...desiredDataSources],
          }),
        })
      }

      // Mark onboarding as complete
      await fetch(`${API_URL}/api/v1/users/${user?.id}/onboarding/complete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens.accessToken}`,
        },
      })

      // Update local user state properly via setter
      useAuthStore.getState().setOnboardingCompleted(true)

      navigate('/dashboard')
    } catch (error) {
      console.error('Failed to complete onboarding:', error)
      // Still navigate even if save fails - user can update settings later
      // Mark onboarding complete locally so they don't get stuck in a loop
      useAuthStore.getState().setOnboardingCompleted(true)
      navigate('/dashboard')
    } finally {
      setIsSaving(false)
    }
  }

  const updateField = (field: string, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const canProceed = () => {
    switch (currentStep) {
      case 2: // Connect CGM - optional, can skip
        return true
      case 3: // Profile - name is nice to have but not required
        return true
      case 4: // Insulin - values have defaults
        return formData.targetBg > 0 && formData.insulinSensitivity > 0 && formData.carbRatio > 0
      default:
        return true
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      {/* Progress */}
      <div className="w-full max-w-3xl mb-8 overflow-x-auto">
        <div className="flex items-center justify-between min-w-[600px]">
          {steps.map((step, i) => {
            const Icon = step.icon
            const isComplete = i < currentStep
            const isCurrent = i === currentStep

            return (
              <div key={step.id} className="flex items-center">
                <div
                  className={`
                    w-10 h-10 rounded-full flex items-center justify-center transition-all
                    ${isComplete ? 'bg-cyan text-black' : ''}
                    ${isCurrent ? 'bg-cyan/20 text-cyan border-2 border-cyan' : ''}
                    ${!isComplete && !isCurrent ? 'bg-slate-800 text-gray-500' : ''}
                  `}
                >
                  {isComplete ? <Check className="w-5 h-5" /> : <Icon className="w-5 h-5" />}
                </div>
                {i < steps.length - 1 && (
                  <div
                    className={`w-8 sm:w-12 h-0.5 mx-1 sm:mx-2 ${
                      i < currentStep ? 'bg-cyan' : 'bg-slate-700'
                    }`}
                  />
                )}
              </div>
            )
          })}
        </div>
        <p className="text-center text-sm text-gray-400 mt-2">
          Step {currentStep + 1} of {steps.length}: {steps[currentStep].title}
        </p>
      </div>

      {/* Content */}
      <div className="w-full max-w-2xl glass-card overflow-hidden">
        <AnimatePresence mode="wait" custom={direction}>
          <motion.div
            key={currentStep}
            custom={direction}
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.3 }}
            className="p-8"
          >
            {/* Step 0: Welcome */}
            {currentStep === 0 && (
              <div className="text-center">
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-r from-cyan to-purple-500 flex items-center justify-center">
                  <Sparkles className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-3xl font-bold mb-4 text-white">Welcome to T1D-AI</h2>
                <p className="text-gray-400 mb-6 max-w-md mx-auto">
                  Let's set up your personalized diabetes management assistant. This wizard will
                  help you:
                </p>
                <div className="text-left max-w-sm mx-auto space-y-3 mb-8">
                  <div className="flex items-center gap-3">
                    <Database className="w-5 h-5 text-cyan flex-shrink-0" />
                    <span className="text-gray-300">Connect your CGM data</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Droplet className="w-5 h-5 text-cyan flex-shrink-0" />
                    <span className="text-gray-300">Configure your insulin settings</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Brain className="w-5 h-5 text-cyan flex-shrink-0" />
                    <span className="text-gray-300">Start training your personalized AI</span>
                  </div>
                </div>
                <p className="text-sm text-gray-500">
                  You can always change these settings later in Settings.
                </p>
              </div>
            )}

            {/* Step 1: Data Sources */}
            {currentStep === 1 && (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <Database className="w-8 h-8 text-cyan" />
                  <h2 className="text-2xl font-bold text-white">Data Sources</h2>
                </div>
                <p className="text-gray-400 mb-6">
                  Select the diabetes devices and apps you use. We'll help you connect them. You
                  can also vote for future integrations!
                </p>
                <DataSourceSelector
                  currentSources={currentDataSources}
                  desiredSources={desiredDataSources}
                  onCurrentChange={setCurrentDataSources}
                  onDesiredChange={setDesiredDataSources}
                />
              </div>
            )}

            {/* Step 2: Connect CGM */}
            {currentStep === 2 && (
              <div>
                {currentDataSources.includes('gluroo') ? (
                  <GlurooInstructions
                    nightscoutUrl={glurooUrl}
                    apiSecret={glurooApiSecret}
                    onUrlChange={setGlurooUrl}
                    onSecretChange={setGlurooApiSecret}
                    onTestConnection={testGlurooConnection}
                    isTestingConnection={isTestingConnection}
                    connectionStatus={connectionStatus}
                    connectionError={connectionError}
                  />
                ) : (
                  <div className="text-center py-8">
                    <Database className="w-16 h-16 mx-auto text-gray-600 mb-4" />
                    <h3 className="text-xl font-semibold mb-2">No Active Data Source Selected</h3>
                    <p className="text-gray-400 mb-4">
                      Go back to select Gluroo to connect your CGM, or skip this step if you want to
                      configure it later.
                    </p>
                    <p className="text-sm text-gray-500">
                      You can always add data sources in Settings after setup.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Step 3: Profile */}
            {currentStep === 3 && (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <User className="w-8 h-8 text-cyan" />
                  <h2 className="text-2xl font-bold text-white">Your Profile</h2>
                </div>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="name">Display Name</Label>
                    <Input
                      id="name"
                      value={formData.name}
                      onChange={(e) => updateField('name', e.target.value)}
                      placeholder="What should we call you?"
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                  </div>
                  <div>
                    <Label htmlFor="timezone">Timezone</Label>
                    <Input
                      id="timezone"
                      value={formData.timezone}
                      onChange={(e) => updateField('timezone', e.target.value)}
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                    <p className="text-xs text-gray-500 mt-1">Auto-detected from your browser</p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="diabetesType">Diabetes Type</Label>
                      <select
                        id="diabetesType"
                        value={formData.diabetesType}
                        onChange={(e) => updateField('diabetesType', e.target.value)}
                        className="mt-1 w-full bg-gray-800/50 border border-gray-700 rounded-md px-3 py-2 text-white"
                      >
                        <option value="type1">Type 1</option>
                        <option value="type2">Type 2</option>
                        <option value="lada">LADA</option>
                        <option value="other">Other</option>
                      </select>
                    </div>
                    <div>
                      <Label htmlFor="diagnosisYear">Year Diagnosed (optional)</Label>
                      <Input
                        id="diagnosisYear"
                        type="number"
                        value={formData.diagnosisYear}
                        onChange={(e) => updateField('diagnosisYear', e.target.value)}
                        placeholder="e.g., 2015"
                        className="mt-1 bg-gray-800/50 border-gray-700"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 4: Insulin Settings */}
            {currentStep === 4 && (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <Droplet className="w-8 h-8 text-cyan" />
                  <h2 className="text-2xl font-bold text-white">Insulin Settings</h2>
                </div>
                <p className="text-gray-400 mb-6">
                  Enter your typical insulin settings. These are used for dose calculations.
                  <span className="text-cyan"> The AI will learn and refine these over time.</span>
                </p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="targetBg">Target BG (mg/dL)</Label>
                    <Input
                      id="targetBg"
                      type="number"
                      value={formData.targetBg}
                      onChange={(e) => updateField('targetBg', parseInt(e.target.value) || 0)}
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                    <p className="text-xs text-gray-500 mt-1">Your desired glucose level</p>
                  </div>
                  <div>
                    <Label htmlFor="insulinSensitivity">ISF (mg/dL per unit)</Label>
                    <Input
                      id="insulinSensitivity"
                      type="number"
                      value={formData.insulinSensitivity}
                      onChange={(e) =>
                        updateField('insulinSensitivity', parseInt(e.target.value) || 0)
                      }
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                    <p className="text-xs text-gray-500 mt-1">How much 1 unit lowers BG</p>
                  </div>
                  <div>
                    <Label htmlFor="carbRatio">Carb Ratio (g per unit)</Label>
                    <Input
                      id="carbRatio"
                      type="number"
                      value={formData.carbRatio}
                      onChange={(e) => updateField('carbRatio', parseInt(e.target.value) || 0)}
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                    <p className="text-xs text-gray-500 mt-1">Grams of carbs covered by 1 unit</p>
                  </div>
                  <div>
                    <Label htmlFor="insulinDuration">Insulin Duration (min)</Label>
                    <Input
                      id="insulinDuration"
                      type="number"
                      value={formData.insulinDuration}
                      onChange={(e) =>
                        updateField('insulinDuration', parseInt(e.target.value) || 0)
                      }
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                    <p className="text-xs text-gray-500 mt-1">Active insulin time (DIA)</p>
                  </div>
                </div>
              </div>
            )}

            {/* Step 5: Alerts */}
            {currentStep === 5 && (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <Bell className="w-8 h-8 text-cyan" />
                  <h2 className="text-2xl font-bold text-white">Alert Thresholds</h2>
                </div>
                <p className="text-gray-400 mb-6">
                  Set your glucose alert thresholds. The dashboard will highlight readings outside
                  these ranges.
                </p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="lowThreshold" className="text-yellow-400">
                      Low (mg/dL)
                    </Label>
                    <Input
                      id="lowThreshold"
                      type="number"
                      value={formData.lowThreshold}
                      onChange={(e) => updateField('lowThreshold', parseInt(e.target.value) || 0)}
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                  </div>
                  <div>
                    <Label htmlFor="highThreshold" className="text-yellow-400">
                      High (mg/dL)
                    </Label>
                    <Input
                      id="highThreshold"
                      type="number"
                      value={formData.highThreshold}
                      onChange={(e) => updateField('highThreshold', parseInt(e.target.value) || 0)}
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                  </div>
                  <div>
                    <Label htmlFor="criticalLow" className="text-red-400">
                      Critical Low (mg/dL)
                    </Label>
                    <Input
                      id="criticalLow"
                      type="number"
                      value={formData.criticalLow}
                      onChange={(e) => updateField('criticalLow', parseInt(e.target.value) || 0)}
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                  </div>
                  <div>
                    <Label htmlFor="criticalHigh" className="text-red-400">
                      Critical High (mg/dL)
                    </Label>
                    <Input
                      id="criticalHigh"
                      type="number"
                      value={formData.criticalHigh}
                      onChange={(e) => updateField('criticalHigh', parseInt(e.target.value) || 0)}
                      className="mt-1 bg-gray-800/50 border-gray-700"
                    />
                  </div>
                </div>
                <div className="mt-4 p-3 bg-gray-800/50 rounded-lg">
                  <p className="text-sm text-gray-400">
                    <span className="text-yellow-400">Yellow</span> = Needs attention |{' '}
                    <span className="text-red-400">Red</span> = Urgent
                  </p>
                </div>
              </div>
            )}

            {/* Step 6: AI Training */}
            {currentStep === 6 && (
              <AITrainingPath
                dataStatus={{
                  daysOfData: 0,
                  isfReady: false,
                  icrReady: false,
                  pirReady: false,
                  tftReady: false,
                }}
              />
            )}

            {/* Step 7: Complete */}
            {currentStep === 7 && (
              <div className="text-center">
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-green-500/20 flex items-center justify-center">
                  <Check className="w-10 h-10 text-green-500" />
                </div>
                <h2 className="text-3xl font-bold mb-4 text-white">You're All Set!</h2>
                <p className="text-gray-400 mb-6 max-w-md mx-auto">
                  Your account is ready. T1D-AI will start syncing your data and learning your
                  patterns. The AI gets smarter every day!
                </p>
                <div className="space-y-3 mb-6">
                  <div className="flex items-center justify-center gap-3 text-sm text-gray-400">
                    <Shield className="w-4 h-4 text-cyan" />
                    <span>Your data is encrypted and secure</span>
                  </div>
                  <div className="flex items-center justify-center gap-3 text-sm text-gray-400">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span>AI learns from your data over time</span>
                  </div>
                </div>
                {connectionStatus === 'success' && (
                  <div className="bg-green-500/10 text-green-400 p-3 rounded-lg text-sm">
                    CGM data connection verified and ready!
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Navigation */}
        <div className="flex items-center justify-between p-6 border-t border-gray-800">
          <Button
            variant="ghost"
            onClick={goBack}
            disabled={currentStep === 0 || isSaving}
            className="text-gray-400"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>

          {currentStep < steps.length - 1 ? (
            <Button onClick={goNext} disabled={!canProceed()} className="btn-primary">
              Continue
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          ) : (
            <Button onClick={handleComplete} disabled={isSaving} className="btn-primary">
              {isSaving ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  Go to Dashboard
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          )}
        </div>
      </div>

      {/* Skip link - only show on first few steps */}
      {currentStep < 3 && (
        <button
          onClick={handleComplete}
          className="mt-6 text-gray-500 hover:text-gray-400 text-sm"
        >
          Skip setup for now
        </button>
      )}
    </div>
  )
}
