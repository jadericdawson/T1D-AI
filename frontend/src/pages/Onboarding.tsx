import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link, useNavigate } from 'react-router-dom'
import {
  Activity, ArrowRight, ArrowLeft, Check, Database,
  User, Droplet, Bell, Shield
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

const steps = [
  { id: 'welcome', title: 'Welcome', icon: Activity },
  { id: 'gluroo', title: 'Connect CGM', icon: Database },
  { id: 'profile', title: 'Your Profile', icon: User },
  { id: 'insulin', title: 'Insulin Settings', icon: Droplet },
  { id: 'alerts', title: 'Alerts', icon: Bell },
  { id: 'complete', title: 'Complete', icon: Check },
]

const slideVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? 200 : -200,
    opacity: 0
  }),
  center: {
    x: 0,
    opacity: 1
  },
  exit: (direction: number) => ({
    x: direction < 0 ? 200 : -200,
    opacity: 0
  })
}

export default function Onboarding() {
  const navigate = useNavigate()
  const [currentStep, setCurrentStep] = useState(0)
  const [direction, setDirection] = useState(0)
  const [formData, setFormData] = useState({
    // Gluroo
    glurooUrl: 'https://share.gluroo.com',
    glurooApiSecret: '',
    // Profile
    name: '',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
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

  const handleComplete = () => {
    // TODO: Save settings to API
    navigate('/dashboard')
  }

  const updateField = (field: string, value: string | number) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      {/* Progress */}
      <div className="w-full max-w-2xl mb-8">
        <div className="flex items-center justify-between mb-2">
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
                  <div className={`w-12 h-0.5 mx-2 ${i < currentStep ? 'bg-cyan' : 'bg-slate-700'}`} />
                )}
              </div>
            )
          })}
        </div>
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
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-r from-cyan to-blue-600 flex items-center justify-center">
                  <Activity className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-3xl font-playfair font-bold mb-4 text-white">
                  Welcome to T1D-AI
                </h2>
                <p className="text-gray-400 mb-8 max-w-md mx-auto">
                  Let's set up your account. This will take about 2 minutes.
                  You can always change these settings later.
                </p>
              </div>
            )}

            {/* Step 1: Gluroo Connection */}
            {currentStep === 1 && (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <Database className="w-8 h-8 text-cyan" />
                  <h2 className="text-2xl font-bold text-white">Connect Your CGM</h2>
                </div>
                <p className="text-gray-400 mb-6">
                  T1D-AI connects to Gluroo to sync your glucose data. Enter your
                  Gluroo Nightscout URL and API secret.
                </p>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="glurooUrl">Gluroo URL</Label>
                    <Input
                      id="glurooUrl"
                      value={formData.glurooUrl}
                      onChange={(e) => updateField('glurooUrl', e.target.value)}
                      placeholder="https://share.gluroo.com"
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="glurooApiSecret">API Secret</Label>
                    <Input
                      id="glurooApiSecret"
                      type="password"
                      value={formData.glurooApiSecret}
                      onChange={(e) => updateField('glurooApiSecret', e.target.value)}
                      placeholder="Your API secret"
                      className="mt-1"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Find this in your Gluroo app under Settings → Share
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Step 2: Profile */}
            {currentStep === 2 && (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <User className="w-8 h-8 text-cyan" />
                  <h2 className="text-2xl font-bold text-white">Your Profile</h2>
                </div>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="name">Your Name</Label>
                    <Input
                      id="name"
                      value={formData.name}
                      onChange={(e) => updateField('name', e.target.value)}
                      placeholder="What should we call you?"
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="timezone">Timezone</Label>
                    <Input
                      id="timezone"
                      value={formData.timezone}
                      onChange={(e) => updateField('timezone', e.target.value)}
                      className="mt-1"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Step 3: Insulin Settings */}
            {currentStep === 3 && (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <Droplet className="w-8 h-8 text-cyan" />
                  <h2 className="text-2xl font-bold text-white">Insulin Settings</h2>
                </div>
                <p className="text-gray-400 mb-6">
                  These settings help calculate dose recommendations. Start with your
                  typical values - the AI will learn your patterns over time.
                </p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="targetBg">Target BG (mg/dL)</Label>
                    <Input
                      id="targetBg"
                      type="number"
                      value={formData.targetBg}
                      onChange={(e) => updateField('targetBg', parseInt(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="insulinSensitivity">ISF (mg/dL per unit)</Label>
                    <Input
                      id="insulinSensitivity"
                      type="number"
                      value={formData.insulinSensitivity}
                      onChange={(e) => updateField('insulinSensitivity', parseInt(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="carbRatio">Carb Ratio (g per unit)</Label>
                    <Input
                      id="carbRatio"
                      type="number"
                      value={formData.carbRatio}
                      onChange={(e) => updateField('carbRatio', parseInt(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="insulinDuration">Insulin Duration (min)</Label>
                    <Input
                      id="insulinDuration"
                      type="number"
                      value={formData.insulinDuration}
                      onChange={(e) => updateField('insulinDuration', parseInt(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Step 4: Alerts */}
            {currentStep === 4 && (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <Bell className="w-8 h-8 text-cyan" />
                  <h2 className="text-2xl font-bold text-white">Alert Thresholds</h2>
                </div>
                <p className="text-gray-400 mb-6">
                  Set your glucose alert thresholds. You'll be notified when your
                  glucose is predicted to go outside these ranges.
                </p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="highThreshold">High (mg/dL)</Label>
                    <Input
                      id="highThreshold"
                      type="number"
                      value={formData.highThreshold}
                      onChange={(e) => updateField('highThreshold', parseInt(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="lowThreshold">Low (mg/dL)</Label>
                    <Input
                      id="lowThreshold"
                      type="number"
                      value={formData.lowThreshold}
                      onChange={(e) => updateField('lowThreshold', parseInt(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="criticalHigh">Critical High (mg/dL)</Label>
                    <Input
                      id="criticalHigh"
                      type="number"
                      value={formData.criticalHigh}
                      onChange={(e) => updateField('criticalHigh', parseInt(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="criticalLow">Critical Low (mg/dL)</Label>
                    <Input
                      id="criticalLow"
                      type="number"
                      value={formData.criticalLow}
                      onChange={(e) => updateField('criticalLow', parseInt(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Step 5: Complete */}
            {currentStep === 5 && (
              <div className="text-center">
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-green-500/20 flex items-center justify-center">
                  <Check className="w-10 h-10 text-green-500" />
                </div>
                <h2 className="text-3xl font-playfair font-bold mb-4 text-white">
                  You're All Set!
                </h2>
                <p className="text-gray-400 mb-8 max-w-md mx-auto">
                  Your account is ready. T1D-AI will start syncing your data and
                  learning your patterns. The AI gets smarter every day!
                </p>
                <div className="flex items-center justify-center gap-3 text-sm text-gray-500">
                  <Shield className="w-4 h-4" />
                  <span>Your data is encrypted and secure</span>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Navigation */}
        <div className="flex items-center justify-between p-6 border-t border-gray-800">
          <Button
            variant="ghost"
            onClick={goBack}
            disabled={currentStep === 0}
            className="text-gray-400"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>

          {currentStep < steps.length - 1 ? (
            <Button onClick={goNext} className="btn-primary">
              Continue
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          ) : (
            <Button onClick={handleComplete} className="btn-primary">
              Go to Dashboard
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          )}
        </div>
      </div>

      {/* Skip */}
      {currentStep < steps.length - 1 && (
        <Link to="/dashboard" className="mt-6 text-gray-500 hover:text-gray-400 text-sm">
          Skip for now
        </Link>
      )}
    </div>
  )
}
