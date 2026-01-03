/**
 * Settings Page
 * User settings and preferences management
 */
import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  Activity, ArrowLeft, User, Database, Droplet, Bell,
  Moon, Trash2, Save, Loader2, Check, AlertCircle,
  RefreshCw, ExternalLink
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { useGlucoseStore } from '@/stores/glucoseStore'
import { usersApi, datasourcesApi } from '@/lib/api'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

const USER_ID = 'demo_user'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } }
}

export default function Settings() {
  const queryClient = useQueryClient()
  const { preferences, updatePreferences } = useGlucoseStore()

  // Form state
  const [formState, setFormState] = useState({
    // Profile
    name: 'Demo User',
    email: 'demo@example.com',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,

    // Data Source (Gluroo)
    glurooUrl: '',
    glurooApiSecret: '',
    syncInterval: 5,

    // Insulin Settings
    targetBg: preferences.targetBg,
    insulinSensitivity: 50,
    carbRatio: 10,
    insulinDuration: 180,
    carbAbsorptionDuration: 180,

    // Alerts
    highThreshold: preferences.highThreshold,
    lowThreshold: preferences.lowThreshold,
    criticalHighThreshold: preferences.criticalHighThreshold,
    criticalLowThreshold: preferences.criticalLowThreshold,
    enableAlerts: preferences.enableAlerts,
    soundEnabled: true,
    predictiveAlerts: preferences.enablePredictiveAlerts,

    // Display
    darkMode: preferences.darkMode,
    showPredictions: true,
    showInsights: true,
  })

  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle')
  const [glurooStatus, setGlurooStatus] = useState<'idle' | 'testing' | 'success' | 'error' | 'syncing'>('idle')

  // Fetch user settings
  const { data: userSettings, isLoading: isLoadingSettings } = useQuery({
    queryKey: ['user', 'settings', USER_ID],
    queryFn: () => usersApi.getSettings(USER_ID),
    retry: false,
  })

  // Update form state when settings load
  useEffect(() => {
    if (userSettings?.settings) {
      const s = userSettings.settings
      setFormState(prev => ({
        ...prev,
        targetBg: s.targetBg ?? prev.targetBg,
        insulinSensitivity: s.insulinSensitivity ?? prev.insulinSensitivity,
        carbRatio: s.carbRatio ?? prev.carbRatio,
        insulinDuration: s.insulinDuration ?? prev.insulinDuration,
        highThreshold: s.highThreshold ?? prev.highThreshold,
        lowThreshold: s.lowThreshold ?? prev.lowThreshold,
        criticalHighThreshold: s.criticalHighThreshold ?? prev.criticalHighThreshold,
        criticalLowThreshold: s.criticalLowThreshold ?? prev.criticalLowThreshold,
        enableAlerts: s.enableAlerts ?? prev.enableAlerts,
        showInsights: s.showInsights ?? prev.showInsights,
      }))
    }
  }, [userSettings])

  // Save settings mutation
  const saveMutation = useMutation({
    mutationFn: async () => {
      // Update backend
      await usersApi.updateSettings(USER_ID, {
        targetBg: formState.targetBg,
        insulinSensitivity: formState.insulinSensitivity,
        carbRatio: formState.carbRatio,
        insulinDuration: formState.insulinDuration,
        highThreshold: formState.highThreshold,
        lowThreshold: formState.lowThreshold,
        criticalHighThreshold: formState.criticalHighThreshold,
        criticalLowThreshold: formState.criticalLowThreshold,
        enableAlerts: formState.enableAlerts,
        enablePredictiveAlerts: formState.predictiveAlerts,
        showInsights: formState.showInsights,
      })

      // Update Zustand store
      updatePreferences({
        targetBg: formState.targetBg,
        highThreshold: formState.highThreshold,
        lowThreshold: formState.lowThreshold,
        criticalHighThreshold: formState.criticalHighThreshold,
        criticalLowThreshold: formState.criticalLowThreshold,
        enableAlerts: formState.enableAlerts,
        enablePredictiveAlerts: formState.predictiveAlerts,
        darkMode: formState.darkMode,
      })
    },
    onSuccess: () => {
      setSaveStatus('success')
      queryClient.invalidateQueries({ queryKey: ['user', 'settings'] })
      setTimeout(() => setSaveStatus('idle'), 2000)
    },
    onError: () => {
      setSaveStatus('error')
      setTimeout(() => setSaveStatus('idle'), 3000)
    },
  })

  // Test Gluroo connection
  const handleTestGluroo = async () => {
    if (!formState.glurooUrl || !formState.glurooApiSecret) {
      return
    }

    setGlurooStatus('testing')
    try {
      await datasourcesApi.connectGluroo(USER_ID, formState.glurooUrl, formState.glurooApiSecret)
      await datasourcesApi.testGluroo(USER_ID)
      setGlurooStatus('success')
      setTimeout(() => setGlurooStatus('idle'), 2000)
    } catch {
      setGlurooStatus('error')
      setTimeout(() => setGlurooStatus('idle'), 3000)
    }
  }

  // Sync data from Gluroo
  const handleSyncGluroo = async () => {
    setGlurooStatus('syncing')
    try {
      await datasourcesApi.sync(USER_ID)
      setGlurooStatus('success')
      queryClient.invalidateQueries({ queryKey: ['glucose'] })
      queryClient.invalidateQueries({ queryKey: ['treatments'] })
      setTimeout(() => setGlurooStatus('idle'), 2000)
    } catch {
      setGlurooStatus('error')
      setTimeout(() => setGlurooStatus('idle'), 3000)
    }
  }

  const updateField = (key: string, value: string | number | boolean) => {
    setFormState(prev => ({ ...prev, [key]: value }))
  }

  const handleSave = () => {
    setSaveStatus('saving')
    saveMutation.mutate()
  }

  if (isLoadingSettings) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-cyan" />
      </div>
    )
  }

  return (
    <div className="min-h-screen p-6 max-w-4xl mx-auto">
      {/* Header */}
      <motion.header
        initial="hidden"
        animate="visible"
        variants={fadeIn}
        className="flex items-center justify-between mb-8"
      >
        <div className="flex items-center gap-4">
          <Link to="/dashboard">
            <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white">
              <ArrowLeft className="w-5 h-5" />
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <Activity className="w-6 h-6 text-cyan" />
            <h1 className="text-2xl font-bold text-white">Settings</h1>
          </div>
        </div>

        {/* Save Button - Always visible */}
        <Button
          onClick={handleSave}
          disabled={saveStatus === 'saving'}
          className={cn(
            'min-w-32',
            saveStatus === 'success' && 'bg-green-600 hover:bg-green-700',
            saveStatus === 'error' && 'bg-red-600 hover:bg-red-700'
          )}
        >
          {saveStatus === 'saving' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
          {saveStatus === 'success' && <Check className="w-4 h-4 mr-2" />}
          {saveStatus === 'error' && <AlertCircle className="w-4 h-4 mr-2" />}
          {saveStatus === 'idle' && <Save className="w-4 h-4 mr-2" />}
          {saveStatus === 'saving' ? 'Saving...' :
            saveStatus === 'success' ? 'Saved!' :
              saveStatus === 'error' ? 'Error' : 'Save Changes'}
        </Button>
      </motion.header>

      {/* Tabs */}
      <Tabs defaultValue="profile" className="space-y-6">
        <TabsList className="bg-slate-800/50 w-full justify-start flex-wrap">
          <TabsTrigger value="profile" className="data-[state=active]:bg-cyan/20 data-[state=active]:text-cyan">
            <User className="w-4 h-4 mr-2" />
            Profile
          </TabsTrigger>
          <TabsTrigger value="datasource" className="data-[state=active]:bg-cyan/20 data-[state=active]:text-cyan">
            <Database className="w-4 h-4 mr-2" />
            Data Source
          </TabsTrigger>
          <TabsTrigger value="insulin" className="data-[state=active]:bg-cyan/20 data-[state=active]:text-cyan">
            <Droplet className="w-4 h-4 mr-2" />
            Insulin
          </TabsTrigger>
          <TabsTrigger value="alerts" className="data-[state=active]:bg-cyan/20 data-[state=active]:text-cyan">
            <Bell className="w-4 h-4 mr-2" />
            Alerts
          </TabsTrigger>
          <TabsTrigger value="display" className="data-[state=active]:bg-cyan/20 data-[state=active]:text-cyan">
            <Moon className="w-4 h-4 mr-2" />
            Display
          </TabsTrigger>
        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="profile">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass-card space-y-6"
          >
            <h2 className="text-xl font-bold text-white">Profile Settings</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <Label htmlFor="name" className="text-gray-300">Name</Label>
                <Input
                  id="name"
                  value={formState.name}
                  onChange={(e) => updateField('name', e.target.value)}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
              </div>
              <div>
                <Label htmlFor="email" className="text-gray-300">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={formState.email}
                  onChange={(e) => updateField('email', e.target.value)}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
              </div>
              <div className="md:col-span-2">
                <Label htmlFor="timezone" className="text-gray-300">Timezone</Label>
                <Input
                  id="timezone"
                  value={formState.timezone}
                  onChange={(e) => updateField('timezone', e.target.value)}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
              </div>
            </div>
          </motion.div>
        </TabsContent>

        {/* Data Source Tab */}
        <TabsContent value="datasource">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass-card space-y-6"
          >
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-white">Gluroo Connection</h2>
              <a
                href="https://gluroo.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-cyan hover:underline text-sm flex items-center gap-1"
              >
                About Gluroo <ExternalLink className="w-3 h-3" />
              </a>
            </div>

            <p className="text-gray-400">
              Connect your Gluroo account to automatically sync glucose readings and treatments.
            </p>

            <div className="space-y-4">
              <div>
                <Label htmlFor="glurooUrl" className="text-gray-300">Gluroo URL</Label>
                <Input
                  id="glurooUrl"
                  value={formState.glurooUrl}
                  onChange={(e) => updateField('glurooUrl', e.target.value)}
                  placeholder="https://share.gluroo.com"
                  className="mt-1 bg-slate-800 border-gray-700"
                />
              </div>
              <div>
                <Label htmlFor="glurooApiSecret" className="text-gray-300">API Secret</Label>
                <Input
                  id="glurooApiSecret"
                  type="password"
                  value={formState.glurooApiSecret}
                  onChange={(e) => updateField('glurooApiSecret', e.target.value)}
                  placeholder="Enter your Gluroo API secret"
                  className="mt-1 bg-slate-800 border-gray-700"
                />
              </div>
              <div>
                <Label htmlFor="syncInterval" className="text-gray-300">Sync Interval (minutes)</Label>
                <Input
                  id="syncInterval"
                  type="number"
                  value={formState.syncInterval}
                  onChange={(e) => updateField('syncInterval', parseInt(e.target.value))}
                  min={1}
                  max={60}
                  className="mt-1 bg-slate-800 border-gray-700 w-32"
                />
              </div>

              <div className="flex gap-3 pt-2">
                <Button
                  variant="outline"
                  onClick={handleTestGluroo}
                  disabled={glurooStatus === 'testing' || !formState.glurooUrl || !formState.glurooApiSecret}
                  className="border-gray-700"
                >
                  {glurooStatus === 'testing' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                  {glurooStatus === 'success' && <Check className="w-4 h-4 mr-2 text-green-500" />}
                  {glurooStatus === 'error' && <AlertCircle className="w-4 h-4 mr-2 text-red-500" />}
                  Test Connection
                </Button>
                <Button
                  variant="outline"
                  onClick={handleSyncGluroo}
                  disabled={glurooStatus === 'syncing'}
                  className="border-gray-700"
                >
                  {glurooStatus === 'syncing' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                  <RefreshCw className={cn('w-4 h-4 mr-2', glurooStatus !== 'syncing' && 'block')} />
                  Sync Now
                </Button>
              </div>

              {glurooStatus === 'success' && (
                <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                  Connected successfully
                </Badge>
              )}
              {glurooStatus === 'error' && (
                <Badge className="bg-red-500/20 text-red-400 border-red-500/30">
                  Connection failed - check credentials
                </Badge>
              )}
            </div>
          </motion.div>
        </TabsContent>

        {/* Insulin Tab */}
        <TabsContent value="insulin">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass-card space-y-6"
          >
            <h2 className="text-xl font-bold text-white">Insulin Settings</h2>
            <p className="text-gray-400">
              These values are used for dose calculations. The AI will learn your patterns
              over time and adjust ISF predictions dynamically.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <Label htmlFor="targetBg" className="text-gray-300">Target BG (mg/dL)</Label>
                <Input
                  id="targetBg"
                  type="number"
                  value={formState.targetBg}
                  onChange={(e) => updateField('targetBg', parseInt(e.target.value))}
                  min={70}
                  max={150}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
                <p className="text-xs text-gray-500 mt-1">Goal glucose level for corrections</p>
              </div>
              <div>
                <Label htmlFor="insulinSensitivity" className="text-gray-300">Default ISF (mg/dL per unit)</Label>
                <Input
                  id="insulinSensitivity"
                  type="number"
                  value={formState.insulinSensitivity}
                  onChange={(e) => updateField('insulinSensitivity', parseInt(e.target.value))}
                  min={10}
                  max={200}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
                <p className="text-xs text-gray-500 mt-1">Fallback when ML prediction unavailable</p>
              </div>
              <div>
                <Label htmlFor="carbRatio" className="text-gray-300">Carb Ratio (g per unit)</Label>
                <Input
                  id="carbRatio"
                  type="number"
                  value={formState.carbRatio}
                  onChange={(e) => updateField('carbRatio', parseInt(e.target.value))}
                  min={1}
                  max={50}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
                <p className="text-xs text-gray-500 mt-1">Grams of carbs covered by 1 unit</p>
              </div>
              <div>
                <Label htmlFor="insulinDuration" className="text-gray-300">Insulin Duration (min)</Label>
                <Input
                  id="insulinDuration"
                  type="number"
                  value={formState.insulinDuration}
                  onChange={(e) => updateField('insulinDuration', parseInt(e.target.value))}
                  min={120}
                  max={360}
                  step={30}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
                <p className="text-xs text-gray-500 mt-1">Active insulin time (default: 180 min)</p>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/30 mt-4">
              <p className="text-sm text-purple-300">
                <strong>Note:</strong> The ML model dynamically predicts your ISF based on time of day,
                recent activity, and glucose patterns. The default ISF above is used as a fallback
                when the model is unavailable.
              </p>
            </div>
          </motion.div>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass-card space-y-6"
          >
            <h2 className="text-xl font-bold text-white">Alert Settings</h2>

            {/* Toggles */}
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50">
                <div>
                  <p className="font-medium text-white">Enable Alerts</p>
                  <p className="text-sm text-gray-400">Receive notifications for glucose events</p>
                </div>
                <Switch
                  checked={formState.enableAlerts}
                  onCheckedChange={(checked) => updateField('enableAlerts', checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50">
                <div>
                  <p className="font-medium text-white">Sound Effects</p>
                  <p className="text-sm text-gray-400">Play sound for critical alerts</p>
                </div>
                <Switch
                  checked={formState.soundEnabled}
                  onCheckedChange={(checked) => updateField('soundEnabled', checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50">
                <div>
                  <p className="font-medium text-white">Predictive Alerts</p>
                  <p className="text-sm text-gray-400">Alert based on predicted low/high values</p>
                </div>
                <Switch
                  checked={formState.predictiveAlerts}
                  onCheckedChange={(checked) => updateField('predictiveAlerts', checked)}
                />
              </div>
            </div>

            {/* Thresholds */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-gray-700">
              <div>
                <Label htmlFor="highThreshold" className="text-gray-300">High Threshold (mg/dL)</Label>
                <Input
                  id="highThreshold"
                  type="number"
                  value={formState.highThreshold}
                  onChange={(e) => updateField('highThreshold', parseInt(e.target.value))}
                  min={140}
                  max={300}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
              </div>
              <div>
                <Label htmlFor="lowThreshold" className="text-gray-300">Low Threshold (mg/dL)</Label>
                <Input
                  id="lowThreshold"
                  type="number"
                  value={formState.lowThreshold}
                  onChange={(e) => updateField('lowThreshold', parseInt(e.target.value))}
                  min={60}
                  max={100}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
              </div>
              <div>
                <Label htmlFor="criticalHighThreshold" className="text-gray-300">Critical High (mg/dL)</Label>
                <Input
                  id="criticalHighThreshold"
                  type="number"
                  value={formState.criticalHighThreshold}
                  onChange={(e) => updateField('criticalHighThreshold', parseInt(e.target.value))}
                  min={200}
                  max={400}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
                <p className="text-xs text-red-400 mt-1">Urgent alert threshold</p>
              </div>
              <div>
                <Label htmlFor="criticalLowThreshold" className="text-gray-300">Critical Low (mg/dL)</Label>
                <Input
                  id="criticalLowThreshold"
                  type="number"
                  value={formState.criticalLowThreshold}
                  onChange={(e) => updateField('criticalLowThreshold', parseInt(e.target.value))}
                  min={40}
                  max={70}
                  className="mt-1 bg-slate-800 border-gray-700"
                />
                <p className="text-xs text-red-400 mt-1">Urgent alert threshold</p>
              </div>
            </div>
          </motion.div>
        </TabsContent>

        {/* Display Tab */}
        <TabsContent value="display">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass-card space-y-6"
          >
            <h2 className="text-xl font-bold text-white">Display Settings</h2>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50">
                <div>
                  <p className="font-medium text-white">Dark Mode</p>
                  <p className="text-sm text-gray-400">Use dark theme (recommended)</p>
                </div>
                <Switch
                  checked={formState.darkMode}
                  onCheckedChange={(checked) => updateField('darkMode', checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50">
                <div>
                  <p className="font-medium text-white">Show Predictions</p>
                  <p className="text-sm text-gray-400">Display ML prediction values on dashboard</p>
                </div>
                <Switch
                  checked={formState.showPredictions}
                  onCheckedChange={(checked) => updateField('showPredictions', checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50">
                <div>
                  <p className="font-medium text-white">Show AI Insights</p>
                  <p className="text-sm text-gray-400">Display GPT-powered pattern insights</p>
                </div>
                <Switch
                  checked={formState.showInsights}
                  onCheckedChange={(checked) => updateField('showInsights', checked)}
                />
              </div>
            </div>
          </motion.div>
        </TabsContent>
      </Tabs>

      {/* Footer actions */}
      <div className="flex justify-between items-center mt-8 pt-6 border-t border-gray-800">
        <Button variant="ghost" className="text-red-400 hover:text-red-300 hover:bg-red-400/10">
          <Trash2 className="w-4 h-4 mr-2" />
          Delete Account
        </Button>

        <p className="text-sm text-gray-500">
          Changes are saved automatically when you click "Save Changes"
        </p>
      </div>
    </div>
  )
}
