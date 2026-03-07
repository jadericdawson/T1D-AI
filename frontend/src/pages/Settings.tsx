/**
 * Settings Page
 * Comprehensive user settings and preferences management
 */
import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Link, useSearchParams, useNavigate } from 'react-router-dom'
import {
  Activity, ArrowLeft, User, Droplet, Bell,
  Moon, Trash2, Save, Loader2, Check, AlertCircle,
  RefreshCw, ExternalLink, LogOut, Shield,
  Share2, Link2, Settings2, Wifi, HelpCircle, Users, Plug
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { ResponsiveLayout } from '@/components/layout/ResponsiveLayout'
import { SharingManager } from '@/components/sharing/SharingManager'
import { ProfilesManager } from '@/components/profiles/ProfilesManager'
import { HomeAssistantKeys } from '@/components/settings/HomeAssistantKeys'
import { cn } from '@/lib/utils'
import { useGlucoseStore } from '@/stores/glucoseStore'
import { useAuthStore } from '@/stores/authStore'
import { usersApi, datasourcesApi } from '@/lib/api'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useIsMobile } from '@/hooks/useResponsive'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } }
}

// Tab configuration
const TABS = [
  { id: 'profile', label: 'Profile', icon: User },
  { id: 'profiles', label: 'Profiles', icon: Users },
  { id: 'datasource', label: 'Data Sources', icon: Link2 },
  { id: 'insulin', label: 'Insulin', icon: Droplet },
  { id: 'alerts', label: 'Alerts', icon: Bell },
  { id: 'sharing', label: 'Sharing', icon: Share2 },
  { id: 'integrations', label: 'Integrations', icon: Plug },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'display', label: 'Display', icon: Moon },
]

export default function Settings() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const queryClient = useQueryClient()
  const isMobile = useIsMobile()
  const { preferences, updatePreferences } = useGlucoseStore()

  // Auth store
  const user = useAuthStore(state => state.user)
  const logout = useAuthStore(state => state.logout)
  const setTimezone = useAuthStore(state => state.setTimezone)
  const userId = user?.id || ''

  // Get active tab from URL or default
  const activeTab = searchParams.get('tab') || 'profile'

  const setActiveTab = (tab: string) => {
    setSearchParams({ tab })
  }

  // Handle logout
  const handleLogout = () => {
    logout()
    navigate('/')
  }

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

    // Data Source (Tandem)
    tandemEmail: '',
    tandemPassword: '',

    // Insulin Settings
    targetBg: preferences.targetBg,
    insulinSensitivity: 50,
    carbRatio: 10,
    insulinDuration: 180,

    // ICR/PIR Learning Settings
    useLearnedICR: false,
    useLearnedPIR: false,
    includeProteinInBolus: false,
    proteinRatio: 25,
    proteinUpfrontPercent: 40,

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

    // Prediction Settings
    useTFTModifiers: true,
    trackPredictionAccuracy: true,
  })

  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle')
  const [glurooStatus, setGlurooStatus] = useState<'idle' | 'testing' | 'success' | 'error' | 'syncing'>('idle')
  const [tandemStatus, setTandemStatus] = useState<'idle' | 'testing' | 'success' | 'error' | 'syncing'>('idle')
  const [tandemConnected, setTandemConnected] = useState(false)

  // Fetch user settings
  const { data: userSettings, isLoading: isLoadingSettings } = useQuery({
    queryKey: ['user', 'settings', userId],
    queryFn: () => usersApi.getSettings(userId),
    enabled: !!userId,
    retry: false,
  })

  // Fetch Gluroo defaults for auto-fill
  const { data: glurooDefaults } = useQuery({
    queryKey: ['gluroo', 'defaults', userId],
    queryFn: () => datasourcesApi.getDefaults(),
    enabled: !!userId,
    retry: false,
  })

  // Fetch Tandem connection status
  const { data: tandemStatusData } = useQuery({
    queryKey: ['tandem', 'status', userId],
    queryFn: () => datasourcesApi.getTandemStatus(),
    enabled: !!userId,
    retry: false,
  })

  // Auto-fill Gluroo credentials when defaults load
  useEffect(() => {
    if (glurooDefaults && glurooDefaults.isOwner && !formState.glurooUrl) {
      setFormState(prev => ({
        ...prev,
        glurooUrl: glurooDefaults.url,
        glurooApiSecret: glurooDefaults.apiSecret,
        syncInterval: glurooDefaults.syncInterval,
      }))
    }
  }, [glurooDefaults])

  // Update Tandem connected state
  useEffect(() => {
    if (tandemStatusData) {
      setTandemConnected(tandemStatusData.connected)
    }
  }, [tandemStatusData])

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
        useTFTModifiers: s.useTFTModifiers ?? prev.useTFTModifiers,
        trackPredictionAccuracy: s.trackPredictionAccuracy ?? prev.trackPredictionAccuracy,
      }))
    }
  }, [userSettings])

  // Save settings mutation
  const saveMutation = useMutation({
    mutationFn: async () => {
      await usersApi.updateSettings(userId, {
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
        useTFTModifiers: formState.useTFTModifiers,
        trackPredictionAccuracy: formState.trackPredictionAccuracy,
        timezone: formState.timezone,
      })
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
      // Persist timezone preference to auth store
      if (formState.timezone) {
        setTimezone(formState.timezone)
      }
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

  // Gluroo handlers
  const handleTestGluroo = async () => {
    if (!formState.glurooUrl || !formState.glurooApiSecret) return
    setGlurooStatus('testing')
    try {
      await datasourcesApi.testGluroo(formState.glurooUrl, formState.glurooApiSecret)
      setGlurooStatus('success')
      setTimeout(() => setGlurooStatus('idle'), 2000)
    } catch {
      setGlurooStatus('error')
      setTimeout(() => setGlurooStatus('idle'), 3000)
    }
  }

  const handleSyncGluroo = async () => {
    setGlurooStatus('syncing')
    try {
      await datasourcesApi.sync()
      setGlurooStatus('success')
      queryClient.invalidateQueries({ queryKey: ['glucose'] })
      queryClient.invalidateQueries({ queryKey: ['treatments'] })
      setTimeout(() => setGlurooStatus('idle'), 2000)
    } catch {
      setGlurooStatus('error')
      setTimeout(() => setGlurooStatus('idle'), 3000)
    }
  }

  // Tandem handlers
  const handleTestTandem = async () => {
    if (!formState.tandemEmail || !formState.tandemPassword) return
    setTandemStatus('testing')
    try {
      const result = await datasourcesApi.testTandem(formState.tandemEmail, formState.tandemPassword)
      if (result.success) {
        setTandemStatus('success')
        setTimeout(() => setTandemStatus('idle'), 2000)
      } else {
        setTandemStatus('error')
        setTimeout(() => setTandemStatus('idle'), 3000)
      }
    } catch {
      setTandemStatus('error')
      setTimeout(() => setTandemStatus('idle'), 3000)
    }
  }

  const handleConnectTandem = async () => {
    if (!formState.tandemEmail || !formState.tandemPassword) return
    setTandemStatus('testing')
    try {
      await datasourcesApi.connectTandem(formState.tandemEmail, formState.tandemPassword)
      setTandemStatus('success')
      setTandemConnected(true)
      queryClient.invalidateQueries({ queryKey: ['tandem', 'status'] })
      setTimeout(() => setTandemStatus('idle'), 2000)
    } catch {
      setTandemStatus('error')
      setTimeout(() => setTandemStatus('idle'), 3000)
    }
  }

  const handleDisconnectTandem = async () => {
    try {
      await datasourcesApi.disconnectTandem()
      setTandemConnected(false)
      setFormState(prev => ({ ...prev, tandemEmail: '', tandemPassword: '' }))
      queryClient.invalidateQueries({ queryKey: ['tandem', 'status'] })
    } catch {
      setTandemStatus('error')
      setTimeout(() => setTandemStatus('idle'), 3000)
    }
  }

  const handleSyncTandem = async (fullSync: boolean = false) => {
    setTandemStatus('syncing')
    try {
      await datasourcesApi.syncTandem(fullSync)
      setTandemStatus('success')
      queryClient.invalidateQueries({ queryKey: ['glucose'] })
      queryClient.invalidateQueries({ queryKey: ['treatments'] })
      setTimeout(() => setTandemStatus('idle'), 2000)
    } catch {
      setTandemStatus('error')
      setTimeout(() => setTandemStatus('idle'), 3000)
    }
  }

  const updateField = (key: string, value: string | number | boolean) => {
    setFormState(prev => ({ ...prev, [key]: value }))
  }

  const handleSave = () => {
    setSaveStatus('saving')
    saveMutation.mutate()
  }

  // Get user initials
  const initials = user?.displayName
    ? user.displayName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : user?.email?.slice(0, 2).toUpperCase() || 'U'

  if (isLoadingSettings) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <ResponsiveLayout title="Settings" hideBottomNav>
      <div className="min-h-screen p-4 md:p-6 max-w-6xl mx-auto">
        {/* Header - Desktop Only */}
        {!isMobile && (
          <motion.header
            initial="hidden"
            animate="visible"
            variants={fadeIn}
            className="flex items-center justify-between mb-6"
          >
            <div className="flex items-center gap-4">
              <Link to="/dashboard">
                <Button variant="ghost" size="icon">
                  <ArrowLeft className="w-5 h-5" />
                </Button>
              </Link>
              <div className="flex items-center gap-2">
                <Settings2 className="w-6 h-6 text-primary" />
                <h1 className="text-2xl font-bold">Settings</h1>
              </div>
            </div>
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
        )}

        {/* Mobile Save Button */}
        {isMobile && (
          <div className="fixed bottom-20 left-4 right-4 z-40">
            <Button
              onClick={handleSave}
              disabled={saveStatus === 'saving'}
              className={cn(
                'w-full shadow-lg',
                saveStatus === 'success' && 'bg-green-600',
                saveStatus === 'error' && 'bg-red-600'
              )}
            >
              {saveStatus === 'saving' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              {saveStatus === 'success' && <Check className="w-4 h-4 mr-2" />}
              {saveStatus === 'idle' && <Save className="w-4 h-4 mr-2" />}
              {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'success' ? 'Saved!' : 'Save Changes'}
            </Button>
          </div>
        )}

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <ScrollArea className="w-full">
            <TabsList className="bg-muted/50 w-full justify-start gap-1 p-1">
              {TABS.map((tab) => {
                const Icon = tab.icon
                return (
                  <TabsTrigger
                    key={tab.id}
                    value={tab.id}
                    className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary gap-2 whitespace-nowrap"
                  >
                    <Icon className="w-4 h-4" />
                    <span className={isMobile ? 'hidden sm:inline' : ''}>{tab.label}</span>
                  </TabsTrigger>
                )
              })}
            </TabsList>
          </ScrollArea>

          {/* Profile Tab */}
          <TabsContent value="profile">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Profile Settings</CardTitle>
                    <Button variant="outline" onClick={handleLogout} className="text-destructive border-destructive/30">
                      <LogOut className="w-4 h-4 mr-2" />
                      Logout
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* User Card */}
                  {user && (
                    <div className="p-4 rounded-lg bg-primary/10 border border-primary/30">
                      <div className="flex items-center gap-4">
                        <div className="w-14 h-14 rounded-full bg-primary/20 flex items-center justify-center">
                          <span className="text-xl font-semibold text-primary">{initials}</span>
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-semibold text-lg">{user.displayName || user.email}</p>
                          <p className="text-sm text-muted-foreground">{user.email}</p>
                          <p className="text-xs text-muted-foreground/70 mt-1 font-mono truncate">ID: {user.id}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <Label>Display Name</Label>
                      <Input value={user?.displayName || user?.email || ''} disabled className="mt-1 opacity-60" />
                      <p className="text-xs text-muted-foreground mt-1">Managed by Microsoft account</p>
                    </div>
                    <div>
                      <Label>Email</Label>
                      <Input type="email" value={user?.email || ''} disabled className="mt-1 opacity-60" />
                      <p className="text-xs text-muted-foreground mt-1">Managed by Microsoft account</p>
                    </div>
                    <div className="md:col-span-2">
                      <Label>Timezone</Label>
                      <Input value={formState.timezone} onChange={(e) => updateField('timezone', e.target.value)} className="mt-1" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Managed Profiles Tab */}
          <TabsContent value="profiles">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Users className="w-5 h-5 text-cyan-500" />
                    Managed Profiles
                  </CardTitle>
                  <CardDescription>
                    Manage profiles for people whose diabetes data you track. Each profile can have its own data sources.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ProfilesManager />
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Gluroo Data Source Tab */}
          <TabsContent value="datasource">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <Link2 className="w-5 h-5 text-blue-500" />
                        Gluroo Connection
                      </CardTitle>
                      <CardDescription>
                        Connect your Gluroo account to sync glucose data automatically
                      </CardDescription>
                    </div>
                    <a href="https://gluroo.com" target="_blank" rel="noopener noreferrer"
                      className="text-primary hover:underline text-sm flex items-center gap-1">
                      About Gluroo <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div>
                      <Label>Gluroo URL</Label>
                      <Input
                        value={formState.glurooUrl}
                        onChange={(e) => updateField('glurooUrl', e.target.value)}
                        placeholder="https://share.gluroo.com"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>API Secret</Label>
                      <Input
                        type="password"
                        value={formState.glurooApiSecret}
                        onChange={(e) => updateField('glurooApiSecret', e.target.value)}
                        placeholder="Enter your Gluroo API secret"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Sync Interval (minutes)</Label>
                      <Input
                        type="number"
                        value={formState.syncInterval}
                        onChange={(e) => updateField('syncInterval', parseInt(e.target.value))}
                        min={1}
                        max={60}
                        className="mt-1 w-32"
                      />
                    </div>

                    <div className="flex gap-3 pt-2">
                      <Button
                        variant="outline"
                        onClick={handleTestGluroo}
                        disabled={glurooStatus === 'testing' || !formState.glurooUrl || !formState.glurooApiSecret}
                      >
                        {glurooStatus === 'testing' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                        {glurooStatus === 'success' && <Check className="w-4 h-4 mr-2 text-green-500" />}
                        {glurooStatus === 'error' && <AlertCircle className="w-4 h-4 mr-2 text-red-500" />}
                        <Wifi className="w-4 h-4 mr-2" />
                        Test Connection
                      </Button>
                      <Button
                        variant="outline"
                        onClick={handleSyncGluroo}
                        disabled={glurooStatus === 'syncing'}
                      >
                        {glurooStatus === 'syncing' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Sync Now
                      </Button>
                    </div>

                    {glurooStatus === 'success' && (
                      <Badge className="bg-green-500/20 text-green-600 border-green-500/30">
                        Connected successfully
                      </Badge>
                    )}
                    {glurooStatus === 'error' && (
                      <Badge className="bg-red-500/20 text-red-600 border-red-500/30">
                        Connection failed - check credentials
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Instructions Card */}
              <Card className="bg-blue-500/10 border-blue-500/30">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-blue-600 dark:text-blue-400">
                    <HelpCircle className="w-5 h-5" />
                    Where to Find Your Gluroo Credentials
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <ol className="list-decimal list-inside space-y-2">
                    <li>Open the <strong>Gluroo app</strong> on your phone</li>
                    <li>Go to <strong>Menu → Settings → Gluroo Global Connect Nightscout</strong></li>
                    <li>Copy your <strong>Nightscout URL</strong> (looks like https://xxxx.ns.gluroo.com)</li>
                    <li>Copy your <strong>API Secret</strong></li>
                    <li>Paste both values above and click "Test Connection"</li>
                  </ol>
                  <p className="text-muted-foreground mt-4">
                    Need help? Visit{' '}
                    <a
                      href="https://gluroo.com/support"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline hover:no-underline"
                    >
                      Gluroo Support
                    </a>
                  </p>
                </CardContent>
              </Card>

              {/* Tandem Source Connection */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <Activity className="w-5 h-5 text-purple-500" />
                        Tandem Pump Connection
                      </CardTitle>
                      <CardDescription>
                        Connect your Tandem t:connect account to sync pump data (basal, bolus, carbs)
                      </CardDescription>
                    </div>
                    {tandemConnected && (
                      <Badge className="bg-green-500/20 text-green-600 border-green-500/30">
                        Connected
                      </Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div>
                      <Label>Tandem Email</Label>
                      <Input
                        type="email"
                        value={formState.tandemEmail}
                        onChange={(e) => updateField('tandemEmail', e.target.value)}
                        placeholder="your@email.com"
                        className="mt-1"
                        disabled={tandemConnected}
                      />
                    </div>
                    <div>
                      <Label>Tandem Password</Label>
                      <Input
                        type="password"
                        value={formState.tandemPassword}
                        onChange={(e) => updateField('tandemPassword', e.target.value)}
                        placeholder="Enter your Tandem password"
                        className="mt-1"
                        disabled={tandemConnected}
                      />
                    </div>

                    <div className="flex gap-3 pt-2">
                      {!tandemConnected ? (
                        <>
                          <Button
                            variant="outline"
                            onClick={handleTestTandem}
                            disabled={tandemStatus === 'testing' || !formState.tandemEmail || !formState.tandemPassword}
                          >
                            {tandemStatus === 'testing' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                            {tandemStatus === 'success' && <Check className="w-4 h-4 mr-2 text-green-500" />}
                            {tandemStatus === 'error' && <AlertCircle className="w-4 h-4 mr-2 text-red-500" />}
                            <Wifi className="w-4 h-4 mr-2" />
                            Test Connection
                          </Button>
                          <Button
                            onClick={handleConnectTandem}
                            disabled={tandemStatus === 'testing' || !formState.tandemEmail || !formState.tandemPassword}
                          >
                            {tandemStatus === 'testing' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                            Connect
                          </Button>
                        </>
                      ) : (
                        <>
                          <Button
                            variant="outline"
                            onClick={() => handleSyncTandem(false)}
                            disabled={tandemStatus === 'syncing'}
                          >
                            {tandemStatus === 'syncing' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                            <RefreshCw className="w-4 h-4 mr-2" />
                            Sync Now
                          </Button>
                          <Button
                            variant="outline"
                            onClick={() => handleSyncTandem(true)}
                            disabled={tandemStatus === 'syncing'}
                          >
                            {tandemStatus === 'syncing' && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                            Full Sync (7 days)
                          </Button>
                          <Button
                            variant="destructive"
                            onClick={handleDisconnectTandem}
                          >
                            <Trash2 className="w-4 h-4 mr-2" />
                            Disconnect
                          </Button>
                        </>
                      )}
                    </div>

                    {tandemStatus === 'success' && (
                      <Badge className="bg-green-500/20 text-green-600 border-green-500/30">
                        {tandemConnected ? 'Synced successfully' : 'Connected successfully'}
                      </Badge>
                    )}
                    {tandemStatus === 'error' && (
                      <Badge className="bg-red-500/20 text-red-600 border-red-500/30">
                        {tandemConnected ? 'Sync failed' : 'Connection failed - check credentials'}
                      </Badge>
                    )}
                    {tandemStatusData?.lastSyncAt && (
                      <p className="text-xs text-muted-foreground">
                        Last synced: {new Date(tandemStatusData.lastSyncAt).toLocaleString()}
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Insulin Tab */}
          <TabsContent value="insulin">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Insulin Settings</CardTitle>
                  <CardDescription>
                    These values are used for dose calculations. The AI learns your patterns over time.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <Label>Target BG (mg/dL)</Label>
                      <Input
                        type="number"
                        value={formState.targetBg}
                        onChange={(e) => updateField('targetBg', parseInt(e.target.value))}
                        min={70}
                        max={150}
                        className="mt-1"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Goal glucose level for corrections</p>
                    </div>
                    <div>
                      <Label>Default ISF (mg/dL per unit)</Label>
                      <Input
                        type="number"
                        value={formState.insulinSensitivity}
                        onChange={(e) => updateField('insulinSensitivity', parseInt(e.target.value))}
                        min={10}
                        max={200}
                        className="mt-1"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Fallback when ML prediction unavailable</p>
                    </div>
                    <div>
                      <Label>Carb Ratio (g per unit)</Label>
                      <Input
                        type="number"
                        value={formState.carbRatio}
                        onChange={(e) => updateField('carbRatio', parseInt(e.target.value))}
                        min={1}
                        max={50}
                        className="mt-1"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Grams of carbs covered by 1 unit</p>
                    </div>
                    <div>
                      <Label>Insulin Duration (min)</Label>
                      <Input
                        type="number"
                        value={formState.insulinDuration}
                        onChange={(e) => updateField('insulinDuration', parseInt(e.target.value))}
                        min={120}
                        max={360}
                        step={30}
                        className="mt-1"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Active insulin time (default: 180 min)</p>
                    </div>
                  </div>

                  <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/30">
                    <p className="text-sm">
                      <strong>Note:</strong> The ML model dynamically predicts your ISF based on time of day,
                      recent activity, and glucose patterns. The default ISF above is used as a fallback
                      when the model is unavailable.
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* AI-Learned Ratios Card */}
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-blue-500" />
                    <CardTitle>AI-Learned Ratios</CardTitle>
                  </div>
                  <CardDescription>
                    Enable AI to learn your personal carb and protein sensitivity from your data.
                    Visit the ML Models page to trigger learning.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* ICR Settings */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Use AI-Learned ICR</p>
                        <p className="text-sm text-muted-foreground">Use AI-learned carb-to-insulin ratio instead of manual</p>
                      </div>
                      <Switch
                        checked={formState.useLearnedICR}
                        onCheckedChange={(checked) => updateField('useLearnedICR', checked)}
                      />
                    </div>
                  </div>

                  <Separator />

                  {/* PIR Settings */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Include Protein in Dose</p>
                        <p className="text-sm text-muted-foreground">Calculate insulin for protein (shows immediate + delayed breakdown)</p>
                      </div>
                      <Switch
                        checked={formState.includeProteinInBolus}
                        onCheckedChange={(checked) => updateField('includeProteinInBolus', checked)}
                      />
                    </div>

                    {formState.includeProteinInBolus && (
                      <>
                        <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                          <div>
                            <p className="font-medium">Use AI-Learned PIR</p>
                            <p className="text-sm text-muted-foreground">Use AI-learned protein-to-insulin ratio + timing</p>
                          </div>
                          <Switch
                            checked={formState.useLearnedPIR}
                            onCheckedChange={(checked) => updateField('useLearnedPIR', checked)}
                          />
                        </div>

                        {!formState.useLearnedPIR && (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pl-4 border-l-2 border-primary/30">
                            <div>
                              <Label>Protein Ratio (g per unit)</Label>
                              <Input
                                type="number"
                                value={formState.proteinRatio}
                                onChange={(e) => updateField('proteinRatio', parseInt(e.target.value))}
                                min={10}
                                max={100}
                                className="mt-1"
                              />
                              <p className="text-xs text-muted-foreground mt-1">Grams of protein covered by 1 unit</p>
                            </div>
                            <div>
                              <Label>Upfront Percent (%)</Label>
                              <Input
                                type="number"
                                value={formState.proteinUpfrontPercent}
                                onChange={(e) => updateField('proteinUpfrontPercent', parseInt(e.target.value))}
                                min={0}
                                max={100}
                                className="mt-1"
                              />
                              <p className="text-xs text-muted-foreground mt-1">% of protein insulin to give at meal (rest delayed)</p>
                            </div>
                          </div>
                        )}

                        <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
                          <p className="text-sm">
                            <strong>Protein Dosing:</strong> Protein affects BG 2-4 hours after eating.
                            The dose breakdown shows how much to give NOW (carbs + upfront protein)
                            and how much to give LATER (delayed protein via extended bolus or manual dose).
                          </p>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="flex gap-3">
                    <Button variant="outline" asChild>
                      <Link to="/ml-models">
                        <Activity className="w-4 h-4 mr-2" />
                        Go to ML Models
                      </Link>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Alerts Tab */}
          <TabsContent value="alerts">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Alert Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Enable Alerts</p>
                        <p className="text-sm text-muted-foreground">Receive notifications for glucose events</p>
                      </div>
                      <Switch
                        checked={formState.enableAlerts}
                        onCheckedChange={(checked) => updateField('enableAlerts', checked)}
                      />
                    </div>
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Sound Effects</p>
                        <p className="text-sm text-muted-foreground">Play sound for critical alerts</p>
                      </div>
                      <Switch
                        checked={formState.soundEnabled}
                        onCheckedChange={(checked) => updateField('soundEnabled', checked)}
                      />
                    </div>
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Predictive Alerts</p>
                        <p className="text-sm text-muted-foreground">Alert based on predicted low/high values</p>
                      </div>
                      <Switch
                        checked={formState.predictiveAlerts}
                        onCheckedChange={(checked) => updateField('predictiveAlerts', checked)}
                      />
                    </div>
                  </div>

                  <Separator />

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <Label>High Threshold (mg/dL)</Label>
                      <Input
                        type="number"
                        value={formState.highThreshold}
                        onChange={(e) => updateField('highThreshold', parseInt(e.target.value))}
                        min={140}
                        max={300}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Low Threshold (mg/dL)</Label>
                      <Input
                        type="number"
                        value={formState.lowThreshold}
                        onChange={(e) => updateField('lowThreshold', parseInt(e.target.value))}
                        min={60}
                        max={100}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Critical High (mg/dL)</Label>
                      <Input
                        type="number"
                        value={formState.criticalHighThreshold}
                        onChange={(e) => updateField('criticalHighThreshold', parseInt(e.target.value))}
                        min={200}
                        max={400}
                        className="mt-1"
                      />
                      <p className="text-xs text-destructive mt-1">Urgent alert threshold</p>
                    </div>
                    <div>
                      <Label>Critical Low (mg/dL)</Label>
                      <Input
                        type="number"
                        value={formState.criticalLowThreshold}
                        onChange={(e) => updateField('criticalLowThreshold', parseInt(e.target.value))}
                        min={40}
                        max={70}
                        className="mt-1"
                      />
                      <p className="text-xs text-destructive mt-1">Urgent alert threshold</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Sharing Tab */}
          <TabsContent value="sharing">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <SharingManager />
            </motion.div>
          </TabsContent>

          {/* Security Tab */}
          <TabsContent value="integrations">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <HomeAssistantKeys />
            </motion.div>
          </TabsContent>

          <TabsContent value="security">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="w-5 h-5" />
                    Privacy & Security
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Two-Factor Authentication</p>
                        <p className="text-sm text-muted-foreground">Add an extra layer of security</p>
                      </div>
                      <Badge variant="secondary">Managed by Microsoft</Badge>
                    </div>
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Data Encryption</p>
                        <p className="text-sm text-muted-foreground">All data is encrypted at rest and in transit</p>
                      </div>
                      <Badge variant="default" className="bg-green-500">Enabled</Badge>
                    </div>
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Anonymous Analytics</p>
                        <p className="text-sm text-muted-foreground">Help improve T1D-AI with anonymous usage data</p>
                      </div>
                      <Switch defaultChecked={false} />
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <h3 className="font-medium mb-4">Active Sessions</h3>
                    <div className="p-4 rounded-lg bg-muted/50">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                            <Activity className="w-5 h-5 text-primary" />
                          </div>
                          <div>
                            <p className="font-medium">Current Session</p>
                            <p className="text-xs text-muted-foreground">This device</p>
                          </div>
                        </div>
                        <Badge>Active</Badge>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-destructive">Delete Account</CardTitle>
                  <CardDescription>
                    Permanently delete your account and all associated data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button variant="destructive">
                        <Trash2 className="w-4 h-4 mr-2" />
                        Delete My Account
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This action cannot be undone. This will permanently delete your account
                          and remove all your data from our servers.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction className="bg-destructive">Delete Account</AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Display Tab */}
          <TabsContent value="display">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Display Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                    <div>
                      <p className="font-medium">Dark Mode</p>
                      <p className="text-sm text-muted-foreground">Use dark theme (recommended)</p>
                    </div>
                    <Switch
                      checked={formState.darkMode}
                      onCheckedChange={(checked) => {
                        updateField('darkMode', checked)
                        if (checked) {
                          document.documentElement.classList.add('dark')
                          localStorage.setItem('theme', 'dark')
                        } else {
                          document.documentElement.classList.remove('dark')
                          localStorage.setItem('theme', 'light')
                        }
                      }}
                    />
                  </div>
                  <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                    <div>
                      <p className="font-medium">Show Predictions</p>
                      <p className="text-sm text-muted-foreground">Display ML prediction values on dashboard</p>
                    </div>
                    <Switch
                      checked={formState.showPredictions}
                      onCheckedChange={(checked) => updateField('showPredictions', checked)}
                    />
                  </div>
                  <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                    <div>
                      <p className="font-medium">Show AI Insights</p>
                      <p className="text-sm text-muted-foreground">Display GPT-powered pattern insights</p>
                    </div>
                    <Switch
                      checked={formState.showInsights}
                      onCheckedChange={(checked) => updateField('showInsights', checked)}
                    />
                  </div>

                  <Separator />

                  <div className="space-y-4">
                    <h4 className="font-medium text-lg">Prediction Settings</h4>
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Use TFT Neural Network Modifiers</p>
                        <p className="text-sm text-muted-foreground">Apply TFT neural network adjustments to physics-based predictions. Turn off for pure physics predictions only.</p>
                      </div>
                      <Switch
                        checked={formState.useTFTModifiers}
                        onCheckedChange={(checked) => updateField('useTFTModifiers', checked)}
                      />
                    </div>
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium">Track Prediction Accuracy</p>
                        <p className="text-sm text-muted-foreground">Log predictions and compare with actual glucose to improve model over time</p>
                      </div>
                      <Switch
                        checked={formState.trackPredictionAccuracy}
                        onCheckedChange={(checked) => updateField('trackPredictionAccuracy', checked)}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>
        </Tabs>

        {/* Bottom padding for mobile */}
        <div className="h-32 md:h-8" />
      </div>
    </ResponsiveLayout>
  )
}
