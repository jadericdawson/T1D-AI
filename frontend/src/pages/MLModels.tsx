/**
 * ML Models Page
 * Comprehensive view of all machine learning models and their status
 */
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  Brain, TrendingUp, Zap, BarChart3, ArrowLeft, Loader2,
  Check, RefreshCw, Clock, Activity, Target,
  Sunrise, Sun, Sunset, Moon as MoonIcon, Utensils, Beef
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { ResponsiveLayout } from '@/components/layout/ResponsiveLayout'
import { formatDateTime } from '@/lib/utils'
import { trainingApi } from '@/lib/api'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useAuthStore } from '@/stores/authStore'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } }
}

export default function MLModels() {
  const queryClient = useQueryClient()
  const user = useAuthStore(state => state.user)
  const userId = user?.id || ''

  // Fetch ML model status
  const { data: trainingStatus, isLoading: isLoadingTraining } = useQuery({
    queryKey: ['training', 'status', userId],
    queryFn: () => trainingApi.getStatus(),
    enabled: !!userId,
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Fetch ISF status
  const { data: isfStatus, isLoading: isLoadingISF } = useQuery({
    queryKey: ['training', 'isf', userId],
    queryFn: () => trainingApi.getISFStatus(),
    enabled: !!userId,
  })

  // Fetch training eligibility (used for future eligibility checks)
  const { data: _eligibility } = useQuery({
    queryKey: ['training', 'eligibility', userId],
    queryFn: () => trainingApi.checkEligibility(),
    enabled: !!userId,
  })

  // Fetch data stats
  const { data: dataStats } = useQuery({
    queryKey: ['training', 'data-stats', userId],
    queryFn: () => trainingApi.getDataStats(),
    enabled: !!userId,
  })

  // ISF Learning mutation
  const learnISFMutation = useMutation({
    mutationFn: () => trainingApi.learnISFEnhanced(30, true),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['training', 'isf'] })
      if (data.success) {
        alert(`ISF Learning Complete!\n${data.message}`)
      } else {
        alert(`ISF Learning Issue:\n${data.message}`)
      }
    },
    onError: (error: Error) => {
      console.error('ISF Learning failed:', error)
      alert(`ISF Learning Failed:\n${error.message || 'Unknown error occurred'}`)
    },
  })

  // Reset ISF mutation
  const resetISFMutation = useMutation({
    mutationFn: () => trainingApi.resetISF(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training', 'isf'] })
      alert('ISF has been reset successfully')
    },
    onError: (error: Error) => {
      console.error('ISF Reset failed:', error)
      alert(`ISF Reset Failed:\n${error.message || 'Unknown error occurred'}`)
    },
  })

  // Fetch ICR status
  const { data: icrStatus, isLoading: isLoadingICR } = useQuery({
    queryKey: ['training', 'icr', userId],
    queryFn: () => trainingApi.getICRStatus(),
    enabled: !!userId,
  })

  // ICR Learning mutation
  const learnICRMutation = useMutation({
    mutationFn: () => trainingApi.learnICR(30),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['training', 'icr'] })
      if (data.success) {
        alert(`ICR Learning Complete!\n${data.message}`)
      } else {
        alert(`ICR Learning Issue:\n${data.message}`)
      }
    },
    onError: (error: Error) => {
      console.error('ICR Learning failed:', error)
      alert(`ICR Learning Failed:\n${error.message || 'Unknown error occurred'}`)
    },
  })

  // Reset ICR mutation
  const resetICRMutation = useMutation({
    mutationFn: () => trainingApi.resetICR(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training', 'icr'] })
      alert('ICR has been reset successfully')
    },
    onError: (error: Error) => {
      console.error('ICR Reset failed:', error)
      alert(`ICR Reset Failed:\n${error.message || 'Unknown error occurred'}`)
    },
  })

  // Fetch PIR status
  const { data: pirStatus, isLoading: isLoadingPIR } = useQuery({
    queryKey: ['training', 'pir', userId],
    queryFn: () => trainingApi.getPIRStatus(),
    enabled: !!userId,
  })

  // PIR Learning mutation
  const learnPIRMutation = useMutation({
    mutationFn: () => trainingApi.learnPIR(30),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['training', 'pir'] })
      if (data.success) {
        alert(`PIR Learning Complete!\n${data.message}`)
      } else {
        alert(`PIR Learning Issue:\n${data.message}`)
      }
    },
    onError: (error: Error) => {
      console.error('PIR Learning failed:', error)
      alert(`PIR Learning Failed:\n${error.message || 'Unknown error occurred'}`)
    },
  })

  // Reset PIR mutation
  const resetPIRMutation = useMutation({
    mutationFn: () => trainingApi.resetPIR(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training', 'pir'] })
      alert('PIR has been reset successfully')
    },
    onError: (error: Error) => {
      console.error('PIR Reset failed:', error)
      alert(`PIR Reset Failed:\n${error.message || 'Unknown error occurred'}`)
    },
  })

  // Start training mutation (used by Start Training button)
  // Temporarily disabled - ACI training needs proper user ID setup
  // const startTrainingMutation = useMutation({
  //   mutationFn: () => trainingApi.startTraining('tft'),
  //   onSuccess: () => {
  //     queryClient.invalidateQueries({ queryKey: ['training', 'status'] })
  //     alert('TFT Model training started! This may take several minutes.')
  //   },
  //   onError: (error: Error) => {
  //     console.error('Training start failed:', error)
  //     alert(`Training Start Failed:\n${error.message || 'Unknown error occurred'}`)
  //   },
  // })

  // Get time of day icon
  const getTimeOfDayIcon = (time: string) => {
    switch (time) {
      case 'morning': return <Sunrise className="w-4 h-4 text-orange-400" />
      case 'afternoon': return <Sun className="w-4 h-4 text-yellow-400" />
      case 'evening': return <Sunset className="w-4 h-4 text-orange-500" />
      case 'night': return <MoonIcon className="w-4 h-4 text-blue-400" />
      default: return null
    }
  }

  return (
    <ResponsiveLayout title="ML Models">
      <div className="min-h-screen p-4 md:p-6 max-w-6xl mx-auto">
        {/* Header */}
        <motion.header
          initial="hidden"
          animate="visible"
          variants={fadeIn}
          className="flex items-center gap-4 mb-6"
        >
          <Link to="/dashboard">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="w-5 h-5" />
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <Brain className="w-6 h-6 text-primary" />
            <h1 className="text-2xl font-bold">Machine Learning Models</h1>
          </div>
        </motion.header>

        <div className="grid gap-6">
          {/* ISF Learning Section */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-orange-500" />
                      Insulin Sensitivity Factor (ISF) Learning
                    </CardTitle>
                    <CardDescription>
                      Learns your ISF from clean correction boluses - how much 1 unit of insulin lowers your blood sugar
                    </CardDescription>
                  </div>
                  <Badge variant={isfStatus?.has_learned_isf ? 'default' : 'secondary'}>
                    {isfStatus?.has_learned_isf ? 'Active' : 'Not Learned'}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {isLoadingISF ? (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading ISF status...
                  </div>
                ) : isfStatus?.has_learned_isf ? (
                  <>
                    {/* Main ISF Values */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 rounded-lg bg-orange-500/10 border border-orange-500/30">
                        <p className="text-xs text-muted-foreground">Fasting ISF</p>
                        <p className="text-3xl font-bold text-orange-500">
                          {isfStatus.fasting_isf?.toFixed(0) || '-'}
                        </p>
                        <p className="text-xs text-muted-foreground">mg/dL per unit</p>
                      </div>
                      <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
                        <p className="text-xs text-muted-foreground">Meal ISF</p>
                        <p className="text-3xl font-bold text-blue-500">
                          {isfStatus.meal_isf?.toFixed(0) || '-'}
                        </p>
                        <p className="text-xs text-muted-foreground">mg/dL per unit</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <p className="text-xs text-muted-foreground">Confidence</p>
                        <p className="text-3xl font-bold">
                          {((isfStatus.fasting_confidence || 0) * 100).toFixed(0)}%
                        </p>
                        <Progress
                          value={(isfStatus.fasting_confidence || 0) * 100}
                          className="mt-2 h-2"
                        />
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <p className="text-xs text-muted-foreground">Clean Boluses</p>
                        <p className="text-3xl font-bold">{isfStatus.fasting_samples || 0}</p>
                        <p className="text-xs text-muted-foreground">samples analyzed</p>
                      </div>
                    </div>

                    {/* Time of Day Pattern */}
                    {isfStatus.time_of_day_pattern && (
                      <div>
                        <h4 className="text-sm font-medium mb-3">Time of Day Patterns</h4>
                        <div className="grid grid-cols-4 gap-3">
                          {Object.entries(isfStatus.time_of_day_pattern).map(([time, value]) => (
                            <div key={time} className="p-3 rounded-lg bg-muted/50 text-center">
                              <div className="flex justify-center mb-1">
                                {getTimeOfDayIcon(time)}
                              </div>
                              <p className="text-xs text-muted-foreground capitalize">{time}</p>
                              <p className="text-lg font-semibold">
                                {value ? `${value.toFixed(0)}` : '-'}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Last Updated */}
                    {isfStatus.last_updated && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Clock className="w-4 h-4" />
                        Last updated: {formatDateTime(new Date(isfStatus.last_updated))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="p-6 rounded-lg border border-dashed text-center">
                    <TrendingUp className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-muted-foreground mb-2">
                      ISF learning analyzes your clean correction boluses (insulin without carbs)
                      to determine how much 1 unit of insulin lowers your blood sugar.
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Click "Learn ISF" to start the learning process.
                    </p>
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-3 pt-2">
                  <Button
                    onClick={() => learnISFMutation.mutate()}
                    disabled={learnISFMutation.isPending}
                  >
                    {learnISFMutation.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    <Brain className="w-4 h-4 mr-2" />
                    {isfStatus?.has_learned_isf ? 'Re-Learn ISF' : 'Learn ISF'}
                  </Button>
                  {isfStatus?.has_learned_isf && (
                    <Button
                      variant="outline"
                      onClick={() => resetISFMutation.mutate()}
                      disabled={resetISFMutation.isPending}
                    >
                      {resetISFMutation.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Reset ISF
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* ICR Learning Section */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn} transition={{ delay: 0.05 }}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Utensils className="w-5 h-5 text-teal-500" />
                      Carb-to-Insulin Ratio (ICR) Learning
                    </CardTitle>
                    <CardDescription>
                      Learns how many grams of carbs are covered by 1 unit of insulin
                    </CardDescription>
                  </div>
                  <Badge variant={icrStatus?.has_learned_icr ? 'default' : 'secondary'}>
                    {icrStatus?.has_learned_icr ? 'Active' : 'Not Learned'}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {isLoadingICR ? (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading ICR status...
                  </div>
                ) : icrStatus?.has_learned_icr ? (
                  <>
                    {/* Main ICR Values */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 rounded-lg bg-teal-500/10 border border-teal-500/30">
                        <p className="text-xs text-muted-foreground">Overall ICR</p>
                        <p className="text-3xl font-bold text-teal-500">
                          {icrStatus.overall_icr?.toFixed(1) || '-'}
                        </p>
                        <p className="text-xs text-muted-foreground">grams per unit</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <p className="text-xs text-muted-foreground">Confidence</p>
                        <p className="text-3xl font-bold">
                          {((icrStatus.overall_confidence || 0) * 100).toFixed(0)}%
                        </p>
                        <Progress
                          value={(icrStatus.overall_confidence || 0) * 100}
                          className="mt-2 h-2"
                        />
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <p className="text-xs text-muted-foreground">Samples</p>
                        <p className="text-3xl font-bold">{icrStatus.overall_samples || 0}</p>
                        <p className="text-xs text-muted-foreground">meal boluses analyzed</p>
                      </div>
                      {icrStatus.icr_range && (
                        <div className="p-4 rounded-lg bg-muted/50">
                          <p className="text-xs text-muted-foreground">Range</p>
                          <p className="text-xl font-bold">
                            {icrStatus.icr_range.min?.toFixed(0) ?? '-'} - {icrStatus.icr_range.max?.toFixed(0) ?? '-'}
                          </p>
                          <p className="text-xs text-muted-foreground">g/U</p>
                        </div>
                      )}
                    </div>

                    {/* Meal Type Pattern */}
                    {icrStatus.meal_type_pattern && Object.keys(icrStatus.meal_type_pattern).length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium mb-3">Meal Type Patterns</h4>
                        <div className="grid grid-cols-3 gap-3">
                          {['breakfast', 'lunch', 'dinner'].map((meal) => {
                            const value = icrStatus.meal_type_pattern?.[meal as keyof typeof icrStatus.meal_type_pattern]
                            return (
                              <div key={meal} className="p-3 rounded-lg bg-muted/50 text-center">
                                <p className="text-xs text-muted-foreground capitalize">{meal}</p>
                                <p className="text-lg font-semibold">
                                  {value ? `${value.toFixed(1)} g/U` : '-'}
                                </p>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}

                    {/* Last Updated */}
                    {icrStatus.last_updated && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Clock className="w-4 h-4" />
                        Last updated: {formatDateTime(new Date(icrStatus.last_updated))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="p-6 rounded-lg border border-dashed text-center">
                    <Utensils className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-muted-foreground mb-2">
                      ICR learning analyzes your meal boluses to determine how many grams of carbs
                      are covered by 1 unit of insulin.
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Click "Learn ICR" to start the learning process.
                    </p>
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-3 pt-2">
                  <Button
                    onClick={() => learnICRMutation.mutate()}
                    disabled={learnICRMutation.isPending}
                  >
                    {learnICRMutation.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    <Brain className="w-4 h-4 mr-2" />
                    {icrStatus?.has_learned_icr ? 'Re-Learn ICR' : 'Learn ICR'}
                  </Button>
                  {icrStatus?.has_learned_icr && (
                    <Button
                      variant="outline"
                      onClick={() => resetICRMutation.mutate()}
                      disabled={resetICRMutation.isPending}
                    >
                      {resetICRMutation.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Reset ICR
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* PIR Learning Section */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn} transition={{ delay: 0.075 }}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Beef className="w-5 h-5 text-rose-500" />
                      Protein-to-Insulin Ratio (PIR) Learning
                    </CardTitle>
                    <CardDescription>
                      Learns how protein affects your blood sugar and timing
                    </CardDescription>
                  </div>
                  <Badge variant={pirStatus?.has_learned_pir ? 'default' : 'secondary'}>
                    {pirStatus?.has_learned_pir ? 'Active' : 'Not Learned'}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {isLoadingPIR ? (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading PIR status...
                  </div>
                ) : pirStatus?.has_learned_pir ? (
                  <>
                    {/* Main PIR Values */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 rounded-lg bg-rose-500/10 border border-rose-500/30">
                        <p className="text-xs text-muted-foreground">Overall PIR</p>
                        <p className="text-3xl font-bold text-rose-500">
                          {pirStatus.overall_pir?.toFixed(1) || '-'}
                        </p>
                        <p className="text-xs text-muted-foreground">grams per unit</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <p className="text-xs text-muted-foreground">Confidence</p>
                        <p className="text-3xl font-bold">
                          {((pirStatus.overall_confidence || 0) * 100).toFixed(0)}%
                        </p>
                        <Progress
                          value={(pirStatus.overall_confidence || 0) * 100}
                          className="mt-2 h-2"
                        />
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <p className="text-xs text-muted-foreground">Samples</p>
                        <p className="text-3xl font-bold">{pirStatus.overall_samples || 0}</p>
                        <p className="text-xs text-muted-foreground">protein events analyzed</p>
                      </div>
                      {pirStatus.pir_range && (
                        <div className="p-4 rounded-lg bg-muted/50">
                          <p className="text-xs text-muted-foreground">Range</p>
                          <p className="text-xl font-bold">
                            {pirStatus.pir_range.min?.toFixed(0) ?? '-'} - {pirStatus.pir_range.max?.toFixed(0) ?? '-'}
                          </p>
                          <p className="text-xs text-muted-foreground">g/U</p>
                        </div>
                      )}
                    </div>

                    {/* Protein Timing */}
                    {(pirStatus.protein_onset_min || pirStatus.protein_peak_min) && (
                      <div>
                        <h4 className="text-sm font-medium mb-3">Protein Effect Timing</h4>
                        <div className="grid grid-cols-3 gap-3">
                          <div className="p-3 rounded-lg bg-rose-500/10 text-center">
                            <p className="text-xs text-muted-foreground">Onset</p>
                            <p className="text-lg font-semibold text-rose-400">
                              {pirStatus.protein_onset_min ? `${Math.round(pirStatus.protein_onset_min)} min` : '-'}
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-rose-500/10 text-center">
                            <p className="text-xs text-muted-foreground">Peak</p>
                            <p className="text-lg font-semibold text-rose-400">
                              {pirStatus.protein_peak_min ? `${Math.round(pirStatus.protein_peak_min)} min` : '-'}
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-rose-500/10 text-center">
                            <p className="text-xs text-muted-foreground">Duration</p>
                            <p className="text-lg font-semibold text-rose-400">
                              {pirStatus.protein_duration_min ? `${Math.round(pirStatus.protein_duration_min)} min` : '-'}
                            </p>
                          </div>
                        </div>
                        <p className="text-xs text-muted-foreground mt-2">
                          💡 Consider using an extended bolus covering {pirStatus.protein_onset_min ? Math.round(pirStatus.protein_onset_min) : 120}-{pirStatus.protein_peak_min ? Math.round(pirStatus.protein_peak_min) : 180} minutes for protein-heavy meals
                        </p>
                      </div>
                    )}

                    {/* Last Updated */}
                    {pirStatus.last_updated && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Clock className="w-4 h-4" />
                        Last updated: {formatDateTime(new Date(pirStatus.last_updated))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="p-6 rounded-lg border border-dashed text-center">
                    <Beef className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-muted-foreground mb-2">
                      PIR learning analyzes high-protein meals to determine how protein affects
                      your blood sugar and when the effect occurs (typically 2-4 hours after eating).
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Click "Learn PIR" to start the learning process.
                    </p>
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-3 pt-2">
                  <Button
                    onClick={() => learnPIRMutation.mutate()}
                    disabled={learnPIRMutation.isPending}
                  >
                    {learnPIRMutation.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    <Brain className="w-4 h-4 mr-2" />
                    {pirStatus?.has_learned_pir ? 'Re-Learn PIR' : 'Learn PIR'}
                  </Button>
                  {pirStatus?.has_learned_pir && (
                    <Button
                      variant="outline"
                      onClick={() => resetPIRMutation.mutate()}
                      disabled={resetPIRMutation.isPending}
                    >
                      {resetPIRMutation.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Reset PIR
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* TFT Prediction Model Section */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn} transition={{ delay: 0.1 }}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Zap className="w-5 h-5 text-purple-500" />
                      TFT Glucose Prediction Model
                    </CardTitle>
                    <CardDescription>
                      Temporal Fusion Transformer for personalized glucose predictions
                    </CardDescription>
                  </div>
                  <Badge variant={trainingStatus?.hasPersonalizedModel ? 'default' : 'secondary'}>
                    {trainingStatus?.modelStatus || 'Not Trained'}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {isLoadingTraining ? (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading model status...
                  </div>
                ) : trainingStatus?.hasPersonalizedModel ? (
                  <>
                    {/* Model Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/30">
                        <p className="text-xs text-muted-foreground">Model Version</p>
                        <p className="text-3xl font-bold text-purple-500">
                          v{trainingStatus.modelVersion}
                        </p>
                      </div>
                      <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/30">
                        <p className="text-xs text-muted-foreground">MAE @ 30min</p>
                        <p className="text-3xl font-bold text-green-500">
                          {trainingStatus.modelMetrics?.mae_30min?.toFixed(1) || '-'}
                        </p>
                        <p className="text-xs text-muted-foreground">mg/dL</p>
                      </div>
                      <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                        <p className="text-xs text-muted-foreground">MAE @ 60min</p>
                        <p className="text-3xl font-bold text-yellow-500">
                          {trainingStatus.modelMetrics?.mae_60min?.toFixed(1) || '-'}
                        </p>
                        <p className="text-xs text-muted-foreground">mg/dL</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <p className="text-xs text-muted-foreground">Last Trained</p>
                        <p className="text-sm font-medium">
                          {trainingStatus.lastTrainedAt
                            ? formatDateTime(new Date(trainingStatus.lastTrainedAt))
                            : '-'}
                        </p>
                      </div>
                    </div>

                    {/* RMSE Metrics */}
                    {trainingStatus.modelMetrics?.rmse_30min && (
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-3 rounded-lg bg-muted/50">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">RMSE @ 30min</span>
                            <span className="font-medium">
                              {trainingStatus.modelMetrics.rmse_30min?.toFixed(1)} mg/dL
                            </span>
                          </div>
                        </div>
                        <div className="p-3 rounded-lg bg-muted/50">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">RMSE @ 60min</span>
                            <span className="font-medium">
                              {trainingStatus.modelMetrics.rmse_60min?.toFixed(1)} mg/dL
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="space-y-4">
                    {/* Physics-based predictions active */}
                    <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/30">
                      <div className="flex items-start gap-3">
                        <Zap className="w-6 h-6 text-purple-400 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="font-medium text-purple-400 mb-1">Physics-Based Predictions Active</p>
                          <p className="text-sm text-muted-foreground">
                            Your predictions use IOB/COB forcing functions with learned ISF, ICR, and PIR.
                            This provides accurate predictions without requiring a personalized TFT model.
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Data Stats */}
                    <div className="p-4 rounded-lg bg-muted/50">
                      <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
                        <Target className="w-4 h-4" />
                        Your Training Data
                      </h4>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>Glucose Readings</span>
                            <span className="text-green-500">
                              {dataStats?.totalReadings?.toLocaleString() || 0}
                              <Check className="w-4 h-4 inline ml-1" />
                            </span>
                          </div>
                          <Progress value={100} className="bg-green-500/20" />
                        </div>
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>Treatments</span>
                            <span className="text-green-500">
                              {dataStats?.totalTreatments?.toLocaleString() || 0}
                              <Check className="w-4 h-4 inline ml-1" />
                            </span>
                          </div>
                          <Progress value={100} className="bg-green-500/20" />
                        </div>
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>Days of Data</span>
                            <span className="text-green-500">
                              {dataStats?.dataSpanDays || 0} days
                              <Check className="w-4 h-4 inline ml-1" />
                            </span>
                          </div>
                          <Progress value={100} className="bg-green-500/20" />
                        </div>
                      </div>
                    </div>

                    {/* Local Training Info */}
                    <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                      <p className="text-sm text-yellow-400">
                        <strong>Note:</strong> Personalized TFT training requires local execution due to compute requirements.
                        Run <code className="bg-black/30 px-1 rounded">python train_local_model.py</code> locally to train.
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* COB Absorption Learning Section */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn} transition={{ delay: 0.2 }}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-green-500" />
                      COB Absorption Learning
                    </CardTitle>
                    <CardDescription>
                      Per-food absorption patterns learned from your BG responses
                    </CardDescription>
                  </div>
                  <Badge variant="outline">Auto-Learning</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="p-6 rounded-lg bg-green-500/10 border border-green-500/30">
                  <div className="flex items-start gap-4">
                    <Activity className="w-8 h-8 text-green-500 flex-shrink-0" />
                    <div>
                      <p className="font-medium mb-2">Automatic Learning Enabled</p>
                      <p className="text-sm text-muted-foreground">
                        The system automatically learns how different foods affect your blood sugar.
                        As you log meals, it builds per-food absorption profiles to improve predictions.
                      </p>
                      <div className="flex flex-wrap gap-2 mt-4">
                        <Badge variant="outline">Physics Formula Baseline</Badge>
                        <Badge variant="outline">Macro Modifiers (Fat/Protein/Fiber)</Badge>
                        <Badge variant="outline">Per-Food Learning</Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Recent Training Jobs */}
          {trainingStatus?.recentJobs && trainingStatus.recentJobs.length > 0 && (
            <motion.div initial="hidden" animate="visible" variants={fadeIn} transition={{ delay: 0.3 }}>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="w-5 h-5" />
                    Training History
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Job ID</TableHead>
                        <TableHead>Model</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead>Completed</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {trainingStatus.recentJobs.map((job) => (
                        <TableRow key={job.id}>
                          <TableCell className="font-mono text-xs">{job.id.slice(0, 8)}...</TableCell>
                          <TableCell>{job.modelType}</TableCell>
                          <TableCell>
                            <Badge variant={
                              job.status === 'completed' ? 'default' :
                              job.status === 'failed' ? 'destructive' :
                              job.status === 'running' ? 'secondary' : 'outline'
                            }>
                              {job.status}
                            </Badge>
                          </TableCell>
                          <TableCell>{formatDateTime(new Date(job.createdAt))}</TableCell>
                          <TableCell>
                            {job.completedAt ? formatDateTime(new Date(job.completedAt)) : '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </div>

        {/* Bottom padding */}
        <div className="h-20" />
      </div>
    </ResponsiveLayout>
  )
}
