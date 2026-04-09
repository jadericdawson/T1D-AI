/**
 * Reports Page
 *
 * Generates comprehensive pump/diabetes reports with charts and tables.
 * Supports 24h, 3-day, 7-day, 30-day, 90-day, and 1-year periods.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  ArrowLeft, FileBarChart, Loader2,
  Droplets, Pill, Activity, Zap, Heart
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@/components/ui/card'
import { ResponsiveLayout } from '@/components/layout/ResponsiveLayout'
import api from '@/lib/api'

// ── Types ─────────────────────────────────────────────────────────────────

interface TirBreakdown {
  very_low: number
  low: number
  in_range: number
  high: number
  very_high: number
}

interface InsulinSplit {
  basal_total: number
  basal_pct: number
  bolus_total: number
  bolus_pct: number
  auto_total: number
  auto_pct: number
}

interface PumpStatus {
  battery_percent: number | null
  control_mode: string | null
  pump_iob: number | null
  daily_basal_units: number | null
  daily_bolus_units: number | null
  daily_total_insulin: number | null
}

interface DailySummary {
  date: string
  basal_units: number
  bolus_units: number
  bolus_count: number
  auto_correction_units: number
  auto_correction_count: number
  total_insulin: number
  carbs: number
  meal_count: number
  avg_bg: number | null
  tir: number | null
  time_low: number | null
  time_high: number | null
  readings_count: number
}

interface HourlyPattern {
  hour: number
  avg_basal_rate: number
  avg_glucose: number | null
  min_glucose: number | null
  max_glucose: number | null
  auto_correction_insulin: number
}

interface GlucosePoint {
  time: string
  value: number
}

interface BolusEvent {
  time: string
  type: string
  units: number
  notes: string
}

interface MealEvent {
  time: string
  carbs: number
  notes: string
}

interface ReportData {
  period_label: string
  start_date: string
  end_date: string
  total_days: number
  timezone: string
  avg_daily_insulin: number
  avg_daily_carbs: number
  avg_glucose: number | null
  gmi: number | null
  cv: number | null
  total_readings: number
  total_auto_corrections: number
  avg_daily_auto_corrections: number
  tir: TirBreakdown
  insulin_split: InsulinSplit
  pump_status: PumpStatus | null
  daily_summaries: DailySummary[]
  hourly_patterns: HourlyPattern[]
  glucose_trace: GlucosePoint[]
  bolus_events: BolusEvent[]
  meal_events: MealEvent[]
}

type Period = '24h' | '3d' | '7d' | '30d' | '90d' | '1y'

const PERIODS: { value: Period; label: string }[] = [
  { value: '24h', label: '24 Hours' },
  { value: '3d', label: '3 Days' },
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' },
  { value: '1y', label: '1 Year' },
]

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } }
}

// ── Chart.js dynamic loading ──────────────────────────────────────────────

let chartJsLoaded = false
let chartJsPromise: Promise<void> | null = null

function loadChartJs(): Promise<void> {
  if (chartJsLoaded) return Promise.resolve()
  if (chartJsPromise) return chartJsPromise

  chartJsPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js'
    script.onload = () => { chartJsLoaded = true; resolve() }
    script.onerror = reject
    document.head.appendChild(script)
  })
  return chartJsPromise
}

// Safe number helper — handles null, undefined, NaN from API responses
const n = (v: any, digits?: number): string => {
  const num = Number(v ?? 0)
  return digits != null ? (isNaN(num) ? '0' : num.toFixed(digits)) : (isNaN(num) ? '0' : String(num))
}

// ── Main Component ────────────────────────────────────────────────────────

export default function Reports() {
  const [selectedPeriod, setSelectedPeriod] = useState<Period>('7d')
  const [report, setReport] = useState<ReportData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const chartsRef = useRef<Record<string, any>>({})

  const generateReport = useCallback(async (period: Period) => {
    setLoading(true)
    setError(null)
    try {
      const tzOffset = -(new Date().getTimezoneOffset()) / 60
      const res = await api.get('/api/v1/reports/pump-report', {
        params: { period, tz_offset: tzOffset }
      })
      setReport(res.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate report')
    } finally {
      setLoading(false)
    }
  }, [])

  // Render charts when report data changes
  useEffect(() => {
    if (!report) return

    loadChartJs().then(() => {
      // Destroy existing charts
      Object.values(chartsRef.current).forEach((c: any) => c?.destroy?.())
      chartsRef.current = {}

      renderGlucoseTrace(report)
      renderDailyInsulin(report)
      renderDailyCarbs(report)
      renderHourlyBasal(report)
      renderHourlyGlucose(report)
    })

    return () => {
      Object.values(chartsRef.current).forEach((c: any) => c?.destroy?.())
    }
  }, [report])

  const Chart = (window as any).Chart

  function renderGlucoseTrace(r: ReportData) {
    const canvas = document.getElementById('glucoseTrace') as HTMLCanvasElement
    if (!canvas || !Chart) return
    chartsRef.current.glucoseTrace = new Chart(canvas, {
      type: 'line',
      data: {
        labels: r.glucose_trace.map(p => p.time),
        datasets: [{
          data: r.glucose_trace.map(p => p.value),
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34,197,94,0.05)',
          borderWidth: 1, pointRadius: 0, fill: true, tension: 0.3,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: (i: any) => i.raw + ' mg/dL' } }
        },
        scales: {
          x: { display: true, ticks: { maxTicksLimit: 12, maxRotation: 0, color: '#8b8fa3', font: { size: 10 } }, grid: { display: false } },
          y: { min: 40, max: 400, grid: { color: 'rgba(46,51,70,0.4)' }, ticks: { color: '#8b8fa3' } }
        }
      },
      plugins: [{
        id: 'targetRange',
        beforeDraw: (chart: any) => {
          const ctx = chart.ctx, yScale = chart.scales.y, xScale = chart.scales.x
          const y70 = yScale.getPixelForValue(70), y180 = yScale.getPixelForValue(180)
          ctx.save()
          ctx.fillStyle = 'rgba(34,197,94,0.06)'
          ctx.fillRect(xScale.left, y180, xScale.width, y70 - y180)
          ctx.strokeStyle = 'rgba(34,197,94,0.2)'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1
          ctx.beginPath(); ctx.moveTo(xScale.left, y70); ctx.lineTo(xScale.right, y70); ctx.stroke()
          ctx.beginPath(); ctx.moveTo(xScale.left, y180); ctx.lineTo(xScale.right, y180); ctx.stroke()
          ctx.restore()
        }
      }]
    })
  }

  function renderDailyInsulin(r: ReportData) {
    const canvas = document.getElementById('dailyInsulin') as HTMLCanvasElement
    if (!canvas || !Chart) return
    const days = r.daily_summaries
    chartsRef.current.dailyInsulin = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: days.map(d => { const dt = new Date(d.date + 'T12:00:00'); return dt.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' }) }),
        datasets: [
          { label: 'Basal', data: days.map(d => d.basal_units ?? 0), backgroundColor: '#3b82f6', borderRadius: 3 },
          { label: 'Bolus', data: days.map(d => d.bolus_units ?? 0), backgroundColor: '#6366f1', borderRadius: 3 },
          { label: 'Auto-Correction', data: days.map(d => d.auto_correction_units ?? 0), backgroundColor: '#a855f7', borderRadius: 3 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: 'top' as const, labels: { usePointStyle: true, pointStyle: 'rectRounded', padding: 15, font: { size: 10 }, color: '#8b8fa3' } } },
        scales: {
          x: { stacked: true, grid: { display: false }, ticks: { color: '#8b8fa3' } },
          y: { stacked: true, grid: { color: 'rgba(46,51,70,0.4)' }, ticks: { color: '#8b8fa3' }, title: { display: true, text: 'Units', color: '#8b8fa3' } }
        }
      }
    })
  }

  function renderDailyCarbs(r: ReportData) {
    const canvas = document.getElementById('dailyCarbs') as HTMLCanvasElement
    if (!canvas || !Chart) return
    const days = r.daily_summaries
    chartsRef.current.dailyCarbs = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: days.map(d => { const dt = new Date(d.date + 'T12:00:00'); return dt.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' }) }),
        datasets: [{ label: 'Carbs', data: days.map(d => d.carbs ?? 0), backgroundColor: 'rgba(249,115,22,0.7)', borderColor: '#f97316', borderWidth: 1, borderRadius: 6 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#8b8fa3' } },
          y: { grid: { color: 'rgba(46,51,70,0.4)' }, ticks: { color: '#8b8fa3' }, title: { display: true, text: 'Grams', color: '#8b8fa3' } }
        }
      }
    })
  }

  function renderHourlyBasal(r: ReportData) {
    const canvas = document.getElementById('hourlyBasal') as HTMLCanvasElement
    if (!canvas || !Chart) return
    chartsRef.current.hourlyBasal = new Chart(canvas, {
      type: 'line',
      data: {
        labels: r.hourly_patterns.map(h => `${h.hour.toString().padStart(2, '0')}:00`),
        datasets: [{
          label: 'Avg Basal Rate', data: r.hourly_patterns.map(h => h.avg_basal_rate ?? 0),
          borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)',
          borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#3b82f6', fill: true, tension: 0.4,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (i: any) => i.raw + ' U/hr' } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#8b8fa3' } },
          y: { min: 0, grid: { color: 'rgba(46,51,70,0.4)' }, ticks: { color: '#8b8fa3' }, title: { display: true, text: 'U/hr', color: '#8b8fa3' } }
        }
      }
    })
  }

  function renderHourlyGlucose(r: ReportData) {
    const canvas = document.getElementById('hourlyGlucose') as HTMLCanvasElement
    if (!canvas || !Chart) return
    const hp = r.hourly_patterns
    chartsRef.current.hourlyGlucose = new Chart(canvas, {
      type: 'line',
      data: {
        labels: hp.map(h => `${h.hour.toString().padStart(2, '0')}:00`),
        datasets: [
          { label: 'Max', data: hp.map(h => h.max_glucose), borderColor: 'rgba(234,179,8,0.3)', backgroundColor: 'rgba(234,179,8,0.05)', borderWidth: 1, pointRadius: 0, fill: '+1', tension: 0.4 },
          { label: 'Avg', data: hp.map(h => h.avg_glucose), borderColor: '#22c55e', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#22c55e', tension: 0.4 },
          { label: 'Min', data: hp.map(h => h.min_glucose), borderColor: 'rgba(239,68,68,0.3)', backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, tension: 0.4 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: 'top' as const, labels: { usePointStyle: true, pointStyle: 'line', padding: 15, font: { size: 10 }, color: '#8b8fa3' } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#8b8fa3' } },
          y: { min: 40, grid: { color: 'rgba(46,51,70,0.4)' }, ticks: { color: '#8b8fa3' }, title: { display: true, text: 'mg/dL', color: '#8b8fa3' } }
        }
      },
      plugins: [{
        id: 'targetRangeH',
        beforeDraw: (chart: any) => {
          const ctx = chart.ctx, yScale = chart.scales.y, xScale = chart.scales.x
          const y70 = yScale.getPixelForValue(70), y180 = yScale.getPixelForValue(180)
          ctx.save(); ctx.fillStyle = 'rgba(34,197,94,0.06)'
          ctx.fillRect(xScale.left, y180, xScale.width, y70 - y180); ctx.restore()
        }
      }]
    })
  }

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <ResponsiveLayout title="Reports">
      <div className="min-h-screen p-4 md:p-6 max-w-7xl mx-auto">
        {/* Header */}
        <motion.header initial="hidden" animate="visible" variants={fadeIn} className="flex items-center gap-4 mb-6">
          <Link to="/dashboard">
            <Button variant="ghost" size="icon"><ArrowLeft className="w-5 h-5" /></Button>
          </Link>
          <div className="flex items-center gap-2">
            <FileBarChart className="w-6 h-6 text-primary" />
            <h1 className="text-2xl font-bold">Reports</h1>
          </div>
        </motion.header>

        {/* Period Selector */}
        <motion.div initial="hidden" animate="visible" variants={fadeIn}>
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Generate Report</CardTitle>
              <CardDescription>Select a time period and generate a comprehensive pump data report</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 mb-4">
                {PERIODS.map(p => (
                  <Button
                    key={p.value}
                    variant={selectedPeriod === p.value ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setSelectedPeriod(p.value)}
                    className="min-w-[80px]"
                  >
                    {p.label}
                  </Button>
                ))}
              </div>
              <Button
                onClick={() => generateReport(selectedPeriod)}
                disabled={loading}
                className="w-full sm:w-auto"
                size="lg"
              >
                {loading ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating...</>
                ) : (
                  <><FileBarChart className="w-4 h-4 mr-2" /> Generate Report</>
                )}
              </Button>
            </CardContent>
          </Card>
        </motion.div>

        {/* Error */}
        {error && (
          <Card className="mb-6 border-destructive">
            <CardContent className="pt-6">
              <p className="text-destructive">{error}</p>
            </CardContent>
          </Card>
        )}

        {/* Report Content */}
        {report && !loading && (
          <motion.div initial="hidden" animate="visible" variants={fadeIn} className="space-y-6">
            {/* Report Header */}
            <div className="text-center py-4">
              <h2 className="text-xl font-bold">
                Pump Data Report
              </h2>
              <p className="text-muted-foreground text-sm mt-1">
                {report.start_date} &rarr; {report.end_date} &middot; {report.total_days} days &middot; {report.timezone}
              </p>
            </div>

            {/* Overview Cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <StatCard icon={<Pill className="w-4 h-4" />} label="Avg Daily Insulin" value={`${report.avg_daily_insulin}`} unit="U/day" color="text-indigo-400" />
              <StatCard icon={<Heart className="w-4 h-4" />} label="Avg Daily Carbs" value={`${report.avg_daily_carbs}`} unit="g/day" color="text-orange-400" />
              <StatCard icon={<Droplets className="w-4 h-4" />} label="Avg Glucose" value={report.avg_glucose ? `${report.avg_glucose}` : '-'} unit="mg/dL"
                color={report.avg_glucose && report.avg_glucose >= 70 && report.avg_glucose <= 180 ? 'text-green-400' : 'text-yellow-400'}
                detail={report.gmi ? `GMI: ${report.gmi}% · CV: ${report.cv}%` : undefined} />
              <StatCard icon={<Activity className="w-4 h-4" />} label="Time in Range" value={`${report.tir.in_range}`} unit="%" color="text-green-400"
                detail="70-180 mg/dL target" />
              <StatCard icon={<Zap className="w-4 h-4" />} label="Auto-Corrections" value={`${report.total_auto_corrections}`}
                color="text-purple-400" detail={`${report.avg_daily_auto_corrections}/day`} />
            </div>

            {/* Insulin Split + TIR */}
            <div className="grid md:grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Insulin Distribution (Daily Avg)</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-2">
                    <InsulinSplitBox label="Basal" value={report.insulin_split.basal_total} pct={report.insulin_split.basal_pct} color="bg-blue-500/15 text-blue-400" />
                    <InsulinSplitBox label="Bolus" value={report.insulin_split.bolus_total} pct={report.insulin_split.bolus_pct} color="bg-indigo-500/15 text-indigo-400" />
                    <InsulinSplitBox label="Auto-Corr" value={report.insulin_split.auto_total} pct={report.insulin_split.auto_pct} color="bg-purple-500/15 text-purple-400" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Time in Range Breakdown</CardTitle></CardHeader>
                <CardContent>
                  <div className="flex h-8 rounded-lg overflow-hidden mb-3">
                    <TirBar width={report.tir.very_low} color="bg-red-800" label={report.tir.very_low} />
                    <TirBar width={report.tir.low} color="bg-red-500" label={report.tir.low} />
                    <TirBar width={report.tir.in_range} color="bg-green-500" label={report.tir.in_range} />
                    <TirBar width={report.tir.high} color="bg-yellow-500" label={report.tir.high} />
                    <TirBar width={report.tir.very_high} color="bg-orange-500" label={report.tir.very_high} />
                  </div>
                  <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                    <TirLegend color="bg-red-800" label={`Very Low <54 (${report.tir.very_low}%)`} />
                    <TirLegend color="bg-red-500" label={`Low 54-70 (${report.tir.low}%)`} />
                    <TirLegend color="bg-green-500" label={`In Range 70-180 (${report.tir.in_range}%)`} />
                    <TirLegend color="bg-yellow-500" label={`High 180-250 (${report.tir.high}%)`} />
                    <TirLegend color="bg-orange-500" label={`Very High >250 (${report.tir.very_high}%)`} />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Pump Status */}
            {report.pump_status && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Current Pump Status</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                    <PumpStatusItem label="Battery" value={report.pump_status.battery_percent != null ? `${n(report.pump_status.battery_percent, 0)}%` : '-'}
                      color={report.pump_status.battery_percent && report.pump_status.battery_percent > 30 ? 'text-green-400' : 'text-red-400'} />
                    <PumpStatusItem label="Mode" value={report.pump_status.control_mode || '-'} color="text-green-400" />
                    <PumpStatusItem label="IOB" value={report.pump_status.pump_iob != null ? `${n(report.pump_status.pump_iob, 2)}U` : '-'} color="text-indigo-400" />
                    <PumpStatusItem label="Today Basal" value={report.pump_status.daily_basal_units != null ? `${n(report.pump_status.daily_basal_units, 2)}U` : '-'} color="text-blue-400" />
                    <PumpStatusItem label="Today Bolus" value={report.pump_status.daily_bolus_units != null ? `${n(report.pump_status.daily_bolus_units, 2)}U` : '-'} color="text-indigo-400" />
                    <PumpStatusItem label="Today Total" value={report.pump_status.daily_total_insulin != null ? `${n(report.pump_status.daily_total_insulin, 2)}U` : '-'} color="text-white" />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Glucose Trace */}
            {report.glucose_trace.length > 0 && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Glucose Trace ({report.total_readings.toLocaleString()} readings)</CardTitle></CardHeader>
                <CardContent><div className="h-[300px] md:h-[350px]"><canvas id="glucoseTrace" /></div></CardContent>
              </Card>
            )}

            {/* Daily Charts */}
            <div className="grid md:grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Daily Insulin Breakdown</CardTitle></CardHeader>
                <CardContent><div className="h-[280px]"><canvas id="dailyInsulin" /></div></CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Daily Carbohydrates</CardTitle></CardHeader>
                <CardContent><div className="h-[280px]"><canvas id="dailyCarbs" /></div></CardContent>
              </Card>
            </div>

            {/* Hourly Patterns */}
            <div className="grid md:grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Average Basal Rate by Hour</CardTitle></CardHeader>
                <CardContent><div className="h-[280px]"><canvas id="hourlyBasal" /></div></CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Average Glucose by Hour</CardTitle></CardHeader>
                <CardContent><div className="h-[280px]"><canvas id="hourlyGlucose" /></div></CardContent>
              </Card>
            </div>

            {/* Daily Summary Table */}
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Daily Summary</CardTitle></CardHeader>
              <CardContent>
                <div className="overflow-x-auto max-h-[500px] overflow-y-auto rounded-lg border border-border">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/50 sticky top-0">
                      <tr>
                        <th className="px-3 py-2 text-left font-semibold">Date</th>
                        <th className="px-2 py-2 text-right font-semibold">Basal</th>
                        <th className="px-2 py-2 text-right font-semibold">Bolus</th>
                        <th className="px-2 py-2 text-right font-semibold">Auto</th>
                        <th className="px-2 py-2 text-right font-semibold">Total</th>
                        <th className="px-2 py-2 text-right font-semibold">Carbs</th>
                        <th className="px-2 py-2 text-right font-semibold">Meals</th>
                        <th className="px-2 py-2 text-right font-semibold">Avg BG</th>
                        <th className="px-2 py-2 text-right font-semibold">TIR</th>
                        <th className="px-2 py-2 text-right font-semibold">Low</th>
                        <th className="px-2 py-2 text-right font-semibold">High</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.daily_summaries.map(d => {
                        const dt = new Date(d.date + 'T12:00:00')
                        const dayLabel = dt.toLocaleDateString('en-US', { weekday: 'short', month: 'numeric', day: 'numeric' })
                        return (
                          <tr key={d.date} className="border-b border-border/50 hover:bg-muted/30">
                            <td className="px-3 py-1.5 font-medium">{dayLabel}</td>
                            <td className="px-2 py-1.5 text-right"><Badge variant="outline" className="text-blue-400 border-blue-400/30 text-[10px]">{n(d.basal_units, 1)}U</Badge></td>
                            <td className="px-2 py-1.5 text-right"><Badge variant="outline" className="text-indigo-400 border-indigo-400/30 text-[10px]">{n(d.bolus_units, 1)}U</Badge> <span className="text-muted-foreground">({d.bolus_count ?? 0})</span></td>
                            <td className="px-2 py-1.5 text-right"><Badge variant="outline" className="text-purple-400 border-purple-400/30 text-[10px]">{n(d.auto_correction_units, 1)}U</Badge> <span className="text-muted-foreground">({d.auto_correction_count ?? 0})</span></td>
                            <td className="px-2 py-1.5 text-right font-semibold">{n(d.total_insulin, 1)}U</td>
                            <td className="px-2 py-1.5 text-right"><Badge variant="outline" className="text-orange-400 border-orange-400/30 text-[10px]">{n(d.carbs)}g</Badge></td>
                            <td className="px-2 py-1.5 text-right text-muted-foreground">{d.meal_count ?? 0}</td>
                            <td className={`px-2 py-1.5 text-right font-semibold ${d.avg_bg && d.avg_bg >= 70 && d.avg_bg <= 180 ? 'text-green-400' : 'text-yellow-400'}`}>{d.avg_bg != null ? n(d.avg_bg) : '-'}</td>
                            <td className={`px-2 py-1.5 text-right font-semibold ${d.tir && d.tir >= 70 ? 'text-green-400' : d.tir && d.tir >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>{d.tir != null ? `${n(d.tir, 0)}%` : '-'}</td>
                            <td className={`px-2 py-1.5 text-right ${d.time_low && d.time_low > 4 ? 'text-red-400' : 'text-muted-foreground'}`}>{d.time_low != null ? `${n(d.time_low, 0)}%` : '-'}</td>
                            <td className={`px-2 py-1.5 text-right ${d.time_high && d.time_high > 25 ? 'text-orange-400' : 'text-muted-foreground'}`}>{d.time_high != null ? `${n(d.time_high, 0)}%` : '-'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Bolus + Meal Tables */}
            <div className="grid md:grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Bolus Events ({report.bolus_events.filter(b => b.type !== 'auto_correction').length} manual)</CardTitle></CardHeader>
                <CardContent>
                  <div className="overflow-y-auto max-h-[400px] rounded-lg border border-border">
                    <table className="w-full text-xs">
                      <thead className="bg-muted/50 sticky top-0">
                        <tr>
                          <th className="px-3 py-2 text-left font-semibold">Time</th>
                          <th className="px-2 py-2 text-left font-semibold">Type</th>
                          <th className="px-2 py-2 text-right font-semibold">Units</th>
                          <th className="px-2 py-2 text-left font-semibold">Notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.bolus_events.filter(b => b.type !== 'auto_correction').map((b, i) => (
                          <tr key={i} className="border-b border-border/50">
                            <td className="px-3 py-1.5">{b.time}</td>
                            <td className="px-2 py-1.5"><Badge variant="outline" className="text-indigo-400 border-indigo-400/30 text-[10px]">{b.type}</Badge></td>
                            <td className="px-2 py-1.5 text-right font-semibold">{n(b.units, 1)}U</td>
                            <td className="px-2 py-1.5 text-muted-foreground truncate max-w-[120px]">{b.notes || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Meal Log ({report.meal_events.length} meals)</CardTitle></CardHeader>
                <CardContent>
                  <div className="overflow-y-auto max-h-[400px] rounded-lg border border-border">
                    <table className="w-full text-xs">
                      <thead className="bg-muted/50 sticky top-0">
                        <tr>
                          <th className="px-3 py-2 text-left font-semibold">Time</th>
                          <th className="px-2 py-2 text-right font-semibold">Carbs</th>
                          <th className="px-2 py-2 text-left font-semibold">Notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.meal_events.map((m, i) => (
                          <tr key={i} className="border-b border-border/50">
                            <td className="px-3 py-1.5">{m.time}</td>
                            <td className="px-2 py-1.5 text-right">
                              <Badge variant="outline" className={`text-[10px] ${(m.carbs ?? 0) <= 20 ? 'text-green-400 border-green-400/30' : (m.carbs ?? 0) <= 50 ? 'text-yellow-400 border-yellow-400/30' : 'text-red-400 border-red-400/30'}`}>{n(m.carbs)}g</Badge>
                            </td>
                            <td className="px-2 py-1.5 text-muted-foreground truncate max-w-[200px]">{m.notes || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Auto-Corrections Table */}
            {report.bolus_events.filter(b => b.type === 'auto_correction').length > 0 && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Control-IQ Auto-Corrections ({report.bolus_events.filter(b => b.type === 'auto_correction').length})</CardTitle></CardHeader>
                <CardContent>
                  <div className="overflow-y-auto max-h-[400px] rounded-lg border border-border">
                    <table className="w-full text-xs">
                      <thead className="bg-muted/50 sticky top-0">
                        <tr>
                          <th className="px-3 py-2 text-left font-semibold">Time</th>
                          <th className="px-2 py-2 text-right font-semibold">Units</th>
                          <th className="px-2 py-2 text-left font-semibold">Notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.bolus_events.filter(b => b.type === 'auto_correction').map((b, i) => (
                          <tr key={i} className="border-b border-border/50">
                            <td className="px-3 py-1.5">{b.time}</td>
                            <td className="px-2 py-1.5 text-right font-semibold text-purple-400">{n(b.units, 1)}U</td>
                            <td className="px-2 py-1.5 text-muted-foreground">{b.notes || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Footer */}
            <p className="text-center text-xs text-muted-foreground py-4">
              Generated {new Date().toLocaleString()} &middot; {(report.total_readings ?? 0).toLocaleString()} glucose readings &middot; {report.daily_summaries.reduce((s, d) => s + (d.meal_count ?? 0) + (d.bolus_count ?? 0) + (d.auto_correction_count ?? 0), 0)} treatment events
            </p>

            <div className="h-20" />
          </motion.div>
        )}

        {/* Empty state */}
        {!report && !loading && !error && (
          <div className="text-center py-20 text-muted-foreground">
            <FileBarChart className="w-12 h-12 mx-auto mb-4 opacity-30" />
            <p>Select a time period and click Generate Report</p>
          </div>
        )}
      </div>
    </ResponsiveLayout>
  )
}

// ── Sub-components ──────────────────────────────────────────────────────

function StatCard({ icon, label, value, unit, color, detail }: {
  icon: React.ReactNode; label: string; value: string; unit?: string; color: string; detail?: string
}) {
  return (
    <Card>
      <CardContent className="pt-4 pb-3 px-4">
        <div className="flex items-center gap-1.5 text-muted-foreground text-[10px] font-semibold uppercase tracking-wider mb-2">
          {icon} {label}
        </div>
        <div className={`text-2xl md:text-3xl font-extrabold tracking-tight ${color}`}>
          {value}{unit && <span className="text-xs font-normal text-muted-foreground ml-1">{unit}</span>}
        </div>
        {detail && <p className="text-[10px] text-muted-foreground mt-1">{detail}</p>}
      </CardContent>
    </Card>
  )
}

function InsulinSplitBox({ label, value, pct, color }: { label: string; value: number; pct: number; color: string }) {
  return (
    <div className={`rounded-lg p-3 text-center ${color}`}>
      <div className="text-[10px] font-semibold uppercase tracking-wider mb-1">{label}</div>
      <div className="text-xl font-bold">{n(value, 1)}U</div>
      <div className="text-[10px] opacity-70">{n(pct, 0)}%</div>
    </div>
  )
}

function TirBar({ width, color, label }: { width: number; color: string; label: number }) {
  if (width < 0.5) return null
  return (
    <div className={`${color} flex items-center justify-center text-[10px] font-semibold text-white`} style={{ width: `${width}%`, minWidth: width > 3 ? undefined : '2px' }}>
      {width > 4 ? `${n(label, 0)}%` : ''}
    </div>
  )
}

function TirLegend({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-2.5 h-2.5 rounded-sm ${color}`} />
      <span>{label}</span>
    </div>
  )
}

function PumpStatusItem({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-muted/50 rounded-lg p-2 text-center">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">{label}</div>
      <div className={`text-lg font-bold mt-0.5 ${color}`}>{value}</div>
    </div>
  )
}
