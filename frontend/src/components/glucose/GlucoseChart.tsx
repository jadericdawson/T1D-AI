/**
 * Glucose Chart Component
 * Displays glucose readings over time with predictions and target range
 */
import { useMemo } from 'react'
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  Scatter,
  Legend,
} from 'recharts'
import { format, subHours, isAfter } from 'date-fns'
import { getGlucoseColor, cn } from '@/lib/utils'

// Types
interface GlucoseReading {
  timestamp: string
  value: number
  trend?: string | null
}

interface Treatment {
  timestamp: string
  type: 'insulin' | 'carbs'
  value: number
}

interface Prediction {
  timestamp: string
  linear?: number
  lstm?: number
}

interface GlucoseChartProps {
  readings: GlucoseReading[]
  predictions?: Prediction[]
  treatments?: Treatment[]
  timeRange: '1hr' | '3hr' | '6hr' | '12hr' | '24hr'
  targetLow?: number
  targetHigh?: number
  criticalLow?: number
  criticalHigh?: number
  showPredictions?: boolean
  showTreatments?: boolean
  showTargetRange?: boolean
  className?: string
}

// Time range in hours
const timeRangeHours: Record<string, number> = {
  '1hr': 1,
  '3hr': 3,
  '6hr': 6,
  '12hr': 12,
  '24hr': 24,
}

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null

  const time = format(new Date(label), 'h:mm a')
  const glucoseData = payload.find((p: any) => p.dataKey === 'value')
  const linearPred = payload.find((p: any) => p.dataKey === 'linear')
  const lstmPred = payload.find((p: any) => p.dataKey === 'lstm')

  return (
    <div className="glass-card p-3 text-sm border border-gray-700/50">
      <p className="text-gray-400 mb-2">{time}</p>
      {glucoseData && (
        <p className="text-white">
          <span className="text-cyan">Glucose:</span>{' '}
          <span style={{ color: getGlucoseColor(glucoseData.value) }}>
            {glucoseData.value} mg/dL
          </span>
        </p>
      )}
      {linearPred && (
        <p className="text-gray-400">
          Linear: {linearPred.value?.toFixed(0)} mg/dL
        </p>
      )}
      {lstmPred && (
        <p className="text-purple-400">
          LSTM: {lstmPred.value?.toFixed(0)} mg/dL
        </p>
      )}
    </div>
  )
}

// Custom dot for treatments
const TreatmentDot = ({ cx, cy, payload }: any) => {
  if (!payload.treatment) return null

  const isCarbs = payload.treatment.type === 'carbs'
  const size = 8

  return (
    <g>
      {isCarbs ? (
        <polygon
          points={`${cx},${cy - size} ${cx + size},${cy + size} ${cx - size},${cy + size}`}
          fill="#22c55e"
          stroke="#166534"
          strokeWidth={1}
        />
      ) : (
        <circle
          cx={cx}
          cy={cy}
          r={size / 2}
          fill="#f97316"
          stroke="#c2410c"
          strokeWidth={1}
        />
      )}
    </g>
  )
}

export function GlucoseChart({
  readings,
  predictions = [],
  treatments = [],
  timeRange = '3hr',
  targetLow = 70,
  targetHigh = 180,
  criticalLow = 54,
  criticalHigh = 250,
  showPredictions = true,
  showTreatments = true,
  showTargetRange = true,
  className,
}: GlucoseChartProps) {
  // Process data for chart
  const chartData = useMemo(() => {
    const hours = timeRangeHours[timeRange]
    const cutoff = subHours(new Date(), hours)

    // Filter readings within time range
    const filteredReadings = readings
      .filter((r) => isAfter(new Date(r.timestamp), cutoff))
      .map((r) => ({
        timestamp: new Date(r.timestamp).getTime(),
        value: r.value,
        trend: r.trend,
      }))

    // Add treatments as markers
    const treatmentMap = new Map<number, Treatment>()
    treatments.forEach((t) => {
      if (isAfter(new Date(t.timestamp), cutoff)) {
        const ts = new Date(t.timestamp).getTime()
        treatmentMap.set(ts, t)
      }
    })

    // Merge with treatment data
    const dataWithTreatments = filteredReadings.map((r) => ({
      ...r,
      treatment: treatmentMap.get(r.timestamp) || null,
    }))

    // Add predictions (extend beyond current time)
    if (showPredictions && predictions.length > 0) {
      const latestReading = filteredReadings[filteredReadings.length - 1]
      if (latestReading) {
        predictions.forEach((pred, i) => {
          const predTime = latestReading.timestamp + (i + 1) * 5 * 60 * 1000 // 5 min intervals
          dataWithTreatments.push({
            timestamp: predTime,
            value: undefined as any,
            trend: null,
            treatment: null,
            linear: pred.linear,
            lstm: pred.lstm,
          })
        })
      }
    }

    return dataWithTreatments.sort((a, b) => a.timestamp - b.timestamp)
  }, [readings, predictions, treatments, timeRange, showPredictions])

  // Calculate Y axis domain
  const yDomain = useMemo(() => {
    const values = chartData
      .flatMap((d) => [d.value, d.linear, d.lstm])
      .filter((v) => v !== undefined && v !== null) as number[]

    if (values.length === 0) return [40, 300]

    const min = Math.min(...values, criticalLow)
    const max = Math.max(...values, criticalHigh)

    return [
      Math.max(40, Math.floor(min / 10) * 10 - 20),
      Math.min(400, Math.ceil(max / 10) * 10 + 20),
    ]
  }, [chartData, criticalLow, criticalHigh])

  // Format X axis
  const formatXAxis = (timestamp: number) => {
    return format(new Date(timestamp), timeRange === '24hr' ? 'HH:mm' : 'h:mm a')
  }

  if (chartData.length === 0) {
    return (
      <div className={cn('flex items-center justify-center h-80 text-gray-500', className)}>
        No glucose data available
      </div>
    )
  }

  return (
    <div className={cn('w-full h-80', className)}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          {/* Grid */}
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#374151"
            opacity={0.5}
          />

          {/* Target range background */}
          {showTargetRange && (
            <ReferenceArea
              y1={targetLow}
              y2={targetHigh}
              fill="#00c6ff"
              fillOpacity={0.1}
            />
          )}

          {/* Critical lines */}
          <ReferenceLine
            y={criticalLow}
            stroke="#dc2626"
            strokeDasharray="5 5"
            strokeWidth={1}
          />
          <ReferenceLine
            y={criticalHigh}
            stroke="#dc2626"
            strokeDasharray="5 5"
            strokeWidth={1}
          />

          {/* Target lines */}
          <ReferenceLine
            y={targetLow}
            stroke="#00c6ff"
            strokeDasharray="3 3"
            strokeWidth={1}
            opacity={0.5}
          />
          <ReferenceLine
            y={targetHigh}
            stroke="#00c6ff"
            strokeDasharray="3 3"
            strokeWidth={1}
            opacity={0.5}
          />

          {/* Axes */}
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatXAxis}
            stroke="#6b7280"
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            domain={yDomain}
            stroke="#6b7280"
            fontSize={11}
            tickLine={false}
            axisLine={false}
            width={40}
          />

          {/* Tooltip */}
          <Tooltip content={<CustomTooltip />} />

          {/* Glucose line */}
          <Line
            type="monotone"
            dataKey="value"
            stroke="#00c6ff"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6, fill: '#00c6ff', stroke: '#fff', strokeWidth: 2 }}
            connectNulls={false}
          />

          {/* Linear prediction line */}
          {showPredictions && (
            <Line
              type="monotone"
              dataKey="linear"
              stroke="#9ca3af"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={false}
              connectNulls={false}
            />
          )}

          {/* LSTM prediction line */}
          {showPredictions && (
            <Line
              type="monotone"
              dataKey="lstm"
              stroke="#a855f7"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              connectNulls={false}
            />
          )}

          {/* Treatment markers */}
          {showTreatments && (
            <Scatter
              dataKey="value"
              shape={<TreatmentDot />}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// Chart legend component
export function ChartLegend() {
  return (
    <div className="flex items-center gap-6 text-sm">
      <div className="flex items-center gap-2">
        <div className="w-4 h-0.5 bg-cyan" />
        <span className="text-gray-400">Glucose</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-4 h-0.5 bg-purple-500 border-dashed" style={{ borderTopWidth: 2, borderTopStyle: 'dashed' }} />
        <span className="text-gray-400">LSTM Prediction</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-4 h-0.5 bg-gray-400" style={{ borderTopWidth: 2, borderTopStyle: 'dashed' }} />
        <span className="text-gray-400">Linear</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-b-[8px] border-b-green-500" />
        <span className="text-gray-400">Carbs</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-orange-500" />
        <span className="text-gray-400">Insulin</span>
      </div>
    </div>
  )
}

export default GlucoseChart
