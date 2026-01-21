/**
 * Data Viewer Page
 * Comprehensive view of glucose readings and treatments with filtering and export
 */
import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  Database, ArrowLeft, Loader2, FileJson, FileSpreadsheet,
  Activity, Droplet, Search, Calendar, ChevronLeft, ChevronRight,
  TrendingUp, TrendingDown, Minus, PieChart, Filter
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Card,
  CardContent,
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ResponsiveLayout } from '@/components/layout/ResponsiveLayout'
import { cn, formatDateTime } from '@/lib/utils'
import { glucoseApi, treatmentsApi, trainingApi } from '@/lib/api'
import { useQuery } from '@tanstack/react-query'
import { useAuthStore } from '@/stores/authStore'
import { useGlucoseStore } from '@/stores/glucoseStore'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } }
}

// Time range options
const TIME_RANGES = [
  { value: '1', label: '24 Hours' },
  { value: '3', label: '3 Days' },
  { value: '7', label: '7 Days' },
  { value: '14', label: '14 Days' },
  { value: '30', label: '30 Days' },
]

// Glucose range filter options
const GLUCOSE_FILTERS = [
  { value: 'all', label: 'All Readings' },
  { value: 'low', label: 'Low (<70 mg/dL)' },
  { value: 'in-range', label: 'In Range (70-180)' },
  { value: 'high', label: 'High (>180 mg/dL)' },
]

// Treatment type filter options
const TREATMENT_FILTERS = [
  { value: 'all', label: 'All Treatments' },
  { value: 'insulin', label: 'Insulin Only' },
  { value: 'carbs', label: 'Carbs Only' },
]

export default function DataViewer() {
  const user = useAuthStore(state => state.user)
  const userId = user?.id || ''
  const { preferences } = useGlucoseStore()

  // State for filters
  const [timeRange, setTimeRange] = useState('7')
  const [glucoseFilter, setGlucoseFilter] = useState('all')
  const [treatmentFilter, setTreatmentFilter] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [glucosePage, setGlucosePage] = useState(0)
  const [treatmentPage, setTreatmentPage] = useState(0)
  const pageSize = 50

  // Fetch data stats
  const { data: dataStats, isLoading: isLoadingStats } = useQuery({
    queryKey: ['training', 'data-stats', userId],
    queryFn: () => trainingApi.getDataStats(),
    enabled: !!userId,
  })

  // Fetch glucose readings
  const { data: glucoseData, isLoading: isLoadingGlucose } = useQuery({
    queryKey: ['glucose', 'history', userId, timeRange],
    queryFn: () => glucoseApi.getHistory(userId, parseInt(timeRange) * 24),
    enabled: !!userId,
  })

  // Fetch treatments
  const { data: treatmentsData, isLoading: isLoadingTreatments } = useQuery({
    queryKey: ['treatments', userId, timeRange],
    queryFn: () => treatmentsApi.getRecent(parseInt(timeRange) * 24),
    enabled: !!userId,
  })

  // Filter glucose readings
  const filteredGlucose = useMemo(() => {
    if (!glucoseData?.readings) return []

    return glucoseData.readings.filter((reading: { value: number }) => {
      if (glucoseFilter === 'low') return reading.value < 70
      if (glucoseFilter === 'in-range') return reading.value >= 70 && reading.value <= 180
      if (glucoseFilter === 'high') return reading.value > 180
      return true
    })
  }, [glucoseData, glucoseFilter])

  // Filter treatments
  const filteredTreatments = useMemo(() => {
    if (!treatmentsData || !Array.isArray(treatmentsData)) return []

    let filtered = treatmentsData

    // Filter by type
    if (treatmentFilter === 'insulin') {
      filtered = filtered.filter((t: { type: string }) => t.type === 'insulin')
    } else if (treatmentFilter === 'carbs') {
      filtered = filtered.filter((t: { type: string }) => t.type === 'carbs')
    }

    // Filter by search query (notes)
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter((t: { notes?: string }) =>
        t.notes?.toLowerCase().includes(query)
      )
    }

    return filtered
  }, [treatmentsData, treatmentFilter, searchQuery])

  // Paginate data
  const paginatedGlucose = filteredGlucose.slice(
    glucosePage * pageSize,
    (glucosePage + 1) * pageSize
  )
  const paginatedTreatments = filteredTreatments.slice(
    treatmentPage * pageSize,
    (treatmentPage + 1) * pageSize
  )

  // Calculate time in range stats
  const tirStats = useMemo(() => {
    if (!glucoseData?.readings || glucoseData.readings.length === 0) {
      return { low: 0, inRange: 0, high: 0, veryHigh: 0 }
    }

    const total = glucoseData.readings.length
    const lowThreshold = preferences.lowThreshold || 70
    const highThreshold = preferences.highThreshold || 180
    const veryHighThreshold = preferences.criticalHighThreshold || 250

    let low = 0, inRange = 0, high = 0, veryHigh = 0

    glucoseData.readings.forEach((r: { value: number }) => {
      if (r.value < lowThreshold) low++
      else if (r.value <= highThreshold) inRange++
      else if (r.value <= veryHighThreshold) high++
      else veryHigh++
    })

    return {
      low: (low / total) * 100,
      inRange: (inRange / total) * 100,
      high: (high / total) * 100,
      veryHigh: (veryHigh / total) * 100,
    }
  }, [glucoseData, preferences])

  // Export functions
  const exportAsJSON = () => {
    const data = {
      glucose: glucoseData?.readings || [],
      treatments: treatmentsData || [],
      exportedAt: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `t1d-data-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportAsCSV = () => {
    // Export glucose readings
    const glucoseCSV = [
      ['timestamp', 'value', 'trend'].join(','),
      ...(glucoseData?.readings || []).map((r) =>
        [r.timestamp, r.value, r.trend || ''].join(',')
      )
    ].join('\n')

    const blob = new Blob([glucoseCSV], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `glucose-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Get trend icon
  const getTrendIcon = (trend: string) => {
    if (trend?.includes('Up') || trend?.includes('Rising')) {
      return <TrendingUp className="w-4 h-4 text-orange-500" />
    }
    if (trend?.includes('Down') || trend?.includes('Falling')) {
      return <TrendingDown className="w-4 h-4 text-blue-500" />
    }
    return <Minus className="w-4 h-4 text-muted-foreground" />
  }

  // Get glucose color
  const getGlucoseColor = (value: number) => {
    if (value < 70) return 'text-red-500'
    if (value > 180) return 'text-orange-500'
    return 'text-green-500'
  }

  return (
    <ResponsiveLayout title="My Data">
      <div className="min-h-screen p-4 md:p-6 max-w-6xl mx-auto">
        {/* Header */}
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
              <Database className="w-6 h-6 text-primary" />
              <h1 className="text-2xl font-bold">My Data</h1>
            </div>
          </div>

          {/* Time Range Selector */}
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-36">
              <Calendar className="w-4 h-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TIME_RANGES.map(range => (
                <SelectItem key={range.value} value={range.value}>
                  {range.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </motion.header>

        <div className="grid gap-6">
          {/* Statistics Cards */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Glucose Readings</p>
                      <p className="text-2xl font-bold">
                        {isLoadingStats ? '...' : dataStats?.totalReadings?.toLocaleString() || '0'}
                      </p>
                    </div>
                    <Activity className="w-8 h-8 text-primary/20" />
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Treatments</p>
                      <p className="text-2xl font-bold">
                        {isLoadingStats ? '...' : dataStats?.totalTreatments?.toLocaleString() || '0'}
                      </p>
                    </div>
                    <Droplet className="w-8 h-8 text-blue-500/20" />
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Days of Data</p>
                      <p className="text-2xl font-bold">
                        {isLoadingStats ? '...' : dataStats?.dataSpanDays || '0'}
                      </p>
                    </div>
                    <Calendar className="w-8 h-8 text-purple-500/20" />
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Readings/Day</p>
                      <p className="text-2xl font-bold">
                        {isLoadingStats ? '...' : dataStats?.readingsPerDay?.toFixed(0) || '0'}
                      </p>
                    </div>
                    <TrendingUp className="w-8 h-8 text-green-500/20" />
                  </div>
                </CardContent>
              </Card>
            </div>
          </motion.div>

          {/* Time in Range Card */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn} transition={{ delay: 0.1 }}>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PieChart className="w-5 h-5" />
                  Time in Range ({TIME_RANGES.find(r => r.value === timeRange)?.label})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4">
                  {/* Visual bar */}
                  <div className="flex-1 h-8 rounded-full overflow-hidden flex">
                    <div
                      className="bg-red-500 transition-all"
                      style={{ width: `${tirStats.low}%` }}
                      title={`Low: ${tirStats.low.toFixed(1)}%`}
                    />
                    <div
                      className="bg-green-500 transition-all"
                      style={{ width: `${tirStats.inRange}%` }}
                      title={`In Range: ${tirStats.inRange.toFixed(1)}%`}
                    />
                    <div
                      className="bg-orange-500 transition-all"
                      style={{ width: `${tirStats.high}%` }}
                      title={`High: ${tirStats.high.toFixed(1)}%`}
                    />
                    <div
                      className="bg-red-700 transition-all"
                      style={{ width: `${tirStats.veryHigh}%` }}
                      title={`Very High: ${tirStats.veryHigh.toFixed(1)}%`}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-4 mt-4 text-center">
                  <div>
                    <div className="w-3 h-3 rounded-full bg-red-500 mx-auto mb-1" />
                    <p className="text-xs text-muted-foreground">Low</p>
                    <p className="font-medium">{tirStats.low.toFixed(1)}%</p>
                  </div>
                  <div>
                    <div className="w-3 h-3 rounded-full bg-green-500 mx-auto mb-1" />
                    <p className="text-xs text-muted-foreground">In Range</p>
                    <p className="font-medium text-green-500">{tirStats.inRange.toFixed(1)}%</p>
                  </div>
                  <div>
                    <div className="w-3 h-3 rounded-full bg-orange-500 mx-auto mb-1" />
                    <p className="text-xs text-muted-foreground">High</p>
                    <p className="font-medium">{tirStats.high.toFixed(1)}%</p>
                  </div>
                  <div>
                    <div className="w-3 h-3 rounded-full bg-red-700 mx-auto mb-1" />
                    <p className="text-xs text-muted-foreground">Very High</p>
                    <p className="font-medium">{tirStats.veryHigh.toFixed(1)}%</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Data Tables */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn} transition={{ delay: 0.2 }}>
            <Card>
              <Tabs defaultValue="glucose">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <TabsList>
                      <TabsTrigger value="glucose">Glucose Readings</TabsTrigger>
                      <TabsTrigger value="treatments">Treatments</TabsTrigger>
                    </TabsList>

                    {/* Export Buttons */}
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={exportAsJSON}>
                        <FileJson className="w-4 h-4 mr-2" />
                        JSON
                      </Button>
                      <Button variant="outline" size="sm" onClick={exportAsCSV}>
                        <FileSpreadsheet className="w-4 h-4 mr-2" />
                        CSV
                      </Button>
                    </div>
                  </div>
                </CardHeader>

                <CardContent>
                  {/* Glucose Tab */}
                  <TabsContent value="glucose" className="mt-0">
                    {/* Filter */}
                    <div className="flex items-center gap-4 mb-4">
                      <Select value={glucoseFilter} onValueChange={setGlucoseFilter}>
                        <SelectTrigger className="w-48">
                          <Filter className="w-4 h-4 mr-2" />
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {GLUCOSE_FILTERS.map(f => (
                            <SelectItem key={f.value} value={f.value}>{f.label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <span className="text-sm text-muted-foreground">
                        {filteredGlucose.length} readings
                      </span>
                    </div>

                    {isLoadingGlucose ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="w-6 h-6 animate-spin" />
                      </div>
                    ) : (
                      <>
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Time</TableHead>
                              <TableHead>Value</TableHead>
                              <TableHead>Trend</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {paginatedGlucose.map((reading, i) => (
                              <TableRow key={i}>
                                <TableCell>{formatDateTime(new Date(reading.timestamp))}</TableCell>
                                <TableCell className={cn('font-medium', getGlucoseColor(reading.value))}>
                                  {reading.value} mg/dL
                                </TableCell>
                                <TableCell>
                                  <div className="flex items-center gap-2">
                                    {getTrendIcon(reading.trend || '')}
                                    <span className="text-sm text-muted-foreground">
                                      {reading.trend || 'Flat'}
                                    </span>
                                  </div>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>

                        {/* Pagination */}
                        <div className="flex items-center justify-between mt-4">
                          <span className="text-sm text-muted-foreground">
                            Showing {glucosePage * pageSize + 1} - {Math.min((glucosePage + 1) * pageSize, filteredGlucose.length)} of {filteredGlucose.length}
                          </span>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={glucosePage === 0}
                              onClick={() => setGlucosePage(p => p - 1)}
                            >
                              <ChevronLeft className="w-4 h-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={(glucosePage + 1) * pageSize >= filteredGlucose.length}
                              onClick={() => setGlucosePage(p => p + 1)}
                            >
                              <ChevronRight className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                      </>
                    )}
                  </TabsContent>

                  {/* Treatments Tab */}
                  <TabsContent value="treatments" className="mt-0">
                    {/* Filters */}
                    <div className="flex items-center gap-4 mb-4">
                      <Select value={treatmentFilter} onValueChange={setTreatmentFilter}>
                        <SelectTrigger className="w-48">
                          <Filter className="w-4 h-4 mr-2" />
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {TREATMENT_FILTERS.map(f => (
                            <SelectItem key={f.value} value={f.value}>{f.label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <div className="relative flex-1 max-w-sm">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                        <Input
                          placeholder="Search by food notes..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="pl-10"
                        />
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {filteredTreatments.length} treatments
                      </span>
                    </div>

                    {isLoadingTreatments ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="w-6 h-6 animate-spin" />
                      </div>
                    ) : (
                      <>
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Time</TableHead>
                              <TableHead>Type</TableHead>
                              <TableHead>Amount</TableHead>
                              <TableHead>Notes</TableHead>
                              <TableHead>GI</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {paginatedTreatments.map((treatment: {
                              id: string
                              timestamp: string
                              type: string
                              insulin?: number
                              carbs?: number
                              notes?: string
                              glycemicIndex?: number
                            }) => (
                              <TableRow key={treatment.id}>
                                <TableCell>{formatDateTime(new Date(treatment.timestamp))}</TableCell>
                                <TableCell>
                                  <Badge variant={treatment.type === 'insulin' ? 'default' : 'secondary'}>
                                    {treatment.type}
                                  </Badge>
                                </TableCell>
                                <TableCell className="font-medium">
                                  {treatment.type === 'insulin'
                                    ? `${treatment.insulin} units`
                                    : `${treatment.carbs}g carbs`}
                                </TableCell>
                                <TableCell className="max-w-xs truncate">
                                  {treatment.notes || '-'}
                                </TableCell>
                                <TableCell>
                                  {treatment.glycemicIndex || '-'}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>

                        {/* Pagination */}
                        <div className="flex items-center justify-between mt-4">
                          <span className="text-sm text-muted-foreground">
                            Showing {treatmentPage * pageSize + 1} - {Math.min((treatmentPage + 1) * pageSize, filteredTreatments.length)} of {filteredTreatments.length}
                          </span>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={treatmentPage === 0}
                              onClick={() => setTreatmentPage(p => p - 1)}
                            >
                              <ChevronLeft className="w-4 h-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={(treatmentPage + 1) * pageSize >= filteredTreatments.length}
                              onClick={() => setTreatmentPage(p => p + 1)}
                            >
                              <ChevronRight className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                      </>
                    )}
                  </TabsContent>
                </CardContent>
              </Tabs>
            </Card>
          </motion.div>
        </div>

        {/* Bottom padding */}
        <div className="h-20" />
      </div>
    </ResponsiveLayout>
  )
}
