/**
 * Data transformation utilities for PlotlyGlucoseChart
 * Converts API data structures to Plotly-compatible format
 */

// Types from API
interface GlucoseReading {
  timestamp: string
  value: number
  trend?: string | null
}

interface Prediction {
  timestamp: string
  linear?: number
  lstm?: number
}

interface TFTPrediction {
  timestamp: string
  horizon: number
  value: number
  lower: number
  upper: number
}

interface EffectPoint {
  minutesAhead: number
  iobEffect: number
  cobEffect: number
  netEffect: number
  remainingIOB?: number
  remainingCOB?: number
  remainingPOB?: number     // Future projected POB (grams)
  insulinActivity?: number  // 0-1 bell-shaped activity curve
  carbActivity?: number     // 0-1 bell-shaped activity curve
  expectedBg?: number       // Projected BG accounting for IOB/COB effects
  bgWithIobOnly?: number    // BG trajectory with only insulin effect (pull-down floor)
  bgWithCobOnly?: number    // BG trajectory with only carb effect (push-up ceiling)
}

interface Treatment {
  timestamp: string
  type: 'insulin' | 'carbs' | 'Correction Bolus' | 'Carb Correction'  // All possible treatment types
  value: number
  notes?: string  // Food description for carbs
  isLiquid?: boolean  // Whether it's a liquid (drink)
}

// Plotly data structures
export interface PlotlyXY {
  x: Date[]
  y: number[]
}

export interface PlotlyPredictions {
  timestamps: Date[]
  linear: (number | null)[]
  lstm: (number | null)[]
}

export interface PlotlyTFT {
  timestamps: Date[]
  values: number[]
  upperBounds: number[]
  lowerBounds: number[]
}

export interface PlotlyEffects {
  timestamps: Date[]
  iob: number[]
  cob: number[]
  pob: number[]              // Future projected POB (grams)
  iobEffect: number[]        // Cumulative IOB effect on BG (negative = lowering)
  cobEffect: number[]        // Cumulative COB effect on BG (positive = raising)
  insulinActivity: number[]  // Bell-shaped activity curve (0-1)
  carbActivity: number[]     // Bell-shaped activity curve (0-1)
  expectedBg: number[]       // Combined BG trajectory
  bgWithIobOnly: number[]    // BG trajectory with only insulin effect (pull-down floor)
  bgWithCobOnly: number[]    // BG trajectory with only carb effect (push-up ceiling)
}

export interface PlotlyTreatments {
  insulin: PlotlyXY & { emojis: string[] }
  carbs: PlotlyXY & { emojis: string[] }
}

/**
 * Map food description to emoji
 * Uses simple keyword matching for common foods
 */
export function getFoodEmoji(notes?: string, isLiquid?: boolean): string {
  if (!notes) return isLiquid ? '🥤' : '🍽️'

  const lower = notes.toLowerCase()

  // Drinks
  if (isLiquid || lower.includes('juice') || lower.includes('milk') || lower.includes('soda') || lower.includes('smoothie') || lower.includes('drink')) {
    if (lower.includes('orange') || lower.includes('oj')) return '🍊'
    if (lower.includes('apple')) return '🧃'
    if (lower.includes('milk') || lower.includes('chocolate milk')) return '🥛'
    if (lower.includes('smoothie')) return '🥤'
    if (lower.includes('coffee') || lower.includes('latte')) return '☕'
    if (lower.includes('tea')) return '🍵'
    return '🥤'
  }

  // Fruits
  if (lower.includes('apple')) return '🍎'
  if (lower.includes('banana')) return '🍌'
  if (lower.includes('orange')) return '🍊'
  if (lower.includes('grape')) return '🍇'
  if (lower.includes('strawberr') || lower.includes('berr')) return '🍓'
  if (lower.includes('watermelon') || lower.includes('melon')) return '🍉'
  if (lower.includes('peach')) return '🍑'
  if (lower.includes('pear')) return '🍐'
  if (lower.includes('cherry') || lower.includes('cherries')) return '🍒'
  if (lower.includes('fruit')) return '🍎'

  // Bread & Bakery
  if (lower.includes('bread') || lower.includes('toast')) return '🍞'
  if (lower.includes('bagel')) return '🥯'
  if (lower.includes('croissant')) return '🥐'
  if (lower.includes('muffin')) return '🧁'
  if (lower.includes('pancake') || lower.includes('waffle')) return '🥞'
  if (lower.includes('pretzel')) return '🥨'

  // Main dishes
  if (lower.includes('pizza')) return '🍕'
  if (lower.includes('burger') || lower.includes('hamburger')) return '🍔'
  if (lower.includes('hot dog')) return '🌭'
  if (lower.includes('taco')) return '🌮'
  if (lower.includes('burrito')) return '🌯'
  if (lower.includes('sandwich') || lower.includes('sub')) return '🥪'
  if (lower.includes('pasta') || lower.includes('spaghetti') || lower.includes('noodle')) return '🍝'
  if (lower.includes('rice') || lower.includes('fried rice')) return '🍚'
  if (lower.includes('sushi')) return '🍣'
  if (lower.includes('ramen')) return '🍜'
  if (lower.includes('soup')) return '🍲'
  if (lower.includes('salad')) return '🥗'
  if (lower.includes('chicken')) return '🍗'
  if (lower.includes('steak') || lower.includes('beef')) return '🥩'
  if (lower.includes('fish') || lower.includes('salmon')) return '🐟'
  if (lower.includes('shrimp') || lower.includes('seafood')) return '🦐'
  if (lower.includes('egg')) return '🍳'

  // Snacks & Sweets
  if (lower.includes('cookie')) return '🍪'
  if (lower.includes('cake')) return '🎂'
  if (lower.includes('donut') || lower.includes('doughnut')) return '🍩'
  if (lower.includes('chocolate')) return '🍫'
  if (lower.includes('candy') || lower.includes('sweet')) return '🍬'
  if (lower.includes('ice cream') || lower.includes('icecream')) return '🍦'
  if (lower.includes('popcorn')) return '🍿'
  if (lower.includes('chip') || lower.includes('crisp')) return '🍟'
  if (lower.includes('fries') || lower.includes('french fries')) return '🍟'
  if (lower.includes('pretzel')) return '🥨'
  if (lower.includes('cracker')) return '🧀'

  // Vegetables & Healthy
  if (lower.includes('carrot')) return '🥕'
  if (lower.includes('corn')) return '🌽'
  if (lower.includes('potato') || lower.includes('mashed')) return '🥔'
  if (lower.includes('broccoli') || lower.includes('vegetable')) return '🥦'
  if (lower.includes('avocado')) return '🥑'

  // Breakfast
  if (lower.includes('cereal') || lower.includes('oat')) return '🥣'
  if (lower.includes('yogurt')) return '🥛'
  if (lower.includes('bacon')) return '🥓'

  // Generic fallbacks
  if (lower.includes('snack')) return '🍿'
  if (lower.includes('meal') || lower.includes('dinner') || lower.includes('lunch')) return '🍽️'
  if (lower.includes('breakfast')) return '🍳'

  return '🍽️'  // Default plate
}

/**
 * Transform glucose readings to Plotly format
 * Filters out invalid values and converts timestamps to Date objects
 */
export function transformGlucoseData(readings: GlucoseReading[]): PlotlyXY {
  const filtered = readings.filter(r =>
    r.value !== undefined &&
    r.value !== null &&
    Number.isFinite(r.value)
  )

  return {
    x: filtered.map(r => new Date(r.timestamp)),
    y: filtered.map(r => r.value)
  }
}

/**
 * Transform linear/LSTM predictions to Plotly format
 * Creates separate arrays for each prediction type
 */
export function transformPredictions(
  predictions: Prediction[],
  baseTime: Date
): PlotlyPredictions {
  const timestamps: Date[] = []
  const linear: (number | null)[] = []
  const lstm: (number | null)[] = []

  predictions.forEach((p, i) => {
    // 5-minute intervals starting from base time
    timestamps.push(new Date(baseTime.getTime() + (i + 1) * 5 * 60 * 1000))
    linear.push(Number.isFinite(p.linear) ? p.linear! : null)
    lstm.push(p.lstm != null && Number.isFinite(p.lstm) ? p.lstm : null)
  })

  return { timestamps, linear, lstm }
}

/**
 * Transform TFT predictions with uncertainty bands
 * Returns values and the distances for error bars (not absolute bounds)
 */
export function transformTftWithBands(
  tftPredictions: TFTPrediction[],
  lastGlucose?: number,
  lastTimestamp?: Date
): PlotlyTFT {
  // Start with bridge point if we have last glucose reading
  const timestamps: Date[] = []
  const values: number[] = []
  const upperBounds: number[] = []
  const lowerBounds: number[] = []

  // Add bridge point to connect TFT line to actual glucose
  if (lastGlucose !== undefined && lastTimestamp !== undefined) {
    timestamps.push(lastTimestamp)
    values.push(lastGlucose)
    upperBounds.push(0) // No uncertainty at current point
    lowerBounds.push(0)
  }

  // Add TFT prediction points
  tftPredictions.forEach(p => {
    if (!Number.isFinite(p.value)) return

    timestamps.push(new Date(p.timestamp))
    values.push(p.value)
    // Error bars use distance from center, not absolute values
    upperBounds.push(Number.isFinite(p.upper) ? p.upper - p.value : 0)
    lowerBounds.push(Number.isFinite(p.lower) ? p.value - p.lower : 0)
  })

  return { timestamps, values, upperBounds, lowerBounds }
}

/**
 * Transform IOB/COB effect curves
 * Used for the decay visualization on secondary Y-axes
 * Includes activity curves for pharmacokinetic visualization
 * Includes three BG trajectory lines:
 * - expectedBg: Combined effect of IOB and COB
 * - bgWithIobOnly: Where BG would go with just insulin (pull-down floor)
 * - bgWithCobOnly: Where BG would go with just carbs (push-up ceiling)
 */
export function transformEffectCurve(
  effectCurve: EffectPoint[],
  baseTime: Date,
  currentBg?: number
): PlotlyEffects {
  const timestamps = effectCurve.map(e =>
    new Date(baseTime.getTime() + e.minutesAhead * 60 * 1000)
  )

  // Expected BG (combined IOB + COB effect)
  const expectedBgValues = effectCurve.map(e => {
    if (e.expectedBg !== undefined && e.expectedBg !== null) {
      return e.expectedBg
    }
    // Fallback: calculate from current BG + net effect
    if (currentBg !== undefined) {
      return currentBg + (e.cobEffect ?? 0) + (e.iobEffect ?? 0)
    }
    return 0
  })

  // BG with IOB only (pull-down floor - shows where BG would go without carbs)
  const bgWithIobOnlyValues = effectCurve.map(e => {
    if (e.bgWithIobOnly !== undefined && e.bgWithIobOnly !== null) {
      return e.bgWithIobOnly
    }
    // Fallback: calculate from current BG + IOB effect only
    if (currentBg !== undefined) {
      return currentBg + (e.iobEffect ?? 0)
    }
    return 0
  })

  // BG with COB only (push-up ceiling - shows where BG would go without insulin)
  const bgWithCobOnlyValues = effectCurve.map(e => {
    if (e.bgWithCobOnly !== undefined && e.bgWithCobOnly !== null) {
      return e.bgWithCobOnly
    }
    // Fallback: calculate from current BG + COB effect only
    if (currentBg !== undefined) {
      return currentBg + (e.cobEffect ?? 0)
    }
    return 0
  })

  return {
    timestamps,
    iob: effectCurve.map(e => e.remainingIOB ?? 0),
    cob: effectCurve.map(e => e.remainingCOB ?? 0),
    pob: effectCurve.map(e => e.remainingPOB ?? 0),
    iobEffect: effectCurve.map(e => e.iobEffect ?? 0),
    cobEffect: effectCurve.map(e => e.cobEffect ?? 0),
    insulinActivity: effectCurve.map(e => e.insulinActivity ?? 0),
    carbActivity: effectCurve.map(e => e.carbActivity ?? 0),
    expectedBg: expectedBgValues,
    bgWithIobOnly: bgWithIobOnlyValues,
    bgWithCobOnly: bgWithCobOnlyValues
  }
}

/**
 * Transform treatments into insulin and carb markers
 * Finds the nearest glucose reading for Y positioning
 * Returns emojis for each treatment marker
 */
export function transformTreatments(
  treatments: Treatment[],
  glucoseReadings: GlucoseReading[]
): PlotlyTreatments {
  const insulin: PlotlyXY & { emojis: string[] } = { x: [], y: [], emojis: [] }
  const carbs: PlotlyXY & { emojis: string[] } = { x: [], y: [], emojis: [] }

  if (glucoseReadings.length === 0) {
    return { insulin, carbs }
  }

  treatments.forEach(t => {
    const treatmentTime = new Date(t.timestamp).getTime()

    // Find nearest glucose reading for Y position
    const nearestReading = glucoseReadings.reduce((nearest, r) => {
      const diff = Math.abs(new Date(r.timestamp).getTime() - treatmentTime)
      const nearestDiff = Math.abs(new Date(nearest.timestamp).getTime() - treatmentTime)
      return diff < nearestDiff ? r : nearest
    })

    // Handle all insulin types: 'insulin' and 'Correction Bolus'
    if (t.type === 'insulin' || t.type === 'Correction Bolus') {
      insulin.x.push(new Date(t.timestamp))
      insulin.y.push(nearestReading.value)
      insulin.emojis.push('💉')  // Syringe emoji for insulin
    // Handle all carb types: 'carbs' and 'Carb Correction'
    } else if (t.type === 'carbs' || t.type === 'Carb Correction') {
      carbs.x.push(new Date(t.timestamp))
      carbs.y.push(nearestReading.value)
      carbs.emojis.push(getFoodEmoji(t.notes, t.isLiquid))  // Food emoji based on description
    }
  })

  return { insulin, carbs }
}

/**
 * Calculate dynamic Y-axis domain based on all data
 */
export function calculateGlucoseDomain(
  glucoseValues: number[],
  predictionValues: (number | null)[],
  tftValues: number[],
  criticalLow: number = 54,
  criticalHigh: number = 250
): [number, number] {
  const allValues = [
    ...glucoseValues,
    ...predictionValues.filter((v): v is number => v !== null && Number.isFinite(v)),
    ...tftValues.filter(v => Number.isFinite(v))
  ]

  if (allValues.length === 0) {
    return [40, 300]
  }

  const min = Math.min(...allValues, criticalLow)
  const max = Math.max(...allValues, criticalHigh)

  // Add padding and round to nice numbers
  const lowerBound = Math.max(40, Math.floor(min / 10) * 10 - 20)
  const upperBound = Math.min(400, Math.ceil(max / 10) * 10 + 20)

  // Safety check
  if (!Number.isFinite(lowerBound) || !Number.isFinite(upperBound) || lowerBound >= upperBound) {
    return [40, 300]
  }

  return [lowerBound, upperBound]
}

/**
 * Calculate IOB domain based on effect curve data
 */
export function calculateIobDomain(iobValues: number[]): [number, number] {
  if (iobValues.length === 0) return [0, 5]

  const max = Math.max(...iobValues.filter(v => Number.isFinite(v)))
  const upperBound = Math.max(5, Math.ceil(max * 1.2))

  return [0, upperBound]
}

/**
 * Calculate COB domain based on effect curve data
 */
export function calculateCobDomain(cobValues: number[]): [number, number] {
  if (cobValues.length === 0) return [0, 50]

  const max = Math.max(...cobValues.filter(v => Number.isFinite(v)))
  const upperBound = Math.max(50, Math.ceil(max * 1.2))

  return [0, upperBound]
}

// Historical IOB/COB/POB point from API (with BG pressure)
interface HistoricalIobCobPoint {
  timestamp: string
  iob: number
  cob: number
  pob: number          // Protein on Board (grams)
  bgPressure?: number  // Where BG is heading based on IOB+COB+POB
  actualBg?: number    // Actual BG at this time
}

export interface PlotlyHistoricalIobCob {
  timestamps: Date[]
  iob: number[]
  cob: number[]
  pob: number[]         // Protein on Board over time
  bgPressure: number[]  // Historical BG pressure line
  actualBg: number[]    // Actual BG readings for comparison
}

/**
 * Transform historical IOB/COB/POB data for continuous plotting
 * Shows how IOB, COB, POB, and BG pressure changed over time from all doses
 */
export function transformHistoricalIobCob(
  historicalData: HistoricalIobCobPoint[]
): PlotlyHistoricalIobCob {
  if (!historicalData || historicalData.length === 0) {
    return { timestamps: [], iob: [], cob: [], pob: [], bgPressure: [], actualBg: [] }
  }

  // Sort by timestamp and filter invalid values
  const sorted = [...historicalData]
    .filter(d => Number.isFinite(d.iob) && Number.isFinite(d.cob))
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())

  return {
    timestamps: sorted.map(d => new Date(d.timestamp)),
    iob: sorted.map(d => d.iob),
    cob: sorted.map(d => d.cob),
    pob: sorted.map(d => d.pob ?? 0),  // POB defaults to 0 if not present
    // Use actualBg as fallback if bgPressure is missing (when IOB=0 and COB=0, pressure equals actual)
    bgPressure: sorted.map(d => d.bgPressure ?? d.actualBg ?? 0),
    actualBg: sorted.map(d => d.actualBg ?? 0)
  }
}

/**
 * Calculate POB domain based on historical data
 */
export function calculatePobDomain(pobValues: number[]): [number, number] {
  if (pobValues.length === 0) return [0, 50]

  const max = Math.max(...pobValues.filter(v => Number.isFinite(v)))
  const upperBound = Math.max(50, Math.ceil(max * 1.2))

  return [0, upperBound]
}
