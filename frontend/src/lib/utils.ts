import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Glucose range helpers
export type GlucoseRange = 'critical-low' | 'low' | 'in-range' | 'high' | 'critical-high'

export function getGlucoseRange(value: number): GlucoseRange {
  if (value < 54) return 'critical-low'
  if (value < 70) return 'low'
  if (value <= 180) return 'in-range'
  if (value <= 250) return 'high'
  return 'critical-high'
}

export function getGlucoseColor(value: number): string {
  const range = getGlucoseRange(value)
  const colors: Record<GlucoseRange, string> = {
    'critical-low': '#dc2626',
    'low': '#f97316',
    'in-range': '#00c6ff',
    'high': '#eab308',
    'critical-high': '#dc2626',
  }
  return colors[range]
}

// Trend arrow mapping (from dexcom_reader_predict_v2.3.py)
export const trendArrows: Record<string, string> = {
  'DoubleUp': '⇈',
  'SingleUp': '↑',
  'FortyFiveUp': '↗',
  'Flat': '→',
  'FortyFiveDown': '↘',
  'SingleDown': '↓',
  'DoubleDown': '⇊',
  'NotComputable': '?',
  'RateOutOfRange': '⚠',
}

export function getTrendArrow(trend: string): string {
  return trendArrows[trend] || '?'
}

// Default timezone for the platform (Eastern Time)
export const DEFAULT_TIMEZONE = 'America/New_York'

// Format time helpers - always use Eastern timezone
export function formatTime(date: Date | string, timezone: string = DEFAULT_TIMEZONE): string {
  const d = typeof date === 'string' ? new Date(date) : date
  return d.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
    timeZone: timezone
  })
}

export function formatDate(date: Date | string, timezone: string = DEFAULT_TIMEZONE): string {
  const d = typeof date === 'string' ? new Date(date) : date
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: timezone
  })
}

// Format datetime with both date and time
export function formatDateTime(date: Date | string, timezone: string = DEFAULT_TIMEZONE): string {
  const d = typeof date === 'string' ? new Date(date) : date
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
    timeZone: timezone
  })
}

// Number formatting
export function formatDecimal(value: number, decimals: number = 1): string {
  return value.toFixed(decimals)
}
