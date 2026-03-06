/**
 * Timezone-aware formatting utilities.
 * Uses the user's configured timezone preference for all time display.
 */

/** Get the user's configured timezone from the auth store, falling back to browser default. */
export function getUserTimezone(): string {
  try {
    const stored = localStorage.getItem('t1d-ai-auth')
    if (stored) {
      const data = JSON.parse(stored)
      const tz = data.state?.timezone
      if (tz) return tz
    }
  } catch {
    // Fall through to default
  }
  return Intl.DateTimeFormat().resolvedOptions().timeZone
}

/** Format a date/timestamp as a time string in the user's timezone. */
export function formatInTimezone(
  date: Date | string,
  timezone?: string,
  options?: Intl.DateTimeFormatOptions
): string {
  const d = typeof date === 'string' ? new Date(date) : date
  const tz = timezone || getUserTimezone()
  return d.toLocaleTimeString('en-US', { timeZone: tz, ...options })
}

/** Format a date/timestamp as a date+time string in the user's timezone. */
export function formatDateTimeInTimezone(
  date: Date | string,
  timezone?: string,
  options?: Intl.DateTimeFormatOptions
): string {
  const d = typeof date === 'string' ? new Date(date) : date
  const tz = timezone || getUserTimezone()
  return d.toLocaleString('en-US', { timeZone: tz, ...options })
}

/**
 * Convert a UTC ISO timestamp to datetime-local format in user's timezone.
 * Used for <input type="datetime-local"> fields.
 */
export function formatDateTimeLocal(isoTimestamp: string, timezone?: string): string {
  const tz = timezone || getUserTimezone()
  const date = new Date(isoTimestamp)
  // Format parts in the target timezone
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: tz,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).formatToParts(date)

  const get = (type: string) => parts.find(p => p.type === type)?.value || '00'
  return `${get('year')}-${get('month')}-${get('day')}T${get('hour')}:${get('minute')}`
}
