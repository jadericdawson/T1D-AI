/**
 * Pump Status Card
 * Displays comprehensive Tandem pump status below the Predicted BG card
 */
import { motion } from 'framer-motion'
import {
  Battery, BatteryLow, BatteryMedium, BatteryFull,
  Activity, AlertTriangle, AlertCircle, Droplets,
  Syringe, Zap, Power, Shield, ShieldAlert, ShieldOff,
  RefreshCw,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import type { PumpStatus } from '@/lib/api'

interface PumpStatusCardProps {
  pumpStatus: PumpStatus | null
  isLoading?: boolean
  className?: string
}

const fadeIn = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
}

// Format timestamp to relative time
function timeAgo(isoStr: string | null | undefined): string {
  if (!isoStr) return '—'
  const diff = (Date.now() - new Date(isoStr).getTime()) / 1000
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`
  return `${Math.round(diff / 86400)}d ago`
}

// Format timestamp to local time string
function formatTime(isoStr: string | null | undefined): string {
  if (!isoStr) return '—'
  return new Date(isoStr).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

function getBatteryIcon(percent: number | null | undefined) {
  if (percent == null) return <Battery className="w-4 h-4 text-gray-500" />
  if (percent <= 15) return <BatteryLow className="w-4 h-4 text-red-500" />
  if (percent <= 40) return <BatteryMedium className="w-4 h-4 text-yellow-500" />
  return <BatteryFull className="w-4 h-4 text-green-500" />
}

function getBatteryColor(percent: number | null | undefined): string {
  if (percent == null) return 'text-gray-500'
  if (percent <= 15) return 'text-red-400'
  if (percent <= 40) return 'text-yellow-400'
  return 'text-green-400'
}

function getModeColor(mode: string | null): string {
  switch (mode) {
    case 'Normal': return 'text-blue-400'
    case 'Sleeping': return 'text-indigo-400'
    case 'Exercising': return 'text-green-400'
    case 'EatingSoon': return 'text-orange-400'
    default: return 'text-gray-400'
  }
}

function getModeLabel(mode: string | null): string {
  switch (mode) {
    case 'EatingSoon': return 'Eating Soon'
    default: return mode || '—'
  }
}

function getControlModeIcon(mode: string | null) {
  switch (mode) {
    case 'ClosedLoop': return <Shield className="w-3.5 h-3.5 text-green-400" />
    case 'OpenLoop': return <ShieldOff className="w-3.5 h-3.5 text-yellow-400" />
    case 'Pining': return <ShieldAlert className="w-3.5 h-3.5 text-orange-400" />
    default: return <ShieldOff className="w-3.5 h-3.5 text-gray-500" />
  }
}

function getControlModeLabel(mode: string | null): string {
  switch (mode) {
    case 'ClosedLoop': return 'Closed Loop'
    case 'OpenLoop': return 'Open Loop'
    case 'Pining': return 'Pining'
    case 'NoControl': return 'No Control'
    default: return mode || '—'
  }
}

function getSiteAgeColor(hours: number | null | undefined): string {
  if (hours == null) return 'text-gray-500'
  if (hours > 72) return 'text-red-400'
  if (hours > 48) return 'text-yellow-400'
  return 'text-green-400'
}

function formatSiteAge(hours: number | null | undefined): string {
  if (hours == null || isNaN(hours)) return '—'
  if (hours < 1) return '<1h'
  if (hours < 24) return `${Math.round(hours)}h`
  const days = Math.floor(hours / 24)
  const remainingHours = Math.round(hours % 24)
  return `${days}d ${remainingHours}h`
}

export function PumpStatusCard({ pumpStatus, isLoading, className }: PumpStatusCardProps) {
  if (isLoading) {
    return (
      <motion.div initial="hidden" animate="visible" variants={fadeIn} className={cn('glass-card', className)}>
        <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan" />
          Pump Status
        </h3>
        <div className="flex items-center justify-center py-6 text-gray-500">
          <RefreshCw className="w-4 h-4 animate-spin mr-2" />
          Loading pump data...
        </div>
      </motion.div>
    )
  }

  if (!pumpStatus) {
    return (
      <motion.div initial="hidden" animate="visible" variants={fadeIn} className={cn('glass-card', className)}>
        <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan" />
          Pump Status
        </h3>
        <p className="text-sm text-gray-500 italic">No pump data available. Tandem sync will populate this on next cycle.</p>
      </motion.div>
    )
  }

  const isSuspended = pumpStatus.is_suspended

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeIn} className={cn('glass-card', className)}>
      <TooltipProvider delayDuration={200}>
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan" />
            Pump Status
          </h3>
          {pumpStatus.last_updated && (
            <span className="text-xs text-gray-500">
              Updated {timeAgo(pumpStatus.last_updated)}
            </span>
          )}
        </div>

        {/* Suspended banner */}
        {isSuspended && (
          <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center gap-2">
            <Power className="w-5 h-5 text-red-400 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-red-400">Pump Suspended</p>
              {pumpStatus.last_suspend_reason && (
                <p className="text-xs text-gray-400">Reason: {pumpStatus.last_suspend_reason}</p>
              )}
            </div>
          </div>
        )}

        {/* Main grid */}
        <div className="grid grid-cols-2 gap-3">
          {/* Battery */}
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-700/50">
                <div className="flex items-center gap-1.5 mb-1">
                  {getBatteryIcon(pumpStatus.battery_percent)}
                  <span className="text-xs text-gray-400">Battery</span>
                </div>
                <div className={cn('text-lg font-bold', getBatteryColor(pumpStatus.battery_percent))}>
                  {pumpStatus.battery_percent != null ? `${Math.round(pumpStatus.battery_percent)}%` : '—'}
                </div>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              {pumpStatus.battery_millivolts
                ? `${pumpStatus.battery_millivolts}mV`
                : 'Battery voltage unavailable'}
            </TooltipContent>
          </Tooltip>

          {/* Pump IOB */}
          <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <Syringe className="w-4 h-4 text-blue-400" />
              <span className="text-xs text-gray-400">Pump IOB</span>
            </div>
            <div className="text-lg font-bold text-blue-400">
              {pumpStatus.pump_iob != null ? `${pumpStatus.pump_iob}u` : '—'}
            </div>
          </div>

          {/* Control-IQ Mode */}
          <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <Zap className="w-4 h-4 text-purple-400" />
              <span className="text-xs text-gray-400">CIQ Mode</span>
            </div>
            <div className={cn('text-sm font-semibold', getModeColor(pumpStatus.current_mode))}>
              {getModeLabel(pumpStatus.current_mode)}
            </div>
          </div>

          {/* Control Mode (Closed/Open Loop) */}
          <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              {getControlModeIcon(pumpStatus.control_mode)}
              <span className="text-xs text-gray-400">Control</span>
            </div>
            <div className={cn('text-sm font-semibold',
              pumpStatus.control_mode === 'ClosedLoop' ? 'text-green-400' :
              pumpStatus.control_mode === 'OpenLoop' ? 'text-yellow-400' : 'text-gray-400'
            )}>
              {getControlModeLabel(pumpStatus.control_mode)}
            </div>
          </div>

          {/* Site Age */}
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-700/50">
                <div className="flex items-center gap-1.5 mb-1">
                  <Droplets className="w-4 h-4 text-teal-400" />
                  <span className="text-xs text-gray-400">Site Age</span>
                </div>
                <div className={cn('text-sm font-semibold', getSiteAgeColor(pumpStatus.site_age_hours))}>
                  {formatSiteAge(pumpStatus.site_age_hours)}
                </div>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              {pumpStatus.last_site_change_at
                ? `Changed ${formatTime(pumpStatus.last_site_change_at)}`
                : 'No site change recorded'}
            </TooltipContent>
          </Tooltip>

          {/* Cartridge */}
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-700/50">
                <div className="flex items-center gap-1.5 mb-1">
                  <Syringe className="w-4 h-4 text-amber-400" />
                  <span className="text-xs text-gray-400">Insulin Left</span>
                </div>
                <div className={cn('text-sm font-semibold',
                  pumpStatus.insulin_remaining != null && pumpStatus.insulin_remaining < 30 ? 'text-red-400' :
                  pumpStatus.insulin_remaining != null && pumpStatus.insulin_remaining < 80 ? 'text-yellow-400' : 'text-amber-400'
                )}>
                  {pumpStatus.insulin_remaining != null
                    ? `${pumpStatus.insulin_remaining}u`
                    : pumpStatus.last_cartridge_volume != null
                    ? `${pumpStatus.last_cartridge_volume}u filled`
                    : '—'}
                </div>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              {pumpStatus.last_cartridge_change_at
                ? `Filled ${pumpStatus.last_cartridge_volume ?? '?'}u ${timeAgo(pumpStatus.last_cartridge_change_at)}`
                : 'No cartridge change recorded'}
            </TooltipContent>
          </Tooltip>
        </div>

        {/* Daily totals */}
        {(pumpStatus.daily_total_insulin != null || pumpStatus.daily_carbs != null) && (
          <div className="mt-3 pt-3 border-t border-slate-700/50">
            <div className="text-xs text-gray-400 mb-2">Today's Totals (Pump)</div>
            <div className="grid grid-cols-5 gap-2 text-center">
              <div>
                <div className="text-sm font-bold text-blue-400">
                  {pumpStatus.daily_basal_units != null ? pumpStatus.daily_basal_units.toFixed(1) : '—'}
                </div>
                <div className="text-[10px] text-gray-500">Basal</div>
              </div>
              <div>
                <div className="text-sm font-bold text-cyan-400">
                  {pumpStatus.daily_bolus_units != null ? pumpStatus.daily_bolus_units.toFixed(1) : '—'}
                </div>
                <div className="text-[10px] text-gray-500">Bolus</div>
              </div>
              <div>
                <div className="text-sm font-bold text-purple-400">
                  {pumpStatus.daily_total_insulin != null ? pumpStatus.daily_total_insulin.toFixed(1) : '—'}
                </div>
                <div className="text-[10px] text-gray-500">Total U</div>
              </div>
              <div>
                <div className="text-sm font-bold text-amber-400">
                  {pumpStatus.daily_carbs != null ? Math.round(pumpStatus.daily_carbs) : '—'}
                </div>
                <div className="text-[10px] text-gray-500">Carbs</div>
              </div>
              <div>
                <div className="text-sm font-bold text-teal-400">
                  {pumpStatus.daily_auto_corrections ?? '—'}
                </div>
                <div className="text-[10px] text-gray-500">Auto Corr</div>
              </div>
            </div>
          </div>
        )}

        {/* Recent alerts */}
        {pumpStatus.recent_alerts && pumpStatus.recent_alerts.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-700/50">
            <div className="flex items-center gap-1.5 mb-2">
              <AlertTriangle className="w-3.5 h-3.5 text-yellow-500" />
              <span className="text-xs text-gray-400">Recent Alerts</span>
            </div>
            <div className="space-y-1">
              {pumpStatus.recent_alerts.slice(-3).reverse().map((alert, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-yellow-500/30 text-yellow-400">
                    Alert {alert.alert}
                  </Badge>
                  <span className="text-gray-500">{timeAgo(alert.at)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recent alarms */}
        {pumpStatus.recent_alarms && pumpStatus.recent_alarms.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-700/50">
            <div className="flex items-center gap-1.5 mb-2">
              <AlertCircle className="w-3.5 h-3.5 text-red-500" />
              <span className="text-xs text-gray-400">Recent Alarms</span>
            </div>
            <div className="space-y-1">
              {pumpStatus.recent_alarms.slice(-3).reverse().map((alarm, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-red-500/30 text-red-400">
                    Alarm {alarm.alarm}
                  </Badge>
                  <span className="text-gray-500">{timeAgo(alarm.at)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recent mode changes */}
        {pumpStatus.recent_mode_changes && pumpStatus.recent_mode_changes.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-700/50">
            <div className="flex items-center gap-1.5 mb-2">
              <Zap className="w-3.5 h-3.5 text-purple-400" />
              <span className="text-xs text-gray-400">Mode Changes</span>
            </div>
            <div className="space-y-1">
              {pumpStatus.recent_mode_changes.slice(-3).reverse().map((mc, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">
                    <span className={getModeColor(mc.from)}>{getModeLabel(mc.from)}</span>
                    {' → '}
                    <span className={getModeColor(mc.to)}>{getModeLabel(mc.to)}</span>
                  </span>
                  <span className="text-gray-500">{timeAgo(mc.at)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </TooltipProvider>
    </motion.div>
  )
}

export default PumpStatusCard
