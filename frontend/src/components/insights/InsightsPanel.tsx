/**
 * AI Insights Panel Component
 * Displays GPT-generated insights with categories and weekly summaries
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Sparkles, TrendingUp, Lightbulb, AlertTriangle, Trophy,
  ChevronRight, RefreshCw, Loader2, Calendar, Clock
} from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn, formatTime } from '@/lib/utils'
import { insightsApi } from '@/lib/api'

const USER_ID = 'demo_user'

// Category icons and colors
const categoryConfig: Record<string, {
  icon: React.ReactNode
  color: string
  bgColor: string
  label: string
}> = {
  pattern: {
    icon: <TrendingUp className="w-4 h-4" />,
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10 border-blue-500/30',
    label: 'Pattern'
  },
  recommendation: {
    icon: <Lightbulb className="w-4 h-4" />,
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-500/10 border-yellow-500/30',
    label: 'Tip'
  },
  warning: {
    icon: <AlertTriangle className="w-4 h-4" />,
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/10 border-orange-500/30',
    label: 'Warning'
  },
  achievement: {
    icon: <Trophy className="w-4 h-4" />,
    color: 'text-green-400',
    bgColor: 'bg-green-500/10 border-green-500/30',
    label: 'Achievement'
  }
}

const fadeIn = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } }
}

interface InsightCardProps {
  content: string
  category: string
  createdAt?: string
}

function InsightCard({ content, category, createdAt }: InsightCardProps) {
  const config = categoryConfig[category] || categoryConfig.recommendation

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className={cn(
        'p-4 rounded-lg border',
        config.bgColor
      )}
    >
      <div className="flex items-start gap-3">
        <div className={cn('mt-0.5', config.color)}>
          {config.icon}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <Badge variant="outline" className={cn('text-xs', config.color)}>
              {config.label}
            </Badge>
            {createdAt && (
              <span className="text-xs text-gray-500">
                {formatTime(createdAt)}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-300 leading-relaxed">
            {content}
          </p>
        </div>
      </div>
    </motion.div>
  )
}

interface WeeklySummaryCardProps {
  stats: {
    avgBg: number
    timeInRange: number
    totalReadings: number
    lows: number
    highs: number
  }
  summary: {
    summary: string
    highlight: string
    focus: string
    motivation: string
  }
  comparison: {
    avgBgChange?: number
    tirChange?: number
  }
}

function WeeklySummaryCard({ stats, summary, comparison }: WeeklySummaryCardProps) {
  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      className="glass-card space-y-4"
    >
      <div className="flex items-center gap-2 mb-4">
        <Calendar className="w-5 h-5 text-purple-500" />
        <h3 className="text-lg font-semibold text-white">Weekly Summary</h3>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-3 rounded-lg bg-slate-800/50">
          <p className="text-2xl font-bold text-cyan">{stats.timeInRange.toFixed(0)}%</p>
          <p className="text-xs text-gray-500">Time in Range</p>
          {comparison.tirChange !== undefined && (
            <p className={cn(
              'text-xs mt-1',
              comparison.tirChange >= 0 ? 'text-green-400' : 'text-red-400'
            )}>
              {comparison.tirChange >= 0 ? '+' : ''}{comparison.tirChange.toFixed(1)}%
            </p>
          )}
        </div>
        <div className="text-center p-3 rounded-lg bg-slate-800/50">
          <p className="text-2xl font-bold text-white">{stats.avgBg.toFixed(0)}</p>
          <p className="text-xs text-gray-500">Avg mg/dL</p>
          {comparison.avgBgChange !== undefined && (
            <p className={cn(
              'text-xs mt-1',
              comparison.avgBgChange <= 0 ? 'text-green-400' : 'text-yellow-400'
            )}>
              {comparison.avgBgChange >= 0 ? '+' : ''}{comparison.avgBgChange.toFixed(0)}
            </p>
          )}
        </div>
        <div className="text-center p-3 rounded-lg bg-slate-800/50">
          <p className="text-2xl font-bold text-orange-400">{stats.lows}</p>
          <p className="text-xs text-gray-500">Lows</p>
        </div>
        <div className="text-center p-3 rounded-lg bg-slate-800/50">
          <p className="text-2xl font-bold text-yellow-400">{stats.highs}</p>
          <p className="text-xs text-gray-500">Highs</p>
        </div>
      </div>

      {/* AI Summary */}
      <div className="space-y-3 pt-4 border-t border-gray-700">
        <p className="text-gray-300">{summary.summary}</p>

        <div className="flex items-start gap-2">
          <Trophy className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-green-400">{summary.highlight}</p>
        </div>

        <div className="flex items-start gap-2">
          <Lightbulb className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-yellow-400">{summary.focus}</p>
        </div>

        <p className="text-sm text-purple-400 italic pt-2">{summary.motivation}</p>
      </div>
    </motion.div>
  )
}

interface InsightsPanelProps {
  className?: string
  compact?: boolean
}

export function InsightsPanel({ className, compact = false }: InsightsPanelProps) {
  const [activeTab, setActiveTab] = useState<'insights' | 'weekly'>('insights')
  const queryClient = useQueryClient()

  // Fetch insights
  const { data: insightsData, isLoading: loadingInsights } = useQuery({
    queryKey: ['insights', USER_ID],
    queryFn: () => insightsApi.getAll(USER_ID, undefined, 10),
    staleTime: 300000, // 5 minutes
  })

  // Fetch weekly summary
  const { data: weeklyData, isLoading: loadingWeekly } = useQuery({
    queryKey: ['insights', 'weekly', USER_ID],
    queryFn: () => insightsApi.getWeeklySummary(USER_ID),
    staleTime: 3600000, // 1 hour
    retry: false,
  })

  // Generate new insights
  const generateMutation = useMutation({
    mutationFn: () => insightsApi.generate(USER_ID, true),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['insights'] })
    }
  })

  const insights = insightsData?.insights || []
  const isLoading = loadingInsights || loadingWeekly

  if (compact) {
    // Compact version for sidebar/header
    return (
      <div className={cn('glass-card border-l-4 border-purple-500', className)}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-purple-500" />
            AI Insights
          </h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => generateMutation.mutate()}
            disabled={generateMutation.isPending}
            className="h-6 px-2"
          >
            {generateMutation.isPending ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <RefreshCw className="w-3 h-3" />
            )}
          </Button>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="w-5 h-5 animate-spin text-purple-500" />
          </div>
        ) : insights.length > 0 ? (
          <div className="space-y-2">
            {insights.slice(0, 3).map((insight, i) => (
              <p key={i} className="text-sm text-gray-300 leading-relaxed">
                {insight.content}
              </p>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500 italic">
            No insights yet. Keep monitoring to generate personalized tips.
          </p>
        )}
      </div>
    )
  }

  // Full panel version
  return (
    <div className={cn('space-y-6', className)}>
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-purple-500" />
          AI Insights
        </h2>
        <Button
          variant="outline"
          size="sm"
          onClick={() => generateMutation.mutate()}
          disabled={generateMutation.isPending}
          className="border-gray-700"
        >
          {generateMutation.isPending ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4 mr-2" />
          )}
          Refresh
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)}>
        <TabsList className="bg-slate-800/50">
          <TabsTrigger
            value="insights"
            className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400"
          >
            <Sparkles className="w-4 h-4 mr-2" />
            Insights
          </TabsTrigger>
          <TabsTrigger
            value="weekly"
            className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400"
          >
            <Calendar className="w-4 h-4 mr-2" />
            Weekly
          </TabsTrigger>
        </TabsList>

        <TabsContent value="insights" className="mt-4">
          <AnimatePresence mode="wait">
            {isLoading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center justify-center py-12"
              >
                <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
              </motion.div>
            ) : insights.length > 0 ? (
              <motion.div
                key="insights"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-3"
              >
                {insights.map((insight, i) => (
                  <InsightCard
                    key={i}
                    content={insight.content}
                    category={insight.category}
                    createdAt={insight.createdAt}
                  />
                ))}
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-center py-12"
              >
                <Sparkles className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-400 mb-4">
                  No insights generated yet
                </p>
                <Button
                  onClick={() => generateMutation.mutate()}
                  disabled={generateMutation.isPending}
                  className="bg-purple-600 hover:bg-purple-700"
                >
                  Generate Insights
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
        </TabsContent>

        <TabsContent value="weekly" className="mt-4">
          <AnimatePresence mode="wait">
            {loadingWeekly ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center justify-center py-12"
              >
                <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
              </motion.div>
            ) : weeklyData?.stats ? (
              <WeeklySummaryCard
                stats={weeklyData.stats}
                summary={weeklyData.summary}
                comparison={weeklyData.comparison || {}}
              />
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-center py-12"
              >
                <Calendar className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-400">
                  Not enough data for a weekly summary yet.
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Keep monitoring to unlock weekly insights!
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default InsightsPanel
