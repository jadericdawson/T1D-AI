/**
 * AI Training Path Explanation
 * Explains how personalized ML models work and what data is needed
 */
import { motion } from 'framer-motion'
import { Brain, Database, TrendingUp, Zap, Clock, CheckCircle, Circle } from 'lucide-react'

interface DataStatus {
  daysOfData: number
  isfReady: boolean
  icrReady: boolean
  pirReady: boolean
  tftReady: boolean
}

interface AITrainingPathProps {
  dataStatus?: DataStatus
}

export default function AITrainingPath({ dataStatus }: AITrainingPathProps) {
  // Default status if not provided
  const status = dataStatus || {
    daysOfData: 0,
    isfReady: false,
    icrReady: false,
    pirReady: false,
    tftReady: false,
  }

  const trainingStages = [
    {
      id: 'data',
      icon: Database,
      title: 'Data Collection',
      description: 'We need at least 7-14 days of glucose and treatment data',
      requirement: '14+ days recommended',
      isReady: status.daysOfData >= 7,
      progress: Math.min(100, (status.daysOfData / 14) * 100),
    },
    {
      id: 'isf',
      icon: TrendingUp,
      title: 'ISF Learning',
      description: 'Learn your insulin sensitivity patterns throughout the day',
      requirement: 'Needs correction boluses',
      isReady: status.isfReady,
      progress: status.isfReady ? 100 : status.daysOfData >= 7 ? 60 : 0,
    },
    {
      id: 'icr',
      icon: Zap,
      title: 'ICR Learning',
      description: 'Learn your insulin-to-carb ratios for different times',
      requirement: 'Needs meal data',
      isReady: status.icrReady,
      progress: status.icrReady ? 100 : status.daysOfData >= 7 ? 50 : 0,
    },
    {
      id: 'pir',
      icon: Brain,
      title: 'PIR Learning',
      description: 'Learn how protein impacts your glucose over time',
      requirement: 'Needs protein tracking',
      isReady: status.pirReady,
      progress: status.pirReady ? 100 : status.daysOfData >= 14 ? 40 : 0,
    },
    {
      id: 'tft',
      icon: Brain,
      title: 'Advanced Predictions',
      description: 'Temporal Fusion Transformer for multi-hour predictions',
      requirement: '30+ days for best results',
      isReady: status.tftReady,
      progress: status.tftReady ? 100 : Math.min(50, (status.daysOfData / 30) * 50),
    },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <span className="text-4xl mb-2 block">🧠</span>
        <h3 className="text-xl font-semibold mb-2">Your AI Training Path</h3>
        <p className="text-gray-400 text-sm">
          T1D-AI learns your unique patterns to provide personalized predictions
        </p>
      </div>

      {/* How it works */}
      <div className="bg-gradient-to-br from-purple-500/10 to-cyan/10 border border-purple-500/20 rounded-lg p-4">
        <h4 className="font-medium mb-2 flex items-center">
          <Brain className="w-5 h-5 mr-2 text-purple-400" />
          How Personalized AI Works
        </h4>
        <p className="text-sm text-gray-300">
          Unlike generic diabetes calculators, T1D-AI builds machine learning models trained
          specifically on <span className="text-cyan">your</span> data. As you log meals, boluses,
          and glucose readings, the AI learns your unique metabolic patterns - how you respond to
          different foods, times of day, and insulin doses.
        </p>
      </div>

      {/* Training stages */}
      <div className="space-y-4">
        <h4 className="font-medium">Training Progress</h4>
        {trainingStages.map((stage, index) => (
          <motion.div
            key={stage.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`
              relative p-4 rounded-lg border
              ${stage.isReady ? 'border-green-500/30 bg-green-500/5' : 'border-gray-700 bg-gray-800/30'}
            `}
          >
            <div className="flex items-start gap-3">
              {/* Status indicator */}
              <div
                className={`
                  flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
                  ${stage.isReady ? 'bg-green-500/20 text-green-400' : 'bg-gray-700 text-gray-400'}
                `}
              >
                {stage.isReady ? (
                  <CheckCircle className="w-5 h-5" />
                ) : (
                  <stage.icon className="w-5 h-5" />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">{stage.title}</span>
                  {stage.isReady ? (
                    <span className="text-xs text-green-400 bg-green-500/20 px-2 py-0.5 rounded">
                      Ready
                    </span>
                  ) : (
                    <span className="text-xs text-gray-500">{stage.requirement}</span>
                  )}
                </div>
                <p className="text-sm text-gray-400 mb-2">{stage.description}</p>

                {/* Progress bar */}
                <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${stage.progress}%` }}
                    transition={{ duration: 0.5, delay: index * 0.1 + 0.2 }}
                    className={`h-full rounded-full ${
                      stage.isReady
                        ? 'bg-green-500'
                        : stage.progress > 0
                        ? 'bg-cyan'
                        : 'bg-gray-600'
                    }`}
                  />
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Timeline estimate */}
      <div className="bg-gray-800/50 rounded-lg p-4">
        <h4 className="font-medium mb-3 flex items-center">
          <Clock className="w-5 h-5 mr-2 text-cyan" />
          What to Expect
        </h4>
        <ul className="space-y-2 text-sm">
          <li className="flex items-start gap-2">
            <Circle className="w-3 h-3 mt-1.5 flex-shrink-0 text-cyan" />
            <span className="text-gray-300">
              <strong className="text-white">Week 1:</strong> Basic predictions start working with
              your configured ISF and ICR values
            </span>
          </li>
          <li className="flex items-start gap-2">
            <Circle className="w-3 h-3 mt-1.5 flex-shrink-0 text-cyan" />
            <span className="text-gray-300">
              <strong className="text-white">Week 2-3:</strong> AI learns your personal patterns and
              predictions become more accurate
            </span>
          </li>
          <li className="flex items-start gap-2">
            <Circle className="w-3 h-3 mt-1.5 flex-shrink-0 text-cyan" />
            <span className="text-gray-300">
              <strong className="text-white">Month 2+:</strong> Advanced multi-hour predictions
              become available
            </span>
          </li>
        </ul>
      </div>

      {/* Current status */}
      {status.daysOfData > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center text-sm"
        >
          <p className="text-gray-400">
            You currently have{' '}
            <span className="text-cyan font-medium">{status.daysOfData} days</span> of data
          </p>
          {status.daysOfData < 14 && (
            <p className="text-gray-500 mt-1">
              Keep logging to improve predictions!
            </p>
          )}
        </motion.div>
      )}
    </div>
  )
}
