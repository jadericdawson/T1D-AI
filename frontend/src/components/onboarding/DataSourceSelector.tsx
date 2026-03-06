/**
 * Data Source Selector Component
 * Allows users to select their current data sources and vote for future ones
 */
import { motion } from 'framer-motion'
import { Check, Clock, ExternalLink } from 'lucide-react'

// Data source definitions
interface DataSource {
  id: string
  name: string
  logo: string // emoji or icon identifier
  description: string
  status: 'active' | 'coming_soon'
  website?: string
}

const dataSources: DataSource[] = [
  // Active data sources
  {
    id: 'gluroo',
    name: 'Gluroo',
    logo: '🔗',
    description: 'Sync via Nightscout Global Connect',
    status: 'active',
    website: 'https://gluroo.com',
  },
  // Coming soon data sources
  {
    id: 'dexcom_share',
    name: 'Dexcom Share',
    logo: '📊',
    description: 'Direct Dexcom CGM data sharing',
    status: 'coming_soon',
    website: 'https://dexcom.com',
  },
  {
    id: 'dexcom_clarity',
    name: 'Dexcom Clarity',
    logo: '📈',
    description: 'Historical reports and data export',
    status: 'coming_soon',
    website: 'https://clarity.dexcom.com',
  },
  {
    id: 'tandem_tconnect',
    name: 'Tandem Source',
    logo: '💉',
    description: 'Tandem Mobi / t:slim X2 pump data',
    status: 'active',
    website: 'https://tandemdiabetes.com',
  },
  {
    id: 'omnipod5',
    name: 'Omnipod 5',
    logo: '🎯',
    description: 'Insulet Omnipod 5 system data',
    status: 'coming_soon',
    website: 'https://omnipod.com',
  },
  {
    id: 'medtronic_carelink',
    name: 'Medtronic CareLink',
    logo: '🏥',
    description: 'Medtronic pump and CGM data',
    status: 'coming_soon',
    website: 'https://carelink.medtronic.com',
  },
  {
    id: 'libre',
    name: 'FreeStyle Libre',
    logo: '📱',
    description: 'Abbott FreeStyle Libre CGM',
    status: 'coming_soon',
    website: 'https://freestylelibre.us',
  },
  {
    id: 'nightscout',
    name: 'Nightscout',
    logo: '🌙',
    description: 'Self-hosted CGM in the cloud',
    status: 'coming_soon',
    website: 'https://nightscout.info',
  },
  {
    id: 'tidepool',
    name: 'Tidepool',
    logo: '🌊',
    description: 'Open platform diabetes data',
    status: 'coming_soon',
    website: 'https://tidepool.org',
  },
  {
    id: 'apple_health',
    name: 'Apple Health',
    logo: '🍎',
    description: 'iOS Health app integration',
    status: 'coming_soon',
  },
  {
    id: 'google_fit',
    name: 'Google Fit',
    logo: '❤️',
    description: 'Android health data platform',
    status: 'coming_soon',
    website: 'https://fit.google.com',
  },
]

interface DataSourceSelectorProps {
  currentSources: string[]
  desiredSources: string[]
  onCurrentChange: (sources: string[]) => void
  onDesiredChange: (sources: string[]) => void
}

export default function DataSourceSelector({
  currentSources,
  desiredSources,
  onCurrentChange,
  onDesiredChange,
}: DataSourceSelectorProps) {
  const activeSources = dataSources.filter((s) => s.status === 'active')
  const comingSoonSources = dataSources.filter((s) => s.status === 'coming_soon')

  const toggleCurrent = (id: string) => {
    if (currentSources.includes(id)) {
      onCurrentChange(currentSources.filter((s) => s !== id))
    } else {
      onCurrentChange([...currentSources, id])
    }
  }

  const toggleDesired = (id: string) => {
    if (desiredSources.includes(id)) {
      onDesiredChange(desiredSources.filter((s) => s !== id))
    } else {
      onDesiredChange([...desiredSources, id])
    }
  }

  return (
    <div className="space-y-8">
      {/* Active Data Sources */}
      <div>
        <h3 className="text-lg font-semibold mb-3 flex items-center">
          <Check className="w-5 h-5 mr-2 text-green-500" />
          Available Now
        </h3>
        <p className="text-sm text-gray-400 mb-4">
          Select the data sources you currently use. We'll help you connect them.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {activeSources.map((source) => (
            <DataSourceCard
              key={source.id}
              source={source}
              selected={currentSources.includes(source.id)}
              onToggle={() => toggleCurrent(source.id)}
            />
          ))}
        </div>
      </div>

      {/* Coming Soon Data Sources */}
      <div>
        <h3 className="text-lg font-semibold mb-3 flex items-center">
          <Clock className="w-5 h-5 mr-2 text-yellow-500" />
          Coming Soon
        </h3>
        <p className="text-sm text-gray-400 mb-4">
          Select the sources you'd like us to support. Your votes help us prioritize development!
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {comingSoonSources.map((source) => (
            <DataSourceCard
              key={source.id}
              source={source}
              selected={desiredSources.includes(source.id)}
              onToggle={() => toggleDesired(source.id)}
              comingSoon
            />
          ))}
        </div>
      </div>

      {/* Vote count */}
      {desiredSources.length > 0 && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-sm text-cyan text-center"
        >
          You've voted for {desiredSources.length} future integration
          {desiredSources.length > 1 ? 's' : ''}
        </motion.p>
      )}
    </div>
  )
}

// Individual data source card component
function DataSourceCard({
  source,
  selected,
  onToggle,
  comingSoon = false,
}: {
  source: DataSource
  selected: boolean
  onToggle: () => void
  comingSoon?: boolean
}) {
  return (
    <motion.button
      type="button"
      onClick={onToggle}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={`
        relative p-4 rounded-lg border text-left transition-all
        ${
          selected
            ? comingSoon
              ? 'border-yellow-500/50 bg-yellow-500/10'
              : 'border-cyan/50 bg-cyan/10'
            : 'border-gray-700 bg-gray-800/30 hover:border-gray-600'
        }
      `}
    >
      {/* Selection indicator */}
      {selected && (
        <div
          className={`absolute top-2 right-2 w-5 h-5 rounded-full flex items-center justify-center ${
            comingSoon ? 'bg-yellow-500' : 'bg-cyan'
          }`}
        >
          <Check className="w-3 h-3 text-black" />
        </div>
      )}

      <div className="flex items-start gap-3">
        {/* Logo */}
        <span className="text-2xl">{source.logo}</span>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium">{source.name}</span>
            {comingSoon && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                Vote
              </span>
            )}
          </div>
          <p className="text-xs text-gray-400 mt-0.5">{source.description}</p>
          {source.website && (
            <a
              href={source.website}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center text-xs text-cyan hover:underline mt-1"
            >
              Learn more <ExternalLink className="w-3 h-3 ml-1" />
            </a>
          )}
        </div>
      </div>
    </motion.button>
  )
}
