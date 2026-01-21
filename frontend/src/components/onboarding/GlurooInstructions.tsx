/**
 * Detailed Gluroo Connection Instructions
 * Step-by-step guide to find Nightscout credentials in Gluroo
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ChevronDown,
  ChevronUp,
  ExternalLink,
  CheckCircle,
  AlertCircle,
  Loader2,
  Copy,
  Check,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface GlurooInstructionsProps {
  nightscoutUrl: string
  apiSecret: string
  onUrlChange: (url: string) => void
  onSecretChange: (secret: string) => void
  onTestConnection: () => Promise<boolean>
  isTestingConnection: boolean
  connectionStatus: 'idle' | 'success' | 'error'
  connectionError?: string
}

export default function GlurooInstructions({
  nightscoutUrl,
  apiSecret,
  onUrlChange,
  onSecretChange,
  onTestConnection,
  isTestingConnection,
  connectionStatus,
  connectionError,
}: GlurooInstructionsProps) {
  const [showFaq, setShowFaq] = useState(false)
  const [copiedField, setCopiedField] = useState<string | null>(null)

  const copyToClipboard = async (text: string, field: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedField(field)
      setTimeout(() => setCopiedField(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <span className="text-4xl mb-2 block">🔗</span>
        <h3 className="text-xl font-semibold mb-2">Connect Gluroo</h3>
        <p className="text-gray-400 text-sm">
          Follow these steps to find your Nightscout credentials in the Gluroo app
        </p>
      </div>

      {/* Step-by-step instructions */}
      <div className="bg-gray-800/50 rounded-lg p-4 space-y-4">
        <h4 className="font-medium text-cyan">Step-by-Step Instructions</h4>

        <ol className="space-y-4">
          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan/20 text-cyan flex items-center justify-center text-sm font-medium">
              1
            </span>
            <div>
              <p className="font-medium">Open the Gluroo app on your phone</p>
              <p className="text-sm text-gray-400">
                Download from{' '}
                <a
                  href="https://apps.apple.com/app/gluroo/id1586395714"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-cyan hover:underline"
                >
                  App Store
                </a>{' '}
                or{' '}
                <a
                  href="https://play.google.com/store/apps/details?id=com.gluroo"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-cyan hover:underline"
                >
                  Google Play
                </a>{' '}
                if you haven't already
              </p>
            </div>
          </li>

          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan/20 text-cyan flex items-center justify-center text-sm font-medium">
              2
            </span>
            <div>
              <p className="font-medium">Tap the menu icon (three lines)</p>
              <p className="text-sm text-gray-400">Located in the top-left corner</p>
            </div>
          </li>

          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan/20 text-cyan flex items-center justify-center text-sm font-medium">
              3
            </span>
            <div>
              <p className="font-medium">Tap "Settings"</p>
            </div>
          </li>

          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan/20 text-cyan flex items-center justify-center text-sm font-medium">
              4
            </span>
            <div>
              <p className="font-medium">Scroll down to "Gluroo Global Connect Nightscout"</p>
              <p className="text-sm text-gray-400">
                This section contains your Nightscout credentials
              </p>
            </div>
          </li>

          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan/20 text-cyan flex items-center justify-center text-sm font-medium">
              5
            </span>
            <div>
              <p className="font-medium">Enable the toggle if it's not already on</p>
              <p className="text-sm text-gray-400">The credentials will appear below</p>
            </div>
          </li>

          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan/20 text-cyan flex items-center justify-center text-sm font-medium">
              6
            </span>
            <div>
              <p className="font-medium">Copy your Nightscout URL</p>
              <p className="text-sm text-gray-400">
                Looks like: <code className="text-cyan">https://xxxxx.ns.gluroo.com</code>
              </p>
            </div>
          </li>

          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan/20 text-cyan flex items-center justify-center text-sm font-medium">
              7
            </span>
            <div>
              <p className="font-medium">Copy your API Secret</p>
              <p className="text-sm text-gray-400">A long string of letters and numbers</p>
            </div>
          </li>
        </ol>
      </div>

      {/* Input fields */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-1">Nightscout URL</label>
          <div className="relative">
            <Input
              type="url"
              value={nightscoutUrl}
              onChange={(e) => onUrlChange(e.target.value)}
              placeholder="https://xxxxx.ns.gluroo.com"
              className="bg-gray-800/50 border-gray-700 pr-10"
            />
            {nightscoutUrl && (
              <button
                type="button"
                onClick={() => copyToClipboard(nightscoutUrl, 'url')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
              >
                {copiedField === 'url' ? (
                  <Check className="w-4 h-4 text-green-500" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </button>
            )}
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">API Secret</label>
          <div className="relative">
            <Input
              type="password"
              value={apiSecret}
              onChange={(e) => onSecretChange(e.target.value)}
              placeholder="Your Nightscout API secret"
              className="bg-gray-800/50 border-gray-700 pr-10"
            />
            {apiSecret && (
              <button
                type="button"
                onClick={() => copyToClipboard(apiSecret, 'secret')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
              >
                {copiedField === 'secret' ? (
                  <Check className="w-4 h-4 text-green-500" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </button>
            )}
          </div>
        </div>

        {/* Test Connection Button */}
        <Button
          type="button"
          onClick={onTestConnection}
          disabled={!nightscoutUrl || !apiSecret || isTestingConnection}
          className="w-full btn-primary"
        >
          {isTestingConnection ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Testing Connection...
            </>
          ) : (
            'Test Connection'
          )}
        </Button>

        {/* Connection Status */}
        <AnimatePresence>
          {connectionStatus === 'success' && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-2 text-green-400 bg-green-500/10 p-3 rounded-lg"
            >
              <CheckCircle className="w-5 h-5" />
              <span>Connection successful! Your CGM data is accessible.</span>
            </motion.div>
          )}

          {connectionStatus === 'error' && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-start gap-2 text-red-400 bg-red-500/10 p-3 rounded-lg"
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Connection failed</p>
                <p className="text-sm">{connectionError || 'Please check your credentials and try again.'}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* FAQ Section */}
      <div className="border-t border-gray-700 pt-4">
        <button
          type="button"
          onClick={() => setShowFaq(!showFaq)}
          className="w-full flex items-center justify-between text-gray-400 hover:text-white"
        >
          <span className="font-medium">Having trouble?</span>
          {showFaq ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
        </button>

        <AnimatePresence>
          {showFaq && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 space-y-4"
            >
              <div>
                <p className="font-medium text-sm">I don't see "Gluroo Global Connect Nightscout"</p>
                <p className="text-sm text-gray-400">
                  Make sure you have the latest version of Gluroo. This feature may require a premium
                  subscription.
                </p>
              </div>

              <div>
                <p className="font-medium text-sm">My connection test keeps failing</p>
                <p className="text-sm text-gray-400">
                  Double-check that you copied the full URL including "https://". The API secret is
                  case-sensitive.
                </p>
              </div>

              <div>
                <p className="font-medium text-sm">I use a different CGM/pump system</p>
                <p className="text-sm text-gray-400">
                  We're actively working on support for Dexcom, Tandem, Omnipod, and more. Check the
                  data sources step to vote for your preferred integration!
                </p>
              </div>

              <a
                href="https://gluroo.com/support"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-sm text-cyan hover:underline"
              >
                Visit Gluroo Support <ExternalLink className="w-4 h-4 ml-1" />
              </a>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
