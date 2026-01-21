/**
 * Invite Modal Component
 * Modal for inviting users to view your glucose data
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Mail, UserPlus, Loader2, Check, AlertCircle,
  Eye, Heart, Shield
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { sharingApi } from '@/lib/api'

interface InviteModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess?: () => void
}

const ROLES = [
  {
    id: 'viewer',
    name: 'Viewer',
    icon: Eye,
    description: 'Can view glucose, treatments, predictions, and insights (read-only)',
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30',
  },
  {
    id: 'caregiver',
    name: 'Caregiver',
    icon: Heart,
    description: 'Can view everything plus log treatments and receive alerts',
    color: 'text-pink-500',
    bgColor: 'bg-pink-500/10',
    borderColor: 'border-pink-500/30',
  },
  {
    id: 'admin',
    name: 'Admin',
    icon: Shield,
    description: 'Full access including settings and all permissions',
    color: 'text-purple-500',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30',
  },
]

export function InviteModal({ open, onOpenChange, onSuccess }: InviteModalProps) {
  const [email, setEmail] = useState('')
  const [role, setRole] = useState('viewer')
  const [status, setStatus] = useState<'idle' | 'sending' | 'success' | 'error'>('idle')
  const [errorMessage, setErrorMessage] = useState('')
  const [successMessage, setSuccessMessage] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!email || !email.includes('@')) {
      setStatus('error')
      setErrorMessage('Please enter a valid email address')
      return
    }

    setStatus('sending')
    setErrorMessage('')

    try {
      const response = await sharingApi.invite(email, role)
      setStatus('success')
      setSuccessMessage(response.message)

      // Reset form after delay
      setTimeout(() => {
        setEmail('')
        setRole('viewer')
        setStatus('idle')
        setSuccessMessage('')
        onOpenChange(false)
        onSuccess?.()
      }, 2000)
    } catch (error: unknown) {
      setStatus('error')
      const err = error as { response?: { data?: { detail?: string } } }
      setErrorMessage(
        err.response?.data?.detail || 'Failed to send invitation. Please try again.'
      )
    }
  }

  const handleClose = () => {
    if (status !== 'sending') {
      setEmail('')
      setRole('viewer')
      setStatus('idle')
      setErrorMessage('')
      setSuccessMessage('')
      onOpenChange(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <UserPlus className="w-5 h-5 text-primary" />
            Invite Someone
          </DialogTitle>
          <DialogDescription>
            Share your glucose data with a family member, caregiver, or healthcare provider.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6 mt-4">
          {/* Email Input */}
          <div>
            <Label htmlFor="email">Email Address</Label>
            <div className="relative mt-1">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="person@example.com"
                className="pl-10"
                disabled={status === 'sending' || status === 'success'}
              />
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              They'll receive an email invitation to access your data
            </p>
          </div>

          {/* Role Selection */}
          <div>
            <Label>Access Level</Label>
            <div className="grid gap-2 mt-2">
              {ROLES.map((r) => {
                const Icon = r.icon
                const isSelected = role === r.id
                return (
                  <button
                    key={r.id}
                    type="button"
                    onClick={() => setRole(r.id)}
                    disabled={status === 'sending' || status === 'success'}
                    className={`
                      flex items-start gap-3 p-3 rounded-lg border text-left transition-all
                      ${isSelected ? `${r.borderColor} ${r.bgColor}` : 'border-border hover:border-muted-foreground/50'}
                      ${status === 'sending' || status === 'success' ? 'opacity-60 cursor-not-allowed' : ''}
                    `}
                  >
                    <div className={`p-2 rounded-full ${r.bgColor} ${r.color}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{r.name}</span>
                        {isSelected && (
                          <Check className={`w-4 h-4 ${r.color}`} />
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {r.description}
                      </p>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Status Messages */}
          <AnimatePresence>
            {status === 'error' && errorMessage && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 p-3 rounded-lg"
              >
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{errorMessage}</span>
              </motion.div>
            )}

            {status === 'success' && successMessage && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2 text-sm text-green-600 bg-green-500/10 p-3 rounded-lg"
              >
                <Check className="w-4 h-4 flex-shrink-0" />
                <span>{successMessage}</span>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Actions */}
          <div className="flex justify-end gap-3">
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
              disabled={status === 'sending'}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={!email || status === 'sending' || status === 'success'}
            >
              {status === 'sending' ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Sending...
                </>
              ) : status === 'success' ? (
                <>
                  <Check className="w-4 h-4 mr-2" />
                  Sent!
                </>
              ) : (
                <>
                  <Mail className="w-4 h-4 mr-2" />
                  Send Invitation
                </>
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

export default InviteModal
