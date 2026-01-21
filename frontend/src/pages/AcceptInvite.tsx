/**
 * Accept Invite Page
 * Allows users to accept sharing invitations from email links
 */
import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Check, X, Loader2, UserPlus, Eye, Heart, Shield,
  LogIn, UserCircle, AlertCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useAuthStore } from '@/stores/authStore'
import { sharingApi, ShareInvitation, AcceptInviteResponse } from '@/lib/api'

const ROLE_CONFIG: Record<string, { icon: typeof Eye; color: string; bgColor: string; label: string; description: string }> = {
  viewer: {
    icon: Eye,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    label: 'Viewer',
    description: 'View glucose readings, treatments, predictions, and insights (read-only)'
  },
  caregiver: {
    icon: Heart,
    color: 'text-pink-500',
    bgColor: 'bg-pink-500/10',
    label: 'Caregiver',
    description: 'View all data plus log treatments and receive alerts'
  },
  admin: {
    icon: Shield,
    color: 'text-purple-500',
    bgColor: 'bg-purple-500/10',
    label: 'Admin',
    description: 'Full access including settings and all permissions'
  },
}

export default function AcceptInvite() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token')

  const { isAuthenticated, user, loadProfiles, switchToUser } = useAuthStore()

  const [status, setStatus] = useState<'loading' | 'ready' | 'accepting' | 'success' | 'error'>('loading')
  const [invitation, setInvitation] = useState<ShareInvitation | null>(null)
  const [errorMessage, setErrorMessage] = useState('')

  // Fetch invitation details when component mounts
  useEffect(() => {
    const fetchInvitation = async () => {
      if (!token) {
        setStatus('error')
        setErrorMessage('Invalid invitation link - no token provided')
        return
      }

      try {
        const inv = await sharingApi.getInvitationByToken(token)
        setInvitation(inv)
        setStatus('ready')
      } catch (error: unknown) {
        setStatus('error')
        const err = error as { response?: { status?: number; data?: { detail?: string } } }
        if (err.response?.status === 404) {
          setErrorMessage('This invitation was not found or has already been used')
        } else if (err.response?.status === 410) {
          setErrorMessage('This invitation has expired')
        } else {
          setErrorMessage(err.response?.data?.detail || 'Failed to load invitation')
        }
      }
    }

    fetchInvitation()
  }, [token])

  const handleAccept = async () => {
    if (!token || !invitation) return

    setStatus('accepting')
    try {
      const result: AcceptInviteResponse = await sharingApi.acceptInvitation(token)
      setStatus('success')

      // Reload profiles to include the new share
      await loadProfiles()

      // Switch to viewing the shared user's data immediately
      switchToUser(result.ownerId, {
        id: result.ownerId,
        email: result.ownerEmail,
        displayName: invitation.profileName || invitation.ownerName || invitation.ownerEmail,
        role: invitation.role,
        permissions: [], // Will be populated from the share
      })

      // Redirect to dashboard after short delay
      setTimeout(() => {
        navigate('/dashboard')
      }, 1500)
    } catch (error: unknown) {
      setStatus('error')
      const err = error as { response?: { data?: { detail?: string } } }
      setErrorMessage(err.response?.data?.detail || 'Failed to accept invitation')
    }
  }

  const handleDecline = () => {
    navigate('/')
  }

  // If not authenticated, show login prompt with return URL
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md"
        >
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader className="text-center">
              <div className="mx-auto w-16 h-16 rounded-full bg-cyan/10 flex items-center justify-center mb-4">
                <UserPlus className="w-8 h-8 text-cyan" />
              </div>
              <CardTitle className="text-xl">Sharing Invitation</CardTitle>
              <CardDescription>
                You've been invited to view someone's glucose data on T1D-AI
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-gray-400 text-center">
                Please log in or create an account to accept this invitation.
              </p>
              <div className="flex flex-col gap-3">
                <Link to={`/login?redirect=/accept-invite?token=${token}`}>
                  <Button className="w-full">
                    <LogIn className="w-4 h-4 mr-2" />
                    Log In
                  </Button>
                </Link>
                <Link to={`/register?redirect=/accept-invite?token=${token}`}>
                  <Button variant="outline" className="w-full">
                    <UserCircle className="w-4 h-4 mr-2" />
                    Create Account
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    )
  }

  // Loading state
  if (status === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-cyan" />
      </div>
    )
  }

  // Error state
  if (status === 'error') {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md"
        >
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader className="text-center">
              <div className="mx-auto w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mb-4">
                <AlertCircle className="w-8 h-8 text-red-500" />
              </div>
              <CardTitle className="text-xl">Invitation Error</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-center text-gray-400">{errorMessage}</p>
              <div className="flex gap-3">
                <Button variant="outline" onClick={() => navigate('/')} className="flex-1">
                  Go Home
                </Button>
                <Button onClick={() => navigate('/dashboard')} className="flex-1">
                  Dashboard
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    )
  }

  // Success state
  if (status === 'success') {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-md"
        >
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader className="text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200, damping: 15 }}
                className="mx-auto w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center mb-4"
              >
                <Check className="w-8 h-8 text-green-500" />
              </motion.div>
              <CardTitle className="text-xl">Invitation Accepted!</CardTitle>
              <CardDescription>
                You can now view {invitation?.ownerName || invitation?.ownerEmail}'s glucose data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-400 text-center">
                Redirecting to dashboard...
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    )
  }

  // Ready state - show invitation details
  const roleConfig = invitation ? (ROLE_CONFIG[invitation.role] || ROLE_CONFIG.viewer) : ROLE_CONFIG.viewer
  const RoleIcon = roleConfig.icon

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <Card className="bg-gray-900/50 border-gray-800">
          <CardHeader className="text-center">
            <div className="mx-auto w-16 h-16 rounded-full bg-cyan/10 flex items-center justify-center mb-4">
              <UserPlus className="w-8 h-8 text-cyan" />
            </div>
            <CardTitle className="text-xl">Sharing Invitation</CardTitle>
            <CardDescription>
              You've been invited to view glucose data
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Invitation details */}
            <div className="bg-gray-800/50 rounded-lg p-4 space-y-3">
              <div>
                <p className="text-sm text-gray-400">From</p>
                <p className="font-medium">{invitation?.ownerName || invitation?.ownerEmail}</p>
                {invitation?.ownerName && invitation?.ownerEmail && (
                  <p className="text-sm text-gray-400">{invitation.ownerEmail}</p>
                )}
              </div>
              <div>
                <p className="text-sm text-gray-400">Access Level</p>
                <div className="flex items-center gap-2 mt-1">
                  <div className={`p-1.5 rounded-full ${roleConfig.bgColor}`}>
                    <RoleIcon className={`w-4 h-4 ${roleConfig.color}`} />
                  </div>
                  <span className="font-medium">{roleConfig.label}</span>
                </div>
                <p className="text-xs text-gray-400 mt-1">{roleConfig.description}</p>
              </div>
            </div>

            {/* Accept/Decline buttons */}
            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={handleDecline}
                className="flex-1"
                disabled={status === 'accepting'}
              >
                <X className="w-4 h-4 mr-2" />
                Decline
              </Button>
              <Button
                onClick={handleAccept}
                className="flex-1"
                disabled={status === 'accepting'}
              >
                {status === 'accepting' ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Accepting...
                  </>
                ) : (
                  <>
                    <Check className="w-4 h-4 mr-2" />
                    Accept
                  </>
                )}
              </Button>
            </div>

            {/* Logged in as */}
            <p className="text-xs text-gray-500 text-center">
              Accepting as {user?.email}
            </p>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
