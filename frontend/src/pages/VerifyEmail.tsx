/**
 * Email Verification Page
 * Handles both:
 * 1. Pending verification (check your email)
 * 2. Verification confirmation (when user clicks link)
 */
import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Mail, CheckCircle, XCircle, Loader2, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useAuthStore } from '@/stores/authStore'

const API_URL = import.meta.env.VITE_API_URL || ''

export default function VerifyEmail() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token')
  const { user, isAuthenticated, logout } = useAuthStore()

  // Get the setter to update emailVerified in local state
  const setEmailVerified = (verified: boolean) => {
    const currentUser = useAuthStore.getState().user
    if (currentUser) {
      useAuthStore.setState({
        user: { ...currentUser, emailVerified: verified }
      })
    }
  }

  const [status, setStatus] = useState<'pending' | 'verifying' | 'success' | 'error'>('pending')
  const [message, setMessage] = useState('')
  const [resendLoading, setResendLoading] = useState(false)
  const [resendMessage, setResendMessage] = useState('')

  // If we have a token in URL, verify it
  useEffect(() => {
    if (token) {
      verifyEmail(token)
    }
  }, [token])

  const verifyEmail = async (verificationToken: string) => {
    setStatus('verifying')
    try {
      const response = await fetch(`${API_URL}/api/auth/verify-email`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: verificationToken }),
      })

      const data = await response.json()

      if (response.ok) {
        setStatus('success')
        setMessage(data.email || 'Your email has been verified!')

        // Update local state if user is logged in
        if (isAuthenticated) {
          setEmailVerified(true)
          // Redirect to onboarding after a moment
          setTimeout(() => {
            navigate('/onboarding')
          }, 2000)
        } else {
          // User is not logged in - redirect to login
          setTimeout(() => {
            navigate('/login?verified=true')
          }, 2000)
        }
      } else {
        setStatus('error')
        setMessage(data.detail || 'Verification failed. The link may have expired.')
      }
    } catch (error) {
      setStatus('error')
      setMessage('Failed to verify email. Please try again.')
    }
  }

  const resendVerification = async () => {
    if (!user?.email) {
      setResendMessage('Please log in to resend verification email')
      return
    }

    setResendLoading(true)
    setResendMessage('')

    try {
      const response = await fetch(`${API_URL}/api/auth/resend-verification`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: user.email }),
      })

      const data = await response.json()
      setResendMessage(data.message || 'Verification email sent!')
    } catch (error) {
      setResendMessage('Failed to send email. Please try again.')
    } finally {
      setResendLoading(false)
    }
  }

  // If verifying with token
  if (token) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md"
        >
          <div className="glass-card p-8 text-center">
            {status === 'verifying' && (
              <>
                <Loader2 className="w-16 h-16 mx-auto text-cyan animate-spin mb-4" />
                <h2 className="text-2xl font-bold mb-2">Verifying Email...</h2>
                <p className="text-gray-400">Please wait while we verify your email address.</p>
              </>
            )}

            {status === 'success' && (
              <>
                <CheckCircle className="w-16 h-16 mx-auto text-green-500 mb-4" />
                <h2 className="text-2xl font-bold mb-2">Email Verified!</h2>
                <p className="text-gray-400 mb-4">
                  Your email has been verified successfully.
                </p>
                <p className="text-sm text-cyan">
                  {isAuthenticated ? 'Redirecting to setup...' : 'Redirecting to login...'}
                </p>
              </>
            )}

            {status === 'error' && (
              <>
                <XCircle className="w-16 h-16 mx-auto text-red-500 mb-4" />
                <h2 className="text-2xl font-bold mb-2">Verification Failed</h2>
                <p className="text-gray-400 mb-6">{message}</p>
                <div className="space-y-3">
                  <Button onClick={() => navigate('/login')} className="w-full btn-primary">
                    Go to Login
                  </Button>
                  {isAuthenticated && (
                    <Button
                      onClick={resendVerification}
                      variant="outline"
                      className="w-full"
                      disabled={resendLoading}
                    >
                      {resendLoading ? (
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      ) : (
                        <RefreshCw className="w-4 h-4 mr-2" />
                      )}
                      Resend Verification Email
                    </Button>
                  )}
                </div>
              </>
            )}
          </div>
        </motion.div>
      </div>
    )
  }

  // Pending verification (no token in URL)
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="glass-card p-8 text-center">
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-cyan/20 to-purple-500/20 flex items-center justify-center">
            <Mail className="w-10 h-10 text-cyan" />
          </div>

          <h1 className="text-2xl font-bold mb-2">Check Your Email</h1>

          {user?.email ? (
            <p className="text-gray-400 mb-6">
              We've sent a verification link to{' '}
              <span className="text-white font-medium">{user.email}</span>
            </p>
          ) : (
            <p className="text-gray-400 mb-6">
              We've sent a verification link to your email address.
            </p>
          )}

          <div className="bg-gray-800/50 rounded-lg p-4 mb-6 text-left">
            <p className="text-sm text-gray-300 mb-2">
              <strong>Next steps:</strong>
            </p>
            <ol className="text-sm text-gray-400 list-decimal list-inside space-y-1">
              <li>Check your inbox <span className="text-yellow-400">(and spam/junk folder!)</span></li>
              <li>Click the verification link</li>
              <li>Complete your account setup</li>
            </ol>
            <p className="text-xs text-gray-500 mt-3">
              The email comes from T1D-AI. If it's in spam, please mark it as "not spam" to receive future emails.
            </p>
          </div>

          {resendMessage && (
            <p className={`text-sm mb-4 ${resendMessage.includes('sent') ? 'text-green-400' : 'text-red-400'}`}>
              {resendMessage}
            </p>
          )}

          <div className="space-y-3">
            <Button
              onClick={resendVerification}
              variant="outline"
              className="w-full"
              disabled={resendLoading}
            >
              {resendLoading ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4 mr-2" />
              )}
              Resend Verification Email
            </Button>

            <p className="text-sm text-gray-500">
              Wrong email?{' '}
              <button
                onClick={() => {
                  logout()
                  navigate('/register')
                }}
                className="text-cyan hover:underline"
              >
                Register again
              </button>
            </p>
          </div>

          <div className="mt-6 pt-6 border-t border-gray-700">
            <Link to="/" className="text-sm text-gray-400 hover:text-white">
              Back to home
            </Link>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
