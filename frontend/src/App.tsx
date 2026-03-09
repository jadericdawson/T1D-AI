import { useEffect } from 'react'
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Onboarding from './pages/Onboarding'
import Settings from './pages/Settings'
import MLModels from './pages/MLModels'
import DataViewer from './pages/DataViewer'
import Admin from './pages/Admin'
import Reports from './pages/Reports'
import Login from './pages/Login'
import Register from './pages/Register'
import VerifyEmail from './pages/VerifyEmail'
import AcceptInvite from './pages/AcceptInvite'
import { useAuthStore } from './stores/authStore'
import { Toaster } from './components/ui/toaster'
import { ErrorBoundary } from './components/ErrorBoundary'

// Protected route component with onboarding guard
function ProtectedRoute({ children, requireOnboarding = true }: { children: React.ReactNode; requireOnboarding?: boolean }) {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated)
  const user = useAuthStore((state) => state.user)
  const location = useLocation()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  // If email not verified, redirect to verify-email page
  if (user && user.emailVerified === false) {
    return <Navigate to="/verify-email" replace />
  }

  // If onboarding is required and not completed, redirect to onboarding
  // Skip this check if we're already on the onboarding page
  if (requireOnboarding && user && user.onboardingCompleted === false && location.pathname !== '/onboarding') {
    return <Navigate to="/onboarding" replace />
  }

  return (
    <ErrorBoundary>
      {children}
    </ErrorBoundary>
  )
}

// Auth event listener hook
function useAuthEventListener() {
  const navigate = useNavigate()
  const logout = useAuthStore((state) => state.logout)

  useEffect(() => {
    const handleUnauthorized = () => {
      console.log('[App] Handling unauthorized event - logging out')
      logout()
      navigate('/login', { replace: true })
    }

    window.addEventListener('auth:unauthorized', handleUnauthorized)
    return () => {
      window.removeEventListener('auth:unauthorized', handleUnauthorized)
    }
  }, [navigate, logout])
}

function App() {
  // Listen for auth events
  useAuthEventListener()

  return (
    <>
      {/* Animated star background */}
      <div className="background-container">
        <div className="stars">
          <div className="twinkling"></div>
        </div>
      </div>

      {/* Main app */}
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/verify-email" element={<VerifyEmail />} />
        <Route path="/accept-invite" element={<AcceptInvite />} />
        <Route path="/dashboard" element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        } />
        <Route path="/onboarding" element={
          <ProtectedRoute requireOnboarding={false}>
            <Onboarding />
          </ProtectedRoute>
        } />
        <Route path="/settings" element={
          <ProtectedRoute>
            <Settings />
          </ProtectedRoute>
        } />
        <Route path="/ml-models" element={
          <ProtectedRoute>
            <MLModels />
          </ProtectedRoute>
        } />
        <Route path="/data" element={
          <ProtectedRoute>
            <DataViewer />
          </ProtectedRoute>
        } />
        <Route path="/reports" element={
          <ProtectedRoute>
            <Reports />
          </ProtectedRoute>
        } />
        <Route path="/admin" element={
          <ProtectedRoute>
            <Admin />
          </ProtectedRoute>
        } />
      </Routes>

      {/* Toast notifications */}
      <Toaster />
    </>
  )
}

export default App
