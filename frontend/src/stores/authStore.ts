/**
 * Authentication store using Zustand
 * Manages user authentication state and tokens
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const API_URL = import.meta.env.VITE_API_URL || ''

// ==================== JWT Utilities ====================

/**
 * Parse a JWT token and extract the expiry timestamp.
 * Returns null if token is invalid or unparseable.
 */
export function parseJwtExpiry(token: string): number | null {
  try {
    // JWT structure: header.payload.signature
    const parts = token.split('.')
    if (parts.length !== 3) return null

    // Decode the payload (base64url)
    const payload = parts[1]
    const decoded = atob(payload.replace(/-/g, '+').replace(/_/g, '/'))
    const data = JSON.parse(decoded)

    // Return exp claim (Unix timestamp in seconds)
    return data.exp || null
  } catch {
    return null
  }
}

/**
 * Check if a JWT token is expired or will expire within the buffer period.
 * @param token The JWT access token
 * @param bufferSeconds Seconds before actual expiry to consider expired (default: 60)
 * @returns true if token is expired or will expire within buffer
 */
export function isTokenExpired(token: string, bufferSeconds: number = 60): boolean {
  const exp = parseJwtExpiry(token)
  if (!exp) return true // Invalid token = expired

  const now = Math.floor(Date.now() / 1000)
  return exp - bufferSeconds <= now
}

/**
 * Get seconds until token expiry.
 * Returns 0 if token is invalid or already expired.
 */
export function getTokenTimeRemaining(token: string): number {
  const exp = parseJwtExpiry(token)
  if (!exp) return 0

  const now = Math.floor(Date.now() / 1000)
  return Math.max(0, exp - now)
}

export interface AuthUser {
  id: string
  email: string
  displayName: string | null
  emailVerified?: boolean
  onboardingCompleted?: boolean
  isAdmin?: boolean
}

export interface SharedUser {
  id: string
  email: string
  displayName: string | null
  role: string
  permissions: string[]
}

// ==================== Profile Types ====================

export type ProfileRelationship = 'self' | 'child' | 'spouse' | 'parent' | 'other'
export type DiabetesType = 'T1D' | 'T2D' | 'LADA' | 'gestational' | 'other'
export type SyncStatus = 'ok' | 'error' | 'pending' | 'disabled'

export interface ProfileSummary {
  id: string
  displayName: string
  relationship: ProfileRelationship
  avatarUrl: string | null
  diabetesType: DiabetesType
  isActive: boolean
  lastDataAt: string | null
  dataSourceCount: number
  syncStatus: SyncStatus
}

interface AuthTokens {
  accessToken: string
  refreshToken: string
  expiresIn: number
}

interface AuthState {
  user: AuthUser | null
  tokens: AuthTokens | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null

  // Viewing another user's data (for shared access)
  viewingUserId: string | null
  viewingUser: SharedUser | null
  sharedWithMe: SharedUser[]

  // Managed profiles (for multi-person support)
  managedProfiles: ProfileSummary[]
  activeProfileId: string | null
  isLoadingProfiles: boolean

  // Actions
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, displayName?: string) => Promise<void>
  logout: () => void
  refreshToken: () => Promise<void>
  clearError: () => void
  loginWithMicrosoft: () => void
  setAuthFromCallback: (data: { access_token: string; refresh_token: string; user: AuthUser; expires_in?: number }) => void

  // Sharing actions
  switchToUser: (userId: string, userInfo: SharedUser) => void
  switchToSelf: () => void
  setSharedWithMe: (users: SharedUser[]) => void
  getEffectiveUserId: () => string

  // Profile actions
  loadProfiles: () => Promise<void>
  switchToProfile: (profileId: string) => void
  getActiveProfile: () => ProfileSummary | null
  getEffectiveProfileId: () => string

  // User state updates
  setOnboardingCompleted: (completed: boolean) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      tokens: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      // Sharing state
      viewingUserId: null,
      viewingUser: null,
      sharedWithMe: [],

      // Managed profiles state
      managedProfiles: [],
      activeProfileId: null,
      isLoadingProfiles: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null })
        try {
          const response = await fetch(`${API_URL}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
          })

          if (!response.ok) {
            const data = await response.json()
            throw new Error(data.detail || 'Login failed')
          }

          const data = await response.json()
          // CRITICAL: Clear any previous user's profile state to prevent data leakage
          // This ensures a new login doesn't inherit activeProfileId from previous session
          set({
            user: data.user,
            tokens: {
              accessToken: data.tokens.access_token,
              refreshToken: data.tokens.refresh_token,
              expiresIn: data.tokens.expires_in,
            },
            isAuthenticated: true,
            isLoading: false,
            // Clear previous user's profile/sharing state
            viewingUserId: null,
            viewingUser: null,
            sharedWithMe: [],
            managedProfiles: [],
            activeProfileId: null,
          })
        } catch (error) {
          set({
            isLoading: false,
            error: error instanceof Error ? error.message : 'Login failed',
          })
          throw error
        }
      },

      register: async (email: string, password: string, displayName?: string) => {
        set({ isLoading: true, error: null })
        try {
          const response = await fetch(`${API_URL}/api/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, displayName }),
          })

          if (!response.ok) {
            const data = await response.json()
            throw new Error(data.detail || 'Registration failed')
          }

          const data = await response.json()
          // CRITICAL: Clear any previous user's profile state to prevent data leakage
          set({
            user: data.user,
            tokens: {
              accessToken: data.tokens.access_token,
              refreshToken: data.tokens.refresh_token,
              expiresIn: data.tokens.expires_in,
            },
            isAuthenticated: true,
            isLoading: false,
            // Clear previous user's profile/sharing state
            viewingUserId: null,
            viewingUser: null,
            sharedWithMe: [],
            managedProfiles: [],
            activeProfileId: null,
          })
        } catch (error) {
          set({
            isLoading: false,
            error: error instanceof Error ? error.message : 'Registration failed',
          })
          throw error
        }
      },

      logout: () => {
        // Note: stopTokenRefreshTimer() is called from clearAllAuthState()
        // or will be stopped when a new login starts a new timer
        set({
          user: null,
          tokens: null,
          isAuthenticated: false,
          error: null,
          viewingUserId: null,
          viewingUser: null,
          // Clear managed profiles
          managedProfiles: [],
          activeProfileId: null,
        })
      },

      refreshToken: async () => {
        const { tokens } = get()
        if (!tokens?.refreshToken) {
          throw new Error('No refresh token')
        }

        try {
          const response = await fetch(`${API_URL}/api/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: tokens.refreshToken }),
          })

          if (!response.ok) {
            throw new Error('Token refresh failed')
          }

          const data = await response.json()
          set({
            tokens: {
              accessToken: data.access_token,
              refreshToken: data.refresh_token,
              expiresIn: data.expires_in,
            },
          })
        } catch {
          // If refresh fails, log out
          get().logout()
          throw new Error('Session expired. Please log in again.')
        }
      },

      clearError: () => set({ error: null }),

      loginWithMicrosoft: async () => {
        set({ isLoading: true, error: null })

        try {
          // Fetch the login URL from the API
          const response = await fetch(`${API_URL}/api/auth/microsoft/login-url`)
          if (!response.ok) {
            const data = await response.json()
            throw new Error(data.detail || 'Microsoft login not available')
          }

          const { login_url } = await response.json()

          // Redirect to Microsoft login (popup blocked due to async fetch)
          // The callback will set localStorage and redirect back
          window.location.href = login_url
        } catch (error) {
          set({
            isLoading: false,
            error: error instanceof Error ? error.message : 'Microsoft login failed',
          })
        }
      },

      setAuthFromCallback: (data: {
        access_token: string
        refresh_token: string
        user: AuthUser
        expires_in?: number
      }) => {
        // CRITICAL: Clear any previous user's profile state to prevent data leakage
        // This ensures OAuth login doesn't inherit activeProfileId from previous session
        set({
          user: data.user,
          tokens: {
            accessToken: data.access_token,
            refreshToken: data.refresh_token,
            expiresIn: data.expires_in || 86400, // Use provided or 24 hours default
          },
          isAuthenticated: true,
          isLoading: false,
          // Clear previous user's profile/sharing state
          viewingUserId: null,
          viewingUser: null,
          sharedWithMe: [],
          managedProfiles: [],
          activeProfileId: null,
        })
      },

      // Sharing actions
      switchToUser: (userId: string, userInfo: SharedUser) => {
        set({
          viewingUserId: userId,
          viewingUser: userInfo,
        })
      },

      switchToSelf: () => {
        set({
          viewingUserId: null,
          viewingUser: null,
        })
      },

      setSharedWithMe: (users: SharedUser[]) => {
        set({ sharedWithMe: users })
      },

      getEffectiveUserId: () => {
        const state = get()
        return state.viewingUserId || state.user?.id || ''
      },

      // Profile actions
      loadProfiles: async () => {
        const { tokens, activeProfileId: storedProfileId } = get()
        if (!tokens?.accessToken) {
          return
        }

        set({ isLoadingProfiles: true })

        try {
          // Fetch both managed profiles and shared-with-me in parallel
          const [profilesResponse, sharedResponse] = await Promise.all([
            fetch(`${API_URL}/api/v1/profiles/summaries`, {
              headers: { 'Authorization': `Bearer ${tokens.accessToken}` },
            }),
            fetch(`${API_URL}/api/v1/sharing/shared-with-me`, {
              headers: { 'Authorization': `Bearer ${tokens.accessToken}` },
            }),
          ])

          // Handle profiles response
          let profiles: ProfileSummary[] = []
          let newActiveProfileId = storedProfileId

          if (profilesResponse.ok) {
            const data = await profilesResponse.json()
            profiles = data.profiles || []

            // Preserve the stored activeProfileId if it's still valid (profile exists)
            // Otherwise fall back to server suggestion or first profile
            if (!storedProfileId || !profiles.find((p: { id: string }) => p.id === storedProfileId)) {
              newActiveProfileId = data.activeProfileId || (profiles[0]?.id ?? null)
            }
          } else if (profilesResponse.status !== 404) {
            console.error('[Auth] Failed to load profiles:', profilesResponse.status)
          }

          // Handle shared-with-me response
          let sharedUsers: SharedUser[] = []
          if (sharedResponse.ok) {
            const sharedData = await sharedResponse.json()
            // Transform the response to SharedUser format
            sharedUsers = (sharedData || []).map((share: {
              ownerId: string
              ownerEmail: string
              ownerName: string | null
              profileId?: string | null
              profileName?: string | null
              role: string
              permissions: string[]
            }) => ({
              id: share.profileId || share.ownerId,  // Use profileId for viewing specific profile
              email: share.ownerEmail,
              displayName: share.profileName || share.ownerName || share.ownerEmail,
              role: share.role,
              permissions: share.permissions,
            }))
            console.log(`[Auth] Loaded ${sharedUsers.length} shared profiles`)
          } else if (sharedResponse.status !== 404) {
            console.error('[Auth] Failed to load shared profiles:', sharedResponse.status)
          }

          set({
            managedProfiles: profiles,
            activeProfileId: newActiveProfileId,
            sharedWithMe: sharedUsers,
            isLoadingProfiles: false,
          })
        } catch (error) {
          console.error('[Auth] Failed to load profiles:', error)
          set({ isLoadingProfiles: false })
        }
      },

      switchToProfile: (profileId: string) => {
        const { managedProfiles, activeProfileId: currentProfileId } = get()
        const profile = managedProfiles.find(p => p.id === profileId)

        if (profile && profileId !== currentProfileId) {
          set({
            activeProfileId: profileId,
            // Clear any shared user viewing when switching profiles
            viewingUserId: null,
            viewingUser: null,
          })
          console.log(`[Auth] Switched to profile: ${profile.displayName}`)
          // Reload page to fetch data for the new profile
          window.location.reload()
        }
      },

      getActiveProfile: () => {
        const { managedProfiles, activeProfileId } = get()
        if (!activeProfileId) return null
        return managedProfiles.find(p => p.id === activeProfileId) || null
      },

      getEffectiveProfileId: () => {
        const state = get()
        // Priority: viewingUserId (shared) > activeProfileId (managed) > user.id (legacy)
        return state.viewingUserId || state.activeProfileId || state.user?.id || ''
      },

      setOnboardingCompleted: (completed: boolean) => {
        const currentUser = get().user
        if (currentUser) {
          set({
            user: { ...currentUser, onboardingCompleted: completed }
          })
        }
      },
    }),
    {
      name: 't1d-ai-auth',
      partialize: (state) => ({
        user: state.user,
        tokens: state.tokens,
        isAuthenticated: state.isAuthenticated,
        viewingUserId: state.viewingUserId,
        viewingUser: state.viewingUser,
        sharedWithMe: state.sharedWithMe,
        // Managed profiles
        managedProfiles: state.managedProfiles,
        activeProfileId: state.activeProfileId,
      }),
    }
  )
)

// Helper hook to get auth header
export function useAuthHeader(): string | null {
  const tokens = useAuthStore((state) => state.tokens)
  return tokens?.accessToken ? `Bearer ${tokens.accessToken}` : null
}

/**
 * Check if stored auth is still valid on app load.
 * Validates token expiry, clears stale tokens, and starts refresh timer.
 */
export function initializeAuth(): boolean {
  const stored = localStorage.getItem('t1d-ai-auth')
  if (stored) {
    try {
      const data = JSON.parse(stored)
      if (data.state?.isAuthenticated && data.state?.tokens?.accessToken) {
        const accessToken = data.state.tokens.accessToken
        const refreshToken = data.state.tokens.refreshToken

        // Check if access token is expired
        if (isTokenExpired(accessToken)) {
          // Access token expired - check if we can refresh
          if (refreshToken && !isTokenExpired(refreshToken)) {
            // Refresh token is still valid - try to refresh
            console.log('[Auth] Access token expired but refresh token valid, will refresh on first API call')
            // Don't clear auth - let the API interceptor handle the refresh
            // Still start the timer which will trigger an immediate refresh
            startTokenRefreshTimer()
            return true
          }

          // Both tokens expired - clear everything
          console.warn('[Auth] Stored tokens are expired, clearing auth state')
          localStorage.removeItem('t1d-ai-auth')
          useAuthStore.getState().logout()
          return false
        }

        // Access token is valid - start proactive refresh timer
        const remaining = getTokenTimeRemaining(accessToken)
        console.log(`[Auth] Token valid, expires in ${Math.round(remaining / 60)} minutes`)
        startTokenRefreshTimer()
        return true
      }
    } catch {
      console.warn('[Auth] Invalid stored auth data, clearing')
      localStorage.removeItem('t1d-ai-auth')
    }
  }
  return false
}

/**
 * Force clear all auth state. Use when login fails due to stale tokens.
 */
export function clearAllAuthState(): void {
  localStorage.removeItem('t1d-ai-auth')
  useAuthStore.getState().logout()
  stopTokenRefreshTimer()
}

// ==================== Proactive Token Refresh ====================

let refreshTimerId: ReturnType<typeof setTimeout> | null = null
let wasAuthenticated = false

// Subscribe to auth state changes to manage the refresh timer
useAuthStore.subscribe((state) => {
  const isNowAuthenticated = state.isAuthenticated && !!state.tokens?.accessToken

  if (isNowAuthenticated && !wasAuthenticated) {
    // Just logged in - start the timer
    console.log('[Auth] Detected login, starting refresh timer')
    // Use setTimeout to avoid calling during render
    setTimeout(() => startTokenRefreshTimer(), 100)
  } else if (!isNowAuthenticated && wasAuthenticated) {
    // Just logged out - stop the timer
    console.log('[Auth] Detected logout, stopping refresh timer')
    stopTokenRefreshTimer()
  }

  wasAuthenticated = isNowAuthenticated
})

/**
 * Stop the background token refresh timer.
 */
export function stopTokenRefreshTimer(): void {
  if (refreshTimerId) {
    clearTimeout(refreshTimerId)
    refreshTimerId = null
    console.log('[Auth] Token refresh timer stopped')
  }
}

/**
 * Start a background timer to refresh tokens before they expire.
 * Refreshes when 5 minutes remain on the access token.
 */
export function startTokenRefreshTimer(): void {
  // Clear any existing timer
  stopTokenRefreshTimer()

  const state = useAuthStore.getState()
  const token = state.tokens?.accessToken

  if (!token) {
    console.log('[Auth] No token to schedule refresh for')
    return
  }

  const remaining = getTokenTimeRemaining(token)

  if (remaining <= 0) {
    console.log('[Auth] Token already expired, triggering refresh')
    // Try to refresh immediately
    state.refreshToken().catch(() => {
      console.warn('[Auth] Immediate refresh failed')
    })
    return
  }

  // Schedule refresh 5 minutes before expiry (or immediately if less than 5 min remaining)
  const refreshBuffer = 5 * 60 // 5 minutes in seconds
  const refreshIn = Math.max(remaining - refreshBuffer, 10) // At least 10 seconds from now

  console.log(`[Auth] Scheduling token refresh in ${Math.round(refreshIn / 60)} minutes`)

  refreshTimerId = setTimeout(async () => {
    console.log('[Auth] Proactive token refresh triggered')
    try {
      await useAuthStore.getState().refreshToken()
      // Schedule next refresh after successful refresh
      startTokenRefreshTimer()
    } catch (error) {
      console.error('[Auth] Proactive refresh failed:', error)
      // Don't logout here - let the API interceptor handle it on next request
    }
  }, refreshIn * 1000)
}
