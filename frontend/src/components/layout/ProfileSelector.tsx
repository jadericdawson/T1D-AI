/**
 * Profile Selector Component
 * Dropdown for switching between managed profiles and shared users
 */
import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Users, ChevronDown, User, Plus, Baby, Heart,
  UserCircle, Activity, AlertCircle, CheckCircle, Clock
} from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { useAuthStore, ProfileSummary, ProfileRelationship, SyncStatus } from '@/stores/authStore'
import { cn } from '@/lib/utils'

interface ProfileSelectorProps {
  className?: string
  compact?: boolean
}

/**
 * Get icon for relationship type
 */
function getRelationshipIcon(relationship: ProfileRelationship) {
  switch (relationship) {
    case 'self':
      return <User className="h-4 w-4" />
    case 'child':
      return <Baby className="h-4 w-4" />
    case 'spouse':
      return <Heart className="h-4 w-4" />
    case 'parent':
      return <UserCircle className="h-4 w-4" />
    default:
      return <Users className="h-4 w-4" />
  }
}

/**
 * Get sync status indicator
 */
function getSyncStatusIndicator(status: SyncStatus) {
  switch (status) {
    case 'ok':
      return <CheckCircle className="h-3 w-3 text-green-500" />
    case 'error':
      return <AlertCircle className="h-3 w-3 text-red-500" />
    case 'pending':
      return <Clock className="h-3 w-3 text-yellow-500" />
    default:
      return null
  }
}

/**
 * Get relationship label
 */
function getRelationshipLabel(relationship: ProfileRelationship): string {
  const labels: Record<ProfileRelationship, string> = {
    self: 'You',
    child: 'Child',
    spouse: 'Spouse',
    parent: 'Parent',
    other: 'Other',
  }
  return labels[relationship] || 'Other'
}

/**
 * Profile avatar component
 */
function ProfileAvatar({ profile, size = 'md' }: { profile: ProfileSummary; size?: 'sm' | 'md' | 'lg' }) {
  const sizeClasses = {
    sm: 'h-6 w-6 text-xs',
    md: 'h-8 w-8 text-sm',
    lg: 'h-10 w-10 text-base',
  }

  const initials = profile.displayName
    .split(' ')
    .map(n => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)

  if (profile.avatarUrl) {
    return (
      <img
        src={profile.avatarUrl}
        alt={profile.displayName}
        className={cn('rounded-full object-cover', sizeClasses[size])}
      />
    )
  }

  return (
    <div
      className={cn(
        'flex items-center justify-center rounded-full bg-primary/10 font-medium text-primary',
        sizeClasses[size]
      )}
    >
      {initials}
    </div>
  )
}

export function ProfileSelector({ className, compact = false }: ProfileSelectorProps) {
  const navigate = useNavigate()
  const {
    managedProfiles,
    activeProfileId,
    isLoadingProfiles,
    sharedWithMe,
    viewingUser,
    loadProfiles,
    switchToProfile,
    switchToSelf,
    getActiveProfile,
    isAuthenticated,
  } = useAuthStore()

  // Load profiles on mount if authenticated and not loaded
  // Also reload if sharedWithMe might be stale (e.g., after accepting an invitation)
  useEffect(() => {
    if (isAuthenticated && !isLoadingProfiles) {
      // Load profiles if we haven't loaded yet OR if we might have stale shared data
      // The viewingUser being set but sharedWithMe empty indicates we accepted an invite but didn't reload
      const needsLoad = managedProfiles.length === 0
      const hasStaleSharedData = viewingUser && sharedWithMe.length === 0

      if (needsLoad || hasStaleSharedData) {
        loadProfiles()
      }
    }
  }, [isAuthenticated, managedProfiles.length, sharedWithMe.length, viewingUser, isLoadingProfiles, loadProfiles])

  // Get active profile
  const activeProfile = getActiveProfile()

  // Get display name - could be own profile OR shared profile (treated identically)
  const displayName = viewingUser
    ? viewingUser.displayName || viewingUser.email
    : activeProfile?.displayName || 'Select Profile'

  // Don't show if not authenticated
  if (!isAuthenticated) {
    return null
  }

  // Loading state
  if (isLoadingProfiles && managedProfiles.length === 0) {
    return (
      <Button variant="outline" disabled className={cn('gap-2', className)}>
        <Users className="h-4 w-4 animate-pulse" />
        Loading...
      </Button>
    )
  }

  // No profiles yet - show prompt to create
  if (managedProfiles.length === 0) {
    return (
      <Button
        variant="outline"
        className={cn('gap-2', className)}
        onClick={() => navigate('/settings?tab=profiles')}
      >
        <Plus className="h-4 w-4" />
        Add Profile
      </Button>
    )
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className={cn('gap-2', className)}>
          {activeProfile ? (
            <>
              <ProfileAvatar profile={activeProfile} size="sm" />
              {!compact && <span className="max-w-[120px] truncate">{displayName}</span>}
              {activeProfile.syncStatus && getSyncStatusIndicator(activeProfile.syncStatus)}
            </>
          ) : viewingUser ? (
            <>
              {/* Shared profile - display same as owned profiles, no special badge */}
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-medium text-primary">
                {(viewingUser.displayName || viewingUser.email || '?').charAt(0).toUpperCase()}
              </div>
              {!compact && <span className="max-w-[120px] truncate">{displayName}</span>}
            </>
          ) : (
            <>
              <Users className="h-4 w-4" />
              {!compact && <span>Select Profile</span>}
            </>
          )}
          <ChevronDown className="h-4 w-4 opacity-50" />
        </Button>
      </DropdownMenuTrigger>

      <DropdownMenuContent className="w-64" align="start" sideOffset={8}>
        {/* My Profiles Section */}
        <DropdownMenuLabel className="text-xs text-muted-foreground uppercase tracking-wider">
          My Profiles
        </DropdownMenuLabel>

        {managedProfiles.map((profile) => (
          <DropdownMenuItem
            key={profile.id}
            onClick={() => {
              // Clear any shared viewing state when selecting own profile
              if (viewingUser) {
                switchToSelf()
              }
              switchToProfile(profile.id)
            }}
            className={cn(
              'flex items-center gap-3 cursor-pointer',
              activeProfileId === profile.id && !viewingUser && 'bg-primary/10'
            )}
          >
            <ProfileAvatar profile={profile} size="md" />
            <div className="flex flex-col flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium truncate">{profile.displayName}</span>
                {activeProfileId === profile.id && !viewingUser && (
                  <Activity className="h-3 w-3 text-primary" />
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                {getRelationshipIcon(profile.relationship)}
                <span>{getRelationshipLabel(profile.relationship)}</span>
                {profile.dataSourceCount > 0 && (
                  <span className="text-muted-foreground/60">
                    • {profile.dataSourceCount} source{profile.dataSourceCount !== 1 ? 's' : ''}
                  </span>
                )}
              </div>
            </div>
            <div className="flex items-center">
              {getSyncStatusIndicator(profile.syncStatus)}
            </div>
          </DropdownMenuItem>
        ))}

        {/* Shared With Me Section */}
        {sharedWithMe.length > 0 && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuLabel className="text-xs text-muted-foreground uppercase tracking-wider">
              Shared With Me
            </DropdownMenuLabel>

            {sharedWithMe.map((shared) => (
              <DropdownMenuItem
                key={shared.id}
                onClick={() => useAuthStore.getState().switchToUser(shared.id, shared)}
                className={cn(
                  'flex items-center gap-3 cursor-pointer',
                  viewingUser?.id === shared.id && 'bg-primary/10'
                )}
              >
                {/* Same avatar style as owned profiles */}
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 font-medium text-primary text-sm">
                  {(shared.displayName || shared.email || '?')
                    .split(' ')
                    .map(n => n[0])
                    .join('')
                    .toUpperCase()
                    .slice(0, 2)}
                </div>
                <div className="flex flex-col flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium truncate">
                      {shared.displayName || shared.email}
                    </span>
                    {viewingUser?.id === shared.id && (
                      <Activity className="h-3 w-3 text-primary" />
                    )}
                  </div>
                  <span className="text-xs text-muted-foreground capitalize">
                    {shared.role} access
                  </span>
                </div>
              </DropdownMenuItem>
            ))}
          </>
        )}

        {/* Actions */}
        <DropdownMenuSeparator />

        {/* Return to own data if viewing shared */}
        {viewingUser && (
          <DropdownMenuItem
            onClick={switchToSelf}
            className="text-primary cursor-pointer"
          >
            <User className="mr-2 h-4 w-4" />
            Return to My Data
          </DropdownMenuItem>
        )}

        {/* Add Profile */}
        <DropdownMenuItem
          onClick={() => navigate('/settings?tab=profiles&action=add')}
          className="cursor-pointer"
        >
          <Plus className="mr-2 h-4 w-4" />
          Add Profile
        </DropdownMenuItem>

        {/* Manage Profiles */}
        <DropdownMenuItem
          onClick={() => navigate('/settings?tab=profiles')}
          className="cursor-pointer text-muted-foreground"
        >
          <Users className="mr-2 h-4 w-4" />
          Manage Profiles
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default ProfileSelector
