/**
 * Sharing Manager Component
 * Displays and manages data sharing with other users
 */
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  UserPlus, Users, Eye, Clock, Trash2, Loader2,
  ChevronRight, Shield, Heart
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { sharingApi, ShareInfo } from '@/lib/api'
import { useAuthStore } from '@/stores/authStore'
import { InviteModal } from './InviteModal'

const ROLE_CONFIG: Record<string, { icon: typeof Eye; color: string; bgColor: string; label: string }> = {
  viewer: { icon: Eye, color: 'text-blue-500', bgColor: 'bg-blue-500/10', label: 'Viewer' },
  caregiver: { icon: Heart, color: 'text-pink-500', bgColor: 'bg-pink-500/10', label: 'Caregiver' },
  admin: { icon: Shield, color: 'text-purple-500', bgColor: 'bg-purple-500/10', label: 'Admin' },
}

export function SharingManager() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { switchToUser } = useAuthStore()
  const [inviteModalOpen, setInviteModalOpen] = useState(false)

  // Fetch shares I created
  const { data: myShares = [], isLoading: isLoadingMyShares } = useQuery({
    queryKey: ['sharing', 'my-shares'],
    queryFn: sharingApi.getMyShares,
  })

  // Fetch shares with me
  const { data: sharedWithMe = [], isLoading: isLoadingSharedWithMe } = useQuery({
    queryKey: ['sharing', 'shared-with-me'],
    queryFn: sharingApi.getSharedWithMe,
  })

  // Fetch pending invitations I sent (not the ones sent to me)
  const { data: pendingInvitations = [], isLoading: isLoadingPending } = useQuery({
    queryKey: ['sharing', 'pending-invitations'],
    queryFn: sharingApi.getPendingInvitations,
  })

  // Revoke share mutation
  const revokeMutation = useMutation({
    mutationFn: sharingApi.revokeShare,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sharing'] })
    },
  })

  const handleViewData = (share: ShareInfo) => {
    // Switch to viewing this user's data
    switchToUser(share.ownerId, {
      id: share.ownerId,
      email: share.ownerEmail || '',
      displayName: share.ownerName || null,
      role: share.role,
      permissions: share.permissions,
    })
    navigate('/dashboard')
  }

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['sharing'] })
  }

  const isLoading = isLoadingMyShares || isLoadingSharedWithMe || isLoadingPending

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
  }

  const formatExpiry = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const daysLeft = Math.ceil((date.getTime() - now.getTime()) / (1000 * 60 * 60 * 24))
    if (daysLeft <= 0) return 'Expired'
    if (daysLeft === 1) return 'Expires tomorrow'
    return `Expires in ${daysLeft} days`
  }

  return (
    <div className="space-y-6">
      {/* Invite Button */}
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-semibold">Data Sharing</h3>
          <p className="text-sm text-muted-foreground">
            Share your glucose data with family, caregivers, or healthcare providers
          </p>
        </div>
        <Button onClick={() => setInviteModalOpen(true)}>
          <UserPlus className="w-4 h-4 mr-2" />
          Invite Someone
        </Button>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <>
          {/* People I'm Sharing With */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Users className="w-5 h-5" />
                People I'm Sharing With
              </CardTitle>
              <CardDescription>
                These people can view your glucose data based on their access level
              </CardDescription>
            </CardHeader>
            <CardContent>
              {myShares.length === 0 ? (
                <div className="text-center py-6 text-muted-foreground">
                  <Users className="w-10 h-10 mx-auto mb-2 opacity-50" />
                  <p>You haven't shared your data with anyone yet</p>
                  <Button
                    variant="link"
                    onClick={() => setInviteModalOpen(true)}
                    className="mt-2"
                  >
                    Invite someone to get started
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {myShares.filter(s => s.isActive).map((share) => {
                    const roleConfig = ROLE_CONFIG[share.role] || ROLE_CONFIG.viewer
                    const RoleIcon = roleConfig.icon
                    return (
                      <div
                        key={share.id}
                        className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                      >
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-full ${roleConfig.bgColor}`}>
                            <RoleIcon className={`w-4 h-4 ${roleConfig.color}`} />
                          </div>
                          <div>
                            <p className="font-medium">
                              {share.sharedWithName || share.sharedWithEmail}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {share.sharedWithEmail}
                            </p>
                            {share.profileName && (
                              <p className="text-xs text-cyan-500 mt-0.5">
                                Sharing {share.profileName}'s data
                              </p>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary" className={roleConfig.bgColor}>
                            {roleConfig.label}
                          </Badge>
                          <AlertDialog>
                            <AlertDialogTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="text-destructive hover:text-destructive"
                              >
                                <Trash2 className="w-4 h-4" />
                              </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                              <AlertDialogHeader>
                                <AlertDialogTitle>Revoke Access</AlertDialogTitle>
                                <AlertDialogDescription>
                                  Are you sure you want to stop sharing your data with{' '}
                                  {share.sharedWithName || share.sharedWithEmail}?
                                  They will no longer be able to view your glucose data.
                                </AlertDialogDescription>
                              </AlertDialogHeader>
                              <AlertDialogFooter>
                                <AlertDialogCancel>Cancel</AlertDialogCancel>
                                <AlertDialogAction
                                  onClick={() => revokeMutation.mutate(share.id)}
                                  className="bg-destructive hover:bg-destructive/90"
                                >
                                  Revoke Access
                                </AlertDialogAction>
                              </AlertDialogFooter>
                            </AlertDialogContent>
                          </AlertDialog>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Pending Invitations to Accept (invitations sent TO me) */}
          {pendingInvitations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Clock className="w-5 h-5" />
                  Pending Invitations
                </CardTitle>
                <CardDescription>
                  Invitations from others to view their data - click to accept
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {pendingInvitations.map((inv) => {
                    const roleConfig = ROLE_CONFIG[inv.role] || ROLE_CONFIG.viewer
                    return (
                      <div
                        key={inv.id}
                        className="flex items-center justify-between p-3 rounded-lg bg-muted/30 border border-dashed"
                      >
                        <div className="flex items-center gap-3">
                          <Clock className="w-5 h-5 text-muted-foreground" />
                          <div>
                            <p className="font-medium">
                              {inv.ownerName || inv.ownerEmail} wants to share
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {formatExpiry(inv.expiresAt)}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{roleConfig.label}</Badge>
                          <Button
                            variant="default"
                            size="sm"
                            onClick={() => navigate(`/accept-invite/${inv.id}`)}
                          >
                            Accept
                          </Button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Shared With Me */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Eye className="w-5 h-5" />
                Shared With Me
              </CardTitle>
              <CardDescription>
                People who have shared their glucose data with you
              </CardDescription>
            </CardHeader>
            <CardContent>
              {sharedWithMe.length === 0 ? (
                <div className="text-center py-6 text-muted-foreground">
                  <Eye className="w-10 h-10 mx-auto mb-2 opacity-50" />
                  <p>No one has shared their data with you yet</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {sharedWithMe.filter(s => s.isActive).map((share) => {
                    const roleConfig = ROLE_CONFIG[share.role] || ROLE_CONFIG.viewer
                    const RoleIcon = roleConfig.icon
                    return (
                      <div
                        key={share.id}
                        className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                      >
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-full ${roleConfig.bgColor}`}>
                            <RoleIcon className={`w-4 h-4 ${roleConfig.color}`} />
                          </div>
                          <div>
                            <p className="font-medium">
                              {share.profileName
                                ? `${share.profileName}'s data`
                                : (share.ownerName || share.ownerEmail)
                              }
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {share.profileName && share.ownerName
                                ? `Shared by ${share.ownerName} on ${formatDate(share.createdAt)}`
                                : `Shared on ${formatDate(share.createdAt)}`
                              }
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary" className={roleConfig.bgColor}>
                            {roleConfig.label}
                          </Badge>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleViewData(share)}
                          >
                            View Data
                            <ChevronRight className="w-4 h-4 ml-1" />
                          </Button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {/* Invite Modal */}
      <InviteModal
        open={inviteModalOpen}
        onOpenChange={setInviteModalOpen}
        onSuccess={handleRefresh}
      />
    </div>
  )
}

export default SharingManager
