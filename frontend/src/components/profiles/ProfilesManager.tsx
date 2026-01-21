/**
 * Profiles Manager Component
 * Manage profiles for people whose diabetes data you track
 */
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  User, Baby, Heart, UserCircle, Users, Plus, Pencil,
  Loader2, AlertCircle, Share2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { useAuthStore } from '@/stores/authStore'
import { profilesApi, sharingApi, ManagedProfile, ProfileRelationship, DiabetesType, ProfileUpdateData, ProfileCreateData } from '@/lib/api'
import { toast } from '@/hooks/useToast'

type Relationship = ProfileRelationship

function getRelationshipIcon(relationship: Relationship) {
  switch (relationship) {
    case 'self':
      return <User className="h-5 w-5" />
    case 'child':
      return <Baby className="h-5 w-5" />
    case 'spouse':
      return <Heart className="h-5 w-5" />
    case 'parent':
      return <UserCircle className="h-5 w-5" />
    default:
      return <Users className="h-5 w-5" />
  }
}

function getRelationshipLabel(relationship: Relationship): string {
  const labels: Record<Relationship, string> = {
    self: 'Yourself',
    child: 'Child',
    spouse: 'Spouse',
    parent: 'Parent',
    other: 'Other',
  }
  return labels[relationship] || 'Other'
}

interface ProfileFormData {
  displayName: string
  relationship: Relationship
  diabetesType: DiabetesType
}

export function ProfilesManager() {
  const queryClient = useQueryClient()
  const { loadProfiles } = useAuthStore()
  const [editingProfile, setEditingProfile] = useState<ManagedProfile | null>(null)
  const [isAddingNew, setIsAddingNew] = useState(false)
  const [sharingProfile, setSharingProfile] = useState<ManagedProfile | null>(null)
  const [shareEmail, setShareEmail] = useState('')
  const [formData, setFormData] = useState<ProfileFormData>({
    displayName: '',
    relationship: 'child',
    diabetesType: 'T1D',
  })

  // Fetch profiles
  const { data: profilesData, isLoading, error } = useQuery({
    queryKey: ['profiles'],
    queryFn: () => profilesApi.list(),
  })

  const profiles = profilesData?.profiles || []

  // Update profile mutation
  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: ProfileUpdateData }) =>
      profilesApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      loadProfiles() // Refresh the profile selector
      setEditingProfile(null)
    },
    onError: (error) => {
      console.error('Failed to update profile:', error)
      alert(`Failed to update profile: ${error instanceof Error ? error.message : 'Unknown error'}`)
    },
  })

  // Create profile mutation
  const createMutation = useMutation({
    mutationFn: (data: ProfileCreateData) => profilesApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] })
      loadProfiles() // Refresh the profile selector
      setIsAddingNew(false)
      setFormData({ displayName: '', relationship: 'child', diabetesType: 'T1D' })
    },
  })

  // Share profile mutation
  const shareMutation = useMutation({
    mutationFn: ({ email, profileId, profileName }: { email: string; profileId: string; profileName: string }) =>
      sharingApi.invite(email, 'viewer', [], profileId, profileName),
    onSuccess: () => {
      toast({
        title: 'Invitation sent',
        description: `${sharingProfile?.displayName}'s data will be shared when they accept.`,
      })
      setSharingProfile(null)
      setShareEmail('')
    },
    onError: (error) => {
      toast({
        title: 'Failed to send invitation',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    },
  })

  const handleEdit = (profile: ManagedProfile) => {
    setEditingProfile(profile)
    setFormData({
      displayName: profile.displayName,
      relationship: profile.relationship,
      diabetesType: (profile.diabetesType || 'T1D') as DiabetesType,
    })
  }

  const handleSave = () => {
    if (editingProfile) {
      const updateData: ProfileUpdateData = {
        displayName: formData.displayName,
        relationship: formData.relationship,
        diabetesType: formData.diabetesType,
      }
      updateMutation.mutate({ id: editingProfile.id, data: updateData })
    }
  }

  const handleCreate = () => {
    const createData: ProfileCreateData = {
      displayName: formData.displayName,
      relationship: formData.relationship,
      diabetesType: formData.diabetesType,
    }
    createMutation.mutate(createData)
  }

  const handleCancel = () => {
    setEditingProfile(null)
    setIsAddingNew(false)
    setFormData({ displayName: '', relationship: 'child', diabetesType: 'T1D' })
  }

  const handleShare = (profile: ManagedProfile) => {
    setSharingProfile(profile)
    setShareEmail('')
  }

  const handleSendInvite = () => {
    if (sharingProfile && shareEmail) {
      shareMutation.mutate({
        email: shareEmail,
        profileId: sharingProfile.id,
        profileName: sharingProfile.displayName,
      })
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-6 w-6 animate-spin text-cyan-500" />
        <span className="ml-2 text-gray-400">Loading profiles...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-8 text-red-400">
        <AlertCircle className="h-5 w-5 mr-2" />
        Failed to load profiles
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Profile List */}
      {profiles.map((profile) => (
        <div
          key={profile.id}
          className={cn(
            'flex items-center justify-between p-4 rounded-lg border',
            'bg-slate-800/50 border-slate-700 hover:border-cyan-500/30 transition-colors'
          )}
        >
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-cyan-500/10 text-cyan-400">
              {getRelationshipIcon(profile.relationship as Relationship)}
            </div>
            <div>
              <h3 className="font-medium text-white">{profile.displayName}</h3>
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <span>{getRelationshipLabel(profile.relationship as Relationship)}</span>
                <span>•</span>
                <Badge variant="outline" className="text-xs">
                  {profile.diabetesType || 'T1D'}
                </Badge>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleShare(profile)}
              className="text-gray-400 hover:text-cyan-400"
              title={`Share ${profile.displayName}'s data`}
            >
              <Share2 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleEdit(profile)}
              className="text-gray-400 hover:text-white"
            >
              <Pencil className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ))}

      {/* Add New Profile Button */}
      <Button
        variant="outline"
        className="w-full border-dashed border-slate-600 hover:border-cyan-500/50"
        onClick={() => setIsAddingNew(true)}
      >
        <Plus className="h-4 w-4 mr-2" />
        Add Profile
      </Button>

      {/* Edit/Create Dialog */}
      <Dialog open={!!editingProfile || isAddingNew} onOpenChange={(open) => !open && handleCancel()}>
        <DialogContent className="bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="text-white">
              {editingProfile ? 'Edit Profile' : 'Add New Profile'}
            </DialogTitle>
            <DialogDescription>
              {editingProfile
                ? 'Update the profile information below.'
                : 'Add a new profile for someone whose diabetes data you manage.'}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="displayName">Name</Label>
              <Input
                id="displayName"
                value={formData.displayName}
                onChange={(e) => setFormData({ ...formData, displayName: e.target.value })}
                placeholder="e.g., Emrys Dawson"
                className="bg-slate-800 border-slate-700"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="relationship">Relationship</Label>
              <Select
                value={formData.relationship}
                onValueChange={(value) => setFormData({ ...formData, relationship: value as Relationship })}
              >
                <SelectTrigger className="bg-slate-800 border-slate-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-700">
                  <SelectItem value="self">Yourself</SelectItem>
                  <SelectItem value="child">Child</SelectItem>
                  <SelectItem value="spouse">Spouse</SelectItem>
                  <SelectItem value="parent">Parent</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="diabetesType">Diabetes Type</Label>
              <Select
                value={formData.diabetesType}
                onValueChange={(value) => setFormData({ ...formData, diabetesType: value as DiabetesType })}
              >
                <SelectTrigger className="bg-slate-800 border-slate-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-700">
                  <SelectItem value="T1D">Type 1</SelectItem>
                  <SelectItem value="T2D">Type 2</SelectItem>
                  <SelectItem value="LADA">LADA</SelectItem>
                  <SelectItem value="gestational">Gestational</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter>
            <Button variant="ghost" onClick={handleCancel}>
              Cancel
            </Button>
            <Button
              onClick={editingProfile ? handleSave : handleCreate}
              disabled={!formData.displayName || updateMutation.isPending || createMutation.isPending}
              className="bg-cyan-600 hover:bg-cyan-500"
            >
              {(updateMutation.isPending || createMutation.isPending) && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              {editingProfile ? 'Save Changes' : 'Create Profile'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Share Profile Dialog */}
      <Dialog open={!!sharingProfile} onOpenChange={(open) => !open && setSharingProfile(null)}>
        <DialogContent className="bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="text-white">
              Share {sharingProfile?.displayName}'s Data
            </DialogTitle>
            <DialogDescription>
              Invite someone to view {sharingProfile?.displayName}'s glucose data.
              They'll receive an email with a link to accept the invitation.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="shareEmail">Email Address</Label>
              <Input
                id="shareEmail"
                type="email"
                value={shareEmail}
                onChange={(e) => setShareEmail(e.target.value)}
                placeholder="caregiver@example.com"
                className="bg-slate-800 border-slate-700"
              />
            </div>
            <p className="text-sm text-gray-400">
              The recipient will be able to view glucose readings, treatments, predictions, and insights
              for {sharingProfile?.displayName}.
            </p>
          </div>

          <DialogFooter>
            <Button variant="ghost" onClick={() => setSharingProfile(null)}>
              Cancel
            </Button>
            <Button
              onClick={handleSendInvite}
              disabled={!shareEmail || shareMutation.isPending}
              className="bg-cyan-600 hover:bg-cyan-500"
            >
              {shareMutation.isPending && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              Send Invitation
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default ProfilesManager
