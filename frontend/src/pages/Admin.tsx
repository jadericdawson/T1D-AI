/**
 * Admin Page
 * Platform administration for managing users and viewing analytics
 */
import { useState } from 'react'
import { motion } from 'framer-motion'
import { Link, useSearchParams } from 'react-router-dom'
import {
  ArrowLeft, Users, BarChart3, Shield, Search, Loader2,
  CheckCircle, UserCheck, UserX, RefreshCw, Trash2, Database
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
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
import { ResponsiveLayout } from '@/components/layout/ResponsiveLayout'
import { useAuthStore } from '@/stores/authStore'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '@/lib/api'
import { format } from 'date-fns'

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } }
}

// Types
interface UserSummary {
  id: string
  email: string
  displayName: string | null
  createdAt: string
  lastLoginAt: string | null
  emailVerified: boolean
  onboardingCompleted: boolean
  authProvider: string
  isAdmin: boolean
  glucoseReadingCount: number
}

interface PlatformStats {
  totalUsers: number
  verifiedUsers: number
  onboardedUsers: number
  activeUsers7d: number
  activeUsers30d: number
  totalGlucoseReadings: number
  totalTreatments: number
  newUsersToday: number
  newUsersThisWeek: number
}

interface DeleteUserResponse {
  message: string
  userId: string
  email: string
  dataRetained: boolean
  glucoseReadings: number
  treatments: number
}

// Admin API functions
const adminApi = {
  getUsers: async (skip = 0, limit = 50, search?: string): Promise<UserSummary[]> => {
    const params = new URLSearchParams({ skip: String(skip), limit: String(limit) })
    if (search) params.append('search', search)
    const response = await api.get(`/admin/users?${params}`)
    return response.data
  },
  getStats: async (): Promise<PlatformStats> => {
    const response = await api.get('/admin/stats')
    return response.data
  },
  toggleAdmin: async (userId: string, isAdmin: boolean): Promise<{ message: string }> => {
    const response = await api.put(`/admin/users/${userId}/admin?is_admin=${isAdmin}`)
    return response.data
  },
  deleteUser: async (userId: string): Promise<DeleteUserResponse> => {
    const response = await api.delete(`/admin/users/${userId}`)
    return response.data
  },
}

// Tab configuration
const TABS = [
  { id: 'users', label: 'Users', icon: Users },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
]

export default function Admin() {
  const [searchParams, setSearchParams] = useSearchParams()
  const queryClient = useQueryClient()
  const user = useAuthStore(state => state.user)

  // Get active tab from URL or default
  const activeTab = searchParams.get('tab') || 'users'
  const [searchTerm, setSearchTerm] = useState('')

  const setActiveTab = (tab: string) => {
    setSearchParams({ tab })
  }

  // Fetch users
  const { data: users, isLoading: isLoadingUsers, refetch: refetchUsers } = useQuery({
    queryKey: ['admin', 'users', searchTerm],
    queryFn: () => adminApi.getUsers(0, 100, searchTerm || undefined),
    enabled: !!user?.isAdmin,
  })

  // Fetch platform stats
  const { data: stats, isLoading: isLoadingStats, refetch: refetchStats } = useQuery({
    queryKey: ['admin', 'stats'],
    queryFn: () => adminApi.getStats(),
    enabled: !!user?.isAdmin,
  })

  // Toggle admin mutation
  const toggleAdminMutation = useMutation({
    mutationFn: ({ userId, isAdmin }: { userId: string; isAdmin: boolean }) =>
      adminApi.toggleAdmin(userId, isAdmin),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'users'] })
    },
  })

  // Delete user mutation
  const deleteUserMutation = useMutation({
    mutationFn: (userId: string) => adminApi.deleteUser(userId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'users'] })
      queryClient.invalidateQueries({ queryKey: ['admin', 'stats'] })
      alert(`User deleted. ${data.glucoseReadings.toLocaleString()} glucose readings and ${data.treatments.toLocaleString()} treatments retained for ML training.`)
    },
    onError: (error: Error) => {
      alert(`Failed to delete user: ${error.message}`)
    },
  })

  // Check if user is admin
  if (!user?.isAdmin) {
    return (
      <ResponsiveLayout
        title="Access Denied"
      >
        <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
          <Shield className="w-16 h-16 text-red-500" />
          <h1 className="text-2xl font-bold text-white">Access Denied</h1>
          <p className="text-gray-400">You don't have permission to access this page.</p>
          <Link to="/dashboard">
            <Button variant="outline">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Dashboard
            </Button>
          </Link>
        </div>
      </ResponsiveLayout>
    )
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'Never'
    try {
      return format(new Date(dateStr), 'MMM d, yyyy HH:mm')
    } catch {
      return 'Invalid date'
    }
  }

  return (
    <ResponsiveLayout
      title="Admin Panel"
    >
      <motion.div
        initial="hidden"
        animate="visible"
        variants={fadeIn}
        className="space-y-6"
      >
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link to="/dashboard">
              <Button variant="ghost" size="icon">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-white">Admin Panel</h1>
              <p className="text-gray-400 text-sm">Manage users and view platform analytics</p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              refetchUsers()
              refetchStats()
            }}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2 max-w-md">
            {TABS.map(tab => (
              <TabsTrigger key={tab.id} value={tab.id} className="flex items-center gap-2">
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>

          {/* Users Tab */}
          <TabsContent value="users" className="space-y-4 mt-4">
            {/* Search */}
            <div className="relative max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                placeholder="Search by email or name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Users Table */}
            <Card className="bg-slate-800/50 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Users ({users?.length || 0})</CardTitle>
                <CardDescription>All registered users on the platform</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingUsers ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 animate-spin text-cyan-500" />
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow className="border-gray-700">
                        <TableHead className="text-gray-400">User</TableHead>
                        <TableHead className="text-gray-400">Status</TableHead>
                        <TableHead className="text-gray-400">Auth</TableHead>
                        <TableHead className="text-gray-400">Readings</TableHead>
                        <TableHead className="text-gray-400">Created</TableHead>
                        <TableHead className="text-gray-400">Last Login</TableHead>
                        <TableHead className="text-gray-400">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {users?.map((u) => (
                        <TableRow key={u.id} className="border-gray-700">
                          <TableCell>
                            <div>
                              <p className="font-medium text-white">{u.displayName || 'No name'}</p>
                              <p className="text-sm text-gray-400">{u.email}</p>
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="flex flex-col gap-1">
                              {u.emailVerified ? (
                                <Badge variant="default" className="bg-green-600 text-xs">Verified</Badge>
                              ) : (
                                <Badge variant="secondary" className="text-xs">Unverified</Badge>
                              )}
                              {u.isAdmin && (
                                <Badge variant="default" className="bg-purple-600 text-xs">Admin</Badge>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="text-xs capitalize">
                              {u.authProvider}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-gray-300">
                            {u.glucoseReadingCount.toLocaleString()}
                          </TableCell>
                          <TableCell className="text-gray-400 text-sm">
                            {formatDate(u.createdAt)}
                          </TableCell>
                          <TableCell className="text-gray-400 text-sm">
                            {formatDate(u.lastLoginAt)}
                          </TableCell>
                          <TableCell>
                            {u.id !== user?.id && (
                              <div className="flex items-center gap-1">
                                {/* Admin toggle */}
                                <AlertDialog>
                                  <AlertDialogTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className={u.isAdmin ? 'text-red-400 hover:text-red-300' : 'text-cyan-400 hover:text-cyan-300'}
                                    >
                                      {u.isAdmin ? (
                                        <>
                                          <UserX className="w-4 h-4 mr-1" />
                                          Revoke
                                        </>
                                      ) : (
                                        <>
                                          <UserCheck className="w-4 h-4 mr-1" />
                                          Grant
                                        </>
                                      )}
                                    </Button>
                                  </AlertDialogTrigger>
                                  <AlertDialogContent className="bg-slate-800 border-gray-700">
                                    <AlertDialogHeader>
                                      <AlertDialogTitle className="text-white">
                                        {u.isAdmin ? 'Revoke Admin Access' : 'Grant Admin Access'}
                                      </AlertDialogTitle>
                                      <AlertDialogDescription>
                                        {u.isAdmin
                                          ? `Are you sure you want to revoke admin privileges from ${u.email}?`
                                          : `Are you sure you want to grant admin privileges to ${u.email}?`}
                                      </AlertDialogDescription>
                                    </AlertDialogHeader>
                                    <AlertDialogFooter>
                                      <AlertDialogCancel className="bg-gray-700 border-gray-600">Cancel</AlertDialogCancel>
                                      <AlertDialogAction
                                        onClick={() => toggleAdminMutation.mutate({ userId: u.id, isAdmin: !u.isAdmin })}
                                        className={u.isAdmin ? 'bg-red-600 hover:bg-red-700' : 'bg-cyan-600 hover:bg-cyan-700'}
                                      >
                                        {u.isAdmin ? 'Revoke' : 'Grant'}
                                      </AlertDialogAction>
                                    </AlertDialogFooter>
                                  </AlertDialogContent>
                                </AlertDialog>

                                {/* Delete user */}
                                <AlertDialog>
                                  <AlertDialogTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className="text-red-400 hover:text-red-300 hover:bg-red-900/20"
                                    >
                                      <Trash2 className="w-4 h-4" />
                                    </Button>
                                  </AlertDialogTrigger>
                                  <AlertDialogContent className="bg-slate-800 border-gray-700">
                                    <AlertDialogHeader>
                                      <AlertDialogTitle className="text-white flex items-center gap-2">
                                        <Trash2 className="w-5 h-5 text-red-500" />
                                        Delete User Account
                                      </AlertDialogTitle>
                                      <AlertDialogDescription className="space-y-3">
                                        <p>
                                          Are you sure you want to delete <strong className="text-white">{u.email}</strong>?
                                        </p>
                                        <div className="bg-slate-900/50 p-3 rounded-lg space-y-2">
                                          <div className="flex items-center gap-2 text-green-400">
                                            <Database className="w-4 h-4" />
                                            <span className="text-sm">Data will be retained for ML training:</span>
                                          </div>
                                          <ul className="text-sm text-gray-400 ml-6 space-y-1">
                                            <li>• {u.glucoseReadingCount.toLocaleString()} glucose readings</li>
                                            <li>• Treatment history</li>
                                          </ul>
                                        </div>
                                        <p className="text-yellow-400 text-sm">
                                          This will remove the user's login credentials and profile. This action cannot be undone.
                                        </p>
                                      </AlertDialogDescription>
                                    </AlertDialogHeader>
                                    <AlertDialogFooter>
                                      <AlertDialogCancel className="bg-gray-700 border-gray-600">Cancel</AlertDialogCancel>
                                      <AlertDialogAction
                                        onClick={() => deleteUserMutation.mutate(u.id)}
                                        className="bg-red-600 hover:bg-red-700"
                                        disabled={deleteUserMutation.isPending}
                                      >
                                        {deleteUserMutation.isPending ? (
                                          <>
                                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                            Deleting...
                                          </>
                                        ) : (
                                          <>
                                            <Trash2 className="w-4 h-4 mr-2" />
                                            Delete User
                                          </>
                                        )}
                                      </AlertDialogAction>
                                    </AlertDialogFooter>
                                  </AlertDialogContent>
                                </AlertDialog>
                              </div>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-4 mt-4">
            {isLoadingStats ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-cyan-500" />
              </div>
            ) : stats ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* User Stats */}
                <Card className="bg-slate-800/50 border-gray-700">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg text-white flex items-center gap-2">
                      <Users className="w-5 h-5 text-cyan-500" />
                      User Statistics
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total Users</span>
                      <span className="text-white font-medium">{stats.totalUsers}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Verified</span>
                      <span className="text-green-400 font-medium">{stats.verifiedUsers}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Onboarded</span>
                      <span className="text-cyan-400 font-medium">{stats.onboardedUsers}</span>
                    </div>
                  </CardContent>
                </Card>

                {/* Activity Stats */}
                <Card className="bg-slate-800/50 border-gray-700">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg text-white flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-green-500" />
                      Activity
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Active (7 days)</span>
                      <span className="text-white font-medium">{stats.activeUsers7d}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Active (30 days)</span>
                      <span className="text-white font-medium">{stats.activeUsers30d}</span>
                    </div>
                  </CardContent>
                </Card>

                {/* Growth Stats */}
                <Card className="bg-slate-800/50 border-gray-700">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg text-white flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-purple-500" />
                      Growth
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">New Today</span>
                      <span className="text-white font-medium">{stats.newUsersToday}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">New This Week</span>
                      <span className="text-white font-medium">{stats.newUsersThisWeek}</span>
                    </div>
                  </CardContent>
                </Card>

                {/* Data Stats */}
                <Card className="bg-slate-800/50 border-gray-700 md:col-span-2 lg:col-span-3">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg text-white flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-orange-500" />
                      Platform Data
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-6">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Total Glucose Readings</span>
                        <span className="text-white font-medium">{stats.totalGlucoseReadings.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Total Treatments</span>
                        <span className="text-white font-medium">{stats.totalTreatments.toLocaleString()}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                No statistics available
              </div>
            )}
          </TabsContent>
        </Tabs>
      </motion.div>
    </ResponsiveLayout>
  )
}
