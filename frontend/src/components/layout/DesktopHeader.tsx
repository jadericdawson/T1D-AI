/**
 * Desktop Header Component
 * Top navigation bar for desktop devices
 */
import { Link } from 'react-router-dom'
import {
  Activity, Bell, RefreshCw, Wifi, WifiOff, Clock,
  AlertCircle, Users, Eye
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { UserMenu } from './UserMenu'
import { useAuthStore } from '@/stores/authStore'
import { cn, formatTime } from '@/lib/utils'
import { useState, useEffect } from 'react'

interface DesktopHeaderProps {
  connectionStatus?: 'connected' | 'polling' | 'disconnected' | 'error'
  onRefresh?: () => void
  isRefreshing?: boolean
  className?: string
}

export function DesktopHeader({
  connectionStatus = 'polling',
  onRefresh,
  isRefreshing = false,
  className,
}: DesktopHeaderProps) {
  const {
    viewingUser,
    viewingUserId,
    sharedWithMe,
    switchToUser,
    switchToSelf,
  } = useAuthStore()

  const [currentTime, setCurrentTime] = useState(new Date())

  // Update time every minute
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 60000)
    return () => clearInterval(timer)
  }, [])

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Wifi className="h-4 w-4 text-green-500" />
      case 'polling':
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />
      case 'disconnected':
        return <WifiOff className="h-4 w-4 text-yellow-500" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <WifiOff className="h-4 w-4 text-gray-500" />
    }
  }

  const getConnectionLabel = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Real-time'
      case 'polling':
        return 'Polling'
      case 'disconnected':
        return 'Disconnected'
      case 'error':
        return 'Error'
      default:
        return 'Unknown'
    }
  }

  return (
    <header
      className={cn(
        'sticky top-0 z-50 flex h-16 items-center justify-between border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6',
        className
      )}
    >
      {/* Left: Logo and Navigation */}
      <div className="flex items-center gap-6">
        <Link to="/dashboard" className="flex items-center gap-2">
          <Activity className="h-6 w-6 text-primary" />
          <span className="text-xl font-bold">T1D-AI</span>
        </Link>

        {/* User Switcher for Shared Access */}
        {(sharedWithMe.length > 0 || viewingUser) && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                {viewingUser ? (
                  <>
                    <Eye className="h-4 w-4" />
                    <span className="max-w-[120px] truncate">
                      {viewingUser.displayName || viewingUser.email}
                    </span>
                  </>
                ) : (
                  <>
                    <Users className="h-4 w-4" />
                    <span>My Data</span>
                  </>
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56">
              <DropdownMenuLabel>Viewing Data For</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => switchToSelf()}
                className={cn(!viewingUserId && 'bg-accent')}
              >
                <Users className="mr-2 h-4 w-4" />
                My Data
                {!viewingUserId && (
                  <Badge variant="secondary" className="ml-auto text-xs">
                    Current
                  </Badge>
                )}
              </DropdownMenuItem>
              {sharedWithMe.length > 0 && (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel className="text-xs text-muted-foreground">
                    Shared With Me
                  </DropdownMenuLabel>
                  {sharedWithMe.map((shared) => (
                    <DropdownMenuItem
                      key={shared.id}
                      onClick={() => switchToUser(shared.id, shared)}
                      className={cn(viewingUserId === shared.id && 'bg-accent')}
                    >
                      <Eye className="mr-2 h-4 w-4" />
                      <span className="truncate">
                        {shared.displayName || shared.email}
                      </span>
                      {viewingUserId === shared.id && (
                        <Badge variant="secondary" className="ml-auto text-xs">
                          Viewing
                        </Badge>
                      )}
                    </DropdownMenuItem>
                  ))}
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        )}

        {/* Viewing Badge */}
        {viewingUser && (
          <Badge variant="outline" className="bg-yellow-500/10 text-yellow-600 border-yellow-500/30">
            <Eye className="mr-1 h-3 w-3" />
            Viewing {viewingUser.displayName?.split(' ')[0] || 'Shared'}'s data
          </Badge>
        )}
      </div>

      {/* Right: Status and Actions */}
      <div className="flex items-center gap-4">
        {/* Connection Status */}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant="outline"
                className={cn(
                  'gap-1.5 cursor-default',
                  connectionStatus === 'connected' && 'border-green-500/30 text-green-600',
                  connectionStatus === 'polling' && 'border-blue-500/30 text-blue-600',
                  connectionStatus === 'error' && 'border-red-500/30 text-red-600'
                )}
              >
                {getConnectionIcon()}
                {getConnectionLabel()}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <p>Data connection status</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {/* Current Time */}
        <div className="hidden lg:flex items-center gap-1.5 text-sm text-muted-foreground">
          <Clock className="h-4 w-4" />
          <span>{formatTime(currentTime)}</span>
          <span className="text-xs opacity-70">EST</span>
        </div>

        {/* Refresh Button */}
        {onRefresh && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onRefresh}
                  disabled={isRefreshing}
                >
                  <RefreshCw
                    className={cn('h-5 w-5', isRefreshing && 'animate-spin')}
                  />
                  <span className="sr-only">Refresh data</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Refresh data</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}

        {/* Notifications */}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="relative">
                <Bell className="h-5 w-5" />
                <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-orange-500" />
                <span className="sr-only">Notifications</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Notifications</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {/* User Menu */}
        <UserMenu />
      </div>
    </header>
  )
}

export default DesktopHeader
