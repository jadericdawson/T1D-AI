/**
 * User Menu Component
 * Comprehensive dropdown menu for user actions and settings
 */
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  User, Settings, LogOut, Database, Brain, Link2, Shield,
  HelpCircle, MessageSquare, ChevronRight, Moon, Sun, Bell,
  Share2, Activity, Plug
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
import { Switch } from '@/components/ui/switch'
import { useAuthStore } from '@/stores/authStore'

interface UserMenuProps {
  className?: string
  trigger?: React.ReactNode
  align?: 'start' | 'center' | 'end'
}

export function UserMenu({ className, trigger, align = 'end' }: UserMenuProps) {
  const navigate = useNavigate()
  const { user, logout, isAuthenticated } = useAuthStore()
  const [darkMode, setDarkMode] = useState(
    document.documentElement.classList.contains('dark')
  )

  const toggleDarkMode = () => {
    const newMode = !darkMode
    setDarkMode(newMode)
    if (newMode) {
      document.documentElement.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      document.documentElement.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  if (!isAuthenticated || !user) {
    return (
      <Button variant="outline" onClick={() => navigate('/login')}>
        Sign In
      </Button>
    )
  }

  // Get user initials for avatar
  const initials = user.displayName
    ? user.displayName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : user.email.slice(0, 2).toUpperCase()

  const defaultTrigger = (
    <Button
      variant="ghost"
      className="relative h-10 w-10 rounded-full bg-primary/10 hover:bg-primary/20"
    >
      <span className="text-sm font-semibold text-primary">{initials}</span>
    </Button>
  )

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild className={className}>
        {trigger || defaultTrigger}
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-72" align={align} sideOffset={8}>
        {/* User Info Header */}
        <DropdownMenuLabel className="font-normal">
          <div className="flex items-center gap-3 py-2">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
              <span className="text-lg font-semibold text-primary">{initials}</span>
            </div>
            <div className="flex flex-col space-y-1">
              <p className="text-sm font-medium leading-none">
                {user.displayName || 'T1D User'}
              </p>
              <p className="text-xs leading-none text-muted-foreground">
                {user.email}
              </p>
            </div>
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />

        {/* Quick Actions */}
        <DropdownMenuItem onClick={() => navigate('/dashboard')}>
          <Activity className="mr-2 h-4 w-4" />
          Dashboard
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        {/* Data & Models Section */}
        <DropdownMenuLabel className="text-xs text-muted-foreground uppercase tracking-wider">
          Data & Models
        </DropdownMenuLabel>

        <DropdownMenuItem onClick={() => navigate('/ml-models')}>
          <Brain className="mr-2 h-4 w-4" />
          ML Models
          <ChevronRight className="ml-auto h-4 w-4 text-muted-foreground" />
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => navigate('/data')}>
          <Database className="mr-2 h-4 w-4" />
          My Data
          <ChevronRight className="ml-auto h-4 w-4 text-muted-foreground" />
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        {/* Connections Section */}
        <DropdownMenuLabel className="text-xs text-muted-foreground uppercase tracking-wider">
          Connections
        </DropdownMenuLabel>

        <DropdownMenuItem onClick={() => navigate('/settings?tab=datasource')}>
          <Link2 className="mr-2 h-4 w-4" />
          Gluroo Connection
          <ChevronRight className="ml-auto h-4 w-4 text-muted-foreground" />
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => navigate('/settings?tab=sharing')}>
          <Share2 className="mr-2 h-4 w-4" />
          Sharing & Access
          <ChevronRight className="ml-auto h-4 w-4 text-muted-foreground" />
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => navigate('/settings?tab=integrations')}>
          <Plug className="mr-2 h-4 w-4" />
          Integrations
          <ChevronRight className="ml-auto h-4 w-4 text-muted-foreground" />
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        {/* Settings Section */}
        <DropdownMenuLabel className="text-xs text-muted-foreground uppercase tracking-wider">
          Preferences
        </DropdownMenuLabel>

        <DropdownMenuItem onClick={() => navigate('/settings?tab=insulin')}>
          <Settings className="mr-2 h-4 w-4" />
          Insulin Settings
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => navigate('/settings?tab=alerts')}>
          <Bell className="mr-2 h-4 w-4" />
          Alerts & Notifications
        </DropdownMenuItem>

        {/* Dark Mode Toggle */}
        <DropdownMenuItem onSelect={(e) => e.preventDefault()} className="cursor-pointer">
          <div className="flex w-full items-center justify-between">
            <div className="flex items-center">
              {darkMode ? (
                <Moon className="mr-2 h-4 w-4" />
              ) : (
                <Sun className="mr-2 h-4 w-4" />
              )}
              <span>Dark Mode</span>
            </div>
            <Switch checked={darkMode} onCheckedChange={toggleDarkMode} />
          </div>
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        {/* Help & Account */}
        <DropdownMenuItem onClick={() => navigate('/settings?tab=profile')}>
          <User className="mr-2 h-4 w-4" />
          Account Settings
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => navigate('/settings?tab=security')}>
          <Shield className="mr-2 h-4 w-4" />
          Privacy & Security
        </DropdownMenuItem>

        {user.isAdmin && (
          <DropdownMenuItem onClick={() => navigate('/admin')}>
            <Shield className="mr-2 h-4 w-4 text-purple-500" />
            <span className="text-purple-400">Admin Panel</span>
          </DropdownMenuItem>
        )}

        <DropdownMenuItem onClick={() => window.open('https://github.com/jadericdawson/T1D-AI/issues', '_blank')}>
          <HelpCircle className="mr-2 h-4 w-4" />
          Help & Support
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => window.open('https://github.com/jadericdawson/T1D-AI/issues/new', '_blank')}>
          <MessageSquare className="mr-2 h-4 w-4" />
          Send Feedback
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        {/* Logout */}
        <DropdownMenuItem onClick={handleLogout} className="text-destructive focus:text-destructive">
          <LogOut className="mr-2 h-4 w-4" />
          Sign Out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default UserMenu
