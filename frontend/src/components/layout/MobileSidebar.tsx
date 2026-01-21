/**
 * Mobile Sidebar Component
 * Navigation menu for mobile devices (appears in sheet)
 */
import { Link, useLocation, useNavigate } from 'react-router-dom'
import {
  Brain, Database, Link2, Share2,
  Bell, User, Shield, HelpCircle, LogOut, ChevronRight,
  Gauge, TrendingUp, Syringe, Apple, Moon, Sun
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useAuthStore } from '@/stores/authStore'
import { cn } from '@/lib/utils'
import { useState } from 'react'

interface MobileSidebarProps {
  onNavigate?: () => void
}

interface NavItemProps {
  icon: React.ElementType
  label: string
  href?: string
  onClick?: () => void
  badge?: string
  badgeVariant?: 'default' | 'secondary' | 'destructive' | 'outline'
  active?: boolean
  className?: string
}

function NavItem({
  icon: Icon,
  label,
  href,
  onClick,
  badge,
  badgeVariant = 'secondary',
  active,
  className,
}: NavItemProps) {
  const content = (
    <div
      className={cn(
        'flex items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors',
        active
          ? 'bg-primary/10 text-primary font-medium'
          : 'hover:bg-accent text-muted-foreground hover:text-foreground',
        className
      )}
    >
      <div className="flex items-center gap-3">
        <Icon className="h-5 w-5" />
        <span>{label}</span>
      </div>
      {badge && (
        <Badge variant={badgeVariant} className="text-xs">
          {badge}
        </Badge>
      )}
      {!badge && <ChevronRight className="h-4 w-4 text-muted-foreground" />}
    </div>
  )

  if (href) {
    return (
      <Link to={href} onClick={onClick}>
        {content}
      </Link>
    )
  }

  return (
    <button onClick={onClick} className="w-full text-left">
      {content}
    </button>
  )
}

function NavSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <h3 className="px-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      <div className="space-y-0.5">{children}</div>
    </div>
  )
}

export function MobileSidebar({ onNavigate }: MobileSidebarProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, logout } = useAuthStore()
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

  const handleNavigate = (path: string) => {
    navigate(path)
    onNavigate?.()
  }

  const handleLogout = () => {
    logout()
    navigate('/login')
    onNavigate?.()
  }

  const isActive = (path: string) => location.pathname === path

  // Get user initials
  const initials = user?.displayName
    ? user.displayName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : user?.email?.slice(0, 2).toUpperCase() || 'U'

  return (
    <ScrollArea className="h-[calc(100vh-64px)]">
      <div className="flex flex-col gap-4 p-4">
        {/* User Card */}
        <div className="rounded-lg bg-accent/50 p-4">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
              <span className="text-lg font-semibold text-primary">{initials}</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">
                {user?.displayName || 'T1D User'}
              </p>
              <p className="text-xs text-muted-foreground truncate">
                {user?.email}
              </p>
            </div>
          </div>
        </div>

        {/* Main Navigation */}
        <NavSection title="Main">
          <NavItem
            icon={Gauge}
            label="Dashboard"
            href="/dashboard"
            onClick={onNavigate}
            active={isActive('/dashboard')}
          />
        </NavSection>

        {/* Quick Actions */}
        <NavSection title="Quick Log">
          <div className="flex gap-2 px-3">
            <Button
              variant="outline"
              className="flex-1 h-auto py-3"
              onClick={() => handleNavigate('/dashboard?action=logCarbs')}
            >
              <div className="flex flex-col items-center gap-1">
                <Apple className="h-5 w-5 text-green-500" />
                <span className="text-xs">Carbs</span>
              </div>
            </Button>
            <Button
              variant="outline"
              className="flex-1 h-auto py-3"
              onClick={() => handleNavigate('/dashboard?action=logInsulin')}
            >
              <div className="flex flex-col items-center gap-1">
                <Syringe className="h-5 w-5 text-blue-500" />
                <span className="text-xs">Insulin</span>
              </div>
            </Button>
          </div>
        </NavSection>

        <Separator />

        {/* ML & Data */}
        <NavSection title="ML & Data">
          <NavItem
            icon={Brain}
            label="ML Models"
            onClick={() => handleNavigate('/settings?tab=models')}
            badge="3 Active"
            badgeVariant="outline"
          />
          <NavItem
            icon={TrendingUp}
            label="ISF Learning"
            onClick={() => handleNavigate('/settings?tab=models')}
          />
          <NavItem
            icon={Database}
            label="My Data"
            onClick={() => handleNavigate('/settings?tab=data')}
          />
        </NavSection>

        <Separator />

        {/* Connections */}
        <NavSection title="Connections">
          <NavItem
            icon={Link2}
            label="Gluroo"
            onClick={() => handleNavigate('/settings?tab=datasource')}
            badge="Connected"
            badgeVariant="default"
          />
          <NavItem
            icon={Share2}
            label="Sharing"
            onClick={() => handleNavigate('/settings?tab=sharing')}
          />
        </NavSection>

        <Separator />

        {/* Settings */}
        <NavSection title="Settings">
          <NavItem
            icon={Syringe}
            label="Insulin Settings"
            onClick={() => handleNavigate('/settings?tab=insulin')}
          />
          <NavItem
            icon={Bell}
            label="Alerts"
            onClick={() => handleNavigate('/settings?tab=alerts')}
          />
          <NavItem
            icon={User}
            label="Account"
            onClick={() => handleNavigate('/settings?tab=profile')}
          />
          <NavItem
            icon={Shield}
            label="Privacy & Security"
            onClick={() => handleNavigate('/settings?tab=security')}
          />
        </NavSection>

        {/* Theme Toggle */}
        <div className="rounded-lg border px-3 py-2.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {darkMode ? (
                <Moon className="h-5 w-5 text-muted-foreground" />
              ) : (
                <Sun className="h-5 w-5 text-muted-foreground" />
              )}
              <span className="text-sm">Dark Mode</span>
            </div>
            <Switch checked={darkMode} onCheckedChange={toggleDarkMode} />
          </div>
        </div>

        <Separator />

        {/* Help & Support */}
        <NavSection title="Support">
          <NavItem
            icon={HelpCircle}
            label="Help & Support"
            onClick={() => window.open('https://github.com/jadericdawson/T1D-AI/issues', '_blank')}
          />
        </NavSection>

        {/* Logout */}
        <Button
          variant="ghost"
          className="justify-start text-destructive hover:text-destructive hover:bg-destructive/10"
          onClick={handleLogout}
        >
          <LogOut className="mr-3 h-5 w-5" />
          Sign Out
        </Button>

        {/* Bottom Padding for Safe Area */}
        <div className="h-8" />
      </div>
    </ScrollArea>
  )
}

export default MobileSidebar
