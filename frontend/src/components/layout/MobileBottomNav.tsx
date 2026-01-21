/**
 * Mobile Bottom Navigation Component
 * Fixed bottom navigation bar for mobile devices
 */
import { useLocation, useNavigate } from 'react-router-dom'
import { Gauge, Activity, Settings, Brain, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface NavItem {
  icon: React.ElementType
  label: string
  href: string
  exact?: boolean
}

const navItems: NavItem[] = [
  { icon: Gauge, label: 'Home', href: '/dashboard', exact: true },
  { icon: Brain, label: 'Models', href: '/settings?tab=models' },
  { icon: Activity, label: 'Data', href: '/settings?tab=data' },
  { icon: Settings, label: 'Settings', href: '/settings' },
]

interface MobileBottomNavProps {
  onQuickLog?: () => void
  className?: string
}

export function MobileBottomNav({ onQuickLog, className }: MobileBottomNavProps) {
  const location = useLocation()
  const navigate = useNavigate()

  const isActive = (item: NavItem) => {
    if (item.exact) {
      return location.pathname === item.href
    }
    // For settings tabs, check if we're on settings and the tab matches
    if (item.href.includes('?tab=')) {
      const [path, query] = item.href.split('?')
      const params = new URLSearchParams(query)
      const tab = params.get('tab')
      const currentParams = new URLSearchParams(location.search)
      return location.pathname === path && currentParams.get('tab') === tab
    }
    return location.pathname.startsWith(item.href.split('?')[0])
  }

  return (
    <nav
      className={cn(
        'fixed bottom-0 left-0 right-0 z-50 flex h-16 items-center justify-around border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 pb-safe',
        className
      )}
    >
      {/* First two nav items */}
      {navItems.slice(0, 2).map((item) => {
        const Icon = item.icon
        const active = isActive(item)
        return (
          <button
            key={item.href}
            onClick={() => navigate(item.href)}
            className={cn(
              'flex flex-col items-center justify-center gap-1 w-16 h-full transition-colors',
              active ? 'text-primary' : 'text-muted-foreground'
            )}
          >
            <Icon className={cn('h-5 w-5', active && 'scale-110 transition-transform')} />
            <span className={cn('text-[10px]', active && 'font-semibold')}>{item.label}</span>
          </button>
        )
      })}

      {/* Center Quick Log Button */}
      <div className="relative -mt-6">
        <Button
          size="lg"
          className="h-14 w-14 rounded-full shadow-lg bg-primary hover:bg-primary/90"
          onClick={onQuickLog}
        >
          <Plus className="h-6 w-6" />
          <span className="sr-only">Quick Log</span>
        </Button>
      </div>

      {/* Last two nav items */}
      {navItems.slice(2).map((item) => {
        const Icon = item.icon
        const active = isActive(item)
        return (
          <button
            key={item.href}
            onClick={() => navigate(item.href)}
            className={cn(
              'flex flex-col items-center justify-center gap-1 w-16 h-full transition-colors',
              active ? 'text-primary' : 'text-muted-foreground'
            )}
          >
            <Icon className={cn('h-5 w-5', active && 'scale-110 transition-transform')} />
            <span className={cn('text-[10px]', active && 'font-semibold')}>{item.label}</span>
          </button>
        )
      })}
    </nav>
  )
}

export default MobileBottomNav
