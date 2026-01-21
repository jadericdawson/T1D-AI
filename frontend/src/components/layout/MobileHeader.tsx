/**
 * Mobile Header Component
 * Sticky header for mobile devices with menu and user actions
 */
import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Menu, Activity, Bell } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet'
import { UserMenu } from './UserMenu'
import { MobileSidebar } from './MobileSidebar'
import { useAuthStore } from '@/stores/authStore'
import { cn } from '@/lib/utils'

interface MobileHeaderProps {
  title?: string
  showBackButton?: boolean
  onBack?: () => void
  className?: string
}

export function MobileHeader({
  title = 'T1D-AI',
  className,
}: MobileHeaderProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()
  const { viewingUser } = useAuthStore()

  // Determine page title from route
  const getPageTitle = () => {
    if (title !== 'T1D-AI') return title
    switch (location.pathname) {
      case '/dashboard':
        return 'Dashboard'
      case '/settings':
        return 'Settings'
      default:
        return 'T1D-AI'
    }
  }

  return (
    <header
      className={cn(
        'sticky top-0 z-50 flex h-14 items-center justify-between border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-4',
        className
      )}
    >
      {/* Left: Menu Button */}
      <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="h-9 w-9">
            <Menu className="h-5 w-5" />
            <span className="sr-only">Open menu</span>
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-[85vw] max-w-[320px] p-0">
          <SheetHeader className="border-b px-4 py-4">
            <SheetTitle className="flex items-center gap-2 text-left">
              <Activity className="h-5 w-5 text-primary" />
              T1D-AI
            </SheetTitle>
          </SheetHeader>
          <MobileSidebar onNavigate={() => setSidebarOpen(false)} />
        </SheetContent>
      </Sheet>

      {/* Center: Title & Viewing Badge */}
      <div className="flex flex-1 items-center justify-center gap-2">
        <Link to="/dashboard" className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary" />
          <span className="font-semibold">{getPageTitle()}</span>
        </Link>
        {viewingUser && (
          <Badge variant="secondary" className="text-xs">
            Viewing {viewingUser.displayName?.split(' ')[0] || 'Shared'}
          </Badge>
        )}
      </div>

      {/* Right: Notifications & User Menu */}
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="icon" className="h-9 w-9 relative">
          <Bell className="h-5 w-5" />
          {/* Notification dot */}
          <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-orange-500" />
          <span className="sr-only">Notifications</span>
        </Button>
        <UserMenu align="end" />
      </div>
    </header>
  )
}

export default MobileHeader
