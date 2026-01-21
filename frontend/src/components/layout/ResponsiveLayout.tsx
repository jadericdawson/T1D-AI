/**
 * Responsive Layout Component
 * Wraps pages with appropriate header/navigation based on screen size
 */
import { ReactNode } from 'react'
import { useIsMobile } from '@/hooks/useResponsive'
import { MobileHeader } from './MobileHeader'
import { MobileBottomNav } from './MobileBottomNav'
import { DesktopHeader } from './DesktopHeader'
import { cn } from '@/lib/utils'

interface ResponsiveLayoutProps {
  children: ReactNode
  title?: string
  connectionStatus?: 'connected' | 'polling' | 'disconnected' | 'error'
  onRefresh?: () => void
  isRefreshing?: boolean
  onQuickLog?: () => void
  className?: string
  /** Hide bottom nav on mobile (e.g., for modal-heavy pages) */
  hideBottomNav?: boolean
}

export function ResponsiveLayout({
  children,
  title,
  connectionStatus = 'polling',
  onRefresh,
  isRefreshing,
  onQuickLog,
  className,
  hideBottomNav = false,
}: ResponsiveLayoutProps) {
  const isMobile = useIsMobile()

  if (isMobile) {
    return (
      <div className="min-h-screen bg-background">
        <MobileHeader title={title} />
        <main
          className={cn(
            'pb-20', // Space for bottom nav
            className
          )}
        >
          {children}
        </main>
        {!hideBottomNav && <MobileBottomNav onQuickLog={onQuickLog} />}
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <DesktopHeader
        connectionStatus={connectionStatus}
        onRefresh={onRefresh}
        isRefreshing={isRefreshing}
      />
      <main className={cn('', className)}>{children}</main>
    </div>
  )
}

export default ResponsiveLayout
