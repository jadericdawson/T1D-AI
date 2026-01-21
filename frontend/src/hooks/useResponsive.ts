/**
 * Responsive breakpoint detection hooks
 * Matches Tailwind CSS breakpoints
 */
import { useState, useEffect } from 'react'

// Tailwind breakpoints
const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
}

interface ResponsiveState {
  isMobile: boolean      // < 768px
  isTablet: boolean      // 768px - 1023px
  isDesktop: boolean     // >= 1024px
  width: number
}

export function useResponsive(): ResponsiveState {
  const [state, setState] = useState<ResponsiveState>(() => {
    const width = typeof window !== 'undefined' ? window.innerWidth : 1024
    return {
      isMobile: width < BREAKPOINTS.md,
      isTablet: width >= BREAKPOINTS.md && width < BREAKPOINTS.lg,
      isDesktop: width >= BREAKPOINTS.lg,
      width,
    }
  })

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth
      setState({
        isMobile: width < BREAKPOINTS.md,
        isTablet: width >= BREAKPOINTS.md && width < BREAKPOINTS.lg,
        isDesktop: width >= BREAKPOINTS.lg,
        width,
      })
    }

    window.addEventListener('resize', handleResize)
    handleResize() // Initial check

    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return state
}

export function useIsMobile(): boolean {
  const { isMobile } = useResponsive()
  return isMobile
}

export function useIsDesktop(): boolean {
  const { isDesktop } = useResponsive()
  return isDesktop
}

export function useIsMobileOrTablet(): boolean {
  const { isMobile, isTablet } = useResponsive()
  return isMobile || isTablet
}
