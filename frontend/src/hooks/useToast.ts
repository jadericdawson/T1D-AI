/**
 * Toast Hook with Global Event System
 * Allows toast notifications from anywhere in the app
 */
import { useState, useCallback, useEffect } from 'react'

export interface Toast {
  id: string
  title?: string
  description?: string
  action?: React.ReactNode
  variant?: 'default' | 'destructive' | 'success'
}

interface ToastState {
  toasts: Toast[]
}

let toastIdCounter = 0

// Global event emitter for toasts
type ToastEventListener = (toast: Omit<Toast, 'id'>) => void
const listeners: Set<ToastEventListener> = new Set()

/**
 * Global toast function - can be called from anywhere
 * Usage: toast({ title: 'Success!', description: 'Action completed' })
 */
export function toast(options: Omit<Toast, 'id'>) {
  listeners.forEach(listener => listener(options))
}

/**
 * Hook for components that need to display and manage toasts
 */
export function useToast() {
  const [state, setState] = useState<ToastState>({ toasts: [] })

  const addToast = useCallback(({ title, description, action, variant = 'default' }: Omit<Toast, 'id'>) => {
    const id = `toast-${++toastIdCounter}`
    const newToast: Toast = { id, title, description, action, variant }

    setState((prev) => ({
      toasts: [...prev.toasts, newToast],
    }))

    // Auto dismiss after 5 seconds
    setTimeout(() => {
      setState((prev) => ({
        toasts: prev.toasts.filter((t) => t.id !== id),
      }))
    }, 5000)

    return id
  }, [])

  const dismiss = useCallback((id: string) => {
    setState((prev) => ({
      toasts: prev.toasts.filter((t) => t.id !== id),
    }))
  }, [])

  // Subscribe to global toast events
  useEffect(() => {
    listeners.add(addToast)
    return () => {
      listeners.delete(addToast)
    }
  }, [addToast])

  return {
    toasts: state.toasts,
    toast: addToast,
    dismiss,
  }
}
