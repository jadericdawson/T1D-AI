/**
 * Error Boundary to catch React errors and display them
 */
import { Component, ErrorInfo, ReactNode } from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    this.setState({ error, errorInfo })
  }

  handleReload = () => {
    window.location.reload()
  }

  handleClearAndReload = () => {
    localStorage.removeItem('t1d-ai-auth')
    window.location.href = '/'
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center p-6 bg-slate-950">
          <div className="max-w-lg w-full bg-slate-900 border border-red-500/30 rounded-lg p-6">
            <div className="flex items-center gap-3 mb-4">
              <AlertCircle className="w-8 h-8 text-red-500" />
              <h1 className="text-xl font-bold text-white">Something went wrong</h1>
            </div>

            <div className="mb-4">
              <p className="text-gray-300 mb-2">Error:</p>
              <pre className="bg-slate-800 p-3 rounded text-sm text-red-400 overflow-auto max-h-32">
                {this.state.error?.message || 'Unknown error'}
              </pre>
            </div>

            {this.state.error?.stack && (
              <div className="mb-4">
                <p className="text-gray-300 mb-2">Stack trace:</p>
                <pre className="bg-slate-800 p-3 rounded text-xs text-gray-400 overflow-auto max-h-48">
                  {this.state.error.stack}
                </pre>
              </div>
            )}

            {this.state.errorInfo?.componentStack && (
              <div className="mb-4">
                <p className="text-gray-300 mb-2">Component stack:</p>
                <pre className="bg-slate-800 p-3 rounded text-xs text-gray-400 overflow-auto max-h-48">
                  {this.state.errorInfo.componentStack}
                </pre>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={this.handleReload}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition"
              >
                <RefreshCw className="w-4 h-4" />
                Reload Page
              </button>
              <button
                onClick={this.handleClearAndReload}
                className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition"
              >
                Clear Auth & Reload
              </button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
