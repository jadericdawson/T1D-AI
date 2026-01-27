# Authentication Error Handling Fix

## Problem

The T1D-AI dashboard was showing empty state instead of redirecting to login when authentication failed:
- Empty glucose chart with "No glucose history"
- Empty activity log with "No recent activity"
- IOB/COB/POB all showing 0
- No error messages shown to user
- All API calls returning 401 Unauthorized but silently failing

## Root Cause

The frontend authentication error handling had a subtle bug:

1. **Backend**: ✅ Working correctly - Returns `401 Unauthorized` for invalid/expired tokens
2. **Axios interceptor**: ✅ Catches 401, attempts refresh, dispatches `auth:unauthorized` event
3. **Event listener**: ✅ Exists in `App.tsx` (lines 42-58) and should redirect to login
4. **React Query**: ❌ Receives rejected promise and silently fails, showing empty state
5. **Error boundary**: ❌ Wasn't detecting auth errors properly

**The issue**: When token refresh failed, the axios interceptor would:
- Dispatch `auth:unauthorized` event ✓
- Return `Promise.reject(error)` ✗

React Query would receive the rejected promise but wouldn't retry (correct), and the UI would show empty state instead of an auth error.

## Solution Implemented

### 1. Improved Axios Interceptor Error Message
**File**: `frontend/src/lib/api.ts`

Changed the error rejection message to be clearer:
```typescript
// Return a rejected promise with a clear message for the UI
return Promise.reject(new Error('Session expired - redirecting to login'))
```

### 2. Enhanced Error Boundary for Auth Errors
**File**: `frontend/src/components/ErrorBoundary.tsx`

Added special handling for authentication errors:
- Detects auth-related error messages (session expired, token, 401, unauthorized)
- Shows a clean "Session Expired" dialog instead of full stack trace
- Provides "Go to Login" button that clears auth and redirects

### 3. Added Error Boundary to Protected Routes
**File**: `frontend/src/App.tsx`

Wrapped all protected route children with ErrorBoundary:
```typescript
return (
  <ErrorBoundary>
    {children}
  </ErrorBoundary>
)
```

### 4. Improved React Query Error Handling
**File**: `frontend/src/hooks/useGlucose.ts`

Added custom retry logic to glucose and treatment queries:
```typescript
retry: (failureCount, error: any) => {
  // Don't retry on auth errors - let the event listener handle redirect
  if (error?.message?.includes('Session expired') || error?.response?.status === 401) {
    console.log('[Hook] Auth error detected, not retrying')
    return false
  }
  return failureCount < 3
}
```

This prevents infinite retries on auth failures and logs the error clearly.

## Files Changed

1. `frontend/src/lib/api.ts` - Improved error message
2. `frontend/src/components/ErrorBoundary.tsx` - Added auth error detection
3. `frontend/src/App.tsx` - Wrapped protected routes with error boundary
4. `frontend/src/hooks/useGlucose.ts` - Added custom retry logic for:
   - `useCurrentGlucose`
   - `useGlucoseHistory`
   - `useRecentTreatments`

## User Experience After Fix

When authentication fails:
1. User sees "Session Expired" dialog with clear message
2. "Go to Login" button clears stale auth data
3. User is redirected to login page
4. No more empty dashboard with confusing "No data" messages

## Testing

Build successful:
```bash
npm run build
✓ built in 51.21s
```

## Deployment

Deploy using the standard deployment script:
```bash
./deploy.sh
```

## Notes

- The auth event listener was already implemented correctly
- The backend authentication is working as expected
- The issue was purely in how the frontend handled auth failures in data fetching hooks
- No backend changes required
