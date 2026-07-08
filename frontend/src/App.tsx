import { useCallback, useEffect, useState } from 'react'
import { getCurrentUser, logout } from './api/auth'
import { getAccessToken } from './api/client'
import AuthScreen from './components/AuthScreen'
import ChatLayout from './components/ChatLayout'

type AuthStatus = 'checking' | 'authenticated' | 'unauthenticated'

function App() {
  const [authStatus, setAuthStatus] = useState<AuthStatus>(() =>
    getAccessToken() ? 'checking' : 'unauthenticated',
  )

  const verifyCurrentUser = useCallback(async () => {
    if (!getAccessToken()) {
      setAuthStatus('unauthenticated')
      return
    }

    setAuthStatus('checking')

    try {
      await getCurrentUser()
      setAuthStatus('authenticated')
    } catch {
      logout()
      setAuthStatus('unauthenticated')
    }
  }, [])

  const handleLogout = useCallback(() => {
    logout()
    setAuthStatus('unauthenticated')
  }, [])

  useEffect(() => {
    let isMounted = true

    const verifyStoredToken = async () => {
      if (!getAccessToken()) {
        if (isMounted) {
          setAuthStatus('unauthenticated')
        }
        return
      }

      try {
        await getCurrentUser()
        if (isMounted) {
          setAuthStatus('authenticated')
        }
      } catch {
        logout()
        if (isMounted) {
          setAuthStatus('unauthenticated')
        }
      }
    }

    void verifyStoredToken()

    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const handleUnauthorized = () => {
      logout()
      setAuthStatus('unauthenticated')
    }

    window.addEventListener('auth:unauthorized', handleUnauthorized)
    return () => {
      window.removeEventListener('auth:unauthorized', handleUnauthorized)
    }
  }, [])

  if (authStatus === 'checking') {
    return (
      <main className="flex min-h-screen items-center justify-center bg-gray-50 px-4 text-gray-950">
        <div className="rounded-3xl border border-gray-200 bg-white px-6 py-5 text-sm font-medium text-gray-600 shadow-sm">
          Đang kiểm tra phiên đăng nhập...
        </div>
      </main>
    )
  }

  if (authStatus === 'unauthenticated') {
    return <AuthScreen onAuthenticated={verifyCurrentUser} />
  }

  return <ChatLayout onLogout={handleLogout} />
}

export default App
