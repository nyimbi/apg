import React, { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import apiClient, { AuthTokens } from '@/api/client'
import toast from 'react-hot-toast'

// Types
export interface User {
  id: string
  email: string
  name: string
  role: string
  permissions: string[]
  avatar_url?: string
  created_at: string
  last_login?: string
  preferences: {
    theme: 'light' | 'dark' | 'system'
    notifications_enabled: boolean
    default_view: string
  }
}

export interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  permissions: string[]
}

export interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  updateUser: (updates: Partial<User>) => Promise<void>
  hasPermission: (permission: string) => boolean
  hasAnyPermission: (permissions: string[]) => boolean
  refreshUser: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Auth Provider Component
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const navigate = useNavigate()
  const location = useLocation()

  const isAuthenticated = !!user
  const permissions = user?.permissions || []

  // Initialize auth state on mount
  useEffect(() => {
    initializeAuth()
  }, [])

  const initializeAuth = async () => {
    try {
      if (apiClient.isAuthenticated()) {
        const userData = await apiClient.getCurrentUser()
        setUser(userData)
      }
    } catch (error) {
      // Token might be invalid, clear it
      await apiClient.logout()
    } finally {
      setIsLoading(false)
    }
  }

  const login = useCallback(async (email: string, password: string) => {
    setIsLoading(true)
    try {
      const tokens = await apiClient.login(email, password)
      const userData = await apiClient.getCurrentUser()

      setUser(userData)

      // Navigate to intended destination or dashboard
      const intendedPath = location.state?.from?.pathname || '/'
      navigate(intendedPath, { replace: true })

    } catch (error: any) {
      toast.error(error.message || 'Login failed')
      throw error
    } finally {
      setIsLoading(false)
    }
  }, [navigate, location.state])

  const logout = useCallback(async () => {
    setIsLoading(true)
    try {
      await apiClient.logout()
      setUser(null)
      navigate('/login', { replace: true })
    } catch (error: any) {
      console.error('Logout error:', error)
      // Even if logout fails, clear local state
      setUser(null)
      navigate('/login', { replace: true })
    } finally {
      setIsLoading(false)
    }
  }, [navigate])

  const updateUser = useCallback(async (updates: Partial<User>) => {
    try {
      const updatedUser = await apiClient.put('/auth/profile', updates)
      setUser(updatedUser)
      toast.success('Profile updated successfully')
    } catch (error: any) {
      toast.error(error.message || 'Failed to update profile')
      throw error
    }
  }, [])

  const refreshUser = useCallback(async () => {
    try {
      if (apiClient.isAuthenticated()) {
        const userData = await apiClient.getCurrentUser()
        setUser(userData)
      }
    } catch (error) {
      console.error('Failed to refresh user:', error)
      // If refresh fails, user might need to re-login
      await logout()
    }
  }, [logout])

  const hasPermission = useCallback((permission: string): boolean => {
    if (!user) return false
    return permissions.includes(permission) || permissions.includes('*')
  }, [permissions, user])

  const hasAnyPermission = useCallback((requiredPermissions: string[]): boolean => {
    if (!user) return false
    return requiredPermissions.some(permission => hasPermission(permission))
  }, [hasPermission, user])

  const contextValue: AuthContextType = {
    user,
    isAuthenticated,
    isLoading,
    permissions,
    login,
    logout,
    updateUser,
    hasPermission,
    hasAnyPermission,
    refreshUser,
  }

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  )
}

// Custom hook to use auth context
export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// Protected Route Component
export function ProtectedRoute({
  children,
  permissions,
  fallback
}: {
  children: React.ReactNode
  permissions?: string[]
  fallback?: React.ReactNode
}) {
  const { isAuthenticated, isLoading, hasAnyPermission } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      navigate('/login', {
        state: { from: location },
        replace: true
      })
    }
  }, [isAuthenticated, isLoading, navigate, location])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="loading-spinner h-8 w-8" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return null
  }

  if (permissions && !hasAnyPermission(permissions)) {
    return fallback || (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-2">
            Access Denied
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            You don't have permission to access this page.
          </p>
        </div>
      </div>
    )
  }

  return <>{children}</>
}

// Permission Guard Component
export function PermissionGuard({
  permission,
  permissions,
  children,
  fallback,
  requireAll = false
}: {
  permission?: string
  permissions?: string[]
  children: React.ReactNode
  fallback?: React.ReactNode
  requireAll?: boolean
}) {
  const { hasPermission, hasAnyPermission } = useAuth()

  let hasAccess = false

  if (permission) {
    hasAccess = hasPermission(permission)
  } else if (permissions) {
    if (requireAll) {
      hasAccess = permissions.every(p => hasPermission(p))
    } else {
      hasAccess = hasAnyPermission(permissions)
    }
  }

  if (!hasAccess) {
    return fallback || null
  }

  return <>{children}</>
}

export default AuthProvider