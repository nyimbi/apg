/**
 * APG Connection Management - API Client
 *
 * Centralized API client with authentication, error handling, and type safety
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios'
import { toast } from 'react-hot-toast'

// Types
export interface ApiError {
  message: string
  code?: string
  details?: any
  status?: number
}

export interface ApiResponse<T = any> {
  data: T
  message?: string
  success: boolean
}

export interface AuthTokens {
  access_token: string
  refresh_token: string
  token_type: 'bearer'
  expires_in: number
}

// API Client Configuration
class APGApiClient {
  private client: AxiosInstance
  private baseURL: string
  private accessToken: string | null = null
  private refreshToken: string | null = null
  private isRefreshing = false
  private failedQueue: Array<{
    resolve: (token: string) => void
    reject: (error: any) => void
  }> = []

  constructor(baseURL = '/api/v1') {
    this.baseURL = baseURL
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
    this.loadTokensFromStorage()
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        if (this.accessToken) {
          config.headers.Authorization = `Bearer ${this.accessToken}`
        }

        // Add request timestamp for debugging
        config.metadata = { startTime: new Date() }

        return config
      },
      (error) => {
        return Promise.reject(this.handleError(error))
      }
    )

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        // Log response time in development
        if (process.env.NODE_ENV === 'development') {
          const endTime = new Date()
          const startTime = response.config.metadata?.startTime
          if (startTime) {
            const duration = endTime.getTime() - startTime.getTime()
            console.log(`API ${response.config.method?.toUpperCase()} ${response.config.url}: ${duration}ms`)
          }
        }
        return response
      },
      async (error: AxiosError) => {
        const originalRequest = error.config as any

        // Handle 401 errors with token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
          if (this.isRefreshing) {
            return new Promise((resolve, reject) => {
              this.failedQueue.push({ resolve, reject })
            }).then((token) => {
              originalRequest.headers.Authorization = `Bearer ${token}`
              return this.client(originalRequest)
            })
          }

          originalRequest._retry = true
          this.isRefreshing = true

          try {
            const newToken = await this.refreshAccessToken()
            this.processQueue(null, newToken)
            originalRequest.headers.Authorization = `Bearer ${newToken}`
            return this.client(originalRequest)
          } catch (refreshError) {
            this.processQueue(refreshError, null)
            this.logout()
            throw refreshError
          } finally {
            this.isRefreshing = false
          }
        }

        return Promise.reject(this.handleError(error))
      }
    )
  }

  private processQueue(error: any, token: string | null) {
    this.failedQueue.forEach(({ resolve, reject }) => {
      if (error) {
        reject(error)
      } else {
        resolve(token!)
      }
    })

    this.failedQueue = []
  }

  private handleError(error: AxiosError): ApiError {
    const apiError: ApiError = {
      message: 'An unexpected error occurred',
      status: error.response?.status,
    }

    if (error.response?.data) {
      const data = error.response.data as any
      apiError.message = data.detail || data.message || apiError.message
      apiError.code = data.code
      apiError.details = data.details
    } else if (error.request) {
      apiError.message = 'Network error - please check your connection'
    } else {
      apiError.message = error.message
    }

    // Show toast for user-facing errors
    if (error.response?.status !== 401) { // Don't show toast for auth errors
      toast.error(apiError.message)
    }

    return apiError
  }

  private loadTokensFromStorage() {
    if (typeof window !== 'undefined') {
      this.accessToken = localStorage.getItem('apg_access_token')
      this.refreshToken = localStorage.getItem('apg_refresh_token')
    }
  }

  private saveTokensToStorage(tokens: AuthTokens) {
    if (typeof window !== 'undefined') {
      localStorage.setItem('apg_access_token', tokens.access_token)
      localStorage.setItem('apg_refresh_token', tokens.refresh_token)
      localStorage.setItem('apg_token_expires_at',
        (Date.now() + tokens.expires_in * 1000).toString()
      )
    }
    this.accessToken = tokens.access_token
    this.refreshToken = tokens.refresh_token
  }

  private clearTokensFromStorage() {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('apg_access_token')
      localStorage.removeItem('apg_refresh_token')
      localStorage.removeItem('apg_token_expires_at')
    }
    this.accessToken = null
    this.refreshToken = null
  }

  private async refreshAccessToken(): Promise<string> {
    if (!this.refreshToken) {
      throw new Error('No refresh token available')
    }

    const response = await axios.post(`${this.baseURL}/auth/refresh`, {
      refresh_token: this.refreshToken
    })

    const tokens: AuthTokens = response.data
    this.saveTokensToStorage(tokens)
    return tokens.access_token
  }

  // Authentication Methods
  async login(email: string, password: string): Promise<AuthTokens> {
    const response = await this.client.post<AuthTokens>('/auth/login', {
      username: email, // FastAPI OAuth2 uses 'username' field
      password
    })

    this.saveTokensToStorage(response.data)
    toast.success('Successfully logged in!')
    return response.data
  }

  async logout(): Promise<void> {
    try {
      if (this.refreshToken) {
        await this.client.post('/auth/logout', {
          refresh_token: this.refreshToken
        })
      }
    } catch (error) {
      // Ignore errors during logout
    } finally {
      this.clearTokensFromStorage()
      toast.success('Successfully logged out!')
    }
  }

  async getCurrentUser(): Promise<any> {
    const response = await this.client.get('/auth/me')
    return response.data
  }

  // Generic API Methods
  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<ApiResponse<T>>(url, config)
    return response.data.data || response.data
  }

  async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<ApiResponse<T>>(url, data, config)
    return response.data.data || response.data
  }

  async put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<ApiResponse<T>>(url, data, config)
    return response.data.data || response.data
  }

  async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<ApiResponse<T>>(url, config)
    return response.data.data || response.data
  }

  // Utility Methods
  isAuthenticated(): boolean {
    return !!this.accessToken
  }

  getAccessToken(): string | null {
    return this.accessToken
  }

  setAccessToken(token: string) {
    this.accessToken = token
    if (typeof window !== 'undefined') {
      localStorage.setItem('apg_access_token', token)
    }
  }
}

// Create singleton instance
export const apiClient = new APGApiClient()

// Export default instance
export default apiClient