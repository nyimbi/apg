import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import {
  EyeIcon,
  EyeSlashIcon,
  SparklesIcon,
  ShieldCheckIcon,
  BoltIcon,
  GlobeAltIcon
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { useAuth } from '@/providers/AuthProvider'
import { cn } from '@/utils/cn'

// Form validation schema
const loginSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(1, 'Password is required'),
  remember: z.boolean().optional()
})

type LoginForm = z.infer<typeof loginSchema>

export function LoginPage() {
  const [showPassword, setShowPassword] = useState(false)
  const { login, isLoading } = useAuth()

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting }
  } = useForm<LoginForm>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      remember: false
    }
  })

  const onSubmit = async (data: LoginForm) => {
    try {
      await login(data.email, data.password)
    } catch (error) {
      // Error is handled by the auth context
    }
  }

  const features = [
    {
      icon: <ShieldCheckIcon className="h-6 w-6" />,
      title: 'Enterprise Security',
      description: 'Bank-level encryption and security controls'
    },
    {
      icon: <BoltIcon className="h-6 w-6" />,
      title: 'Lightning Fast',
      description: 'Process millions of records in seconds'
    },
    {
      icon: <GlobeAltIcon className="h-6 w-6" />,
      title: 'Global Scale',
      description: 'Connect data sources worldwide'
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800 flex">
      {/* Left Side - Branding */}
      <div className="hidden lg:flex lg:flex-1 lg:flex-col lg:justify-center lg:px-12">
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-md"
        >
          {/* Logo and Title */}
          <div className="flex items-center space-x-3 mb-8">
            <div className="p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl shadow-glow">
              <SparklesIcon className="h-10 w-10 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                APG Platform
              </h1>
              <p className="text-primary-600 dark:text-primary-400 font-medium">
                Connection Management
              </p>
            </div>
          </div>

          {/* Description */}
          <div className="space-y-6 mb-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                The Future of Data Integration
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed">
                Connect, transform, and orchestrate your data with our revolutionary
                platform that's 10x better than traditional solutions.
              </p>
            </div>

            {/* Features */}
            <div className="space-y-4">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  className="flex items-start space-x-4"
                >
                  <div className="p-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg text-primary-600 dark:text-primary-400">
                    {feature.icon}
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 text-sm">
                      {feature.description}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 pt-6 border-t border-gray-200 dark:border-gray-700">
            {[
              { label: 'Data Sources', value: '500+' },
              { label: 'Records/sec', value: '100K+' },
              { label: 'Uptime', value: '99.9%' }
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.6 + index * 0.1 }}
                className="text-center"
              >
                <p className="text-2xl font-bold text-primary-600 dark:text-primary-400">
                  {stat.value}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {stat.label}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Right Side - Login Form */}
      <div className="flex flex-1 flex-col justify-center px-6 py-12 lg:px-8 lg:flex-none lg:w-96">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mx-auto w-full max-w-sm"
        >
          {/* Mobile Logo */}
          <div className="lg:hidden flex items-center justify-center mb-8">
            <div className="p-2 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg shadow-glow">
              <SparklesIcon className="h-8 w-8 text-white" />
            </div>
          </div>

          <Card>
            <CardHeader className="text-center">
              <CardTitle className="text-2xl font-bold">
                Welcome Back
              </CardTitle>
              <p className="text-gray-600 dark:text-gray-400">
                Sign in to your APG account
              </p>
            </CardHeader>

            <CardContent>
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                {/* Email Field */}
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Email Address
                  </label>
                  <input
                    {...register('email')}
                    type="email"
                    id="email"
                    autoComplete="email"
                    placeholder="you@company.com"
                    className={cn(
                      'input',
                      errors.email && 'input-error'
                    )}
                  />
                  {errors.email && (
                    <p className="text-danger-600 dark:text-danger-400 text-sm mt-1">
                      {errors.email.message}
                    </p>
                  )}
                </div>

                {/* Password Field */}
                <div>
                  <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Password
                  </label>
                  <div className="relative">
                    <input
                      {...register('password')}
                      type={showPassword ? 'text' : 'password'}
                      id="password"
                      autoComplete="current-password"
                      placeholder="Enter your password"
                      className={cn(
                        'input pr-10',
                        errors.password && 'input-error'
                      )}
                    />
                    <button
                      type="button"
                      className="absolute inset-y-0 right-0 pr-3 flex items-center"
                      onClick={() => setShowPassword(!showPassword)}
                    >
                      {showPassword ? (
                        <EyeSlashIcon className="h-5 w-5 text-gray-400" />
                      ) : (
                        <EyeIcon className="h-5 w-5 text-gray-400" />
                      )}
                    </button>
                  </div>
                  {errors.password && (
                    <p className="text-danger-600 dark:text-danger-400 text-sm mt-1">
                      {errors.password.message}
                    </p>
                  )}
                </div>

                {/* Remember Me & Forgot Password */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <input
                      {...register('remember')}
                      id="remember"
                      type="checkbox"
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                    <label htmlFor="remember" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                      Remember me
                    </label>
                  </div>
                  <button
                    type="button"
                    className="text-sm text-primary-600 hover:text-primary-500 dark:text-primary-400 dark:hover:text-primary-300"
                  >
                    Forgot password?
                  </button>
                </div>

                {/* Submit Button */}
                <Button
                  type="submit"
                  className="w-full"
                  size="lg"
                  loading={isSubmitting || isLoading}
                  disabled={isSubmitting || isLoading}
                >
                  Sign In
                </Button>

                {/* Demo Credentials */}
                <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    <strong>Demo Credentials:</strong>
                  </p>
                  <div className="text-sm space-y-1">
                    <p><strong>Email:</strong> admin@datacraft.co.ke</p>
                    <p><strong>Password:</strong> admin123</p>
                  </div>
                </div>
              </form>
            </CardContent>
          </Card>

          {/* Footer */}
          <div className="mt-8 text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Don't have an account?{' '}
              <button className="text-primary-600 hover:text-primary-500 dark:text-primary-400 dark:hover:text-primary-300 font-medium">
                Contact your administrator
              </button>
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  )
}