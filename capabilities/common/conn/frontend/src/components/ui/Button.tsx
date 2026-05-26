import React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/utils/cn'

const buttonVariants = cva(
  'inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-primary-600 text-white shadow-elegant hover:bg-primary-700 focus-visible:ring-primary-500',
        destructive: 'bg-danger-600 text-white shadow-elegant hover:bg-danger-700 focus-visible:ring-danger-500',
        outline: 'border border-gray-300 bg-white text-gray-700 shadow-sm hover:bg-gray-50 focus-visible:ring-primary-500 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700',
        secondary: 'bg-secondary-100 text-secondary-900 shadow-sm hover:bg-secondary-200 focus-visible:ring-secondary-500 dark:bg-secondary-800 dark:text-secondary-100 dark:hover:bg-secondary-700',
        ghost: 'text-gray-700 hover:bg-gray-100 focus-visible:ring-primary-500 dark:text-gray-300 dark:hover:bg-gray-800',
        link: 'text-primary-600 underline-offset-4 hover:underline focus-visible:ring-primary-500',
        gradient: 'bg-gradient-to-r from-primary-500 to-primary-600 text-white shadow-elegant hover:from-primary-600 hover:to-primary-700 focus-visible:ring-primary-500',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-8 px-3 text-xs',
        lg: 'h-12 px-8 text-base',
        xl: 'h-14 px-10 text-lg',
        icon: 'h-10 w-10',
        'icon-sm': 'h-8 w-8',
        'icon-lg': 'h-12 w-12',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean
  icon?: React.ReactNode
  iconPosition?: 'left' | 'right'
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, loading, icon, iconPosition = 'left', children, disabled, ...props }, ref) => {
    const isDisabled = disabled || loading

    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={isDisabled}
        {...props}
      >
        {loading && (
          <div className="loading-spinner h-4 w-4 mr-2" />
        )}
        {!loading && icon && iconPosition === 'left' && (
          <span className={cn('flex-shrink-0', children && 'mr-2')}>
            {icon}
          </span>
        )}
        {children}
        {!loading && icon && iconPosition === 'right' && (
          <span className={cn('flex-shrink-0', children && 'ml-2')}>
            {icon}
          </span>
        )}
      </button>
    )
  }
)
Button.displayName = 'Button'

export { Button, buttonVariants }