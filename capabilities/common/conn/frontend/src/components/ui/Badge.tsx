import React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/utils/cn'

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-300',
        secondary: 'border-transparent bg-secondary-100 text-secondary-800 dark:bg-secondary-800 dark:text-secondary-300',
        success: 'border-transparent bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-300',
        warning: 'border-transparent bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-300',
        destructive: 'border-transparent bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-300',
        outline: 'text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-600',
        glow: 'border-transparent bg-primary-500 text-white shadow-glow',
      },
      size: {
        default: 'px-2.5 py-0.5 text-xs',
        sm: 'px-2 py-0.5 text-xs',
        lg: 'px-3 py-1 text-sm',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {
  icon?: React.ReactNode
  dot?: boolean
}

function Badge({ className, variant, size, icon, dot, children, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant, size }), className)} {...props}>
      {dot && (
        <div className="w-2 h-2 bg-current rounded-full mr-1.5" />
      )}
      {icon && (
        <span className={cn('flex-shrink-0', children && 'mr-1')}>
          {icon}
        </span>
      )}
      {children}
    </div>
  )
}

export { Badge, badgeVariants }