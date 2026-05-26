import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  HomeIcon,
  LinkIcon,
  WorkflowIcon,
  BarChart3Icon,
  SettingsIcon,
  BrainIcon,
  DatabaseIcon,
  GitBranchIcon,
  SearchIcon,
  PlayIcon,
  PauseIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  SparklesIcon,
} from 'lucide-react'
import { cn } from '@/utils/cn'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'

interface SidebarProps {
  isCollapsed: boolean
  onToggle: () => void
}

const navigationItems = [
  {
    name: 'Dashboard',
    href: '/',
    icon: HomeIcon,
    description: 'Overview and analytics'
  },
  {
    name: 'Connections',
    href: '/connections',
    icon: LinkIcon,
    description: 'Manage data connections',
    badge: 'Hot'
  },
  {
    name: 'Data Flows',
    href: '/flows',
    icon: WorkflowIcon,
    description: 'Design and execute flows'
  },
  {
    name: 'Visual Designer',
    href: '/designer',
    icon: BrainIcon,
    description: 'Drag & drop flow builder',
    badge: 'New'
  },
  {
    name: 'Data Lineage',
    href: '/lineage',
    icon: GitBranchIcon,
    description: 'Track data relationships'
  },
  {
    name: 'Singer Taps',
    href: '/taps',
    icon: DatabaseIcon,
    description: 'Manage Singer.io taps'
  },
  {
    name: 'Analytics',
    href: '/analytics',
    icon: BarChart3Icon,
    description: 'Performance insights'
  },
  {
    name: 'Search',
    href: '/search',
    icon: SearchIcon,
    description: 'Find connections & flows'
  },
]

const bottomNavItems = [
  {
    name: 'Settings',
    href: '/settings',
    icon: SettingsIcon,
    description: 'Application settings'
  }
]

export function Sidebar({ isCollapsed, onToggle }: SidebarProps) {
  const location = useLocation()

  const sidebarVariants = {
    expanded: { width: 280 },
    collapsed: { width: 80 }
  }

  const textVariants = {
    expanded: { opacity: 1, x: 0 },
    collapsed: { opacity: 0, x: -10 }
  }

  return (
    <motion.div
      initial={false}
      animate={isCollapsed ? 'collapsed' : 'expanded'}
      variants={sidebarVariants}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="relative flex flex-col h-full bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 shadow-elegant"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
        <AnimatePresence>
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="flex items-center space-x-2"
            >
              <div className="p-2 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg shadow-glow">
                <SparklesIcon className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">APG</h1>
                <p className="text-xs text-gray-600 dark:text-gray-400">Connection Hub</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onToggle}
          className="flex-shrink-0"
        >
          {isCollapsed ? (
            <ChevronRightIcon className="h-4 w-4" />
          ) : (
            <ChevronLeftIcon className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto scrollbar-hide">
        {navigationItems.map((item) => {
          const isActive = location.pathname === item.href
          const Icon = item.icon

          return (
            <NavLink
              key={item.name}
              to={item.href}
              className={cn(
                'group flex items-center p-3 rounded-lg transition-all duration-200 relative',
                isActive
                  ? 'bg-primary-100 text-primary-900 dark:bg-primary-900/30 dark:text-primary-300'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-300'
              )}
              title={isCollapsed ? item.name : undefined}
            >
              <Icon className={cn(
                'h-5 w-5 flex-shrink-0 transition-colors duration-200',
                isActive ? 'text-primary-600 dark:text-primary-400' : ''
              )} />

              <AnimatePresence>
                {!isCollapsed && (
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.2 }}
                    className="ml-3 flex-1 min-w-0"
                  >
                    <div className="flex items-center justify-between">
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate">
                          {item.name}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-500 truncate">
                          {item.description}
                        </p>
                      </div>
                      {item.badge && (
                        <Badge
                          variant={item.badge === 'New' ? 'success' : 'warning'}
                          size="sm"
                          className="ml-2"
                        >
                          {item.badge}
                        </Badge>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Active indicator */}
              {isActive && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute right-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-primary-600 rounded-l-full"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
            </NavLink>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-800 space-y-2">
        {bottomNavItems.map((item) => {
          const isActive = location.pathname === item.href
          const Icon = item.icon

          return (
            <NavLink
              key={item.name}
              to={item.href}
              className={cn(
                'group flex items-center p-3 rounded-lg transition-all duration-200',
                isActive
                  ? 'bg-primary-100 text-primary-900 dark:bg-primary-900/30 dark:text-primary-300'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-300'
              )}
              title={isCollapsed ? item.name : undefined}
            >
              <Icon className="h-5 w-5 flex-shrink-0" />

              <AnimatePresence>
                {!isCollapsed && (
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.2 }}
                    className="ml-3"
                  >
                    <p className="text-sm font-medium">{item.name}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-500">
                      {item.description}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </NavLink>
          )
        })}
      </div>

      {/* Collapse indicator for collapsed state */}
      {isCollapsed && (
        <div className="absolute left-1/2 bottom-4 transform -translate-x-1/2">
          <div className="w-8 h-1 bg-gray-300 dark:bg-gray-600 rounded-full" />
        </div>
      )}
    </motion.div>
  )
}