import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  SearchIcon,
  BellIcon,
  UserIcon,
  SunIcon,
  MoonIcon,
  ComputerDesktopIcon,
  ChevronDownIcon,
  ActivityIcon,
  LogOutIcon,
  SettingsIcon,
} from '@heroicons/react/24/outline'
import { Menu, Transition } from '@headlessui/react'
import { useTheme } from '@/providers/ThemeProvider'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { cn } from '@/utils/cn'

interface HeaderProps {
  title: string
  subtitle?: string
  actions?: React.ReactNode
}

export function Header({ title, subtitle, actions }: HeaderProps) {
  const { theme, setTheme, currentTheme } = useTheme()

  const themeOptions = [
    { value: 'light', label: 'Light', icon: SunIcon },
    { value: 'dark', label: 'Dark', icon: MoonIcon },
    { value: 'system', label: 'System', icon: ComputerDesktopIcon },
  ]

  const notifications = [
    { id: 1, title: 'Flow completed successfully', time: '2 min ago', type: 'success' },
    { id: 2, title: 'Connection health check failed', time: '5 min ago', type: 'warning' },
    { id: 3, title: 'New Singer tap available', time: '1 hour ago', type: 'info' },
  ]

  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 shadow-elegant"
    >
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Title Section */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-4">
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  {title}
                </h1>
                {subtitle && (
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {subtitle}
                  </p>
                )}
              </div>

              {/* Status Indicators */}
              <div className="hidden lg:flex items-center space-x-3">
                <Badge variant="success" dot>
                  System Healthy
                </Badge>
                <Badge variant="default">
                  <ActivityIcon className="h-3 w-3 mr-1" />
                  12 Active Flows
                </Badge>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-4">
            {/* Search */}
            <div className="hidden md:flex relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <SearchIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                placeholder="Search connections, flows..."
                className="input pl-10 pr-4 py-2 w-64 text-sm"
              />
              <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                <kbd className="px-2 py-1 text-xs text-gray-500 bg-gray-100 dark:bg-gray-800 rounded border">
                  ⌘K
                </kbd>
              </div>
            </div>

            {/* Mobile Search Button */}
            <Button variant="ghost" size="icon" className="md:hidden">
              <SearchIcon className="h-5 w-5" />
            </Button>

            {/* Notifications */}
            <Menu as="div" className="relative">
              <Menu.Button as={Button} variant="ghost" size="icon" className="relative">
                <BellIcon className="h-5 w-5" />
                {notifications.length > 0 && (
                  <span className="absolute top-0 right-0 h-2 w-2 bg-danger-500 rounded-full transform translate-x-1 -translate-y-1" />
                )}
              </Menu.Button>

              <Transition
                as={React.Fragment}
                enter="transition ease-out duration-100"
                enterFrom="transform opacity-0 scale-95"
                enterTo="transform opacity-100 scale-100"
                leave="transition ease-in duration-75"
                leaveFrom="transform opacity-100 scale-100"
                leaveTo="transform opacity-0 scale-95"
              >
                <Menu.Items className="absolute right-0 z-50 mt-2 w-80 origin-top-right bg-white dark:bg-gray-800 rounded-lg shadow-elegant-lg border border-gray-200 dark:border-gray-700 focus:outline-none">
                  <div className="p-4">
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                      Notifications
                    </h3>
                    <div className="space-y-3">
                      {notifications.map((notification) => (
                        <Menu.Item key={notification.id}>
                          <div className="flex items-start space-x-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                            <div className={cn(
                              'w-2 h-2 rounded-full mt-2 flex-shrink-0',
                              notification.type === 'success' && 'bg-success-500',
                              notification.type === 'warning' && 'bg-warning-500',
                              notification.type === 'info' && 'bg-primary-500'
                            )} />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-gray-900 dark:text-white">
                                {notification.title}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400">
                                {notification.time}
                              </p>
                            </div>
                          </div>
                        </Menu.Item>
                      ))}
                    </div>
                    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                      <Button variant="outline" size="sm" className="w-full">
                        View all notifications
                      </Button>
                    </div>
                  </div>
                </Menu.Items>
              </Transition>
            </Menu>

            {/* Theme Selector */}
            <Menu as="div" className="relative">
              <Menu.Button as={Button} variant="ghost" size="icon">
                {currentTheme === 'dark' ? (
                  <MoonIcon className="h-5 w-5" />
                ) : (
                  <SunIcon className="h-5 w-5" />
                )}
              </Menu.Button>

              <Transition
                as={React.Fragment}
                enter="transition ease-out duration-100"
                enterFrom="transform opacity-0 scale-95"
                enterTo="transform opacity-100 scale-100"
                leave="transition ease-in duration-75"
                leaveFrom="transform opacity-100 scale-100"
                leaveTo="transform opacity-0 scale-95"
              >
                <Menu.Items className="absolute right-0 z-50 mt-2 w-48 origin-top-right bg-white dark:bg-gray-800 rounded-lg shadow-elegant-lg border border-gray-200 dark:border-gray-700 focus:outline-none">
                  <div className="p-2">
                    {themeOptions.map((option) => {
                      const Icon = option.icon
                      return (
                        <Menu.Item key={option.value}>
                          {({ active }) => (
                            <button
                              onClick={() => setTheme(option.value as any)}
                              className={cn(
                                'flex items-center space-x-3 w-full px-3 py-2 text-sm rounded-md transition-colors',
                                active && 'bg-gray-100 dark:bg-gray-700',
                                theme === option.value && 'text-primary-600 dark:text-primary-400'
                              )}
                            >
                              <Icon className="h-4 w-4" />
                              <span>{option.label}</span>
                              {theme === option.value && (
                                <div className="w-2 h-2 bg-primary-500 rounded-full ml-auto" />
                              )}
                            </button>
                          )}
                        </Menu.Item>
                      )
                    })}
                  </div>
                </Menu.Items>
              </Transition>
            </Menu>

            {/* User Menu */}
            <Menu as="div" className="relative">
              <Menu.Button className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center">
                  <UserIcon className="h-5 w-5 text-white" />
                </div>
                <div className="hidden lg:block text-left">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    Admin User
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    admin@company.com
                  </p>
                </div>
                <ChevronDownIcon className="hidden lg:block h-4 w-4 text-gray-400" />
              </Menu.Button>

              <Transition
                as={React.Fragment}
                enter="transition ease-out duration-100"
                enterFrom="transform opacity-0 scale-95"
                enterTo="transform opacity-100 scale-100"
                leave="transition ease-in duration-75"
                leaveFrom="transform opacity-100 scale-100"
                leaveTo="transform opacity-0 scale-95"
              >
                <Menu.Items className="absolute right-0 z-50 mt-2 w-56 origin-top-right bg-white dark:bg-gray-800 rounded-lg shadow-elegant-lg border border-gray-200 dark:border-gray-700 focus:outline-none">
                  <div className="p-2">
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          className={cn(
                            'flex items-center space-x-3 w-full px-3 py-2 text-sm rounded-md transition-colors',
                            active && 'bg-gray-100 dark:bg-gray-700'
                          )}
                        >
                          <SettingsIcon className="h-4 w-4" />
                          <span>Settings</span>
                        </button>
                      )}
                    </Menu.Item>
                    <div className="my-1 border-t border-gray-200 dark:border-gray-700" />
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          className={cn(
                            'flex items-center space-x-3 w-full px-3 py-2 text-sm rounded-md transition-colors text-danger-600 dark:text-danger-400',
                            active && 'bg-gray-100 dark:bg-gray-700'
                          )}
                        >
                          <LogOutIcon className="h-4 w-4" />
                          <span>Sign out</span>
                        </button>
                      )}
                    </Menu.Item>
                  </div>
                </Menu.Items>
              </Transition>
            </Menu>

            {/* Custom Actions */}
            {actions && (
              <div className="flex items-center space-x-2">
                {actions}
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.header>
  )
}