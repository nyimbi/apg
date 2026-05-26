import React from 'react'
import { motion } from 'framer-motion'
import {
  DatabaseIcon,
  CloudIcon,
  ServerIcon,
  WifiIcon,
  PlayIcon,
  PauseIcon,
  MoreVerticalIcon,
  EditIcon,
  TrashIcon,
  TestTubeIcon,
  ActivityIcon,
} from 'lucide-react'
import { Menu, Transition } from '@headlessui/react'
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { cn } from '@/utils/cn'

interface Connection {
  id: string
  name: string
  description?: string
  connection_type: 'database' | 'api' | 'file' | 'stream'
  status: 'active' | 'inactive' | 'error' | 'testing'
  singer_tap?: string
  singer_target?: string
  last_sync?: string
  last_success?: string
  last_error?: string
  error_count: number
  tags: string[]
  health_score?: number
  records_processed?: number
}

interface ConnectionCardProps {
  connection: Connection
  onEdit?: (connection: Connection) => void
  onDelete?: (connection: Connection) => void
  onTest?: (connection: Connection) => void
  onToggle?: (connection: Connection) => void
  onView?: (connection: Connection) => void
}

const connectionIcons = {
  database: DatabaseIcon,
  api: CloudIcon,
  file: ServerIcon,
  stream: WifiIcon,
}

const statusColors = {
  active: 'success',
  inactive: 'secondary',
  error: 'destructive',
  testing: 'warning',
} as const

export function ConnectionCard({
  connection,
  onEdit,
  onDelete,
  onTest,
  onToggle,
  onView
}: ConnectionCardProps) {
  const Icon = connectionIcons[connection.connection_type] || DatabaseIcon
  const statusColor = statusColors[connection.status]

  const handleAction = (action: () => void) => (e: React.MouseEvent) => {
    e.stopPropagation()
    action()
  }

  const formatLastSync = (dateString?: string) => {
    if (!dateString) return 'Never'
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / (1000 * 60))

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`
    return `${Math.floor(diffMins / 1440)}d ago`
  }

  const getHealthColor = (score?: number) => {
    if (!score) return 'text-gray-400'
    if (score >= 0.8) return 'text-success-600'
    if (score >= 0.6) return 'text-warning-600'
    return 'text-danger-600'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2 }}
      className="group"
    >
      <Card
        hoverable
        className={cn(
          'cursor-pointer transition-all duration-200',
          connection.status === 'error' && 'border-danger-200 dark:border-danger-800',
          connection.status === 'active' && 'border-success-200 dark:border-success-800'
        )}
        onClick={() => onView?.(connection)}
      >
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-3">
              <div className={cn(
                'p-3 rounded-xl shadow-sm',
                connection.status === 'active' && 'bg-success-100 dark:bg-success-900/30',
                connection.status === 'error' && 'bg-danger-100 dark:bg-danger-900/30',
                connection.status === 'inactive' && 'bg-gray-100 dark:bg-gray-800',
                connection.status === 'testing' && 'bg-warning-100 dark:bg-warning-900/30'
              )}>
                <Icon className={cn(
                  'h-6 w-6',
                  connection.status === 'active' && 'text-success-600 dark:text-success-400',
                  connection.status === 'error' && 'text-danger-600 dark:text-danger-400',
                  connection.status === 'inactive' && 'text-gray-600 dark:text-gray-400',
                  connection.status === 'testing' && 'text-warning-600 dark:text-warning-400'
                )} />
              </div>

              <div className="flex-1 min-w-0">
                <CardTitle className="text-base">{connection.name}</CardTitle>
                <p className="text-sm text-gray-600 dark:text-gray-400 truncate">
                  {connection.description || `${connection.connection_type} connection`}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Badge variant={statusColor} size="sm" className="capitalize">
                {connection.status}
              </Badge>

              <Menu as="div" className="relative">
                <Menu.Button
                  as={Button}
                  variant="ghost"
                  size="icon-sm"
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={(e: React.MouseEvent) => e.stopPropagation()}
                >
                  <MoreVerticalIcon className="h-4 w-4" />
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
                      <Menu.Item>
                        {({ active }) => (
                          <button
                            onClick={handleAction(() => onTest?.(connection))}
                            className={cn(
                              'flex items-center space-x-3 w-full px-3 py-2 text-sm rounded-md transition-colors',
                              active && 'bg-gray-100 dark:bg-gray-700'
                            )}
                          >
                            <TestTubeIcon className="h-4 w-4" />
                            <span>Test Connection</span>
                          </button>
                        )}
                      </Menu.Item>
                      <Menu.Item>
                        {({ active }) => (
                          <button
                            onClick={handleAction(() => onToggle?.(connection))}
                            className={cn(
                              'flex items-center space-x-3 w-full px-3 py-2 text-sm rounded-md transition-colors',
                              active && 'bg-gray-100 dark:bg-gray-700'
                            )}
                          >
                            {connection.status === 'active' ? (
                              <>
                                <PauseIcon className="h-4 w-4" />
                                <span>Pause</span>
                              </>
                            ) : (
                              <>
                                <PlayIcon className="h-4 w-4" />
                                <span>Activate</span>
                              </>
                            )}
                          </button>
                        )}
                      </Menu.Item>
                      <Menu.Item>
                        {({ active }) => (
                          <button
                            onClick={handleAction(() => onEdit?.(connection))}
                            className={cn(
                              'flex items-center space-x-3 w-full px-3 py-2 text-sm rounded-md transition-colors',
                              active && 'bg-gray-100 dark:bg-gray-700'
                            )}
                          >
                            <EditIcon className="h-4 w-4" />
                            <span>Edit</span>
                          </button>
                        )}
                      </Menu.Item>
                      <div className="my-1 border-t border-gray-200 dark:border-gray-700" />
                      <Menu.Item>
                        {({ active }) => (
                          <button
                            onClick={handleAction(() => onDelete?.(connection))}
                            className={cn(
                              'flex items-center space-x-3 w-full px-3 py-2 text-sm rounded-md transition-colors text-danger-600 dark:text-danger-400',
                              active && 'bg-gray-100 dark:bg-gray-700'
                            )}
                          >
                            <TrashIcon className="h-4 w-4" />
                            <span>Delete</span>
                          </button>
                        )}
                      </Menu.Item>
                    </div>
                  </Menu.Items>
                </Transition>
              </Menu>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          <div className="space-y-4">
            {/* Health Score */}
            {connection.health_score && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Health Score</span>
                <div className="flex items-center space-x-2">
                  <div className="w-16 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={cn(
                        'h-full transition-all duration-300',
                        connection.health_score >= 0.8 ? 'bg-success-500' :
                        connection.health_score >= 0.6 ? 'bg-warning-500' : 'bg-danger-500'
                      )}
                      style={{ width: `${connection.health_score * 100}%` }}
                    />
                  </div>
                  <span className={cn('text-sm font-medium', getHealthColor(connection.health_score))}>
                    {Math.round(connection.health_score * 100)}%
                  </span>
                </div>
              </div>
            )}

            {/* Connection Details */}
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-600 dark:text-gray-400">Type</span>
                <p className="font-medium capitalize">{connection.connection_type}</p>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Last Sync</span>
                <p className="font-medium">{formatLastSync(connection.last_sync)}</p>
              </div>
              {connection.singer_tap && (
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Singer Tap</span>
                  <p className="font-medium text-xs truncate">{connection.singer_tap}</p>
                </div>
              )}
              {connection.records_processed && (
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Records</span>
                  <p className="font-medium">{connection.records_processed.toLocaleString()}</p>
                </div>
              )}
            </div>

            {/* Error Information */}
            {connection.error_count > 0 && (
              <div className="p-3 bg-danger-50 dark:bg-danger-900/20 rounded-lg border border-danger-200 dark:border-danger-800">
                <div className="flex items-center space-x-2">
                  <ActivityIcon className="h-4 w-4 text-danger-600" />
                  <span className="text-sm font-medium text-danger-800 dark:text-danger-200">
                    {connection.error_count} error{connection.error_count !== 1 ? 's' : ''}
                  </span>
                </div>
                {connection.last_error && (
                  <p className="text-xs text-danger-600 dark:text-danger-400 mt-1 truncate">
                    {connection.last_error}
                  </p>
                )}
              </div>
            )}

            {/* Tags */}
            {connection.tags.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {connection.tags.slice(0, 3).map((tag) => (
                  <Badge key={tag} variant="outline" size="sm">
                    {tag}
                  </Badge>
                ))}
                {connection.tags.length > 3 && (
                  <Badge variant="outline" size="sm">
                    +{connection.tags.length - 3}
                  </Badge>
                )}
              </div>
            )}
          </div>
        </CardContent>

        <CardFooter>
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center space-x-2">
              {connection.status === 'active' && (
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-success-500 rounded-full animate-pulse" />
                  <span className="text-xs text-success-600 dark:text-success-400">Live</span>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleAction(() => onTest?.(connection))}
                icon={<TestTubeIcon className="h-3 w-3" />}
              >
                Test
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleAction(() => onEdit?.(connection))}
                icon={<EditIcon className="h-3 w-3" />}
              >
                Edit
              </Button>
            </div>
          </div>
        </CardFooter>
      </Card>
    </motion.div>
  )
}