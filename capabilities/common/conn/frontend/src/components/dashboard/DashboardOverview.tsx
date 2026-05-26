import React from 'react'
import { motion } from 'framer-motion'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import {
  DatabaseIcon,
  WorkflowIcon,
  ActivityIcon,
  AlertTriangleIcon,
  CheckCircleIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  ClockIcon,
  ZapIcon,
  ShieldIcon,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { cn } from '@/utils/cn'

interface DashboardData {
  connections: {
    total: number
    active: number
    inactive: number
    error: number
    health_avg: number
  }
  flows: {
    total: number
    running: number
    success_rate: number
    avg_runtime: number
  }
  data_processing: {
    records_today: number
    records_trend: number
    throughput: number
    errors: number
  }
  system_health: {
    cpu_usage: number
    memory_usage: number
    disk_usage: number
    uptime: number
  }
}

// Sample data - in production, this would come from props or API
const mockData: DashboardData = {
  connections: {
    total: 24,
    active: 18,
    inactive: 4,
    error: 2,
    health_avg: 0.87
  },
  flows: {
    total: 47,
    running: 32,
    success_rate: 0.94,
    avg_runtime: 3.2
  },
  data_processing: {
    records_today: 2847593,
    records_trend: 12.5,
    throughput: 1250,
    errors: 23
  },
  system_health: {
    cpu_usage: 0.34,
    memory_usage: 0.67,
    disk_usage: 0.42,
    uptime: 99.8
  }
}

const hourlyData = [
  { time: '00:00', records: 45000, throughput: 750 },
  { time: '04:00', records: 32000, throughput: 533 },
  { time: '08:00', records: 78000, throughput: 1300 },
  { time: '12:00', records: 95000, throughput: 1583 },
  { time: '16:00', records: 87000, throughput: 1450 },
  { time: '20:00', records: 65000, throughput: 1083 },
]

const connectionTypeData = [
  { name: 'Database', value: 12, color: '#3B82F6' },
  { name: 'API', value: 8, color: '#10B981' },
  { name: 'File', value: 3, color: '#F59E0B' },
  { name: 'Stream', value: 1, color: '#EF4444' },
]

const performanceData = [
  { date: 'Mon', success: 94, error: 6, runtime: 2.8 },
  { date: 'Tue', success: 97, error: 3, runtime: 3.1 },
  { date: 'Wed', success: 91, error: 9, runtime: 3.5 },
  { date: 'Thu', success: 95, error: 5, runtime: 2.9 },
  { date: 'Fri', success: 98, error: 2, runtime: 2.6 },
  { date: 'Sat', success: 93, error: 7, runtime: 3.8 },
  { date: 'Sun', success: 96, error: 4, runtime: 3.0 },
]

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  trend?: number
  icon: React.ReactNode
  color?: 'primary' | 'success' | 'warning' | 'danger'
  onClick?: () => void
}

function MetricCard({ title, value, subtitle, trend, icon, color = 'primary', onClick }: MetricCardProps) {
  const colorClasses = {
    primary: 'bg-primary-50 text-primary-600 dark:bg-primary-900/20 dark:text-primary-400',
    success: 'bg-success-50 text-success-600 dark:bg-success-900/20 dark:text-success-400',
    warning: 'bg-warning-50 text-warning-600 dark:bg-warning-900/20 dark:text-warning-400',
    danger: 'bg-danger-50 text-danger-600 dark:bg-danger-900/20 dark:text-danger-400',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2 }}
    >
      <Card hoverable className={cn(onClick && 'cursor-pointer')} onClick={onClick}>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                {title}
              </p>
              <div className="flex items-baseline space-x-2">
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {typeof value === 'number' ? value.toLocaleString() : value}
                </p>
                {trend !== undefined && (
                  <div className={cn(
                    'flex items-center text-sm font-medium',
                    trend >= 0 ? 'text-success-600 dark:text-success-400' : 'text-danger-600 dark:text-danger-400'
                  )}>
                    {trend >= 0 ? (
                      <TrendingUpIcon className="h-4 w-4 mr-1" />
                    ) : (
                      <TrendingDownIcon className="h-4 w-4 mr-1" />
                    )}
                    {Math.abs(trend)}%
                  </div>
                )}
              </div>
              {subtitle && (
                <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                  {subtitle}
                </p>
              )}
            </div>
            <div className={cn('p-3 rounded-xl', colorClasses[color])}>
              {icon}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}

interface StatusIndicatorProps {
  label: string
  value: number
  total?: number
  color?: 'success' | 'warning' | 'danger'
}

function StatusIndicator({ label, value, total, color = 'success' }: StatusIndicatorProps) {
  const percentage = total ? (value / total) * 100 : value * 100
  const colorClasses = {
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500',
  }

  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>
      <div className="flex items-center space-x-2">
        <span className="text-sm font-medium text-gray-900 dark:text-white">
          {total ? `${value}/${total}` : `${value}%`}
        </span>
        <div className="w-16 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className={cn('h-full transition-all duration-300', colorClasses[color])}
            style={{ width: `${Math.min(percentage, 100)}%` }}
          />
        </div>
      </div>
    </div>
  )
}

export function DashboardOverview() {
  const data = mockData

  return (
    <div className="space-y-6">
      {/* Key Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Connections"
          value={data.connections.total}
          subtitle={`${data.connections.active} active`}
          trend={8.2}
          icon={<DatabaseIcon className="h-6 w-6" />}
          color="primary"
        />

        <MetricCard
          title="Active Flows"
          value={data.flows.running}
          subtitle={`${data.flows.total} total flows`}
          trend={5.7}
          icon={<WorkflowIcon className="h-6 w-6" />}
          color="success"
        />

        <MetricCard
          title="Records Processed"
          value={`${(data.data_processing.records_today / 1000000).toFixed(1)}M`}
          subtitle="Today"
          trend={data.data_processing.records_trend}
          icon={<ZapIcon className="h-6 w-6" />}
          color="primary"
        />

        <MetricCard
          title="Success Rate"
          value={`${(data.flows.success_rate * 100).toFixed(1)}%`}
          subtitle="Last 24 hours"
          trend={2.1}
          icon={<CheckCircleIcon className="h-6 w-6" />}
          color="success"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data Processing Timeline */}
        <Card>
          <CardHeader>
            <CardTitle>Data Processing Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={hourlyData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="time"
                    fontSize={12}
                    tick={{ fill: 'currentColor' }}
                  />
                  <YAxis
                    fontSize={12}
                    tick={{ fill: 'currentColor' }}
                    tickFormatter={(value) => `${(value / 1000).toFixed(0)}K`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    formatter={(value: number) => [value.toLocaleString(), 'Records']}
                  />
                  <Area
                    type="monotone"
                    dataKey="records"
                    stroke="#3B82F6"
                    fill="url(#colorRecords)"
                    strokeWidth={2}
                  />
                  <defs>
                    <linearGradient id="colorRecords" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Performance Metrics */}
        <Card>
          <CardHeader>
            <CardTitle>Weekly Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="date"
                    fontSize={12}
                    tick={{ fill: 'currentColor' }}
                  />
                  <YAxis
                    fontSize={12}
                    tick={{ fill: 'currentColor' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Bar dataKey="success" fill="#10B981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="error" fill="#EF4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Status and Details Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* System Health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <ShieldIcon className="h-5 w-5 text-success-600" />
              <span>System Health</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <StatusIndicator
                label="CPU Usage"
                value={data.system_health.cpu_usage * 100}
                color="success"
              />
              <StatusIndicator
                label="Memory Usage"
                value={data.system_health.memory_usage * 100}
                color="warning"
              />
              <StatusIndicator
                label="Disk Usage"
                value={data.system_health.disk_usage * 100}
                color="success"
              />
              <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Uptime</span>
                  <Badge variant="success">
                    {data.system_health.uptime}%
                  </Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Connection Types */}
        <Card>
          <CardHeader>
            <CardTitle>Connection Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={connectionTypeData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={70}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {connectionTypeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-4">
              {connectionTypeData.map((item) => (
                <div key={item.name} className="flex items-center space-x-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {item.name}
                  </span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white ml-auto">
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Recent Activity</span>
              <Button variant="ghost" size="sm">
                View All
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {[
                {
                  type: 'success',
                  message: 'PostgreSQL sync completed',
                  time: '2 minutes ago',
                  icon: <CheckCircleIcon className="h-4 w-4" />,
                },
                {
                  type: 'warning',
                  message: 'API rate limit approaching',
                  time: '5 minutes ago',
                  icon: <AlertTriangleIcon className="h-4 w-4" />,
                },
                {
                  type: 'info',
                  message: 'New Singer tap installed',
                  time: '1 hour ago',
                  icon: <ActivityIcon className="h-4 w-4" />,
                },
                {
                  type: 'error',
                  message: 'Connection timeout error',
                  time: '2 hours ago',
                  icon: <ClockIcon className="h-4 w-4" />,
                },
              ].map((activity, index) => (
                <div key={index} className="flex items-start space-x-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                  <div className={cn(
                    'p-1 rounded-full flex-shrink-0',
                    activity.type === 'success' && 'text-success-600 bg-success-100 dark:bg-success-900/20',
                    activity.type === 'warning' && 'text-warning-600 bg-warning-100 dark:bg-warning-900/20',
                    activity.type === 'error' && 'text-danger-600 bg-danger-100 dark:bg-danger-900/20',
                    activity.type === 'info' && 'text-primary-600 bg-primary-100 dark:bg-primary-900/20'
                  )}>
                    {activity.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900 dark:text-white">
                      {activity.message}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-500">
                      {activity.time}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}