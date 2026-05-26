import React from 'react'
import { motion } from 'framer-motion'
import { Header } from '@/components/layout/Header'
import { BarChart3Icon } from 'lucide-react'

export function AnalyticsPage() {
  return (
    <>
      <Header title="Analytics" subtitle="Performance insights and data processing metrics" />
      <motion.main
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex-1 p-6 flex items-center justify-center"
      >
        <div className="text-center">
          <BarChart3Icon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Analytics
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            This page will contain detailed analytics and reporting.
          </p>
        </div>
      </motion.main>
    </>
  )
}