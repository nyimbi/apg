import React from 'react'
import { motion } from 'framer-motion'
import { Header } from '@/components/layout/Header'
import { SettingsIcon } from 'lucide-react'

export function SettingsPage() {
  return (
    <>
      <Header title="Settings" subtitle="Configure your APG Connection Management platform" />
      <motion.main
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex-1 p-6 flex items-center justify-center"
      >
        <div className="text-center">
          <SettingsIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Settings
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            This page will contain application settings and configuration.
          </p>
        </div>
      </motion.main>
    </>
  )
}