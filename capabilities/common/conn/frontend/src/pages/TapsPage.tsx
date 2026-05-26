import React from 'react'
import { motion } from 'framer-motion'
import { Header } from '@/components/layout/Header'
import { DatabaseIcon } from 'lucide-react'

export function TapsPage() {
  return (
    <>
      <Header title="Singer Taps" subtitle="Manage Singer.io taps and data connectors" />
      <motion.main
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex-1 p-6 flex items-center justify-center"
      >
        <div className="text-center">
          <DatabaseIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Singer Taps
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            This page will contain Singer.io tap management.
          </p>
        </div>
      </motion.main>
    </>
  )
}