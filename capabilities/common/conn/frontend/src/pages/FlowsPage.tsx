import React from 'react'
import { motion } from 'framer-motion'
import { Header } from '@/components/layout/Header'
import { Button } from '@/components/ui/Button'
import { PlusIcon, WorkflowIcon } from 'lucide-react'

export function FlowsPage() {
  return (
    <>
      <Header title="Data Flows" subtitle="Orchestrate and manage your data processing workflows" />
      <motion.main
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex-1 p-6 flex items-center justify-center"
      >
        <div className="text-center">
          <WorkflowIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Flows Page
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            This page will contain flow management functionality.
          </p>
          <Button icon={<PlusIcon className="h-4 w-4" />}>
            New Flow
          </Button>
        </div>
      </motion.main>
    </>
  )
}