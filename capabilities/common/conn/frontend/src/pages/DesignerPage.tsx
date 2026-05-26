import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Header } from '@/components/layout/Header'
import { FlowDesignerWrapper } from '@/components/designer/FlowDesigner'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  PlayIcon,
  SaveIcon,
  ShareIcon,
  FolderIcon,
  PlusIcon,
  HistoryIcon,
  SettingsIcon
} from 'lucide-react'
import toast from 'react-hot-toast'

export function DesignerPage() {
  const [flowName, setFlowName] = useState('Untitled Flow')
  const [isReadonly] = useState(false)
  const [isSaved, setIsSaved] = useState(true)

  const handleSave = (flowDefinition: any) => {
    console.log('Saving flow:', flowDefinition)
    toast.success('Flow saved successfully!')
    setIsSaved(true)
  }

  const handleExecute = (flowDefinition: any) => {
    console.log('Executing flow:', flowDefinition)
    toast.success('Flow execution started!')
  }

  const handleNewFlow = () => {
    setFlowName('Untitled Flow')
    setIsSaved(false)
    toast.success('New flow created!')
  }

  const handleLoadTemplate = () => {
    toast.success('Template loaded!')
  }

  const handleShare = () => {
    toast.success('Flow share link copied to clipboard!')
  }

  const headerActions = (
    <div className="flex items-center space-x-3">
      {/* Flow Status */}
      <div className="flex items-center space-x-2">
        <Badge variant={isSaved ? 'success' : 'warning'} size="sm">
          {isSaved ? 'Saved' : 'Unsaved'}
        </Badge>
        {!isReadonly && (
          <Badge variant="default" size="sm">
            Editing
          </Badge>
        )}
      </div>

      <div className="w-px h-6 bg-gray-300 dark:bg-gray-600" />

      {/* Action Buttons */}
      <Button
        variant="outline"
        size="sm"
        icon={<FolderIcon className="h-4 w-4" />}
        onClick={handleLoadTemplate}
      >
        Templates
      </Button>

      <Button
        variant="outline"
        size="sm"
        icon={<HistoryIcon className="h-4 w-4" />}
      >
        History
      </Button>

      <Button
        variant="outline"
        size="sm"
        icon={<ShareIcon className="h-4 w-4" />}
        onClick={handleShare}
      >
        Share
      </Button>

      <Button
        variant="outline"
        size="sm"
        icon={<SettingsIcon className="h-4 w-4" />}
      >
        Settings
      </Button>

      <div className="w-px h-6 bg-gray-300 dark:bg-gray-600" />

      <Button
        variant="outline"
        size="sm"
        icon={<PlusIcon className="h-4 w-4" />}
        onClick={handleNewFlow}
      >
        New Flow
      </Button>

      <Button
        variant="default"
        size="sm"
        icon={<PlayIcon className="h-4 w-4" />}
        disabled={isReadonly}
      >
        Execute Flow
      </Button>
    </div>
  )

  return (
    <>
      <Header
        title={
          <div className="flex items-center space-x-3">
            <span>Visual Flow Designer</span>
            <Badge variant="success" size="sm">
              Beta
            </Badge>
          </div>
        }
        subtitle={`Design and orchestrate data flows with drag-and-drop simplicity • ${flowName}`}
        actions={headerActions}
      />

      <motion.main
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex-1 flex flex-col min-h-0"
      >
        {/* Designer Canvas */}
        <div className="flex-1">
          <FlowDesignerWrapper
            onSave={handleSave}
            onExecute={handleExecute}
            readonly={isReadonly}
          />
        </div>

        {/* Status Bar */}
        <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
              <span>Flow: {flowName}</span>
              <span>•</span>
              <span>Last saved: 2 minutes ago</span>
              <span>•</span>
              <span>Auto-save enabled</span>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm">
                <div className="w-2 h-2 bg-success-500 rounded-full animate-pulse" />
                <span className="text-gray-600 dark:text-gray-400">
                  Real-time collaboration active
                </span>
              </div>

              <div className="flex items-center space-x-1">
                <div className="w-6 h-6 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center text-xs font-medium text-white">
                  A
                </div>
                <div className="w-6 h-6 bg-gradient-to-br from-success-500 to-success-600 rounded-full flex items-center justify-center text-xs font-medium text-white">
                  B
                </div>
                <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">
                  2 collaborators
                </span>
              </div>
            </div>
          </div>
        </div>
      </motion.main>
    </>
  )
}