/**
 * APG Workflow Orchestration UI - Main Entry Point
 * 
 * Main application entry point for the React drag-drop canvas interface.
 * 
 * © 2025 Datacraft. All rights reserved.
 * Author: Nyimbi Odero <nyimbi@gmail.com>
 */

import React from 'react';
import { createRoot } from 'react-dom/client';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Import main components
import WorkflowCanvas from './workflow_canvas';
import WorkflowApp from './components/WorkflowApp';

// Import styles
import '../css/workflow_canvas.css';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0'
    },
    secondary: {
      main: '#dc004e',
      light: '#ff4081',
      dark: '#c51162'
    },
    background: {
      default: '#fafafa',
      paper: '#ffffff'
    }
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h6: {
      fontWeight: 500
    }
  },
  shape: {
    borderRadius: 8
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          '&:hover': {
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.15)'
          }
        }
      }
    }
  }
});

/**
 * Initialize the workflow canvas application
 */
function initializeWorkflowCanvas() {
  const canvasContainer = document.getElementById('workflow-canvas-root');
  
  if (canvasContainer) {
    const root = createRoot(canvasContainer);
    
    // Get initial workflow data from data attributes
    const workflowData = canvasContainer.dataset.workflow 
      ? JSON.parse(canvasContainer.dataset.workflow) 
      : null;
    
    const isReadOnly = canvasContainer.dataset.readonly === 'true';
    const collaborators = canvasContainer.dataset.collaborators
      ? JSON.parse(canvasContainer.dataset.collaborators)
      : [];

    root.render(
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <WorkflowCanvas
          workflow={workflowData}
          collaborators={collaborators}
          isReadOnly={isReadOnly}
          onWorkflowUpdate={(updatedWorkflow) => {
            // Send updates to backend via API
            console.log('Workflow updated:', updatedWorkflow);
            // This would typically call an API endpoint
          }}
        />
      </ThemeProvider>
    );
  }
}

/**
 * Initialize the full workflow application
 */
function initializeWorkflowApp() {
  const appContainer = document.getElementById('workflow-app-root');
  
  if (appContainer) {
    const root = createRoot(appContainer);
    
    root.render(
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <WorkflowApp />
      </ThemeProvider>
    );
  }
}

/**
 * API utilities for communicating with the backend
 */
export const WorkflowAPI = {
  /**
   * Save workflow to backend
   */
  async saveWorkflow(workflow) {
    try {
      const response = await fetch('/api/v1/workflow_orchestration/workflows', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': document.querySelector('meta[name=csrf-token]')?.content
        },
        body: JSON.stringify(workflow)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to save workflow:', error);
      throw error;
    }
  },

  /**
   * Load workflow from backend
   */
  async loadWorkflow(workflowId) {
    try {
      const response = await fetch(`/api/v1/workflow_orchestration/workflows/${workflowId}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to load workflow:', error);
      throw error;
    }
  },

  /**
   * Validate workflow
   */
  async validateWorkflow(workflow) {
    try {
      const response = await fetch('/api/v1/workflow_orchestration/workflows/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(workflow)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to validate workflow:', error);
      throw error;
    }
  },

  /**
   * Execute workflow
   */
  async executeWorkflow(workflowId, parameters = {}) {
    try {
      const response = await fetch(`/api/v1/workflow_orchestration/workflows/${workflowId}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': document.querySelector('meta[name=csrf-token]')?.content
        },
        body: JSON.stringify({ parameters })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to execute workflow:', error);
      throw error;
    }
  },

  /**
   * Get workflow templates
   */
  async getTemplates(category = null) {
    try {
      const url = category 
        ? `/api/v1/workflow_orchestration/templates?category=${category}`
        : '/api/v1/workflow_orchestration/templates';
        
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to load templates:', error);
      throw error;
    }
  }
};

/**
 * WebSocket connection for real-time collaboration
 */
export class WorkflowCollaboration {
  constructor(workflowId, userId) {
    this.workflowId = workflowId;
    this.userId = userId;
    this.socket = null;
    this.isConnected = false;
    this.eventHandlers = new Map();
  }

  connect() {
    if (this.socket) {
      return;
    }

    this.socket = new WebSocket(`ws://localhost:5000/ws/workflow/${this.workflowId}`);
    
    this.socket.onopen = () => {
      this.isConnected = true;
      console.log('WebSocket connected for collaboration');
      
      // Send initial presence
      this.socket.send(JSON.stringify({
        type: 'user_joined',
        user_id: this.userId,
        workflow_id: this.workflowId
      }));
    };

    this.socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.socket.onclose = () => {
      this.isConnected = false;
      console.log('WebSocket disconnected');
      
      // Attempt to reconnect after delay
      setTimeout(() => {
        if (!this.isConnected) {
          this.connect();
        }
      }, 5000);
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }
  }

  sendMessage(type, data) {
    if (this.isConnected && this.socket) {
      this.socket.send(JSON.stringify({
        type,
        user_id: this.userId,
        workflow_id: this.workflowId,
        ...data
      }));
    }
  }

  handleMessage(message) {
    const handlers = this.eventHandlers.get(message.type);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          console.error('Error in message handler:', error);
        }
      });
    }
  }

  on(eventType, handler) {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, []);
    }
    this.eventHandlers.get(eventType).push(handler);
  }

  off(eventType, handler) {
    const handlers = this.eventHandlers.get(eventType);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  // Collaboration methods
  broadcastCursorPosition(x, y) {
    this.sendMessage('cursor_move', { x, y });
  }

  broadcastComponentEdit(componentId, field, value) {
    this.sendMessage('component_edit', {
      component_id: componentId,
      field,
      value
    });
  }

  broadcastComponentSelect(componentId) {
    this.sendMessage('component_select', {
      component_id: componentId
    });
  }
}

// Global utilities
window.WorkflowAPI = WorkflowAPI;
window.WorkflowCollaboration = WorkflowCollaboration;

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initializeWorkflowCanvas();
  initializeWorkflowApp();
});

// Hot module replacement for development
if (module.hot) {
  module.hot.accept('./workflow_canvas', () => {
    console.log('Hot reloading workflow canvas...');
    initializeWorkflowCanvas();
  });
}