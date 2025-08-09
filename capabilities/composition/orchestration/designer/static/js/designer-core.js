/**
 * APG Workflow Designer Core
 * 
 * Core functionality for the workflow designer interface.
 * 
 * © 2025 Datacraft. All rights reserved.
 * Author: Nyimbi Odero <nyimbi@gmail.com>
 */

class WorkflowDesigner {
    constructor(config) {
        this.config = config;
        this.sessionId = null;
        this.nodes = new Map();
        this.connections = new Map();
        this.selectedNodes = new Set();
        this.selectedConnections = new Set();
        this.clipboard = null;
        this.history = [];
        this.historyIndex = -1;
        this.isDirty = false;
        this.isLoading = false;
        
        // Event handlers
        this.eventHandlers = {
            nodeAdded: [],
            nodeRemoved: [],
            nodeSelected: [],
            connectionAdded: [],
            connectionRemoved: [],
            workflowChanged: [],
            validationChanged: []
        };
        
        // Initialize
        this.initialize();
    }
    
    async initialize() {
        try {
            this.isLoading = true;
            this.showLoading('Initializing designer...');
            
            // Create designer session
            await this.createSession();
            
            // Initialize components
            this.canvas = new CanvasEngine(this);
            this.componentLibrary = new ComponentLibrary(this);
            this.propertyPanels = new PropertyPanels(this);
            this.collaboration = new CollaborationManager(this);
            
            // Setup UI event handlers
            this.setupEventHandlers();
            
            // Load initial data
            await this.loadInitialData();
            
            this.hideLoading();
            this.isLoading = false;
            
            console.log('Workflow designer initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize designer:', error);
            this.showError('Failed to initialize designer: ' + error.message);
            this.isLoading = false;
        }
    }
    
    async createSession() {
        try {
            const response = await fetch(`${this.config.api_base_url}/session`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    workflow_id: this.config.workflow_id,
                    template_id: this.config.template_id
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to create session');
            }
            
            this.sessionId = result.session.session_id;
            console.log('Designer session created:', this.sessionId);
            
        } catch (error) {
            console.error('Failed to create session:', error);
            throw error;
        }
    }
    
    async loadInitialData() {
        try {
            // Load workflow if specified
            if (this.config.workflow_id) {
                await this.loadWorkflow(this.config.workflow_id);
            } else if (this.config.template_id) {
                await this.loadTemplate(this.config.template_id);
            }
            
            // Load components library
            await this.componentLibrary.loadComponents();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            throw error;
        }
    }
    
    setupEventHandlers() {
        // Toolbar buttons
        document.getElementById('save-btn').addEventListener('click', () => this.saveWorkflow());
        document.getElementById('validate-btn').addEventListener('click', () => this.validateWorkflow());
        document.getElementById('export-btn').addEventListener('click', () => this.showExportModal());
        document.getElementById('undo-btn').addEventListener('click', () => this.undo());
        document.getElementById('redo-btn').addEventListener('click', () => this.redo());
        document.getElementById('settings-btn').addEventListener('click', () => this.showSettingsModal());
        document.getElementById('help-btn').addEventListener('click', () => this.showHelp());
        
        // Canvas controls
        document.getElementById('zoom-in-btn').addEventListener('click', () => this.canvas.zoomIn());
        document.getElementById('zoom-out-btn').addEventListener('click', () => this.canvas.zoomOut());
        document.getElementById('zoom-fit-btn').addEventListener('click', () => this.canvas.fitToScreen());
        document.getElementById('grid-toggle').addEventListener('click', () => this.canvas.toggleGrid());
        document.getElementById('minimap-toggle').addEventListener('click', () => this.canvas.toggleMinimap());
        document.getElementById('auto-layout').addEventListener('click', () => this.canvas.autoLayout());
        
        // Bottom panel tabs
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });
        
        // Modal handlers
        this.setupModalHandlers();
        
        // Keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Auto-save
        if (this.config.auto_save_interval > 0) {
            setInterval(() => {
                if (this.isDirty && !this.isLoading) {
                    this.autoSave();
                }
            }, this.config.auto_save_interval * 1000);
        }
    }
    
    setupModalHandlers() {
        // Export modal
        document.getElementById('export-confirm').addEventListener('click', () => {
            this.exportWorkflow();
        });
        
        // Settings modal
        document.getElementById('settings-save').addEventListener('click', () => {
            this.saveSettings();
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd combinations
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 's':
                        e.preventDefault();
                        this.saveWorkflow();
                        break;
                    case 'z':
                        e.preventDefault();
                        if (e.shiftKey) {
                            this.redo();
                        } else {
                            this.undo();
                        }
                        break;
                    case 'c':
                        e.preventDefault();
                        this.copySelected();
                        break;
                    case 'v':
                        e.preventDefault();
                        this.paste();
                        break;
                    case 'a':
                        e.preventDefault();
                        this.selectAll();
                        break;
                }
            }
            
            // Other keys
            switch (e.key) {
                case 'Delete':
                case 'Backspace':
                    e.preventDefault();
                    this.deleteSelected();
                    break;
                case 'Escape':
                    this.clearSelection();
                    break;
            }
        });
    }
    
    // === Node Management ===
    
    async addNode(componentType, position, config = {}) {
        try {
            const response = await fetch(`${this.config.api_base_url}/session/${this.sessionId}/component`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    component_type: componentType,
                    position: position,
                    config: config
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to add component');
            }
            
            const node = result.component;
            this.nodes.set(node.id, node);
            
            // Add to canvas
            this.canvas.addNode(node);
            
            // Update history
            this.addToHistory('addNode', { node });
            
            // Trigger events
            this.triggerEvent('nodeAdded', node);
            this.triggerEvent('workflowChanged');
            
            this.markDirty();
            
            return node;
            
        } catch (error) {
            console.error('Failed to add node:', error);
            this.showError('Failed to add component: ' + error.message);
            throw error;
        }
    }
    
    async removeNode(nodeId) {
        try {
            const response = await fetch(`${this.config.api_base_url}/session/${this.sessionId}/component/${nodeId}`, {
                method: 'DELETE',
                headers: {
                    'X-CSRFToken': this.getCSRFToken()
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to remove component');
            }
            
            const node = this.nodes.get(nodeId);
            this.nodes.delete(nodeId);
            this.selectedNodes.delete(nodeId);
            
            // Remove from canvas
            this.canvas.removeNode(nodeId);
            
            // Remove connected connections
            const connectionsToRemove = [];
            this.connections.forEach((conn, connId) => {
                if (conn.source_node_id === nodeId || conn.target_node_id === nodeId) {
                    connectionsToRemove.push(connId);
                }
            });
            
            for (const connId of connectionsToRemove) {
                await this.removeConnection(connId);
            }
            
            // Update history
            this.addToHistory('removeNode', { node, nodeId });
            
            // Trigger events
            this.triggerEvent('nodeRemoved', nodeId);
            this.triggerEvent('workflowChanged');
            
            this.markDirty();
            
        } catch (error) {
            console.error('Failed to remove node:', error);
            this.showError('Failed to remove component: ' + error.message);
            throw error;
        }
    }
    
    async moveNode(nodeId, position) {
        try {
            const response = await fetch(`${this.config.api_base_url}/session/${this.sessionId}/component/${nodeId}/move`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    position: position
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to move component');
            }
            
            // Update local node
            const node = this.nodes.get(nodeId);
            if (node) {
                const oldPosition = { ...node.position };
                node.position = position;
                
                // Update canvas
                this.canvas.moveNode(nodeId, position);
                
                // Update history
                this.addToHistory('moveNode', { nodeId, oldPosition, newPosition: position });
                
                // Trigger events
                this.triggerEvent('workflowChanged');
                
                this.markDirty();
            }
            
        } catch (error) {
            console.error('Failed to move node:', error);
            this.showError('Failed to move component: ' + error.message);
            throw error;
        }
    }
    
    // === Connection Management ===
    
    async addConnection(sourceId, targetId, sourcePort = 'output', targetPort = 'input') {
        try {
            const response = await fetch(`${this.config.api_base_url}/session/${this.sessionId}/connection`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    source_id: sourceId,
                    target_id: targetId,
                    source_port: sourcePort,
                    target_port: targetPort
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to add connection');
            }
            
            const connection = result.connection;
            this.connections.set(connection.id, connection);
            
            // Add to canvas
            this.canvas.addConnection(connection);
            
            // Update history
            this.addToHistory('addConnection', { connection });
            
            // Trigger events
            this.triggerEvent('connectionAdded', connection);
            this.triggerEvent('workflowChanged');
            
            this.markDirty();
            
            return connection;
            
        } catch (error) {
            console.error('Failed to add connection:', error);
            this.showError('Failed to add connection: ' + error.message);
            throw error;
        }
    }
    
    async removeConnection(connectionId) {
        try {
            const response = await fetch(`${this.config.api_base_url}/session/${this.sessionId}/connection/${connectionId}`, {
                method: 'DELETE',
                headers: {
                    'X-CSRFToken': this.getCSRFToken()
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to remove connection');
            }
            
            const connection = this.connections.get(connectionId);
            this.connections.delete(connectionId);
            this.selectedConnections.delete(connectionId);
            
            // Remove from canvas
            this.canvas.removeConnection(connectionId);
            
            // Update history
            this.addToHistory('removeConnection', { connection, connectionId });
            
            // Trigger events
            this.triggerEvent('connectionRemoved', connectionId);
            this.triggerEvent('workflowChanged');
            
            this.markDirty();
            
        } catch (error) {
            console.error('Failed to remove connection:', error);
            this.showError('Failed to remove connection: ' + error.message);
            throw error;
        }
    }
    
    // === Selection Management ===
    
    selectNode(nodeId, addToSelection = false) {
        if (!addToSelection) {
            this.clearSelection();
        }
        
        this.selectedNodes.add(nodeId);
        this.canvas.selectNode(nodeId);
        
        // Update properties panel
        if (this.selectedNodes.size === 1) {
            this.propertyPanels.showNodeProperties(nodeId);
        } else {
            this.propertyPanels.showMultiSelection();
        }
        
        this.triggerEvent('nodeSelected', Array.from(this.selectedNodes));
    }
    
    selectConnection(connectionId, addToSelection = false) {
        if (!addToSelection) {
            this.clearSelection();
        }
        
        this.selectedConnections.add(connectionId);
        this.canvas.selectConnection(connectionId);
    }
    
    clearSelection() {
        this.selectedNodes.clear();
        this.selectedConnections.clear();
        this.canvas.clearSelection();
        this.propertyPanels.showEmpty();
    }
    
    selectAll() {
        this.selectedNodes.clear();
        this.selectedConnections.clear();
        
        this.nodes.forEach((node, nodeId) => {
            this.selectedNodes.add(nodeId);
        });
        
        this.connections.forEach((conn, connId) => {
            this.selectedConnections.add(connId);
        });
        
        this.canvas.selectAll();
        this.propertyPanels.showMultiSelection();
    }
    
    // === Copy/Paste ===
    
    copySelected() {
        if (this.selectedNodes.size === 0) return;
        
        const nodesToCopy = [];
        const connectionsToCopy = [];
        
        // Copy selected nodes
        this.selectedNodes.forEach(nodeId => {
            const node = this.nodes.get(nodeId);
            if (node) {
                nodesToCopy.push({ ...node });
            }
        });
        
        // Copy connections between selected nodes
        this.connections.forEach(conn => {
            if (this.selectedNodes.has(conn.source_node_id) && 
                this.selectedNodes.has(conn.target_node_id)) {
                connectionsToCopy.push({ ...conn });
            }
        });
        
        this.clipboard = {
            nodes: nodesToCopy,
            connections: connectionsToCopy,
            timestamp: Date.now()
        };
        
        this.showNotification('Copied to clipboard', 'success');
    }
    
    async paste() {
        if (!this.clipboard || !this.clipboard.nodes.length) return;
        
        try {
            const offset = { x: 50, y: 50 };
            const nodeIdMap = new Map();
            
            // Paste nodes
            for (const originalNode of this.clipboard.nodes) {
                const position = {
                    x: originalNode.position.x + offset.x,
                    y: originalNode.position.y + offset.y
                };
                
                const newNode = await this.addNode(
                    originalNode.component_type,
                    position,
                    { ...originalNode.config }
                );
                
                nodeIdMap.set(originalNode.id, newNode.id);
            }
            
            // Paste connections
            for (const originalConn of this.clipboard.connections) {
                const sourceId = nodeIdMap.get(originalConn.source_node_id);
                const targetId = nodeIdMap.get(originalConn.target_node_id);
                
                if (sourceId && targetId) {
                    await this.addConnection(
                        sourceId,
                        targetId,
                        originalConn.source_port,
                        originalConn.target_port
                    );
                }
            }
            
            this.showNotification('Pasted from clipboard', 'success');
            
        } catch (error) {
            console.error('Failed to paste:', error);
            this.showError('Failed to paste: ' + error.message);
        }
    }
    
    // === Delete ===
    
    async deleteSelected() {
        if (this.selectedNodes.size === 0 && this.selectedConnections.size === 0) return;
        
        try {
            // Delete selected connections
            for (const connectionId of this.selectedConnections) {
                await this.removeConnection(connectionId);
            }
            
            // Delete selected nodes
            for (const nodeId of this.selectedNodes) {
                await this.removeNode(nodeId);
            }
            
            this.clearSelection();
            
        } catch (error) {
            console.error('Failed to delete selected:', error);
            this.showError('Failed to delete selected items: ' + error.message);
        }
    }
    
    // === History Management ===
    
    addToHistory(action, data) {
        // Remove any redo history
        this.history = this.history.slice(0, this.historyIndex + 1);
        
        // Add new action
        this.history.push({
            action,
            data,
            timestamp: Date.now()
        });
        
        this.historyIndex++;
        
        // Limit history size
        if (this.history.length > this.config.max_undo_steps) {
            this.history.shift();
            this.historyIndex--;
        }
        
        this.updateHistoryButtons();
    }
    
    async undo() {
        if (this.historyIndex < 0) return;
        
        const action = this.history[this.historyIndex];
        this.historyIndex--;
        
        try {
            await this.revertAction(action);
            this.updateHistoryButtons();
        } catch (error) {
            console.error('Failed to undo:', error);
            this.showError('Failed to undo action: ' + error.message);
            this.historyIndex++; // Restore index on error
        }
    }
    
    async redo() {
        if (this.historyIndex >= this.history.length - 1) return;
        
        this.historyIndex++;
        const action = this.history[this.historyIndex];
        
        try {
            await this.replayAction(action);
            this.updateHistoryButtons();
        } catch (error) {
            console.error('Failed to redo:', error);
            this.showError('Failed to redo action: ' + error.message);
            this.historyIndex--; // Restore index on error
        }
    }
    
    async revertAction(action) {
        switch (action.action) {
            case 'addNode':
                await this.removeNode(action.data.node.id);
                break;
            case 'removeNode':
                await this.addNode(
                    action.data.node.component_type,
                    action.data.node.position,
                    action.data.node.config
                );
                break;
            case 'moveNode':
                await this.moveNode(action.data.nodeId, action.data.oldPosition);
                break;
            case 'addConnection':
                await this.removeConnection(action.data.connection.id);
                break;
            case 'removeConnection':
                const conn = action.data.connection;
                await this.addConnection(
                    conn.source_node_id,
                    conn.target_node_id,
                    conn.source_port,
                    conn.target_port
                );
                break;
        }
    }
    
    async replayAction(action) {
        switch (action.action) {
            case 'addNode':
                const node = action.data.node;
                await this.addNode(node.component_type, node.position, node.config);
                break;
            case 'removeNode':
                await this.removeNode(action.data.nodeId);
                break;
            case 'moveNode':
                await this.moveNode(action.data.nodeId, action.data.newPosition);
                break;
            case 'addConnection':
                const conn = action.data.connection;
                await this.addConnection(
                    conn.source_node_id,
                    conn.target_node_id,
                    conn.source_port,
                    conn.target_port
                );
                break;
            case 'removeConnection':
                await this.removeConnection(action.data.connectionId);
                break;
        }
    }
    
    updateHistoryButtons() {
        document.getElementById('undo-btn').disabled = this.historyIndex < 0;
        document.getElementById('redo-btn').disabled = this.historyIndex >= this.history.length - 1;
    }
    
    // === Workflow Operations ===
    
    async saveWorkflow() {
        try {
            this.showLoading('Saving workflow...');
            
            const workflowData = this.getWorkflowData();
            
            const response = await fetch(`${this.config.api_base_url}/session/${this.sessionId}/save`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    workflow: workflowData
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to save workflow');
            }
            
            this.isDirty = false;
            this.showNotification('Workflow saved successfully', 'success');
            
            this.hideLoading();
            
        } catch (error) {
            console.error('Failed to save workflow:', error);
            this.showError('Failed to save workflow: ' + error.message);
            this.hideLoading();
        }
    }
    
    async autoSave() {
        try {
            await this.saveWorkflow();
            console.log('Auto-saved workflow');
        } catch (error) {
            console.error('Auto-save failed:', error);
        }
    }
    
    async validateWorkflow() {
        try {
            this.showLoading('Validating workflow...');
            
            const response = await fetch(`${this.config.api_base_url}/session/${this.sessionId}/validate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to validate workflow');
            }
            
            this.displayValidationResults(result.validation);
            
            this.hideLoading();
            
        } catch (error) {
            console.error('Failed to validate workflow:', error);
            this.showError('Failed to validate workflow: ' + error.message);
            this.hideLoading();
        }
    }
    
    // === Helper Methods ===
    
    getWorkflowData() {
        const nodes = Array.from(this.nodes.values());
        const connections = Array.from(this.connections.values());
        
        return {
            name: document.getElementById('workflow-name').textContent,
            description: '',
            definition: {
                nodes: nodes,
                connections: connections
            },
            metadata: {
                created_at: new Date().toISOString(),
                designer_version: '1.0.0'
            }
        };
    }
    
    markDirty() {
        this.isDirty = true;
        document.getElementById('save-btn').classList.add('btn-warning');
        document.getElementById('save-btn').innerHTML = '<i class="fas fa-save"></i> Save*';
    }
    
    getCSRFToken() {
        const token = document.querySelector('meta[name=csrf-token]');
        return token ? token.getAttribute('content') : '';
    }
    
    // === Event System ===
    
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        }
    }
    
    off(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
    }
    
    triggerEvent(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }
    
    // === UI Methods ===
    
    showLoading(message = 'Loading...') {
        // Implementation would show loading overlay
        console.log('Loading:', message);
    }
    
    hideLoading() {
        // Implementation would hide loading overlay
        console.log('Loading complete');
    }
    
    showNotification(message, type = 'info') {
        // Implementation would show toast notification
        console.log(`${type.toUpperCase()}: ${message}`);
    }
    
    showError(message) {
        this.showNotification(message, 'error');
    }
    
    switchTab(tabName) {
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.remove('active');
        });
        
        // Show selected tab
        document.getElementById(`${tabName}-tab`).classList.add('active');
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    }
    
    displayValidationResults(validation) {
        const container = document.getElementById('validation-results');
        container.innerHTML = '';
        
        if (validation.issues && validation.issues.length > 0) {
            validation.issues.forEach(issue => {
                const div = document.createElement('div');
                div.className = `validation-item ${issue.severity}`;
                div.innerHTML = `
                    <i class="fas fa-${issue.severity === 'error' ? 'times-circle' : 
                                      issue.severity === 'warning' ? 'exclamation-triangle' : 
                                      'info-circle'} validation-icon"></i>
                    <div class="validation-content">
                        <div class="validation-message">${issue.message}</div>
                        ${issue.description ? `<div class="validation-description">${issue.description}</div>` : ''}
                    </div>
                `;
                container.appendChild(div);
            });
            
            // Update validation count
            document.getElementById('validation-count').textContent = validation.issues.length;
        } else {
            container.innerHTML = '<div class="text-center text-muted p-3">No validation issues found</div>';
            document.getElementById('validation-count').textContent = '0';
        }
        
        // Switch to validation tab
        this.switchTab('validation');
    }
    
    showExportModal() {
        $('#export-modal').modal('show');
    }
    
    showSettingsModal() {
        $('#settings-modal').modal('show');
    }
    
    showHelp() {
        window.open('/workflow/designer/help', '_blank');
    }
    
    async exportWorkflow() {
        try {
            const format = document.getElementById('export-format').value;
            const options = {
                include_metadata: document.getElementById('include-metadata').checked,
                include_config: document.getElementById('include-config').checked,
                include_layout: document.getElementById('include-layout').checked
            };
            
            const response = await fetch(`${this.config.api_base_url}/session/${this.sessionId}/export`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    format: format,
                    options: options
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to export workflow');
            }
            
            // Download the exported file
            this.downloadFile(result.export.content, `workflow.${format}`);
            
            $('#export-modal').modal('hide');
            this.showNotification('Workflow exported successfully', 'success');
            
        } catch (error) {
            console.error('Failed to export workflow:', error);
            this.showError('Failed to export workflow: ' + error.message);
        }
    }
    
    downloadFile(content, filename) {
        const blob = new Blob([atob(content)], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    saveSettings() {
        // Implementation would save designer settings
        $('#settings-modal').modal('hide');
        this.showNotification('Settings saved', 'success');
    }
}

// Global designer instance
window.workflowDesigner = null;