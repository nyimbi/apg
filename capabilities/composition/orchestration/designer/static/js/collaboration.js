/**
 * APG Workflow Collaboration Manager
 * 
 * Real-time collaboration features for workflow design.
 * 
 * © 2025 Datacraft. All rights reserved.
 * Author: Nyimbi Odero <nyimbi@gmail.com>
 */

class CollaborationManager {
    constructor(designer) {
        this.designer = designer;
        this.socket = null;
        this.collaborators = new Map();
        this.cursors = new Map();
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.chatMessages = [];
        
        // DOM elements
        this.collaboratorsList = document.getElementById('collaborators');
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.chatSend = document.getElementById('chat-send');
        this.collaboratorCount = document.getElementById('collaborator-count');
        
        this.initialize();
    }
    
    async initialize() {
        try {
            // Initialize WebSocket connection for real-time collaboration
            await this.connectWebSocket();
            
            // Setup event handlers
            this.setupEventHandlers();
            
            // Setup presence tracking
            this.setupPresenceTracking();
            
            console.log('Collaboration manager initialized');
            
        } catch (error) {
            console.error('Failed to initialize collaboration:', error);
            this.showOfflineMode();
        }
    }
    
    async connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/v1/workflow/collaborate/${this.designer.sessionId}`;
            
            this.socket = new WebSocket(wsUrl);
            
            this.socket.onopen = () => {
                this.isConnected = true;
                this.reconnectAttempts = 0;
                console.log('WebSocket connected for collaboration');
                this.sendPresence();
            };
            
            this.socket.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.socket.onclose = () => {
                this.isConnected = false;
                console.log('WebSocket disconnected');
                this.attemptReconnect();
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.isConnected = false;
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            throw error;
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff
            
            console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
            
            setTimeout(() => {
                this.connectWebSocket();
            }, delay);
        } else {
            console.error('Max reconnect attempts reached');
            this.showOfflineMode();
        }
    }
    
    setupEventHandlers() {
        // Chat functionality
        this.chatSend.addEventListener('click', () => {
            this.sendChatMessage();
        });
        
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendChatMessage();
            }
        });
        
        // Listen to designer events
        this.designer.on('nodeAdded', (node) => {
            this.broadcastChange('node_added', { node });
        });
        
        this.designer.on('nodeRemoved', (nodeId) => {
            this.broadcastChange('node_removed', { nodeId });
        });
        
        this.designer.on('connectionAdded', (connection) => {
            this.broadcastChange('connection_added', { connection });
        });
        
        this.designer.on('connectionRemoved', (connectionId) => {
            this.broadcastChange('connection_removed', { connectionId });
        });
        
        this.designer.on('nodeSelected', (selectedNodes) => {
            this.broadcastCursor('selection', { selectedNodes });
        });
        
        // Mouse tracking for cursors
        document.addEventListener('mousemove', (e) => {
            this.throttledCursorUpdate(e);
        });
    }
    
    setupPresenceTracking() {
        // Send presence updates every 30 seconds
        setInterval(() => {
            if (this.isConnected) {
                this.sendPresence();
            }
        }, 30000);
        
        // Send presence when tab becomes visible
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.isConnected) {
                this.sendPresence();
            }
        });
        
        // Send presence on beforeunload
        window.addEventListener('beforeunload', () => {
            if (this.isConnected) {
                this.sendPresence('offline');
            }
        });
    }
    
    // === WebSocket Message Handling ===
    
    handleWebSocketMessage(event) {
        try {
            const message = JSON.parse(event.data);
            
            switch (message.type) {
                case 'collaborator_joined':
                    this.handleCollaboratorJoined(message.data);
                    break;
                case 'collaborator_left':
                    this.handleCollaboratorLeft(message.data);
                    break;
                case 'cursor_update':
                    this.handleCursorUpdate(message.data);
                    break;
                case 'workflow_change':
                    this.handleWorkflowChange(message.data);
                    break;
                case 'chat_message':
                    this.handleChatMessage(message.data);
                    break;
                case 'presence_update':
                    this.handlePresenceUpdate(message.data);
                    break;
                case 'error':
                    console.error('Collaboration error:', message.data);
                    break;
                default:
                    console.warn('Unknown message type:', message.type);
            }
            
        } catch (error) {
            console.error('Failed to handle WebSocket message:', error);
        }
    }
    
    handleCollaboratorJoined(data) {
        const collaborator = data.collaborator;\n        this.collaborators.set(collaborator.id, collaborator);
        this.updateCollaboratorsList();
        this.updateCollaboratorCount();
        
        // Show notification
        this.designer.showNotification(`${collaborator.name} joined the session`, 'info');
        
        // Add chat message
        this.addSystemMessage(`${collaborator.name} joined the collaboration`);
    }
    
    handleCollaboratorLeft(data) {
        const collaboratorId = data.collaborator_id;
        const collaborator = this.collaborators.get(collaboratorId);
        
        if (collaborator) {
            this.collaborators.delete(collaboratorId);
            this.cursors.delete(collaboratorId);
            this.updateCollaboratorsList();
            this.updateCollaboratorCount();
            this.removeCursor(collaboratorId);
            
            // Show notification
            this.designer.showNotification(`${collaborator.name} left the session`, 'info');
            
            // Add chat message
            this.addSystemMessage(`${collaborator.name} left the collaboration`);
        }
    }
    
    handleCursorUpdate(data) {
        const { collaborator_id, position, action } = data;
        const collaborator = this.collaborators.get(collaborator_id);
        
        if (collaborator && collaborator_id !== this.getCurrentUserId()) {
            this.updateCursor(collaborator_id, collaborator, position, action);
        }
    }
    
    handleWorkflowChange(data) {
        const { action, payload, collaborator_id } = data;
        const collaborator = this.collaborators.get(collaborator_id);
        
        // Skip if this change came from current user
        if (collaborator_id === this.getCurrentUserId()) return;
        
        console.log('Received workflow change:', action, payload);
        
        // Apply the change to local workflow
        this.applyRemoteChange(action, payload);
        
        // Show notification for significant changes
        if (collaborator && ['node_added', 'node_removed', 'connection_added', 'connection_removed'].includes(action)) {
            this.designer.showNotification(`${collaborator.name} ${action.replace('_', ' ')} a component`, 'info');
        }
    }
    
    handleChatMessage(data) {
        this.addChatMessage(data);
    }
    
    handlePresenceUpdate(data) {
        const { collaborators } = data;
        
        // Update collaborators list
        this.collaborators.clear();
        collaborators.forEach(collaborator => {
            this.collaborators.set(collaborator.id, collaborator);
        });
        
        this.updateCollaboratorsList();
        this.updateCollaboratorCount();
    }
    
    // === Broadcasting ===
    
    broadcastChange(action, payload) {
        if (!this.isConnected) return;
        
        const message = {
            type: 'workflow_change',
            data: {
                action,
                payload,
                session_id: this.designer.sessionId,
                timestamp: Date.now()
            }
        };
        
        this.socket.send(JSON.stringify(message));
    }
    
    broadcastCursor(action, data) {
        if (!this.isConnected) return;
        
        const message = {
            type: 'cursor_update',
            data: {
                action,
                data,
                session_id: this.designer.sessionId,
                timestamp: Date.now()
            }
        };
        
        this.socket.send(JSON.stringify(message));
    }
    
    sendPresence(status = 'online') {
        if (!this.isConnected) return;
        
        const message = {
            type: 'presence',
            data: {
                status,
                session_id: this.designer.sessionId,
                timestamp: Date.now()
            }
        };
        
        this.socket.send(JSON.stringify(message));
    }
    
    sendChatMessage() {
        const messageText = this.chatInput.value.trim();
        if (!messageText || !this.isConnected) return;
        
        const message = {
            type: 'chat_message',
            data: {
                message: messageText,
                session_id: this.designer.sessionId,
                timestamp: Date.now()
            }
        };
        
        this.socket.send(JSON.stringify(message));
        this.chatInput.value = '';
    }
    
    // === UI Updates ===
    
    updateCollaboratorsList() {
        this.collaboratorsList.innerHTML = '';
        
        Array.from(this.collaborators.values()).forEach(collaborator => {
            const collaboratorElement = this.createCollaboratorElement(collaborator);
            this.collaboratorsList.appendChild(collaboratorElement);
        });
    }
    
    createCollaboratorElement(collaborator) {
        const div = document.createElement('div');
        div.className = 'collaborator-item';
        div.dataset.collaboratorId = collaborator.id;
        
        const avatar = document.createElement('div');
        avatar.className = 'collaborator-avatar';
        avatar.style.backgroundColor = collaborator.color || this.generateColor(collaborator.id);
        avatar.textContent = collaborator.name.charAt(0).toUpperCase();
        
        const info = document.createElement('div');
        info.className = 'collaborator-info';
        
        const name = document.createElement('div');
        name.className = 'collaborator-name';
        name.textContent = collaborator.name;
        
        const status = document.createElement('div');
        status.className = 'collaborator-status';
        status.textContent = this.getStatusText(collaborator);
        
        info.appendChild(name);
        info.appendChild(status);
        
        div.appendChild(avatar);
        div.appendChild(info);
        
        return div;
    }
    
    updateCollaboratorCount() {
        const count = this.collaborators.size;
        this.collaboratorCount.textContent = count;
        this.collaboratorCount.style.backgroundColor = count > 1 ? '#28a745' : '#6c757d';
    }
    
    // === Cursor Management ===
    
    updateCursor(collaboratorId, collaborator, position, action) {
        let cursor = this.cursors.get(collaboratorId);
        
        if (!cursor) {
            cursor = this.createCursor(collaborator);
            this.cursors.set(collaboratorId, cursor);
            document.body.appendChild(cursor);
        }
        
        // Update cursor position
        cursor.style.left = position.x + 'px';
        cursor.style.top = position.y + 'px';
        
        // Update cursor action
        this.updateCursorAction(cursor, action);
        
        // Reset cursor visibility timer
        this.resetCursorTimer(collaboratorId);
    }
    
    createCursor(collaborator) {
        const cursor = document.createElement('div');
        cursor.className = 'collaboration-cursor';
        cursor.style.position = 'fixed';
        cursor.style.pointerEvents = 'none';
        cursor.style.zIndex = '10000';
        cursor.style.transition = 'all 0.1s ease';
        
        const color = collaborator.color || this.generateColor(collaborator.id);
        
        cursor.innerHTML = `
            <div class="cursor-pointer" style="
                width: 12px;
                height: 12px;
                background: ${color};
                border: 2px solid white;
                border-radius: 50%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            "></div>
            <div class="cursor-label" style="
                background: ${color};
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 500;
                margin-left: 15px;
                margin-top: -8px;
                white-space: nowrap;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            ">${collaborator.name}</div>
        `;
        
        return cursor;
    }
    
    updateCursorAction(cursor, action) {
        const label = cursor.querySelector('.cursor-label');
        const pointer = cursor.querySelector('.cursor-pointer');
        
        if (action && action.type) {
            switch (action.type) {
                case 'selection':
                    pointer.style.transform = 'scale(1.2)';
                    label.textContent = `${label.textContent.split(' - ')[0]} - selecting`;
                    break;
                case 'dragging':
                    pointer.style.transform = 'scale(1.5)';
                    label.textContent = `${label.textContent.split(' - ')[0]} - dragging`;
                    break;
                default:
                    pointer.style.transform = 'scale(1)';
                    label.textContent = label.textContent.split(' - ')[0];
            }
        }
    }
    
    resetCursorTimer(collaboratorId) {
        const cursor = this.cursors.get(collaboratorId);
        if (!cursor) return;
        
        // Clear existing timer
        if (cursor._hideTimer) {
            clearTimeout(cursor._hideTimer);
        }
        
        // Set new timer to hide cursor after 5 seconds of inactivity
        cursor._hideTimer = setTimeout(() => {
            this.hideCursor(collaboratorId);
        }, 5000);
    }
    
    hideCursor(collaboratorId) {
        const cursor = this.cursors.get(collaboratorId);
        if (cursor) {
            cursor.style.opacity = '0.3';
        }
    }
    
    removeCursor(collaboratorId) {
        const cursor = this.cursors.get(collaboratorId);
        if (cursor) {
            if (cursor._hideTimer) {
                clearTimeout(cursor._hideTimer);
            }
            cursor.remove();
            this.cursors.delete(collaboratorId);
        }
    }
    
    // === Chat Management ===
    
    addChatMessage(messageData) {
        const messageElement = this.createChatMessageElement(messageData);
        this.chatMessages.appendChild(messageElement);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    addSystemMessage(text) {
        const messageElement = document.createElement('div');
        messageElement.className = 'chat-message system';
        messageElement.innerHTML = `
            <div class="message-content">
                <i class="fas fa-info-circle"></i> ${text}
            </div>
            <div class="message-time">${new Date().toLocaleTimeString()}</div>
        `;
        
        this.chatMessages.appendChild(messageElement);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    createChatMessageElement(messageData) {
        const div = document.createElement('div');
        div.className = 'chat-message';
        
        const collaborator = this.collaborators.get(messageData.collaborator_id);
        const isCurrentUser = messageData.collaborator_id === this.getCurrentUserId();
        
        if (isCurrentUser) {
            div.classList.add('own-message');
        }
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.style.backgroundColor = collaborator?.color || this.generateColor(messageData.collaborator_id);
        avatar.textContent = (collaborator?.name || 'Unknown').charAt(0).toUpperCase();
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const header = document.createElement('div');
        header.className = 'message-header';
        header.innerHTML = `
            <span class="message-author">${collaborator?.name || 'Unknown'}</span>
            <span class="message-time">${new Date(messageData.timestamp).toLocaleTimeString()}</span>
        `;
        
        const text = document.createElement('div');
        text.className = 'message-text';
        text.textContent = messageData.message;
        
        content.appendChild(header);
        content.appendChild(text);
        
        if (!isCurrentUser) {
            div.appendChild(avatar);
        }
        div.appendChild(content);
        if (isCurrentUser) {
            div.appendChild(avatar);
        }
        
        return div;
    }
    
    // === Remote Change Application ===
    
    applyRemoteChange(action, payload) {
        switch (action) {
            case 'node_added':
                this.applyRemoteNodeAdded(payload.node);
                break;
            case 'node_removed':
                this.applyRemoteNodeRemoved(payload.nodeId);
                break;
            case 'connection_added':
                this.applyRemoteConnectionAdded(payload.connection);
                break;
            case 'connection_removed':
                this.applyRemoteConnectionRemoved(payload.connectionId);
                break;
            case 'node_moved':
                this.applyRemoteNodeMoved(payload.nodeId, payload.position);
                break;
        }
    }
    
    applyRemoteNodeAdded(node) {
        // Add node to local state without triggering events
        this.designer.nodes.set(node.id, node);
        
        // Add to canvas
        this.designer.canvas.addNode(node);
        
        // Don't add to history (this is a remote change)
    }
    
    applyRemoteNodeRemoved(nodeId) {
        // Remove from local state
        this.designer.nodes.delete(nodeId);
        this.designer.selectedNodes.delete(nodeId);
        
        // Remove from canvas
        this.designer.canvas.removeNode(nodeId);
    }
    
    applyRemoteConnectionAdded(connection) {
        // Add connection to local state
        this.designer.connections.set(connection.id, connection);
        
        // Add to canvas
        this.designer.canvas.addConnection(connection);
    }
    
    applyRemoteConnectionRemoved(connectionId) {
        // Remove from local state
        this.designer.connections.delete(connectionId);
        this.designer.selectedConnections.delete(connectionId);
        
        // Remove from canvas
        this.designer.canvas.removeConnection(connectionId);
    }
    
    applyRemoteNodeMoved(nodeId, position) {
        // Update local node position
        const node = this.designer.nodes.get(nodeId);
        if (node) {
            node.position = position;
            this.designer.canvas.moveNode(nodeId, position);
        }
    }
    
    // === Utility Methods ===
    
    throttledCursorUpdate = this.throttle((e) => {
        if (!this.isConnected) return;
        
        const canvasRect = this.designer.canvas.svg.getBoundingClientRect();
        if (e.clientX >= canvasRect.left && e.clientX <= canvasRect.right &&
            e.clientY >= canvasRect.top && e.clientY <= canvasRect.bottom) {
            
            this.broadcastCursor('move', {
                x: e.clientX,
                y: e.clientY
            });
        }
    }, 100);
    
    throttle(func, delay) {
        let timeoutId;
        let lastExecTime = 0;
        return function (...args) {
            const currentTime = Date.now();
            
            if (currentTime - lastExecTime > delay) {
                func.apply(this, args);
                lastExecTime = currentTime;
            } else {
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    func.apply(this, args);
                    lastExecTime = Date.now();
                }, delay - (currentTime - lastExecTime));
            }
        };
    }
    
    generateColor(id) {
        const colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F06292', '#AED581', '#FFB74D'
        ];
        const hash = id.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
        return colors[hash % colors.length];
    }
    
    getCurrentUserId() {
        return this.designer.config.current_user_id || 'anonymous';
    }
    
    getStatusText(collaborator) {
        if (collaborator.status === 'online') {
            return collaborator.last_seen ? 'Active' : 'Online';
        } else {
            return 'Away';
        }
    }
    
    showOfflineMode() {
        this.designer.showNotification('Collaboration unavailable - working offline', 'warning');
        this.collaboratorCount.textContent = '1';
        this.collaboratorCount.style.backgroundColor = '#6c757d';
        
        // Disable chat
        this.chatInput.disabled = true;
        this.chatSend.disabled = true;
        this.chatInput.placeholder = 'Chat unavailable (offline mode)';
    }
    
    // === Public API ===
    
    disconnect() {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
        this.isConnected = false;
        
        // Clear cursors
        this.cursors.forEach((cursor, collaboratorId) => {
            this.removeCursor(collaboratorId);
        });
        
        this.showOfflineMode();
    }
    
    isOnline() {
        return this.isConnected;
    }
    
    getCollaborators() {
        return Array.from(this.collaborators.values());
    }
}

// Add styles for collaboration features
const collaborationStyles = `
<style>
.collaboration-cursor {
    transition: all 0.1s ease;
}

.chat-message {
    display: flex;
    align-items: flex-start;
    margin-bottom: 12px;
    font-size: 13px;
}

.chat-message.own-message {
    flex-direction: row-reverse;
}

.chat-message.own-message .message-content {
    background: #007bff;
    color: white;
    border-radius: 12px 12px 4px 12px;
}

.chat-message.system {
    justify-content: center;
    font-style: italic;
    color: #6c757d;
}

.message-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 500;
    font-size: 11px;
    margin: 0 8px;
    flex-shrink: 0;
}

.message-content {
    max-width: 60%;
    background: #f8f9fa;
    border-radius: 12px 12px 12px 4px;
    padding: 8px 12px;
    word-wrap: break-word;
}

.message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
    font-size: 11px;
}

.message-author {
    font-weight: 600;
    color: #495057;
}

.message-time {
    color: #6c757d;
    font-size: 10px;
}

.message-text {
    line-height: 1.4;
}

#chat-messages {
    max-height: 120px;
    font-size: 13px;
}

.collaborator-item {
    padding: 6px 8px;
    margin: 2px 0;
}

.collaborator-avatar {
    width: 28px;
    height: 28px;
    font-size: 11px;
}

.collaborator-name {
    font-size: 12px;
}

.collaborator-status {
    font-size: 10px;
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', collaborationStyles);