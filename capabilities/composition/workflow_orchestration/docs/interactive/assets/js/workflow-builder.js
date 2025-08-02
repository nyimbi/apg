/**
 * APG Workflow Orchestration - Workflow Builder JavaScript
 * Interactive drag-and-drop workflow builder with canvas manipulation
 */

class WorkflowBuilder {
    constructor() {
        this.canvas = null;
        this.canvasContainer = null;
        this.nodes = new Map();
        this.connections = new Map();
        this.selectedNode = null;
        this.draggedNode = null;
        this.isConnecting = false;
        this.connectionStart = null;
        this.canvasOffset = { x: 0, y: 0 };
        this.zoomLevel = 1;
        this.history = [];
        this.historyIndex = -1;
        this.maxHistory = 50;
        
        this.nodeTypes = {
            start: { icon: 'play', color: '#10b981', label: 'Start' },
            end: { icon: 'stop', color: '#ef4444', label: 'End' },
            task: { icon: 'cog', color: '#2563eb', label: 'Task' },
            decision: { icon: 'question', color: '#f59e0b', label: 'Decision' },
            transform: { icon: 'exchange-alt', color: '#06b6d4', label: 'Transform' },
            filter: { icon: 'filter', color: '#8b5cf6', label: 'Filter' },
            aggregate: { icon: 'calculator', color: '#ec4899', label: 'Aggregate' },
            http: { icon: 'globe', color: '#059669', label: 'HTTP Request' },
            database: { icon: 'database', color: '#dc2626', label: 'Database' },
            email: { icon: 'envelope', color: '#7c3aed', label: 'Email' }
        };
        
        this.init();
    }
    
    init() {
        this.setupCanvas();
        this.setupToolbar();
        this.setupComponentPalette();
        this.setupPropertiesPanel();
        this.bindEvents();
        this.createSampleWorkflow();
    }
    
    setupCanvas() {
        this.canvasContainer = document.getElementById('workflowCanvas');
        if (!this.canvasContainer) return;
        
        // Create SVG canvas
        this.canvas = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.canvas.setAttribute('width', '100%');
        this.canvas.setAttribute('height', '100%');
        this.canvas.style.background = 'radial-gradient(circle, #f1f5f9 1px, transparent 1px)';
        this.canvas.style.backgroundSize = '20px 20px';
        this.canvas.style.cursor = 'grab';
        
        // Create groups for different layers
        this.connectionsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.connectionsGroup.setAttribute('class', 'connections-layer');
        
        this.nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.nodesGroup.setAttribute('class', 'nodes-layer');
        
        this.canvas.appendChild(this.connectionsGroup);
        this.canvas.appendChild(this.nodesGroup);
        this.canvasContainer.appendChild(this.canvas);
    }
    
    setupToolbar() {
        const saveBtn = document.getElementById('saveWorkflow');
        const validateBtn = document.getElementById('validateWorkflow');
        const runBtn = document.getElementById('runWorkflow');
        const undoBtn = document.getElementById('undoAction');
        const redoBtn = document.getElementById('redoAction');
        const zoomInBtn = document.getElementById('zoomIn');
        const zoomOutBtn = document.getElementById('zoomOut');
        const fitBtn = document.getElementById('fitToScreen');
        
        if (saveBtn) saveBtn.addEventListener('click', () => this.saveWorkflow());
        if (validateBtn) validateBtn.addEventListener('click', () => this.validateWorkflow());
        if (runBtn) runBtn.addEventListener('click', () => this.runWorkflow());
        if (undoBtn) undoBtn.addEventListener('click', () => this.undo());
        if (redoBtn) redoBtn.addEventListener('click', () => this.redo());
        if (zoomInBtn) zoomInBtn.addEventListener('click', () => this.zoom(1.2));
        if (zoomOutBtn) zoomOutBtn.addEventListener('click', () => this.zoom(0.8));
        if (fitBtn) fitBtn.addEventListener('click', () => this.fitToScreen());
    }
    
    setupComponentPalette() {
        const componentItems = document.querySelectorAll('.component-item');
        componentItems.forEach(item => {
            item.addEventListener('dragstart', (e) => {
                const nodeType = e.target.getAttribute('data-type');
                e.dataTransfer.setData('text/plain', nodeType);
                e.dataTransfer.effectAllowed = 'copy';
            });
            
            item.setAttribute('draggable', 'true');
        });
    }
    
    setupPropertiesPanel() {
        this.propertiesPanel = document.getElementById('propertiesPanel');
        if (!this.propertiesPanel) return;
        
        this.propertiesContent = this.propertiesPanel.querySelector('.properties-content');
    }
    
    bindEvents() {
        if (!this.canvas) return;
        
        // Canvas drag and drop
        this.canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });
        
        this.canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            const nodeType = e.dataTransfer.getData('text/plain');
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - this.canvasOffset.x) / this.zoomLevel;
            const y = (e.clientY - rect.top - this.canvasOffset.y) / this.zoomLevel;
            
            this.addNode(nodeType, x, y);
        });
        
        // Canvas panning
        let isPanning = false;
        let panStart = { x: 0, y: 0 };
        
        this.canvas.addEventListener('mousedown', (e) => {
            if (e.target === this.canvas) {
                isPanning = true;
                panStart = { x: e.clientX - this.canvasOffset.x, y: e.clientY - this.canvasOffset.y };
                this.canvas.style.cursor = 'grabbing';
            }
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (isPanning) {
                this.canvasOffset.x = e.clientX - panStart.x;
                this.canvasOffset.y = e.clientY - panStart.y;
                this.updateCanvasTransform();
            }
        });
        
        this.canvas.addEventListener('mouseup', () => {
            isPanning = false;
            this.canvas.style.cursor = 'grab';
        });
        
        // Canvas zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.zoom(delta, e.clientX, e.clientY);
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
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
                    case 'a':
                        e.preventDefault();
                        this.selectAll();
                        break;
                }
            } else if (e.key === 'Delete' && this.selectedNode) {
                this.deleteNode(this.selectedNode);
            }
        });
    }
    
    addNode(type, x, y) {
        const nodeId = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const nodeConfig = this.nodeTypes[type];
        
        if (!nodeConfig) return;
        
        const node = {
            id: nodeId,
            type: type,
            x: x,
            y: y,
            width: 120,
            height: 60,
            config: nodeConfig,
            properties: this.getDefaultProperties(type),
            inputs: this.getNodeInputs(type),
            outputs: this.getNodeOutputs(type)
        };
        
        this.nodes.set(nodeId, node);
        this.renderNode(node);
        this.saveToHistory();
        
        return node;
    }
    
    renderNode(node) {
        // Create node group
        const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        nodeGroup.setAttribute('class', 'workflow-node');
        nodeGroup.setAttribute('data-node-id', node.id);
        nodeGroup.setAttribute('transform', `translate(${node.x}, ${node.y})`);
        
        // Node background
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('width', node.width);
        rect.setAttribute('height', node.height);
        rect.setAttribute('rx', '8');
        rect.setAttribute('fill', node.config.color);
        rect.setAttribute('stroke', '#e5e7eb');
        rect.setAttribute('stroke-width', '2');
        
        // Node icon
        const iconGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        iconGroup.setAttribute('transform', `translate(${node.width/2 - 8}, 15)`);
        
        const iconText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        iconText.setAttribute('x', '8');
        iconText.setAttribute('y', '12');
        iconText.setAttribute('text-anchor', 'middle');
        iconText.setAttribute('fill', 'white');
        iconText.setAttribute('font-family', 'FontAwesome');
        iconText.setAttribute('font-size', '16');
        iconText.textContent = this.getIconChar(node.config.icon);
        
        // Node label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', node.width/2);
        label.setAttribute('y', 45);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', 'white');
        label.setAttribute('font-size', '12');
        label.setAttribute('font-weight', '500');
        label.textContent = node.config.label;
        
        // Connection points
        const inputPoint = this.createConnectionPoint(10, node.height/2, 'input');
        const outputPoint = this.createConnectionPoint(node.width - 10, node.height/2, 'output');
        
        nodeGroup.appendChild(rect);
        nodeGroup.appendChild(iconGroup);
        iconGroup.appendChild(iconText);
        nodeGroup.appendChild(label);
        nodeGroup.appendChild(inputPoint);
        nodeGroup.appendChild(outputPoint);
        
        // Bind node events
        this.bindNodeEvents(nodeGroup, node);
        
        this.nodesGroup.appendChild(nodeGroup);
    }
    
    createConnectionPoint(x, y, type) {
        const point = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        point.setAttribute('cx', x);
        point.setAttribute('cy', y);
        point.setAttribute('r', '6');
        point.setAttribute('fill', '#f3f4f6');
        point.setAttribute('stroke', '#6b7280');
        point.setAttribute('stroke-width', '2');
        point.setAttribute('class', `connection-point ${type}`);
        point.style.cursor = 'crosshair';
        
        return point;
    }
    
    bindNodeEvents(nodeGroup, node) {
        let isDragging = false;
        let dragStart = { x: 0, y: 0 };
        
        nodeGroup.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            
            if (e.target.classList.contains('connection-point')) {
                this.startConnection(e.target, node);
                return;
            }
            
            this.selectNode(node);
            isDragging = true;
            dragStart = { x: e.clientX - node.x * this.zoomLevel, y: e.clientY - node.y * this.zoomLevel };
            nodeGroup.style.cursor = 'grabbing';
        });
        
        nodeGroup.addEventListener('mousemove', (e) => {
            if (isDragging) {
                node.x = (e.clientX - dragStart.x) / this.zoomLevel;
                node.y = (e.clientY - dragStart.y) / this.zoomLevel;
                nodeGroup.setAttribute('transform', `translate(${node.x}, ${node.y})`);
                this.updateConnections(node);
            }
        });
        
        nodeGroup.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                nodeGroup.style.cursor = 'grab';
                this.saveToHistory();
            }
        });
        
        nodeGroup.addEventListener('dblclick', () => {
            this.editNodeProperties(node);
        });
    }
    
    startConnection(connectionPoint, node) {
        this.isConnecting = true;
        this.connectionStart = {
            node: node,
            type: connectionPoint.classList.contains('input') ? 'input' : 'output',
            point: connectionPoint
        };
        
        // Create temporary connection line
        this.tempConnection = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        this.tempConnection.setAttribute('stroke', '#2563eb');
        this.tempConnection.setAttribute('stroke-width', '2');
        this.tempConnection.setAttribute('fill', 'none');
        this.tempConnection.setAttribute('stroke-dasharray', '5,5');
        this.connectionsGroup.appendChild(this.tempConnection);
        
        // Follow mouse movement
        const handleMouseMove = (e) => {
            if (this.tempConnection) {
                const rect = this.canvas.getBoundingClientRect();
                const endX = (e.clientX - rect.left - this.canvasOffset.x) / this.zoomLevel;
                const endY = (e.clientY - rect.top - this.canvasOffset.y) / this.zoomLevel;
                
                const startX = node.x + (this.connectionStart.type === 'output' ? node.width - 10 : 10);
                const startY = node.y + node.height / 2;
                
                const path = this.createConnectionPath(startX, startY, endX, endY);
                this.tempConnection.setAttribute('d', path);
            }
        };
        
        const handleMouseUp = (e) => {
            this.finishConnection(e);
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
        
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    }
    
    finishConnection(e) {
        if (this.tempConnection) {
            this.tempConnection.remove();
            this.tempConnection = null;
        }
        
        // Check if we're over a valid connection point
        const target = e.target;
        if (target && target.classList.contains('connection-point')) {
            const targetNodeGroup = target.closest('[data-node-id]');
            const targetNodeId = targetNodeGroup.getAttribute('data-node-id');
            const targetNode = this.nodes.get(targetNodeId);
            
            if (targetNode && targetNode !== this.connectionStart.node) {
                const targetType = target.classList.contains('input') ? 'input' : 'output';
                
                // Validate connection (output to input)
                if (this.connectionStart.type === 'output' && targetType === 'input') {
                    this.createConnection(this.connectionStart.node, targetNode);
                } else if (this.connectionStart.type === 'input' && targetType === 'output') {
                    this.createConnection(targetNode, this.connectionStart.node);
                }
            }
        }
        
        this.isConnecting = false;
        this.connectionStart = null;
    }
    
    createConnection(fromNode, toNode) {
        const connectionId = `${fromNode.id}_to_${toNode.id}`;
        
        // Check if connection already exists
        if (this.connections.has(connectionId)) return;
        
        const connection = {
            id: connectionId,
            from: fromNode,
            to: toNode
        };
        
        this.connections.set(connectionId, connection);
        this.renderConnection(connection);
        this.saveToHistory();
    }
    
    renderConnection(connection) {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'workflow-connection');
        path.setAttribute('data-connection-id', connection.id);
        path.setAttribute('stroke', '#6b7280');
        path.setAttribute('stroke-width', '2');
        path.setAttribute('fill', 'none');
        path.setAttribute('marker-end', 'url(#arrowhead)');
        
        this.updateConnectionPath(connection, path);
        
        // Add click handler for connection selection/deletion
        path.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectConnection(connection, path);
        });
        
        this.connectionsGroup.appendChild(path);
        
        // Create arrowhead marker if it doesn't exist
        this.createArrowheadMarker();
    }
    
    createArrowheadMarker() {
        if (this.canvas.querySelector('#arrowhead')) return;
        
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'arrowhead');
        marker.setAttribute('markerWidth', '10');
        marker.setAttribute('markerHeight', '7');
        marker.setAttribute('refX', '10');
        marker.setAttribute('refY', '3.5');
        marker.setAttribute('orient', 'auto');
        
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
        polygon.setAttribute('fill', '#6b7280');
        
        marker.appendChild(polygon);
        defs.appendChild(marker);
        this.canvas.insertBefore(defs, this.connectionsGroup);
    }
    
    updateConnectionPath(connection, pathElement) {
        const fromX = connection.from.x + connection.from.width - 10;
        const fromY = connection.from.y + connection.from.height / 2;
        const toX = connection.to.x + 10;
        const toY = connection.to.y + connection.to.height / 2;
        
        const path = this.createConnectionPath(fromX, fromY, toX, toY);
        pathElement.setAttribute('d', path);
    }
    
    createConnectionPath(x1, y1, x2, y2) {
        const dx = x2 - x1;
        const dy = y2 - y1;
        const controlOffset = Math.max(50, Math.abs(dx) * 0.5);
        
        return `M ${x1} ${y1} C ${x1 + controlOffset} ${y1}, ${x2 - controlOffset} ${y2}, ${x2} ${y2}`;
    }
    
    updateConnections(node) {
        this.connections.forEach(connection => {
            if (connection.from === node || connection.to === node) {
                const pathElement = this.canvas.querySelector(`[data-connection-id="${connection.id}"]`);
                if (pathElement) {
                    this.updateConnectionPath(connection, pathElement);
                }
            }
        });
    }
    
    selectNode(node) {
        // Remove previous selection
        this.canvas.querySelectorAll('.workflow-node').forEach(n => {
            n.classList.remove('selected');
        });
        
        // Select new node
        const nodeElement = this.canvas.querySelector(`[data-node-id="${node.id}"]`);
        if (nodeElement) {
            nodeElement.classList.add('selected');
        }
        
        this.selectedNode = node;
        this.updatePropertiesPanel(node);
    }
    
    updatePropertiesPanel(node) {
        if (!this.propertiesContent) return;
        
        const properties = node.properties || {};
        
        this.propertiesContent.innerHTML = `
            <div class="property-group">
                <h5>Basic Information</h5>
                <div class="property-item">
                    <label>Name:</label>
                    <input type="text" value="${node.config.label}" data-property="name">
                </div>
                <div class="property-item">
                    <label>Type:</label>
                    <span class="property-value">${node.type}</span>
                </div>
                <div class="property-item">
                    <label>ID:</label>
                    <span class="property-value">${node.id}</span>
                </div>
            </div>
            
            <div class="property-group">
                <h5>Position</h5>
                <div class="property-item">
                    <label>X:</label>
                    <input type="number" value="${Math.round(node.x)}" data-property="x">
                </div>
                <div class="property-item">
                    <label>Y:</label>
                    <input type="number" value="${Math.round(node.y)}" data-property="y">
                </div>
            </div>
            
            ${this.getTypeSpecificProperties(node)}
        `;
        
        // Bind property change events
        this.propertiesContent.querySelectorAll('input, select, textarea').forEach(input => {
            input.addEventListener('change', (e) => {
                this.updateNodeProperty(node, e.target.getAttribute('data-property'), e.target.value);
            });
        });
    }
    
    getTypeSpecificProperties(node) {
        switch (node.type) {
            case 'http':
                return `
                    <div class="property-group">
                        <h5>HTTP Configuration</h5>
                        <div class="property-item">
                            <label>Method:</label>
                            <select data-property="method">
                                <option value="GET">GET</option>
                                <option value="POST">POST</option>
                                <option value="PUT">PUT</option>
                                <option value="DELETE">DELETE</option>
                            </select>
                        </div>
                        <div class="property-item">
                            <label>URL:</label>
                            <input type="url" placeholder="https://api.example.com/endpoint" data-property="url">
                        </div>
                        <div class="property-item">
                            <label>Headers:</label>
                            <textarea placeholder="Content-Type: application/json" data-property="headers"></textarea>
                        </div>
                    </div>
                `;
            case 'database':
                return `
                    <div class="property-group">
                        <h5>Database Configuration</h5>
                        <div class="property-item">
                            <label>Operation:</label>
                            <select data-property="operation">
                                <option value="select">SELECT</option>
                                <option value="insert">INSERT</option>
                                <option value="update">UPDATE</option>
                                <option value="delete">DELETE</option>
                            </select>
                        </div>
                        <div class="property-item">
                            <label>Query:</label>
                            <textarea placeholder="SELECT * FROM users" data-property="query"></textarea>
                        </div>
                    </div>
                `;
            case 'email':
                return `
                    <div class="property-group">
                        <h5>Email Configuration</h5>
                        <div class="property-item">
                            <label>To:</label>
                            <input type="email" placeholder="recipient@example.com" data-property="to">
                        </div>
                        <div class="property-item">
                            <label>Subject:</label>
                            <input type="text" placeholder="Email subject" data-property="subject">
                        </div>
                        <div class="property-item">
                            <label>Template:</label>
                            <select data-property="template">
                                <option value="default">Default Template</option>
                                <option value="notification">Notification</option>
                                <option value="alert">Alert</option>
                            </select>
                        </div>
                    </div>
                `;
            default:
                return '';
        }
    }
    
    updateNodeProperty(node, property, value) {
        if (property === 'x' || property === 'y') {
            node[property] = parseFloat(value);
            const nodeElement = this.canvas.querySelector(`[data-node-id="${node.id}"]`);
            if (nodeElement) {
                nodeElement.setAttribute('transform', `translate(${node.x}, ${node.y})`);
                this.updateConnections(node);
            }
        } else if (property === 'name') {
            node.config.label = value;
            const labelElement = this.canvas.querySelector(`[data-node-id="${node.id}"] text:last-child`);
            if (labelElement) {
                labelElement.textContent = value;
            }
        } else {
            if (!node.properties) node.properties = {};
            node.properties[property] = value;
        }
        
        this.saveToHistory();
    }
    
    deleteNode(node) {
        // Remove connections
        const connectionsToRemove = [];
        this.connections.forEach((connection, id) => {
            if (connection.from === node || connection.to === node) {
                connectionsToRemove.push(id);
            }
        });
        
        connectionsToRemove.forEach(id => {
            this.connections.delete(id);
            const pathElement = this.canvas.querySelector(`[data-connection-id="${id}"]`);
            if (pathElement) pathElement.remove();
        });
        
        // Remove node
        this.nodes.delete(node.id);
        const nodeElement = this.canvas.querySelector(`[data-node-id="${node.id}"]`);
        if (nodeElement) nodeElement.remove();
        
        // Clear selection
        if (this.selectedNode === node) {
            this.selectedNode = null;
            this.clearPropertiesPanel();
        }
        
        this.saveToHistory();
    }
    
    clearPropertiesPanel() {
        if (this.propertiesContent) {
            this.propertiesContent.innerHTML = '<p class="text-muted">Select a component to view its properties</p>';
        }
    }
    
    zoom(factor, centerX, centerY) {
        const newZoom = Math.max(0.1, Math.min(3, this.zoomLevel * factor));
        
        if (centerX && centerY) {
            // Zoom towards a specific point
            const rect = this.canvas.getBoundingClientRect();
            const canvasX = centerX - rect.left;
            const canvasY = centerY - rect.top;
            
            this.canvasOffset.x = canvasX - (canvasX - this.canvasOffset.x) * (newZoom / this.zoomLevel);
            this.canvasOffset.y = canvasY - (canvasY - this.canvasOffset.y) * (newZoom / this.zoomLevel);
        }
        
        this.zoomLevel = newZoom;
        this.updateCanvasTransform();
    }
    
    updateCanvasTransform() {
        if (this.nodesGroup && this.connectionsGroup) {
            const transform = `translate(${this.canvasOffset.x}, ${this.canvasOffset.y}) scale(${this.zoomLevel})`;
            this.nodesGroup.setAttribute('transform', transform);
            this.connectionsGroup.setAttribute('transform', transform);
        }
    }
    
    fitToScreen() {
        if (this.nodes.size === 0) return;
        
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        this.nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + node.width);
            maxY = Math.max(maxY, node.y + node.height);
        });
        
        const padding = 50;
        const contentWidth = maxX - minX + padding * 2;
        const contentHeight = maxY - minY + padding * 2;
        
        const canvasRect = this.canvas.getBoundingClientRect();
        const scaleX = canvasRect.width / contentWidth;
        const scaleY = canvasRect.height / contentHeight;
        
        this.zoomLevel = Math.min(scaleX, scaleY, 1);
        this.canvasOffset.x = (canvasRect.width - contentWidth * this.zoomLevel) / 2 - (minX - padding) * this.zoomLevel;
        this.canvasOffset.y = (canvasRect.height - contentHeight * this.zoomLevel) / 2 - (minY - padding) * this.zoomLevel;
        
        this.updateCanvasTransform();
    }
    
    saveToHistory() {
        const state = this.serializeWorkflow();
        
        // Remove any future history if we're not at the end
        this.history = this.history.slice(0, this.historyIndex + 1);
        
        // Add new state
        this.history.push(JSON.stringify(state));
        this.historyIndex++;
        
        // Limit history size
        if (this.history.length > this.maxHistory) {
            this.history.shift();
            this.historyIndex--;
        }
    }
    
    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            const state = JSON.parse(this.history[this.historyIndex]);
            this.loadWorkflow(state);
        }
    }
    
    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            const state = JSON.parse(this.history[this.historyIndex]);
            this.loadWorkflow(state);
        }
    }
    
    serializeWorkflow() {
        const nodes = [];
        const connections = [];
        
        this.nodes.forEach(node => {
            nodes.push({
                id: node.id,
                type: node.type,
                x: node.x,
                y: node.y,
                properties: node.properties
            });
        });
        
        this.connections.forEach(connection => {
            connections.push({
                id: connection.id,
                from: connection.from.id,
                to: connection.to.id
            });
        });
        
        return { nodes, connections };
    }
    
    loadWorkflow(data) {
        // Clear existing workflow
        this.clearCanvas();
        
        // Load nodes
        data.nodes.forEach(nodeData => {
            const node = this.addNode(nodeData.type, nodeData.x, nodeData.y);
            if (node) {
                node.id = nodeData.id;
                node.properties = nodeData.properties || {};
                
                // Update the DOM element ID
                const nodeElement = this.canvas.querySelector(`[data-node-id="${node.id}"]`);
                if (nodeElement) {
                    nodeElement.setAttribute('data-node-id', nodeData.id);
                }
                
                // Update the nodes map
                this.nodes.delete(node.id);
                this.nodes.set(nodeData.id, node);
            }
        });
        
        // Load connections
        data.connections.forEach(connData => {
            const fromNode = this.nodes.get(connData.from);
            const toNode = this.nodes.get(connData.to);
            if (fromNode && toNode) {
                this.createConnection(fromNode, toNode);
            }
        });
    }
    
    clearCanvas() {
        this.nodes.clear();
        this.connections.clear();
        this.selectedNode = null;
        this.nodesGroup.innerHTML = '';
        this.connectionsGroup.innerHTML = '';
        this.clearPropertiesPanel();
    }
    
    createSampleWorkflow() {
        // Create a sample workflow to demonstrate functionality
        const startNode = this.addNode('start', 100, 200);
        const httpNode = this.addNode('http', 300, 200);
        const transformNode = this.addNode('transform', 500, 200);
        const endNode = this.addNode('end', 700, 200);
        
        if (startNode && httpNode) {
            this.createConnection(startNode, httpNode);
        }
        if (httpNode && transformNode) {
            this.createConnection(httpNode, transformNode);
        }
        if (transformNode && endNode) {
            this.createConnection(transformNode, endNode);
        }
        
        // Save initial state
        this.saveToHistory();
    }
    
    saveWorkflow() {
        const workflow = this.serializeWorkflow();
        const blob = new Blob([JSON.stringify(workflow, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'workflow.json';
        a.click();
        
        URL.revokeObjectURL(url);
        this.showToast('Workflow saved successfully!', 'success');
    }
    
    validateWorkflow() {
        const errors = [];
        
        // Check for start and end nodes
        const hasStart = Array.from(this.nodes.values()).some(node => node.type === 'start');
        const hasEnd = Array.from(this.nodes.values()).some(node => node.type === 'end');
        
        if (!hasStart) errors.push('Workflow must have a start node');
        if (!hasEnd) errors.push('Workflow must have an end node');
        
        // Check for orphaned nodes
        this.nodes.forEach(node => {
            if (node.type !== 'start' && node.type !== 'end') {
                const hasIncoming = Array.from(this.connections.values()).some(conn => conn.to === node);
                const hasOutgoing = Array.from(this.connections.values()).some(conn => conn.from === node);
                
                if (!hasIncoming) errors.push(`Node "${node.config.label}" has no incoming connections`);
                if (!hasOutgoing) errors.push(`Node "${node.config.label}" has no outgoing connections`);
            }
        });
        
        if (errors.length === 0) {
            this.showToast('Workflow is valid!', 'success');
        } else {
            this.showValidationErrors(errors);
        }
    }
    
    runWorkflow() {
        this.showToast('Workflow execution started...', 'info');
        
        // Simulate workflow execution
        setTimeout(() => {
            this.showToast('Workflow completed successfully!', 'success');
        }, 2000);
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    showValidationErrors(errors) {
        const modal = document.createElement('div');
        modal.className = 'validation-modal';
        modal.innerHTML = `
            <div class="modal-overlay"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-exclamation-triangle"></i> Validation Errors</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <ul class="error-list">
                        ${errors.map(error => `<li><i class="fas fa-times"></i> ${error}</li>`).join('')}
                    </ul>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary modal-close">Close</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        setTimeout(() => modal.classList.add('show'), 10);
        
        modal.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', () => {
                modal.classList.remove('show');
                setTimeout(() => modal.remove(), 300);
            });
        });
    }
    
    getDefaultProperties(type) {
        const defaults = {
            http: { method: 'GET', url: '', headers: '' },
            database: { operation: 'select', query: '' },
            email: { to: '', subject: '', template: 'default' }
        };
        return defaults[type] || {};
    }
    
    getNodeInputs(type) {
        return type === 'start' ? [] : ['input'];
    }
    
    getNodeOutputs(type) {
        return type === 'end' ? [] : ['output'];
    }
    
    getIconChar(iconName) {
        const iconMap = {
            'play': '▶',
            'stop': '⏹',
            'cog': '⚙',
            'question': '?',
            'exchange-alt': '⇄',
            'filter': '⚗',
            'calculator': '∑',
            'globe': '🌐',
            'database': '💾',
            'envelope': '✉'
        };
        return iconMap[iconName] || '⚡';
    }
}

// Initialize workflow builder when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.workflowBuilder = new WorkflowBuilder();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WorkflowBuilder;
}