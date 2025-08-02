/**
 * APG Workflow Canvas Engine
 * 
 * High-performance SVG-based canvas for workflow visualization.
 * 
 * © 2025 Datacraft. All rights reserved.
 * Author: Nyimbi Odero <nyimbi@gmail.com>
 */

class CanvasEngine {
    constructor(designer) {
        this.designer = designer;
        this.svg = document.getElementById('workflow-canvas');
        this.contentGroup = document.getElementById('canvas-content');
        this.nodesLayer = document.getElementById('nodes-layer');
        this.connectionsLayer = document.getElementById('connections-layer');
        this.selectionLayer = document.getElementById('selection-layer');
        
        // Canvas state
        this.transform = {
            x: 0,
            y: 0,
            scale: 1
        };
        
        this.isDragging = false;
        this.isPanning = false;
        this.isConnecting = false;
        this.dragStartPoint = null;
        this.dragTarget = null;
        this.connectionPreview = null;
        
        // Grid settings
        this.gridSize = 20;
        this.snapToGrid = true;
        this.showGrid = true;
        this.showMinimap = true;
        
        // Performance optimization
        this.viewportBounds = { x: 0, y: 0, width: 0, height: 0 };
        this.visibleNodes = new Set();
        
        this.initialize();
    }
    
    initialize() {
        this.setupEventHandlers();
        this.updateViewport();
        this.setupMinimap();
        
        // Initial canvas setup
        this.centerView();
        
        console.log('Canvas engine initialized');
    }
    
    setupEventHandlers() {
        // Mouse events
        this.svg.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.svg.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.svg.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.svg.addEventListener('wheel', (e) => this.onWheel(e));
        this.svg.addEventListener('contextmenu', (e) => this.onContextMenu(e));
        
        // Drag and drop from component palette
        this.svg.addEventListener('dragover', (e) => this.onDragOver(e));
        this.svg.addEventListener('drop', (e) => this.onDrop(e));
        
        // Resize handling
        window.addEventListener('resize', () => this.updateViewport());
        
        // Keyboard events for canvas
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
    }
    
    onMouseDown(e) {
        e.preventDefault();
        
        const point = this.screenToCanvas(e.clientX, e.clientY);
        const target = e.target;
        
        // Check what was clicked
        if (target.classList.contains('node-rect') || target.parentElement.classList.contains('workflow-node')) {
            this.startNodeDrag(target, point, e);
        } else if (target.classList.contains('node-port')) {
            this.startConnection(target, point);
        } else if (target.classList.contains('workflow-connection')) {
            this.selectConnection(target.dataset.connectionId);
        } else {
            this.startCanvasDrag(point, e);
        }
        
        this.dragStartPoint = point;
    }
    
    onMouseMove(e) {
        if (!this.isDragging && !this.isPanning && !this.isConnecting) return;
        
        const point = this.screenToCanvas(e.clientX, e.clientY);
        
        if (this.isDragging && this.dragTarget) {
            this.handleNodeDrag(point);
        } else if (this.isPanning) {
            this.handleCanvasPan(point);
        } else if (this.isConnecting) {
            this.updateConnectionPreview(point);
        }
    }
    
    onMouseUp(e) {
        const point = this.screenToCanvas(e.clientX, e.clientY);
        
        if (this.isDragging && this.dragTarget) {
            this.endNodeDrag(point);
        } else if (this.isPanning) {
            this.endCanvasPan();
        } else if (this.isConnecting) {
            this.endConnection(e.target, point);
        }
        
        this.isDragging = false;
        this.isPanning = false;
        this.isConnecting = false;
        this.dragTarget = null;
        this.dragStartPoint = null;
    }
    
    onWheel(e) {
        e.preventDefault();
        
        const point = this.screenToCanvas(e.clientX, e.clientY);
        const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
        
        this.zoomAt(point, scaleFactor);
    }
    
    onContextMenu(e) {
        e.preventDefault();
        
        const point = this.screenToCanvas(e.clientX, e.clientY);
        this.showContextMenu(e.clientX, e.clientY, point);
    }
    
    onDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    }
    
    onDrop(e) {
        e.preventDefault();
        
        const componentType = e.dataTransfer.getData('text/plain');
        if (!componentType) return;
        
        const point = this.screenToCanvas(e.clientX, e.clientY);
        const snappedPoint = this.snapToGrid ? this.snapPoint(point) : point;
        
        // Add component to workflow
        this.designer.addNode(componentType, snappedPoint);
    }
    
    onKeyDown(e) {
        // Canvas-specific keyboard shortcuts
        switch (e.key) {
            case ' ':
                if (!this.isPanning) {
                    e.preventDefault();
                    this.svg.style.cursor = 'grab';
                }
                break;
        }
    }
    
    // === Node Management ===
    
    addNode(node) {
        const nodeElement = this.createNodeElement(node);
        this.nodesLayer.appendChild(nodeElement);
        
        // Update visible nodes
        this.visibleNodes.add(node.id);
        
        // Animate node appearance
        this.animateNodeIn(nodeElement);
        
        return nodeElement;
    }
    
    removeNode(nodeId) {
        const nodeElement = document.getElementById(`node-${nodeId}`);
        if (nodeElement) {
            this.animateNodeOut(nodeElement, () => {
                nodeElement.remove();
            });
        }
        
        this.visibleNodes.delete(nodeId);
    }
    
    moveNode(nodeId, position) {
        const nodeElement = document.getElementById(`node-${nodeId}`);
        if (nodeElement) {
            const snappedPosition = this.snapToGrid ? this.snapPoint(position) : position;
            
            nodeElement.setAttribute('transform', `translate(${snappedPosition.x}, ${snappedPosition.y})`);
            
            // Update connected connections
            this.updateNodeConnections(nodeId);
        }
    }
    
    selectNode(nodeId) {
        const nodeElement = document.getElementById(`node-${nodeId}`);
        if (nodeElement) {
            nodeElement.classList.add('selected');
        }
    }
    
    createNodeElement(node) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.id = `node-${node.id}`;
        group.className = 'workflow-node';
        group.dataset.nodeId = node.id;
        group.setAttribute('transform', `translate(${node.position.x}, ${node.position.y})`);
        
        // Node rectangle
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.className = `node-rect ${this.getNodeTypeClass(node.component_type)}`;
        rect.setAttribute('width', node.size?.width || 200);
        rect.setAttribute('height', node.size?.height || 100);
        rect.setAttribute('rx', 8);
        
        // Node icon
        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        icon.className = 'node-icon';
        icon.setAttribute('x', 20);
        icon.setAttribute('y', 30);
        icon.textContent = this.getNodeIcon(node.component_type);
        
        // Node text
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.className = 'node-text';
        text.setAttribute('x', (node.size?.width || 200) / 2);
        text.setAttribute('y', (node.size?.height || 100) / 2 + 5);
        text.textContent = node.label || node.component_type;
        
        // Input ports
        const inputPorts = this.createPorts(node, 'input');
        
        // Output ports  
        const outputPorts = this.createPorts(node, 'output');
        
        // Assemble node
        group.appendChild(rect);
        group.appendChild(icon);
        group.appendChild(text);
        inputPorts.forEach(port => group.appendChild(port));
        outputPorts.forEach(port => group.appendChild(port));
        
        return group;
    }
    
    createPorts(node, type) {
        const ports = [];
        const portData = type === 'input' ? node.input_ports : node.output_ports;
        const nodeWidth = node.size?.width || 200;
        const nodeHeight = node.size?.height || 100;
        
        if (portData && portData.length > 0) {
            portData.forEach((port, index) => {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.className = `node-port ${type}`;
                circle.dataset.portName = port.name;
                circle.dataset.nodeId = node.id;
                circle.setAttribute('r', 4);
                
                if (type === 'input') {
                    circle.setAttribute('cx', -4);
                    circle.setAttribute('cy', (index + 1) * (nodeHeight / (portData.length + 1)));
                } else {
                    circle.setAttribute('cx', nodeWidth + 4);
                    circle.setAttribute('cy', (index + 1) * (nodeHeight / (portData.length + 1)));
                }
                
                ports.push(circle);
            });
        } else {
            // Default single port
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.className = `node-port ${type}`;
            circle.dataset.portName = type;
            circle.dataset.nodeId = node.id;
            circle.setAttribute('r', 4);
            
            if (type === 'input') {
                circle.setAttribute('cx', -4);
                circle.setAttribute('cy', nodeHeight / 2);
            } else {
                circle.setAttribute('cx', nodeWidth + 4);
                circle.setAttribute('cy', nodeHeight / 2);
            }
            
            ports.push(circle);
        }
        
        return ports;
    }
    
    getNodeTypeClass(componentType) {
        if (componentType.includes('trigger')) return 'trigger';
        if (componentType.includes('data')) return 'data';
        if (componentType.includes('logic') || componentType.includes('condition')) return 'logic';
        if (componentType.includes('integration') || componentType.includes('http') || componentType.includes('database')) return 'integration';
        if (componentType.includes('ai') || componentType.includes('ml')) return 'ai';
        return 'utility';
    }
    
    getNodeIcon(componentType) {
        const iconMap = {
            'http_trigger': '\uf0ac',        // fa-globe
            'schedule_trigger': '\uf017',     // fa-clock
            'data_transform': '\uf362',       // fa-exchange-alt
            'filter_data': '\uf0b0',          // fa-filter
            'condition': '\uf126',            // fa-code-branch
            'loop': '\uf01e',                 // fa-repeat
            'http_request': '\uf35d',         // fa-arrow-right
            'database_query': '\uf1c0'        // fa-database
        };
        
        return iconMap[componentType] || '\uf013'; // fa-cog default
    }
    
    // === Connection Management ===
    
    addConnection(connection) {
        const connectionElement = this.createConnectionElement(connection);
        this.connectionsLayer.appendChild(connectionElement);
        return connectionElement;
    }
    
    removeConnection(connectionId) {
        const connectionElement = document.getElementById(`connection-${connectionId}`);
        if (connectionElement) {
            connectionElement.remove();
        }
    }
    
    selectConnection(connectionId) {
        const connectionElement = document.getElementById(`connection-${connectionId}`);
        if (connectionElement) {
            connectionElement.classList.add('selected');
        }
        
        this.designer.selectConnection(connectionId);
    }
    
    createConnectionElement(connection) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.id = `connection-${connection.id}`;
        group.className = 'workflow-connection-group';
        group.dataset.connectionId = connection.id;
        
        // Get source and target positions
        const sourcePos = this.getPortPosition(connection.source_node_id, connection.source_port, 'output');
        const targetPos = this.getPortPosition(connection.target_node_id, connection.target_port, 'input');
        
        // Create path
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.className = 'workflow-connection';
        path.dataset.connectionId = connection.id;
        
        const pathData = this.createConnectionPath(sourcePos, targetPos);
        path.setAttribute('d', pathData);
        
        // Create arrow marker
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        marker.className = 'connection-arrow';
        marker.setAttribute('points', '0,0 -8,-4 -8,4');
        marker.setAttribute('transform', `translate(${targetPos.x}, ${targetPos.y}) rotate(${this.getArrowRotation(sourcePos, targetPos)})`);
        
        group.appendChild(path);
        group.appendChild(marker);
        
        return group;
    }
    
    createConnectionPath(source, target) {
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        
        // Bezier curve for smooth connections
        const cp1x = source.x + Math.max(100, Math.abs(dx) * 0.5);
        const cp1y = source.y;
        const cp2x = target.x - Math.max(100, Math.abs(dx) * 0.5);
        const cp2y = target.y;
        
        return `M ${source.x} ${source.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${target.x} ${target.y}`;
    }
    
    getPortPosition(nodeId, portName, portType) {
        const nodeElement = document.getElementById(`node-${nodeId}`);
        if (!nodeElement) return { x: 0, y: 0 };
        
        const transform = nodeElement.getAttribute('transform');
        const match = transform.match(/translate\(([^,]+),([^)]+)\)/);
        const nodeX = match ? parseFloat(match[1]) : 0;
        const nodeY = match ? parseFloat(match[2]) : 0;
        
        // Find the specific port
        const port = nodeElement.querySelector(`.node-port.${portType}[data-port-name="${portName}"]`);
        if (port) {
            const portX = parseFloat(port.getAttribute('cx'));
            const portY = parseFloat(port.getAttribute('cy'));
            return {
                x: nodeX + portX,
                y: nodeY + portY
            };
        }
        
        // Default port position
        const node = this.designer.nodes.get(nodeId);
        const nodeWidth = node?.size?.width || 200;
        const nodeHeight = node?.size?.height || 100;
        
        return {
            x: nodeX + (portType === 'output' ? nodeWidth + 4 : -4),
            y: nodeY + nodeHeight / 2
        };
    }
    
    getArrowRotation(source, target) {
        const angle = Math.atan2(target.y - source.y, target.x - source.x);
        return (angle * 180 / Math.PI);
    }
    
    updateNodeConnections(nodeId) {
        // Update all connections connected to this node
        this.designer.connections.forEach(connection => {
            if (connection.source_node_id === nodeId || connection.target_node_id === nodeId) {
                const connectionElement = document.getElementById(`connection-${connection.id}`);
                if (connectionElement) {
                    const path = connectionElement.querySelector('.workflow-connection');
                    const arrow = connectionElement.querySelector('.connection-arrow');
                    
                    const sourcePos = this.getPortPosition(connection.source_node_id, connection.source_port, 'output');
                    const targetPos = this.getPortPosition(connection.target_node_id, connection.target_port, 'input');
                    
                    path.setAttribute('d', this.createConnectionPath(sourcePos, targetPos));
                    arrow.setAttribute('transform', `translate(${targetPos.x}, ${targetPos.y}) rotate(${this.getArrowRotation(sourcePos, targetPos)})`);
                }
            }
        });
    }
    
    // === Drag and Drop ===
    
    startNodeDrag(target, point, event) {
        this.isDragging = true;
        this.dragTarget = target.closest('.workflow-node');
        this.svg.style.cursor = 'grabbing';
        
        const nodeId = this.dragTarget.dataset.nodeId;
        
        // Select node if not already selected
        if (!event.ctrlKey && !event.metaKey) {
            this.designer.selectNode(nodeId);
        } else {
            this.designer.selectNode(nodeId, true);
        }
        
        // Add dragging class
        this.dragTarget.classList.add('dragging');
    }
    
    handleNodeDrag(point) {
        if (!this.dragTarget || !this.dragStartPoint) return;
        
        const dx = point.x - this.dragStartPoint.x;
        const dy = point.y - this.dragStartPoint.y;
        
        // Move all selected nodes
        this.designer.selectedNodes.forEach(nodeId => {
            const nodeElement = document.getElementById(`node-${nodeId}`);
            if (nodeElement) {
                const node = this.designer.nodes.get(nodeId);
                const newPosition = {
                    x: node.position.x + dx,
                    y: node.position.y + dy
                };
                
                const snappedPosition = this.snapToGrid ? this.snapPoint(newPosition) : newPosition;
                nodeElement.setAttribute('transform', `translate(${snappedPosition.x}, ${snappedPosition.y})`);
                
                // Update connections
                this.updateNodeConnections(nodeId);
            }
        });
    }
    
    endNodeDrag(point) {
        if (!this.dragTarget || !this.dragStartPoint) return;
        
        const dx = point.x - this.dragStartPoint.x;
        const dy = point.y - this.dragStartPoint.y;
        
        // Update node positions in backend
        this.designer.selectedNodes.forEach(nodeId => {
            const node = this.designer.nodes.get(nodeId);
            const newPosition = {
                x: node.position.x + dx,
                y: node.position.y + dy
            };
            
            const snappedPosition = this.snapToGrid ? this.snapPoint(newPosition) : newPosition;
            this.designer.moveNode(nodeId, snappedPosition);
        });
        
        // Remove dragging class
        this.dragTarget.classList.remove('dragging');
        this.svg.style.cursor = 'default';
    }
    
    startCanvasDrag(point, event) {
        if (event.button === 0 && !event.ctrlKey && !event.metaKey) {
            // Left click - start selection rectangle
            this.startSelectionRectangle(point);
        } else if (event.button === 1 || event.button === 0 && (event.ctrlKey || event.metaKey)) {
            // Middle click or Ctrl+left click - start panning
            this.isPanning = true;
            this.svg.style.cursor = 'grabbing';
        }
    }
    
    handleCanvasPan(point) {
        if (!this.dragStartPoint) return;
        
        const dx = (point.x - this.dragStartPoint.x) * this.transform.scale;
        const dy = (point.y - this.dragStartPoint.y) * this.transform.scale;
        
        this.transform.x += dx;
        this.transform.y += dy;
        
        this.updateTransform();
        this.updateMinimap();
    }
    
    endCanvasPan() {
        this.svg.style.cursor = 'default';
    }
    
    // === Connection Creation ===
    
    startConnection(portElement, point) {
        this.isConnecting = true;
        this.connectionPreview = {
            sourceNodeId: portElement.dataset.nodeId,
            sourcePort: portElement.dataset.portName,
            sourcePoint: point
        };
        
        // Show connection preview
        const previewSvg = document.getElementById('connection-preview');
        const previewLine = document.getElementById('preview-line');
        
        previewSvg.style.display = 'block';
        previewLine.setAttribute('x1', point.x);
        previewLine.setAttribute('y1', point.y);
        previewLine.setAttribute('x2', point.x);
        previewLine.setAttribute('y2', point.y);
    }
    
    updateConnectionPreview(point) {
        if (!this.connectionPreview) return;
        
        const previewLine = document.getElementById('preview-line');
        previewLine.setAttribute('x2', point.x);
        previewLine.setAttribute('y2', point.y);
    }
    
    endConnection(targetElement, point) {
        if (!this.connectionPreview) return;
        
        // Hide connection preview
        document.getElementById('connection-preview').style.display = 'none';
        
        // Check if dropped on a valid port
        if (targetElement.classList.contains('node-port')) {
            const targetNodeId = targetElement.dataset.nodeId;
            const targetPort = targetElement.dataset.portName;
            
            // Validate connection
            if (targetNodeId !== this.connectionPreview.sourceNodeId) {
                this.designer.addConnection(
                    this.connectionPreview.sourceNodeId,
                    targetNodeId,
                    this.connectionPreview.sourcePort,
                    targetPort
                );
            }
        }
        
        this.connectionPreview = null;
    }
    
    // === Transform and Zoom ===
    
    updateTransform() {
        this.contentGroup.setAttribute('transform', 
            `translate(${this.transform.x}, ${this.transform.y}) scale(${this.transform.scale})`
        );
    }
    
    zoomIn() {
        this.zoomAt({ x: this.viewportBounds.width / 2, y: this.viewportBounds.height / 2 }, 1.2);
    }
    
    zoomOut() {
        this.zoomAt({ x: this.viewportBounds.width / 2, y: this.viewportBounds.height / 2 }, 0.8);
    }
    
    zoomAt(point, scaleFactor) {
        const newScale = Math.max(0.1, Math.min(5.0, this.transform.scale * scaleFactor));
        
        if (newScale === this.transform.scale) return;
        
        // Zoom towards the point
        const scaleChange = newScale / this.transform.scale;
        this.transform.x = point.x - (point.x - this.transform.x) * scaleChange;
        this.transform.y = point.y - (point.y - this.transform.y) * scaleChange;
        this.transform.scale = newScale;
        
        this.updateTransform();
        this.updateMinimap();
    }
    
    fitToScreen() {
        if (this.designer.nodes.size === 0) return;
        
        // Calculate bounds of all nodes
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        this.designer.nodes.forEach(node => {
            const x = node.position.x;
            const y = node.position.y;
            const w = node.size?.width || 200;
            const h = node.size?.height || 100;
            
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x + w);
            maxY = Math.max(maxY, y + h);
        });
        
        // Add padding
        const padding = 50;
        minX -= padding;
        minY -= padding;
        maxX += padding;
        maxY += padding;
        
        // Calculate scale to fit
        const contentWidth = maxX - minX;
        const contentHeight = maxY - minY;
        const scaleX = this.viewportBounds.width / contentWidth;
        const scaleY = this.viewportBounds.height / contentHeight;
        
        this.transform.scale = Math.min(scaleX, scaleY, 1.0);
        this.transform.x = (this.viewportBounds.width - contentWidth * this.transform.scale) / 2 - minX * this.transform.scale;
        this.transform.y = (this.viewportBounds.height - contentHeight * this.transform.scale) / 2 - minY * this.transform.scale;
        
        this.updateTransform();
        this.updateMinimap();
    }
    
    centerView() {
        this.transform.x = this.viewportBounds.width / 2;
        this.transform.y = this.viewportBounds.height / 2;
        this.transform.scale = 1.0;
        
        this.updateTransform();
        this.updateMinimap();
    }
    
    // === Grid and Snapping ===
    
    toggleGrid() {
        this.showGrid = !this.showGrid;
        const gridElement = document.querySelector('.canvas-grid');
        if (gridElement) {
            gridElement.style.display = this.showGrid ? 'block' : 'none';
        }
        
        const button = document.getElementById('grid-toggle');
        button.classList.toggle('active', this.showGrid);
    }
    
    snapPoint(point) {
        return {
            x: Math.round(point.x / this.gridSize) * this.gridSize,
            y: Math.round(point.y / this.gridSize) * this.gridSize
        };
    }
    
    // === Coordinate Conversion ===
    
    screenToCanvas(screenX, screenY) {
        const rect = this.svg.getBoundingClientRect();
        const x = ((screenX - rect.left - this.transform.x) / this.transform.scale);
        const y = ((screenY - rect.top - this.transform.y) / this.transform.scale);
        return { x, y };
    }
    
    canvasToScreen(canvasX, canvasY) {
        const rect = this.svg.getBoundingClientRect();
        const x = canvasX * this.transform.scale + this.transform.x + rect.left;
        const y = canvasY * this.transform.scale + this.transform.y + rect.top;
        return { x, y };
    }
    
    // === Selection and Context Menu ===
    
    clearSelection() {
        document.querySelectorAll('.workflow-node.selected').forEach(node => {
            node.classList.remove('selected');
        });
        document.querySelectorAll('.workflow-connection.selected').forEach(conn => {
            conn.classList.remove('selected');
        });
    }
    
    selectAll() {
        document.querySelectorAll('.workflow-node').forEach(node => {
            node.classList.add('selected');
        });
        document.querySelectorAll('.workflow-connection').forEach(conn => {
            conn.classList.add('selected');
        });
    }
    
    showContextMenu(screenX, screenY, canvasPoint) {
        const contextMenu = document.getElementById('context-menu');
        contextMenu.style.display = 'block';
        contextMenu.style.left = screenX + 'px';
        contextMenu.style.top = screenY + 'px';
        
        // Hide menu when clicking elsewhere
        const hideMenu = (e) => {
            if (!contextMenu.contains(e.target)) {
                contextMenu.style.display = 'none';
                document.removeEventListener('click', hideMenu);
            }
        };
        
        setTimeout(() => {
            document.addEventListener('click', hideMenu);
        }, 0);
    }
    
    // === Minimap ===
    
    setupMinimap() {
        this.minimapSvg = document.querySelector('.minimap-canvas');
        this.minimapContent = document.getElementById('minimap-content');
        this.minimapViewport = document.getElementById('minimap-viewport');
        
        this.updateMinimap();
    }
    
    updateMinimap() {
        if (!this.showMinimap) return;
        
        // Update viewport rectangle
        const scale = 0.1; // Minimap scale
        this.minimapViewport.setAttribute('x', -this.transform.x * scale);
        this.minimapViewport.setAttribute('y', -this.transform.y * scale);
        this.minimapViewport.setAttribute('width', this.viewportBounds.width * scale / this.transform.scale);
        this.minimapViewport.setAttribute('height', this.viewportBounds.height * scale / this.transform.scale);
    }
    
    toggleMinimap() {
        this.showMinimap = !this.showMinimap;
        const minimap = document.getElementById('minimap');
        if (minimap) {
            minimap.style.display = this.showMinimap ? 'block' : 'none';
        }
        
        const button = document.getElementById('minimap-toggle');
        button.classList.toggle('active', this.showMinimap);
    }
    
    // === Utility Methods ===
    
    updateViewport() {
        const rect = this.svg.getBoundingClientRect();
        this.viewportBounds = {
            x: 0,
            y: 0,
            width: rect.width,
            height: rect.height
        };
    }
    
    animateNodeIn(nodeElement) {
        nodeElement.style.opacity = '0';
        nodeElement.style.transform = 'scale(0.5)';
        
        requestAnimationFrame(() => {
            nodeElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            nodeElement.style.opacity = '1';
            nodeElement.style.transform = 'scale(1)';
        });
    }
    
    animateNodeOut(nodeElement, callback) {
        nodeElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        nodeElement.style.opacity = '0';
        nodeElement.style.transform = 'scale(0.5)';
        
        setTimeout(callback, 300);
    }
    
    autoLayout() {
        try {
            const nodes = Array.from(this.designer.nodes.values());
            const connections = Array.from(this.designer.connections.values());
            
            if (nodes.length === 0) {
                this.designer.showNotification('No nodes to layout', 'warning');
                return;
            }
            
            // Use hierarchical layout algorithm
            const layout = this.calculateHierarchicalLayout(nodes, connections);
            
            // Apply the new positions
            this.applyLayout(layout);
            
            // Fit the canvas to show all nodes
            this.fitToScreen();
            
            this.designer.showNotification('Auto-layout applied successfully', 'success');
            
        } catch (error) {
            console.error('Auto-layout failed:', error);
            this.designer.showError('Auto-layout failed: ' + error.message);
        }
    }
    
    calculateHierarchicalLayout(nodes, connections) {
        const layout = {};
        const nodeMap = new Map();
        const incomingEdges = new Map();
        const outgoingEdges = new Map();
        
        // Initialize data structures
        nodes.forEach(node => {
            nodeMap.set(node.id, node);
            incomingEdges.set(node.id, []);
            outgoingEdges.set(node.id, []);
        });
        
        // Build edge lists
        connections.forEach(conn => {
            if (nodeMap.has(conn.source_node_id) && nodeMap.has(conn.target_node_id)) {
                outgoingEdges.get(conn.source_node_id).push(conn.target_node_id);
                incomingEdges.get(conn.target_node_id).push(conn.source_node_id);
            }
        });
        
        // Find root nodes (nodes with no incoming edges)
        const rootNodes = nodes.filter(node => incomingEdges.get(node.id).length === 0);
        
        if (rootNodes.length === 0) {
            // Handle circular dependencies - find node with minimum incoming edges
            let minIncoming = Infinity;
            let selectedRoot = nodes[0];
            nodes.forEach(node => {
                const incomingCount = incomingEdges.get(node.id).length;
                if (incomingCount < minIncoming) {
                    minIncoming = incomingCount;
                    selectedRoot = node;
                }
            });
            rootNodes.push(selectedRoot);
        }
        
        // Perform hierarchical layout
        const levels = this.assignLevels(nodes, connections, rootNodes);
        const positions = this.calculatePositions(levels);
        
        return positions;
    }
    
    assignLevels(nodes, connections, rootNodes) {
        const levels = [];
        const nodeLevel = new Map();
        const visited = new Set();
        const processing = new Set();
        
        // Build adjacency list
        const adjacencyList = new Map();
        nodes.forEach(node => {
            adjacencyList.set(node.id, []);
        });
        
        connections.forEach(conn => {
            if (adjacencyList.has(conn.source_node_id) && adjacencyList.has(conn.target_node_id)) {
                adjacencyList.get(conn.source_node_id).push(conn.target_node_id);
            }
        });
        
        // DFS to assign levels
        const assignLevel = (nodeId, level) => {
            if (processing.has(nodeId)) {
                // Circular dependency detected - break it by assigning current level
                return level;
            }
            
            if (visited.has(nodeId)) {
                return nodeLevel.get(nodeId);
            }
            
            processing.add(nodeId);
            
            let maxChildLevel = level;
            const children = adjacencyList.get(nodeId) || [];
            
            children.forEach(childId => {
                const childLevel = assignLevel(childId, level + 1);
                maxChildLevel = Math.max(maxChildLevel, childLevel);
            });
            
            processing.delete(nodeId);
            visited.add(nodeId);
            nodeLevel.set(nodeId, level);
            
            // Ensure levels array exists
            while (levels.length <= level) {
                levels.push([]);
            }
            
            levels[level].push(nodeId);
            return maxChildLevel;
        };
        
        // Start from root nodes
        rootNodes.forEach(root => {
            assignLevel(root.id, 0);
        });
        
        // Handle any unvisited nodes (disconnected components)
        nodes.forEach(node => {
            if (!visited.has(node.id)) {
                assignLevel(node.id, levels.length);
            }
        });
        
        return levels;
    }
    
    calculatePositions(levels) {
        const positions = {};
        const nodeSpacing = { x: 250, y: 150 };
        const levelPadding = 100;
        
        levels.forEach((level, levelIndex) => {
            const levelY = levelIndex * nodeSpacing.y + levelPadding;
            const totalWidth = (level.length - 1) * nodeSpacing.x;
            const startX = -totalWidth / 2;
            
            level.forEach((nodeId, nodeIndex) => {
                const x = startX + nodeIndex * nodeSpacing.x;
                const y = levelY;
                
                positions[nodeId] = { x, y };
            });
        });
        
        return positions;
    }
    
    applyLayout(layout) {
        // Save current state for undo
        this.designer.addToHistory('autoLayout', {
            oldPositions: this.getCurrentPositions(),
            newPositions: layout
        });
        
        // Apply new positions
        Object.entries(layout).forEach(([nodeId, position]) => {
            const node = this.designer.nodes.get(nodeId);
            if (node) {
                node.position = position;
                this.moveNode(nodeId, position);
            }
        });
        
        // Mark workflow as dirty
        this.designer.markDirty();
    }
    
    getCurrentPositions() {
        const positions = {};
        this.designer.nodes.forEach((node, nodeId) => {
            positions[nodeId] = { ...node.position };
        });
        return positions;
    }
}