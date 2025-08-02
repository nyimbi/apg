/**
 * APG Workflow Property Panels
 * 
 * Dynamic property UI management for workflow components.
 * 
 * © 2025 Datacraft. All rights reserved.
 * Author: Nyimbi Odero <nyimbi@gmail.com>
 */

class PropertyPanels {
    constructor(designer) {
        this.designer = designer;
        this.currentNode = null;
        this.currentConnection = null;
        this.propertyCache = new Map();
        
        // DOM elements
        this.panelTitle = document.getElementById('properties-title');
        this.panelContent = document.getElementById('properties-content');
        this.closeButton = document.getElementById('properties-close');
        
        this.initialize();
    }
    
    initialize() {
        this.setupEventHandlers();
        console.log('Property panels initialized');
    }
    
    setupEventHandlers() {
        // Close button
        this.closeButton.addEventListener('click', () => {
            this.showEmpty();
        });
        
        // Listen for node selection changes
        this.designer.on('nodeSelected', (selectedNodes) => {
            if (selectedNodes.length === 1) {
                this.showNodeProperties(selectedNodes[0]);
            } else if (selectedNodes.length > 1) {
                this.showMultiSelection();
            } else {
                this.showEmpty();
            }
        });
    }
    
    // === Property Display Methods ===
    
    showEmpty() {
        this.currentNode = null;
        this.currentConnection = null;
        this.panelTitle.textContent = 'Properties';
        
        this.panelContent.innerHTML = `
            <div class="no-selection">
                <i class="fas fa-mouse-pointer"></i>
                <p>Select a component to view properties</p>
            </div>
        `;
    }
    
    async showNodeProperties(nodeId) {
        try {
            const node = this.designer.nodes.get(nodeId);
            if (!node) return;
            
            this.currentNode = node;
            this.currentConnection = null;
            this.panelTitle.textContent = node.label || node.component_type;
            
            // Get component definition for property schema
            const component = this.designer.componentLibrary.getComponent(node.component_type);
            if (!component) {
                this.showError('Component definition not found');
                return;
            }
            
            // Render properties form
            this.renderNodePropertiesForm(node, component);
            
        } catch (error) {
            console.error('Failed to show node properties:', error);
            this.showError('Failed to load properties: ' + error.message);
        }
    }
    
    showMultiSelection() {
        this.currentNode = null;
        this.currentConnection = null;
        this.panelTitle.textContent = 'Multiple Selection';
        
        const selectedCount = this.designer.selectedNodes.size;
        
        this.panelContent.innerHTML = `
            <div class="multi-selection">
                <div class="selection-info">
                    <i class="fas fa-layer-group"></i>
                    <h6>${selectedCount} components selected</h6>
                </div>
                
                <div class="bulk-actions">
                    <button class="btn btn-sm btn-danger full-width" onclick="workflowDesigner.deleteSelected()">
                        <i class="fas fa-trash"></i> Delete Selected
                    </button>
                    <button class="btn btn-sm btn-secondary full-width mt-2" onclick="workflowDesigner.copySelected()">
                        <i class="fas fa-copy"></i> Copy Selected
                    </button>
                </div>
                
                <div class="common-properties">
                    <h6 class="property-group-title">Common Properties</h6>
                    <!-- Common properties for multi-selection would go here -->
                </div>
            </div>
        `;
    }
    
    showConnectionProperties(connectionId) {
        try {
            const connection = this.designer.connections.get(connectionId);
            if (!connection) return;
            
            this.currentConnection = connection;
            this.currentNode = null;
            this.panelTitle.textContent = 'Connection Properties';
            
            this.renderConnectionPropertiesForm(connection);
            
        } catch (error) {
            console.error('Failed to show connection properties:', error);
            this.showError('Failed to load connection properties: ' + error.message);
        }
    }
    
    showError(message) {
        this.panelContent.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i>
                ${message}
            </div>
        `;
    }
    
    // === Form Rendering ===
    
    renderNodePropertiesForm(node, component) {
        const form = document.createElement('form');
        form.className = 'properties-form';
        form.id = 'node-properties-form';
        
        // Basic information
        const basicGroup = this.createPropertyGroup('Basic Information', [
            {
                name: 'label',
                label: 'Display Name',
                type: 'text',
                value: node.label || node.component_type,
                description: 'Name displayed on the workflow canvas'
            },
            {
                name: 'description',
                label: 'Description',
                type: 'textarea',
                value: node.description || '',
                description: 'Optional description of this component'
            }
        ]);
        
        form.appendChild(basicGroup);
        
        // Position information
        const positionGroup = this.createPropertyGroup('Position', [
            {
                name: 'position_x',
                label: 'X Position',
                type: 'number',
                value: Math.round(node.position.x),
                readonly: true
            },
            {
                name: 'position_y',
                label: 'Y Position',
                type: 'number',
                value: Math.round(node.position.y),
                readonly: true
            }
        ]);
        
        form.appendChild(positionGroup);
        
        // Configuration properties
        if (component.properties && component.properties.length > 0) {
            const configGroup = this.createConfigurationGroup(node, component);
            form.appendChild(configGroup);
        }
        
        // Input/Output ports information
        if (component.input_ports || component.output_ports) {
            const portsGroup = this.createPortsGroup(component);
            form.appendChild(portsGroup);
        }
        
        // Advanced settings
        const advancedGroup = this.createAdvancedGroup(node);
        form.appendChild(advancedGroup);
        
        // Setup form submission
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveNodeProperties(node);
        });
        
        // Setup real-time validation
        form.addEventListener('input', (e) => {
            this.validateProperty(e.target, component);
        });
        
        this.panelContent.innerHTML = '';
        this.panelContent.appendChild(form);
        
        // Add save button
        const saveButton = document.createElement('button');
        saveButton.type = 'submit';
        saveButton.form = 'node-properties-form';
        saveButton.className = 'btn btn-primary full-width mt-3';
        saveButton.innerHTML = '<i class="fas fa-save"></i> Save Properties';
        
        this.panelContent.appendChild(saveButton);
    }
    
    createPropertyGroup(title, properties) {
        const group = document.createElement('div');
        group.className = 'property-group';
        
        const titleElement = document.createElement('div');
        titleElement.className = 'property-group-title';
        titleElement.textContent = title;
        group.appendChild(titleElement);
        
        properties.forEach(property => {
            const propertyElement = this.createPropertyField(property);
            group.appendChild(propertyElement);
        });
        
        return group;
    }
    
    createPropertyField(property) {
        const item = document.createElement('div');
        item.className = 'property-item';
        
        const label = document.createElement('label');
        label.className = 'property-label';
        label.setAttribute('for', property.name);
        label.textContent = property.label;
        if (property.required) {
            label.innerHTML += ' <span class="text-danger">*</span>';
        }
        
        let input;
        switch (property.type) {
            case 'textarea':
                input = document.createElement('textarea');
                input.rows = 3;
                break;
            case 'select':
                input = document.createElement('select');
                if (property.options) {
                    property.options.forEach(option => {
                        const optionElement = document.createElement('option');
                        optionElement.value = option.value;
                        optionElement.textContent = option.label;
                        input.appendChild(optionElement);
                    });
                }
                break;
            case 'checkbox':
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = property.value || false;
                break;
            case 'number':
                input = document.createElement('input');
                input.type = 'number';
                if (property.min !== undefined) input.min = property.min;
                if (property.max !== undefined) input.max = property.max;
                if (property.step !== undefined) input.step = property.step;
                break;
            case 'json':
                input = document.createElement('textarea');
                input.rows = 6;
                input.className = 'json-editor';
                break;
            default:
                input = document.createElement('input');
                input.type = property.type || 'text';
        }
        
        input.id = property.name;
        input.name = property.name;
        input.className += ' property-input form-control';
        
        if (property.value !== undefined && property.type !== 'checkbox') {
            input.value = typeof property.value === 'object' ? 
                JSON.stringify(property.value, null, 2) : property.value;
        }
        
        if (property.placeholder) input.placeholder = property.placeholder;
        if (property.readonly) input.readOnly = true;
        if (property.required) input.required = true;
        
        item.appendChild(label);
        
        if (property.description) {
            const description = document.createElement('div');
            description.className = 'property-description';
            description.textContent = property.description;
            item.appendChild(description);
        }
        
        item.appendChild(input);
        
        // Error container
        const errorContainer = document.createElement('div');
        errorContainer.className = 'property-error';
        errorContainer.id = `${property.name}-error`;
        item.appendChild(errorContainer);
        
        return item;
    }
    
    createConfigurationGroup(node, component) {
        const properties = component.properties.map(prop => ({
            ...prop,
            value: node.config ? node.config[prop.name] : prop.default_value
        }));
        
        return this.createPropertyGroup('Configuration', properties);
    }
    
    createPortsGroup(component) {
        const group = document.createElement('div');
        group.className = 'property-group';
        
        const titleElement = document.createElement('div');
        titleElement.className = 'property-group-title';
        titleElement.textContent = 'Ports & Connections';
        group.appendChild(titleElement);
        
        // Input ports
        if (component.input_ports && component.input_ports.length > 0) {
            const inputHeader = document.createElement('h6');
            inputHeader.textContent = 'Input Ports';
            inputHeader.className = 'mt-3 mb-2';
            group.appendChild(inputHeader);
            
            component.input_ports.forEach(port => {
                const portInfo = document.createElement('div');
                portInfo.className = 'port-info';
                portInfo.innerHTML = `
                    <div class="d-flex justify-content-between">
                        <strong>${port.name}</strong>
                        <span class="badge badge-info">${port.type}</span>
                    </div>
                    ${port.description ? `<small class="text-muted">${port.description}</small>` : ''}
                    ${port.required ? '<span class="badge badge-danger badge-sm">Required</span>' : ''}
                `;
                group.appendChild(portInfo);
            });
        }
        
        // Output ports
        if (component.output_ports && component.output_ports.length > 0) {
            const outputHeader = document.createElement('h6');
            outputHeader.textContent = 'Output Ports';
            outputHeader.className = 'mt-3 mb-2';
            group.appendChild(outputHeader);
            
            component.output_ports.forEach(port => {
                const portInfo = document.createElement('div');
                portInfo.className = 'port-info';
                portInfo.innerHTML = `
                    <div class="d-flex justify-content-between">
                        <strong>${port.name}</strong>
                        <span class="badge badge-success">${port.type}</span>
                    </div>
                    ${port.description ? `<small class="text-muted">${port.description}</small>` : ''}
                `;
                group.appendChild(portInfo);
            });
        }
        
        return group;
    }
    
    createAdvancedGroup(node) {
        return this.createPropertyGroup('Advanced Settings', [
            {
                name: 'timeout',
                label: 'Timeout (seconds)',
                type: 'number',
                value: node.timeout || 30,
                min: 1,
                max: 3600,
                description: 'Maximum execution time for this component'
            },
            {
                name: 'retry_count',
                label: 'Retry Count',
                type: 'number',
                value: node.retry_count || 0,
                min: 0,
                max: 10,
                description: 'Number of retries on failure'
            },
            {
                name: 'enabled',
                label: 'Enabled',
                type: 'checkbox',
                value: node.enabled !== false,
                description: 'Whether this component is enabled for execution'
            }
        ]);
    }
    
    renderConnectionPropertiesForm(connection) {
        const form = document.createElement('form');
        form.className = 'properties-form';
        form.id = 'connection-properties-form';
        
        // Connection information
        const basicGroup = this.createPropertyGroup('Connection Information', [
            {
                name: 'source_node',
                label: 'Source Node',
                type: 'text',
                value: this.getNodeDisplayName(connection.source_node_id),
                readonly: true
            },
            {
                name: 'source_port',
                label: 'Source Port',
                type: 'text',
                value: connection.source_port,
                readonly: true
            },
            {
                name: 'target_node',
                label: 'Target Node',
                type: 'text',
                value: this.getNodeDisplayName(connection.target_node_id),
                readonly: true
            },
            {
                name: 'target_port',
                label: 'Target Port',
                type: 'text',
                value: connection.target_port,
                readonly: true
            }
        ]);
        
        form.appendChild(basicGroup);
        
        // Connection settings
        const settingsGroup = this.createPropertyGroup('Settings', [
            {
                name: 'label',
                label: 'Connection Label',
                type: 'text',
                value: connection.label || '',
                description: 'Optional label for this connection'
            },
            {
                name: 'condition',
                label: 'Condition',
                type: 'text',
                value: connection.condition || '',
                description: 'Optional condition for this connection (e.g., "success", "error")'
            }
        ]);
        
        form.appendChild(settingsGroup);
        
        this.panelContent.innerHTML = '';
        this.panelContent.appendChild(form);
        
        // Add save button
        const saveButton = document.createElement('button');
        saveButton.type = 'submit';
        saveButton.className = 'btn btn-primary full-width mt-3';
        saveButton.innerHTML = '<i class="fas fa-save"></i> Save Connection';
        saveButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.saveConnectionProperties(connection);
        });
        
        this.panelContent.appendChild(saveButton);
    }
    
    // === Property Validation ===
    
    validateProperty(input, component) {
        const property = component.properties?.find(p => p.name === input.name);
        if (!property) return true;
        
        const errorContainer = document.getElementById(`${input.name}-error`);
        const errors = [];
        
        // Required validation
        if (property.required && !input.value.trim()) {
            errors.push('This field is required');
        }
        
        // Type validation
        if (input.value.trim()) {
            switch (property.type) {
                case 'number':
                    if (isNaN(input.value)) {
                        errors.push('Must be a valid number');
                    } else {
                        const num = parseFloat(input.value);
                        if (property.min !== undefined && num < property.min) {
                            errors.push(`Must be at least ${property.min}`);
                        }
                        if (property.max !== undefined && num > property.max) {
                            errors.push(`Must be at most ${property.max}`);
                        }
                    }
                    break;
                case 'email':
                    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                    if (!emailRegex.test(input.value)) {
                        errors.push('Must be a valid email address');
                    }
                    break;
                case 'url':
                    try {
                        new URL(input.value);
                    } catch {
                        errors.push('Must be a valid URL');
                    }
                    break;
                case 'json':
                    try {
                        JSON.parse(input.value);
                    } catch {
                        errors.push('Must be valid JSON');
                    }
                    break;
            }
        }
        
        // Custom validation
        if (property.pattern) {
            const regex = new RegExp(property.pattern);
            if (!regex.test(input.value)) {
                errors.push(property.pattern_message || 'Invalid format');
            }
        }
        
        // Display errors
        if (errors.length > 0) {
            errorContainer.textContent = errors[0];
            input.classList.add('is-invalid');
            return false;
        } else {
            errorContainer.textContent = '';
            input.classList.remove('is-invalid');
            return true;
        }
    }
    
    // === Property Saving ===
    
    async saveNodeProperties(node) {
        try {
            const form = document.getElementById('node-properties-form');
            const formData = new FormData(form);
            const properties = {};
            
            // Extract form values
            for (const [key, value] of formData.entries()) {
                const input = form.querySelector(`[name="${key}"]`);
                
                if (input.type === 'checkbox') {
                    properties[key] = input.checked;
                } else if (input.type === 'number') {
                    properties[key] = parseFloat(value) || 0;
                } else if (input.classList.contains('json-editor')) {
                    try {
                        properties[key] = JSON.parse(value);
                    } catch {
                        properties[key] = value;
                    }
                } else {
                    properties[key] = value;
                }
            }
            
            // Update node configuration
            const updatedConfig = {
                ...node.config,
                ...Object.fromEntries(
                    Object.entries(properties).filter(([key]) => 
                        !['label', 'description', 'position_x', 'position_y', 'timeout', 'retry_count', 'enabled'].includes(key)
                    )
                )
            };
            
            // Update basic properties
            if (properties.label !== node.label) {
                node.label = properties.label;
            }
            if (properties.description !== node.description) {
                node.description = properties.description;
            }
            if (properties.timeout !== node.timeout) {
                node.timeout = properties.timeout;
            }
            if (properties.retry_count !== node.retry_count) {
                node.retry_count = properties.retry_count;
            }
            if (properties.enabled !== node.enabled) {
                node.enabled = properties.enabled;
            }
            
            // Save to backend
            const response = await fetch(`${this.designer.config.api_base_url}/session/${this.designer.sessionId}/component/${node.id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.designer.getCSRFToken()
                },
                body: JSON.stringify({
                    label: node.label,
                    description: node.description,
                    config: updatedConfig,
                    timeout: node.timeout,
                    retry_count: node.retry_count,
                    enabled: node.enabled
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to save properties');
            }
            
            // Update local node
            Object.assign(node, result.component);
            
            // Update canvas display
            this.updateNodeDisplay(node);
            
            // Mark workflow as dirty
            this.designer.markDirty();
            
            this.designer.showNotification('Properties saved successfully', 'success');
            
        } catch (error) {
            console.error('Failed to save node properties:', error);
            this.designer.showError('Failed to save properties: ' + error.message);
        }
    }
    
    async saveConnectionProperties(connection) {
        try {
            const form = document.getElementById('connection-properties-form');
            const formData = new FormData(form);
            
            const updatedConnection = {
                ...connection,
                label: formData.get('label'),
                condition: formData.get('condition')
            };
            
            // Save to backend
            const response = await fetch(`${this.designer.config.api_base_url}/session/${this.designer.sessionId}/connection/${connection.id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.designer.getCSRFToken()
                },
                body: JSON.stringify({
                    label: updatedConnection.label,
                    condition: updatedConnection.condition
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to save connection properties');
            }
            
            // Update local connection
            Object.assign(connection, result.connection);
            
            // Mark workflow as dirty
            this.designer.markDirty();
            
            this.designer.showNotification('Connection properties saved successfully', 'success');
            
        } catch (error) {
            console.error('Failed to save connection properties:', error);
            this.designer.showError('Failed to save connection properties: ' + error.message);
        }
    }
    
    // === Helper Methods ===
    
    getNodeDisplayName(nodeId) {
        const node = this.designer.nodes.get(nodeId);
        return node ? (node.label || node.component_type) : 'Unknown Node';
    }
    
    updateNodeDisplay(node) {
        // Update the node text on canvas
        const nodeElement = document.getElementById(`node-${node.id}`);
        if (nodeElement) {
            const textElement = nodeElement.querySelector('.node-text');
            if (textElement) {
                textElement.textContent = node.label || node.component_type;
            }
        }
    }
    
    // === Property Templates ===
    
    getPropertyTemplate(type) {
        const templates = {
            'trigger': {
                properties: [
                    { name: 'enabled', label: 'Enabled', type: 'checkbox', default_value: true },
                    { name: 'schedule', label: 'Schedule', type: 'text', description: 'Cron expression for scheduling' }
                ]
            },
            'data_transform': {
                properties: [
                    { name: 'mapping', label: 'Field Mapping', type: 'json', description: 'JSON object defining field mappings' },
                    { name: 'validation', label: 'Validation Rules', type: 'json', description: 'JSON schema for validation' }
                ]
            },
            'condition': {
                properties: [
                    { name: 'expression', label: 'Condition Expression', type: 'text', required: true },
                    { name: 'operator', label: 'Operator', type: 'select', options: [
                        { value: 'equals', label: 'Equals' },
                        { value: 'not_equals', label: 'Not Equals' },
                        { value: 'greater_than', label: 'Greater Than' },
                        { value: 'less_than', label: 'Less Than' }
                    ]}
                ]
            }
        };
        
        return templates[type] || { properties: [] };
    }
    
    // === Cache Management ===
    
    cacheProperty(nodeId, propertyName, value) {
        const cacheKey = `${nodeId}_${propertyName}`;
        this.propertyCache.set(cacheKey, value);
    }
    
    getCachedProperty(nodeId, propertyName) {
        const cacheKey = `${nodeId}_${propertyName}`;
        return this.propertyCache.get(cacheKey);
    }
    
    clearPropertyCache(nodeId = null) {
        if (nodeId) {
            // Clear cache for specific node
            for (const key of this.propertyCache.keys()) {
                if (key.startsWith(`${nodeId}_`)) {
                    this.propertyCache.delete(key);
                }
            }
        } else {
            // Clear all cache
            this.propertyCache.clear();
        }
    }
}

// Add styles for property panels
const propertyPanelStyles = `
<style>
.properties-form {
    font-size: 13px;
}

.property-item {
    margin-bottom: 16px;
}

.property-input.is-invalid {
    border-color: #dc3545;
}

.property-error {
    color: #dc3545;
    font-size: 11px;
    margin-top: 4px;
    min-height: 14px;
}

.port-info {
    padding: 8px;
    margin: 4px 0;
    background: #f8f9fa;
    border-radius: 4px;
    font-size: 12px;
}

.multi-selection {
    text-align: center;
    padding: 20px;
}

.selection-info {
    margin-bottom: 20px;
}

.selection-info i {
    font-size: 32px;
    color: #6c757d;
    margin-bottom: 8px;
}

.bulk-actions {
    margin-bottom: 20px;
}

.json-editor {
    font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
    font-size: 12px;
}

.badge-sm {
    font-size: 9px;
    padding: 2px 6px;
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', propertyPanelStyles);