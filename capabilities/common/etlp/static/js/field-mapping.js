/**
 * APG ETLP Visual Field Mapping JavaScript
 * Advanced drag-and-drop field mapping with AI intelligence
 */

class FieldMappingInterface {
    constructor() {
        this.sourceSchema = null;
        this.targetSchema = null;
        this.mappings = new Map();
        this.connections = new Map();
        this.draggedElement = null;
        this.currentMapping = null;
        this.aiSuggestions = [];
        
        this.initializeEventListeners();
        this.setupDragAndDrop();
    }

    initializeEventListeners() {
        // Window resize handler for connection lines
        window.addEventListener('resize', () => this.redrawConnections());
        
        // Transformation type change handler
        document.getElementById('transformationType').addEventListener('change', (e) => {
            this.updateTransformationOptions(e.target.value);
        });
    }

    setupDragAndDrop() {
        // Setup drag and drop for field items
        document.addEventListener('dragstart', (e) => {
            if (e.target.classList.contains('field-item')) {
                this.draggedElement = e.target;
                e.target.classList.add('dragging');
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/html', e.target.outerHTML);
            }
        });

        document.addEventListener('dragend', (e) => {
            if (e.target.classList.contains('field-item')) {
                e.target.classList.remove('dragging');
                this.draggedElement = null;
            }
        });

        document.addEventListener('dragover', (e) => {
            if (e.target.classList.contains('field-item') || e.target.closest('.field-item')) {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                
                const targetField = e.target.closest('.field-item');
                if (targetField && !targetField.classList.contains('dragging')) {
                    targetField.classList.add('drop-target');
                }
            }
        });

        document.addEventListener('dragleave', (e) => {
            if (e.target.classList.contains('drop-target')) {
                e.target.classList.remove('drop-target');
            }
        });

        document.addEventListener('drop', (e) => {
            e.preventDefault();
            const targetField = e.target.closest('.field-item');
            
            if (targetField && this.draggedElement && targetField !== this.draggedElement) {
                targetField.classList.remove('drop-target');
                this.createFieldMapping(this.draggedElement, targetField);
            }
        });
    }

    async loadSourceSchema(dataSourceId, tableName) {
        try {
            this.showStatus('Loading source schema...', 'loading');
            
            const response = await fetch(`/api/etlp/field-mapping/schema/${dataSourceId}/${tableName}`);
            const data = await response.json();
            
            this.sourceSchema = data.schema;
            this.renderSourceFields();
            this.showStatus('Source schema loaded', 'success');
            
        } catch (error) {
            console.error('Error loading source schema:', error);
            this.showStatus('Error loading source schema', 'error');
        }
    }

    async loadTargetSchema(dataSourceId, tableName) {
        try {
            this.showStatus('Loading target schema...', 'loading');
            
            const response = await fetch(`/api/etlp/field-mapping/schema/${dataSourceId}/${tableName}`);
            const data = await response.json();
            
            this.targetSchema = data.schema;
            this.renderTargetFields();
            this.showStatus('Target schema loaded', 'success');
            
        } catch (error) {
            console.error('Error loading target schema:', error);
            this.showStatus('Error loading target schema', 'error');
        }
    }

    renderSourceFields() {
        const container = document.getElementById('sourceFields');
        container.innerHTML = '';

        if (!this.sourceSchema || !this.sourceSchema.fields) {
            return;
        }

        this.sourceSchema.fields.forEach((field, index) => {
            const fieldElement = this.createFieldElement(field, 'source', index);
            container.appendChild(fieldElement);
        });
    }

    renderTargetFields() {
        const container = document.getElementById('targetFields');
        container.innerHTML = '';

        if (!this.targetSchema || !this.targetSchema.fields) {
            return;
        }

        this.targetSchema.fields.forEach((field, index) => {
            const fieldElement = this.createFieldElement(field, 'target', index);
            container.appendChild(fieldElement);
        });
    }

    createFieldElement(field, type, index) {
        const fieldDiv = document.createElement('div');
        fieldDiv.className = 'field-item';
        fieldDiv.draggable = type === 'source';
        fieldDiv.dataset.fieldName = field.name;
        fieldDiv.dataset.fieldType = type;
        fieldDiv.dataset.fieldIndex = index;

        // Add connection point
        const connectionPoint = document.createElement('div');
        connectionPoint.className = 'connection-point';
        connectionPoint.style.cssText = `
            position: absolute;
            ${type === 'source' ? 'right: -6px' : 'left: -6px'};
            top: 50%;
            transform: translateY(-50%);
            width: 12px;
            height: 12px;
            background: #007bff;
            border-radius: 50%;
            cursor: pointer;
        `;

        fieldDiv.innerHTML = `
            <div class="field-name">${field.name}</div>
            <div class="field-type">${this.formatDataType(field.data_type)}</div>
            <div class="field-meta">
                ${field.nullable ? '<span title="Nullable">NULL</span>' : '<span title="Not Null">NOT NULL</span>'}
                ${field.primary_key ? '<span title="Primary Key">PK</span>' : ''}
                ${field.max_length ? `<span title="Max Length">${field.max_length}</span>` : ''}
            </div>
        `;

        fieldDiv.appendChild(connectionPoint);

        // Add event listeners
        fieldDiv.addEventListener('mouseenter', (e) => this.showFieldTooltip(e, field));
        fieldDiv.addEventListener('mouseleave', () => this.hideTooltip());
        
        if (type === 'target') {
            fieldDiv.addEventListener('click', () => this.selectTargetField(field, fieldDiv));
        }

        return fieldDiv;
    }

    createFieldMapping(sourceElement, targetElement) {
        const sourceField = sourceElement.dataset.fieldName;
        const targetField = targetElement.dataset.fieldName;
        
        // Check if mapping already exists
        if (this.mappings.has(sourceField)) {
            this.removeMapping(sourceField);
        }

        // Create mapping
        const mapping = {
            id: this.generateMappingId(),
            source_field: sourceField,
            target_field: targetField,
            transformation: 'direct_copy',
            transformation_config: {},
            created_by: 'current_user', // TODO: Get from auth
            validation_rules: [],
            connection_points: this.calculateConnectionPoints(sourceElement, targetElement)
        };

        this.mappings.set(sourceField, mapping);
        
        // Update UI
        sourceElement.classList.add('mapped');
        targetElement.classList.add('mapped');
        
        // Draw connection line
        this.drawConnection(mapping);
        
        // Update mapping count
        this.updateMappingCount();
        
        // Show transformation panel
        this.showTransformationPanel(mapping);
        
        this.showStatus('Field mapping created', 'success');
    }

    calculateConnectionPoints(sourceElement, targetElement) {
        const sourceRect = sourceElement.getBoundingClientRect();
        const targetRect = targetElement.getBoundingClientRect();
        const canvasRect = document.getElementById('mappingCanvas').getBoundingClientRect();

        return {
            source: {
                x: sourceRect.right - canvasRect.left,
                y: sourceRect.top + sourceRect.height / 2 - canvasRect.top
            },
            target: {
                x: targetRect.left - canvasRect.left,
                y: targetRect.top + targetRect.height / 2 - canvasRect.top
            }
        };
    }

    drawConnection(mapping) {
        const svg = document.getElementById('connectionSvg');
        const points = mapping.connection_points;
        
        // Create SVG path element
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.classList.add('connection-line');
        path.dataset.mappingId = mapping.id;
        
        // Calculate control points for smooth curve
        const controlPoint1X = points.source.x + (points.target.x - points.source.x) * 0.5;
        const controlPoint2X = points.source.x + (points.target.x - points.source.x) * 0.5;
        
        const pathData = `M ${points.source.x} ${points.source.y} 
                         C ${controlPoint1X} ${points.source.y}, 
                           ${controlPoint2X} ${points.target.y}, 
                           ${points.target.x} ${points.target.y}`;
        
        path.setAttribute('d', pathData);
        
        // Add click handler for editing transformation
        path.addEventListener('click', () => {
            this.editMapping(mapping.id);
        });
        
        svg.appendChild(path);
        this.connections.set(mapping.id, path);
    }

    redrawConnections() {
        // Clear existing connections
        const svg = document.getElementById('connectionSvg');
        svg.innerHTML = '';
        this.connections.clear();
        
        // Redraw all mappings
        this.mappings.forEach(mapping => {
            // Recalculate connection points
            const sourceElement = document.querySelector(`[data-field-name="${mapping.source_field}"][data-field-type="source"]`);
            const targetElement = document.querySelector(`[data-field-name="${mapping.target_field}"][data-field-type="target"]`);
            
            if (sourceElement && targetElement) {
                mapping.connection_points = this.calculateConnectionPoints(sourceElement, targetElement);
                this.drawConnection(mapping);
            }
        });
    }

    async generateIntelligentMappings() {
        if (!this.sourceSchema || !this.targetSchema) {
            this.showStatus('Please load both schemas first', 'error');
            return;
        }

        try {
            this.showStatus('Generating AI suggestions...', 'loading');
            
            const response = await fetch('/api/etlp/field-mapping/suggest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    source_schema: this.sourceSchema,
                    target_schema: this.targetSchema
                })
            });
            
            const data = await response.json();
            this.aiSuggestions = data.suggestions;
            
            this.renderAISuggestions();
            this.showStatus('AI suggestions generated', 'success');
            
        } catch (error) {
            console.error('Error generating suggestions:', error);
            this.showStatus('Error generating AI suggestions', 'error');
        }
    }

    renderAISuggestions() {
        const container = document.getElementById('suggestionsList');
        container.innerHTML = '';

        this.aiSuggestions.forEach((suggestion, index) => {
            const suggestionDiv = document.createElement('div');
            suggestionDiv.className = 'suggestion-item';
            suggestionDiv.dataset.suggestionIndex = index;

            const confidence = Math.round(suggestion.confidence * 100);
            const confidenceClass = confidence >= 80 ? 'status-success' : confidence >= 60 ? 'status-warning' : 'status-error';

            suggestionDiv.innerHTML = `
                <div class="suggestion-match">
                    ${suggestion.source_field} → ${suggestion.target_field}
                </div>
                <div class="suggestion-confidence">
                    <span class="status-indicator ${confidenceClass}">
                        ${confidence}% confidence
                    </span>
                    <span style="margin-left: 10px; font-size: 10px;">
                        ${suggestion.transformation || 'direct_copy'}
                    </span>
                </div>
            `;

            suggestionDiv.addEventListener('click', () => {
                this.applySuggestion(suggestion);
            });

            container.appendChild(suggestionDiv);
        });
    }

    applySuggestion(suggestion) {
        const sourceElement = document.querySelector(`[data-field-name="${suggestion.source_field}"][data-field-type="source"]`);
        const targetElement = document.querySelector(`[data-field-name="${suggestion.target_field}"][data-field-type="target"]`);

        if (sourceElement && targetElement) {
            // Remove existing mapping if it exists
            if (this.mappings.has(suggestion.source_field)) {
                this.removeMapping(suggestion.source_field);
            }

            // Create new mapping with suggestion details
            const mapping = {
                id: this.generateMappingId(),
                source_field: suggestion.source_field,
                target_field: suggestion.target_field,
                transformation: suggestion.transformation || 'direct_copy',
                transformation_config: suggestion.config || {},
                created_by: 'current_user',
                validation_rules: suggestion.validation_rules || [],
                connection_points: this.calculateConnectionPoints(sourceElement, targetElement)
            };

            this.mappings.set(suggestion.source_field, mapping);
            
            // Update UI
            sourceElement.classList.add('mapped');
            targetElement.classList.add('mapped');
            
            // Draw connection
            this.drawConnection(mapping);
            
            this.updateMappingCount();
            this.showStatus(`Applied suggestion: ${suggestion.source_field} → ${suggestion.target_field}`, 'success');
        }
    }

    showTransformationPanel(mapping) {
        this.currentMapping = mapping;
        const panel = document.getElementById('transformationPanel');
        const typeSelect = document.getElementById('transformationType');
        
        typeSelect.value = mapping.transformation;
        this.updateTransformationOptions(mapping.transformation, mapping.transformation_config);
        
        panel.style.display = 'block';
    }

    updateTransformationOptions(transformationType, config = {}) {
        const container = document.getElementById('configurationOptions');
        container.innerHTML = '';

        switch (transformationType) {
            case 'type_convert':
                container.innerHTML = `
                    <div class="form-group">
                        <label class="form-label">Target Type</label>
                        <select class="form-control" id="targetType">
                            <option value="string" ${config.target_type === 'string' ? 'selected' : ''}>String</option>
                            <option value="integer" ${config.target_type === 'integer' ? 'selected' : ''}>Integer</option>
                            <option value="float" ${config.target_type === 'float' ? 'selected' : ''}>Float</option>
                            <option value="boolean" ${config.target_type === 'boolean' ? 'selected' : ''}>Boolean</option>
                            <option value="date" ${config.target_type === 'date' ? 'selected' : ''}>Date</option>
                        </select>
                    </div>
                `;
                break;

            case 'format_string':
                container.innerHTML = `
                    <div class="form-group">
                        <label class="form-label">Format Pattern</label>
                        <input type="text" class="form-control" id="formatPattern" 
                               placeholder="e.g., {0} - {1}" value="${config.format || ''}">
                        <small style="color: #6c757d;">Use {0}, {1}, etc. as placeholders</small>
                    </div>
                `;
                break;

            case 'substring':
                container.innerHTML = `
                    <div class="form-group">
                        <label class="form-label">Start Position</label>
                        <input type="number" class="form-control" id="startPos" 
                               min="0" value="${config.start || 0}">
                    </div>
                    <div class="form-group">
                        <label class="form-label">End Position (optional)</label>
                        <input type="number" class="form-control" id="endPos" 
                               min="0" value="${config.end || ''}">
                    </div>
                `;
                break;

            case 'regex_extract':
                container.innerHTML = `
                    <div class="form-group">
                        <label class="form-label">Regular Expression</label>
                        <input type="text" class="form-control" id="regexPattern" 
                               placeholder="e.g., ^(\w+)" value="${config.pattern || ''}">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Group Index</label>
                        <input type="number" class="form-control" id="regexGroup" 
                               min="0" value="${config.group || 1}">
                    </div>
                `;
                break;

            case 'conditional':
                container.innerHTML = `
                    <div class="form-group">
                        <label class="form-label">Condition</label>
                        <textarea class="form-control" id="conditionLogic" rows="3" 
                                  placeholder="e.g., value > 0 ? 'positive' : 'negative'">${config.condition || ''}</textarea>
                    </div>
                `;
                break;

            case 'custom_function':
                container.innerHTML = `
                    <div class="form-group">
                        <label class="form-label">Function Name</label>
                        <input type="text" class="form-control" id="functionName" 
                               placeholder="e.g., format_phone_number" value="${config.function_name || ''}">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Parameters (JSON)</label>
                        <textarea class="form-control" id="functionParams" rows="3" 
                                  placeholder='{"param1": "value1"}'>${JSON.stringify(config.parameters || {}, null, 2)}</textarea>
                    </div>
                `;
                break;

            default:
                container.innerHTML = '<p style="color: #6c757d; font-style: italic;">No additional configuration required.</p>';
        }
    }

    applyTransformation() {
        if (!this.currentMapping) return;

        const transformationType = document.getElementById('transformationType').value;
        const config = {};

        // Collect configuration based on transformation type
        switch (transformationType) {
            case 'type_convert':
                config.target_type = document.getElementById('targetType').value;
                break;

            case 'format_string':
                config.format = document.getElementById('formatPattern').value;
                break;

            case 'substring':
                config.start = parseInt(document.getElementById('startPos').value) || 0;
                const endPos = document.getElementById('endPos').value;
                if (endPos) config.end = parseInt(endPos);
                break;

            case 'regex_extract':
                config.pattern = document.getElementById('regexPattern').value;
                config.group = parseInt(document.getElementById('regexGroup').value) || 1;
                break;

            case 'conditional':
                config.condition = document.getElementById('conditionLogic').value;
                break;

            case 'custom_function':
                config.function_name = document.getElementById('functionName').value;
                try {
                    config.parameters = JSON.parse(document.getElementById('functionParams').value || '{}');
                } catch (e) {
                    this.showStatus('Invalid JSON in parameters', 'error');
                    return;
                }
                break;
        }

        // Update the mapping
        this.currentMapping.transformation = transformationType;
        this.currentMapping.transformation_config = config;
        this.mappings.set(this.currentMapping.source_field, this.currentMapping);

        // Update connection line style based on transformation
        const connectionLine = this.connections.get(this.currentMapping.id);
        if (connectionLine) {
            if (transformationType === 'direct_copy') {
                connectionLine.style.stroke = '#28a745';
            } else {
                connectionLine.style.stroke = '#ffc107';
            }
        }

        this.closeTransformationPanel();
        this.showStatus('Transformation applied', 'success');
    }

    closeTransformationPanel() {
        document.getElementById('transformationPanel').style.display = 'none';
        this.currentMapping = null;
    }

    editMapping(mappingId) {
        const mapping = Array.from(this.mappings.values()).find(m => m.id === mappingId);
        if (mapping) {
            this.showTransformationPanel(mapping);
        }
    }

    removeMapping(sourceField) {
        const mapping = this.mappings.get(sourceField);
        if (!mapping) return;

        // Remove connection line
        const connectionLine = this.connections.get(mapping.id);
        if (connectionLine) {
            connectionLine.remove();
            this.connections.delete(mapping.id);
        }

        // Remove mapping
        this.mappings.delete(sourceField);

        // Update UI
        const sourceElement = document.querySelector(`[data-field-name="${sourceField}"][data-field-type="source"]`);
        const targetElement = document.querySelector(`[data-field-name="${mapping.target_field}"][data-field-type="target"]`);
        
        if (sourceElement) sourceElement.classList.remove('mapped');
        if (targetElement) targetElement.classList.remove('mapped');

        this.updateMappingCount();
    }

    async saveMappingConfiguration() {
        if (this.mappings.size === 0) {
            this.showStatus('No mappings to save', 'warning');
            return;
        }

        try {
            this.showStatus('Saving mapping configuration...', 'loading');

            const configuration = {
                name: 'Field Mapping Configuration',
                description: 'Visual field mapping configuration',
                source_schema: this.sourceSchema,
                target_schema: this.targetSchema,
                field_mappings: Array.from(this.mappings.values()),
                tenant_id: 'current_tenant', // TODO: Get from context
                created_by: 'current_user'
            };

            const response = await fetch('/api/etlp/field-mapping/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(configuration)
            });

            if (response.ok) {
                const result = await response.json();
                this.showStatus(`Configuration saved: ${result.id}`, 'success');
            } else {
                throw new Error('Save failed');
            }

        } catch (error) {
            console.error('Error saving configuration:', error);
            this.showStatus('Error saving configuration', 'error');
        }
    }

    // Utility Methods

    generateMappingId() {
        return 'mapping_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    formatDataType(dataType) {
        const typeMap = {
            'string': 'VARCHAR',
            'integer': 'INT',
            'float': 'FLOAT',
            'decimal': 'DECIMAL',
            'boolean': 'BOOLEAN',
            'date': 'DATE',
            'datetime': 'DATETIME',
            'timestamp': 'TIMESTAMP',
            'json': 'JSON',
            'uuid': 'UUID'
        };
        return typeMap[dataType] || dataType.toUpperCase();
    }

    updateMappingCount() {
        document.getElementById('mappingCount').textContent = this.mappings.size;
        
        // Update progress bar
        if (this.targetSchema && this.targetSchema.fields) {
            const progress = (this.mappings.size / this.targetSchema.fields.length) * 100;
            document.getElementById('progressFill').style.width = Math.min(progress, 100) + '%';
        }
    }

    showFieldTooltip(event, field) {
        const tooltip = document.getElementById('tooltip');
        const sampleValues = field.sample_values ? field.sample_values.slice(0, 3).join(', ') : 'No samples';
        
        tooltip.innerHTML = `
            <strong>${field.name}</strong><br>
            Type: ${this.formatDataType(field.data_type)}<br>
            ${field.max_length ? `Length: ${field.max_length}<br>` : ''}
            ${field.nullable ? 'Nullable' : 'Not Null'}<br>
            Samples: ${sampleValues}
        `;
        
        tooltip.style.left = (event.pageX + 10) + 'px';
        tooltip.style.top = (event.pageY - 10) + 'px';
        tooltip.classList.add('visible');
    }

    hideTooltip() {
        document.getElementById('tooltip').classList.remove('visible');
    }

    showStatus(message, type) {
        const indicator = document.getElementById('statusIndicator');
        indicator.textContent = message;
        indicator.className = 'status-indicator';
        
        switch (type) {
            case 'success':
                indicator.classList.add('status-success');
                break;
            case 'warning':
                indicator.classList.add('status-warning');
                break;
            case 'error':
                indicator.classList.add('status-error');
                break;
            default:
                indicator.classList.add('status-info');
        }

        if (type !== 'loading') {
            setTimeout(() => {
                indicator.textContent = 'Ready';
                indicator.className = 'status-indicator status-success';
            }, 3000);
        }
    }
}

// Global instance
let fieldMappingInterface;

// Global functions for HTML button handlers
function initializeFieldMapping() {
    fieldMappingInterface = new FieldMappingInterface();
    
    // Load sample data for demonstration
    loadSampleSchemas();
}

function loadSchema() {
    // Open schema selection dialog
    // For now, use sample data
    loadSampleSchemas();
}

function generateMappings() {
    if (fieldMappingInterface) {
        fieldMappingInterface.generateIntelligentMappings();
    }
}

function saveMappings() {
    if (fieldMappingInterface) {
        fieldMappingInterface.saveMappingConfiguration();
    }
}

function closeTransformationPanel() {
    if (fieldMappingInterface) {
        fieldMappingInterface.closeTransformationPanel();
    }
}

function applyTransformation() {
    if (fieldMappingInterface) {
        fieldMappingInterface.applyTransformation();
    }
}

// Sample data for demonstration
function loadSampleSchemas() {
    const sampleSourceSchema = {
        id: 'source_schema_1',
        name: 'customers',
        database: 'crm',
        fields: [
            {
                name: 'id',
                data_type: 'integer',
                nullable: false,
                primary_key: true,
                sample_values: [1, 2, 3]
            },
            {
                name: 'first_name',
                data_type: 'string',
                nullable: false,
                max_length: 50,
                sample_values: ['John', 'Jane', 'Bob']
            },
            {
                name: 'last_name',
                data_type: 'string',
                nullable: false,
                max_length: 50,
                sample_values: ['Smith', 'Doe', 'Johnson']
            },
            {
                name: 'email_address',
                data_type: 'email',
                nullable: false,
                max_length: 255,
                sample_values: ['john@example.com', 'jane@test.com', 'bob@demo.org']
            },
            {
                name: 'phone_number',
                data_type: 'phone',
                nullable: true,
                max_length: 20,
                sample_values: ['+1234567890', '555-0123', '(555) 456-7890']
            },
            {
                name: 'birth_date',
                data_type: 'date',
                nullable: true,
                sample_values: ['1990-05-15', '1985-12-03', '1992-08-22']
            },
            {
                name: 'created_at',
                data_type: 'timestamp',
                nullable: false,
                sample_values: ['2024-01-15 10:30:00', '2024-01-16 14:22:15', '2024-01-17 09:45:30']
            }
        ]
    };

    const sampleTargetSchema = {
        id: 'target_schema_1',
        name: 'customer_profile',
        database: 'warehouse',
        fields: [
            {
                name: 'customer_id',
                data_type: 'integer',
                nullable: false,
                primary_key: true
            },
            {
                name: 'full_name',
                data_type: 'string',
                nullable: false,
                max_length: 100
            },
            {
                name: 'email',
                data_type: 'string',
                nullable: false,
                max_length: 255
            },
            {
                name: 'phone',
                data_type: 'string',
                nullable: true,
                max_length: 15
            },
            {
                name: 'date_of_birth',
                data_type: 'date',
                nullable: true
            },
            {
                name: 'registration_timestamp',
                data_type: 'timestamp',
                nullable: false
            },
            {
                name: 'age_group',
                data_type: 'string',
                nullable: true,
                max_length: 20
            }
        ]
    };

    fieldMappingInterface.sourceSchema = sampleSourceSchema;
    fieldMappingInterface.targetSchema = sampleTargetSchema;
    
    fieldMappingInterface.renderSourceFields();
    fieldMappingInterface.renderTargetFields();
    
    fieldMappingInterface.showStatus('Sample schemas loaded', 'success');
}