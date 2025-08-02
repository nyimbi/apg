/**
 * APG Workflow Component Library
 * 
 * Component palette and library management for the workflow designer.
 * 
 * © 2025 Datacraft. All rights reserved.
 * Author: Nyimbi Odero <nyimbi@gmail.com>
 */

class ComponentLibrary {
    constructor(designer) {
        this.designer = designer;
        this.components = new Map();
        this.categories = new Map();
        this.filteredComponents = new Map();
        this.currentCategory = 'all';
        this.searchQuery = '';
        
        // DOM elements
        this.searchInput = document.getElementById('component-search');
        this.categoriesContainer = document.querySelector('.component-categories');
        this.componentsList = document.querySelector('.component-list');
        
        this.initialize();
    }
    
    initialize() {
        this.setupEventHandlers();
        console.log('Component library initialized');
    }
    
    setupEventHandlers() {
        // Search functionality
        this.searchInput.addEventListener('input', (e) => {
            this.searchQuery = e.target.value.toLowerCase();
            this.filterComponents();
        });
        
        // Clear search on escape
        this.searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.searchInput.value = '';
                this.searchQuery = '';
                this.filterComponents();
            }
        });
    }
    
    async loadComponents() {
        try {
            const response = await fetch(`${this.designer.config.api_base_url}/components`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Failed to load components');
            }
            
            // Process components
            this.processComponents(result.components);
            
            // Render UI
            this.renderCategories();
            this.renderComponents();
            
            console.log(`Loaded ${this.components.size} components`);
            
        } catch (error) {
            console.error('Failed to load components:', error);
            this.designer.showError('Failed to load components: ' + error.message);
        }
    }
    
    processComponents(componentsData) {
        // Clear existing data
        this.components.clear();
        this.categories.clear();
        
        // Group components by category
        const categoryGroups = {};
        
        componentsData.forEach(component => {
            this.components.set(component.id, component);
            
            const category = component.category || 'other';
            if (!categoryGroups[category]) {
                categoryGroups[category] = [];
            }
            categoryGroups[category].push(component);
            
            // Extract category info
            if (!this.categories.has(category)) {
                this.categories.set(category, {
                    id: category,
                    name: this.formatCategoryName(category),
                    icon: this.getCategoryIcon(category),
                    color: this.getCategoryColor(category),
                    count: 0
                });
            }
            
            this.categories.get(category).count++;
        });
        
        // Initialize filtered components
        this.filteredComponents = new Map(this.components);
    }
    
    formatCategoryName(category) {
        return category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    getCategoryIcon(category) {
        const iconMap = {
            'triggers': 'fa-play-circle',
            'data': 'fa-database',
            'logic': 'fa-code-branch',
            'integrations': 'fa-plug',
            'ai_ml': 'fa-brain',
            'utilities': 'fa-tools'
        };
        return iconMap[category] || 'fa-folder';
    }
    
    getCategoryColor(category) {
        const colorMap = {
            'triggers': '#e74c3c',
            'data': '#3498db',
            'logic': '#f39c12',
            'integrations': '#9b59b6',
            'ai_ml': '#1abc9c',
            'utilities': '#95a5a6'
        };
        return colorMap[category] || '#6c757d';
    }
    
    renderCategories() {
        this.categoriesContainer.innerHTML = '';
        
        // Add "All" category
        const allCategory = this.createCategoryElement({
            id: 'all',
            name: 'All Components',
            icon: 'fa-th-large',
            count: this.components.size
        }, this.currentCategory === 'all');
        
        this.categoriesContainer.appendChild(allCategory);
        
        // Add other categories
        Array.from(this.categories.values())
            .sort((a, b) => a.name.localeCompare(b.name))
            .forEach(category => {
                const categoryElement = this.createCategoryElement(category, this.currentCategory === category.id);
                this.categoriesContainer.appendChild(categoryElement);
            });
    }
    
    createCategoryElement(category, isActive = false) {
        const button = document.createElement('button');
        button.className = `category-item ${isActive ? 'active' : ''}`;
        button.dataset.categoryId = category.id;
        
        button.innerHTML = `
            <i class="fas ${category.icon}"></i>
            <span class="category-name">${category.name}</span>
            <span class="badge badge-secondary">${category.count}</span>
        `;
        
        button.addEventListener('click', () => {
            this.selectCategory(category.id);
        });
        
        return button;
    }
    
    selectCategory(categoryId) {
        this.currentCategory = categoryId;
        
        // Update UI
        document.querySelectorAll('.category-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-category-id="${categoryId}"]`).classList.add('active');
        
        // Filter and render components
        this.filterComponents();
    }
    
    filterComponents() {
        this.filteredComponents.clear();
        
        this.components.forEach(component => {
            // Category filter
            const categoryMatch = this.currentCategory === 'all' || component.category === this.currentCategory;
            
            // Search filter
            const searchMatch = !this.searchQuery || 
                component.name.toLowerCase().includes(this.searchQuery) ||
                component.description.toLowerCase().includes(this.searchQuery) ||
                (component.tags && component.tags.some(tag => tag.toLowerCase().includes(this.searchQuery)));
            
            if (categoryMatch && searchMatch) {
                this.filteredComponents.set(component.id, component);
            }
        });
        
        this.renderComponents();
    }
    
    renderComponents() {
        this.componentsList.innerHTML = '';
        
        if (this.filteredComponents.size === 0) {
            this.componentsList.innerHTML = `
                <div class="no-components">
                    <i class="fas fa-search"></i>
                    <p>No components found</p>
                </div>
            `;
            return;
        }
        
        // Sort components by name
        const sortedComponents = Array.from(this.filteredComponents.values())
            .sort((a, b) => a.name.localeCompare(b.name));
        
        sortedComponents.forEach(component => {
            const componentElement = this.createComponentElement(component);
            this.componentsList.appendChild(componentElement);
        });
    }
    
    createComponentElement(component) {
        const div = document.createElement('div');
        div.className = 'component-item';
        div.draggable = true;
        div.dataset.componentType = component.id;
        
        const iconClass = this.getComponentIconClass(component.category);
        const iconColor = this.getCategoryColor(component.category);
        
        div.innerHTML = `
            <div class="component-icon ${iconClass}" style="background-color: ${iconColor}">
                <i class="fas ${component.icon || this.getDefaultIcon(component.category)}"></i>
            </div>
            <div class="component-details">
                <div class="component-name">${component.name}</div>
                <div class="component-description">${component.description}</div>
            </div>
        `;
        
        // Setup drag and drop
        this.setupComponentDragDrop(div, component);
        
        // Setup click handler for component info
        div.addEventListener('click', (e) => {
            if (!e.defaultPrevented) {
                this.showComponentInfo(component);
            }
        });
        
        return div;
    }
    
    getComponentIconClass(category) {
        const classMap = {
            'triggers': 'trigger',
            'data': 'data',
            'logic': 'logic',
            'integrations': 'integration',
            'ai_ml': 'ai',
            'utilities': 'utility'
        };
        return classMap[category] || 'utility';
    }
    
    getDefaultIcon(category) {
        const iconMap = {
            'triggers': 'fa-play',
            'data': 'fa-table',
            'logic': 'fa-code',
            'integrations': 'fa-link',
            'ai_ml': 'fa-robot',
            'utilities': 'fa-wrench'
        };
        return iconMap[category] || 'fa-cog';
    }
    
    setupComponentDragDrop(element, component) {
        element.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', component.id);
            e.dataTransfer.effectAllowed = 'copy';
            
            // Add visual feedback
            element.classList.add('dragging');
            
            // Create drag image
            this.createDragImage(element, component);
        });
        
        element.addEventListener('dragend', (e) => {
            element.classList.remove('dragging');
        });
        
        // Double-click to add component at center
        element.addEventListener('dblclick', (e) => {
            e.preventDefault();
            this.addComponentAtCenter(component);
        });
    }
    
    createDragImage(element, component) {
        // Create a custom drag image
        const dragImage = element.cloneNode(true);
        dragImage.style.position = 'absolute';
        dragImage.style.top = '-1000px';
        dragImage.style.opacity = '0.8';
        dragImage.style.transform = 'rotate(5deg)';
        dragImage.style.pointerEvents = 'none';
        
        document.body.appendChild(dragImage);
        
        setTimeout(() => {
            document.body.removeChild(dragImage);
        }, 0);
    }
    
    addComponentAtCenter(component) {
        // Calculate center position of visible canvas
        const canvasBounds = this.designer.canvas.viewportBounds;
        const centerPoint = this.designer.canvas.screenToCanvas(
            canvasBounds.width / 2,
            canvasBounds.height / 2
        );
        
        // Add component
        this.designer.addNode(component.id, centerPoint);
    }
    
    showComponentInfo(component) {
        // Show component information in a modal or panel
        const modal = this.createComponentInfoModal(component);
        document.body.appendChild(modal);
        
        // Show modal
        $(modal).modal('show');
        
        // Remove modal when hidden
        $(modal).on('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
    }
    
    createComponentInfoModal(component) {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas ${component.icon || this.getDefaultIcon(component.category)}"></i>
                            ${component.name}
                        </h5>
                        <button type="button" class="close" data-dismiss="modal">
                            <span>&times;</span>
                        </button>
                    </div>
                    <div class="modal-body">
                        <div class="component-info">
                            <div class="row">
                                <div class="col-md-8">
                                    <p class="component-description">${component.description}</p>
                                    
                                    ${component.input_ports && component.input_ports.length > 0 ? `
                                    <h6>Input Ports</h6>
                                    <ul class="port-list">
                                        ${component.input_ports.map(port => `
                                            <li>
                                                <strong>${port.name}</strong> (${port.type})
                                                ${port.required ? '<span class="badge badge-danger">Required</span>' : ''}
                                                ${port.description ? `<br><small class="text-muted">${port.description}</small>` : ''}
                                            </li>
                                        `).join('')}
                                    </ul>
                                    ` : ''}
                                    
                                    ${component.output_ports && component.output_ports.length > 0 ? `
                                    <h6>Output Ports</h6>
                                    <ul class="port-list">
                                        ${component.output_ports.map(port => `
                                            <li>
                                                <strong>${port.name}</strong> (${port.type})
                                                ${port.description ? `<br><small class="text-muted">${port.description}</small>` : ''}
                                            </li>
                                        `).join('')}
                                    </ul>
                                    ` : ''}
                                    
                                    ${component.properties && component.properties.length > 0 ? `
                                    <h6>Configuration Properties</h6>
                                    <ul class="properties-list">
                                        ${component.properties.map(prop => `
                                            <li>
                                                <strong>${prop.label || prop.name}</strong> (${prop.type})
                                                ${prop.required ? '<span class="badge badge-danger">Required</span>' : ''}
                                                ${prop.description ? `<br><small class="text-muted">${prop.description}</small>` : ''}
                                                ${prop.default_value ? `<br><small><em>Default: ${prop.default_value}</em></small>` : ''}
                                            </li>
                                        `).join('')}
                                    </ul>
                                    ` : ''}
                                </div>
                                <div class="col-md-4">
                                    <div class="component-meta">
                                        <p><strong>Category:</strong> ${this.formatCategoryName(component.category)}</p>
                                        <p><strong>Version:</strong> ${component.version || '1.0.0'}</p>
                                        <p><strong>Execution:</strong> ${component.execution_type || 'sync'}</p>
                                        
                                        ${component.tags && component.tags.length > 0 ? `
                                        <p><strong>Tags:</strong></p>
                                        <div class="component-tags">
                                            ${component.tags.map(tag => `<span class="badge badge-secondary">${tag}</span>`).join(' ')}
                                        </div>
                                        ` : ''}
                                        
                                        ${component.documentation_url ? `
                                        <p><a href="${component.documentation_url}" target="_blank" class="btn btn-sm btn-outline-primary">
                                            <i class="fas fa-external-link-alt"></i> Documentation
                                        </a></p>
                                        ` : ''}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="workflowDesigner.componentLibrary.addComponentAtCenter(${JSON.stringify(component).replace(/"/g, '&quot;')})">
                            <i class="fas fa-plus"></i> Add to Workflow
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        return modal;
    }
    
    // === Search and Filter Methods ===
    
    searchComponents(query) {
        this.searchInput.value = query;
        this.searchQuery = query.toLowerCase();
        this.filterComponents();
    }
    
    clearSearch() {
        this.searchInput.value = '';
        this.searchQuery = '';
        this.filterComponents();
    }
    
    getComponentsByCategory(categoryId) {
        const components = [];
        this.components.forEach(component => {
            if (categoryId === 'all' || component.category === categoryId) {
                components.push(component);
            }
        });
        return components;
    }
    
    getComponent(componentId) {
        return this.components.get(componentId);
    }
    
    // === Component Usage Tracking ===
    
    trackComponentUsage(componentId) {
        // Track component usage for analytics
        const component = this.components.get(componentId);
        if (component) {
            // Send usage analytics
            this.sendUsageAnalytics(componentId);
        }
    }
    
    sendUsageAnalytics(componentId) {
        // Send usage data to analytics service
        try {
            fetch('/api/v1/analytics/component-usage', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.designer.getCSRFToken()
                },
                body: JSON.stringify({
                    component_id: componentId,
                    session_id: this.designer.sessionId,
                    timestamp: new Date().toISOString()
                })
            }).catch(error => {
                console.warn('Failed to send usage analytics:', error);
            });
        } catch (error) {
            console.warn('Failed to send usage analytics:', error);
        }
    }
    
    // === Keyboard Shortcuts ===
    
    handleKeyboardShortcut(event) {
        // Handle component library specific shortcuts
        if (event.target === this.searchInput) {
            switch (event.key) {
                case 'ArrowDown':
                    event.preventDefault();
                    this.focusFirstComponent();
                    break;
                case 'Enter':
                    event.preventDefault();
                    this.addFirstFilteredComponent();
                    break;
            }
        }
    }
    
    focusFirstComponent() {
        const firstComponent = this.componentsList.querySelector('.component-item');
        if (firstComponent) {
            firstComponent.focus();
        }
    }
    
    addFirstFilteredComponent() {
        if (this.filteredComponents.size > 0) {
            const firstComponent = Array.from(this.filteredComponents.values())[0];
            this.addComponentAtCenter(firstComponent);
        }
    }
    
    // === Component Validation ===
    
    validateComponent(component) {
        const errors = [];
        
        // Basic validation
        if (!component.name) {
            errors.push('Component name is required');
        }
        
        if (!component.id) {
            errors.push('Component ID is required');
        }
        
        if (!component.category) {
            errors.push('Component category is required');
        }
        
        // Port validation
        if (component.input_ports) {
            component.input_ports.forEach((port, index) => {
                if (!port.name) {
                    errors.push(`Input port ${index + 1} is missing a name`);
                }
                if (!port.type) {
                    errors.push(`Input port ${port.name || index + 1} is missing a type`);
                }
            });
        }
        
        if (component.output_ports) {
            component.output_ports.forEach((port, index) => {
                if (!port.name) {
                    errors.push(`Output port ${index + 1} is missing a name`);
                }
                if (!port.type) {
                    errors.push(`Output port ${port.name || index + 1} is missing a type`);
                }
            });
        }
        
        // Property validation
        if (component.properties) {
            component.properties.forEach((prop, index) => {
                if (!prop.name) {
                    errors.push(`Property ${index + 1} is missing a name`);
                }
                if (!prop.type) {
                    errors.push(`Property ${prop.name || index + 1} is missing a type`);
                }
            });
        }
        
        return {
            valid: errors.length === 0,
            errors: errors
        };
    }
    
    // === Refresh and Reload ===
    
    async refresh() {
        try {
            await this.loadComponents();
            this.designer.showNotification('Component library refreshed', 'success');
        } catch (error) {
            console.error('Failed to refresh component library:', error);
            this.designer.showError('Failed to refresh component library: ' + error.message);
        }
    }
}

// Add styles for component library
const componentLibraryStyles = `
<style>
.no-components {
    text-align: center;
    padding: 40px 20px;
    color: #6c757d;
}

.no-components i {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.5;
}

.port-list, .properties-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.port-list li, .properties-list li {
    padding: 8px 0;
    border-bottom: 1px solid #dee2e6;
}

.port-list li:last-child, .properties-list li:last-child {
    border-bottom: none;
}

.component-tags {
    margin-top: 8px;
}

.component-tags .badge {
    margin-right: 4px;
    margin-bottom: 4px;
}

.component-meta {
    background: #f8f9fa;
    padding: 16px;
    border-radius: 4px;
    font-size: 13px;
}

.component-meta p {
    margin-bottom: 8px;
}

.component-meta p:last-child {
    margin-bottom: 0;
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', componentLibraryStyles);