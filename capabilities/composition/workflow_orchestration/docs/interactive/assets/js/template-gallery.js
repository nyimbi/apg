/**
 * APG Workflow Orchestration - Template Gallery JavaScript
 * Manages template browsing, filtering, and usage functionality
 */

class TemplateGallery {
    constructor() {
        this.templates = [];
        this.filteredTemplates = [];
        this.categories = new Set();
        this.tags = new Set();
        this.currentFilter = {
            category: '',
            complexity: '',
            search: '',
            tags: []
        };
        
        this.init();
    }
    
    init() {
        this.loadTemplates();
        this.setupFilters();
        this.setupSearch();
        this.setupSorting();
        this.bindEvents();
    }
    
    loadTemplates() {
        // Enhanced template data with more details
        this.templates = [
            {
                id: 'data-processing-pipeline',
                name: 'Data Processing Pipeline',
                description: 'Complete ETL pipeline for processing large datasets with validation, transformation, and error handling',
                category: 'data_processing',
                complexity: 'intermediate',
                tags: ['etl', 'data', 'transformation', 'validation', 'batch'],
                author: 'APG Team',
                rating: 4.8,
                usageCount: 1250,
                downloads: 3456,
                thumbnail: 'assets/images/templates/data-pipeline.svg',
                createdAt: '2024-01-10',
                updatedAt: '2024-01-28',
                version: '2.1.0',
                features: [
                    'Multi-source data ingestion',
                    'Real-time validation',
                    'Parallel processing',
                    'Error recovery',
                    'Performance monitoring'
                ],
                requirements: ['PostgreSQL', 'Redis', 'Python 3.9+'],
                estimatedTime: '30 minutes',
                difficulty: 'Intermediate'
            },
            {
                id: 'approval-workflow',
                name: 'Multi-Stage Approval Workflow',
                description: 'Configurable approval process with parallel approvers, escalation, and automatic notifications',
                category: 'business_process',
                complexity: 'beginner',
                tags: ['approval', 'business', 'workflow', 'notifications', 'escalation'],
                author: 'APG Team',
                rating: 4.6,
                usageCount: 890,
                downloads: 2123,
                thumbnail: 'assets/images/templates/approval-workflow.svg',
                createdAt: '2024-01-05',
                updatedAt: '2024-01-25',
                version: '1.3.2',
                features: [
                    'Multi-level approvals',
                    'Parallel approval paths',
                    'Auto-escalation',
                    'Email notifications',
                    'Audit trails'
                ],
                requirements: ['Email service', 'User management'],
                estimatedTime: '15 minutes',
                difficulty: 'Beginner'
            },
            {
                id: 'api-integration',
                name: 'REST API Integration Hub',
                description: 'Comprehensive API integration with rate limiting, retry logic, and data synchronization',
                category: 'integration',
                complexity: 'advanced',
                tags: ['api', 'integration', 'sync', 'rest', 'webhooks'],
                author: 'Community',
                rating: 4.7,
                usageCount: 567,
                downloads: 1789,
                thumbnail: 'assets/images/templates/api-integration.svg',
                createdAt: '2024-01-15',
                updatedAt: '2024-01-30',
                version: '3.0.1',
                features: [
                    'Multiple API endpoints',
                    'Rate limiting',
                    'Retry mechanisms',
                    'Data mapping',
                    'Error handling'
                ],
                requirements: ['HTTP client', 'JSON parser'],
                estimatedTime: '45 minutes',
                difficulty: 'Advanced'
            },
            {
                id: 'ml-data-pipeline',
                name: 'Machine Learning Data Pipeline',
                description: 'End-to-end ML pipeline with data preprocessing, model training, and deployment',
                category: 'analytics',
                complexity: 'advanced',
                tags: ['ml', 'ai', 'data', 'training', 'deployment'],
                author: 'ML Team',
                rating: 4.9,
                usageCount: 342,
                downloads: 987,
                thumbnail: 'assets/images/templates/ml-pipeline.svg',
                createdAt: '2024-01-20',
                updatedAt: '2024-01-31',
                version: '1.0.0',
                features: [
                    'Data preprocessing',
                    'Feature engineering',
                    'Model training',
                    'Hyperparameter tuning',
                    'Model deployment'
                ],
                requirements: ['scikit-learn', 'TensorFlow', 'GPU (optional)'],
                estimatedTime: '90 minutes',
                difficulty: 'Advanced'
            },
            {
                id: 'devops-cicd',
                name: 'CI/CD Deployment Pipeline',
                description: 'Complete DevOps pipeline with testing, building, and multi-environment deployment',
                category: 'devops',
                complexity: 'intermediate',
                tags: ['cicd', 'deployment', 'testing', 'docker', 'kubernetes'],
                author: 'DevOps Team',
                rating: 4.5,
                usageCount: 678,
                downloads: 1456,
                thumbnail: 'assets/images/templates/cicd-pipeline.svg',
                createdAt: '2024-01-12',
                updatedAt: '2024-01-29',
                version: '2.2.0',
                features: [
                    'Automated testing',
                    'Docker containerization',
                    'Multi-stage deployment',
                    'Rollback mechanisms',
                    'Environment promotion'
                ],
                requirements: ['Docker', 'Kubernetes', 'Git'],
                estimatedTime: '60 minutes',
                difficulty: 'Intermediate'
            },
            {
                id: 'ecommerce-order-processing',
                name: 'E-commerce Order Processing',
                description: 'Complete order lifecycle management with payment, inventory, and fulfillment',
                category: 'business_process',
                complexity: 'intermediate',
                tags: ['ecommerce', 'orders', 'payment', 'inventory', 'fulfillment'],
                author: 'E-commerce Team',
                rating: 4.4,
                usageCount: 445,
                downloads: 1234,
                thumbnail: 'assets/images/templates/ecommerce-order.svg',
                createdAt: '2024-01-08',
                updatedAt: '2024-01-27',
                version: '1.5.3',
                features: [
                    'Order validation',
                    'Payment processing',
                    'Inventory management',
                    'Shipping integration',
                    'Customer notifications'
                ],
                requirements: ['Payment gateway', 'Inventory system'],
                estimatedTime: '40 minutes',
                difficulty: 'Intermediate'
            }
        ];
        
        // Build categories and tags sets
        this.templates.forEach(template => {
            this.categories.add(template.category);
            template.tags.forEach(tag => this.tags.add(tag));
        });
        
        this.filteredTemplates = [...this.templates];
    }
    
    setupFilters() {
        const categoryFilter = document.getElementById('categoryFilter');
        const complexityFilter = document.getElementById('complexityFilter');
        
        if (categoryFilter) {
            // Populate category options
            this.categories.forEach(category => {
                const option = document.createElement('option');
                option.value = category;
                option.textContent = this.formatCategoryName(category);
                categoryFilter.appendChild(option);
            });
            
            categoryFilter.addEventListener('change', (e) => {
                this.currentFilter.category = e.target.value;
                this.applyFilters();
            });
        }
        
        if (complexityFilter) {
            complexityFilter.addEventListener('change', (e) => {
                this.currentFilter.complexity = e.target.value;
                this.applyFilters();
            });
        }
    }
    
    setupSearch() {
        const searchInput = document.getElementById('templateSearch');
        if (!searchInput) return;
        
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.currentFilter.search = e.target.value.toLowerCase();
                this.applyFilters();
            }, 300);
        });
    }
    
    setupSorting() {
        // Create sorting controls
        const filtersContainer = document.querySelector('.template-filters');
        if (!filtersContainer) return;
        
        const sortGroup = document.createElement('div');
        sortGroup.className = 'filter-group';
        
        const sortLabel = document.createElement('label');
        sortLabel.textContent = 'Sort by:';
        
        const sortSelect = document.createElement('select');
        sortSelect.id = 'sortFilter';
        
        const sortOptions = [
            { value: 'popular', text: 'Most Popular' },
            { value: 'rating', text: 'Highest Rated' },
            { value: 'newest', text: 'Newest' },
            { value: 'name', text: 'Name A-Z' },
            { value: 'downloads', text: 'Most Downloaded' }
        ];
        
        sortOptions.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option.value;
            optionElement.textContent = option.text;
            sortSelect.appendChild(optionElement);
        });
        
        sortSelect.addEventListener('change', (e) => {
            this.sortTemplates(e.target.value);
            this.renderTemplates();
        });
        
        sortGroup.appendChild(sortLabel);
        sortGroup.appendChild(sortSelect);
        filtersContainer.appendChild(sortGroup);
    }
    
    bindEvents() {
        // Add tag filter functionality
        this.createTagFilter();
        
        // Add view mode toggle
        this.createViewModeToggle();
        
        // Handle keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'f') {
                e.preventDefault();
                document.getElementById('templateSearch')?.focus();
            }
        });
    }
    
    createTagFilter() {
        const filtersContainer = document.querySelector('.template-filters');
        if (!filtersContainer) return;
        
        const tagGroup = document.createElement('div');
        tagGroup.className = 'filter-group tag-filter-group';
        
        const tagLabel = document.createElement('label');
        tagLabel.textContent = 'Tags:';
        
        const tagContainer = document.createElement('div');
        tagContainer.className = 'tag-filter-container';
        
        // Create popular tags
        const popularTags = Array.from(this.tags).slice(0, 8);
        popularTags.forEach(tag => {
            const tagButton = document.createElement('button');
            tagButton.className = 'tag-filter-btn';
            tagButton.textContent = tag;
            tagButton.addEventListener('click', () => {
                tagButton.classList.toggle('active');
                this.toggleTagFilter(tag);
            });
            tagContainer.appendChild(tagButton);
        });
        
        tagGroup.appendChild(tagLabel);
        tagGroup.appendChild(tagContainer);
        filtersContainer.appendChild(tagGroup);
    }
    
    createViewModeToggle() {
        const gallerySection = document.getElementById('template-gallery');
        if (!gallerySection) return;
        
        const header = gallerySection.querySelector('.section-header');
        if (!header) return;
        
        const viewToggle = document.createElement('div');
        viewToggle.className = 'view-mode-toggle';
        
        const gridBtn = document.createElement('button');
        gridBtn.className = 'view-btn active';
        gridBtn.innerHTML = '<i class="fas fa-th"></i>';
        gridBtn.title = 'Grid View';
        
        const listBtn = document.createElement('button');
        listBtn.className = 'view-btn';
        listBtn.innerHTML = '<i class="fas fa-list"></i>';
        listBtn.title = 'List View';
        
        gridBtn.addEventListener('click', () => {
            this.setViewMode('grid');
            gridBtn.classList.add('active');
            listBtn.classList.remove('active');
        });
        
        listBtn.addEventListener('click', () => {
            this.setViewMode('list');
            listBtn.classList.add('active');
            gridBtn.classList.remove('active');
        });
        
        viewToggle.appendChild(gridBtn);
        viewToggle.appendChild(listBtn);
        header.appendChild(viewToggle);
    }
    
    toggleTagFilter(tag) {
        const index = this.currentFilter.tags.indexOf(tag);
        if (index > -1) {
            this.currentFilter.tags.splice(index, 1);
        } else {
            this.currentFilter.tags.push(tag);
        }
        this.applyFilters();
    }
    
    applyFilters() {
        this.filteredTemplates = this.templates.filter(template => {
            // Category filter
            if (this.currentFilter.category && template.category !== this.currentFilter.category) {
                return false;
            }
            
            // Complexity filter
            if (this.currentFilter.complexity && template.complexity !== this.currentFilter.complexity) {
                return false;
            }
            
            // Search filter
            if (this.currentFilter.search) {
                const searchTerm = this.currentFilter.search;
                const searchableText = `${template.name} ${template.description} ${template.tags.join(' ')}`.toLowerCase();
                if (!searchableText.includes(searchTerm)) {
                    return false;
                }
            }
            
            // Tag filters
            if (this.currentFilter.tags.length > 0) {
                const hasAllTags = this.currentFilter.tags.every(tag => 
                    template.tags.includes(tag)
                );
                if (!hasAllTags) {
                    return false;
                }
            }
            
            return true;
        });
        
        this.renderTemplates();
        this.updateResultsCount();
    }
    
    sortTemplates(sortBy) {
        this.filteredTemplates.sort((a, b) => {
            switch (sortBy) {
                case 'popular':
                    return b.usageCount - a.usageCount;
                case 'rating':
                    return b.rating - a.rating;
                case 'newest':
                    return new Date(b.createdAt) - new Date(a.createdAt);
                case 'name':
                    return a.name.localeCompare(b.name);
                case 'downloads':
                    return b.downloads - a.downloads;
                default:
                    return 0;
            }
        });
    }
    
    renderTemplates() {
        const templateGrid = document.getElementById('templateGrid');
        if (!templateGrid) return;
        
        if (this.filteredTemplates.length === 0) {
            templateGrid.innerHTML = this.renderEmptyState();
            return;
        }
        
        templateGrid.innerHTML = this.filteredTemplates.map(template => 
            this.renderTemplateCard(template)
        ).join('');
        
        // Bind template action events
        this.bindTemplateEvents();
    }
    
    renderTemplateCard(template) {
        const featuresHtml = template.features.slice(0, 3).map(feature => 
            `<li><i class="fas fa-check"></i> ${feature}</li>`
        ).join('');
        
        const tagsHtml = template.tags.slice(0, 3).map(tag => 
            `<span class="template-tag">${tag}</span>`
        ).join('');
        
        return `
            <div class="template-card" data-template="${template.id}">
                <div class="template-thumbnail">
                    <img src="${template.thumbnail}" alt="${template.name}" 
                         onerror="this.src='assets/images/template-placeholder.svg'">
                    <div class="template-overlay">
                        <button class="btn btn-primary preview-btn" data-template="${template.id}">
                            <i class="fas fa-eye"></i> Preview
                        </button>
                    </div>
                </div>
                
                <div class="template-content">
                    <div class="template-header">
                        <h3>${template.name}</h3>
                        <span class="template-version">v${template.version}</span>
                    </div>
                    
                    <p class="template-description">${template.description}</p>
                    
                    <div class="template-features">
                        <ul>${featuresHtml}</ul>
                    </div>
                    
                    <div class="template-meta">
                        <span class="template-category">${this.formatCategoryName(template.category)}</span>
                        <span class="template-complexity complexity-${template.complexity}">${template.complexity}</span>
                        <span class="template-time"><i class="fas fa-clock"></i> ${template.estimatedTime}</span>
                    </div>
                    
                    <div class="template-tags">
                        ${tagsHtml}
                    </div>
                    
                    <div class="template-stats">
                        <div class="stat-item">
                            <i class="fas fa-star"></i>
                            <span>${template.rating}</span>
                        </div>
                        <div class="stat-item">
                            <i class="fas fa-download"></i>
                            <span>${this.formatNumber(template.downloads)}</span>
                        </div>
                        <div class="stat-item">
                            <i class="fas fa-users"></i>
                            <span>${this.formatNumber(template.usageCount)}</span>
                        </div>
                    </div>
                    
                    <div class="template-actions">
                        <button class="btn btn-primary use-template-btn" data-template="${template.id}">
                            <i class="fas fa-plus"></i> Use Template
                        </button>
                        <button class="btn btn-outline details-btn" data-template="${template.id}">
                            <i class="fas fa-info-circle"></i> Details
                        </button>
                        <button class="btn btn-outline favorite-btn" data-template="${template.id}">
                            <i class="far fa-heart"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }
    
    renderEmptyState() {
        return `
            <div class="empty-state">
                <i class="fas fa-search fa-3x"></i>
                <h3>No templates found</h3>
                <p>Try adjusting your filters or search terms</p>
                <button class="btn btn-primary" onclick="templateGallery.clearFilters()">
                    <i class="fas fa-refresh"></i> Clear Filters
                </button>
            </div>
        `;
    }
    
    bindTemplateEvents() {
        // Use template buttons
        document.querySelectorAll('.use-template-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const templateId = e.target.getAttribute('data-template');
                this.useTemplate(templateId);
            });
        });
        
        // Preview buttons
        document.querySelectorAll('.preview-btn, .details-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const templateId = e.target.getAttribute('data-template');
                this.showTemplateDetails(templateId);
            });
        });
        
        // Favorite buttons
        document.querySelectorAll('.favorite-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const templateId = e.target.getAttribute('data-template');
                this.toggleFavorite(templateId, btn);
            });
        });
    }
    
    useTemplate(templateId) {
        const template = this.templates.find(t => t.id === templateId);
        if (!template) return;
        
        // Show loading state
        this.showLoadingModal('Creating workflow from template...');
        
        // Simulate template usage (in real implementation, this would call the API)
        setTimeout(() => {
            this.hideLoadingModal();
            this.showSuccessMessage(`Workflow created from "${template.name}" template!`);
            
            // Navigate to workflow builder
            if (window.interactiveDocs) {
                window.interactiveDocs.showSection('workflow-builder');
            }
        }, 2000);
    }
    
    showTemplateDetails(templateId) {
        const template = this.templates.find(t => t.id === templateId);
        if (!template) return;
        
        const modal = this.createTemplateModal(template);
        document.body.appendChild(modal);
        
        // Show modal with animation
        setTimeout(() => modal.classList.add('show'), 10);
    }
    
    createTemplateModal(template) {
        const modal = document.createElement('div');
        modal.className = 'template-modal';
        modal.innerHTML = `
            <div class="modal-overlay"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h2>${template.name}</h2>
                    <button class="modal-close">&times;</button>
                </div>
                
                <div class="modal-body">
                    <div class="template-details-grid">
                        <div class="details-main">
                            <img src="${template.thumbnail}" alt="${template.name}" class="template-image">
                            
                            <div class="template-info">
                                <p class="template-full-description">${template.description}</p>
                                
                                <div class="template-features-full">
                                    <h4>Features</h4>
                                    <ul>
                                        ${template.features.map(feature => `<li><i class="fas fa-check"></i> ${feature}</li>`).join('')}
                                    </ul>
                                </div>
                                
                                <div class="template-requirements">
                                    <h4>Requirements</h4>
                                    <div class="requirement-tags">
                                        ${template.requirements.map(req => `<span class="requirement-tag">${req}</span>`).join('')}
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="details-sidebar">
                            <div class="template-metadata">
                                <div class="meta-item">
                                    <label>Category:</label>
                                    <span>${this.formatCategoryName(template.category)}</span>
                                </div>
                                <div class="meta-item">
                                    <label>Complexity:</label>
                                    <span class="complexity-${template.complexity}">${template.complexity}</span>
                                </div>
                                <div class="meta-item">
                                    <label>Estimated Time:</label>
                                    <span><i class="fas fa-clock"></i> ${template.estimatedTime}</span>
                                </div>
                                <div class="meta-item">
                                    <label>Version:</label>
                                    <span>v${template.version}</span>
                                </div>
                                <div class="meta-item">
                                    <label>Author:</label>
                                    <span>${template.author}</span>
                                </div>
                                <div class="meta-item">
                                    <label>Downloads:</label>
                                    <span><i class="fas fa-download"></i> ${this.formatNumber(template.downloads)}</span>
                                </div>
                                <div class="meta-item">
                                    <label>Rating:</label>
                                    <span><i class="fas fa-star"></i> ${template.rating}/5</span>
                                </div>
                            </div>
                            
                            <div class="template-tags-full">
                                ${template.tags.map(tag => `<span class="template-tag">${tag}</span>`).join('')}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="modal-footer">
                    <button class="btn btn-outline modal-cancel">Cancel</button>
                    <button class="btn btn-primary use-template-modal" data-template="${template.id}">
                        <i class="fas fa-plus"></i> Use This Template
                    </button>
                </div>
            </div>
        `;
        
        // Bind modal events
        const closeBtn = modal.querySelector('.modal-close');
        const cancelBtn = modal.querySelector('.modal-cancel');
        const overlay = modal.querySelector('.modal-overlay');
        const useBtn = modal.querySelector('.use-template-modal');
        
        [closeBtn, cancelBtn, overlay].forEach(element => {
            element.addEventListener('click', () => this.closeModal(modal));
        });
        
        useBtn.addEventListener('click', () => {
            this.closeModal(modal);
            this.useTemplate(template.id);
        });
        
        return modal;
    }
    
    closeModal(modal) {
        modal.classList.remove('show');
        setTimeout(() => modal.remove(), 300);
    }
    
    toggleFavorite(templateId, button) {
        const icon = button.querySelector('i');
        const isFavorited = icon.classList.contains('fas');
        
        if (isFavorited) {
            icon.classList.remove('fas');
            icon.classList.add('far');
            button.title = 'Add to favorites';
        } else {
            icon.classList.remove('far');
            icon.classList.add('fas');
            button.title = 'Remove from favorites';
        }
        
        // In real implementation, this would sync with backend
        console.log(`Template ${templateId} ${isFavorited ? 'removed from' : 'added to'} favorites`);
    }
    
    setViewMode(mode) {
        const templateGrid = document.getElementById('templateGrid');
        if (!templateGrid) return;
        
        templateGrid.className = mode === 'list' ? 'template-list' : 'template-grid';
        this.currentViewMode = mode;
    }
    
    clearFilters() {
        this.currentFilter = {
            category: '',
            complexity: '',
            search: '',
            tags: []
        };
        
        // Reset form elements
        document.getElementById('categoryFilter').value = '';
        document.getElementById('complexityFilter').value = '';
        document.getElementById('templateSearch').value = '';
        
        // Reset tag buttons
        document.querySelectorAll('.tag-filter-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        this.applyFilters();
    }
    
    updateResultsCount() {
        const header = document.querySelector('#template-gallery .section-subtitle');
        if (!header) return;
        
        const count = this.filteredTemplates.length;
        const total = this.templates.length;
        
        if (count === total) {
            header.textContent = `${total} templates available`;
        } else {
            header.textContent = `Showing ${count} of ${total} templates`;
        }
    }
    
    showLoadingModal(message) {
        const modal = document.createElement('div');
        modal.className = 'loading-modal';
        modal.innerHTML = `
            <div class="modal-overlay"></div>
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <p>${message}</p>
            </div>
        `;
        document.body.appendChild(modal);
        setTimeout(() => modal.classList.add('show'), 10);
        this.currentLoadingModal = modal;
    }
    
    hideLoadingModal() {
        if (this.currentLoadingModal) {
            this.currentLoadingModal.classList.remove('show');
            setTimeout(() => this.currentLoadingModal.remove(), 300);
            this.currentLoadingModal = null;
        }
    }
    
    showSuccessMessage(message) {
        const toast = document.createElement('div');
        toast.className = 'toast toast-success';
        toast.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    formatCategoryName(category) {
        return category.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
}

// Initialize template gallery when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.templateGallery = new TemplateGallery();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TemplateGallery;
}