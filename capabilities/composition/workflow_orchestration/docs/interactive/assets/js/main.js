/**
 * APG Workflow Orchestration - Interactive Documentation
 * Main JavaScript functionality
 */

class InteractiveDocumentation {
    constructor() {
        this.currentSection = 'overview';
        this.searchIndex = new Map();
        this.apiEndpoints = new Map();
        this.templates = [];
        
        this.init();
    }
    
    init() {
        this.setupNavigation();
        this.setupSearch();
        this.setupThemeToggle();
        this.setupCodeTabs();
        this.loadAPIEndpoints();
        this.loadTemplates();
        this.setupEventListeners();
        
        // Initialize URL routing
        this.handleUrlRoute();
        window.addEventListener('popstate', () => this.handleUrlRoute());
    }
    
    setupNavigation() {
        const navLinks = document.querySelectorAll('[data-section]');
        const navToggle = document.getElementById('navToggle');
        const docsNav = document.getElementById('docsNav');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.getAttribute('data-section');
                this.showSection(section);
                
                // Close mobile nav
                if (window.innerWidth <= 1024) {
                    docsNav.classList.remove('open');
                }
            });
        });
        
        // Mobile navigation toggle
        if (navToggle) {
            navToggle.addEventListener('click', () => {
                docsNav.classList.toggle('open');
            });
        }
        
        // Close nav when clicking outside
        document.addEventListener('click', (e) => {
            if (!docsNav.contains(e.target) && !navToggle.contains(e.target)) {
                docsNav.classList.remove('open');
            }
        });
    }
    
    setupSearch() {
        const searchInput = document.getElementById('searchInput');
        if (!searchInput) return;
        
        // Build search index
        this.buildSearchIndex();
        
        // Setup search functionality
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.performSearch(e.target.value);
            }, 300);
        });
        
        // Handle search keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                searchInput.focus();
            }
            
            if (e.key === 'Escape' && document.activeElement === searchInput) {
                searchInput.blur();
                this.clearSearch();
            }
        });
    }
    
    buildSearchIndex() {
        const sections = document.querySelectorAll('.content-section');
        
        sections.forEach(section => {
            const sectionId = section.id;
            const title = section.querySelector('h1')?.textContent || '';
            const content = section.textContent.toLowerCase();
            
            this.searchIndex.set(sectionId, {
                title,
                content,
                element: section
            });
        });
    }
    
    performSearch(query) {
        if (!query.trim()) {
            this.clearSearch();
            return;
        }
        
        query = query.toLowerCase();
        const results = [];
        
        this.searchIndex.forEach((data, sectionId) => {
            if (data.title.toLowerCase().includes(query) || 
                data.content.includes(query)) {
                results.push({
                    sectionId,
                    title: data.title,
                    relevance: this.calculateRelevance(query, data)
                });
            }
        });
        
        // Sort by relevance
        results.sort((a, b) => b.relevance - a.relevance);
        
        this.displaySearchResults(results);
    }
    
    calculateRelevance(query, data) {
        let score = 0;
        const titleLower = data.title.toLowerCase();
        
        // Title exact match
        if (titleLower === query) score += 100;
        // Title starts with query
        else if (titleLower.startsWith(query)) score += 50;
        // Title contains query
        else if (titleLower.includes(query)) score += 25;
        
        // Content matches
        const contentMatches = (data.content.match(new RegExp(query, 'gi')) || []).length;
        score += contentMatches * 5;
        
        return score;
    }
    
    displaySearchResults(results) {
        const searchResultsContainer = document.getElementById('searchResults');
        if (!searchResultsContainer) {
            // Create search results container if it doesn't exist
            const container = document.createElement('div');
            container.id = 'searchResults';
            container.className = 'search-results';
            
            const sidebar = document.querySelector('.nav-sidebar');
            if (sidebar) {
                sidebar.appendChild(container);
            }
        }
        
        const resultsContainer = document.getElementById('searchResults');
        resultsContainer.innerHTML = '';
        
        if (results.length === 0) {
            resultsContainer.innerHTML = '<div class="no-results">No results found</div>';
            return;
        }
        
        const resultsHeader = document.createElement('h4');
        resultsHeader.textContent = `Search Results (${results.length})`;
        resultsContainer.appendChild(resultsHeader);
        
        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'search-result-item';
            resultItem.innerHTML = `
                <div class="result-title">${result.title}</div>
                <div class="result-snippet">${this.highlightSearchTerms(result.snippet, this.lastSearchTerm)}</div>
                <div class="result-meta">Section: ${result.section}</div>
            `;
            
            resultItem.addEventListener('click', () => {
                // Navigate to the section and highlight the result
                this.showSection(result.section);
                this.highlightSearchResult(result.element);
                this.clearSearch();
            });
            
            resultsContainer.appendChild(resultItem);
        });
        
        resultsContainer.style.display = 'block';
    }
    
    clearSearch() {
        const searchInput = document.getElementById('searchInput');
        const searchResults = document.getElementById('searchResults');
        
        if (searchInput) {
            searchInput.value = '';
        }
        
        if (searchResults) {
            searchResults.style.display = 'none';
            searchResults.innerHTML = '';
        }
        
        // Clear any highlighted search results
        document.querySelectorAll('.search-highlight').forEach(element => {
            element.classList.remove('search-highlight');
        });
        
        this.lastSearchTerm = '';
    }
    
    setupThemeToggle() {
        // Create theme toggle button
        const themeToggle = document.createElement('button');
        themeToggle.className = 'theme-toggle';
        themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        themeToggle.setAttribute('aria-label', 'Toggle dark mode');
        
        const navHeader = document.querySelector('.nav-header');
        if (navHeader) {
            navHeader.appendChild(themeToggle);
        }
        
        // Get saved theme
        const savedTheme = localStorage.getItem('docs-theme') || 'light';
        this.setTheme(savedTheme);
        
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            this.setTheme(newTheme);
        });
    }
    
    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('docs-theme', theme);
        
        const themeToggle = document.querySelector('.theme-toggle i');
        if (themeToggle) {
            themeToggle.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
        }
    }
    
    setupCodeTabs() {
        const codeTabs = document.querySelectorAll('.code-tabs');
        
        codeTabs.forEach(tabContainer => {
            const buttons = tabContainer.querySelectorAll('.tab-button');
            const panes = tabContainer.querySelectorAll('.tab-pane');
            
            buttons.forEach(button => {
                button.addEventListener('click', () => {
                    const targetTab = button.getAttribute('data-tab');
                    
                    // Remove active class from all buttons and panes
                    buttons.forEach(btn => btn.classList.remove('active'));
                    panes.forEach(pane => pane.classList.remove('active'));
                    
                    // Add active class to clicked button and corresponding pane
                    button.classList.add('active');
                    const targetPane = tabContainer.querySelector(`#${targetTab}`);
                    if (targetPane) {
                        targetPane.classList.add('active');
                    }
                });
            });
        });
    }
    
    loadAPIEndpoints() {
        // Mock API endpoints data
        const endpoints = [
            {
                id: 'get-workflows',
                method: 'GET',
                path: '/workflows',
                summary: 'List all workflows',
                description: 'Retrieve a list of all workflows with optional filtering and pagination.',
                parameters: [
                    { name: 'limit', type: 'integer', description: 'Maximum number of results' },
                    { name: 'offset', type: 'integer', description: 'Number of results to skip' },
                    { name: 'category', type: 'string', description: 'Filter by category' },
                    { name: 'status', type: 'string', description: 'Filter by status' }
                ],
                responses: {
                    200: { description: 'Success', example: { workflows: [], total: 0 } },
                    400: { description: 'Bad Request' },
                    401: { description: 'Unauthorized' }
                }
            },
            {
                id: 'post-workflows',
                method: 'POST',
                path: '/workflows',
                summary: 'Create a new workflow',
                description: 'Create a new workflow with the provided definition.',
                requestBody: {
                    name: 'string',
                    description: 'string',
                    definition: 'object',
                    category: 'string'
                },
                responses: {
                    201: { description: 'Created', example: { id: 'wf-123', status: 'created' } },
                    400: { description: 'Bad Request' },
                    401: { description: 'Unauthorized' }
                }
            }
            // Add more endpoints as needed
        ];
        
        endpoints.forEach(endpoint => {
            this.apiEndpoints.set(endpoint.id, endpoint);
        });
    }
    
    loadTemplates() {
        // Mock template data
        this.templates = [
            {
                id: 'data-processing-pipeline',
                name: 'Data Processing Pipeline',
                description: 'ETL pipeline for processing customer data',
                category: 'data_processing',
                complexity: 'intermediate',
                tags: ['etl', 'data', 'transformation'],
                author: 'APG Team',
                rating: 4.8,
                usageCount: 1250,
                thumbnail: 'assets/images/templates/data-pipeline.svg'
            },
            {
                id: 'approval-workflow',
                name: 'Approval Workflow',
                description: 'Multi-stage approval process for business documents',
                category: 'business_process',
                complexity: 'beginner',
                tags: ['approval', 'business', 'workflow'],
                author: 'APG Team',
                rating: 4.6,
                usageCount: 890,
                thumbnail: 'assets/images/templates/approval-workflow.svg'
            },
            {
                id: 'api-integration',
                name: 'API Integration',
                description: 'Sync data between multiple REST APIs',
                category: 'integration',
                complexity: 'advanced',
                tags: ['api', 'integration', 'sync'],
                author: 'Community',
                rating: 4.7,
                usageCount: 567,
                thumbnail: 'assets/images/templates/api-integration.svg'
            }
        ];
    }
    
    setupEventListeners() {
        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Navigation shortcuts
            if (e.altKey && e.key >= '1' && e.key <= '9') {
                e.preventDefault();
                const sectionIndex = parseInt(e.key) - 1;
                const sections = ['overview', 'quick-start', 'api-explorer', 'template-gallery', 'workflow-builder'];
                if (sections[sectionIndex]) {
                    this.showSection(sections[sectionIndex]);
                }
            }
        });
        
        // Smooth scrolling for anchor links
        document.addEventListener('click', (e) => {
            if (e.target.matches('a[href^="#"]')) {
                e.preventDefault();
                const targetId = e.target.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
        
        // Copy code button functionality
        this.setupCodeCopyButtons();
    }
    
    setupCodeCopyButtons() {
        const codeBlocks = document.querySelectorAll('pre code');
        
        codeBlocks.forEach(codeBlock => {
            const pre = codeBlock.parentElement;
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-code-btn';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            copyButton.setAttribute('aria-label', 'Copy code');
            
            pre.style.position = 'relative';
            pre.appendChild(copyButton);
            
            copyButton.addEventListener('click', async () => {
                try {
                    await navigator.clipboard.writeText(codeBlock.textContent);
                    copyButton.innerHTML = '<i class="fas fa-check"></i>';
                    copyButton.classList.add('copied');
                    
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                        copyButton.classList.remove('copied');
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy code:', err);
                }
            });
        });
    }
    
    showSection(sectionId) {
        // Hide all sections
        const sections = document.querySelectorAll('.content-section');
        sections.forEach(section => {
            section.classList.remove('active');
        });
        
        // Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionId;
            
            // Update URL without page reload
            const newUrl = `${window.location.pathname}#${sectionId}`;
            history.pushState({ section: sectionId }, '', newUrl);
            
            // Update navigation
            this.updateNavigation(sectionId);
            
            // Initialize section-specific functionality
            this.initializeSection(sectionId);
        }
    }
    
    updateNavigation(sectionId) {
        // Remove active class from all nav links
        const navLinks = document.querySelectorAll('[data-section]');
        navLinks.forEach(link => {
            link.classList.remove('active');
        });
        
        // Add active class to current section link
        const activeLink = document.querySelector(`[data-section="${sectionId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }
    
    initializeSection(sectionId) {
        switch (sectionId) {
            case 'api-explorer':
                this.initializeAPIExplorer();
                break;
            case 'template-gallery':
                this.initializeTemplateGallery();
                break;
            case 'workflow-builder':
                this.initializeWorkflowBuilder();
                break;
        }
    }
    
    initializeAPIExplorer() {
        const apiLinks = document.querySelectorAll('.api-link');
        const apiDetails = document.getElementById('apiDetails');
        
        apiLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const endpointId = link.getAttribute('data-endpoint');
                this.showAPIEndpoint(endpointId);
            });
        });
    }
    
    showAPIEndpoint(endpointId) {
        const endpoint = this.apiEndpoints.get(endpointId);
        if (!endpoint) return;
        
        const apiDetails = document.getElementById('apiDetails');
        if (!apiDetails) return;
        
        apiDetails.innerHTML = `
            <div class="endpoint-details">
                <div class="endpoint-header">
                    <h3>${endpoint.method} ${endpoint.path}</h3>
                    <span class="method-badge ${endpoint.method.toLowerCase()}">${endpoint.method}</span>
                </div>
                
                <p class="endpoint-summary">${endpoint.summary}</p>
                <p class="endpoint-description">${endpoint.description}</p>
                
                ${endpoint.parameters ? this.renderParameters(endpoint.parameters) : ''}
                ${endpoint.requestBody ? this.renderRequestBody(endpoint.requestBody) : ''}
                ${endpoint.responses ? this.renderResponses(endpoint.responses) : ''}
                
                <div class="try-it-section">
                    <h4>Try it out</h4>
                    <button class="btn btn-primary try-endpoint-btn" data-endpoint="${endpointId}">
                        <i class="fas fa-play"></i> Send Request
                    </button>
                </div>
            </div>
        `;
        
        // Setup try it functionality
        const tryButton = apiDetails.querySelector('.try-endpoint-btn');
        if (tryButton) {
            tryButton.addEventListener('click', () => {
                this.tryAPIEndpoint(endpointId);
            });
        }
    }
    
    renderParameters(parameters) {
        return `
            <div class="parameters-section">
                <h4>Parameters</h4>
                <div class="parameters-list">
                    ${parameters.map(param => `
                        <div class="parameter-item">
                            <strong>${param.name}</strong>
                            <span class="param-type">${param.type}</span>
                            <p>${param.description}</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    renderRequestBody(requestBody) {
        return `
            <div class="request-body-section">
                <h4>Request Body</h4>
                <pre><code class="language-json">${JSON.stringify(requestBody, null, 2)}</code></pre>
            </div>
        `;
    }
    
    renderResponses(responses) {
        return `
            <div class="responses-section">
                <h4>Responses</h4>
                ${Object.entries(responses).map(([code, response]) => `
                    <div class="response-item">
                        <div class="response-code">${code}</div>
                        <div class="response-description">${response.description}</div>
                        ${response.example ? `<pre><code class="language-json">${JSON.stringify(response.example, null, 2)}</code></pre>` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    tryAPIEndpoint(endpointId) {
        const endpoint = this.apiEndpoints.find(ep => ep.id === endpointId);
        if (!endpoint) return;
        
        // Create API testing modal
        const modal = document.createElement('div');
        modal.className = 'api-test-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Test API Endpoint: ${endpoint.name}</h3>
                    <button class="close-btn" onclick="this.closest('.api-test-modal').remove()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="endpoint-info">
                        <div class="method ${endpoint.method.toLowerCase()}">${endpoint.method}</div>
                        <div class="url">${endpoint.url}</div>
                        <div class="description">${endpoint.description}</div>
                    </div>
                    
                    <div class="request-section">
                        <h4>Request Parameters</h4>
                        <textarea id="requestBody" placeholder="Enter JSON request body...">${JSON.stringify(endpoint.example_request || {}, null, 2)}</textarea>
                    </div>
                    
                    <div class="headers-section">
                        <h4>Headers</h4>
                        <textarea id="requestHeaders" placeholder="Enter headers as JSON...">{
  "Content-Type": "application/json",
  "Authorization": "Bearer your-token-here"
}</textarea>
                    </div>
                    
                    <div class="actions">
                        <button class="btn btn-primary" onclick="this.sendTestRequest('${endpointId}')" data-endpoint="${endpointId}">Send Request</button>
                        <button class="btn btn-secondary" onclick="this.generateCurlCommand('${endpointId}')">Generate cURL</button>
                    </div>
                    
                    <div class="response-section">
                        <h4>Response</h4>
                        <pre id="responseOutput">Click "Send Request" to test the endpoint</pre>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Add event listeners
        modal.querySelector('[data-endpoint]').addEventListener('click', (e) => {
            this.sendTestRequest(e.target.dataset.endpoint);
        });
    }
    
    initializeTemplateGallery() {
        this.renderTemplates();
        this.setupTemplateFilters();
    }
    
    renderTemplates(filteredTemplates = this.templates) {
        const templateGrid = document.getElementById('templateGrid');
        if (!templateGrid) return;
        
        templateGrid.innerHTML = filteredTemplates.map(template => `
            <div class="template-card" data-template="${template.id}">
                <div class="template-thumbnail">
                    <img src="${template.thumbnail}" alt="${template.name}" onerror="this.src='assets/images/template-placeholder.svg'">
                </div>
                
                <div class="template-content">
                    <h3>${template.name}</h3>
                    <p>${template.description}</p>
                    
                    <div class="template-meta">
                        <span class="template-category">${template.category.replace('_', ' ')}</span>
                        <span class="template-complexity complexity-${template.complexity}">${template.complexity}</span>
                    </div>
                    
                    <div class="template-stats">
                        <div class="template-rating">
                            <i class="fas fa-star"></i>
                            <span>${template.rating}</span>
                        </div>
                        <div class="template-usage">
                            <i class="fas fa-download"></i>
                            <span>${template.usageCount}</span>
                        </div>
                    </div>
                    
                    <div class="template-actions">
                        <button class="btn btn-primary use-template-btn" data-template="${template.id}">
                            <i class="fas fa-plus"></i> Use Template
                        </button>
                        <button class="btn btn-outline preview-template-btn" data-template="${template.id}">
                            <i class="fas fa-eye"></i> Preview
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
        
        // Setup template action buttons
        templateGrid.querySelectorAll('.use-template-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const templateId = e.target.getAttribute('data-template');
                this.useTemplate(templateId);
            });
        });
        
        templateGrid.querySelectorAll('.preview-template-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const templateId = e.target.getAttribute('data-template');
                this.previewTemplate(templateId);
            });
        });
    }
    
    setupTemplateFilters() {
        const categoryFilter = document.getElementById('categoryFilter');
        const complexityFilter = document.getElementById('complexityFilter');
        const templateSearch = document.getElementById('templateSearch');
        
        const applyFilters = () => {
            let filtered = this.templates;
            
            // Category filter
            if (categoryFilter && categoryFilter.value) {
                filtered = filtered.filter(t => t.category === categoryFilter.value);
            }
            
            // Complexity filter
            if (complexityFilter && complexityFilter.value) {
                filtered = filtered.filter(t => t.complexity === complexityFilter.value);
            }
            
            // Search filter
            if (templateSearch && templateSearch.value) {
                const query = templateSearch.value.toLowerCase();
                filtered = filtered.filter(t => 
                    t.name.toLowerCase().includes(query) ||
                    t.description.toLowerCase().includes(query) ||
                    t.tags.some(tag => tag.toLowerCase().includes(query))
                );
            }
            
            this.renderTemplates(filtered);
        };
        
        [categoryFilter, complexityFilter, templateSearch].forEach(element => {
            if (element) {
                element.addEventListener('change', applyFilters);
                element.addEventListener('input', applyFilters);
            }
        });
    }
    
    useTemplate(templateId) {
        const template = this.templates.find(t => t.id === templateId);
        if (!template) return;
        
        // Create template usage modal
        const modal = document.createElement('div');
        modal.className = 'template-usage-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Use Template: ${template.name}</h3>
                    <button class="close-btn" onclick="this.closest('.template-usage-modal').remove()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="template-summary">
                        <img src="${template.thumbnail}" alt="${template.name}" class="template-image">
                        <div class="template-details">
                            <p><strong>Category:</strong> ${template.category.replace('_', ' ')}</p>
                            <p><strong>Complexity:</strong> ${template.complexity}</p>
                            <p><strong>Estimated Time:</strong> ${template.estimated_time}</p>
                            <p><strong>Description:</strong> ${template.description}</p>
                        </div>
                    </div>
                    
                    <div class="template-options">
                        <h4>Template Configuration</h4>
                        <div class="form-group">
                            <label for="workflowName">Workflow Name:</label>
                            <input type="text" id="workflowName" value="${template.name} - Copy" />
                        </div>
                        
                        <div class="form-group">
                            <label for="workflowDescription">Description:</label>
                            <textarea id="workflowDescription">${template.description}</textarea>
                        </div>
                        
                        <div class="form-group">
                            <label for="templateMode">Usage Mode:</label>
                            <select id="templateMode">
                                <option value="copy">Create a copy</option>
                                <option value="import">Import as-is</option>
                                <option value="customize">Customize before import</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="actions">
                        <button class="btn btn-primary" onclick="this.createFromTemplate('${templateId}')">Create Workflow</button>
                        <button class="btn btn-secondary" onclick="this.previewTemplate('${templateId}')">Preview First</button>
                        <button class="btn btn-outline" onclick="this.closest('.template-usage-modal').remove()">Cancel</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    previewTemplate(templateId) {
        const template = this.templates.find(t => t.id === templateId);
        if (!template) return;
        
        // Create template preview modal
        const modal = document.createElement('div');
        modal.className = 'template-preview-modal';
        modal.innerHTML = `
            <div class="modal-content large">
                <div class="modal-header">
                    <h3>Template Preview: ${template.name}</h3>
                    <button class="close-btn" onclick="this.closest('.template-preview-modal').remove()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="preview-tabs">
                        <button class="tab-btn active" data-tab="overview">Overview</button>
                        <button class="tab-btn" data-tab="workflow">Workflow Diagram</button>
                        <button class="tab-btn" data-tab="code">Configuration</button>
                        <button class="tab-btn" data-tab="requirements">Requirements</button>
                    </div>
                    
                    <div class="tab-content">
                        <div class="tab-pane active" id="overview">
                            <div class="template-overview">
                                <img src="${template.thumbnail}" alt="${template.name}" class="preview-image">
                                <div class="overview-details">
                                    <h4>${template.name}</h4>
                                    <p class="category">Category: ${template.category.replace('_', ' ')}</p>
                                    <p class="complexity">Complexity: <span class="complexity-${template.complexity}">${template.complexity}</span></p>
                                    <p class="description">${template.description}</p>
                                    
                                    <div class="features">
                                        <h5>Key Features:</h5>
                                        <ul>
                                            ${template.features ? template.features.map(f => `<li>${f}</li>`).join('') : '<li>Advanced workflow automation</li><li>Error handling and retries</li><li>Real-time monitoring</li>'}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="tab-pane" id="workflow">
                            <div class="workflow-diagram">
                                <svg width="100%" height="400" class="workflow-preview">
                                    <!-- Simplified workflow visualization -->
                                    <g transform="translate(50, 50)">
                                        <rect x="0" y="0" width="120" height="40" rx="5" fill="#4CAF50" />
                                        <text x="60" y="25" text-anchor="middle" fill="white">Start</text>
                                        
                                        <line x1="120" y1="20" x2="180" y2="20" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)" />
                                        
                                        <rect x="180" y="0" width="120" height="40" rx="5" fill="#2196F3" />
                                        <text x="240" y="25" text-anchor="middle" fill="white">Process</text>
                                        
                                        <line x1="300" y1="20" x2="360" y2="20" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)" />
                                        
                                        <rect x="360" y="0" width="120" height="40" rx="5" fill="#FF9800" />
                                        <text x="420" y="25" text-anchor="middle" fill="white">Complete</text>
                                    </g>
                                    
                                    <defs>
                                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
                                        </marker>
                                    </defs>
                                </svg>
                                <p class="diagram-note">This is a simplified view. The actual workflow contains ${template.node_count || '10+'} nodes with complex logic.</p>
                            </div>
                        </div>
                        
                        <div class="tab-pane" id="code">
                            <pre class="code-preview"><code>${JSON.stringify(template.configuration || {name: template.name, description: template.description, steps: []}, null, 2)}</code></pre>
                        </div>
                        
                        <div class="tab-pane" id="requirements">
                            <div class="requirements">
                                <h5>System Requirements:</h5>
                                <ul>
                                    <li>Python 3.8+ runtime</li>
                                    <li>PostgreSQL database</li>
                                    <li>Redis for caching</li>
                                    ${template.requirements ? template.requirements.map(r => `<li>${r}</li>`).join('') : ''}
                                </ul>
                                
                                <h5>Estimated Resources:</h5>
                                <ul>
                                    <li>CPU: ${template.cpu_requirement || '2 cores'}</li>
                                    <li>Memory: ${template.memory_requirement || '4GB RAM'}</li>
                                    <li>Storage: ${template.storage_requirement || '10GB'}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <div class="actions">
                        <button class="btn btn-primary" onclick="this.useTemplate('${templateId}')">Use This Template</button>
                        <button class="btn btn-secondary" onclick="this.downloadTemplate('${templateId}')">Download</button>
                        <button class="btn btn-outline" onclick="this.closest('.template-preview-modal').remove()">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Setup tab switching
        modal.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                
                // Update active tab button
                modal.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                // Update active tab content
                modal.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
                modal.querySelector(`#${tabName}`).classList.add('active');
            });
        });
    }
    
    initializeWorkflowBuilder() {
        const builderContainer = document.getElementById('workflowBuilder');
        if (!builderContainer) return;
        
        builderContainer.innerHTML = `
            <div class="workflow-builder">
                <div class="builder-toolbar">
                    <div class="toolbar-section">
                        <button class="btn btn-primary" onclick="this.createNewWorkflow()">New Workflow</button>
                        <button class="btn btn-secondary" onclick="this.saveWorkflow()">Save</button>
                        <button class="btn btn-outline" onclick="this.loadWorkflow()">Load</button>
                    </div>
                    
                    <div class="toolbar-section">
                        <button class="btn btn-sm" onclick="this.validateWorkflow()" title="Validate">
                            <i class="fas fa-check"></i>
                        </button>
                        <button class="btn btn-sm" onclick="this.runWorkflow()" title="Test Run">
                            <i class="fas fa-play"></i>
                        </button>
                        <button class="btn btn-sm" onclick="this.exportWorkflow()" title="Export">
                            <i class="fas fa-download"></i>
                        </button>
                    </div>
                </div>
                
                <div class="builder-content">
                    <div class="component-palette">
                        <h4>Components</h4>
                        
                        <div class="component-category">
                            <h5>Triggers</h5>
                            <div class="component-item draggable" data-type="http-trigger">
                                <i class="fas fa-globe"></i> HTTP Request
                            </div>
                            <div class="component-item draggable" data-type="schedule-trigger">
                                <i class="fas fa-clock"></i> Schedule
                            </div>
                            <div class="component-item draggable" data-type="file-trigger">
                                <i class="fas fa-file"></i> File Watcher
                            </div>
                        </div>
                        
                        <div class="component-category">
                            <h5>Actions</h5>
                            <div class="component-item draggable" data-type="api-call">
                                <i class="fas fa-exchange-alt"></i> API Call
                            </div>
                            <div class="component-item draggable" data-type="data-transform">
                                <i class="fas fa-cogs"></i> Transform Data
                            </div>
                            <div class="component-item draggable" data-type="email-send">
                                <i class="fas fa-envelope"></i> Send Email
                            </div>
                            <div class="component-item draggable" data-type="database-query">
                                <i class="fas fa-database"></i> Database Query
                            </div>
                        </div>
                        
                        <div class="component-category">
                            <h5>Logic</h5>
                            <div class="component-item draggable" data-type="condition">
                                <i class="fas fa-code-branch"></i> Condition
                            </div>
                            <div class="component-item draggable" data-type="loop">
                                <i class="fas fa-redo"></i> Loop
                            </div>
                            <div class="component-item draggable" data-type="parallel">
                                <i class="fas fa-share-alt"></i> Parallel
                            </div>
                        </div>
                    </div>
                    
                    <div class="workflow-canvas">
                        <div class="canvas-grid" id="workflowCanvas">
                            <div class="canvas-placeholder">
                                <i class="fas fa-plus-circle"></i>
                                <p>Drag components here to build your workflow</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="properties-panel">
                        <h4>Properties</h4>
                        <div class="property-content">
                            <p>Select a component to edit its properties</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Initialize drag and drop functionality
        this.setupDragAndDrop();
        
        console.log('Workflow builder initialized with drag-and-drop interface');
    }
    
    // Helper methods for the implemented functionality
    highlightSearchTerms(text, searchTerm) {
        if (!searchTerm) return text;
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
    
    highlightSearchResult(element) {
        if (element) {
            element.classList.add('search-highlight');
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
    
    sendTestRequest(endpointId) {
        const modal = document.querySelector('.api-test-modal');
        const requestBody = modal.querySelector('#requestBody').value;
        const requestHeaders = modal.querySelector('#requestHeaders').value;
        const responseOutput = modal.querySelector('#responseOutput');
        
        try {
            const headers = JSON.parse(requestHeaders);
            const body = requestBody ? JSON.parse(requestBody) : {};
            
            responseOutput.textContent = 'Sending request...';
            
            // Simulate API call with mock response
            setTimeout(() => {
                const mockResponse = {
                    status: 200,
                    headers: { 'content-type': 'application/json' },
                    data: {
                        success: true,
                        message: 'Mock API response',
                        timestamp: new Date().toISOString(),
                        request_id: Math.random().toString(36).substr(2, 9)
                    }
                };
                
                responseOutput.textContent = JSON.stringify(mockResponse, null, 2);
            }, 1000);
            
        } catch (error) {
            responseOutput.textContent = `Error: ${error.message}`;
        }
    }
    
    generateCurlCommand(endpointId) {
        const endpoint = this.apiEndpoints.find(ep => ep.id === endpointId);
        const modal = document.querySelector('.api-test-modal');
        const requestBody = modal.querySelector('#requestBody').value;
        const requestHeaders = modal.querySelector('#requestHeaders').value;
        
        let curlCommand = `curl -X ${endpoint.method} \\\n  "${endpoint.url}"`;
        
        try {
            const headers = JSON.parse(requestHeaders);
            Object.entries(headers).forEach(([key, value]) => {
                curlCommand += ` \\\n  -H "${key}: ${value}"`;
            });
            
            if (requestBody) {
                curlCommand += ` \\\n  -d '${requestBody}'`;
            }
        } catch (error) {
            console.error('Error generating cURL command:', error);
        }
        
        // Copy to clipboard
        navigator.clipboard.writeText(curlCommand).then(() => {
            alert('cURL command copied to clipboard!');
        });
    }
    
    createFromTemplate(templateId) {
        const modal = document.querySelector('.template-usage-modal');
        const workflowName = modal.querySelector('#workflowName').value;
        const workflowDescription = modal.querySelector('#workflowDescription').value;
        const templateMode = modal.querySelector('#templateMode').value;
        
        console.log(`Creating workflow from template ${templateId}:`, {
            name: workflowName,
            description: workflowDescription,
            mode: templateMode
        });
        
        // Simulate workflow creation
        alert(`Workflow "${workflowName}" created successfully! Redirecting to workflow builder...`);
        modal.remove();
        this.showSection('workflow-builder');
    }
    
    downloadTemplate(templateId) {
        const template = this.templates.find(t => t.id === templateId);
        if (!template) return;
        
        const templateData = {
            name: template.name,
            description: template.description,
            category: template.category,
            configuration: template.configuration || {},
            metadata: {
                version: '1.0.0',
                created_at: new Date().toISOString(),
                complexity: template.complexity
            }
        };
        
        const blob = new Blob([JSON.stringify(templateData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${template.name.replace(/\s+/g, '_').toLowerCase()}_template.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    setupDragAndDrop() {
        const draggables = document.querySelectorAll('.draggable');
        const canvas = document.getElementById('workflowCanvas');
        
        if (!canvas) return;
        
        draggables.forEach(draggable => {
            draggable.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', draggable.dataset.type);
            });
        });
        
        canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            canvas.classList.add('drag-over');
        });
        
        canvas.addEventListener('dragleave', (e) => {
            canvas.classList.remove('drag-over');
        });
        
        canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            canvas.classList.remove('drag-over');
            
            const componentType = e.dataTransfer.getData('text/plain');
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            this.addComponentToCanvas(componentType, x, y);
        });
    }
    
    addComponentToCanvas(componentType, x, y) {
        const canvas = document.getElementById('workflowCanvas');
        const placeholder = canvas.querySelector('.canvas-placeholder');
        
        if (placeholder) {
            placeholder.remove();
        }
        
        const componentElement = document.createElement('div');
        componentElement.className = 'workflow-component';
        componentElement.style.position = 'absolute';
        componentElement.style.left = `${x}px`;
        componentElement.style.top = `${y}px`;
        componentElement.dataset.type = componentType;
        
        const componentConfig = this.getComponentConfig(componentType);
        componentElement.innerHTML = `
            <div class="component-header">
                <i class="${componentConfig.icon}"></i>
                <span>${componentConfig.name}</span>
            </div>
            <div class="component-ports">
                <div class="input-port"></div>
                <div class="output-port"></div>
            </div>
        `;
        
        componentElement.addEventListener('click', () => {
            this.selectComponent(componentElement);
        });
        
        canvas.appendChild(componentElement);
    }
    
    getComponentConfig(type) {
        const configs = {
            'http-trigger': { name: 'HTTP Request', icon: 'fas fa-globe' },
            'schedule-trigger': { name: 'Schedule', icon: 'fas fa-clock' },
            'file-trigger': { name: 'File Watcher', icon: 'fas fa-file' },
            'api-call': { name: 'API Call', icon: 'fas fa-exchange-alt' },
            'data-transform': { name: 'Transform Data', icon: 'fas fa-cogs' },
            'email-send': { name: 'Send Email', icon: 'fas fa-envelope' },
            'database-query': { name: 'Database Query', icon: 'fas fa-database' },
            'condition': { name: 'Condition', icon: 'fas fa-code-branch' },
            'loop': { name: 'Loop', icon: 'fas fa-redo' },
            'parallel': { name: 'Parallel', icon: 'fas fa-share-alt' }
        };
        
        return configs[type] || { name: 'Unknown', icon: 'fas fa-question' };
    }
    
    selectComponent(componentElement) {
        // Remove previous selection
        document.querySelectorAll('.workflow-component.selected').forEach(comp => {
            comp.classList.remove('selected');
        });
        
        // Select current component
        componentElement.classList.add('selected');
        
        // Update properties panel
        const propertiesPanel = document.querySelector('.properties-panel .property-content');
        const componentType = componentElement.dataset.type;
        const componentConfig = this.getComponentConfig(componentType);
        
        propertiesPanel.innerHTML = `
            <h5>${componentConfig.name} Properties</h5>
            <div class="form-group">
                <label>Component Name:</label>
                <input type="text" value="${componentConfig.name}" />
            </div>
            <div class="form-group">
                <label>Description:</label>
                <textarea placeholder="Enter component description..."></textarea>
            </div>
            <div class="form-group">
                <label>Configuration:</label>
                <textarea placeholder="Enter component configuration as JSON..."></textarea>
            </div>
            <div class="form-actions">
                <button class="btn btn-sm btn-primary">Save Changes</button>
                <button class="btn btn-sm btn-danger" onclick="this.deleteComponent()">Delete</button>
            </div>
        `;
    }
    
    handleUrlRoute() {
        const hash = window.location.hash.substring(1);
        if (hash && document.getElementById(hash)) {
            this.showSection(hash);
        } else {
            this.showSection('overview');
        }
    }
}

// Global functions for external access
window.showSection = function(sectionId) {
    if (window.interactiveDocs) {
        window.interactiveDocs.showSection(sectionId);
    }
};

window.openWorkflowBuilder = function() {
    if (window.interactiveDocs) {
        window.interactiveDocs.showSection('workflow-builder');
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.interactiveDocs = new InteractiveDocumentation();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        // Refresh dynamic content when page becomes visible
        console.log('Page became visible, refreshing content');
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InteractiveDocumentation;
}