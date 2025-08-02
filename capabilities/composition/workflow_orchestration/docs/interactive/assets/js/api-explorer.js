/**
 * APG Workflow Orchestration - API Explorer
 * Interactive API testing and documentation
 */

class APIExplorer {
    constructor() {
        this.baseUrl = 'http://localhost:5000/api/v1/workflow_orchestration';
        this.authToken = null;
        this.requestHistory = [];
        this.environments = {
            local: 'http://localhost:5000/api/v1/workflow_orchestration',
            staging: 'https://staging-api.example.com/api/v1/workflow_orchestration',
            production: 'https://api.example.com/api/v1/workflow_orchestration'
        };
        
        this.init();
    }
    
    init() {
        this.setupEnvironmentSelector();
        this.setupAuthentication();
        this.loadEndpointDefinitions();
        this.setupRequestTester();
        this.loadRequestHistory();
    }
    
    setupEnvironmentSelector() {
        const envSelector = document.createElement('div');
        envSelector.className = 'environment-selector';
        envSelector.innerHTML = `
            <label for="envSelect">Environment:</label>
            <select id="envSelect" class="env-select">
                <option value="local">Local Development</option>
                <option value="staging">Staging</option>
                <option value="production">Production</option>
            </select>
        `;
        
        const apiExplorer = document.querySelector('.api-explorer');
        if (apiExplorer) {
            apiExplorer.insertBefore(envSelector, apiExplorer.firstChild);
        }
        
        document.getElementById('envSelect')?.addEventListener('change', (e) => {
            this.baseUrl = this.environments[e.target.value];
            this.updateEndpointUrls();
        });
    }
    
    setupAuthentication() {
        const authSection = document.createElement('div');
        authSection.className = 'auth-section';
        authSection.innerHTML = `
            <div class="auth-header">
                <h4>Authentication</h4>
                <div class="auth-status" id="authStatus">
                    <i class="fas fa-times-circle text-error"></i>
                    <span>Not authenticated</span>
                </div>
            </div>
            
            <div class="auth-methods">
                <div class="auth-method">
                    <label for="authMethod">Method:</label>
                    <select id="authMethod" class="auth-method-select">
                        <option value="none">None</option>
                        <option value="bearer">Bearer Token</option>
                        <option value="api_key">API Key</option>
                        <option value="basic">Basic Auth</option>
                    </select>
                </div>
                
                <div class="auth-inputs" id="authInputs">
                    <!-- Auth inputs will be populated based on selected method -->
                </div>
                
                <div class="auth-actions">
                    <button class="btn btn-primary" id="testAuthBtn">Test Authentication</button>
                    <button class="btn btn-outline" id="clearAuthBtn">Clear</button>
                </div>
            </div>
        `;
        
        const apiSidebar = document.querySelector('.api-sidebar');
        if (apiSidebar) {
            apiSidebar.insertBefore(authSection, apiSidebar.firstChild);
        }
        
        this.setupAuthMethodHandler();
    }
    
    setupAuthMethodHandler() {
        const authMethod = document.getElementById('authMethod');
        const authInputs = document.getElementById('authInputs');
        const testAuthBtn = document.getElementById('testAuthBtn');
        const clearAuthBtn = document.getElementById('clearAuthBtn');
        
        authMethod?.addEventListener('change', (e) => {
            this.renderAuthInputs(e.target.value);
        });
        
        testAuthBtn?.addEventListener('click', () => {
            this.testAuthentication();
        });
        
        clearAuthBtn?.addEventListener('click', () => {
            this.clearAuthentication();
        });
    }
    
    renderAuthInputs(method) {
        const authInputs = document.getElementById('authInputs');
        if (!authInputs) return;
        
        let inputsHtml = '';
        
        switch (method) {
            case 'bearer':
                inputsHtml = `
                    <div class="input-group">
                        <label for="bearerToken">Bearer Token:</label>
                        <input type="password" id="bearerToken" placeholder="Enter your bearer token" class="auth-input">
                    </div>
                `;
                break;
                
            case 'api_key':
                inputsHtml = `
                    <div class="input-group">
                        <label for="apiKey">API Key:</label>
                        <input type="password" id="apiKey" placeholder="Enter your API key" class="auth-input">
                    </div>
                    <div class="input-group">
                        <label for="apiKeyHeader">Header Name:</label>
                        <input type="text" id="apiKeyHeader" value="X-API-Key" class="auth-input">
                    </div>
                `;
                break;
                
            case 'basic':
                inputsHtml = `
                    <div class="input-group">
                        <label for="username">Username:</label>
                        <input type="text" id="username" placeholder="Enter username" class="auth-input">
                    </div>
                    <div class="input-group">
                        <label for="password">Password:</label>
                        <input type="password" id="password" placeholder="Enter password" class="auth-input">
                    </div>
                `;
                break;
        }
        
        authInputs.innerHTML = inputsHtml;
    }
    
    async testAuthentication() {
        const method = document.getElementById('authMethod')?.value;
        const authStatus = document.getElementById('authStatus');
        
        if (method === 'none') {
            this.authToken = null;
            this.updateAuthStatus(false, 'No authentication method selected');
            return;
        }
        
        try {
            const authHeaders = this.getAuthHeaders();
            
            // Test auth with a simple endpoint
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                headers: authHeaders
            });
            
            if (response.ok) {
                this.updateAuthStatus(true, 'Authentication successful');
                this.saveAuthConfig();
            } else {
                this.updateAuthStatus(false, `Authentication failed: ${response.status}`);
            }
        } catch (error) {
            this.updateAuthStatus(false, `Authentication error: ${error.message}`);
        }
    }
    
    getAuthHeaders() {
        const method = document.getElementById('authMethod')?.value;
        const headers = { 'Content-Type': 'application/json' };
        
        switch (method) {
            case 'bearer':
                const bearerToken = document.getElementById('bearerToken')?.value;
                if (bearerToken) {
                    headers['Authorization'] = `Bearer ${bearerToken}`;
                }
                break;
                
            case 'api_key':
                const apiKey = document.getElementById('apiKey')?.value;
                const headerName = document.getElementById('apiKeyHeader')?.value || 'X-API-Key';
                if (apiKey) {
                    headers[headerName] = apiKey;
                }
                break;
                
            case 'basic':
                const username = document.getElementById('username')?.value;
                const password = document.getElementById('password')?.value;
                if (username && password) {
                    const credentials = btoa(`${username}:${password}`);
                    headers['Authorization'] = `Basic ${credentials}`;
                }
                break;
        }
        
        return headers;
    }
    
    updateAuthStatus(isAuthenticated, message) {
        const authStatus = document.getElementById('authStatus');
        if (!authStatus) return;
        
        const icon = isAuthenticated ? 
            '<i class="fas fa-check-circle text-success"></i>' : 
            '<i class="fas fa-times-circle text-error"></i>';
        
        authStatus.innerHTML = `${icon} <span>${message}</span>`;
    }
    
    clearAuthentication() {
        this.authToken = null;
        document.getElementById('authMethod').value = 'none';
        document.getElementById('authInputs').innerHTML = '';
        this.updateAuthStatus(false, 'Authentication cleared');
        localStorage.removeItem('api-explorer-auth');
    }
    
    saveAuthConfig() {
        const method = document.getElementById('authMethod')?.value;
        if (method === 'none') return;
        
        const config = { method };
        
        // Don't save sensitive data, just the method
        localStorage.setItem('api-explorer-auth', JSON.stringify(config));
    }
    
    loadEndpointDefinitions() {
        // Extended endpoint definitions with more details
        this.endpoints = {
            // Workflows
            'get-workflows': {
                method: 'GET',
                path: '/workflows',
                summary: 'List workflows',
                description: 'Retrieve a list of workflows with optional filtering, sorting, and pagination.',
                parameters: [
                    { name: 'limit', type: 'integer', in: 'query', description: 'Maximum number of results (1-1000)', default: 50 },
                    { name: 'offset', type: 'integer', in: 'query', description: 'Number of results to skip', default: 0 },
                    { name: 'category', type: 'string', in: 'query', description: 'Filter by workflow category' },
                    { name: 'status', type: 'string', in: 'query', description: 'Filter by workflow status', enum: ['draft', 'active', 'inactive', 'archived'] },
                    { name: 'search', type: 'string', in: 'query', description: 'Search in workflow names and descriptions' },
                    { name: 'sort', type: 'string', in: 'query', description: 'Sort field', enum: ['name', 'created_at', 'updated_at'] },
                    { name: 'order', type: 'string', in: 'query', description: 'Sort order', enum: ['asc', 'desc'] }
                ],
                responses: {
                    200: {
                        description: 'Success',
                        example: {
                            success: true,
                            data: [
                                {
                                    id: '01HQNZ2G5XKJ8P7M9N6V3R4T5W',
                                    name: 'Data Processing Pipeline',
                                    description: 'ETL pipeline for customer data',
                                    status: 'active',
                                    category: 'data_processing',
                                    created_at: '2024-12-01T10:00:00Z'
                                }
                            ],
                            metadata: {
                                total_count: 1,
                                limit: 50,
                                offset: 0
                            }
                        }
                    },
                    400: { description: 'Bad Request - Invalid parameters' },
                    401: { description: 'Unauthorized - Authentication required' },
                    403: { description: 'Forbidden - Insufficient permissions' }
                }
            },
            
            'post-workflows': {
                method: 'POST',
                path: '/workflows',
                summary: 'Create workflow',
                description: 'Create a new workflow with the provided definition and configuration.',
                requestBody: {
                    type: 'object',
                    required: ['name', 'definition'],
                    properties: {
                        name: { type: 'string', description: 'Workflow name (1-255 characters)' },
                        description: { type: 'string', description: 'Workflow description (optional)' },
                        definition: {
                            type: 'object',
                            description: 'Workflow definition with components and connections',
                            properties: {
                                components: { type: 'array', description: 'Array of workflow components' },
                                connections: { type: 'array', description: 'Array of component connections' },
                                parameters: { type: 'array', description: 'Workflow parameters' }
                            }
                        },
                        category: { type: 'string', description: 'Workflow category' },
                        tags: { type: 'array', items: { type: 'string' }, description: 'Workflow tags' },
                        priority: { type: 'integer', minimum: 1, maximum: 10, description: 'Execution priority' }
                    },
                    example: {
                        name: 'My New Workflow',
                        description: 'A sample workflow for testing',
                        definition: {
                            components: [
                                { id: 'start', type: 'start', config: { trigger_type: 'manual' } },
                                { id: 'end', type: 'end', config: { status: 'success' } }
                            ],
                            connections: [
                                { source: 'start', target: 'end', type: 'success' }
                            ]
                        },
                        category: 'test',
                        tags: ['sample', 'test']
                    }
                },
                responses: {
                    201: {
                        description: 'Created successfully',
                        example: {
                            success: true,
                            data: {
                                id: '01HQNZ2G5XKJ8P7M9N6V3R4T5W',
                                name: 'My New Workflow',
                                status: 'draft',
                                created_at: '2025-01-01T12:00:00Z'
                            },
                            message: 'Workflow created successfully'
                        }
                    },
                    400: { description: 'Bad Request - Invalid workflow definition' },
                    401: { description: 'Unauthorized' },
                    409: { description: 'Conflict - Workflow name already exists' }
                }
            },
            
            'execute-workflow': {
                method: 'POST',
                path: '/workflows/{id}/execute',
                summary: 'Execute workflow',
                description: 'Start execution of a workflow with optional parameters.',
                parameters: [
                    { name: 'id', type: 'string', in: 'path', required: true, description: 'Workflow ID' }
                ],
                requestBody: {
                    type: 'object',
                    properties: {
                        parameters: { type: 'object', description: 'Workflow execution parameters' },
                        priority: { type: 'integer', minimum: 1, maximum: 10, description: 'Execution priority' },
                        scheduled_at: { type: 'string', format: 'datetime', description: 'Schedule execution for future time' },
                        tags: { type: 'array', items: { type: 'string' }, description: 'Execution tags' }
                    },
                    example: {
                        parameters: {
                            input_data: { message: 'Hello World' },
                            timeout: 300
                        },
                        priority: 5
                    }
                },
                responses: {
                    200: {
                        description: 'Execution started',
                        example: {
                            success: true,
                            data: {
                                execution_id: '01HQNZ2G5XKJ8P7M9N6V3R4T5Y',
                                status: 'queued',
                                workflow_id: '01HQNZ2G5XKJ8P7M9N6V3R4T5W',
                                estimated_duration: 300,
                                queue_position: 2
                            },
                            message: 'Workflow execution started'
                        }
                    },
                    400: { description: 'Bad Request - Invalid parameters' },
                    404: { description: 'Workflow not found' },
                    409: { description: 'Workflow cannot be executed in current state' }
                }
            }
        };
    }
    
    setupRequestTester() {
        this.setupEndpointSelection();
        this.setupRequestForm();
        this.setupResponseViewer();
    }
    
    setupEndpointSelection() {
        const endpointLinks = document.querySelectorAll('.api-link');
        
        endpointLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const endpointId = link.getAttribute('data-endpoint');
                this.selectEndpoint(endpointId);
                
                // Update active state
                endpointLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            });
        });
    }
    
    selectEndpoint(endpointId) {
        const endpoint = this.endpoints[endpointId];
        if (!endpoint) return;
        
        const apiDetails = document.getElementById('apiDetails');
        if (!apiDetails) return;
        
        apiDetails.innerHTML = this.renderEndpointDetails(endpoint, endpointId);
        this.setupEndpointTester(endpoint, endpointId);
    }
    
    renderEndpointDetails(endpoint, endpointId) {
        return `
            <div class="endpoint-details">
                <div class="endpoint-header">
                    <div class="endpoint-title">
                        <span class="method-badge ${endpoint.method.toLowerCase()}">${endpoint.method}</span>
                        <code class="endpoint-path">${endpoint.path}</code>
                    </div>
                    <h3>${endpoint.summary}</h3>
                </div>
                
                <div class="endpoint-description">
                    <p>${endpoint.description}</p>
                </div>
                
                ${this.renderEndpointParameters(endpoint.parameters)}
                ${this.renderEndpointRequestBody(endpoint.requestBody)}
                
                <div class="try-it-section">
                    <h4><i class="fas fa-play"></i> Try it out</h4>
                    
                    <div class="request-form" id="requestForm-${endpointId}">
                        <!-- Request form will be populated -->
                    </div>
                    
                    <div class="request-actions">
                        <button class="btn btn-primary execute-btn" data-endpoint="${endpointId}">
                            <i class="fas fa-paper-plane"></i> Send Request
                        </button>
                        <button class="btn btn-outline clear-btn" data-endpoint="${endpointId}">
                            <i class="fas fa-eraser"></i> Clear
                        </button>
                        <button class="btn btn-outline copy-curl-btn" data-endpoint="${endpointId}">
                            <i class="fas fa-copy"></i> Copy as cURL
                        </button>
                    </div>
                    
                    <div class="response-section" id="responseSection-${endpointId}" style="display: none;">
                        <h5>Response</h5>
                        <div class="response-content"></div>
                    </div>
                </div>
                
                ${this.renderEndpointResponses(endpoint.responses)}
            </div>
        `;
    }
    
    renderEndpointParameters(parameters) {
        if (!parameters || parameters.length === 0) return '';
        
        return `
            <div class="parameters-section">
                <h4><i class="fas fa-sliders-h"></i> Parameters</h4>
                <div class="parameters-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Type</th>
                                <th>In</th>
                                <th>Required</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${parameters.map(param => `
                                <tr>
                                    <td><code>${param.name}</code></td>
                                    <td><span class="param-type">${param.type}</span></td>
                                    <td><span class="param-in">${param.in}</span></td>
                                    <td>${param.required ? '<i class="fas fa-check text-success"></i>' : '<i class="fas fa-times text-muted"></i>'}</td>
                                    <td>${param.description}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }
    
    renderEndpointRequestBody(requestBody) {
        if (!requestBody) return '';
        
        return `
            <div class="request-body-section">
                <h4><i class="fas fa-file-code"></i> Request Body</h4>
                <div class="request-body-details">
                    <div class="schema-info">
                        <p><strong>Type:</strong> ${requestBody.type}</p>
                        ${requestBody.required ? `<p><strong>Required fields:</strong> ${requestBody.required.join(', ')}</p>` : ''}
                    </div>
                    
                    ${requestBody.example ? `
                        <div class="request-example">
                            <h5>Example</h5>
                            <pre><code class="language-json">${JSON.stringify(requestBody.example, null, 2)}</code></pre>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    renderEndpointResponses(responses) {
        return `
            <div class="responses-section">
                <h4><i class="fas fa-exchange-alt"></i> Responses</h4>
                <div class="responses-list">
                    ${Object.entries(responses).map(([code, response]) => `
                        <div class="response-item">
                            <div class="response-header">
                                <span class="response-code code-${code}">${code}</span>
                                <span class="response-description">${response.description}</span>
                            </div>
                            
                            ${response.example ? `
                                <div class="response-example">
                                    <h6>Example response</h6>
                                    <pre><code class="language-json">${JSON.stringify(response.example, null, 2)}</code></pre>
                                </div>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    setupEndpointTester(endpoint, endpointId) {
        // Setup request form
        this.setupRequestForm(endpoint, endpointId);
        
        // Setup action buttons
        const executeBtn = document.querySelector(`[data-endpoint="${endpointId}"].execute-btn`);
        const clearBtn = document.querySelector(`[data-endpoint="${endpointId}"].clear-btn`);
        const copyCurlBtn = document.querySelector(`[data-endpoint="${endpointId}"].copy-curl-btn`);
        
        executeBtn?.addEventListener('click', () => this.executeRequest(endpoint, endpointId));
        clearBtn?.addEventListener('click', () => this.clearRequestForm(endpointId));
        copyCurlBtn?.addEventListener('click', () => this.copyCurlCommand(endpoint, endpointId));
    }
    
    setupRequestForm(endpoint, endpointId) {
        const requestForm = document.getElementById(`requestForm-${endpointId}`);
        if (!requestForm) return;
        
        let formHtml = '';
        
        // Path parameters
        const pathParams = endpoint.parameters?.filter(p => p.in === 'path') || [];
        if (pathParams.length > 0) {
            formHtml += `
                <div class="form-section">
                    <h5>Path Parameters</h5>
                    ${pathParams.map(param => `
                        <div class="form-group">
                            <label for="${param.name}-${endpointId}">${param.name} ${param.required ? '*' : ''}</label>
                            <input type="text" id="${param.name}-${endpointId}" class="form-input" 
                                   placeholder="${param.description}" ${param.required ? 'required' : ''}>
                            <small class="form-help">${param.description}</small>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        // Query parameters
        const queryParams = endpoint.parameters?.filter(p => p.in === 'query') || [];
        if (queryParams.length > 0) {
            formHtml += `
                <div class="form-section">
                    <h5>Query Parameters</h5>
                    ${queryParams.map(param => `
                        <div class="form-group">
                            <label for="${param.name}-${endpointId}">${param.name} ${param.required ? '*' : ''}</label>
                            ${param.enum ? `
                                <select id="${param.name}-${endpointId}" class="form-input">
                                    <option value="">Select ${param.name}</option>
                                    ${param.enum.map(option => `<option value="${option}">${option}</option>`).join('')}
                                </select>
                            ` : `
                                <input type="${param.type === 'integer' ? 'number' : 'text'}" 
                                       id="${param.name}-${endpointId}" class="form-input" 
                                       placeholder="${param.default || param.description}" 
                                       ${param.required ? 'required' : ''}>
                            `}
                            <small class="form-help">${param.description}</small>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        // Request body
        if (endpoint.requestBody) {
            formHtml += `
                <div class="form-section">
                    <h5>Request Body</h5>
                    <div class="form-group">
                        <label for="requestBody-${endpointId}">JSON Body</label>
                        <textarea id="requestBody-${endpointId}" class="form-textarea" rows="10" 
                                  placeholder="Enter JSON request body">${JSON.stringify(endpoint.requestBody.example || {}, null, 2)}</textarea>
                        <div class="textarea-actions">
                            <button type="button" class="btn btn-small format-json-btn" data-target="requestBody-${endpointId}">
                                <i class="fas fa-code"></i> Format JSON
                            </button>
                            <button type="button" class="btn btn-small validate-json-btn" data-target="requestBody-${endpointId}">
                                <i class="fas fa-check"></i> Validate
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }
        
        requestForm.innerHTML = formHtml;
        
        // Setup JSON formatting and validation
        this.setupJSONHelpers(endpointId);
    }
    
    setupJSONHelpers(endpointId) {
        const formatBtn = document.querySelector(`[data-target="requestBody-${endpointId}"].format-json-btn`);
        const validateBtn = document.querySelector(`[data-target="requestBody-${endpointId}"].validate-json-btn`);
        
        formatBtn?.addEventListener('click', (e) => {
            const target = e.target.getAttribute('data-target');
            this.formatJSON(target);
        });
        
        validateBtn?.addEventListener('click', (e) => {
            const target = e.target.getAttribute('data-target');
            this.validateJSON(target);
        });
    }
    
    formatJSON(textareaId) {
        const textarea = document.getElementById(textareaId);
        if (!textarea) return;
        
        try {
            const json = JSON.parse(textarea.value);
            textarea.value = JSON.stringify(json, null, 2);
            this.showMessage('JSON formatted successfully', 'success');
        } catch (error) {
            this.showMessage(`Invalid JSON: ${error.message}`, 'error');
        }
    }
    
    validateJSON(textareaId) {
        const textarea = document.getElementById(textareaId);
        if (!textarea) return;
        
        try {
            JSON.parse(textarea.value);
            this.showMessage('JSON is valid', 'success');
        } catch (error) {
            this.showMessage(`Invalid JSON: ${error.message}`, 'error');
        }
    }
    
    async executeRequest(endpoint, endpointId) {
        const executeBtn = document.querySelector(`[data-endpoint="${endpointId}"].execute-btn`);
        const responseSection = document.getElementById(`responseSection-${endpointId}`);
        
        if (!executeBtn || !responseSection) return;
        
        // Show loading state
        executeBtn.disabled = true;
        executeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
        
        try {
            // Build request
            const request = this.buildRequest(endpoint, endpointId);
            
            // Record start time
            const startTime = performance.now();
            
            // Execute request
            const response = await fetch(request.url, {
                method: request.method,
                headers: request.headers,
                body: request.body
            });
            
            // Calculate duration
            const duration = Math.round(performance.now() - startTime);
            
            // Parse response
            const responseData = await this.parseResponse(response);
            
            // Display response
            this.displayResponse(responseSection, {
                status: response.status,
                statusText: response.statusText,
                headers: Object.fromEntries(response.headers.entries()),
                data: responseData,
                duration,
                request: request
            });
            
            // Save to history
            this.saveRequestToHistory(endpoint, request, response, responseData, duration);
            
        } catch (error) {
            this.displayError(responseSection, error);
        } finally {
            // Reset button
            executeBtn.disabled = false;
            executeBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Send Request';
        }
    }
    
    buildRequest(endpoint, endpointId) {
        let url = `${this.baseUrl}${endpoint.path}`;
        const headers = this.getAuthHeaders();
        let body = null;
        const queryParams = new URLSearchParams();
        
        // Process path parameters
        const pathParams = endpoint.parameters?.filter(p => p.in === 'path') || [];
        pathParams.forEach(param => {
            const input = document.getElementById(`${param.name}-${endpointId}`);
            if (input && input.value) {
                url = url.replace(`{${param.name}}`, encodeURIComponent(input.value));
            }
        });
        
        // Process query parameters
        const queryParamsFromEndpoint = endpoint.parameters?.filter(p => p.in === 'query') || [];
        queryParamsFromEndpoint.forEach(param => {
            const input = document.getElementById(`${param.name}-${endpointId}`);
            if (input && input.value) {
                queryParams.append(param.name, input.value);
            }
        });
        
        // Add query parameters to URL
        if (queryParams.toString()) {
            url += `?${queryParams.toString()}`;
        }
        
        // Process request body
        if (endpoint.requestBody) {
            const bodyTextarea = document.getElementById(`requestBody-${endpointId}`);
            if (bodyTextarea && bodyTextarea.value.trim()) {
                try {
                    body = JSON.stringify(JSON.parse(bodyTextarea.value));
                } catch (error) {
                    throw new Error(`Invalid JSON in request body: ${error.message}`);
                }
            }
        }
        
        return {
            url,
            method: endpoint.method,
            headers,
            body
        };
    }
    
    async parseResponse(response) {
        const contentType = response.headers.get('content-type');
        
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        } else if (contentType && contentType.includes('text/')) {
            return await response.text();
        } else {
            return await response.blob();
        }
    }
    
    displayResponse(responseSection, responseInfo) {
        responseSection.style.display = 'block';
        
        const responseContent = responseSection.querySelector('.response-content');
        if (!responseContent) return;
        
        const statusClass = responseInfo.status >= 200 && responseInfo.status < 300 ? 'success' : 
                           responseInfo.status >= 400 ? 'error' : 'warning';
        
        responseContent.innerHTML = `
            <div class="response-header">
                <div class="response-status">
                    <span class="status-code ${statusClass}">${responseInfo.status}</span>
                    <span class="status-text">${responseInfo.statusText}</span>
                    <span class="response-time">${responseInfo.duration}ms</span>
                </div>
                
                <div class="response-actions">
                    <button class="btn btn-small copy-response-btn">
                        <i class="fas fa-copy"></i> Copy Response
                    </button>
                </div>
            </div>
            
            <div class="response-tabs">
                <button class="tab-button active" data-tab="body">Body</button>
                <button class="tab-button" data-tab="headers">Headers</button>
                <button class="tab-button" data-tab="request">Request</button>
            </div>
            
            <div class="response-body tab-pane active" id="response-body">
                <pre><code class="language-json">${typeof responseInfo.data === 'string' ? responseInfo.data : JSON.stringify(responseInfo.data, null, 2)}</code></pre>
            </div>
            
            <div class="response-headers tab-pane" id="response-headers">
                <pre><code class="language-json">${JSON.stringify(responseInfo.headers, null, 2)}</code></pre>
            </div>
            
            <div class="response-request tab-pane" id="response-request">
                <pre><code class="language-json">${JSON.stringify(responseInfo.request, null, 2)}</code></pre>
            </div>
        `;
        
        // Setup response tabs
        this.setupResponseTabs(responseContent);
        
        // Setup copy button
        const copyBtn = responseContent.querySelector('.copy-response-btn');
        copyBtn?.addEventListener('click', () => {
            navigator.clipboard.writeText(JSON.stringify(responseInfo.data, null, 2));
            this.showMessage('Response copied to clipboard', 'success');
        });
        
        // Trigger syntax highlighting
        if (window.Prism) {
            window.Prism.highlightAllUnder(responseContent);
        }
    }
    
    setupResponseTabs(container) {
        const tabs = container.querySelectorAll('.tab-button');
        const panes = container.querySelectorAll('.tab-pane');
        
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetTab = tab.getAttribute('data-tab');
                
                tabs.forEach(t => t.classList.remove('active'));
                panes.forEach(p => p.classList.remove('active'));
                
                tab.classList.add('active');
                const targetPane = container.querySelector(`#response-${targetTab}`);
                targetPane?.classList.add('active');
            });
        });
    }
    
    displayError(responseSection, error) {
        responseSection.style.display = 'block';
        
        const responseContent = responseSection.querySelector('.response-content');
        if (!responseContent) return;
        
        responseContent.innerHTML = `
            <div class="error-response">
                <div class="error-header">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h5>Request Failed</h5>
                </div>
                <div class="error-message">
                    <pre><code>${error.message}</code></pre>
                </div>
            </div>
        `;
    }
    
    copyCurlCommand(endpoint, endpointId) {
        try {
            const request = this.buildRequest(endpoint, endpointId);
            
            let curlCommand = `curl -X ${request.method} "${request.url}"`;
            
            // Add headers
            Object.entries(request.headers).forEach(([key, value]) => {
                curlCommand += ` \\\n  -H "${key}: ${value}"`;
            });
            
            // Add body
            if (request.body) {
                curlCommand += ` \\\n  -d '${request.body}'`;
            }
            
            navigator.clipboard.writeText(curlCommand);
            this.showMessage('cURL command copied to clipboard', 'success');
        } catch (error) {
            this.showMessage(`Failed to generate cURL: ${error.message}`, 'error');
        }
    }
    
    clearRequestForm(endpointId) {
        const requestForm = document.getElementById(`requestForm-${endpointId}`);
        if (!requestForm) return;
        
        // Clear all form inputs
        const inputs = requestForm.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            if (input.type === 'checkbox' || input.type === 'radio') {
                input.checked = false;
            } else {
                input.value = '';
            }
        });
        
        // Hide response section
        const responseSection = document.getElementById(`responseSection-${endpointId}`);
        if (responseSection) {
            responseSection.style.display = 'none';
        }
    }
    
    saveRequestToHistory(endpoint, request, response, responseData, duration) {
        const historyEntry = {
            id: Date.now().toString(),
            timestamp: new Date().toISOString(),
            endpoint: endpoint.summary,
            method: request.method,
            url: request.url,
            status: response.status,
            duration,
            request: request,
            response: responseData
        };
        
        this.requestHistory.unshift(historyEntry);
        
        // Keep only last 50 requests
        if (this.requestHistory.length > 50) {
            this.requestHistory = this.requestHistory.slice(0, 50);
        }
        
        // Save to localStorage
        localStorage.setItem('api-explorer-history', JSON.stringify(this.requestHistory));
    }
    
    loadRequestHistory() {
        try {
            const savedHistory = localStorage.getItem('api-explorer-history');
            if (savedHistory) {
                this.requestHistory = JSON.parse(savedHistory);
            }
        } catch (error) {
            console.warn('Failed to load request history:', error);
            this.requestHistory = [];
        }
    }
    
    updateEndpointUrls() {
        // Update any displayed URLs when environment changes
        const pathElements = document.querySelectorAll('.endpoint-path');
        pathElements.forEach(element => {
            // URLs are already relative, no update needed
        });
    }
    
    showMessage(message, type = 'info') {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'times' : 'info'}-circle"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(toast);
        
        // Show toast
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Hide and remove toast
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Initialize API Explorer when section is shown
window.initializeAPIExplorer = function() {
    if (!window.apiExplorer) {
        window.apiExplorer = new APIExplorer();
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = APIExplorer;
}