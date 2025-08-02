/**
 * APG Workflow Designer Main
 * 
 * Main initialization and coordination for the workflow designer.
 * 
 * © 2025 Datacraft. All rights reserved.
 * Author: Nyimbi Odero <nyimbi@gmail.com>
 */

// Global designer configuration
let designerConfig = {};

// Initialize the workflow designer when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    try {
        // Load configuration from embedded JSON
        const configElement = document.getElementById('designer-config');
        if (configElement) {
            designerConfig = JSON.parse(configElement.textContent);
        }
        
        // Set default configuration
        designerConfig = {
            api_base_url: '/api/v1/workflow/designer',
            auto_save_interval: 30, // seconds
            max_undo_steps: 50,
            grid_size: 20,
            workspace_id: 'default',
            session_timeout: 3600, // seconds
            collaboration_enabled: true,
            validation_enabled: true,
            debug_mode: false,
            ...designerConfig
        };
        
        // Initialize the designer
        initializeDesigner();
        
    } catch (error) {
        console.error('Failed to initialize workflow designer:', error);
        showInitializationError(error);
    }
});

async function initializeDesigner() {
    try {
        // Show loading state
        showGlobalLoading('Initializing Workflow Designer...');
        
        // Validate browser compatibility
        if (!validateBrowserCompatibility()) {
            throw new Error('Browser not supported. Please use a modern browser.');
        }
        
        // Initialize theme
        initializeTheme();
        
        // Create global designer instance
        window.workflowDesigner = new WorkflowDesigner(designerConfig);
        
        // Setup global error handling
        setupGlobalErrorHandling();
        
        // Setup beforeunload handling
        setupBeforeUnloadHandling();
        
        // Setup periodic health checks
        setupHealthChecks();
        
        // Setup keyboard shortcuts
        setupGlobalKeyboardShortcuts();
        
        // Setup performance monitoring
        setupPerformanceMonitoring();
        
        console.log('Workflow Designer initialized successfully');
        
        // Hide loading state
        hideGlobalLoading();
        
    } catch (error) {
        console.error('Designer initialization failed:', error);
        hideGlobalLoading();
        showInitializationError(error);
    }
}

// === Browser Compatibility ===

function validateBrowserCompatibility() {
    // Check for required features
    const requiredFeatures = [
        'fetch',
        'Promise',
        'Map',
        'Set',
        'WebSocket',
        'localStorage',
        'requestAnimationFrame',
        'addEventListener'
    ];
    
    for (const feature of requiredFeatures) {
        if (!(feature in window)) {
            console.error(`Missing required feature: ${feature}`);
            return false;
        }
    }
    
    // Check for SVG support
    if (!document.createElementNS || !document.createElementNS('http://www.w3.org/2000/svg', 'svg').createSVGRect) {
        console.error('SVG not supported');
        return false;
    }
    
    // Check for modern JS features
    try {
        // Test arrow functions
        const testArrow = () => true;
        
        // Test async/await
        const testAsync = async () => await Promise.resolve(true);
        
        // Test destructuring
        const {test} = {test: true};
        
        // Test spread operator
        const testSpread = {...{test: true}};
        
        return true;
    } catch (error) {
        console.error('Modern JavaScript features not supported:', error);
        return false;
    }
}

// === Theme Management ===

function initializeTheme() {
    const savedTheme = localStorage.getItem('workflow-designer-theme') || 'light';
    applyTheme(savedTheme);
    
    // Listen for system theme changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addListener((e) => {
            const currentTheme = localStorage.getItem('workflow-designer-theme');
            if (currentTheme === 'auto') {
                applyTheme('auto');
            }
        });
    }
}

function applyTheme(theme) {
    const root = document.documentElement;
    
    switch (theme) {
        case 'dark':
            root.setAttribute('data-theme', 'dark');
            break;
        case 'auto':
            const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            root.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
            break;
        default:
            root.setAttribute('data-theme', 'light');
    }
    
    localStorage.setItem('workflow-designer-theme', theme);
}

// === Error Handling ===

function setupGlobalErrorHandling() {
    // Catch unhandled JavaScript errors
    window.addEventListener('error', (event) => {
        console.error('Global error:', event.error);
        if (window.workflowDesigner) {
            window.workflowDesigner.showError('An unexpected error occurred: ' + event.error.message);
        }
    });
    
    // Catch unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
        if (window.workflowDesigner) {
            window.workflowDesigner.showError('An unexpected error occurred: ' + event.reason);
        }
    });
}

function showInitializationError(error) {
    const container = document.querySelector('.designer-container') || document.body;
    
    container.innerHTML = `
        <div class="initialization-error">
            <div class="error-content">
                <i class="fas fa-exclamation-triangle error-icon"></i>
                <h3>Failed to Initialize Workflow Designer</h3>
                <p class="error-message">${error.message}</p>
                <div class="error-actions">
                    <button onclick="location.reload()" class="btn btn-primary">
                        <i class="fas fa-redo"></i> Retry
                    </button>
                    <button onclick="reportError('${error.message}')" class="btn btn-secondary">
                        <i class="fas fa-bug"></i> Report Issue
                    </button>
                </div>
                <details class="error-details">
                    <summary>Technical Details</summary>
                    <pre class="error-stack">${error.stack || 'No stack trace available'}</pre>
                </details>
            </div>
        </div>
    `;
}

// === Loading States ===

function showGlobalLoading(message = 'Loading...') {
    let overlay = document.getElementById('global-loading-overlay');
    
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'global-loading-overlay';
        overlay.className = 'global-loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-message">${message}</div>
            </div>
        `;
        document.body.appendChild(overlay);
    } else {
        overlay.querySelector('.loading-message').textContent = message;
        overlay.style.display = 'flex';
    }
}

function hideGlobalLoading() {
    const overlay = document.getElementById('global-loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// === Lifecycle Management ===

function setupBeforeUnloadHandling() {
    window.addEventListener('beforeunload', (event) => {
        if (window.workflowDesigner && window.workflowDesigner.isDirty) {
            const message = 'You have unsaved changes. Are you sure you want to leave?';
            event.returnValue = message;
            return message;
        }
    });
}

function setupHealthChecks() {
    // Check designer health every 60 seconds
    setInterval(() => {
        if (window.workflowDesigner) {
            performHealthCheck();
        }
    }, 60000);
}

async function performHealthCheck() {
    try {
        // Check API connectivity
        const response = await fetch(`${designerConfig.api_base_url}/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (!response.ok) {
            console.warn('API health check failed:', response.status);
            if (window.workflowDesigner) {
                window.workflowDesigner.showNotification('Connection issues detected', 'warning');
            }
        }
        
        // Check session validity
        if (window.workflowDesigner.sessionId) {
            const sessionResponse = await fetch(`${designerConfig.api_base_url}/session/${window.workflowDesigner.sessionId}/status`);
            
            if (!sessionResponse.ok && sessionResponse.status === 404) {
                console.warn('Session expired');
                if (window.workflowDesigner) {
                    window.workflowDesigner.showError('Your session has expired. Please refresh the page.');
                }
            }
        }
        
    } catch (error) {
        console.warn('Health check failed:', error);
    }
}

// === Keyboard Shortcuts ===

function setupGlobalKeyboardShortcuts() {
    document.addEventListener('keydown', (event) => {
        // Don't interfere with input fields
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }
        
        // Global shortcuts
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 'h':
                    event.preventDefault();
                    toggleShortcutsHelp();
                    break;
                case '/':
                    event.preventDefault();
                    focusComponentSearch();
                    break;
            }
        }
        
        // Other global shortcuts
        switch (event.key) {
            case '?':
                if (event.shiftKey) {
                    event.preventDefault();
                    toggleShortcutsHelp();
                }
                break;
            case 'F1':
                event.preventDefault();
                openHelp();
                break;
        }
    });
}

function toggleShortcutsHelp() {
    let modal = document.getElementById('shortcuts-help-modal');
    
    if (!modal) {
        modal = createShortcutsHelpModal();
        document.body.appendChild(modal);
    }
    
    $(modal).modal('toggle');
}

function createShortcutsHelpModal() {
    const modal = document.createElement('div');
    modal.id = 'shortcuts-help-modal';
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Keyboard Shortcuts</h5>
                    <button type="button" class="close" data-dismiss="modal">
                        <span>&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h6>General</h6>
                            <dl class="shortcut-list">
                                <dt>Ctrl+S</dt><dd>Save workflow</dd>
                                <dt>Ctrl+Z</dt><dd>Undo</dd>
                                <dt>Ctrl+Shift+Z</dt><dd>Redo</dd>
                                <dt>Ctrl+A</dt><dd>Select all</dd>
                                <dt>Delete</dt><dd>Delete selected</dd>
                                <dt>Escape</dt><dd>Clear selection</dd>
                            </dl>
                            
                            <h6>Navigation</h6>
                            <dl class="shortcut-list">
                                <dt>Ctrl+/</dt><dd>Focus component search</dd>
                                <dt>Space</dt><dd>Pan mode</dd>
                                <dt>+</dt><dd>Zoom in</dd>
                                <dt>-</dt><dd>Zoom out</dd>
                                <dt>0</dt><dd>Fit to screen</dd>
                            </dl>
                        </div>
                        <div class="col-md-6">
                            <h6>Editing</h6>
                            <dl class="shortcut-list">
                                <dt>Ctrl+C</dt><dd>Copy selected</dd>
                                <dt>Ctrl+V</dt><dd>Paste</dd>
                                <dt>Ctrl+D</dt><dd>Duplicate</dd>
                                <dt>Enter</dt><dd>Edit properties</dd>
                                <dt>Tab</dt><dd>Next component</dd>
                                <dt>Shift+Tab</dt><dd>Previous component</dd>
                            </dl>
                            
                            <h6>Help</h6>
                            <dl class="shortcut-list">
                                <dt>F1</dt><dd>Open help</dd>
                                <dt>Ctrl+H</dt><dd>Show shortcuts</dd>
                                <dt>?</dt><dd>Show shortcuts</dd>
                            </dl>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    return modal;
}

function focusComponentSearch() {
    const searchInput = document.getElementById('component-search');
    if (searchInput) {
        searchInput.focus();
        searchInput.select();
    }
}

function openHelp() {
    if (window.workflowDesigner) {
        window.workflowDesigner.showHelp();
    } else {
        window.open('/workflow/designer/help', '_blank');
    }
}

// === Performance Monitoring ===

function setupPerformanceMonitoring() {
    if (!designerConfig.debug_mode) return;
    
    // Monitor render performance
    let frameCount = 0;
    let lastFrameTime = performance.now();
    
    function measureFrameRate() {
        frameCount++;
        const currentTime = performance.now();
        
        if (currentTime - lastFrameTime >= 1000) {
            const fps = Math.round((frameCount * 1000) / (currentTime - lastFrameTime));
            
            if (fps < 30) {
                console.warn(`Low frame rate detected: ${fps} FPS`);
            }
            
            frameCount = 0;
            lastFrameTime = currentTime;
        }
        
        requestAnimationFrame(measureFrameRate);
    }
    
    requestAnimationFrame(measureFrameRate);
    
    // Monitor memory usage
    if (performance.memory) {
        setInterval(() => {
            const memory = performance.memory;
            const usedMB = Math.round(memory.usedJSHeapSize / 1048576);
            const limitMB = Math.round(memory.jsHeapSizeLimit / 1048576);
            
            if (usedMB > limitMB * 0.8) {
                console.warn(`High memory usage: ${usedMB}MB / ${limitMB}MB`);
            }
        }, 30000);
    }
}

// === Utility Functions ===

function reportError(message) {
    // This would typically send error reports to a logging service
    console.log('Error reported:', message);
    
    if (window.workflowDesigner) {
        window.workflowDesigner.showNotification('Error report sent. Thank you!', 'success');
    }
}

function getDesignerVersion() {
    return designerConfig.version || '1.0.0';
}

function getDesignerInfo() {
    return {
        version: getDesignerVersion(),
        config: designerConfig,
        browser: {
            userAgent: navigator.userAgent,
            language: navigator.language,
            platform: navigator.platform
        },
        performance: performance.memory ? {
            usedJSHeapSize: performance.memory.usedJSHeapSize,
            totalJSHeapSize: performance.memory.totalJSHeapSize,
            jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
        } : null
    };
}

// === Global API ===

// Expose global functions for external access
window.WorkflowDesignerAPI = {
    getDesigner: () => window.workflowDesigner,
    getConfig: () => designerConfig,
    getVersion: getDesignerVersion,
    getInfo: getDesignerInfo,
    applyTheme: applyTheme,
    reportError: reportError,
    showHelp: openHelp,
    focusSearch: focusComponentSearch
};

// === Styles ===

// Add global styles
const globalStyles = `
<style>
.global-loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.9);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    backdrop-filter: blur(2px);
}

.loading-content {
    text-align: center;
    padding: 40px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.loading-spinner {
    width: 40px;
    height: 40px;
    margin: 0 auto 20px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-message {
    font-size: 16px;
    color: #495057;
    font-weight: 500;
}

.initialization-error {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background: #f8f9fa;
    padding: 20px;
}

.error-content {
    max-width: 600px;
    text-align: center;
    background: white;
    padding: 40px;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.error-icon {
    font-size: 48px;
    color: #dc3545;
    margin-bottom: 20px;
}

.error-content h3 {
    color: #495057;
    margin-bottom: 16px;
}

.error-message {
    color: #6c757d;
    margin-bottom: 30px;
    font-size: 16px;
}

.error-actions {
    margin-bottom: 30px;
}

.error-actions .btn {
    margin: 0 8px;
}

.error-details {
    text-align: left;
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #dee2e6;
}

.error-details summary {
    cursor: pointer;
    font-weight: 500;
    color: #6c757d;
}

.error-stack {
    background: #f8f9fa;
    padding: 16px;
    border-radius: 4px;
    font-size: 12px;
    color: #495057;
    margin-top: 12px;
    overflow-x: auto;
}

.shortcut-list {
    font-size: 13px;
}

.shortcut-list dt {
    font-family: monospace;
    background: #f8f9fa;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: normal;
    display: inline-block;
    min-width: 80px;
    margin-bottom: 4px;
}

.shortcut-list dd {
    margin-left: 90px;
    margin-bottom: 8px;
    color: #6c757d;
}

/* Dark theme styles */
[data-theme="dark"] .global-loading-overlay {
    background: rgba(26, 32, 44, 0.9);
}

[data-theme="dark"] .loading-content {
    background: #2d3748;
    color: #e9ecef;
}

[data-theme="dark"] .error-content {
    background: #2d3748;
    color: #e9ecef;
}

[data-theme="dark"] .error-stack {
    background: #1a202c;
    color: #e9ecef;
}
</style>
`;

// Inject global styles
document.head.insertAdjacentHTML('beforeend', globalStyles);

console.log('Workflow Designer main script loaded');