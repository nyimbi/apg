/**
 * APG Workflow Orchestration - Navigation Utilities
 * Enhanced navigation with breadcrumbs, history, and URL management
 */

class NavigationManager {
    constructor() {
        this.currentPath = [];
        this.navigationHistory = [];
        this.historyIndex = -1;
        this.maxHistory = 50;
        this.breadcrumbContainer = null;
        this.sectionMap = new Map();
        
        this.init();
    }
    
    init() {
        this.buildSectionMap();
        this.createBreadcrumbs();
        this.setupNavigationEnhancements();
        this.bindEvents();
        this.initializeFromURL();
    }
    
    buildSectionMap() {
        // Build a comprehensive section map with metadata
        this.sectionMap.set('overview', {
            title: 'Overview',
            parent: null,
            icon: 'fas fa-home',
            description: 'APG Workflow Orchestration introduction and features'
        });
        
        this.sectionMap.set('quick-start', {
            title: 'Quick Start',
            parent: null,
            icon: 'fas fa-rocket',
            description: 'Get started with workflow orchestration'
        });
        
        this.sectionMap.set('installation', {
            title: 'Installation',
            parent: 'quick-start',
            icon: 'fas fa-download',
            description: 'Installation instructions and setup'
        });
        
        this.sectionMap.set('first-workflow', {
            title: 'Your First Workflow',
            parent: 'quick-start',
            icon: 'fas fa-play-circle',
            description: 'Create and run your first workflow'
        });
        
        this.sectionMap.set('workflows', {
            title: 'Workflows',
            parent: null,
            icon: 'fas fa-sitemap',
            description: 'Understanding workflow concepts and design'
        });
        
        this.sectionMap.set('components', {
            title: 'Components',
            parent: null,
            icon: 'fas fa-puzzle-piece',
            description: 'Workflow components and building blocks'
        });
        
        this.sectionMap.set('connectors', {
            title: 'Connectors',
            parent: null,
            icon: 'fas fa-plug',
            description: 'External system integrations'
        });
        
        this.sectionMap.set('templates', {
            title: 'Templates',
            parent: null,
            icon: 'fas fa-layer-group',
            description: 'Pre-built workflow templates'
        });
        
        this.sectionMap.set('api-explorer', {
            title: 'API Explorer',
            parent: null,
            icon: 'fas fa-code',
            description: 'Interactive API testing and exploration'
        });
        
        this.sectionMap.set('workflow-builder', {
            title: 'Workflow Builder',
            parent: null,
            icon: 'fas fa-tools',
            description: 'Visual workflow design interface'
        });
        
        this.sectionMap.set('template-gallery', {
            title: 'Template Gallery',
            parent: null,
            icon: 'fas fa-images',
            description: 'Browse and use workflow templates'
        });
        
        this.sectionMap.set('code-examples', {
            title: 'Code Examples',
            parent: null,
            icon: 'fas fa-file-code',
            description: 'Code samples and implementation examples'
        });
        
        this.sectionMap.set('performance', {
            title: 'Performance',
            parent: null,
            icon: 'fas fa-tachometer-alt',
            description: 'Performance optimization and tuning'
        });
        
        this.sectionMap.set('security', {
            title: 'Security',
            parent: null,
            icon: 'fas fa-shield-alt',
            description: 'Security features and best practices'
        });
        
        this.sectionMap.set('scaling', {
            title: 'Scaling',
            parent: null,
            icon: 'fas fa-expand-arrows-alt',
            description: 'Horizontal and vertical scaling strategies'
        });
        
        this.sectionMap.set('monitoring', {
            title: 'Monitoring',
            parent: null,
            icon: 'fas fa-chart-line',
            description: 'Monitoring and observability'
        });
        
        this.sectionMap.set('tutorials', {
            title: 'Tutorials',
            parent: null,
            icon: 'fas fa-graduation-cap',
            description: 'Step-by-step learning guides'
        });
        
        this.sectionMap.set('examples', {
            title: 'Examples',
            parent: null,
            icon: 'fas fa-lightbulb',
            description: 'Real-world implementation examples'
        });
        
        this.sectionMap.set('troubleshooting', {
            title: 'Troubleshooting',
            parent: null,
            icon: 'fas fa-wrench',
            description: 'Common issues and solutions'
        });
        
        this.sectionMap.set('support', {
            title: 'Support',
            parent: null,
            icon: 'fas fa-life-ring',
            description: 'Getting help and community resources'
        });
    }
    
    createBreadcrumbs() {
        const navHeader = document.querySelector('.nav-header');
        if (!navHeader) return;
        
        this.breadcrumbContainer = document.createElement('div');
        this.breadcrumbContainer.className = 'breadcrumb-container';
        this.breadcrumbContainer.innerHTML = `
            <nav class=\"breadcrumb-nav\" aria-label=\"Breadcrumb\">
                <ol class=\"breadcrumb-list\" id=\"breadcrumbList\">
                    <!-- Breadcrumbs will be inserted here -->
                </ol>
            </nav>
        `;
        
        // Insert after the main nav header
        navHeader.parentNode.insertBefore(this.breadcrumbContainer, navHeader.nextSibling);
    }
    
    setupNavigationEnhancements() {
        // Add keyboard navigation indicators
        const navItems = document.querySelectorAll('.section-items a[data-section]');
        navItems.forEach((item, index) => {
            // Add keyboard shortcut hints
            if (index < 9) {
                const shortcut = document.createElement('span');
                shortcut.className = 'nav-shortcut';
                shortcut.textContent = `Alt+${index + 1}`;
                shortcut.title = `Press Alt+${index + 1} to navigate here`;
                item.appendChild(shortcut);
            }
            
            // Add section descriptions as tooltips
            const sectionId = item.getAttribute('data-section');
            const sectionInfo = this.sectionMap.get(sectionId);
            if (sectionInfo && sectionInfo.description) {
                item.title = sectionInfo.description;
            }
        });
        
        // Create back/forward navigation buttons
        this.createNavigationControls();
        
        // Add progress indicator
        this.createProgressIndicator();
    }
    
    createNavigationControls() {
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'nav-controls';
        controlsContainer.innerHTML = `
            <button class=\"nav-control-btn\" id=\"navBack\" title=\"Go back (Alt+Left)\" disabled>
                <i class=\"fas fa-chevron-left\"></i>
            </button>
            <button class=\"nav-control-btn\" id=\"navForward\" title=\"Go forward (Alt+Right)\" disabled>
                <i class=\"fas fa-chevron-right\"></i>
            </button>
            <button class=\"nav-control-btn\" id=\"navHome\" title=\"Go to overview (Alt+Home)\">
                <i class=\"fas fa-home\"></i>
            </button>
        `;
        
        // Insert navigation controls in the header
        const searchContainer = document.querySelector('.search-container');
        if (searchContainer) {
            searchContainer.parentNode.insertBefore(controlsContainer, searchContainer);
        }
        
        // Bind control events
        document.getElementById('navBack')?.addEventListener('click', () => this.goBack());
        document.getElementById('navForward')?.addEventListener('click', () => this.goForward());
        document.getElementById('navHome')?.addEventListener('click', () => this.navigateToSection('overview'));
    }
    
    createProgressIndicator() {
        const progressContainer = document.createElement('div');
        progressContainer.className = 'nav-progress';
        progressContainer.innerHTML = `
            <div class=\"progress-bar\">
                <div class=\"progress-fill\" id=\"navProgress\"></div>
            </div>
            <span class=\"progress-text\" id=\"navProgressText\">1 of 20</span>
        `;
        
        // Add to navigation
        const navMenu = document.getElementById('navMenu');
        if (navMenu) {
            navMenu.parentNode.insertBefore(progressContainer, navMenu.nextSibling);
        }
    }
    
    bindEvents() {
        // Enhanced keyboard navigation
        document.addEventListener('keydown', (e) => {
            // Alt + number for quick navigation
            if (e.altKey && e.key >= '1' && e.key <= '9') {
                e.preventDefault();
                const navItems = document.querySelectorAll('.section-items a[data-section]');
                const index = parseInt(e.key) - 1;
                if (navItems[index]) {
                    const sectionId = navItems[index].getAttribute('data-section');
                    this.navigateToSection(sectionId);
                }
            }
            
            // Navigation shortcuts
            if (e.altKey) {
                switch (e.key) {
                    case 'ArrowLeft':
                        e.preventDefault();
                        this.goBack();
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        this.goForward();
                        break;
                    case 'Home':
                        e.preventDefault();
                        this.navigateToSection('overview');
                        break;
                }
            }
            
            // Tab navigation within sections
            if (e.key === 'Tab' && !e.shiftKey && !e.ctrlKey && !e.altKey) {
                this.handleTabNavigation(e);
            }
        });
        
        // Listen for navigation events from main documentation
        document.addEventListener('sectionChanged', (e) => {
            this.handleSectionChange(e.detail.sectionId, e.detail.fromHistory);
        });
        
        // Handle browser back/forward
        window.addEventListener('popstate', (e) => {
            if (e.state && e.state.section) {
                this.navigateToSection(e.state.section, true);
            }
        });
    }
    
    navigateToSection(sectionId, fromHistory = false) {
        const sectionInfo = this.sectionMap.get(sectionId);
        if (!sectionInfo) return;
        
        // Add to navigation history if not from history navigation
        if (!fromHistory) {
            this.addToHistory(sectionId);
        }
        
        // Update breadcrumb
        this.updateBreadcrumb(sectionId);
        
        // Update progress
        this.updateProgress(sectionId);
        
        // Update navigation controls
        this.updateNavigationControls();
        
        // Update URL
        this.updateURL(sectionId);
        
        // Dispatch navigation event
        this.dispatchNavigationEvent(sectionId);
        
        // Scroll to top of content
        this.scrollToTop();
    }
    
    addToHistory(sectionId) {
        // Remove any future history if we're not at the end
        this.navigationHistory = this.navigationHistory.slice(0, this.historyIndex + 1);
        
        // Add new entry
        this.navigationHistory.push(sectionId);
        this.historyIndex++;
        
        // Limit history size
        if (this.navigationHistory.length > this.maxHistory) {
            this.navigationHistory.shift();
            this.historyIndex--;
        }
    }
    
    goBack() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            const sectionId = this.navigationHistory[this.historyIndex];
            this.navigateToSection(sectionId, true);
        }
    }
    
    goForward() {
        if (this.historyIndex < this.navigationHistory.length - 1) {
            this.historyIndex++;
            const sectionId = this.navigationHistory[this.historyIndex];
            this.navigateToSection(sectionId, true);
        }
    }
    
    updateBreadcrumb(sectionId) {
        const breadcrumbList = document.getElementById('breadcrumbList');
        if (!breadcrumbList) return;
        
        const path = this.buildBreadcrumbPath(sectionId);
        
        breadcrumbList.innerHTML = path.map((section, index) => {
            const sectionInfo = this.sectionMap.get(section);
            const isLast = index === path.length - 1;
            
            return `
                <li class="breadcrumb-item ${isLast ? 'active' : ''}">
                    ${!isLast ? `
                        <a href="#${section}" class="breadcrumb-link" data-section="${section}">
                            <i class="${sectionInfo.icon}"></i>
                            <span>${sectionInfo.title}</span>
                        </a>
                    ` : `
                        <span class="breadcrumb-current">
                            <i class="${sectionInfo.icon}"></i>
                            <span>${sectionInfo.title}</span>
                        </span>
                    `}
                    ${!isLast ? '<i class="fas fa-chevron-right breadcrumb-separator"></i>' : ''}
                </li>
            `;
        }).join('');
        
        // Bind breadcrumb navigation
        breadcrumbList.querySelectorAll('.breadcrumb-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetSection = link.getAttribute('data-section');
                this.navigateToSection(targetSection);
            });
        });
    }
    
    buildBreadcrumbPath(sectionId) {
        const path = [];
        let currentSection = sectionId;
        
        while (currentSection) {
            path.unshift(currentSection);
            const sectionInfo = this.sectionMap.get(currentSection);
            currentSection = sectionInfo ? sectionInfo.parent : null;
        }
        
        return path;
    }
    
    updateProgress(sectionId) {
        const allSections = Array.from(this.sectionMap.keys());
        const currentIndex = allSections.indexOf(sectionId);
        const progress = ((currentIndex + 1) / allSections.length) * 100;
        
        const progressFill = document.getElementById('navProgress');
        const progressText = document.getElementById('navProgressText');
        
        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${currentIndex + 1} of ${allSections.length}`;
        }
    }
    
    updateNavigationControls() {
        const backBtn = document.getElementById('navBack');
        const forwardBtn = document.getElementById('navForward');
        
        if (backBtn) {
            backBtn.disabled = this.historyIndex <= 0;
        }
        
        if (forwardBtn) {
            forwardBtn.disabled = this.historyIndex >= this.navigationHistory.length - 1;
        }
    }
    
    updateURL(sectionId) {
        const url = `${window.location.pathname}#${sectionId}`;
        const state = { section: sectionId };
        history.pushState(state, '', url);
    }
    
    dispatchNavigationEvent(sectionId) {
        const event = new CustomEvent('navigationChanged', {
            detail: {
                sectionId: sectionId,
                sectionInfo: this.sectionMap.get(sectionId),
                breadcrumbPath: this.buildBreadcrumbPath(sectionId)
            }
        });
        document.dispatchEvent(event);
    }
    
    handleSectionChange(sectionId, fromHistory = false) {
        this.navigateToSection(sectionId, fromHistory);
    }
    
    handleTabNavigation(e) {
        const focusableElements = this.getFocusableElements();
        const currentIndex = focusableElements.indexOf(document.activeElement);
        
        if (currentIndex === focusableElements.length - 1) {
            // At the last element, wrap to first
            e.preventDefault();
            focusableElements[0]?.focus();
        }
    }
    
    getFocusableElements() {
        const selectors = [
            'a[href]',
            'button:not([disabled])',
            'input:not([disabled])',
            'select:not([disabled])',
            'textarea:not([disabled])',
            '[tabindex]:not([tabindex="-1"])'
        ];
        
        return Array.from(document.querySelectorAll(selectors.join(', ')))
            .filter(el => !el.hidden && el.offsetParent !== null);
    }
    
    scrollToTop() {
        const content = document.querySelector('.docs-content');
        if (content) {
            content.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }
    
    initializeFromURL() {
        const hash = window.location.hash.substring(1);
        if (hash && this.sectionMap.has(hash)) {
            this.navigateToSection(hash, true);
        } else {
            this.navigateToSection('overview', true);
        }
    }
    
    // Public API methods
    getCurrentSection() {
        return this.navigationHistory[this.historyIndex];
    }
    
    getSectionInfo(sectionId) {
        return this.sectionMap.get(sectionId);
    }
    
    getAllSections() {
        return Array.from(this.sectionMap.entries()).map(([id, info]) => ({
            id,
            ...info
        }));
    }
    
    isValidSection(sectionId) {
        return this.sectionMap.has(sectionId);
    }
    
    getNextSection(currentSectionId) {
        const allSections = Array.from(this.sectionMap.keys());
        const currentIndex = allSections.indexOf(currentSectionId);
        return currentIndex < allSections.length - 1 ? allSections[currentIndex + 1] : null;
    }
    
    getPreviousSection(currentSectionId) {
        const allSections = Array.from(this.sectionMap.keys());
        const currentIndex = allSections.indexOf(currentSectionId);
        return currentIndex > 0 ? allSections[currentIndex - 1] : null;
    }
    
    createSectionLinks() {
        // Create next/previous navigation at the bottom of sections
        const sections = document.querySelectorAll('.content-section');
        sections.forEach(section => {
            const sectionId = section.id;
            const nextSection = this.getNextSection(sectionId);
            const prevSection = this.getPreviousSection(sectionId);
            
            if (nextSection || prevSection) {
                const nav = document.createElement('nav');
                nav.className = 'section-navigation';
                nav.innerHTML = `
                    <div class="section-nav-links">
                        ${prevSection ? `
                            <a href="#${prevSection}" class="section-nav-link prev" data-section="${prevSection}">
                                <i class="fas fa-chevron-left"></i>
                                <div class="nav-link-content">
                                    <span class="nav-link-label">Previous</span>
                                    <span class="nav-link-title">${this.sectionMap.get(prevSection).title}</span>
                                </div>
                            </a>
                        ` : '<div></div>'}
                        
                        ${nextSection ? `
                            <a href="#${nextSection}" class="section-nav-link next" data-section="${nextSection}">
                                <div class="nav-link-content">
                                    <span class="nav-link-label">Next</span>
                                    <span class="nav-link-title">${this.sectionMap.get(nextSection).title}</span>
                                </div>
                                <i class="fas fa-chevron-right"></i>
                            </a>
                        ` : '<div></div>'}
                    </div>
                `;
                
                section.appendChild(nav);
                
                // Bind navigation events
                nav.querySelectorAll('.section-nav-link').forEach(link => {
                    link.addEventListener('click', (e) => {
                        e.preventDefault();
                        const targetSection = link.getAttribute('data-section');
                        this.navigateToSection(targetSection);
                    });
                });
            }
        });
    }
}

// Initialize navigation manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.navigationManager = new NavigationManager();
    
    // Create section navigation links
    setTimeout(() => {
        window.navigationManager.createSectionLinks();
    }, 100);
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NavigationManager;
}