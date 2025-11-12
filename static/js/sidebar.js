// static/JS/sidebar.js
class SidebarManager {
    constructor() {
        this.sidebar = document.getElementById('sidebar');
        this.sidebarToggle = document.getElementById('sidebarToggle');
        this.mainContent = document.querySelector('.main-content');
        this.isMobile = window.innerWidth <= 768;
        this.isCollapsed = false;
        
        this.init();
    }
    
    init() {
        // Load saved state from localStorage
        this.loadState();
        
        this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        
        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
        
        // Create overlay for mobile
        if (this.isMobile) {
            this.createOverlay();
        }
        
        // Apply initial state
        this.applyState();
    }
    
    toggleSidebar() {
        if (this.isMobile) {
            // On mobile, just show/hide the sidebar
            this.sidebar.classList.toggle('active');
            document.querySelector('.sidebar-overlay')?.classList.toggle('active');
            
            // Prevent body scroll when sidebar is open on mobile
            if (this.sidebar.classList.contains('active')) {
                document.body.style.overflow = 'hidden';
            } else {
                document.body.style.overflow = '';
            }
        } else {
            // On desktop, toggle collapsed state
            this.isCollapsed = !this.isCollapsed;
            this.applyState();
            this.saveState();
        }
    }
    
    applyState() {
        if (this.isMobile) {
            // Don't apply collapsed state on mobile
            this.sidebar.classList.remove('collapsed');
            if (this.mainContent) {
                this.mainContent.style.marginLeft = '0';
            }
        } else {
            if (this.isCollapsed) {
                this.sidebar.classList.add('collapsed');
                if (this.mainContent) {
                    this.mainContent.style.marginLeft = '55px';
                }
            } else {
                this.sidebar.classList.remove('collapsed');
                if (this.mainContent) {
                    this.mainContent.style.marginLeft = '200px';
                }
            }
        }
    }
    
    createOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        overlay.addEventListener('click', () => {
            this.sidebar.classList.remove('active');
            overlay.classList.remove('active');
            document.body.style.overflow = '';
        });
        document.body.appendChild(overlay);
    }
    
    handleResize() {
        const wasMobile = this.isMobile;
        this.isMobile = window.innerWidth <= 768;
        
        if (wasMobile !== this.isMobile) {
            if (this.isMobile) {
                // Switching to mobile - hide sidebar and remove collapsed state
                this.sidebar.classList.remove('active', 'collapsed');
                document.querySelector('.sidebar-overlay')?.classList.remove('active');
                document.body.style.overflow = '';
                if (this.mainContent) {
                    this.mainContent.style.marginLeft = '0';
                }
            } else {
                // Switching to desktop - apply saved collapsed state
                this.applyState();
            }
        }
    }
    
    saveState() {
        if (!this.isMobile) {
            localStorage.setItem('sidebarCollapsed', this.isCollapsed);
        }
    }
    
    loadState() {
        if (!this.isMobile) {
            const savedState = localStorage.getItem('sidebarCollapsed');
            if (savedState !== null) {
                this.isCollapsed = savedState === 'true';
            }
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SidebarManager();
});