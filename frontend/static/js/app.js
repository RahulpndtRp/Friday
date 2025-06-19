// Updated Frontend JavaScript for /api Routes
// Clean separation between frontend routes and API routes

// ============================================================================
// 1. UPDATED APP.JS - Fix API Routes
// ============================================================================

// frontend/static/js/app.js - UPDATED VERSION
class FridayApp {
    constructor() {
        this.config = window.FRIDAY_CONFIG || {};
        this.apiBaseUrl = this.config.apiBaseUrl || window.location.origin;
        this.userId = this.config.userId || 'friday_user_001';
        this.version = this.config.version || '1.0.0';

        this.theme = localStorage.getItem('friday-theme') || 'light';
        this.debugMode = localStorage.getItem('friday-debug') === 'true';

        this.init();
    }

    init() {
        this.setTheme(this.theme);
        this.setupGlobalEventListeners();
        this.log('FRIDAY App initialized', {
            apiBaseUrl: this.apiBaseUrl,
            userId: this.userId,
            version: this.version
        });
    }

    // Logging utility
    log(message, data = null, level = 'info') {
        const timestamp = new Date().toISOString();
        const logData = { timestamp, level, message, data };

        if (this.debugMode || level === 'error') {
            console[level](
                `[FRIDAY ${level.toUpperCase()}] ${message}`,
                data ? data : ''
            );
        }

        this.storeLogs(logData);
    }

    storeLogs(logData) {
        try {
            const logs = JSON.parse(localStorage.getItem('friday-logs') || '[]');
            logs.push(logData);

            if (logs.length > 100) {
                logs.splice(0, logs.length - 100);
            }

            localStorage.setItem('friday-logs', JSON.stringify(logs));
        } catch (error) {
            console.warn('Failed to store logs:', error);
        }
    }

    // Theme management
    setTheme(theme) {
        this.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('friday-theme', theme);
        this.log('Theme changed', { theme });
    }

    toggleDebug() {
        this.debugMode = !this.debugMode;
        localStorage.setItem('friday-debug', this.debugMode.toString());
        this.log('Debug mode toggled', { debugMode: this.debugMode });
    }

    // API utilities with /api prefix
    async apiRequest(endpoint, options = {}) {
        // Automatically prepend /api to endpoints
        const apiEndpoint = endpoint.startsWith('/api') ? endpoint : `/api${endpoint}`;
        const url = `${this.apiBaseUrl}${apiEndpoint}`;

        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        };

        const requestOptions = { ...defaultOptions, ...options };

        this.log('API Request', {
            endpoint: apiEndpoint,
            url,
            method: requestOptions.method || 'GET'
        });

        try {
            const response = await fetch(url, requestOptions);

            // Handle non-JSON responses (like streaming)
            if (response.headers.get('content-type')?.includes('text/plain')) {
                this.log('API Response (streaming)', { url, status: response.status });
                return response;
            }

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            this.log('API Response', { url, status: response.status, data });
            return data;
        } catch (error) {
            this.log('API Error', { url, error: error.message }, 'error');
            throw error;
        }
    }

    // Health check - keep at root level (not /api/health)
    async checkHealth() {
        try {
            // Health endpoint stays at root level
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const health = await response.json();

            if (!response.ok) {
                throw new Error(health.detail || `HTTP ${response.status}`);
            }

            this.log('Health check successful', health);
            return health;
        } catch (error) {
            this.log('Health check failed', error.message, 'error');
            throw error;
        }
    }

    // Enhanced notification system
    showNotification(message, type = 'info', duration = 5000) {
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                pointer-events: none;
            `;
            document.body.appendChild(container);
        }

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            background: ${type === 'error' ? '#f44336' : type === 'success' ? '#4caf50' : '#2196f3'};
            color: white;
            padding: 12px 20px;
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            pointer-events: auto;
            cursor: pointer;
            transition: all 0.3s ease;
            max-width: 300px;
            word-wrap: break-word;
            animation: slideInRight 0.3s ease-out;
        `;

        // Add slide animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        if (!document.head.querySelector('#notification-styles')) {
            style.id = 'notification-styles';
            document.head.appendChild(style);
        }

        notification.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>${message}</span>
                <span style="margin-left: 10px; cursor: pointer; font-weight: bold;" onclick="this.parentElement.parentElement.remove()">×</span>
            </div>
        `;

        container.appendChild(notification);

        setTimeout(() => {
            if (notification.parentElement) {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => notification.remove(), 300);
            }
        }, duration);

        this.log('Notification shown', { message, type });
    }

    // Utility functions
    formatTime(date) {
        return new Intl.DateTimeFormat('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        }).format(date);
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Global event listeners
    setupGlobalEventListeners() {
        window.addEventListener('online', () => {
            this.showNotification('Connection restored', 'success');
        });

        window.addEventListener('offline', () => {
            this.showNotification('Connection lost', 'error');
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.log('Unhandled promise rejection', event.reason, 'error');
            this.showNotification('An unexpected error occurred', 'error');
        });

        window.addEventListener('error', (event) => {
            this.log('JavaScript error', {
                message: event.message,
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno
            }, 'error');
        });
    }
}

// Initialize global app instance
window.fridayApp = new FridayApp();
