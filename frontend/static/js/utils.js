// ============================================================================
// 3. UTILITIES MODULE - frontend/static/js/utils.js
// ============================================================================

// Additional utility functions for the frontend
class FridayUtils {
    // File upload utilities
    static createFileUploader(options = {}) {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = options.multiple || false;
        input.accept = options.accept || '*/*';

        return new Promise((resolve, reject) => {
            input.onchange = (e) => {
                const files = Array.from(e.target.files);
                if (files.length > 0) {
                    resolve(files);
                } else {
                    reject(new Error('No files selected'));
                }
            };

            input.click();
        });
    }

    // Local storage helpers
    static storage = {
        set(key, value) {
            try {
                localStorage.setItem(`friday-${key}`, JSON.stringify(value));
                return true;
            } catch (error) {
                console.warn('Failed to save to localStorage:', error);
                return false;
            }
        },

        get(key, defaultValue = null) {
            try {
                const item = localStorage.getItem(`friday-${key}`);
                return item ? JSON.parse(item) : defaultValue;
            } catch (error) {
                console.warn('Failed to read from localStorage:', error);
                return defaultValue;
            }
        },

        remove(key) {
            try {
                localStorage.removeItem(`friday-${key}`);
                return true;
            } catch (error) {
                console.warn('Failed to remove from localStorage:', error);
                return false;
            }
        }
    };

    // Animation helpers
    static animate(element, keyframes, options = {}) {
        return element.animate(keyframes, {
            duration: 300,
            easing: 'ease-out',
            ...options
        });
    }

    // Clipboard utilities
    static async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (error) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();

            try {
                document.execCommand('copy');
                document.body.removeChild(textArea);
                return true;
            } catch (fallbackError) {
                document.body.removeChild(textArea);
                return false;
            }
        }
    }

    // URL helpers
    static buildUrl(base, path, params = {}) {
        const url = new URL(path, base);
        Object.entries(params).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
                url.searchParams.append(key, value);
            }
        });
        return url.toString();
    }

    // Validation helpers
    static validators = {
        email: (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email),
        url: (url) => {
            try {
                new URL(url);
                return true;
            } catch {
                return false;
            }
        },
        notEmpty: (value) => value && value.trim().length > 0
    };
}

// Make utilities globally available
window.FridayUtils = FridayUtils;

console.log('🚀 FRIDAY Frontend Modules Loaded Successfully!');