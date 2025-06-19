
// ============================================================================
// 2. UPDATED CHAT.JS - Use /api Routes
// ============================================================================

// frontend/static/js/chat.js - UPDATED VERSION
class FridayChat {
    constructor() {
        this.app = window.fridayApp;
        this.elements = this.initializeElements();
        this.state = {
            isConnected: false,
            isStreaming: true,
            currentConversationId: null,
            isProcessing: false,
            messageHistory: []
        };

        this.currentStreamingMessage = null;
        this.init();
    }

    initializeElements() {
        return {
            chatStatus: document.getElementById('chatStatus'),
            chatMessages: document.getElementById('chatMessages'),
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            streamingToggle: document.getElementById('streamingToggle'),
            debugToggle: document.getElementById('debugToggle'),
            settingsBtn: document.getElementById('settingsBtn'),
            clearBtn: document.getElementById('clearBtn'),
            settingsModal: document.getElementById('settingsModal'),
            closeSettings: document.getElementById('closeSettings'),
            themeSelect: document.getElementById('themeSelect')
        };
    }

    init() {
        this.setupEventListeners();
        this.loadSettings();
        this.checkConnection();
        this.addWelcomeMessage();
        this.app.log('Chat module initialized');
    }

    setupEventListeners() {
        if (this.elements.sendBtn) {
            this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        }

        if (this.elements.messageInput) {
            this.elements.messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey && !this.state.isProcessing) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            this.elements.messageInput.addEventListener('input', () => {
                this.autoResizeTextarea();
            });
        }

        if (this.elements.streamingToggle) {
            this.elements.streamingToggle.addEventListener('change', (e) => {
                this.state.isStreaming = e.target.checked;
                this.saveSettings();
                this.app.log('Streaming toggled', { isStreaming: this.state.isStreaming });
            });
        }

        if (this.elements.debugToggle) {
            this.elements.debugToggle.addEventListener('change', (e) => {
                this.app.toggleDebug();
            });
        }

        if (this.elements.clearBtn) {
            this.elements.clearBtn.addEventListener('click', () => this.clearChat());
        }

        if (this.elements.settingsBtn) {
            this.elements.settingsBtn.addEventListener('click', () => this.openSettings());
        }

        if (this.elements.closeSettings) {
            this.elements.closeSettings.addEventListener('click', () => this.closeSettings());
        }

        if (this.elements.themeSelect) {
            this.elements.themeSelect.addEventListener('change', (e) => {
                this.app.setTheme(e.target.value);
            });
        }

        if (this.elements.settingsModal) {
            this.elements.settingsModal.addEventListener('click', (e) => {
                if (e.target === this.elements.settingsModal) {
                    this.closeSettings();
                }
            });
        }
    }

    loadSettings() {
        const streamingPref = localStorage.getItem('friday-streaming');
        if (streamingPref !== null) {
            this.state.isStreaming = streamingPref === 'true';
            if (this.elements.streamingToggle) {
                this.elements.streamingToggle.checked = this.state.isStreaming;
            }
        }

        if (this.elements.debugToggle) {
            this.elements.debugToggle.checked = this.app.debugMode;
        }

        if (this.elements.themeSelect) {
            this.elements.themeSelect.value = this.app.theme;
        }
    }

    saveSettings() {
        localStorage.setItem('friday-streaming', this.state.isStreaming.toString());
    }

    async checkConnection() {
        try {
            await this.app.api.checkHealth();
            this.updateConnectionStatus('Connected', 'success');
            this.state.isConnected = true;
            this.app.showNotification('Connected to FRIDAY', 'success', 2000);
        } catch (error) {
            this.updateConnectionStatus('Disconnected', 'error');
            this.state.isConnected = false;
            this.addMessage('❌ Could not connect to FRIDAY Assistant API', 'error');
            this.app.showNotification('Connection failed - Check if server is running', 'error');
        }
    }

    updateConnectionStatus(message, type) {
        if (this.elements.chatStatus) {
            this.elements.chatStatus.textContent = message;
            this.elements.chatStatus.className = `chat-status ${type}`;
        }
    }

    addWelcomeMessage() {
        this.addMessage(
            '👋 Hello! I\'m FRIDAY, your personal AI assistant. I can help you with questions, conversations, and accessing your documents and memories.',
            'assistant'
        );
        this.addMessage(
            '💡 Try asking: "What can you help me with?" or "Tell me about your capabilities"',
            'info'
        );
    }

    addMessage(content, type, isStreaming = false) {
        if (!this.elements.chatMessages) return null;

        const messageEl = document.createElement('div');

        if (type === 'info') {
            messageEl.className = 'info-message';
        } else if (type === 'error') {
            messageEl.className = 'error-message';
        } else {
            messageEl.className = `message ${type}`;
            if (isStreaming) {
                messageEl.classList.add('streaming');
            }
        }

        messageEl.innerHTML = this.formatMessage(content);
        this.elements.chatMessages.appendChild(messageEl);
        this.scrollToBottom();

        return messageEl;
    }

    formatMessage(content) {
        return content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code style="background: #f1f1f1; padding: 2px 4px; border-radius: 3px;">$1</code>')
            .replace(/\n/g, '<br>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    }

    addTypingIndicator() {
        if (!this.elements.chatMessages) return null;

        const messageEl = document.createElement('div');
        messageEl.className = 'message assistant typing-indicator';
        messageEl.id = 'typing-indicator';
        messageEl.innerHTML = `
            <span>FRIDAY is thinking</span>
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;

        this.elements.chatMessages.appendChild(messageEl);
        this.scrollToBottom();
        return messageEl;
    }

    removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    scrollToBottom() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        }
    }

    autoResizeTextarea() {
        if (!this.elements.messageInput) return;

        const textarea = this.elements.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    async sendMessage() {
        if (!this.elements.messageInput || !this.elements.sendBtn) return;

        const message = this.elements.messageInput.value.trim();
        if (!message || this.state.isProcessing) return;

        this.app.log('Sending message', { message, isStreaming: this.state.isStreaming });

        // Update UI state
        this.state.isProcessing = true;
        this.elements.sendBtn.disabled = true;
        this.elements.sendBtn.innerHTML = '<span class="btn-text">Sending...</span>';
        this.updateConnectionStatus('Processing...', 'info');

        // Add user message
        this.addMessage(message, 'user');
        this.state.messageHistory.push({ role: 'user', content: message });
        this.elements.messageInput.value = '';
        this.autoResizeTextarea();

        try {
            if (this.state.isStreaming) {
                await this.sendStreamingMessage(message);
            } else {
                await this.sendNonStreamingMessage(message);
            }
        } catch (error) {
            this.app.log('Message sending failed', error.message, 'error');
            this.removeTypingIndicator();
            this.addMessage(`❌ Error: ${error.message}`, 'error');
            this.app.showNotification('Failed to send message', 'error');
        } finally {
            // Reset UI state
            this.state.isProcessing = false;
            this.elements.sendBtn.disabled = false;
            this.elements.sendBtn.innerHTML = '<span class="btn-text">Send</span><span class="btn-icon">📤</span>';
            this.updateConnectionStatus('Connected', 'success');
            this.elements.messageInput.focus();
        }
    }

    async sendStreamingMessage(message) {
        const startTime = Date.now();

        this.addTypingIndicator();

        try {
            // Use /api/chat/message endpoint
            const response = await fetch(`${this.app.apiBaseUrl}api/chat/message?user_id=${this.app.userId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/plain'
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.state.currentConversationId,
                    stream: true
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            this.removeTypingIndicator();
            this.currentStreamingMessage = this.addMessage('', 'assistant', true);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';
            let chunkCount = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const dataStr = line.slice(6).trim();
                            if (!dataStr) continue;

                            const data = JSON.parse(dataStr);

                            if (data.content) {
                                fullResponse += data.content;
                                chunkCount++;

                                this.currentStreamingMessage.innerHTML =
                                    this.formatMessage(fullResponse) + '<span class="cursor">|</span>';
                                this.scrollToBottom();
                            }

                            if (data.is_final) {
                                this.currentStreamingMessage.classList.remove('streaming');
                                this.currentStreamingMessage.innerHTML = this.formatMessage(fullResponse);

                                this.addMessageStats(this.currentStreamingMessage, {
                                    processingTime: (Date.now() - startTime) / 1000,
                                    chunkCount: chunkCount,
                                    tokenCount: data.metadata?.token_count
                                });

                                this.state.messageHistory.push({ role: 'assistant', content: fullResponse });
                                this.scrollToBottom();
                                return;
                            }
                        } catch (e) {
                            this.app.log('Failed to parse streaming data', e.message, 'error');
                        }
                    }
                }
            }
        } catch (error) {
            this.removeTypingIndicator();
            throw error;
        }
    }

    async sendNonStreamingMessage(message) {
        this.addTypingIndicator();

        try {
            // Use /api/chat/message endpoint
            const data = await this.app.apiRequest('/chat/message', {
                method: 'POST',
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.state.currentConversationId,
                    stream: false
                }),
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            // Add query parameter for user_id
            const url = new URL(`${this.app.apiBaseUrl}/api/chat/message`);
            url.searchParams.append('user_id', this.app.userId);

            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.state.currentConversationId,
                    stream: false
                })
            });

            this.removeTypingIndicator();

            if (response.ok) {
                const data = await response.json();

                const messageEl = this.addMessage(data.response, 'assistant');

                this.addMessageStats(messageEl, {
                    processingTime: data.processing_time,
                    tokenCount: data.token_count
                });

                this.state.messageHistory.push({ role: 'assistant', content: data.response });
                this.state.currentConversationId = data.conversation_id;

            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || response.statusText);
            }
        } catch (error) {
            this.removeTypingIndicator();
            throw error;
        }
    }

    addMessageStats(messageEl, stats) {
        if (!messageEl) return;

        const statsEl = document.createElement('div');
        statsEl.className = 'message-stats';

        let statsHtml = `⚡ ${stats.processingTime.toFixed(2)}s`;
        if (stats.tokenCount) {
            statsHtml += ` | 📝 ${stats.tokenCount} tokens`;
        }
        if (stats.chunkCount) {
            statsHtml += ` | 📦 ${stats.chunkCount} chunks`;
        }

        statsEl.innerHTML = statsHtml;
        messageEl.appendChild(statsEl);
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the chat?')) {
            if (this.elements.chatMessages) {
                this.elements.chatMessages.innerHTML = '';
            }
            this.state.messageHistory = [];
            this.state.currentConversationId = null;
            this.addWelcomeMessage();
            this.app.log('Chat cleared');
        }
    }

    openSettings() {
        if (this.elements.settingsModal) {
            this.elements.settingsModal.classList.add('active');
        }
    }

    closeSettings() {
        if (this.elements.settingsModal) {
            this.elements.settingsModal.classList.remove('active');
        }
    }

    // Additional API methods using /api routes
    async getConversations() {
        try {
            return await this.app.apiRequest(`/chat/conversations?user_id=${this.app.userId}`);
        } catch (error) {
            this.app.log('Failed to get conversations', error.message, 'error');
            throw error;
        }
    }

    async searchConversations(query) {
        try {
            return await this.app.apiRequest(`/chat/search?q=${encodeURIComponent(query)}&user_id=${this.app.userId}`);
        } catch (error) {
            this.app.log('Failed to search conversations', error.message, 'error');
            throw error;
        }
    }
}

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('chatMessages')) {
        window.fridayChat = new FridayChat();
    }
});

console.log('🚀 FRIDAY Chat Module Loaded Successfully with /api routes!');
