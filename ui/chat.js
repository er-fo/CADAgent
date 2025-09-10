/**
 * CADAgent AI CAD Chat Interface JavaScript
 * Following Fusion 360 best practices for Qt browser compatibility
 * 
 * Key requirements:
 * - Handle both Promise and synchronous returns from adsk.fusionSendData()
 * - Implement window.fusionJavaScriptHandler for Python → HTML communication
 * - No Node.js/Electron APIs (pure browser environment)
 */

// Global state
let isConnected = false;
let currentApiKey = null;
let currentModelId = null;
let currentParameters = [];
let isProcessing = false;
let pendingApiKey = null;

// Parameter update debouncing
let parameterUpdateTimeout = null;
const PARAMETER_UPDATE_DELAY = 300; // 300ms debounce

// Backend configuration - will be set by Python init message
let BACKEND_CONFIG = {
    // Default fallback to remote server
    baseUrl: 'https://spacecad.fly.dev',
    endpoints: {
        status: '/api/v1/status',
        generate: '/api/v1/direct/generate',
        iterate: '/api/v1/direct/iterate',
        parameters: '/api/v1/parameters'
    },
    timeout: 60000 // 60 seconds
};

// Utility: Generate idempotency keys for POST/PUT requests (backend best practice)
function generateIdempotencyKey() {
    try {
        if (window.crypto && typeof window.crypto.randomUUID === 'function') {
            return window.crypto.randomUUID();
        }
        // Fallback UUID v4-ish
        const s4 = () => Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
        return `${s4()}${s4()}-${s4()}-${s4()}-${s4()}-${s4()}${s4()}${s4()}`;
    } catch (_) {
        return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
}

// Utility: Create compatible abort signal for older browsers (Fusion 360 compatibility)
function createTimeoutSignal(timeoutMs) {
    try {
        // Use modern AbortSignal.timeout if available
        if (typeof AbortSignal?.timeout === 'function') {
            return AbortSignal.timeout(timeoutMs);
        }
        
        // Fallback for older browsers
        const controller = new AbortController();
        setTimeout(() => controller.abort(), timeoutMs);
        return controller.signal;
    } catch (_) {
        // Ultimate fallback - return undefined (no timeout)
        console.warn('CADAgent: AbortController not available, requests may not timeout');
        return undefined;
    }
}

// Global error logging
window.addEventListener('error', function(e){
    try { console.error('CADAgent: Uncaught error', e.error || e.message || e); } catch {}
});
window.addEventListener('unhandledrejection', function(e){
    try { console.error('CADAgent: Unhandled promise rejection', e.reason || e); } catch {}
});

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('CADAgent: Initializing chat interface...');
    initializeApp();
    // Notify Python that HTML is ready
    try {
        if (window.adsk && typeof window.adsk.fusionSendData === 'function') {
            sendToFusion('html_ready', { ts: Date.now() })
              .then(r => console.log('CADAgent: html_ready ack:', r))
              .catch(e => console.warn('CADAgent: html_ready failed:', e));
        }
    } catch (e) {
        console.warn('CADAgent: html_ready exception:', e);
    }
});


/**
 * Initialize the application
 */
function initializeApp() {
    // Poll for Fusion 360 communication bridge (adsk injection can race)
    let pollAttempts = 0;
    const pollMax = 40; // ~12s total
    const pollDelay = 300;
    const poll = () => {
        const ready = !!(window.adsk && typeof window.adsk.fusionSendData === 'function');
        if (ready) {
            console.log('CADAgent: Bridge detected during poll at attempt', pollAttempts);
            checkFusionBridge();
        } else if (pollAttempts < pollMax) {
            pollAttempts += 1;
            setTimeout(poll, pollDelay);
        } else {
            console.warn('CADAgent: Bridge not detected after polling');
            checkFusionBridge(); // final attempt will set error state
        }
    };
    poll();
    
    // Initialize dark mode
    initializeDarkMode();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load saved API key
    loadApiKey();
    
    // Ensure the UI is docked to the right by default (remove undocked class)
    try {
        const appEl = document.querySelector('.space-app');
        if (appEl) {
            appEl.classList.remove('undocked');
            // Ensure expected fixed positioning in case host resets styles
            if (!appEl.style.position) appEl.style.position = 'fixed';
            appEl.style.right = '0';
            appEl.style.top = '0';
            appEl.style.bottom = '0';
        }
    } catch (e) {
        console.warn('CADAgent: failed to enforce docked layout', e);
    }

    console.log('CADAgent: Initialization complete');
}

/**
 * Check if we're running in Fusion 360 and test communication
 */
function checkFusionBridge() {
    // Following best practices: check for adsk object and fusionSendData function
    const canSendToFusion = !!(window.adsk && typeof window.adsk.fusionSendData === 'function');
    
    if (canSendToFusion) {
        console.log('CADAgent: Fusion 360 bridge detected');
        updateConnectionStatus('connecting');
        
        // Send ping to test communication with simple retry/backoff
        let attempts = 0;
        const maxAttempts = 10;
        const delayMs = 300;
        const tryPing = () => {
            attempts += 1;
        console.log('CADAgent: Sending ping attempt', attempts);
        sendToFusion('ping', { timestamp: Date.now(), attempt: attempts })
                .then(result => {
            console.log('CADAgent: Ping response:', result);
                    if (result === 'pong') {
                        updateConnectionStatus('connected');
                        console.log('CADAgent: Communication bridge established');
                    } else if (attempts < maxAttempts) {
                        setTimeout(tryPing, delayMs);
                    } else {
                        updateConnectionStatus('error');
                        console.error('CADAgent: Unexpected ping response after retries:', result);
                    }
                })
                .catch(err => {
                    if (attempts < maxAttempts) {
                        setTimeout(tryPing, delayMs);
                    } else {
                        updateConnectionStatus('error');
                        console.error('CADAgent: Bridge test failed after retries:', err);
                    }
                });
        };
        tryPing();
    } else {
        updateConnectionStatus('error');
        console.log('CADAgent: Not running in Fusion 360 environment');
    }
}

/**
 * Send data to Fusion Python backend
 * Handles both Promise and synchronous returns per best practices
 */
function sendToFusion(action, data) {
    return new Promise((resolve, reject) => {
        if (!window.adsk || typeof window.adsk.fusionSendData !== 'function') {
            reject(new Error('Fusion 360 communication not available'));
            return;
        }
        
        try {
            const payload = JSON.stringify(data);
            const result = window.adsk.fusionSendData(action, payload);
            
            // Handle both Promise (Qt browser) and synchronous (legacy) returns
            if (result && typeof result.then === 'function') {
                // Promise-based (Qt browser)
                result.then(resolve).catch(reject);
            } else {
                // Synchronous return (legacy CEF)
                resolve(result);
            }
        } catch (error) {
            reject(error);
        }
    });
}

/**
 * Show native Fusion 360 notification
 * Following best practices: Python handles UI dialogs
 */
function showNativeNotification(message, type = 'info') {
    return sendToFusion('show_notification', {
        message: message,
        type: type // 'info', 'warning', 'error'
    }).catch(err => {
        // Fallback to console if native notification fails
        console.log(`CADAgent: ${type.toUpperCase()}: ${message}`);
    });
}

/**
 * Set up event listeners for UI elements
 */
function setupEventListeners() {
    // API key management
    document.getElementById('saveApiKey').addEventListener('click', saveApiKey);
    document.getElementById('testApiKey').addEventListener('click', testApiKey);
    document.getElementById('apiKeyInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            saveApiKey();
        }
    });
    
    // Chat input
    document.getElementById('sendButton').addEventListener('click', sendMessage);
    document.getElementById('chatInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Dynamic input growth
    const chatInput = document.getElementById('chatInput');
    chatInput.addEventListener('input', handleInputChange);
    chatInput.addEventListener('focus', handleInputFocus);
    chatInput.addEventListener('blur', handleInputBlur);
    
    // Dark mode toggle
    document.getElementById('darkModeToggle').addEventListener('click', toggleDarkMode);
    
    document.getElementById('chatInput').addEventListener('input', function() {
        const button = document.getElementById('sendButton');
        const hasText = this.value.trim().length > 0;
        button.disabled = !hasText || isProcessing;
    });
    
    // Parameter updates
    document.getElementById('updateParameters').addEventListener('click', updateParameters);
    
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(status) {
    const statusElement = document.getElementById('connectionStatus');
    const dot = statusElement.querySelector('.status-dot');
    const text = statusElement.querySelector('.status-text');
    
    dot.className = 'status-dot';
    
    switch (status) {
        case 'connecting':
            text.textContent = 'connecting...';
            break;
        case 'connected':
            isConnected = true;
            dot.classList.add('connected');
            text.textContent = 'connected';
            break;
        case 'error':
            isConnected = false;
            dot.classList.add('error');
            text.textContent = 'connection error';
            break;
    }
}

/**
 * Toggle section visibility
 */
function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    section.classList.toggle('collapsed');
}

/**
 * Handle dynamic input container growth
 */
function handleInputChange() {
    const input = document.getElementById('chatInput');
    const container = document.getElementById('inputContainer');
    
    // Auto-resize the textarea based on content (no height limit)
    input.style.height = 'auto';
    input.style.height = input.scrollHeight + 'px';
    
    // Update send button state
    const button = document.getElementById('sendButton');
    const hasText = input.value.trim().length > 0;
    button.disabled = !hasText || isProcessing;
}

function handleInputFocus() {
    const container = document.getElementById('inputContainer');
    container.classList.add('focused');
}

function handleInputBlur() {
    const container = document.getElementById('inputContainer');
    container.classList.remove('focused');
}

/**
 * Dark Mode Functions
 */
function toggleDarkMode() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    
    // Save preference in localStorage
    localStorage.setItem('cadagent-theme', newTheme);
    
    console.log(`CADAgent: Switched to ${newTheme} mode`);
}

function initializeDarkMode() {
    // Check for saved theme preference or default to 'light'
    const savedTheme = localStorage.getItem('cadagent-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    console.log(`CADAgent: Initialized with ${savedTheme} mode`);
}

/**
 * Add progress message with animated states
 */
function addProgressMessage() {
    const messagesDiv = document.getElementById('chatMessages');
    const mainContent = document.querySelector('.main-content');
    
    // Create progress message element
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message progress fade-in';
    messageDiv.id = 'currentProgressMessage';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    
    const iconDiv = document.createElement('div');
    iconDiv.className = 'progress-icon';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'progress-text';
    textDiv.textContent = 'processing';
    
    contentDiv.appendChild(iconDiv);
    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(contentDiv);
    
    messagesDiv.appendChild(messageDiv);
    
    // Hide welcome message when first message is added
    mainContent.classList.add('has-messages');
    
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    // Start progress animation sequence
    updateProgressState('processing');
    
    return messageDiv;
}

/**
 * Update progress message state with animations
 */
function updateProgressState(state) {
    const progressMessage = document.getElementById('currentProgressMessage');
    if (!progressMessage) return;
    
    const iconDiv = progressMessage.querySelector('.progress-icon');
    const textDiv = progressMessage.querySelector('.progress-text');
    
    // Update text
    textDiv.textContent = state;
    
    // Update icon based on state
    switch(state) {
        case 'processing':
        case 'planning':
        case 'generating':
            // Show loading ring
            iconDiv.innerHTML = '<svg class="progress-icon"><use href="#icon-loading"/></svg>';
            break;
        case 'completed':
            // Show checkmark and collapse
            iconDiv.innerHTML = '<svg class="progress-icon"><use href="#icon-checkmark"/></svg>';
            progressMessage.classList.add('completed');
            break;
    }
}

/**
 * Complete progress message and collapse
 */
function completeProgressMessage() {
    updateProgressState('completed');
    
    // Remove the progress message after animation
    setTimeout(() => {
        const progressMessage = document.getElementById('currentProgressMessage');
        if (progressMessage) {
            progressMessage.remove();
        }
    }, 2000);
}

/**
 * Save API key - store in cache and test with backend
 */
function saveApiKey() {
    const input = document.getElementById('apiKeyInput');
    const key = input.value.trim();
    
    if (!key) {
        showNativeNotification('Please enter your API key to continue.', 'warning');
        return;
    }
    
    const saveButton = document.getElementById('saveApiKey');
    saveButton.classList.add('loading');
    saveButton.disabled = true;
    
    // Kick off async validation via Python; await api_validation_result event
    pendingApiKey = key;
    sendToFusion('validate_api_key', { api_key: key })
        .then(() => { /* ACK received; wait for api_validation_result */ })
        .catch(err => {
            showNativeNotification('Error starting validation: ' + err.message, 'error');
            pendingApiKey = null;
        })
        .finally(() => {
            // Keep button disabled until we get a result event for better UX
            setTimeout(() => { saveButton.classList.remove('loading'); /* keep disabled state handled by event */ }, 200);
        });
}

/**
 * Test API key using Python bridge
 * Following Fusion 360 best practices: Python handles network requests
 */
function testApiKey() {
    const input = document.getElementById('apiKeyInput');
    const key = input.value.trim();
    
    if (!key) {
        showNativeNotification('Please enter your API key first.', 'warning');
        return;
    }
    
    const testButton = document.getElementById('testApiKey');
    testButton.classList.add('loading');
    testButton.disabled = true;
    
    pendingApiKey = key;
    sendToFusion('validate_api_key', { api_key: key })
        .then(() => { /* ACK received; wait for api_validation_result */ })
        .catch(err => {
            console.error('CADAgent: Backend test start error:', err);
            showNativeNotification('API key verification error: ' + err.message, 'error');
            pendingApiKey = null;
        })
        .finally(() => {
            setTimeout(() => { testButton.classList.remove('loading'); /* keep disabled handled by event */ }, 200);
        });
}

/**
 * Load API key from cache
 */
function loadApiKey() {
    console.log('CADAgent: Loading cached API key...');
    
    sendToFusion('get_cached_api_key', {})
        .then(result => {
            if (result && result.success && result.has_cached_key && result.api_key) {
                console.log('CADAgent: Found cached API key');
                
                // Set the cached key
                currentApiKey = result.api_key;
                
                // Update the input field
                const input = document.getElementById('apiKeyInput');
                if (input) {
                    input.value = result.api_key;
                }
                
                // Enable send button if chat input has text
                const chatInput = document.getElementById('chatInput');
                const sendButton = document.getElementById('sendButton');
                if (chatInput && sendButton) {
                    sendButton.disabled = !chatInput.value.trim() || isProcessing;
                }
                
                // Collapse API key section since we have a cached key
                const apiKeySection = document.getElementById('apiKeySection');
                if (apiKeySection) {
                    apiKeySection.classList.add('collapsed');
                }
                
                // Show native notification instead of chat message
                showNativeNotification('Cached API key loaded successfully. Ready to create models.', 'info');
            } else {
                console.log('CADAgent: No cached API key found');
                // Show API key section if no cached key
                const apiKeySection = document.getElementById('apiKeySection');
                if (apiKeySection) {
                    apiKeySection.classList.remove('collapsed');
                }
            }
        })
        .catch(err => {
            console.warn('CADAgent: Error loading cached API key:', err);
            // Show API key section on error
            const apiKeySection = document.getElementById('apiKeySection');
            if (apiKeySection) {
                apiKeySection.classList.remove('collapsed');
            }
        });
}

/**
 * Send chat message - using direct backend API calls with sleek progress animation
 * Following Fusion 360 best practices: JavaScript handles network, Python handles Fusion API
 */
function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message || isProcessing) return;
    
    // Add user message to chat
    addMessage('user', message);
    
    // Clear input and reset height
    input.value = '';
    input.style.height = 'auto';
    
    // Set processing state
    setProcessing(true);
    
    // Add progress message with animation
    addProgressMessage();
    
    // Start progress sequence
    setTimeout(() => updateProgressState('planning'), 5000);
    setTimeout(() => updateProgressState('generating'), 10000);
    
    // Simple rule: if model exists, use iteration
    const isIteration = currentModelId !== null;
    
    if (isIteration) {
        // Call iterate using Python bridge (proper Fusion 360 pattern)
        sendToFusion('iterate_model', {
            model_id: currentModelId,
            prompt: message,
            anthropic_api_key: currentApiKey || ''  // Let Python handle .env fallback
        })
            .then(result => {
                // Expect async ACK; final result will arrive via iteration_complete event
                try {
                    const response = JSON.parse(result);
                    if (!response || response.processing !== true) {
                        // Back-compat path if backend still returns final result
                        if (response && response.success && !response.processing) {
                            handleSuccessfulGeneration(response, true);
                        }
                    }
                } catch (_) { /* ignore */ }
            })
            .catch(err => {
                completeProgressMessage();
                addMessage('assistant', 'Model iteration error: ' + err.message);
                setProcessing(false);
            });
    } else {
        // Call generate using Python bridge (proper Fusion 360 pattern)
        sendToFusion('generate_model', {
            prompt: message,
            anthropic_api_key: currentApiKey || ''  // Let Python handle .env fallback
        })
            .then(result => {
                // Expect async ACK; final result will arrive via generation_complete event
                try {
                    const response = JSON.parse(result);
                    if (!response || response.processing !== true) {
                        // Back-compat path if backend still returns final result
                        if (response && response.success && !response.processing) {
                            handleSuccessfulGeneration(response, false);
                        }
                    }
                } catch (_) { /* ignore */ }
            })
            .catch(err => {
                completeProgressMessage();
                addMessage('assistant', 'Model generation error: ' + err.message);
                setProcessing(false);
            });
    }
}

/**
 * Debounced parameter update function
 */
function debouncedParameterUpdate() {
    // Clear existing timeout
    if (parameterUpdateTimeout) {
        clearTimeout(parameterUpdateTimeout);
    }
    
    // Set new timeout
    parameterUpdateTimeout = setTimeout(() => {
        updateParameters();
    }, PARAMETER_UPDATE_DELAY);
}

/**
 * Update model parameters (enhanced version)
 */
function updateParameters() {
    if (!currentModelId || isProcessing) return;

    // Collect parameter values from enhanced parameter items
    const updates = [];
    const parameterItems = document.querySelectorAll('.parameter-item');

    parameterItems.forEach(item => {
        const paramId = item.getAttribute('data-param-id');
        const slider = item.querySelector('.parameter-slider');
        const numericInput = item.querySelector('.parameter-numeric-input');
        
        if (slider && numericInput && paramId) {
            updates.push({
                id: paramId,
                value: parseFloat(slider.value)
            });
        }
    });

    if (updates.length === 0) {
        console.warn('CADAgent: No parameters to update');
        return;
    }

    // Visual feedback - mark parameters as updating
    parameterItems.forEach(item => {
        item.classList.add('parameter-updating');
    });

    setProcessing(true);

    // Send parameter updates to backend using Python bridge (following Fusion 360 best practices)
    sendToFusion('update_parameters', { 
        model_id: currentModelId, 
        updates: updates 
    })
        .then(result => {
            // Parse response from Python bridge
            const response = typeof result === 'string' ? JSON.parse(result) : result;
            
            // Clean up visual feedback
            const cleanupUpdatingState = () => {
                document.querySelectorAll('.parameter-item').forEach(item => {
                    item.classList.remove('parameter-updating');
                });
            };
            
            if (!response.success) {
                addMessage('assistant', 'failed to update parameters: ' + (response.error || 'unknown error'));
                cleanupUpdatingState();
                setProcessing(false);
                return;
            }
            // Import updated STEP if present
            if (response.step_file) {
                sendToFusion('import_step_file', {
                    step_file_data: response.step_file,
                    model_id: response.model_id || currentModelId,
                    is_iteration: true
                }).then(r => {
                    const importResult = JSON.parse(r);
                    if (importResult.success) {
                        const replacedText = importResult.model_replaced ? ' (previous model replaced)' : '';
                        addMessage('assistant', 'parameters updated and model re-imported' + replacedText);
                    } else {
                        addMessage('assistant', 'parameters updated but import failed: ' + (importResult.error || 'unknown error'));
                    }
                    cleanupUpdatingState();
                    setProcessing(false);
                }).catch(e => {
                    addMessage('assistant', 'parameters updated but import failed: ' + e.message);
                    cleanupUpdatingState();
                    setProcessing(false);
                });
            } else if (response.location) {
                // Fetch STEP from presigned URL then import
                fetchStepAsBase64(response.location)
                    .then(b64 => sendToFusion('import_step_file', {
                        step_file_data: b64,
                        model_id: response.model_id || currentModelId,
                        is_iteration: true
                    }))
                    .then(r => {
                        const importResult = JSON.parse(r);
                        if (importResult.success) {
                            const replacedText = importResult.model_replaced ? ' (previous model replaced)' : '';
                            addMessage('assistant', 'parameters updated and model re-imported' + replacedText);
                        } else {
                            addMessage('assistant', 'parameters updated but import failed: ' + (importResult.error || 'unknown error'));
                        }
                        cleanupUpdatingState();
                        setProcessing(false);
                    })
                    .catch(e => {
                        addMessage('assistant', 'failed to download STEP for updated parameters: ' + e.message);
                        cleanupUpdatingState();
                        setProcessing(false);
                    });
            } else {
                addMessage('assistant', 'parameters updated');
                cleanupUpdatingState();
                setProcessing(false);
            }
        })
        .catch(err => {
            addMessage('assistant', 'error updating parameters: ' + err.message);
            document.querySelectorAll('.parameter-item').forEach(item => {
                item.classList.remove('parameter-updating');
            });
            setProcessing(false);
        });
}

/**
 * Add message to chat
 */
function addMessage(type, content) {
    const messagesContainer = document.getElementById('chatMessages');
    const mainContent = document.querySelector('.main-content');
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${type}`;
    
    const contentElement = document.createElement('div');
    contentElement.className = 'content';
    contentElement.textContent = content;
    
    messageElement.appendChild(contentElement);
    messagesContainer.appendChild(messageElement);
    
    // Hide welcome message when first message is added
    mainContent.classList.add('has-messages');
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Set processing state
 */
function setProcessing(processing) {
    isProcessing = processing;
    
    const sendButton = document.getElementById('sendButton');
    const chatInput = document.getElementById('chatInput');
    const updateButton = document.getElementById('updateParameters');
    
    sendButton.disabled = processing || !chatInput.value.trim();
    chatInput.disabled = processing;
    updateButton.disabled = processing || !currentModelId;
    
    if (processing) {
        sendButton.classList.add('loading');
        updateButton.classList.add('loading');
    } else {
        sendButton.classList.remove('loading');
        updateButton.classList.remove('loading');
    }
}

// ========================================
// Backend API Functions (JavaScript → Python Bridge → Backend)
// Following Fusion 360 best practices: Python handles network requests
// ========================================

// Note: Direct HTTP functions removed - now using Python bridge pattern following Fusion 360 best practices

/**
 * Handle successful generation/iteration response
 * Pass STEP file data to Python for Fusion API operations
 */
function handleSuccessfulGeneration(response, isIteration) {
    try {
        // Store model ID
        if (response.model_id) {
            currentModelId = response.model_id;
        }
        
        // Generate parameter sliders if provided, otherwise auto-extract
        if (response.parameters) {
            generateParameterSliders(response.parameters);
        } else if (response.model_id) {
            // Auto-extract parameters after model generation
            autoExtractParameters(response.model_id);
        }
        
    // Check if we have STEP file data to import
    if (response.step_file) {
            addMessage('assistant', 'model generated successfully. importing into fusion 360...');
            
            // Send STEP file data to Python for Fusion API import
            // This follows best practices: Python handles Fusion API, JavaScript handles network
            sendToFusion('import_step_file', {
                step_file_data: response.step_file,
                model_id: response.model_id, // Use real model_id from backend, no fallback
                is_iteration: isIteration
            })
            .then(result => {
                const importResult = JSON.parse(result);
                if (importResult.success) {
                    const replacedText = importResult.model_replaced ? ' (previous model replaced)' : '';
                    addMessage('assistant', 'model imported successfully into fusion 360!' + replacedText);
                } else {
                    addMessage('assistant', 'model generated but import failed: ' + (importResult.error || 'unknown error'));
                }
                setProcessing(false);
            })
            .catch(err => {
                addMessage('assistant', 'model generated but import failed: ' + err.message);
                setProcessing(false);
            });
        } else if (response.location) {
            // Presigned URL case: fetch STEP then import
            addMessage('assistant', 'model generated. downloading step and importing...');
            fetchStepAsBase64(response.location)
                .then(b64 => sendToFusion('import_step_file', {
                    step_file_data: b64,
                    model_id: response.model_id, // Use real model_id from backend, no fallback
                    is_iteration: isIteration
                }))
                .then(result => {
                    const importResult = JSON.parse(result);
                    if (importResult.success) {
                        const replacedText = importResult.model_replaced ? ' (previous model replaced)' : '';
                        addMessage('assistant', 'model imported successfully into fusion 360!' + replacedText);
                    } else {
                        addMessage('assistant', 'model generated but import failed: ' + (importResult.error || 'unknown error'));
                    }
                    setProcessing(false);
                })
                .catch(err => {
                    addMessage('assistant', 'failed to download STEP: ' + err.message);
                    setProcessing(false);
                });
        } else {
            addMessage('assistant', isIteration ? 'model updated successfully' : 'model generated successfully');
            setProcessing(false);
        }
    } catch (error) {
        addMessage('assistant', 'error processing response: ' + error.message);
        setProcessing(false);
    }
}

// Download a STEP file from presigned URL and return base64 string
async function fetchStepAsBase64(url) {
    const res = await fetch(url, { method: 'GET', signal: AbortSignal.timeout(BACKEND_CONFIG.timeout) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    // Convert ArrayBuffer to base64
    let binary = '';
    const bytes = new Uint8Array(buf);
    const chunkSize = 0x8000; // avoid call stack limits
    for (let i = 0; i < bytes.length; i += chunkSize) {
        const chunk = bytes.subarray(i, i + chunkSize);
        binary += String.fromCharCode.apply(null, chunk);
    }
    // btoa expects binary string
    return btoa(binary);
}

/**
 * Generate enhanced parameter sliders from API response with smart filtering and ranges
 */
function generateParameterSliders(parameters) {
    const container = document.getElementById('parameterControls');
    container.innerHTML = '';
    
    currentParameters = parameters || [];
    
    // Filter out position-related parameters (origin, center, position)
    const filteredParameters = currentParameters.filter(param => {
        const pathLower = (param.path || '').toLowerCase();
        const nameLower = (param.name || '').toLowerCase();
        return !pathLower.includes('origin') && 
               !pathLower.includes('position') && 
               !pathLower.includes('center') &&
               !nameLower.includes('origin') &&
               !nameLower.includes('position') &&
               !nameLower.includes('center');
    });
    
    if (filteredParameters.length === 0) {
        const parametersSection = document.getElementById('parametersSection');
        parametersSection.classList.add('collapsed');
        return;
    }
    
    // Generate enhanced sliders for each filtered parameter
    filteredParameters.forEach(param => {
        const controlDiv = document.createElement('div');
        controlDiv.className = 'parameter-item';
        controlDiv.setAttribute('data-param-id', param.id);
        
        // Parameter header with name and current value
        const headerDiv = document.createElement('div');
        headerDiv.className = 'parameter-header';
        
        const labelSpan = document.createElement('span');
        labelSpan.className = 'parameter-label';
        labelSpan.textContent = param.name || 'Parameter';
        
        const currentValueSpan = document.createElement('span');
        currentValueSpan.className = 'parameter-current-value';
        const currentValue = param.value || param.default_value || 0;
        currentValueSpan.textContent = `${currentValue}${param.unit || ''}`;
        
        headerDiv.appendChild(labelSpan);
        headerDiv.appendChild(currentValueSpan);
        
        // Controls row with slider and numeric input
        const controlsRow = document.createElement('div');
        controlsRow.className = 'parameter-controls-row';
        
        // Slider container
        const sliderContainer = document.createElement('div');
        sliderContainer.className = 'parameter-slider-container';
        
        // Smart range calculation: 0 to 2x current value (current value in middle)
        const smartMin = 0;
        const smartMax = Math.max(currentValue * 2, currentValue + 100); // Ensure reasonable range
        const smartStep = param.step || calculateSmartStep(currentValue);
        
        // Create slider
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'parameter-slider';
        slider.setAttribute('data-param-id', param.id);
        slider.min = smartMin;
        slider.max = smartMax;
        slider.step = smartStep;
        slider.value = currentValue;
        
        // Create numeric input
        const numericInput = document.createElement('input');
        numericInput.type = 'number';
        numericInput.className = 'parameter-numeric-input';
        numericInput.setAttribute('data-param-id', param.id);
        numericInput.min = smartMin;
        numericInput.max = smartMax;
        numericInput.step = smartStep;
        numericInput.value = currentValue;
        
        // Range labels
        const rangeLabels = document.createElement('div');
        rangeLabels.className = 'parameter-range-labels';
        rangeLabels.innerHTML = `<span>${smartMin}${param.unit || ''}</span><span>${smartMax}${param.unit || ''}</span>`;
        
        // Bidirectional sync between slider and numeric input
        slider.addEventListener('input', function() {
            const value = parseFloat(this.value);
            numericInput.value = value;
            currentValueSpan.textContent = `${value}${param.unit || ''}`;
            debouncedParameterUpdate();
        });
        
        numericInput.addEventListener('input', function() {
            const value = parseFloat(this.value);
            if (!isNaN(value) && value >= smartMin && value <= smartMax) {
                slider.value = value;
                currentValueSpan.textContent = `${value}${param.unit || ''}`;
                debouncedParameterUpdate();
            }
        });
        
        // Assemble slider container
        sliderContainer.appendChild(slider);
        sliderContainer.appendChild(rangeLabels);
        
        // Assemble controls row
        controlsRow.appendChild(sliderContainer);
        controlsRow.appendChild(numericInput);
        
        // Assemble parameter item
        controlDiv.appendChild(headerDiv);
        controlDiv.appendChild(controlsRow);
        
        container.appendChild(controlDiv);
    });
    
    // Show parameters section and expand it (same as API key section)
    const parametersSection = document.getElementById('parametersSection');
    parametersSection.style.display = 'block';
    parametersSection.classList.remove('collapsed');
}

/**
 * Calculate smart step size based on parameter value
 */
function calculateSmartStep(value) {
    const absValue = Math.abs(value);
    if (absValue >= 100) return 5;
    if (absValue >= 10) return 1;
    if (absValue >= 1) return 0.1;
    return 0.01;
}

/**
 * Auto-extract parameters from backend after model generation using Python bridge
 */
function autoExtractParameters(modelId) {
    if (!modelId) return;
    
    console.log('CADAgent: Auto-extracting parameters for model:', modelId);
    
    // Use Python bridge to call deployment backend
    sendToFusion('get_parameters', { model_id: modelId })
        .then(result => {
            const data = typeof result === 'string' ? JSON.parse(result) : result;
            if (data && data.success && data.parameters && data.parameters.length > 0) {
                console.log(`CADAgent: Auto-extracted ${data.parameters.length} parameters`);
                generateParameterSliders(data.parameters);
            } else {
                console.log('CADAgent: No parameters found for auto-extraction');
            }
        })
        .catch(error => {
            console.warn('CADAgent: Failed to auto-extract parameters:', error.message);
            // Don't show error to user - parameter extraction is optional
        });
}


/**
 * Handle incoming messages from Python (Fusion → HTML communication)
 * Following best practices for window.fusionJavaScriptHandler
 */
window.fusionJavaScriptHandler = {
    handle: function(action, data) {
    console.log('CADAgent: Received from Python:', action, data);
        
        try {
            switch (action) {
                case 'api_validation_result':
                    const val = JSON.parse(data);
                    if (val && val.success) {
                        // Promote pending key
                        if (pendingApiKey) {
                            currentApiKey = pendingApiKey;
                        }
                        // Cache in Python
                        sendToFusion('store_api_key', { api_key: currentApiKey || '' })
                          .then(() => {
                              // Collapse API key section and notify
                              document.getElementById('apiKeySection').classList.add('collapsed');
                              showNativeNotification('API key saved and verified successfully. You can now create models.', 'info');
                              // Enable send button if chat input has text
                              const chatInput = document.getElementById('chatInput');
                              const sendButton = document.getElementById('sendButton');
                              sendButton.disabled = !chatInput.value.trim() || isProcessing;
                          })
                          .catch(() => {})
                          .finally(() => { pendingApiKey = null; });
                    } else {
                        showNativeNotification('API key verification failed: ' + (val && (val.error || val.message) || 'unknown error'), 'error');
                        pendingApiKey = null;
                    }
                    return 'OK';
                case 'init':
                    // Handle initialization message from Python
                    const initData = JSON.parse(data);
                    console.log('CADAgent: Received init message:', initData);
                    
                    // Update backend configuration if provided
                    if (initData.backend_url) {
                        BACKEND_CONFIG.baseUrl = initData.backend_url;
                        console.log('CADAgent: Backend URL updated to:', initData.backend_url);
                    }
                    
                    // Don't add system message to chat - keep chat clean for actual conversations
                    // Mark connection as established on init as well
                    updateConnectionStatus('connected');
                    return 'INIT_OK';
                    
                case 'generation_complete':
                    // Handle completed generation
                    const genData = JSON.parse(data);
                    addMessage('assistant', 'model generated successfully');
                    if (genData.model_id) currentModelId = genData.model_id;
                    if (genData.parameters) {
                        generateParameterSliders(genData.parameters);
                    } else if (genData.model_id) {
                        // Auto-extract parameters if not provided in response
                        autoExtractParameters(genData.model_id);
                    }
                    // Mark progress as completed
                    try { completeProgressMessage(); } catch(_) {}
                    setProcessing(false);
                    return 'OK';
                    
                case 'step_ready_for_import':
                    // Handle STEP data ready for import (event-driven pattern)
                    const stepData = JSON.parse(data);
                    console.log('CADAgent: STEP data ready for import, triggering main thread import...');
                    
                    // Send import action back to Python (which runs on main thread)
                    sendToFusion('step_import_from_background', stepData)
                        .then(response => {
                            console.log('CADAgent: STEP import response:', response);
                            // Import response is handled separately via generation_complete
                        })
                        .catch(error => {
                            console.error('CADAgent: STEP import failed:', error);
                            showNativeNotification('Import failed: ' + error.message, 'error');
                            try { completeProgressMessage(); } catch(_) {}
                            setProcessing(false);
                        });
                    return 'OK';
                    
                case 'iteration_complete':
                    // Handle completed iteration
                    const itData = JSON.parse(data);
                    if (itData.model_id) currentModelId = itData.model_id;
                    addMessage('assistant', 'model updated successfully');
                    if (itData.parameters) {
                        generateParameterSliders(itData.parameters);
                    } else if (itData.model_id) {
                        // Auto-extract parameters after iteration if not provided
                        autoExtractParameters(itData.model_id);
                    }
                    try { completeProgressMessage(); } catch(_) {}
                    setProcessing(false);
                    return 'OK';
                    
                case 'parameters_updated':
                    // Handle parameter update completion
                    const upData = JSON.parse(data);
                    addMessage('assistant', 'parameters updated successfully');
                    setProcessing(false);
                    return 'OK';
                    
                case 'error':
                    // Handle error from Python - show as native notification
                    const errorData = JSON.parse(data);
                    showNativeNotification('Error: ' + (errorData.message || errorData.error || 'unknown error'), 'error');
                    try { completeProgressMessage(); } catch(_) {}
                    setProcessing(false);
                    return 'OK';
                    
                default:
                    console.warn('CADAgent: Unknown action from Python:', action);
                    return 'UNKNOWN_ACTION';
            }
        } catch (error) {
            console.error('CADAgent: Error handling Python message:', error);
            return 'ERROR';
        }
    }
};

// ========================================
// Parameters Backend API (PUT)
// ========================================

async function callBackendUpdateParameters(modelId, updates, apiKey) {
    try {
        const endpoint = `${BACKEND_CONFIG.endpoints.parameters}/${encodeURIComponent(modelId)}`;
        const response = await fetch(`${BACKEND_CONFIG.baseUrl}${endpoint}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'CADAgent-Fusion-Addon/1.0',
                'Idempotency-Key': generateIdempotencyKey()
            },
            body: JSON.stringify({
                updates: updates,
                anthropic_api_key: apiKey
            }),
            signal: AbortSignal.timeout(BACKEND_CONFIG.timeout)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        return { success: true, ...data };
    } catch (error) {
        return { success: false, error: error.message };
    }
}
