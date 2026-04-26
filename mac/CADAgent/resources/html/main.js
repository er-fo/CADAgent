/**
 * CADAgent Palette JavaScript
 * Handles UI interactions and communication with the add-in backend
 */

// Configure marked.js for safe markdown rendering (DEPRECATED - backend now sends HTML)
// Kept for backward compatibility and potential client-side use
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false
    });
}

/**
 * Safely render markdown to HTML with XSS protection
 * @param {string} text - Markdown text to render
 * @returns {string} Sanitized HTML
 */
function renderMarkdown(text) {
    if (!text || typeof text !== 'string') {
        return '';
    }

    // If marked is not available, return escaped plain text
    if (typeof marked === 'undefined') {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    try {
        // Parse markdown to HTML
        const html = marked.parse(text);

        // Basic XSS sanitization - remove script tags and event handlers
        const temp = document.createElement('div');
        temp.innerHTML = html;

        // Remove script tags
        const scripts = temp.querySelectorAll('script');
        scripts.forEach(script => script.remove());

        // Remove event handler attributes
        const allElements = temp.querySelectorAll('*');
        allElements.forEach(el => {
            Array.from(el.attributes).forEach(attr => {
                if (attr.name.startsWith('on')) {
                    el.removeAttribute(attr.name);
                }
            });
        });

        return temp.innerHTML;
    } catch (error) {
        console.error('Markdown rendering error:', error);
        // Fallback to escaped plain text
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// State management
// Guard against duplicate initialization/binding in case the palette HTML is injected twice
let initialized = false;
let listenersBound = false;

let state = {
    connected: false,
    sessionId: null,
    isAuthenticated: false,
    apiKeyStatus: {
        anthropic: false,
        openai: false
    },
    processing: false,
    cancelRequested: false,
    executeRequestCount: 0,
    visualContextEnabled: false,
    captureRunLogs: false,
    activeRun: null,
    selectedModel: 'gpt-5.4',
    reasoningEffort: 'medium',  // Default for OpenAI; 'off' for Anthropic
    reasoning: {
        activeSession: null,
        sessionCounter: 0
    },
    pendingUserMessage: null,  // Track last user message for checkpoint association
    pendingCheckpointQueue: [], // Checkpoint IDs waiting for matching DOM nodes
    requestCounter: 0,  // Counter for generating unique request IDs for checkpoint correlation
    loginStage: 'cta',
    testingMode: false,
    testingEmail: '',
    attachedImage: null  // {data: base64string, format: 'png'|'jpg'}
};

let otpFocused = false;

// Track run iteration counter per doc in memory (UI-only)
const runIterationCounters = new Map();

// Per-document state storage: docId -> structured thread state
const docStates = new Map();
let currentDocId = null;
const pendingMessagesByDoc = new Map();
const docOrder = [];

// Track consecutive substantive messages for different doc (for self-heal after reconnect)
let consecutiveMessagesForDifferentDoc = { docId: null, count: 0 };
const LOCAL_STORAGE_KEY = 'cadagent_docstate_v1';

function generateId(prefix = 'id') {
    return `${prefix}_${Date.now()}_${Math.floor(Math.random() * 100000)}`;
}

function migrateDefaultStateTo(targetDocId) {
    if (!targetDocId || targetDocId === 'default') return;
    const source = docStates.get('default');
    if (!source || (!source.items || source.items.length === 0)) return;
    const target = getDocState(targetDocId, true);
    if (!target) return;
    if (!target.items || target.items.length === 0) {
        target.items = source.items;
        target.activeRunId = source.activeRunId;
        target.lastStatus = source.lastStatus || target.lastStatus;
        target.lastUpdated = Date.now();
    }
    docStates.delete('default');
    const idx = docOrder.indexOf('default');
    if (idx >= 0) docOrder.splice(idx, 1);
    persistDocState(targetDocId);
}

function loadDocStatesFromStorage() {
    try {
        const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
        if (!raw) return;
        const data = JSON.parse(raw);
        if (Array.isArray(data)) {
            data.forEach((entry) => {
                if (entry && entry.id) {
                    docStates.set(entry.id, entry);
                    if (!docOrder.includes(entry.id)) {
                        docOrder.push(entry.id);
                    }
                }
            });
        }
    } catch (e) {
        console.warn('Failed to load cached doc state', e);
    }
}

function persistDocState(docId) {
    if (!docId) return;
    const snapshot = Array.from(docStates.values());
    try {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(snapshot));
    } catch (e) {
        console.warn('Failed to persist doc state', e);
    }
}

function resolveDocId(docId) {
    return docId || currentDocId || 'default';
}

function getDocState(docId, create = true) {
    const id = resolveDocId(docId);
    if (!docStates.has(id) && create) {
        docStates.set(id, {
            id,
            name: 'Untitled',
            sessionId: null,
            connected: false,
            items: [],
            activeRunId: null,
            lastStatus: 'idle',
            lastUpdated: Date.now()
        });
        docOrder.push(id);
    }
    return docStates.get(id) || null;
}

function updateDocOrder(docId) {
    if (!docId) return;
    const idx = docOrder.indexOf(docId);
    if (idx >= 0) {
        docOrder.splice(idx, 1);
    }
    docOrder.unshift(docId);
}

function getActiveDocState() {
    return getDocState(currentDocId, false);
}

/**
 * Keep the global processing flag in sync with the active document.
 * Prevents the send button from staying in "cancel" mode when the user
 * switches to a different Fusion document that is not running.
 */
function syncProcessingState(docId = currentDocId) {
    const docState = getDocState(docId, false);
    const isProcessing = !!(docState && docState.lastStatus === 'in-progress');

    // If nothing changed, avoid redundant DOM updates
    if (state.processing === isProcessing) {
        return;
    }

    state.processing = isProcessing;

    if (!isProcessing) {
        // Ensure progress UI and controls reflect idle state for this doc
        hideProgress();
    } else {
        updateExecuteButtonState();
        updateRevertButtonStates();
    }
}

function isDocConnected(docId = currentDocId) {
    const docState = getDocState(docId, false);
    if (docState) return !!docState.connected;
    return !!state.connected; // legacy fallback
}

// DOM elements
const elements = {
    headerSection: document.querySelector('.header-section'),
    connectionStatus: document.getElementById('connectionStatus'),
    statusText: document.getElementById('statusText'),
    sessionId: document.getElementById('sessionId'),
    reconnectBtn: document.getElementById('reconnectBtn'),
    welcomeMessage: document.getElementById('welcomeMessage'),
    chatMessages: document.getElementById('chatMessages'),
    cadRequest: document.getElementById('cadRequest'),
    modelSelect: document.getElementById('modelSelect'),
    reasoningSelect: document.getElementById('reasoningSelect'),
    planningMode: document.getElementById('planningMode'),
    planningModeState: document.getElementById('planningModeState'),
    visualContextToggle: document.getElementById('visualContextToggle'),
    visualContextState: document.getElementById('visualContextState'),
    executeBtn: document.getElementById('executeBtn'),
    inlineReconnect: document.getElementById('inlineReconnect'),
    clearBtn: document.getElementById('clearBtn'),
    progressSection: document.getElementById('progressSection'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),
    toastContainer: document.getElementById('toastContainer'),
    // Auth elements
    loginBtn: document.getElementById('loginBtn'),
    logoutBtn: document.getElementById('logoutBtn'),
    userInfo: document.getElementById('userInfo'),
    loginModal: document.getElementById('loginModal'),
    loginForm: document.getElementById('loginForm'),
    emailInput: document.getElementById('emailInput'),
    cancelLoginBtn: document.getElementById('cancelLoginBtn'),
    sendOtpBtn: document.getElementById('sendOtpBtn'),
    verifyOtpBtn: document.getElementById('verifyOtpBtn'),
    otpSection: document.getElementById('otpSection'),
    otpInput: document.getElementById('otpInput'),
    otpBoxes: Array.from(document.querySelectorAll('.otp-visual span')),
    loginStatus: document.getElementById('loginStatus'),
    loginStartBtn: document.getElementById('loginStartBtn'),
    switchEmailBtn: document.getElementById('switchEmailBtn'),
    loginCta: document.getElementById('loginCta'),
    emailRow: document.getElementById('emailRow'),
    tosConsent: document.getElementById('tosConsent'),
    loginVersion: document.getElementById('loginVersion'),
    passwordRow: document.getElementById('passwordRow'),
    passwordInput: document.getElementById('passwordInput'),
    passwordLoginBtn: document.getElementById('passwordLoginBtn'),
    showPasswordLoginBtn: document.getElementById('showPasswordLoginBtn'),
    // API Keys elements
    apiKeysBtn: document.getElementById('apiKeysBtn'),
    apiKeysModal: document.getElementById('apiKeysModal'),
    apiKeysForm: document.getElementById('apiKeysForm'),
    anthropicKeyInput: document.getElementById('anthropicKeyInput'),
    openaiKeyInput: document.getElementById('openaiKeyInput'),
    toggleAnthropicKey: document.getElementById('toggleAnthropicKey'),
    toggleOpenaiKey: document.getElementById('toggleOpenaiKey'),
    anthropicKeyStatus: document.getElementById('anthropicKeyStatus'),
    openaiKeyStatus: document.getElementById('openaiKeyStatus'),
    apiKeysStatus: document.getElementById('apiKeysStatus'),
    apiKeysFooter: document.getElementById('apiKeysFooter'),
    closeApiKeysBtn: document.getElementById('closeApiKeysBtn'),
    saveApiKeysBtn: document.getElementById('saveApiKeysBtn'),
    // Image attachment elements
    attachImageBtn: document.getElementById('attachImageBtn'),
    imageInput: document.getElementById('imageInput'),
    imagePreview: document.getElementById('imagePreview'),
    previewImage: document.getElementById('previewImage'),
    removeImageBtn: document.getElementById('removeImageBtn'),
    // Build Plan Bar elements
    buildPlanBar: document.getElementById('buildPlanBar'),
    buildPlanBarToggle: document.getElementById('buildPlanBarToggle'),
    buildPlanBarIcon: document.getElementById('buildPlanBarIcon'),
    buildPlanBarText: document.getElementById('buildPlanBarText'),
    buildPlanBarProgress: document.getElementById('buildPlanBarProgress'),
    buildPlanBarContent: document.getElementById('buildPlanBarContent'),
    buildPlanBarTitle: document.getElementById('buildPlanBarTitle'),
    buildPlanBarSteps: document.getElementById('buildPlanBarSteps')
};

// Action identifier for Fusion bridge communication
const FUSION_ACTION = 'messageFromPalette';

// Fusion -> HTML bridge handler (called via palette.sendInfoToHTML)
window.fusionJavaScriptHandler = {
    handle(action, data) {
        console.log('='.repeat(60));
        console.log('← FUSION MESSAGE');
        console.log('='.repeat(60));
        console.log('← Action:', action);
        console.log('← Data:', data);

        try {
            handleFusionMessage(action, data);
            return 'OK';
        } catch (error) {
            console.error('❌ Error handling Fusion message:', error);
            addLog('error', `Fusion message error: ${error.message}`, { scope: 'global' });
            return 'FAILED';
        }
    }
};

/**
 * Initialize the palette
 */
function initialize() {
    // Prevent duplicate initialization that can stack event listeners and timers
    if (initialized) {
        console.warn('[CADAgent] initialize called more than once; ignoring duplicate call');
        return;
    }
    initialized = true;

    loadDocStatesFromStorage();
    if (!currentDocId) {
        currentDocId = docOrder[0] || 'default';
        getDocState(currentDocId, true);
    }
    detectEnvironment();
    setupEventListeners();
    setupChatInputAutoResize(elements.cadRequest);
    // Ensure the model selector sizes to its text
    try { autoSizeModelSelect(); } catch (e) {}
    if (elements.planningMode) {
        setPlanningModeActive(isPlanningModeActive());
    }
    if (elements.visualContextToggle) {
        // Ensure internal state reflects default active state on load
        const active = elements.visualContextToggle.getAttribute('aria-pressed') === 'true';
        setVisualContextActive(active);
    }
    requestConnectionStatus();

    if (!currentDocId && docOrder.length) {
        currentDocId = docOrder[0];
        renderActiveDoc();
        updateDocSwitcherUI();
        syncProcessingState(currentDocId);
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    if (listenersBound) {
        return;
    }
    listenersBound = true;

    if (elements.executeBtn) {
        // Remove any prior handler that may have been left on the DOM element from a previous script load
        if (typeof elements.executeBtn.__cadagentClickHandler === 'function') {
            elements.executeBtn.removeEventListener('click', elements.executeBtn.__cadagentClickHandler);
        }

        const executeClickHandler = () => {
            // Count to help diagnose unexpected multiple invocations
            console.count('[CADAgent] executeBtn click');

            if (state.processing) {
                // Only treat as cancel if we actually have a run in flight
                if (hasActiveRun()) {
                    handleCancel();
                } else {
                    console.warn('[CADAgent] Cancel click ignored: no active run');
                }
            } else {
                handleExecute();
            }
        };

        elements.executeBtn.addEventListener('click', executeClickHandler);
        elements.executeBtn.__cadagentClickHandler = executeClickHandler;
    }
    if (elements.clearBtn) elements.clearBtn.addEventListener('click', handleClear);
    if (elements.reconnectBtn) elements.reconnectBtn.addEventListener('click', handleReconnect);
    if (elements.inlineReconnect) elements.inlineReconnect.addEventListener('click', handleReconnect);

    // Image attachment event listeners
    if (elements.attachImageBtn) {
        elements.attachImageBtn.addEventListener('click', handleAttachImage);
    }
    if (elements.imageInput) {
        elements.imageInput.addEventListener('change', handleImageSelected);
    }
    if (elements.removeImageBtn) {
        elements.removeImageBtn.addEventListener('click', handleRemoveImage);
    }

    // Build plan bar toggle
    if (elements.buildPlanBarToggle) {
        elements.buildPlanBarToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            const isExpanded = elements.buildPlanBar.classList.contains('expanded');
            elements.buildPlanBar.classList.toggle('expanded');
            elements.buildPlanBarToggle.setAttribute('aria-expanded', !isExpanded);
        });
        
        // Close bar content when clicking outside
        document.addEventListener('click', (e) => {
            if (elements.buildPlanBar && 
                !elements.buildPlanBar.contains(e.target)) {
                elements.buildPlanBar.classList.remove('expanded');
                elements.buildPlanBarToggle.setAttribute('aria-expanded', 'false');
            }
        });
    }

    if (elements.modelSelect) {
        elements.modelSelect.addEventListener('change', (event) => {
            const requestedModel = event.target.value;
            if (!isModelSelectable(requestedModel)) {
                const provider = getProviderForModel(requestedModel);
                addLog('warning', buildMissingApiKeyMessage(provider), { scope: 'global' });
                enforceModelSelectionByKeys();
                return;
            }

            state.selectedModel = requestedModel;
            console.log('Model selection changed to:', state.selectedModel);
            // Recompute width to match new text
            try { autoSizeModelSelect(); } catch (e) {}
            // Update reasoning options for new model's provider
            updateReasoningOptions();
        });
        // Reflow size on resize and after fonts load
        window.addEventListener('resize', () => {
            try { autoSizeModelSelect(); } catch (e) {}
            try { autoSizeReasoningSelect(); } catch (e) {}
        });
        if (document.fonts && typeof document.fonts.ready?.then === 'function') {
            document.fonts.ready.then(() => {
                try { autoSizeModelSelect(); } catch (e) {}
                try { autoSizeReasoningSelect(); } catch (e) {}
            });
        }
    }

    if (elements.reasoningSelect) {
        elements.reasoningSelect.addEventListener('change', (event) => {
            state.reasoningEffort = event.target.value;
            console.log('Reasoning effort changed to:', state.reasoningEffort);
            try { autoSizeReasoningSelect(); } catch (e) {}
        });
    }

    // Initialize reasoning options based on default model
    updateReasoningOptions();
    enforceModelSelectionByKeys();
    updateImageUploadAvailability();

    if (elements.planningMode) {
        elements.planningMode.addEventListener('click', () => {
            const active = !isPlanningModeActive();
            setPlanningModeActive(active);
        });
    }
    if (elements.visualContextToggle) {
        elements.visualContextToggle.addEventListener('click', () => {
            const active = !isVisualContextActive();
            setVisualContextActive(active);
        });
    }

    if (elements.chatMessages) {
        elements.chatMessages.addEventListener('click', (event) => {
            const target = event.target;
            if (!target || typeof target.closest !== 'function') {
                return;
            }
            const revertBtn = target.closest('.revert-btn');
            if (!revertBtn || !elements.chatMessages.contains(revertBtn)) {
                return;
            }
            event.preventDefault();
            const messageEl = revertBtn.closest('.message');
            const messageId = messageEl ? messageEl.getAttribute('data-message-id') : null;
            if (!messageId) {
                console.warn('Revert button clicked without associated message_id');
                return;
            }
            handleRevertToCheckpoint(messageId);
        });
    }

    elements.cadRequest.addEventListener('input', () => {
        if (!state.processing) {
            // Allow sending if there's text OR an attached image
            const hasContent = elements.cadRequest.value.trim() || state.attachedImage;
            elements.executeBtn.disabled = !hasContent || !state.connected;
        }
    });

    elements.cadRequest.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleExecute();
        }
    });

    // Auth event listeners
    if (elements.loginBtn) {
        elements.loginBtn.addEventListener('click', handleLoginClick);
    }
    if (elements.loginStartBtn) {
        elements.loginStartBtn.addEventListener('click', () => {
            setLoginStage('email');
            showLoginOverlay();
            if (elements.emailInput) elements.emailInput.focus();
        });
    }
    if (elements.logoutBtn) {
        elements.logoutBtn.addEventListener('click', handleLogout);
    }
    if (elements.cancelLoginBtn) {
        elements.cancelLoginBtn.addEventListener('click', handleCancelLogin);
    }
    if (elements.loginForm) {
        elements.loginForm.addEventListener('submit', handleSendOtp);
    }
    if (elements.verifyOtpBtn) {
        elements.verifyOtpBtn.addEventListener('click', handleVerifyOtp);
    }
    if (elements.switchEmailBtn) {
        elements.switchEmailBtn.addEventListener('click', handleSwitchEmail);
    }
    if (elements.showPasswordLoginBtn) {
        elements.showPasswordLoginBtn.addEventListener('click', () => {
            togglePasswordLogin(true);
            if (elements.passwordInput) elements.passwordInput.focus();
        });
    }
    if (elements.passwordLoginBtn) {
        elements.passwordLoginBtn.addEventListener('click', handlePasswordLogin);
    }
    if (elements.passwordInput) {
        elements.passwordInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                handlePasswordLogin();
            }
        });
    }
    if (elements.otpInput) {
        elements.otpInput.addEventListener('input', handleOtpInputChange);
        elements.otpInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                handleVerifyOtp();
            }
        });
        elements.otpInput.addEventListener('focus', () => highlightOtpCursor());
        elements.otpInput.addEventListener('blur', () => highlightOtpCursor(false));
    }
    if (elements.loginModal) {
        // Close modal when clicking backdrop (optional)
        elements.loginModal.querySelector('.login-overlay-backdrop')?.addEventListener('click', handleCancelLogin);
    }

    // API Keys event listeners
    if (elements.apiKeysBtn) {
        elements.apiKeysBtn.addEventListener('click', handleOpenApiKeys);
    }
    if (elements.closeApiKeysBtn) {
        elements.closeApiKeysBtn.addEventListener('click', handleCloseApiKeys);
    }
    if (elements.apiKeysForm) {
        elements.apiKeysForm.addEventListener('submit', handleSaveApiKeys);
    }
    if (elements.toggleAnthropicKey) {
        elements.toggleAnthropicKey.addEventListener('click', () => toggleKeyVisibility('anthropicKeyInput'));
    }
    if (elements.toggleOpenaiKey) {
        elements.toggleOpenaiKey.addEventListener('click', () => toggleKeyVisibility('openaiKeyInput'));
    }
    if (elements.anthropicKeyInput) {
        elements.anthropicKeyInput.addEventListener('input', () => {
            elements.anthropicKeyInput.dataset.userEdited = 'true';
        });
    }
    if (elements.openaiKeyInput) {
        elements.openaiKeyInput.addEventListener('input', () => {
            elements.openaiKeyInput.dataset.userEdited = 'true';
        });
    }
    if (elements.apiKeysModal) {
        elements.apiKeysModal.querySelector('.login-overlay-backdrop')?.addEventListener('click', handleCloseApiKeys);
    }

    // External link handler - intercept clicks on external links and send to add-in
    document.addEventListener('click', (event) => {
        const target = event.target;
        if (!target || typeof target.closest !== 'function') {
            return;
        }
        const externalLink = target.closest('.external-link');
        if (externalLink) {
            event.preventDefault();
            const url = externalLink.getAttribute('data-external-url');
            if (url) {
                handleExternalLink(url);
            }
        }
    });

    // Check for auth callback on startup (from magic link)
    checkAuthCallback();
}

/**
 * Auto-resize the chat input textarea.
 * Collapses to one row when blurred, expands to two rows on focus,
 * and grows up to eight rows before enabling scrolling.
 * @param {HTMLTextAreaElement|null} textarea
 */
function setupChatInputAutoResize(textarea) {
    if (!textarea || textarea.dataset.autoResize === 'initialized') return;
    textarea.dataset.autoResize = 'initialized';

    const MIN_ROWS_BLUR = 1;
    const MIN_ROWS_FOCUS = 2;
    const MAX_ROWS = 8;
    
    let isFocused = false;

    const getMetrics = () => {
        const style = window.getComputedStyle(textarea);
        const fontSize = parseFloat(style.fontSize) || 16;
        const rawLineHeight = style.lineHeight || '';
        let lineHeight;

        if (rawLineHeight === 'normal') {
            lineHeight = fontSize * 1.35;
        } else if (rawLineHeight.endsWith('px')) {
            lineHeight = parseFloat(rawLineHeight);
        } else if (!Number.isNaN(parseFloat(rawLineHeight))) {
            lineHeight = parseFloat(rawLineHeight) * fontSize;
        } else {
            lineHeight = fontSize * 1.35;
        }

        const paddingTop = parseFloat(style.paddingTop) || 0;
        const paddingBottom = parseFloat(style.paddingBottom) || 0;
        const borderTop = parseFloat(style.borderTopWidth) || 0;
        const borderBottom = parseFloat(style.borderBottomWidth) || 0;

        return {
            lineHeight,
            padding: paddingTop + paddingBottom,
            border: borderTop + borderBottom
        };
    };

    const heightForRows = (rows, metrics) =>
        (rows * metrics.lineHeight) + metrics.padding + metrics.border;

    const adjustHeight = () => {
        const metrics = getMetrics();
        const minRows = isFocused ? MIN_ROWS_FOCUS : MIN_ROWS_BLUR;
        const minHeight = heightForRows(minRows, metrics);
        const maxHeight = heightForRows(MAX_ROWS, metrics);

        // Reset to allow accurate content measurement
        // Setting height to 0 collapses the textarea so scrollHeight reflects true content height
        textarea.style.height = '0px';
        textarea.style.minHeight = '0px';
        textarea.style.overflowY = 'hidden';

        // Force a reflow to ensure the browser recalculates scrollHeight
        void textarea.offsetHeight;

        // scrollHeight includes padding but not border; add border for box-sizing: border-box
        const intrinsic = textarea.scrollHeight + metrics.border;
        const target = Math.min(Math.max(intrinsic, minHeight), maxHeight);
        const needsScroll = intrinsic > maxHeight;

        textarea.style.height = `${target}px`;
        textarea.style.minHeight = '';  // Clear inline minHeight, let CSS handle it
        textarea.style.overflowY = needsScroll ? 'auto' : 'hidden';
        if (!needsScroll) textarea.scrollTop = 0;
    };

    textarea.addEventListener('focus', () => {
        isFocused = true;
        requestAnimationFrame(adjustHeight);
    });

    textarea.addEventListener('blur', () => {
        isFocused = false;
        adjustHeight();
    });

    textarea.addEventListener('input', () => requestAnimationFrame(adjustHeight));

    // Track width changes so wrapping recomputes height
    if (typeof ResizeObserver === 'function') {
        const resizeObserver = new ResizeObserver(() => adjustHeight());
        resizeObserver.observe(textarea);
    } else {
        window.addEventListener('resize', adjustHeight);
    }

    // Initial adjustment (after fonts load if possible)
    const init = () => {
        if (document.activeElement === textarea) {
            isFocused = true;
        }
        adjustHeight();
    };

    if (document.fonts && typeof document.fonts.ready?.then === 'function') {
        document.fonts.ready.then(init);
    } else {
        setTimeout(init, 0);
    }
}

/**
 * Handle execute button click
 */
function handleExecute() {
    const request = elements.cadRequest.value.trim();
    const planningMode = isPlanningModeActive();
    const includeVisualContext = !planningMode
        && state.executeRequestCount > 0
        && state.visualContextEnabled;

    // Reset cancel latch for the new request
    state.cancelRequested = false;

    // Allow image-only requests
    if ((!request && !state.attachedImage) || !isDocConnected()) {
        return;
    }

    const selectedProvider = getProviderForModel(state.selectedModel);
    if (!isProviderConfigured(selectedProvider)) {
        addLog('error', buildMissingApiKeyMessage(selectedProvider), { scope: 'global' });
        return;
    }

    if (state.attachedImage && !state.apiKeyStatus.openai) {
        addLog('error', buildMissingApiKeyMessage('openai'), { scope: 'global' });
        return;
    }

    if (!planningMode && state.activeRun) {
        appendRunLogEntry('warning', 'Run superseded by new request');
        finalizeActiveRun('cancelled');
    }

    resetRunLogState();

    // Generate unique request ID for checkpoint correlation
    state.requestCounter += 1;
    const requestId = `req_${Date.now()}_${state.requestCounter}`;

    // Keep user message as chat bubble (right side) - must come BEFORE run log
    // Store requestId for checkpoint matching
    const displayMessage = state.attachedImage && !request 
        ? '[Sketch attached]' 
        : (state.attachedImage ? `${request} [+ sketch]` : request);
    appendMessage('user', displayMessage, { sender: 'You', requestId: requestId });

    if (!planningMode) {
        // Immediately create a fresh run log so the user sees activity even before the backend responds
        startRunLogSession({ forceNew: true });
    } else {
        state.captureRunLogs = false;
    }

    // Clear the input field
    elements.cadRequest.value = '';
    elements.cadRequest.dispatchEvent(new Event('input'));

    // Build payload
    const payload = {
        action: 'execute_request',
        request: request,
        planning_mode: planningMode,
        include_visual_context: includeVisualContext,
        model_name: state.selectedModel,
        reasoning_effort: state.reasoningEffort,
        request_id: requestId
    };
    
    // Add image data if attached
    if (state.attachedImage) {
        payload.image_data = state.attachedImage.data;
        payload.image_format = state.attachedImage.format;
        console.log('[CADAgent] Including image in request:', state.attachedImage.format);
        
        // Clear attached image after sending
        handleRemoveImage();
    }
    
    // Send to Python add-in
    sendToAddin(payload);

    if (planningMode) {
        state.executeRequestCount = 0;
    } else {
        state.executeRequestCount += 1;
    }

    // Show progress and morph send button to cancel
    showProgress('Sending request to backend...');
    state.processing = true;
    updateExecuteButtonState();
    updateRevertButtonStates();
}

/**
 * Handle attach image button click
 */
function handleAttachImage() {
    if (!state.apiKeyStatus.openai) {
        addLog('warning', buildMissingApiKeyMessage('openai'), { scope: 'global' });
        return;
    }

    if (elements.imageInput) {
        elements.imageInput.click();
    }
}

/**
 * Handle image file selection
 */
function handleImageSelected(event) {
    if (!state.apiKeyStatus.openai) {
        addLog('warning', buildMissingApiKeyMessage('openai'), { scope: 'global' });
        if (elements.imageInput) {
            elements.imageInput.value = '';
        }
        return;
    }

    const file = event.target.files[0];
    if (!file) return;
    
    // Validate file type
    if (!file.type.match(/^image\/(png|jpeg|jpg)$/)) {
        addLog('error', 'Please select a PNG or JPG image', { scope: 'global' });
        return;
    }
    
    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        addLog('error', 'Image too large. Maximum size is 10MB', { scope: 'global' });
        return;
    }
    
    // Read and encode image
    const reader = new FileReader();
    reader.onload = (e) => {
        const base64data = e.target.result;
        
        // Extract base64 string (remove "data:image/png;base64," prefix)
        const base64 = base64data.split(',')[1];
        const format = file.type.split('/')[1]; // 'png' or 'jpeg'
        
        // Store in state
        state.attachedImage = {
            data: base64,
            format: format
        };
        
        // Show preview
        if (elements.previewImage && elements.imagePreview) {
            elements.previewImage.src = base64data;
            elements.imagePreview.classList.remove('hidden');
        }
        
        // Enable execute button if connected
        if (isDocConnected() && elements.executeBtn) {
            elements.executeBtn.disabled = false;
        }
        
        addLog('success', `Sketch attached: ${file.name}`, { scope: 'global' });
        console.log('[CADAgent] Image attached:', {
            name: file.name,
            size: file.size,
            format: format
        });
    };
    
    reader.onerror = () => {
        addLog('error', 'Failed to read image file', { scope: 'global' });
    };
    
    reader.readAsDataURL(file);
}

/**
 * Handle remove image button click
 */
function handleRemoveImage() {
    state.attachedImage = null;
    
    if (elements.imagePreview) {
        elements.imagePreview.classList.add('hidden');
    }
    if (elements.previewImage) {
        elements.previewImage.src = '';
    }
    if (elements.imageInput) {
        elements.imageInput.value = ''; // Clear file input
    }
    
    // Update execute button state
    if (elements.executeBtn && elements.cadRequest) {
        elements.executeBtn.disabled = !isDocConnected() || !elements.cadRequest.value.trim();
    }
}

/**
 * Handle cancel button click
 */
function handleCancel() {
    if (!isDocConnected()) {
        return;
    }

    // Prevent duplicate cancel logs if multiple click handlers ever fire
    if (state.cancelRequested) {
        console.warn('[CADAgent] Cancel request already sent; ignoring duplicate');
        return;
    }
    state.cancelRequested = true;

    // Send cancel request to backend
    sendToAddin({
        action: 'cancel_request'
    });

    addLog('warning', 'Cancelling request...', { scope: 'run' });
}

/**
 * Update execute button appearance based on processing state
 */
function updateExecuteButtonState() {
    if (!elements.executeBtn) return;

    const sendIconUse = elements.executeBtn.querySelector('.send-icon-svg use');
    if (!sendIconUse) return;

    if (state.processing) {
        // Morph to cancel button
        elements.executeBtn.classList.add('cancel-mode');
        elements.executeBtn.setAttribute('aria-label', 'Cancel request');
        elements.executeBtn.disabled = false;
        sendIconUse.setAttribute('href', '#icon-cancel');
    } else {
        // Morph back to send button
        elements.executeBtn.classList.remove('cancel-mode');
        elements.executeBtn.setAttribute('aria-label', 'Send');
        const connected = isDocConnected();
        elements.executeBtn.disabled = !connected || !elements.cadRequest.value.trim();
        sendIconUse.setAttribute('href', '#icon-send');
    }

    if (elements.inlineReconnect) {
        if (isDocConnected()) {
            elements.inlineReconnect.classList.add('hidden');
        } else {
            elements.inlineReconnect.classList.remove('hidden');
        }
    }
}

/**
 * Update revert button states based on processing state
 */
function updateRevertButtonStates() {
    if (!elements.chatMessages) return;

    const revertButtons = elements.chatMessages.querySelectorAll('.revert-btn');
    revertButtons.forEach(btn => {
        if (state.processing) {
            btn.disabled = true;
            btn.classList.add('disabled');
            btn.setAttribute('title', 'Cannot revert during execution');
        } else {
            btn.disabled = false;
            btn.classList.remove('disabled');
            btn.setAttribute('title', 'Revert timeline to this state');
        }
    });
}

/**
 * Handle clear button click
 */
function handleClear() {
    elements.cadRequest.value = '';
    elements.cadRequest.dispatchEvent(new Event('input'));
    elements.executeBtn.disabled = true;
    elements.cadRequest.focus();
    addLog('info', 'Input cleared', { scope: 'global' });
}

/**
 * Handle reconnect button click
 */
function handleReconnect() {
    if (!elements.reconnectBtn) {
        return;
    }

    // Disable button and show spinning animation
    elements.reconnectBtn.disabled = true;
    elements.reconnectBtn.classList.add('reconnecting');

    addLog('info', 'Attempting to reconnect...', { scope: 'global' });

    // Send reconnect request to add-in
    sendToAddin({ action: 'reconnect_request' });

    // Re-enable after 2 seconds
    setTimeout(() => {
        if (elements.reconnectBtn) {
            elements.reconnectBtn.disabled = false;
            elements.reconnectBtn.classList.remove('reconnecting');
        }
    }, 2000);
}

/**
 * Auth handler functions
 */
function handleLoginClick() {
    showLoginOverlay();
    setLoginStage('cta');
}

function handleCancelLogin() {
    resetLoginFields();
    // Keep overlay visible for logged-out state; only hide if already authenticated
    if (state && state.connected && elements.loginModal) {
        hideLoginOverlay();
    } else {
        setLoginStage('cta');
    }
}

function handleSendOtp(event) {
    event.preventDefault();

    if (!elements.emailInput) return;

    const email = elements.emailInput.value.trim();
    if (!email) {
        showLoginStatus('Please enter your email address', 'error');
        return;
    }

    if (elements.tosConsent && !elements.tosConsent.checked) {
        showLoginStatus('Please agree to the Terms of Service and Privacy Policy to continue.', 'error');
        return;
    }

    if (elements.sendOtpBtn) {
        elements.sendOtpBtn.disabled = true;
        elements.sendOtpBtn.textContent = 'Checking...';
    }

    // Ensure account exists, then always require OTP verification.
    sendToAddin({
        action: 'check_and_handle_signup',
        email: email
    });

    // Note: Response will be handled in dispatchAddinMessage via auth_otp_required.
}

function handleVerifyOtp() {
    if (!elements.emailInput || !elements.otpInput) return;
    const email = elements.emailInput.value.trim();
    const code = elements.otpInput.value.trim();
    if (!email || !code) {
        showLoginStatus('Email and code are required', 'error');
        return;
    }
    if (!/^[0-9]{6}$/.test(code)) {
        showLoginStatus('Enter the 6-digit code', 'error');
        return;
    }
    if (elements.verifyOtpBtn) {
        elements.verifyOtpBtn.disabled = true;
        elements.verifyOtpBtn.textContent = 'Verifying...';
    }
    
    // Reset verification state for new attempt
    authVerificationCompleted = false;
    
    sendToAddin({
        action: 'verify_otp_code',
        email,
        code
    });

    // Start auth watchdog to detect stuck OTP verification
    startAuthWatchdog();
}

function handleSwitchEmail() {
    resetLoginFields();
    setLoginStage('email');
    if (elements.emailInput) elements.emailInput.focus();
}

function togglePasswordLogin(show) {
    if (!elements.passwordRow || !elements.showPasswordLoginBtn) return;
    elements.passwordRow.classList.toggle('hidden', !show);
    elements.showPasswordLoginBtn.classList.toggle('hidden', show);
}

function handlePasswordLogin() {
    if (!state.testingMode) {
        showLoginStatus('Password login is disabled', 'error');
        return;
    }
    if (!elements.emailInput || !elements.passwordInput) return;
    const email = elements.emailInput.value.trim();
    const password = elements.passwordInput.value;
    if (!email || !password) {
        showLoginStatus('Email and password are required', 'error');
        return;
    }
    if (state.testingEmail && email.toLowerCase() !== state.testingEmail) {
        showLoginStatus('Password login only allowed for the test account', 'error');
        return;
    }
    if (elements.passwordLoginBtn) {
        elements.passwordLoginBtn.disabled = true;
        elements.passwordLoginBtn.textContent = 'Signing in...';
    }
    sendToAddin({
        action: 'login_with_password',
        email,
        password
    });
}

function handleOtpInputChange(event) {
    if (!event || !event.target) return;
    const raw = String(event.target.value || '');
    const digits = raw.replace(/\D/g, '').slice(0, 6);
    event.target.value = digits;
    if (elements.verifyOtpBtn) {
        elements.verifyOtpBtn.disabled = digits.length !== 6;
    }
    updateOtpVisual(digits);
}

function updateOtpVisual(digits) {
    if (!Array.isArray(elements.otpBoxes)) return;
    const chars = (digits || '').split('');
    elements.otpBoxes.forEach((box, idx) => {
        const char = chars[idx] || '';
        box.textContent = char || ' ';
        box.classList.toggle('filled', !!char);
        const activeIdx = Math.min(chars.length, 5);
        box.classList.toggle('active', otpFocused && idx === activeIdx);
    });
}

function highlightOtpCursor(show = true) {
    if (!Array.isArray(elements.otpBoxes)) return;
    otpFocused = show;
    const length = (elements.otpInput?.value || '').length;
    const idx = Math.min(length, 5);
    elements.otpBoxes.forEach((box, i) => {
        box.classList.toggle('active', show && i === idx);
    });
}

function setLoginStage(stage = 'cta') {
    if (!elements.loginModal) return;
    const stageNormalized = ['cta', 'email', 'code'].includes(stage) ? stage : 'cta';
    state.loginStage = stageNormalized;

    if (elements.loginCta) {
        elements.loginCta.classList.toggle('hidden', stageNormalized !== 'cta');
    }
    if (elements.loginForm) {
        elements.loginForm.classList.toggle('hidden', stageNormalized === 'cta');
    }
    if (elements.otpSection) {
        elements.otpSection.classList.toggle('hidden', stageNormalized !== 'code');
    }
    if (elements.emailRow) {
        elements.emailRow.classList.toggle('hidden', stageNormalized === 'code');
    }
    if (elements.passwordRow) {
        const showPasswordRow = stageNormalized !== 'code' && state.testingMode && !elements.passwordRow.classList.contains('hidden');
        elements.passwordRow.classList.toggle('hidden', !showPasswordRow);
    }

    // Adjust email field mode when code stage is active
    if (elements.emailInput) {
        elements.emailInput.readOnly = stageNormalized === 'code';
    }
    if (elements.sendOtpBtn) {
        elements.sendOtpBtn.textContent = stageNormalized === 'code' ? 'Resend' : 'Continue';
        elements.sendOtpBtn.disabled = false;
    }
    if (elements.verifyOtpBtn) {
        const digits = elements.otpInput ? String(elements.otpInput.value || '') : '';
        const ready = stageNormalized === 'code' && digits.length === 6;
        elements.verifyOtpBtn.disabled = !ready;
    }

    if (stageNormalized === 'email' && elements.emailInput) {
        elements.emailInput.focus();
    } else if (stageNormalized === 'code' && elements.otpInput) {
        elements.otpInput.focus();
        updateOtpVisual(elements.otpInput.value || '');
    }
}

function showLoginOverlay() {
    if (!elements.loginModal) return;
    elements.loginModal.classList.remove('hidden');
    if (elements.loginVersion && window.SPACE_APP_VERSION) {
        elements.loginVersion.textContent = window.SPACE_APP_VERSION;
    }
    // Sync visual OTP boxes with any existing input (e.g., after reconnect)
    if (elements.otpInput) {
        updateOtpVisual(elements.otpInput.value || '');
    }
}

function hideLoginOverlay() {
    if (!elements.loginModal) return;
    elements.loginModal.classList.add('hidden');
}

function resetLoginFields() {
    if (elements.emailInput) {
        elements.emailInput.value = '';
        elements.emailInput.readOnly = false;
    }
    if (elements.otpInput) {
        elements.otpInput.value = '';
    }
    if (elements.passwordInput) {
        elements.passwordInput.value = '';
    }
    if (elements.loginStatus) {
        elements.loginStatus.classList.add('hidden');
        elements.loginStatus.textContent = '';
    }
    if (elements.verifyOtpBtn) {
        elements.verifyOtpBtn.disabled = true;
        elements.verifyOtpBtn.textContent = 'Verify';
    }
    if (elements.sendOtpBtn) {
        elements.sendOtpBtn.disabled = false;
        elements.sendOtpBtn.textContent = 'Continue';
    }
    if (elements.passwordLoginBtn) {
        elements.passwordLoginBtn.disabled = false;
        elements.passwordLoginBtn.textContent = 'Log in';
    }
    if (elements.passwordRow && state.testingMode) {
        togglePasswordLogin(false);
    }
    updateOtpVisual('');
}

// ==================== API KEYS FUNCTIONS ====================

function handleOpenApiKeys() {
    console.log('[API Keys] Opening settings modal');
    // Auth guard: prevent opening modal if not authenticated
    if (!state.isAuthenticated) {
        console.warn('[API Keys] Blocked: user not authenticated');
        addLog('warning', 'Please log in to manage API keys', { scope: 'run' });
        return;
    }
    // Request current key status from backend
    sendToAddin({ action: 'get_api_keys_status' });
    showApiKeysModal();
}

function handleCloseApiKeys() {
    hideApiKeysModal();
    resetApiKeysForm();
}

function handleSaveApiKeys(event) {
    event.preventDefault();

    // Auth guard: prevent saving if not authenticated
    if (!state.isAuthenticated) {
        console.warn('[API Keys] Save blocked: user not authenticated');
        showApiKeysStatus('Please log in to save API keys', 'error');
        return;
    }

    const anthropicKey = elements.anthropicKeyInput?.value?.trim() || '';
    const openaiKey = elements.openaiKeyInput?.value?.trim() || '';

    // Basic validation
    if (anthropicKey && !anthropicKey.startsWith('sk-ant-')) {
        showApiKeysStatus('Invalid Anthropic key format (should start with sk-ant-)', 'error');
        return;
    }
    if (openaiKey && !openaiKey.startsWith('sk-')) {
        showApiKeysStatus('Invalid OpenAI key format (should start with sk-)', 'error');
        return;
    }

    // At least one key required
    if (!anthropicKey && !openaiKey) {
        showApiKeysStatus('Please provide at least one API key (Anthropic or OpenAI)', 'error');
        return;
    }

    console.log('[API Keys] Saving keys...');
    if (elements.saveApiKeysBtn) {
        elements.saveApiKeysBtn.disabled = true;
        elements.saveApiKeysBtn.textContent = 'Saving...';
    }

    sendToAddin({
        action: 'save_api_keys',
        keys: {
            anthropic_api_key: anthropicKey,
            openai_api_key: openaiKey,
            google_api_key: ''
        }
    });
}

function toggleKeyVisibility(inputId) {
    const input = elements[inputId];
    if (!input) return;

    // Find the toggle button which is the next sibling
    const btn = input.nextElementSibling;
    const svgUse = btn ? btn.querySelector('use') : null;

    if (input.type === 'password') {
        input.type = 'text';
        if (svgUse) svgUse.setAttribute('href', '#icon-eye-off');
    } else {
        input.type = 'password';
        if (svgUse) svgUse.setAttribute('href', '#icon-eye');
    }
}

function showApiKeysModal() {
    if (!elements.apiKeysModal) return;
    elements.apiKeysModal.classList.remove('hidden');
}

function hideApiKeysModal() {
    if (!elements.apiKeysModal) return;
    elements.apiKeysModal.classList.add('hidden');
}

function resetApiKeysForm() {
    if (elements.anthropicKeyInput) {
        elements.anthropicKeyInput.value = '';
        elements.anthropicKeyInput.type = 'password';
        elements.anthropicKeyInput.dataset.userEdited = 'false';
    }
    if (elements.openaiKeyInput) {
        elements.openaiKeyInput.value = '';
        elements.openaiKeyInput.type = 'password';
        elements.openaiKeyInput.dataset.userEdited = 'false';
    }
    if (elements.apiKeysStatus) {
        elements.apiKeysStatus.classList.add('hidden');
        elements.apiKeysStatus.textContent = '';
    }
    if (elements.saveApiKeysBtn) {
        elements.saveApiKeysBtn.disabled = false;
        elements.saveApiKeysBtn.textContent = 'Save Configuration';
    }
    // Reset visibility toggles and icons
    elements.toggleAnthropicKey?.querySelector('use')?.setAttribute('href', '#icon-eye');
    elements.toggleOpenaiKey?.querySelector('use')?.setAttribute('href', '#icon-eye');
}

function showApiKeysStatus(message, type = 'info') {
    if (!elements.apiKeysStatus) return;
    elements.apiKeysStatus.textContent = message;
    elements.apiKeysStatus.className = `login-status ${type}`;
    elements.apiKeysStatus.classList.remove('hidden');
}

function updateApiKeysUI(status) {
    console.log('[API Keys] Updating UI with status:', status);

    const shouldOverrideInput = (inputEl) => inputEl && inputEl.dataset.userEdited !== 'true';
    state.apiKeyStatus.anthropic = !!status?.anthropic?.configured;
    state.apiKeyStatus.openai = !!status?.openai?.configured;

    // Update status badges
    if (elements.anthropicKeyStatus) {
        if (status.anthropic?.configured) {
            elements.anthropicKeyStatus.textContent = 'Configured';
            elements.anthropicKeyStatus.className = 'key-status configured';
        } else {
            elements.anthropicKeyStatus.textContent = 'Not set';
            elements.anthropicKeyStatus.className = 'key-status missing';
        }
    }

    if (elements.anthropicKeyInput) {
        const keyValue = status.anthropic?.value || '';
        if (shouldOverrideInput(elements.anthropicKeyInput)) {
            elements.anthropicKeyInput.value = keyValue;
        }
        elements.anthropicKeyInput.placeholder = status.anthropic?.masked || 'sk-ant-...';
        elements.anthropicKeyInput.type = 'password'; // default hidden until eye is toggled
        elements.anthropicKeyInput.dataset.hasStoredValue = keyValue ? 'true' : 'false';
        elements.toggleAnthropicKey?.querySelector('use')?.setAttribute('href', '#icon-eye');
    }

    if (elements.openaiKeyStatus) {
        if (status.openai?.configured) {
            elements.openaiKeyStatus.textContent = 'Configured';
            elements.openaiKeyStatus.className = 'key-status configured';
        } else {
            elements.openaiKeyStatus.textContent = 'Not set';
            elements.openaiKeyStatus.className = 'key-status missing';
        }
    }

    if (elements.openaiKeyInput) {
        const keyValue = status.openai?.value || '';
        if (shouldOverrideInput(elements.openaiKeyInput)) {
            elements.openaiKeyInput.value = keyValue;
        }
        elements.openaiKeyInput.placeholder = status.openai?.masked || 'sk-proj-...';
        elements.openaiKeyInput.type = 'password';
        elements.openaiKeyInput.dataset.hasStoredValue = keyValue ? 'true' : 'false';
        elements.toggleOpenaiKey?.querySelector('use')?.setAttribute('href', '#icon-eye');
    }

    // Update footer message based on whether any keys are configured
    const hasAnyKey = status.anthropic?.configured || status.openai?.configured;
    if (elements.apiKeysFooter) {
        if (hasAnyKey) {
            elements.apiKeysFooter.classList.add('hidden');
        } else {
            elements.apiKeysFooter.classList.remove('hidden');
        }
    }

    enforceModelSelectionByKeys();
    updateImageUploadAvailability();
}

function handleExternalLink(url) {
    console.log('[External Link] Opening:', url);
    sendToAddin({
        action: 'open_external_url',
        url: url
    });
}

// ==================== END API KEYS FUNCTIONS ====================

function startResendCooldown(seconds) {
    if (!elements.sendOtpBtn) return;
    let remaining = seconds;
    elements.sendOtpBtn.disabled = true;
    const originalText = elements.sendOtpBtn.textContent || 'Continue';
    const timer = setInterval(() => {
        remaining -= 1;
        if (remaining <= 0) {
            clearInterval(timer);
            elements.sendOtpBtn.disabled = false;
            elements.sendOtpBtn.textContent = originalText;
            return;
        }
        elements.sendOtpBtn.textContent = `Resend (${remaining}s)`;
    }, 1000);
}

function handleLogout() {
    if (!confirm('Are you sure you want to logout?')) {
        return;
    }

    sendToAddin({ action: 'logout' });
}

function checkAuthCallback() {
    // Check if URL contains auth callback params (from magic link)
    const params = new URLSearchParams(window.location.hash.substring(1));
    const accessToken = params.get('access_token');
    const refreshToken = params.get('refresh_token');

    if (accessToken && refreshToken) {
        console.log('Auth callback detected, processing tokens...');

        // Send tokens to Python backend
        sendToAddin({
            action: 'auth_callback',
            access_token: accessToken,
            refresh_token: refreshToken
        });

        // Clear URL hash
        window.location.hash = '';
    } else {
        // Request profile on startup to check if already logged in
        sendToAddin({ action: 'get_profile' });
    }
}

function showLoginStatus(message, type = 'info') {
    if (!elements.loginStatus) return;

    elements.loginStatus.textContent = message;
    elements.loginStatus.className = `login-status ${type}`;
    elements.loginStatus.classList.remove('hidden');
}

function updateAuthUI(userEmail, options = {}) {
    const { preserveStage = false } = options;
    state.isAuthenticated = !!userEmail;

    if (!userEmail) {
        state.apiKeyStatus.anthropic = false;
        state.apiKeyStatus.openai = false;
        enforceModelSelectionByKeys();
        updateImageUploadAvailability();

        // Not logged in
        if (elements.loginBtn) elements.loginBtn.classList.remove('hidden');
        if (elements.userInfo) elements.userInfo.classList.add('hidden');
        showLoginOverlay();
        if (!preserveStage || state.loginStage === 'cta') {
            resetLoginFields();
            setLoginStage('cta');
        } else {
            // Preserve user input and stay on current stage
            setLoginStage(state.loginStage);
        }
    } else {
        // Logged in
        if (elements.loginBtn) elements.loginBtn.classList.add('hidden');
        if (elements.userInfo) elements.userInfo.classList.remove('hidden');
        hideLoginOverlay();

        // Refresh key status so model/image availability reflects current configuration
        sendToAddin({ action: 'get_api_keys_status' });
    }
}

function isPlanningModeActive() {
    return !!(elements.planningMode && elements.planningMode.getAttribute('aria-pressed') === 'true');
}

function setPlanningModeActive(active) {
    if (!elements.planningMode) return;
    elements.planningMode.setAttribute('aria-pressed', active ? 'true' : 'false');
    elements.planningMode.classList.toggle('active', active);
    if (elements.planningModeState) {
        elements.planningModeState.textContent = active ? 'On' : 'Off';
    }
    const tooltip = active
        ? 'Planning mode is ON. Click to disable.'
        : 'Planning mode is OFF. Click to enable.';
    elements.planningMode.setAttribute('title', tooltip);
    elements.planningMode.setAttribute('aria-label', tooltip);
}

function isVisualContextActive() {
    return !!(elements.visualContextToggle && elements.visualContextToggle.getAttribute('aria-pressed') === 'true');
}

function setVisualContextActive(active) {
    if (!elements.visualContextToggle) {
        return;
    }

    elements.visualContextToggle.setAttribute('aria-pressed', active ? 'true' : 'false');
    elements.visualContextToggle.classList.toggle('active', active);
    state.visualContextEnabled = active;

    if (elements.visualContextState) {
        elements.visualContextState.textContent = active ? 'On' : 'Off';
    }

    const tooltip = active
        ? 'Visual context is ON. Click to disable.'
        : 'Visual context is OFF. Click to enable.';
    elements.visualContextToggle.setAttribute('title', tooltip);
    elements.visualContextToggle.setAttribute('aria-label', tooltip);
}

function truncateText(text, limit = 25) {
    if (typeof text !== 'string') {
        return '';
    }
    if (text.length <= limit) {
        return text;
    }
    return `${text.slice(0, Math.max(1, limit - 1)).trimEnd()}…`;
}

function resetRunLogState() {
    resetReasoningState({ forceComplete: true });
    state.activeRun = null;
    state.captureRunLogs = false;
}

function buildThreadSnapshot() {
    return null; // legacy placeholder retained for compatibility
}

function persistCurrentThreadState() {
    if (!currentDocId) return;
    const docState = getDocState(currentDocId, false);
    if (docState) {
        docState.lastUpdated = Date.now();
        persistDocState(currentDocId);
    }
}

function renderMessageFromData(item) {
    if (!item || item.type !== 'message') return;
    const opts = item.options || {};
    opts.__skipStore = true;
    appendMessage(item.role, item.text, opts);
}

function createRunEntryElement(entry) {
    const itemEl = document.createElement('li');
    itemEl.classList.add('run-entry');
    const tone = determineRunEntryTone(entry.level, entry.text) || 'neutral';
    if (tone && tone !== 'neutral') {
        itemEl.classList.add(`run-entry-tone-${tone}`);
    }

    const markerEl = document.createElement('div');
    markerEl.classList.add('run-entry-marker');
    const dotEl = document.createElement('span');
    dotEl.classList.add('run-entry-dot');
    if (tone && tone !== 'neutral') {
        dotEl.classList.add(`run-entry-dot-${tone}`);
    }
    markerEl.appendChild(dotEl);
    itemEl.appendChild(markerEl);

    const bodyEl = document.createElement('div');
    bodyEl.classList.add('run-entry-body');
    const textEl = document.createElement('span');
    textEl.classList.add('run-entry-text');

    if (entry.format === 'html') {
        textEl.innerHTML = entry.text;
    } else if (entry.format === 'markdown') {
        const rendered = (typeof renderMarkdown === 'function') ? renderMarkdown(entry.text) : entry.text;
        textEl.innerHTML = rendered;
    } else {
        textEl.textContent = entry.text;
    }
    textEl.setAttribute('title', entry.text);
    bodyEl.appendChild(textEl);
    itemEl.appendChild(bodyEl);
    return itemEl;
}

function renderRunFromData(runData, options = {}) {
    if (!elements.chatMessages || !runData) return null;

    const runFeedEl = document.createElement('section');
    runFeedEl.classList.add('run-feed');

    const detailsEl = document.createElement('details');
    detailsEl.classList.add('run-details');

    const summaryEl = document.createElement('summary');
    summaryEl.classList.add('run-summary');
    summaryEl.setAttribute('title', 'Toggle detailed actions');

    const latestTextEl = document.createElement('span');
    latestTextEl.classList.add('run-summary-text');
    latestTextEl.textContent = runData.summaryText || 'Designing';
    if (runData.status === 'in-progress') {
        latestTextEl.classList.add('shimmer');
    }

    summaryEl.appendChild(latestTextEl);

    // Feedback buttons (thumbs up/down) for iteration evaluation
    const feedbackWrap = document.createElement('div');
    feedbackWrap.classList.add('run-feedback');
    const upBtn = document.createElement('button');
    upBtn.type = 'button';
    upBtn.classList.add('thumb-btn', 'thumb-up');
    upBtn.setAttribute('aria-label', 'Mark iteration as good');
    upBtn.innerHTML = '<svg class="icon-thumb" aria-hidden="true"><use href="#icon-thumb-up"></use></svg>';
    const downBtn = document.createElement('button');
    downBtn.type = 'button';
    downBtn.classList.add('thumb-btn', 'thumb-down');
    downBtn.setAttribute('aria-label', 'Mark iteration as bad');
    downBtn.innerHTML = '<svg class="icon-thumb" aria-hidden="true"><use href="#icon-thumb-down"></use></svg>';

    const iterationNumber = runData.iteration || options.iteration || null;
    const handleClick = (verdict) => {
        if (!iterationNumber) {
            showToast('Iteration unknown');
            return;
        }
        sendIterationFeedback(iterationNumber, verdict);
    };

    upBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        handleClick('positive');
    });
    downBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        handleClick('negative');
    });

    feedbackWrap.appendChild(upBtn);
    feedbackWrap.appendChild(downBtn);
    summaryEl.appendChild(feedbackWrap);
    detailsEl.appendChild(summaryEl);

    const actionsListEl = document.createElement('ol');
    actionsListEl.classList.add('run-actions');
    detailsEl.appendChild(actionsListEl);
    runFeedEl.appendChild(detailsEl);

    if (Array.isArray(runData.entries)) {
        runData.entries.forEach((entry) => {
            const itemEl = createRunEntryElement(entry);
            if (itemEl) actionsListEl.appendChild(itemEl);
        });
    }

    // Apply status classes
    runFeedEl.classList.add(
        runData.status === 'success'
            ? 'run-complete-success'
            : runData.status === 'error'
                ? 'run-complete-error'
                : runData.status === 'cancelled'
                    ? 'run-complete-cancelled'
                    : 'run-in-progress'
    );

    elements.chatMessages.appendChild(runFeedEl);

    const runState = {
        container: runFeedEl,
        detailsEl,
        summaryEl,
        summaryTextEl: latestTextEl,
        actionsListEl,
        entries: runData.entries || [],
        status: runData.status,
        runData,
        docId: currentDocId
    };

    if (options.setActive) {
        state.activeRun = runState;
        state.captureRunLogs = runData.status === 'in-progress';
    }

    // Keep run logs expanded across all states, including after end_turn.
    detailsEl.open = true;

    detailsEl.addEventListener('toggle', () => {
        try { animateRunDetails(detailsEl); } catch (e) { /* no-op */ }
    });

    return runState;
}

function renderActiveDoc() {
    if (!elements.chatMessages) return;
    elements.chatMessages.innerHTML = '';
    resetRunLogState();

    const docState = getDocState(currentDocId, false);
    if (!docState || !Array.isArray(docState.items) || docState.items.length === 0) {
        if (elements.welcomeMessage) elements.welcomeMessage.classList.remove('hidden');
        if (elements.headerSection) elements.headerSection.classList.remove('compact-mode');
        return;
    }

    if (elements.welcomeMessage) elements.welcomeMessage.classList.add('hidden');
    if (elements.headerSection) elements.headerSection.classList.add('compact-mode');

    let runCounter = 0;
    docState.items.forEach((item) => {
        if (item.type === 'message') {
            renderMessageFromData(item);
        } else if (item.type === 'run') {
            runCounter += 1;
            if (!item.iteration) {
                item.iteration = runCounter;
            }
            const isActive = docState.activeRunId && docState.activeRunId === item.id && item.status === 'in-progress';
            renderRunFromData(item, { setActive: isActive });
        }
    });
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

/**
 * Update doc switcher UI.
 * 
 * Currently a no-op because:
 * - Fusion 360 already provides document tabs
 * - Palette seamlessly follows Fusion's active document via documentActivated event
 * - No need for redundant tab UI in the palette
 * 
 * State is still tracked per-doc in `docStates` for history preservation.
 * The palette automatically switches view when receiving `document_switched` messages.
 */
function updateDocSwitcherUI() {
    // Intentionally empty - Fusion handles doc switching UI
    return;
}

function showToast(message) {
    if (!elements.toastContainer || !message) return;
    const toast = document.createElement('div');
    toast.classList.add('toast');
    toast.textContent = message;
    elements.toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-6px)';
        setTimeout(() => toast.remove(), 250);
    }, 2200);
}

function rebuildDocStateFromDom(docId) {
    const docState = getDocState(docId, false);
    if (!docState || !elements.chatMessages) return;
    const items = [];
    const children = elements.chatMessages.children;
    Array.from(children).forEach((node) => {
        if (node.classList.contains('message')) {
            const roleClass = node.classList.contains('user-message')
                ? 'user'
                : node.classList.contains('assistant-message')
                    ? 'assistant'
                    : 'system';
            const textEl = node.querySelector('.message-text');
            const headerSender = node.querySelector('.message-sender');
            const messageId = node.getAttribute('data-message-id');
            items.push({
                type: 'message',
                role: roleClass,
                text: textEl ? textEl.innerHTML : '',
                options: {
                    sender: headerSender ? headerSender.textContent : undefined,
                    variant: Array.from(node.classList).find((c) => ['info', 'error', 'warning', 'success', 'agent'].includes(c)) || undefined,
                    contentFormat: 'html',
                    messageId: messageId || null
                }
            });
        } else if (node.classList.contains('run-feed')) {
            const summaryTextEl = node.querySelector('.run-summary-text');
            const entries = [];
            node.querySelectorAll('.run-entry-text').forEach((el) => {
                entries.push({ level: 'info', text: el.textContent || '', format: 'text' });
            });
            let status = 'in-progress';
            if (node.classList.contains('run-complete-success')) status = 'success';
            else if (node.classList.contains('run-complete-error')) status = 'error';
            else if (node.classList.contains('run-complete-cancelled')) status = 'cancelled';
            items.push({
                type: 'run',
                id: generateId('run'),
                status,
                summaryText: summaryTextEl ? summaryTextEl.textContent : 'Designing',
                entries,
                createdAt: Date.now()
            });
        }
    });
    docState.items = items;
    docState.activeRunId = null;
    docState.lastUpdated = Date.now();
    persistDocState(docId);
}

function hasActiveRun() {
    return !!(state.activeRun && state.activeRun.container && state.activeRun.container.isConnected);
}

function startRunLogSession(options = {}) {
    const { forceNew = false } = options;
    state.captureRunLogs = true;

    if (forceNew) {
        log('[RUNLOG] startRunLogSession(forceNew=true) resetting activeRun');
        state.activeRun = null;
    } else if (hasActiveRun()) {
        log('[RUNLOG] startRunLogSession reuse active run container');
        return state.activeRun;
    }

    return ensureRunLogContainer();
}

function ensureRunLogContainer() {
    if (!state.captureRunLogs) {
        warn('[RUNLOG] ensureRunLogContainer called while captureRunLogs=false');
        return null;
    }

    if (state.activeRun && state.activeRun.container?.isConnected) {
        return state.activeRun;
    }

    if (!elements.chatMessages) {
        warn('[RUNLOG] chatMessages element not found; cannot create run log');
        return null;
    }

    const docState = getDocState(currentDocId, true);
    if (!docState) {
        warn('[RUNLOG] no docState available; cannot create run log');
        return null;
    }

    resetReasoningState({ forceComplete: true });

    if (elements.welcomeMessage && !elements.welcomeMessage.classList.contains('hidden')) {
        elements.welcomeMessage.classList.add('hidden');
    }

    // Rehydrate existing in-progress run if present
    if (docState.activeRunId) {
        const existing = docState.items.find((item) => item.type === 'run' && item.id === docState.activeRunId);
        if (existing && existing.status === 'in-progress') {
            const runState = renderRunFromData(existing, { setActive: true });
            if (runState) {
                persistCurrentThreadState();
                return runState;
            }
        }
    }

    // Create a new run record
    const existingRuns = docState.items.filter((i) => i.type === 'run');
    const nextIteration = existingRuns.length + 1;

    const runData = {
        type: 'run',
        id: generateId('run'),
        status: 'in-progress',
        summaryText: 'Designing',
        entries: [],
        createdAt: Date.now(),
        iteration: nextIteration
    };
    log(`[RUNLOG] Creating new run container docId=${docState.id} iteration=${nextIteration}`);
    docState.items.push(runData);
    docState.activeRunId = runData.id;
    docState.lastStatus = 'in-progress';
    docState.lastUpdated = Date.now();

    const runState = renderRunFromData(runData, { setActive: true });

    state.reasoning.activeSession = null;
    state.reasoning.sessionCounter = 0;

    persistDocState(docState.id);
    return runState;
}

function shouldSkipRunEntry(level, text) {
    if (!text) {
        log(`[RUNLOG] shouldSkipRunEntry -> false (no text) level='${level}'`);
        return false;
    }
    const normalized = String(text).trim().toLowerCase();
    if (!normalized) {
        log(`[RUNLOG] shouldSkipRunEntry -> false (empty normalized) level='${level}'`);
        return false;
    }
    if (normalized.startsWith('status update')) {
        log(`[RUNLOG] shouldSkipRunEntry -> true (status update) text='${normalized.slice(0,80)}'`);
        return true;
    }
    if (normalized.startsWith('submitting request')) {
        log(`[RUNLOG] shouldSkipRunEntry -> true (submitting request) text='${normalized.slice(0,80)}'`);
        return true;
    }
    log(`[RUNLOG] shouldSkipRunEntry -> false text='${normalized.slice(0,80)}' level='${level}'`);
    return false;
}

function determineRunEntryTone(level, text) {
    const normalizedLevel = (level || '').toLowerCase();
    const rawText = typeof text === 'string' ? text : '';
    const lower = rawText.toLowerCase();

    if (normalizedLevel === 'error') {
        return 'error';
    }
    if (normalizedLevel === 'warning') {
        return 'warning';
    }
    if (normalizedLevel === 'success' || lower.startsWith('✓') || lower.includes('completed')) {
        return 'success';
    }
    if (normalizedLevel === 'agent') {
        return 'accent';
    }
    if (lower.includes('search')) {
        return 'neutral';
    }
    if (lower.startsWith('read') || lower.includes('open ') || lower.includes('loading')) {
        return 'neutral';
    }
    if (lower.includes('execute') || lower.includes('executing') || lower.includes('running') || lower.includes('launching')) {
        return 'accent';
    }

    return 'neutral';
}

function appendRunLogEntry(level, text, format = 'text') {
    log(`[RUNLOG] appendRunLogEntry entered level='${level}' format='${format}' cap=${state.captureRunLogs} currentDoc=${currentDocId}`);

    if (shouldSkipRunEntry(level, text)) {
        log(`[RUNLOG] Skipping entry level='${level}' text='${(text || '').slice(0,80)}'...`);
        ensureRunLogContainer();
        return true;
    }

    const runState = ensureRunLogContainer();
    if (!runState) {
        warn(`[RUNLOG] Could not obtain run container for level='${level}'`);
        return false;
    }

    const entry = { level, text, format };
    // Keep a single source of truth for entries. When the run state is
    // rehydrated, `runState.entries` and `runState.runData.entries` often
    // point to the same array. Pushing to both would duplicate entries that
    // later show up after a document switch. Push once to the canonical array
    // and re-sync both references.
    const entriesArray = (runState.runData && Array.isArray(runState.runData.entries))
        ? runState.runData.entries
        : (Array.isArray(runState.entries) ? runState.entries : []);

    entriesArray.push(entry);
    if (runState.runData) {
        runState.runData.entries = entriesArray;
    }
    runState.entries = entriesArray;

    const itemEl = document.createElement('li');
    itemEl.classList.add('run-entry');

    const tone = determineRunEntryTone(level, text) || 'neutral';
    if (tone && tone !== 'neutral') {
        itemEl.classList.add(`run-entry-tone-${tone}`);
    }

    const markerEl = document.createElement('div');
    markerEl.classList.add('run-entry-marker');

    const dotEl = document.createElement('span');
    dotEl.classList.add('run-entry-dot');
    if (tone && tone !== 'neutral') {
        dotEl.classList.add(`run-entry-dot-${tone}`);
    }
    markerEl.appendChild(dotEl);
    itemEl.appendChild(markerEl);

    const bodyEl = document.createElement('div');
    bodyEl.classList.add('run-entry-body');

    const textEl = document.createElement('span');
    textEl.classList.add('run-entry-text');
    
    // Respect message format - use innerHTML for HTML, textContent for plain text
    if (format === 'html') {
        textEl.innerHTML = text;
    } else if (format === 'markdown') {
        const rendered = (typeof renderMarkdown === 'function') ? renderMarkdown(text) : text;
        if (rendered && rendered.trim()) {
            textEl.innerHTML = rendered;
        } else {
            textEl.textContent = text;
        }
    } else {
        textEl.textContent = text;
    }
    
    // Set title attribute to plain text version for tooltip
    if (format === 'html' || format === 'markdown') {
        // Strip HTML tags for tooltip
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = format === 'markdown' && typeof renderMarkdown === 'function' ? renderMarkdown(text) : text;
        textEl.setAttribute('title', tempDiv.textContent || tempDiv.innerText || text);
    } else {
        textEl.setAttribute('title', text);
    }
    
    bodyEl.appendChild(textEl);

    itemEl.appendChild(bodyEl);

    if (!runState.actionsListEl) {
        warn('[RUNLOG] Missing actionsListEl on runState; cannot append');
        return false;
    }

    runState.actionsListEl.appendChild(itemEl);
    updateRunSummary(runState, entry);
    persistCurrentThreadState();
    log(`[RUNLOG] Appended entry level='${level}' format='${format}' text='${(text || '').slice(0,80)}'...`);
    return true;
}

function updateRunSummary(runState, entry) {
    // Keep summary as "Designing" throughout execution
    // Only update to "Designed" on completion (handled in completed message handler)
    return;
}

function resetReasoningState(options = {}) {
    if (!state.reasoning) {
        state.reasoning = { activeSession: null, sessionCounter: 0 };
        return;
    }

    if (options.forceComplete && state.reasoning.activeSession && !state.reasoning.activeSession.completed) {
        completeActiveReasoning({ reason: 'cancelled', collapse: true });
    }

    state.reasoning.activeSession = null;
    state.reasoning.sessionCounter = 0;
}

function ensureReasoningSession(runState) {
    if (!state.reasoning) {
        state.reasoning = { activeSession: null, sessionCounter: 0 };
    }

    if (state.reasoning.activeSession && !state.reasoning.activeSession.completed) {
        return state.reasoning.activeSession;
    }

    const targetRun = runState || state.activeRun;
    if (!targetRun || !targetRun.actionsListEl) {
        return null;
    }

    return createReasoningSession(targetRun);
}

function createReasoningSession(runState) {
    if (!runState || !runState.actionsListEl) {
        return null;
    }

    state.reasoning.sessionCounter = (state.reasoning.sessionCounter || 0) + 1;
    const index = state.reasoning.sessionCounter;

    const itemEl = document.createElement('li');
    itemEl.classList.add('run-entry', 'reasoning-entry', 'reasoning-live');
    itemEl.dataset.reasoningIndex = String(index);

    const markerEl = document.createElement('div');
    markerEl.classList.add('run-entry-marker');

    const dotEl = document.createElement('span');
    dotEl.classList.add('run-entry-dot', 'run-entry-dot-accent', 'reasoning-dot');
    markerEl.appendChild(dotEl);
    itemEl.appendChild(markerEl);

    const bodyEl = document.createElement('div');
    bodyEl.classList.add('run-entry-body');

    const detailsEl = document.createElement('details');
    detailsEl.classList.add('reasoning-details');
    detailsEl.open = true;

    const summaryEl = document.createElement('summary');
    summaryEl.classList.add('reasoning-summary');

    const previewEl = document.createElement('span');
    previewEl.classList.add('reasoning-preview');
    previewEl.textContent = 'Thinking…';
    summaryEl.appendChild(previewEl);

    const statusWrap = document.createElement('span');
    statusWrap.classList.add('reasoning-status');

    const pulseEl = document.createElement('span');
    pulseEl.classList.add('reasoning-pulse');
    statusWrap.appendChild(pulseEl);

    const statusLabel = document.createElement('span');
    statusLabel.classList.add('reasoning-status-label');
    statusLabel.textContent = 'Thinking';
    statusWrap.appendChild(statusLabel);

    summaryEl.appendChild(statusWrap);
    detailsEl.appendChild(summaryEl);

    const streamEl = document.createElement('div');
    streamEl.classList.add('reasoning-stream');
    streamEl.setAttribute('role', 'log');
    streamEl.setAttribute('aria-live', 'polite');
    streamEl.setAttribute('aria-atomic', 'false');
    const textNode = document.createTextNode('');
    streamEl.appendChild(textNode);
    detailsEl.appendChild(streamEl);

    bodyEl.appendChild(detailsEl);
    itemEl.appendChild(bodyEl);

    runState.actionsListEl.appendChild(itemEl);

    const session = {
        index,
        container: itemEl,
        detailsEl,
        summaryEl,
        previewEl,
        statusLabel,
        pulseEl,
        contentEl: streamEl,
        chunkCount: 0,
        completed: false,
        previewText: '',
        textNode,
        streamText: '',
        userCollapsed: false
    };

    detailsEl.addEventListener('toggle', () => {
        session.userCollapsed = !detailsEl.open;
        try { animateReasoningDetails(detailsEl); } catch (e) { /* no-op */ }
    });

    runState.reasoningSessions = runState.reasoningSessions || [];
    runState.reasoningSessions.push(session);
    state.reasoning.activeSession = session;

    persistCurrentThreadState();
    return session;
}

function appendReasoningChunk(content) {
    const text = typeof content === 'string' ? content : String(content ?? '');
    if (!text.trim()) {
        return;
    }

    startRunLogSession();
    const runState = ensureRunLogContainer();
    if (!runState) {
        return;
    }

    // Check if this chunk starts a new thinking block (bold header pattern)
    // Pattern: **Header** at the start or after newlines, indicating a new section
    const newBlockPattern = /^\s*\*\*[^*]+\*\*/;
    const existingSession = state.reasoning?.activeSession;
    
    // If we have an existing session with content, and this chunk starts with a bold header,
    // complete the old session and start a new one
    if (existingSession && 
        !existingSession.completed && 
        existingSession.streamText && 
        existingSession.streamText.trim().length > 0 &&
        newBlockPattern.test(text)) {
        // Complete the current session before starting new one
        completeActiveReasoning({ reason: 'newblock', statusText: '' });
    }

    const session = ensureReasoningSession(runState);
    if (!session) {
        return;
    }

    session.chunkCount += 1;

    session.streamText = `${session.streamText || ''}${text}`;
    if (session.textNode) {
        session.textNode.textContent = session.streamText;
    } else {
        session.contentEl.textContent = session.streamText;
    }

    setReasoningPreview(session, session.streamText);

    session.container.classList.add('reasoning-live');
    session.container.classList.remove('reasoning-complete');
    session.detailsEl.setAttribute('data-state', 'live');
    if (!session.userCollapsed) {
        session.detailsEl.open = true;
    }

    if (session.statusLabel) {
        session.statusLabel.textContent = 'Thinking';
    }
    if (session.pulseEl) {
        session.pulseEl.classList.remove('hidden');
    }

    session.contentEl.scrollTop = session.contentEl.scrollHeight;
    if (elements.chatMessages) {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }

    persistCurrentThreadState();
}

function setReasoningPreview(session, text) {
    if (!session || !session.previewEl) {
        return;
    }
    const clean = (text || '').replace(/\s+/g, ' ').trim();
    if (!clean) {
        return;
    }
    session.previewText = clean;
    const truncated = truncateText(clean, 72);
    session.previewEl.textContent = truncated;

    if (clean.length > truncated.length) {
        session.previewEl.setAttribute('title', clean);
    } else {
        session.previewEl.removeAttribute('title');
    }
}

function completeActiveReasoning(options = {}) {
    if (!state.reasoning) {
        return;
    }

    const session = state.reasoning.activeSession;
    if (!session || session.completed) {
        return;
    }

    const reason = options.reason || 'complete';
    const statusText = options.statusText !== undefined 
        ? options.statusText 
        : formatReasoningStatusText(reason, options);

    if (session.pulseEl) {
        session.pulseEl.classList.add('hidden');
    }

    // Only set status label if there's text to show
    if (session.statusLabel) {
        if (statusText) {
            session.statusLabel.textContent = statusText;
        } else if (reason !== 'newblock') {
            session.statusLabel.textContent = 'Complete';
        } else {
            // For newblock, hide the status label entirely
            session.statusLabel.textContent = '';
        }
    }

    session.container.classList.remove('reasoning-live');
    session.container.classList.add('reasoning-complete');
    session.detailsEl.setAttribute('data-state', 'complete');

    if (reason === 'tool' && options.operation) {
        const label = formatReasoningOperationName(options.operation);
        if (label && session.previewEl) {
            const text = `NEXT: ${label.toUpperCase()}`;
            session.previewEl.textContent = text;
            session.previewEl.setAttribute('title', text);
        }
    } else if (options.previewText) {
        setReasoningPreview(session, options.previewText);
    }

    if (reason === 'error') {
        session.container.classList.add('reasoning-error');
    } else if (reason === 'cancelled') {
        session.container.classList.add('reasoning-cancelled');
    }

    const shouldCollapse = options.collapse === true || options.collapse === undefined || options.collapse === null;
    if (shouldCollapse) {
        session.detailsEl.open = false;
        session.userCollapsed = true;
    }

    session.completed = true;
    state.reasoning.activeSession = null;
    persistCurrentThreadState();
}

function formatReasoningStatusText(reason, options = {}) {
    switch (reason) {
        case 'tool': {
            const label = formatReasoningOperationName(options.operation);
            return label ? `NEXT: ${label.toUpperCase()}` : 'NEXT ACTION';
        }
        case 'plan':
            return 'PLAN READY';
        case 'cancelled':
            return 'CANCELLED';
        case 'error':
            return 'ERROR';
        case 'newblock':
            return '';  // No status label for block transitions
        default:
            return 'COMPLETE';
    }
}

function formatReasoningOperationName(operation) {
    if (!operation) {
        return '';
    }
    const normalized = String(operation)
        .replace(/[_-]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    if (!normalized) {
        return '';
    }
    return normalized
        .split(' ')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function animateReasoningDetails(detailsEl) {
    const stream = detailsEl.querySelector('.reasoning-stream');
    if (!stream) return;

    const duration = 220;

    if (detailsEl.open) {
        stream.style.willChange = 'max-height, opacity, transform';
        stream.style.overflow = 'hidden';
        stream.style.maxHeight = '0px';
        stream.style.opacity = '0';
        stream.style.transform = 'translateY(-4px)';

        requestAnimationFrame(() => {
            const height = stream.scrollHeight;
            stream.style.transition = `max-height ${duration}ms ease, opacity ${duration}ms ease, transform ${duration}ms ease`;
            stream.style.maxHeight = `${height}px`;
            stream.style.opacity = '1';
            stream.style.transform = 'translateY(0)';
        });

        setTimeout(() => {
            stream.style.maxHeight = '';
            stream.style.overflow = '';
            stream.style.transition = '';
            stream.style.willChange = '';
        }, duration + 20);
    } else {
        const height = stream.scrollHeight;
        stream.style.willChange = 'max-height, opacity, transform';
        stream.style.overflow = 'hidden';
        stream.style.maxHeight = `${height}px`;
        stream.style.opacity = '1';
        stream.style.transform = 'translateY(0)';

        requestAnimationFrame(() => {
            stream.style.transition = `max-height ${duration}ms ease, opacity ${duration}ms ease, transform ${duration}ms ease`;
            stream.style.maxHeight = '0px';
            stream.style.opacity = '0';
            stream.style.transform = 'translateY(-4px)';
        });

        setTimeout(() => {
            stream.style.maxHeight = '';
            stream.style.overflow = '';
            stream.style.transition = '';
            stream.style.willChange = '';
        }, duration + 20);
    }
}

function setRunStatus(status) {
    const runState = state.activeRun;
    if (!runState || !runState.container) {
        return;
    }

    runState.container.classList.remove(
        'run-in-progress',
        'run-complete-success',
        'run-complete-error',
        'run-complete-cancelled'
    );

    if (status === 'success') {
        runState.container.classList.add('run-complete-success');
    } else if (status === 'error') {
        runState.container.classList.add('run-complete-error');
    } else if (status === 'cancelled') {
        runState.container.classList.add('run-complete-cancelled');
    } else {
        runState.container.classList.add('run-in-progress');
    }

    runState.status = status;
    if (runState.runData) {
        runState.runData.status = status;
        if (status === 'success') {
            runState.runData.summaryText = 'Designed';
        }
    }
    const docState = getDocState(runState.docId || currentDocId, false);
    if (docState) {
        docState.lastStatus = status;
        if (docState.activeRunId && status !== 'in-progress') {
            docState.activeRunId = null;
        }
        docState.lastUpdated = Date.now();
        persistDocState(docState.id);
    }
    persistCurrentThreadState();
}

function finalizeActiveRun(status = 'success') {
    if (!state.activeRun) {
        state.captureRunLogs = false;
        state.cancelRequested = false;
        return;
    }

    // Keep the run log expanded when a turn finishes.
    if (state.activeRun.detailsEl) {
        state.activeRun.detailsEl.open = true;
    }

    setRunStatus(status);
    state.captureRunLogs = false;
    state.activeRun = null;
    state.cancelRequested = false;
}

function shouldAutoStartRunLog(level, message) {
    if (!message) {
        return false;
    }

    const trimmed = String(message).trim();
    if (!trimmed) {
        return false;
    }

    const lower = trimmed.toLowerCase();

    if (['success', 'error', 'warning'].includes(level)) {
        if (hasActiveRun()) {
            return true;
        }
        if (trimmed.startsWith('✓') || trimmed.startsWith('✗')) {
            return true;
        }
        if (lower.includes('operation') || lower.startsWith('run ') || lower.startsWith('execution')) {
            return true;
        }
        return false;
    }

    if (trimmed.startsWith('✓') || trimmed.startsWith('✗')) {
        return true;
    }

    if (lower.startsWith('executing')) {
        return true;
    }

    if (lower.includes('operation')) {
        return true;
    }

    if (lower.startsWith('plan approved') || lower.startsWith('plan rejected')) {
        return true;
    }

    return false;
}

/**
 * Add entry to activity log
 */
function addLog(type, message, options = {}) {
    const logType = type || 'info';
    const scopePreference = options.scope || 'auto';
    const formatPreference = typeof options.messageFormat === 'string'
        ? options.messageFormat.trim().toLowerCase()
        : null;

    const preState = {
        captureRunLogs: state.captureRunLogs,
        scopePreference,
        logType,
        currentDocId,
    };

    if (scopePreference === 'run') {
        log(`[RUNLOG] addLog scope=run type='${logType}' msgPreview='${(message || '').slice(0,60)}' cap=${state.captureRunLogs}`);
        startRunLogSession({ forceNew: false });
    } else if (scopePreference === 'auto' && !state.captureRunLogs && shouldAutoStartRunLog(logType, message)) {
        log(`[RUNLOG] addLog auto-start run for type='${logType}' msgPreview='${(message || '').slice(0,60)}'`);
        startRunLogSession();
    }

    const wantsRunLog = scopePreference === 'run' || (scopePreference === 'auto' && state.captureRunLogs);

    if (wantsRunLog && state.captureRunLogs) {
        const appended = appendRunLogEntry(logType, message, formatPreference || 'text');
        if (!appended) {
            warn(`[RUNLOG] addLog failed to append (wantsRunLog=${wantsRunLog}) type='${logType}' doc=${currentDocId}`);
        }
        return appended;
    }

    // Fall back to chat bubble when not capturing run logs
    let sender = 'System';
    let variant = logType;
    let dismissHero = true;

    if (logType === 'agent') {
        sender = 'CADAgent';
        variant = 'agent';
    } else if (logType === 'info') {
        sender = 'System';
        variant = 'info';
        dismissHero = false;
    } else if (logType === 'success') {
        sender = 'Success';
    } else if (logType === 'error') {
        sender = 'Error';
    } else if (logType === 'warning') {
        sender = 'Warning';
    }

    if (typeof options.dismissHero === 'boolean') {
        dismissHero = options.dismissHero;
    }

    const appendOptions = { sender, variant, dismissHero };
    if (formatPreference === 'markdown' || formatPreference === 'html' || formatPreference === 'text' || formatPreference === 'plaintext') {
        appendOptions.contentFormat = formatPreference === 'plaintext' ? 'text' : formatPreference;
    }

    appendMessage('assistant', message, appendOptions);
}

/**
 * Update connection status
 */
function updateConnectionStatus(connected, sessionId = null, docId = null) {
    console.log('='.repeat(60));
    console.log('UPDATE CONNECTION STATUS');
    console.log('='.repeat(60));
    console.log('→ connected:', connected);
    console.log('→ sessionId:', sessionId);

    const targetDocId = resolveDocId(docId);
    if ((!currentDocId || currentDocId === 'default') && targetDocId) {
        migrateDefaultStateTo(targetDocId);
        currentDocId = targetDocId;
        updateDocOrder(targetDocId);
        renderActiveDoc();
        flushPendingMessages(targetDocId);
    }
    const docState = getDocState(targetDocId, true);
    const previouslyConnected = docState ? docState.connected : state.connected;
    const previousSessionId = docState ? docState.sessionId : state.sessionId;

    if (docState) {
        docState.connected = connected;
        if (sessionId) docState.sessionId = sessionId;
        docState.lastStatus = connected ? docState.lastStatus : 'idle';
        docState.lastUpdated = Date.now();
        persistDocState(docState.id);
    }

    state.connected = connected; // legacy fallback
    state.sessionId = sessionId || state.sessionId;

    const isActiveDoc = targetDocId === currentDocId || (!currentDocId && targetDocId);
    const statusDot = elements.connectionStatus ? elements.connectionStatus.querySelector('.status-dot') : null;

    if (isActiveDoc && statusDot && elements.statusText && elements.sessionId) {
        if (connected) {
            console.log('→ Setting UI to CONNECTED');
            statusDot.classList.remove('disconnected');
            statusDot.classList.add('connected');
            elements.statusText.textContent = 'Connected';
            const displaySessionId = sessionId ? `${sessionId.substring(0, 8)}...` : '-';
            elements.sessionId.textContent = displaySessionId;
            if (!previouslyConnected) {
                addLog('success', 'Connected to backend', { scope: 'global' });
            }
        } else {
            console.log('→ Setting UI to DISCONNECTED');
            statusDot.classList.remove('connected');
            statusDot.classList.add('disconnected');
            elements.statusText.textContent = 'Disconnected';
            elements.sessionId.textContent = '-';
            if (previouslyConnected) {
                addLog('error', 'Disconnected from backend', { scope: 'global' });
                if (state.activeRun) {
                    appendRunLogEntry('error', 'Connection lost – run aborted');
                    finalizeActiveRun('error');
                } else {
                    resetRunLogState();
                }
            }
        }
    }

    // Show/hide reconnect button based on connection status (active doc only)
    if (elements.reconnectBtn && isActiveDoc) {
        if (connected) {
            elements.reconnectBtn.classList.add('hidden');
        } else {
            elements.reconnectBtn.classList.remove('hidden');
        }
    }

    updateDocSwitcherUI();
    updateExecuteButtonState();
    console.log('→ Execute button disabled:', elements.executeBtn.disabled);
    console.log('='.repeat(60));

    if (!connected || (sessionId && sessionId !== previousSessionId)) {
        state.executeRequestCount = 0;

        // Clear checkpoint state on disconnect or session change to prevent stale associations
        if (state.pendingCheckpointQueue && state.pendingCheckpointQueue.length > 0) {
            console.log('Clearing pending checkpoint queue due to session change/disconnect');
            state.pendingCheckpointQueue = [];
        }

        // Remove stale data-message-id and data-request-id attributes from DOM
        if (elements.chatMessages) {
            const messagesWithCheckpoints = elements.chatMessages.querySelectorAll('[data-message-id]');
            if (messagesWithCheckpoints.length > 0) {
                console.log('Removing', messagesWithCheckpoints.length, 'stale checkpoint associations');
                messagesWithCheckpoints.forEach(el => {
                    el.removeAttribute('data-message-id');
                    el.removeAttribute('data-request-id');  // Also clear request correlation IDs
                    // Also remove the revert button
                    const revertBtn = el.querySelector('.revert-btn');
                    if (revertBtn) {
                        revertBtn.remove();
                    }
                });
            }
        }
    }
}

/**
 * Show progress indicator
 */
function showProgress(message, progress = null) {
    // Keep the deprecated toast hidden but preserve internal state updates.
    state.processing = true;
    if (elements.progressSection) {
        elements.progressSection.classList.add('hidden');
    }
    if (elements.progressText) {
        elements.progressText.textContent = message || '';
    }
    if (elements.progressFill) {
        if (progress !== null) {
            elements.progressFill.style.width = `${progress}%`;
        } else {
            elements.progressFill.style.width = '0%';
        }
    }
    if (elements.executeBtn) {
        elements.executeBtn.disabled = true;
    }
    updateRevertButtonStates();
}

/**
 * Hide progress indicator
 */
function hideProgress() {
    if (elements.progressSection) {
        elements.progressSection.classList.add('hidden');
    }
    if (elements.progressFill) {
        elements.progressFill.style.width = '0%';
    }
    state.processing = false;
    updateExecuteButtonState();
    updateRevertButtonStates();
}

/**
 * Append a message bubble to the chat transcript
 */
function appendMessage(role, text, options = {}) {
    if (!elements.chatMessages) {
        return;
    }

    const defaultDismiss = role === 'user' || role === 'assistant';
    const shouldDismissHero = options.dismissHero !== undefined ? options.dismissHero : defaultDismiss;

    if (shouldDismissHero && elements.welcomeMessage && !elements.welcomeMessage.classList.contains('hidden')) {
        elements.welcomeMessage.classList.add('hidden');
    }

    if (shouldDismissHero && elements.headerSection && !elements.headerSection.classList.contains('compact-mode')) {
        elements.headerSection.classList.add('compact-mode');
    }

    const messageEl = document.createElement('div');
    messageEl.classList.add('message');

    const roleClass = role === 'user' ? 'user-message' : role === 'system' ? 'system-message' : 'assistant-message';
    messageEl.classList.add(roleClass);

    if (options.variant) {
        messageEl.classList.add(options.variant);
    }

    // Store message_id for checkpoint revert functionality
    if (role === 'user' && options.messageId) {
        messageEl.setAttribute('data-message-id', options.messageId);
    }

    // Store request_id for checkpoint correlation (new approach)
    if (role === 'user' && options.requestId) {
        messageEl.setAttribute('data-request-id', options.requestId);
    }

    const contentEl = document.createElement('div');
    contentEl.classList.add('message-content');

    const headerEl = document.createElement('div');
    headerEl.classList.add('message-header');

    const senderEl = document.createElement('span');
    senderEl.classList.add('message-sender');
    senderEl.textContent = options.sender || (role === 'user' ? 'You' : 'CADAgent');

    headerEl.appendChild(senderEl);

    // Add revert button for user messages with checkpoints
    if (role === 'user' && options.messageId) {
        const actionsEl = document.createElement('div');
        actionsEl.classList.add('message-actions');

        const revertBtn = document.createElement('button');
        revertBtn.classList.add('revert-btn');
        revertBtn.setAttribute('title', 'Revert timeline to this state');
        revertBtn.setAttribute('aria-label', 'Revert timeline to this state');
        revertBtn.innerHTML = '<span class="revert-icon">⟲</span> Revert';

        actionsEl.appendChild(revertBtn);
        headerEl.appendChild(actionsEl);
    }

    contentEl.appendChild(headerEl);

    const textEl = document.createElement('div');
    textEl.classList.add('message-text');
    const defaultFormat = (role === 'assistant' || role === 'system') ? 'html' : 'text';
    const requestedFormat = typeof options.contentFormat === 'string' ? options.contentFormat : defaultFormat;
    const format = requestedFormat || defaultFormat;

    console.log('appendMessage DEBUG - role:', role, 'options.contentFormat:', options.contentFormat, 'final format:', format);
    console.log('appendMessage DEBUG - text preview:', text.substring(0, 100));

    if (format === 'markdown') {
        console.log('appendMessage: Using MARKDOWN rendering');
        const rendered = (typeof renderMarkdown === 'function') ? renderMarkdown(text) : '';
        if (rendered && rendered.trim()) {
            textEl.innerHTML = rendered;
        } else {
            textEl.textContent = text;
        }
    } else if (format === 'text') {
        console.log('appendMessage: Using TEXT rendering (textContent)');
        textEl.textContent = text;
    } else {
        console.log('appendMessage: Using HTML rendering (innerHTML)');
        textEl.innerHTML = text;
    }

    contentEl.appendChild(textEl);
    messageEl.appendChild(contentEl);

    elements.chatMessages.appendChild(messageEl);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    // Persist to doc state unless rendering from stored data
    const shouldStore = options.__skipStore !== true;
    if (shouldStore && currentDocId) {
        const docState = getDocState(currentDocId, true);
        docState.items.push({
            type: 'message',
            role,
            text,
            options: {
                sender: options.sender,
                variant: options.variant,
                contentFormat: format,
                messageId: options.messageId || null
            }
        });
        docState.lastUpdated = Date.now();
        persistDocState(currentDocId);
    }

    // Track user messages for checkpoint association
    if (role === 'user') {
        state.pendingUserMessage = messageEl;
        processPendingCheckpoints({ skipPersist: true });
    }

    persistCurrentThreadState();
}

function saveCurrentThread() {
    if (!currentDocId) return;
    persistCurrentThreadState();
}

function loadThread(docId) {
    currentDocId = docId;
    renderActiveDoc();
    // Reset run state and deliver any queued messages for this document
    resetRunLogState();
    flushPendingMessages(docId);
    processPendingCheckpoints({ skipPersist: true });
    persistCurrentThreadState();
}

function switchToDocument(docId) {
    if (!docId || docId === currentDocId) return;
    saveCurrentThread();
    currentDocId = docId;
    updateDocOrder(docId);
    renderActiveDoc();
    updateDocSwitcherUI();
    flushPendingMessages(docId);
    processPendingCheckpoints({ skipPersist: true });
    syncProcessingState(docId);
    const target = getDocState(docId, false);
    if (target) {
        showToast(`Switched to ${target.name || 'design'} — history restored`);
    }
    if (elements.cadRequest) {
        elements.cadRequest.focus();
    }
}

// Track environment and browser mode
let browserModeWarningShown = false;
let inFusion360 = false;
let fusionBridgeReady = false;
let environmentCheckTimer = null;
let environmentCheckAttempts = 0;
let handshakeTimer = null;

const FUSION_LOG_FORWARDING_ENABLED = false;
const LOG_FORWARDING_WINDOW_MS = 1000;
// DEBUG: loosen throttle so we can see palette probes during run-log investigation
const LOG_FORWARDING_MAX_PER_WINDOW = 100;
const LOG_FORWARDING_MAX_MESSAGE_LENGTH = 1200;

const _fallbackConsoleFn = () => {};
const _originalConsole = {
    log: (typeof console !== 'undefined' && typeof console.log === 'function') ? console.log.bind(console) : _fallbackConsoleFn,
    warn: (typeof console !== 'undefined' && typeof console.warn === 'function') ? console.warn.bind(console) : _fallbackConsoleFn,
    error: (typeof console !== 'undefined' && typeof console.error === 'function') ? console.error.bind(console) : _fallbackConsoleFn
};

const _fusionLogState = {
    windowStart: 0,
    sent: 0,
    suppressed: 0,
    guard: false
};

let _fusionConsoleOverridden = false;

function _resetFusionLogThrottle() {
    _fusionLogState.windowStart = Date.now();
    _fusionLogState.sent = 0;
    _fusionLogState.suppressed = 0;
}

function _safeSerializeValue(value, seen) {
    if (value === null) {
        return 'null';
    }
    if (value === undefined) {
        return 'undefined';
    }
    if (typeof value === 'string') {
        return value;
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
        return String(value);
    }
    if (typeof value === 'symbol') {
        return value.toString();
    }
    if (value instanceof Error) {
        return value.stack || value.message || String(value);
    }

    if (typeof value === 'object') {
        if (seen.has(value)) {
            return '[Circular]';
        }
        seen.add(value);
        try {
            const serialized = JSON.stringify(value, (_key, val) => {
                if (typeof val === 'undefined') {
                    return 'undefined';
                }
                return val;
            });
            seen.delete(value);
            return serialized;
        } catch (error) {
            seen.delete(value);
            return Object.prototype.toString.call(value);
        }
    }

    return String(value);
}

function _serializeLogArguments(args) {
    if (!args || args.length === 0) {
        return '';
    }
    const seen = new WeakSet();
    const combined = args.map((arg) => _safeSerializeValue(arg, seen)).join(' ');
    if (combined.length > LOG_FORWARDING_MAX_MESSAGE_LENGTH) {
        return `${combined.slice(0, LOG_FORWARDING_MAX_MESSAGE_LENGTH)}…`;
    }
    return combined;
}

function _canForwardFusionLog() {
    return FUSION_LOG_FORWARDING_ENABLED
        && fusionBridgeReady
        && fusionBridgeAvailable()
        && !_fusionLogState.guard;
}

function _forwardLogToFusion(level, args) {
    if (!_canForwardFusionLog()) {
        return;
    }

    const now = Date.now();
    if (now - _fusionLogState.windowStart >= LOG_FORWARDING_WINDOW_MS) {
        _fusionLogState.windowStart = now;
        _fusionLogState.sent = 0;
    }

    if (_fusionLogState.sent >= LOG_FORWARDING_MAX_PER_WINDOW) {
        _fusionLogState.suppressed += 1;
        return;
    }

    const serialized = _serializeLogArguments(args);
    if (!serialized) {
        return;
    }

    const suffix = _fusionLogState.suppressed > 0 ? ` (+${_fusionLogState.suppressed} suppressed)` : '';
    const payloadMessage = `${serialized}${suffix}`;
    const payloadData = {
        action: 'log_to_fusion',
        level,
        message: payloadMessage,
        ts: new Date().toISOString()
    };

    try {
        _fusionLogState.guard = true;
        const sent = sendToAddin(payloadData);
        if (sent) {
            _fusionLogState.sent += 1;
            _fusionLogState.suppressed = 0;
        } else {
            _fusionLogState.suppressed += 1;
        }
    } catch (error) {
        _fusionLogState.suppressed += 1;
        _originalConsole.error('[CADAgent] Failed to forward log to Fusion:', error);
    } finally {
        _fusionLogState.guard = false;
    }
}

function _createForwardingMethod(level) {
    return function (...args) {
        _originalConsole[level](...args);
        _forwardLogToFusion(level, args);
    };
}

const fusionPaletteLogger = {
    log: _createForwardingMethod('log'),
    warn: _createForwardingMethod('warn'),
    error: _createForwardingMethod('error')
};

const log = fusionPaletteLogger.log;
const warn = fusionPaletteLogger.warn;
const error = fusionPaletteLogger.error;

function _overrideConsoleMethod(level) {
    if (typeof console === 'undefined' || typeof console[level] !== 'function') {
        return;
    }
    console[level] = (...args) => fusionPaletteLogger[level](...args);
}

function initializeFusionLogForwarding() {
    if (!FUSION_LOG_FORWARDING_ENABLED || _fusionConsoleOverridden) {
        return;
    }
    _fusionConsoleOverridden = true;
    _resetFusionLogThrottle();
    _overrideConsoleMethod('log');
    _overrideConsoleMethod('warn');
    _overrideConsoleMethod('error');
}


// Message queue for add-in communication before handshake completes
let pendingToAddin = [];
const PENDING_QUEUE_CAP = 5;
const AUTH_ACTIONS = ['check_and_handle_signup', 'send_otp_code', 'verify_otp_code', 'auth_callback', 'logout', 'login_with_password', 'save_api_keys', 'get_api_keys_status', 'open_external_url'];

// Retry mechanism state
let retryTimer = null;
let retryDelayMs = 300; // Start at 300ms
const RETRY_DELAY_MIN_MS = 300;
const RETRY_DELAY_MAX_MS = 10000; // Cap at 10 seconds
const RETRY_JITTER_MAX_MS = 100; // +/- 100ms jitter

// Auth watchdog state
let authWatchdogTimer = null;
const AUTH_WATCHDOG_TIMEOUT_MS = 20000; // 20 seconds (increased for slow networks)
let authWatchdogRetryAttempted = false;
let authVerificationCompleted = false;  // Guard against retries after success/error

// Telemetry counters with enhanced tracking
let queueTelemetry = {
    enqueued: 0,
    flushed: 0,
    dropped_full: 0,
    dropped_duplicate: 0,
    retry_count: 0
};

const ENVIRONMENT_CHECK_INTERVAL_MS = 250;
const ENVIRONMENT_MAX_ATTEMPTS = 40;
const ENVIRONMENT_FALLBACK_INTERVAL_MS = 2000;
const HANDSHAKE_RETRY_INTERVAL_MS = 1500;

/**
 * Check if message should coalesce with existing queued message
 */
function shouldCoalesceMessage(newMsg, existingMsg) {
    if (!newMsg.action || !existingMsg.action) return false;

    // For auth actions, replace existing with same action
    if (AUTH_ACTIONS.includes(newMsg.action)) {
        return newMsg.action === existingMsg.action;
    }

    // For other actions, check full equality
    return JSON.stringify(newMsg) === JSON.stringify(existingMsg);
}

/**
 * Enqueue message to pending queue with deduplication and timestamps
 */
function enqueueToAddin(data, reason) {
    const action = data.action || 'unknown';
    const enqueuedAt = Date.now();

    // Check for duplicate/coalescable message
    const existingIdx = pendingToAddin.findIndex(item => shouldCoalesceMessage(data, item.data));
    if (existingIdx !== -1) {
        // Replace existing auth action with latest
        if (AUTH_ACTIONS.includes(action)) {
            console.log(`[Queue] Coalescing ${action} (replacing older)`);
            pendingToAddin[existingIdx] = {
                data,
                options: { requireHandshake: false },
                enqueuedAt,
                retryCount: pendingToAddin[existingIdx].retryCount || 0
            };
        } else {
            console.log(`[Queue] Dropping duplicate ${action}`);
            queueTelemetry.dropped_duplicate++;
        }
        return;
    }

    // Check queue capacity
    if (pendingToAddin.length >= PENDING_QUEUE_CAP) {
        // Try to drop oldest non-auth entry
        let droppedIdx = -1;
        for (let i = 0; i < pendingToAddin.length; i++) {
            if (!AUTH_ACTIONS.includes(pendingToAddin[i].data.action)) {
                droppedIdx = i;
                break;
            }
        }

        if (droppedIdx !== -1) {
            const dropped = pendingToAddin.splice(droppedIdx, 1)[0];
            console.warn(`[Queue] Capacity reached, dropped non-auth: ${dropped.data.action}`);
            queueTelemetry.dropped_full++;
        } else {
            // All auth - drop oldest
            const dropped = pendingToAddin.shift();
            console.warn(`[Queue] Capacity reached, dropped oldest: ${dropped.data.action}`);
            queueTelemetry.dropped_full++;
        }
    }

    // Enqueue message with timestamp
    pendingToAddin.push({
        data,
        options: { requireHandshake: false },
        enqueuedAt,
        retryCount: 0
    });
    queueTelemetry.enqueued++;
    console.log(`[Queue] Enqueued ${action} (reason: ${reason}, queue size: ${pendingToAddin.length})`);

    // Arm retry timer if not already active
    scheduleRetryFlush();
}

/**
 * Flush pending messages to add-in with self-healing retry logic
 */
function flushAddinQueue() {
    if (pendingToAddin.length === 0) {
        // Queue empty - force reset UI state and clear retry timer
        resetQueueUIState();
        clearRetryTimer();
        return;
    }

    // Calculate and log telemetry
    const telemetry = getQueueTelemetry();
    console.log(`[Queue] Flushing ${pendingToAddin.length} pending messages`, telemetry);

    // Warn if oldest message is stale
    if (telemetry.oldest_age_ms > 5000) {
        console.warn(`[Queue] ⚠️ Oldest message age: ${telemetry.oldest_age_ms}ms (threshold: 5000ms)`);
    }

    const toFlush = [...pendingToAddin];
    pendingToAddin = [];

    // Track which items failed for non-blocking requeue
    const failedItems = [];

    for (let i = 0; i < toFlush.length; i++) {
        const item = toFlush[i];
        const { data, options } = item;

        try {
            const wasSent = sendToAddin(data, options);
            if (wasSent) {
                queueTelemetry.flushed++;
                // Message sent successfully - continue to next
            } else {
                // sendToAddin returned false - re-queue this item only, continue with others
                item.retryCount = (item.retryCount || 0) + 1;
                failedItems.push(item);
                console.warn(`[Queue] Send failed for ${data.action} (retry ${item.retryCount}), continuing flush`);
            }
        } catch (error) {
            // Exception during send - re-queue this item only, continue with others
            console.error(`[Queue] Exception flushing ${data.action}:`, error);
            item.retryCount = (item.retryCount || 0) + 1;
            failedItems.push(item);
        }
    }

    // Non-blocking: Re-queue only failed items, preserve already-sent entries
    if (failedItems.length > 0) {
        pendingToAddin = failedItems;
        queueTelemetry.retry_count += failedItems.length;
        console.warn(`[Queue] ${failedItems.length} messages re-queued for retry`);

        // Schedule next retry with backoff
        scheduleRetryFlush();
    } else {
        // All messages sent successfully
        console.log(`[Queue] ✓ Flush complete. Telemetry:`, queueTelemetry);
        resetQueueUIState();
        clearRetryTimer();
    }
}

/**
 * Schedule retry flush with exponential backoff + jitter
 */
function scheduleRetryFlush() {
    // Don't schedule if timer already active
    if (retryTimer !== null) {
        return;
    }

    // Calculate delay with exponential backoff
    const jitter = Math.random() * RETRY_JITTER_MAX_MS * 2 - RETRY_JITTER_MAX_MS; // +/- jitter
    const delayWithJitter = Math.max(0, retryDelayMs + jitter);

    console.log(`[Queue] Scheduling retry in ${delayWithJitter.toFixed(0)}ms (base: ${retryDelayMs}ms, jitter: ${jitter.toFixed(0)}ms)`);

    retryTimer = setTimeout(() => {
        retryTimer = null;
        console.log(`[Queue] Retry timer fired, attempting flush`);
        flushAddinQueue();

        // Increase delay for next retry (exponential backoff)
        retryDelayMs = Math.min(retryDelayMs * 2, RETRY_DELAY_MAX_MS);
    }, delayWithJitter);
}

/**
 * Clear retry timer and reset delay
 */
function clearRetryTimer() {
    if (retryTimer !== null) {
        clearTimeout(retryTimer);
        retryTimer = null;
        console.log(`[Queue] Retry timer cleared`);
    }
    // Reset delay to minimum for next time
    retryDelayMs = RETRY_DELAY_MIN_MS;
}

/**
 * Get comprehensive queue telemetry
 */
function getQueueTelemetry() {
    const now = Date.now();
    let oldest_age_ms = 0;

    if (pendingToAddin.length > 0) {
        const ages = pendingToAddin.map(item => now - item.enqueuedAt);
        oldest_age_ms = Math.max(...ages);
    }

    return {
        queue_depth: pendingToAddin.length,
        retry_count: queueTelemetry.retry_count,
        oldest_age_ms,
        retry_timer_active: retryTimer !== null,
        enqueued: queueTelemetry.enqueued,
        flushed: queueTelemetry.flushed,
        dropped_full: queueTelemetry.dropped_full,
        dropped_duplicate: queueTelemetry.dropped_duplicate
    };
}

/**
 * Force-reset UI state when queue empties (prevents stuck spinners)
 */
function resetQueueUIState() {
    console.log(`[Queue] Force-resetting UI state (queue empty)`);

    // Reset OTP verify button
    if (elements.verifyOtpBtn) {
        elements.verifyOtpBtn.disabled = false;
        elements.verifyOtpBtn.textContent = 'Verify Code';
    }

    // Reset send OTP button
    if (elements.sendOtpBtn && !elements.sendOtpBtn.dataset.cooldown) {
        elements.sendOtpBtn.disabled = false;
        elements.sendOtpBtn.textContent = 'Continue';
    }
}

/**
 * Start auth watchdog for OTP verification (20s timeout)
 */
function startAuthWatchdog() {
    // Clear any existing watchdog timer (but don't reset retry state)
    if (authWatchdogTimer !== null) {
        clearTimeout(authWatchdogTimer);
        authWatchdogTimer = null;
    }

    console.log(`[Queue] Starting auth watchdog (timeout: ${AUTH_WATCHDOG_TIMEOUT_MS}ms, retryAttempted: ${authWatchdogRetryAttempted})`);

    authWatchdogTimer = setTimeout(() => {
        // Guard: If verification already completed, do nothing
        if (authVerificationCompleted) {
            console.log(`[Queue] Watchdog fired but verification already completed, ignoring`);
            clearAuthWatchdog();
            return;
        }
        
        console.warn(`[Queue] ⚠️ Auth watchdog triggered - no auth_success/auth_error received within ${AUTH_WATCHDOG_TIMEOUT_MS}ms`);

        if (!authWatchdogRetryAttempted) {
            // First timeout - retry once
            authWatchdogRetryAttempted = true;

            // Show toast to user
            showLoginStatus('Retrying verification...', 'info');

            // Find verify_otp_code in queue and re-queue it if not present
            const hasVerifyOtp = pendingToAddin.some(item => item.data.action === 'verify_otp_code');

            if (!hasVerifyOtp && elements.emailInput && elements.otpInput) {
                const email = elements.emailInput.value.trim();
                const code = elements.otpInput.value.trim();

                if (email && code) {
                    console.log(`[Queue] Re-queuing verify_otp_code for retry`);
                    enqueueToAddin({
                        action: 'verify_otp_code',
                        email,
                        code
                    }, 'watchdog_retry');
                }
            }

            // Re-arm watchdog for final attempt (do NOT call startAuthWatchdog recursively
            // to avoid resetting authWatchdogRetryAttempted)
            authWatchdogTimer = setTimeout(() => {
                // Guard: If verification already completed, do nothing
                if (authVerificationCompleted) {
                    console.log(`[Queue] Second watchdog fired but verification already completed, ignoring`);
                    clearAuthWatchdog();
                    return;
                }
                
                // Second timeout - give up, reset UI
                console.error(`[Queue] Auth watchdog retry failed, resetting UI`);
                showLoginStatus('Verification timeout. Please try again.', 'error');
                resetQueueUIState();
                clearAuthWatchdog();
            }, AUTH_WATCHDOG_TIMEOUT_MS);
        } else {
            // Already retried once (shouldn't reach here with new logic, but safety fallback)
            console.error(`[Queue] Auth watchdog retry already attempted, resetting UI`);
            showLoginStatus('Verification timeout. Please try again.', 'error');
            resetQueueUIState();
            clearAuthWatchdog();
        }
    }, AUTH_WATCHDOG_TIMEOUT_MS);
}

/**
 * Clear auth watchdog and mark verification as complete
 */
function clearAuthWatchdog() {
    if (authWatchdogTimer !== null) {
        clearTimeout(authWatchdogTimer);
        authWatchdogTimer = null;
        console.log(`[Queue] Auth watchdog cleared`);
    }
    // Reset for next verification attempt (will be set false again in handleVerifyOtp)
    authWatchdogRetryAttempted = false;
}

/**
 * Determine whether Fusion's JavaScript bridge is available
 */
function fusionBridgeAvailable() {
    return typeof window.adsk !== 'undefined'
        && !!window.adsk
        && typeof window.adsk.fusionSendData === 'function';
}

/**
 * Apply environment-specific UI updates
 */
function applyEnvironmentState(isFusion) {
    inFusion360 = isFusion;

    const statusDot = elements.connectionStatus
        ? elements.connectionStatus.querySelector('.status-dot')
        : null;

    if (inFusion360) {
        console.log('[CADAgent] ✓ Fusion 360 bridge detected - switching to palette mode');

        if (elements.statusText) {
            elements.statusText.textContent = fusionBridgeReady
                ? (state.connected ? 'Connected' : 'Disconnected')
                : 'Connecting...';
        }

        if (statusDot) {
            statusDot.classList.remove('connected');
            statusDot.classList.add('disconnected');
        }

        if (elements.sessionId) {
            if (state.sessionId) {
                const displaySessionId = `${state.sessionId.substring(0, 8)}...`;
                elements.sessionId.textContent = displaySessionId;
            } else {
                elements.sessionId.textContent = '-';
            }
        }
    } else {
        if (!browserModeWarningShown) {
            console.warn('[CADAgent] ❌ Fusion 360 API not detected - running in browser mode');
            console.warn('[CADAgent] This HTML is intended to run inside a Fusion 360 palette');
            browserModeWarningShown = true;
        }

        fusionBridgeReady = false;
        if (handshakeTimer) {
            clearTimeout(handshakeTimer);
            handshakeTimer = null;
        }

        if (elements.statusText) {
            elements.statusText.textContent = 'Browser Mode';
        }

        if (statusDot) {
            statusDot.classList.remove('connected');
            statusDot.classList.add('disconnected');
        }

        if (elements.sessionId) {
            elements.sessionId.textContent = 'N/A';
        }
    }
}

/**
 * Schedule the next environment probe
 */
function scheduleEnvironmentProbe(delayMs) {
    if (environmentCheckTimer) {
        clearTimeout(environmentCheckTimer);
    }

    environmentCheckTimer = setTimeout(() => {
        environmentCheckTimer = null;
        attemptEnvironmentDetection();
    }, delayMs);
}

/**
 * Attempt to detect the Fusion host environment
 */
function attemptEnvironmentDetection() {
    console.log('='.repeat(60));
    console.log('ENVIRONMENT PROBE');
    console.log('='.repeat(60));
    console.log('→ Probe attempt:', environmentCheckAttempts + 1);

    if (fusionBridgeAvailable()) {
        console.log('→ Fusion bridge detected');
        applyEnvironmentState(true);
        environmentCheckAttempts = 0;
        scheduleHandshake(0);
        console.log('='.repeat(60));
        return;
    }

    console.log('→ Fusion bridge not available yet');

    environmentCheckAttempts += 1;
    if (environmentCheckAttempts === 1) {
        applyEnvironmentState(false);
    }
    if (environmentCheckAttempts >= ENVIRONMENT_MAX_ATTEMPTS) {
        applyEnvironmentState(false);
        scheduleEnvironmentProbe(ENVIRONMENT_FALLBACK_INTERVAL_MS);
    } else {
        scheduleEnvironmentProbe(ENVIRONMENT_CHECK_INTERVAL_MS);
    }

    console.log('→ Next probe scheduled');
    console.log('='.repeat(60));
}

/**
 * Detect environment on load
 */
function detectEnvironment() {
    console.log('='.repeat(60));
    console.log('DETECTING ENVIRONMENT');
    console.log('='.repeat(60));
    environmentCheckAttempts = 0;
    attemptEnvironmentDetection();
}

/**
 * Schedule/retry a Fusion handshake
 */
function scheduleHandshake(delayMs) {
    if (fusionBridgeReady) {
        if (handshakeTimer) {
            clearTimeout(handshakeTimer);
            handshakeTimer = null;
        }
        return;
    }

    if (!fusionBridgeAvailable()) {
        return;
    }

    if (handshakeTimer) {
        clearTimeout(handshakeTimer);
    }

    handshakeTimer = setTimeout(() => {
        handshakeTimer = null;
        initiateHandshake();
    }, Math.max(0, delayMs));
}

/**
 * Initiate handshake with the Fusion add-in
 */
function initiateHandshake() {
    if (!fusionBridgeAvailable()) {
        return;
    }

    console.log('[CADAgent] → Initiating Fusion handshake');

    const payload = {
        action: 'handshake',
        app_version: window.SPACE_APP_VERSION || null,
        timestamp: Date.now()
    };

    sendToAddin(payload, { requireHandshake: false });

    if (!fusionBridgeReady) {
        scheduleHandshake(HANDSHAKE_RETRY_INTERVAL_MS);
    }
}

/**
 * Send message to Python add-in
 *
 * @returns {boolean} true if message was sent, false if queued/failed
 */
function sendToAddin(data, options = {}) {
    const { requireHandshake = true } = options;

    if (!fusionBridgeAvailable()) {
        enqueueToAddin(data, 'bridge_unavailable');
        applyEnvironmentState(false);
        scheduleEnvironmentProbe(ENVIRONMENT_CHECK_INTERVAL_MS);
        return false;
    }

    if (requireHandshake && !fusionBridgeReady) {
        enqueueToAddin(data, 'handshake_incomplete');
        scheduleHandshake(0);
        return false;
    }

    console.log('='.repeat(60));
    console.log('→ SENDING TO ADD-IN');
    console.log('→ Payload:', data);

    try {
        const payload = JSON.stringify(data);

        // Use Promise-based API for Qt browser (Fusion 360's current browser)
        const result = window.adsk.fusionSendData(FUSION_ACTION, payload);

        if (result && typeof result.then === 'function') {
            result.catch((error) => {
                console.error('❌ Fusion bridge promise rejected:', error);
                addLog('error', `Fusion bridge error: ${error.message || error}`, { scope: 'global' });
            });
        }

        console.log('✓ Message sent to add-in');
        console.log('='.repeat(60));
        return true;
    } catch (error) {
        console.error('❌ Failed to send to add-in:', error);
        addLog('error', `Communication error: ${error.message}`, { scope: 'global' });
        return false;
    }
}

function sendIterationFeedback(iteration, verdict) {
    if (!isDocConnected()) {
        addLog('error', 'Cannot send feedback: not connected', { scope: 'global' });
        return;
    }
    const iter = Number.parseInt(iteration, 10);
    if (!Number.isFinite(iter) || iter <= 0) {
        addLog('warning', 'Invalid iteration number', { scope: 'global' });
        return;
    }
    const verdictNorm = (verdict || '').toLowerCase();
    sendToAddin({
        action: 'iteration_feedback',
        iteration: iter,
        verdict: verdictNorm
    });
    showToast(`Feedback saved: iteration ${iter} → ${verdictNorm || 'unknown'}`);
}

/**
 * Request current connection status from add-in
 */
function requestConnectionStatus() {
    if (!fusionBridgeReady) {
        return;
    }

    sendToAddin({ action: 'get_status' });
}

/**
 * Safely parse JSON payloads originating from Fusion
 */
function safeParseJson(value, fallback = {}) {
    if (value === null || value === undefined) {
        return fallback;
    }

    if (typeof value === 'string') {
        if (!value.trim()) {
            return fallback;
        }

        try {
            return JSON.parse(value);
        } catch (error) {
            console.warn('[CADAgent] Failed to parse JSON payload:', error);
            return fallback;
        }
    }

    if (typeof value === 'object') {
        return value;
    }

    return fallback;
}

/**
 * Handle top-level messages from the Fusion palette bridge
 */
function handleFusionMessage(action, payload) {
    switch (action) {
        case 'fusionReady':
            handleFusionReady(payload);
            break;

        case 'cadagent_message':
            if (typeof payload !== 'string') {
                payload = JSON.stringify(payload || {});
            }
            handleAddinMessage(payload);
            break;

        default:
            console.warn('← Unknown action from Fusion:', action, payload);
    }
}

/**
 * Handle Fusion handshake response
 */
function handleFusionReady(payload) {
    const data = safeParseJson(payload, {});
    const wasReady = fusionBridgeReady;
    fusionBridgeReady = true;
    _resetFusionLogThrottle();

    // Flush any queued messages now that handshake is complete
    flushAddinQueue();

    if (handshakeTimer) {
        clearTimeout(handshakeTimer);
        handshakeTimer = null;
    }

    if (!inFusion360) {
        applyEnvironmentState(true);
    }

    const sessionId = data.session_id || data.sessionId || state.sessionId;
    const backendConnected = typeof data.backend_connected === 'boolean'
        ? data.backend_connected
        : null;

    if (sessionId) {
        state.sessionId = sessionId;
    }

    state.testingMode = !!data.testing_mode;
    state.testingEmail = (data.testing_email || '').toLowerCase();

    if (elements.showPasswordLoginBtn) {
        elements.showPasswordLoginBtn.classList.toggle('hidden', !state.testingMode);
    }
    if (state.testingMode) {
        togglePasswordLogin(false);
    }

    // Now that the bridge is confirmed, ask the add-in for current profile
    sendToAddin({ action: 'get_profile' });

    if (!wasReady) {
        initializeFusionLogForwarding();
        addLog('info', 'Fusion palette bridge established', { scope: 'global' });
    }

    if (backendConnected !== null) {
        updateConnectionStatus(backendConnected, sessionId, currentDocId);
    } else {
        if (elements.statusText) {
            elements.statusText.textContent = state.connected ? 'Connected' : 'Disconnected';
        }

        if (elements.sessionId) {
            if (state.sessionId) {
                elements.sessionId.textContent = `${state.sessionId.substring(0, 8)}...`;
            } else {
                elements.sessionId.textContent = '-';
            }
        }

        if (elements.connectionStatus) {
            const statusDot = elements.connectionStatus.querySelector('.status-dot');
            if (statusDot) {
                statusDot.classList.toggle('connected', state.connected);
                statusDot.classList.toggle('disconnected', !state.connected);
            }
        }

        requestConnectionStatus();
    }

    if (elements.executeBtn) {
        updateExecuteButtonState();
    }
}

function queuePendingMessage(docId, message) {
    if (!docId) {
        return;
    }
    if (!pendingMessagesByDoc.has(docId)) {
        pendingMessagesByDoc.set(docId, []);
    }
    pendingMessagesByDoc.get(docId).push(message);
    log(`↳ Queued message of type '${message.type}' for doc ${docId} (currentDocId=${currentDocId})`);
}

function flushPendingMessages(docId) {
    if (!docId) {
        return;
    }
    const queue = pendingMessagesByDoc.get(docId);
    if (!queue || queue.length === 0) {
        return;
    }
    log(`↳ Flushing ${queue.length} queued message(s) for doc ${docId} (currentDocId=${currentDocId})`);
    pendingMessagesByDoc.delete(docId);
    queue.forEach((msg) => dispatchAddinMessage(msg, { bypassDefer: true }));
}

/**
 * Determine if a message should be deferred (queued for later).
 *
 * Key insight: Only `document_switched` should trigger a view change.
 * All other messages update the target doc's state in the background.
 *
 * CRITICAL: Auth messages (auth_success, auth_error, user_profile) are GLOBAL
 * and must NEVER be deferred - they have no document context.
 */
function shouldDeferMessage(message, targetDocId) {
    // AUTH MESSAGES: Never defer - they are global, not document-scoped
    const AUTH_MESSAGE_TYPES = ['auth_success', 'auth_error', 'user_profile'];
    if (AUTH_MESSAGE_TYPES.includes(message.type)) {
        console.log(`↳ Auth message '${message.type}' bypasses deferral (global scope)`);
        return false;
    }

    if (!targetDocId) {
        return false;  // No doc context, process in current view
    }

    // Bootstrap: If we have no current doc, adopt this one
    if (!currentDocId || currentDocId === 'default') {
        migrateDefaultStateTo(targetDocId);
        currentDocId = targetDocId;
        updateDocOrder(targetDocId);
        renderActiveDoc();
        flushPendingMessages(targetDocId);
        return false;
    }

    // SELF-HEAL: If current doc is disconnected and we get substantive activity for a different doc, adopt it
    // This fixes stale-doc queueing when the UI has cached an old/disconnected doc_id
    // Triggers on: connection_status (original), llm_message, log, progress, completed (expanded resilience)
    //
    // EXTENDED SELF-HEAL: Also adopt new doc if we get multiple substantive messages for it,
    // even when current doc is still marked connected. This handles reconnect scenarios where
    // document_switched is dropped but we get a stream of messages for the new doc_id.
    if (targetDocId !== currentDocId) {
        const currentDocState = getDocState(currentDocId, false);
        const isCurrentDocDisconnected = !currentDocState || !currentDocState.connected;

        // Trigger self-heal on connection_status OR any substantive message type
        const isConnectingDoc = message.type === 'connection_status' && message.connected === true;
        const isSubstantiveMessage = ['log', 'llm_message', 'progress', 'completed'].includes(message.type);

        // Track consecutive substantive messages for the same different doc
        // connection_status with connected=true also counts to handle post-auth race
        if (isSubstantiveMessage || isConnectingDoc) {
            if (consecutiveMessagesForDifferentDoc.docId === targetDocId) {
                consecutiveMessagesForDifferentDoc.count++;
            } else {
                consecutiveMessagesForDifferentDoc = { docId: targetDocId, count: 1 };
            }
        }

        // Original self-heal: current doc is disconnected
        const shouldHealDisconnected = isCurrentDocDisconnected && (isConnectingDoc || isSubstantiveMessage);

        // Extended self-heal: current doc is connected but we got 1+ substantive messages for a different doc
        // This prevents stale-doc lock after reconnect when document_switched is dropped.
        // Lowered threshold from 2 to 1 because:
        // - llm_message/log/completed are ESSENTIAL for run-log display
        // - If document_switched is dropped, the first llm_message must adopt the new doc
        // - Waiting for 2+ messages causes the first message to be silently queued forever
        const shouldHealConnectedStale = !isCurrentDocDisconnected &&
                                         consecutiveMessagesForDifferentDoc.docId === targetDocId &&
                                         consecutiveMessagesForDifferentDoc.count >= 1;

        if (shouldHealDisconnected || shouldHealConnectedStale) {
            const reason = shouldHealDisconnected ? 'disconnected' : 'stale (connected but inactive)';
            console.log(`↳ Self-heal: Current doc ${currentDocId} is ${reason}, adopting active doc ${targetDocId} (trigger: ${message.type}, consecutive_msgs: ${consecutiveMessagesForDifferentDoc.count})`);
            migrateDefaultStateTo(targetDocId);
            currentDocId = targetDocId;
            updateDocOrder(targetDocId);
            renderActiveDoc();
            flushPendingMessages(targetDocId);
            // Reset counter after successful heal
            consecutiveMessagesForDifferentDoc = { docId: null, count: 0 };
            return false;  // Process this message immediately
        }
    } else {
        // Message is for current doc - reset the consecutive counter
        consecutiveMessagesForDifferentDoc = { docId: null, count: 0 };
    }

    // CRITICAL FIX: Only `document_switched` triggers view change
    // This message comes from Fusion's documentActivated event
    if (message.type === 'document_switched') {
        // Prime doc metadata before switching so the toast uses the real tab name
        const docState = getDocState(targetDocId, true);
        if (docState) {
            docState.name = message.doc_name || docState.name;
            docState.sessionId = message.session_id || docState.sessionId;
            docState.connected = true;
            persistDocState(targetDocId);
        }
        // User switched tabs in Fusion - follow them
        switchToDocument(targetDocId);
        flushPendingMessages(targetDocId);
        return false;
    }

    // All other messages: update background state, don't switch view
    if (targetDocId !== currentDocId) {
        // Message is for a background document
        // Update its state but don't disrupt current view
        updateBackgroundDocState(targetDocId, message);
        log(`↳ Deferring message type='${message.type}' for targetDocId=${targetDocId} (currentDocId=${currentDocId})`);
        return true;  // DEFER - queue it for when that doc becomes active
    }

    return false;  // Same doc, process immediately
}

/**
 * Update a background document's state without switching view.
 */
function updateBackgroundDocState(docId, message) {
    const docState = getDocState(docId, true);
    docState.lastUpdated = Date.now();
    
    // Track that there's activity (could show subtle indicator later)
    if (!docState.pendingActivityCount) {
        docState.pendingActivityCount = 0;
    }
    docState.pendingActivityCount++;
    
    // Update connection status in background if that's what this is
    if (message.type === 'connection_status') {
        docState.connected = message.connected;
        if (message.session_id) {
            docState.sessionId = message.session_id;
        }
    }
    
    persistDocState(docId);
}

function dispatchAddinMessage(message, options = {}) {
    const { bypassDefer = false } = options;
    const targetDocId = typeof message.doc_id === 'string' ? message.doc_id : null;
    if (!bypassDefer && shouldDeferMessage(message, targetDocId)) {
        queuePendingMessage(targetDocId, message);
        return;
    }
    // If this message carries a doc_id and we have no current doc, set it now
    if ((!currentDocId || currentDocId === 'default') && targetDocId) {
        migrateDefaultStateTo(targetDocId);
        currentDocId = targetDocId;
        updateDocOrder(targetDocId);
        renderActiveDoc();
        updateDocSwitcherUI();
        flushPendingMessages(targetDocId);
    }
    processAddinMessage(message);
}

function handleSelectionFeedback(feedback) {
    if (!feedback || typeof feedback !== 'object') {
        return;
    }

    const geometry = ['edge', 'face', 'body'].includes(feedback.geometry) ? feedback.geometry : 'edge';
    const operation = typeof feedback.operation === 'string' ? feedback.operation : '';
    const success = feedback.success !== false;
    const message = typeof feedback.message === 'string' ? feedback.message.trim() : '';
    const counts = typeof feedback.counts === 'object' && feedback.counts !== null ? feedback.counts : {};
    const missingTokens = Array.isArray(feedback.missing_tokens) ? feedback.missing_tokens : [];
    const clearedExisting = typeof feedback.cleared_existing === 'boolean'
        ? feedback.cleared_existing
        : undefined;

    const title = getSelectionTitle(operation, geometry);
    const senderLabel = geometry === 'face'
        ? 'Face tools'
        : geometry === 'body'
            ? 'Body tools'
            : 'Edge tools';

    if (!success) {
        const body = message || 'Operation failed.';
        appendMessage('assistant', `**${title}**\n${body}`, {
            sender: senderLabel,
            variant: 'error',
            contentFormat: 'markdown'
        });
        return;
    }

    const summary = buildSelectionSummary(operation, geometry, counts, missingTokens, message);
    const variant = chooseSelectionVariant(operation, missingTokens, counts);

    const content = summary || title;

    const simpleSelectionOps = [
        'list_edges',
        'list_faces',
        'list_bodies',
        'select_edges',
        'select_faces',
        'select_bodies'
    ];

    // Skip rich bubble rendering for simple list/select feedback; run log entries already cover these.
    if (simpleSelectionOps.includes(operation)) {
        return;
    }

    appendMessage('assistant', content, {
        sender: senderLabel,
        variant,
    });
}

function handleReasoningChunk(content) {
    if (typeof content === 'undefined' || content === null) {
        return;
    }
    appendReasoningChunk(content);
}

function getSelectionTitle(operation, geometry) {
    const normalized = operation || '';
    if (normalized === 'list_edges') return 'Edge listing';
    if (normalized === 'select_edges') return 'Edge selection';
    if (normalized === 'clear_edge_selection') return 'Edge selection cleared';
    if (normalized === 'list_bodies') return 'Body listing';
    if (normalized === 'select_bodies') return 'Body selection';
    if (normalized === 'clear_body_selection') return 'Body selection cleared';
    if (normalized === 'list_faces') return 'Face listing';
    if (normalized === 'select_faces') return 'Face selection';
    if (normalized === 'clear_face_selection') return 'Face selection cleared';
    if (geometry === 'face') return 'Face operation';
    if (geometry === 'body') return 'Body operation';
    return 'Edge operation';
}

function buildSelectionSummary(operation, geometry, counts, missingTokens, fallbackMessage) {
    const total = getNumber(counts.total);
    const selected = getNumber(counts.selected);
    const missing = getNumber(counts.missing) ?? missingTokens.length;
    const cleared = getNumber(counts.cleared);
    const geometryLabel = geometry === 'face' ? 'face' : (geometry === 'body' ? 'body' : 'edge');
    const plural = (value) => (value === 1 ? geometryLabel : `${geometryLabel}s`);

    switch (operation) {
        case 'list_edges':
        case 'list_faces':
        case 'list_bodies':
            if (total !== null) {
                return `Found ${total} ${plural(total)}.`;
            }
            return `Found ${plural(2)}.`;

        case 'select_edges':
        case 'select_faces':
        case 'select_bodies': {
            if (selected !== null) {
                let text = `Selected ${selected} ${plural(selected)}.`;
                if (missing) {
                    text += ` ${missing} missing token${missing === 1 ? '' : 's'}.`;
                }
                return text;
            }
            return fallbackMessage || `Selection updated.`;
        }

        case 'clear_edge_selection':
        case 'clear_face_selection':
        case 'clear_body_selection':
            if (cleared !== null) {
                return `Cleared ${geometryLabel} selection (${cleared} removed).`;
            }
            return `Cleared ${geometryLabel} selection.`;

        default:
            return fallbackMessage || 'Operation completed.';
    }
}

function chooseSelectionVariant(operation, missingTokens, counts) {
    if (operation === 'select_edges' || operation === 'select_faces' || operation === 'select_bodies') {
        const missingCount = getNumber(counts.missing) ?? missingTokens.length;
        if (missingCount > 0) {
            return 'warning';
        }
        return 'success';
    }
    if (operation === 'clear_edge_selection' || operation === 'clear_face_selection' || operation === 'clear_body_selection') {
        const cleared = getNumber(counts.cleared);
        return cleared && cleared > 0 ? 'success' : 'info';
    }
    return 'info';
}

function getNumber(value) {
    if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
    }
    if (typeof value === 'string') {
        const parsed = Number(value);
        if (Number.isFinite(parsed)) {
            return parsed;
        }
    }
    return null;
}

/**
 * Find the most recent user message without an associated checkpoint.
 * @returns {HTMLElement|null}
 */
function findLastUnassociatedUserMessage() {
    if (!elements.chatMessages) {
        return null;
    }
    const candidates = elements.chatMessages.querySelectorAll('.message.user-message');
    for (let i = candidates.length - 1; i >= 0; i -= 1) {
        const candidate = candidates[i];
        if (!candidate.getAttribute('data-message-id')) {
            return candidate;
        }
    }
    return null;
}

/**
 * Ensure a message has the revert UI wired up and tagged with the checkpoint id.
 * @param {HTMLElement} messageEl
 * @param {string} messageId
 * @param {{ skipPersist?: boolean }} options
 * @returns {boolean}
 */
function associateCheckpointWithMessage(messageEl, messageId, options = {}) {
    if (!messageEl || !elements.chatMessages || !elements.chatMessages.contains(messageEl)) {
        return false;
    }

    const { skipPersist = false } = options;
    const existingId = messageEl.getAttribute('data-message-id');
    if (!existingId && messageId) {
        messageEl.setAttribute('data-message-id', messageId);
    } else if (existingId && messageId && existingId !== messageId) {
        console.warn('Message already associated with different checkpoint:', existingId, messageId);
        return false;
    }

    const header = messageEl.querySelector('.message-header');
    if (header) {
        let actionsEl = header.querySelector('.message-actions');
        if (!actionsEl) {
            actionsEl = document.createElement('div');
            actionsEl.classList.add('message-actions');
            header.appendChild(actionsEl);
        }

        if (!actionsEl.querySelector('.revert-btn')) {
            const revertBtn = document.createElement('button');
            revertBtn.classList.add('revert-btn');
            revertBtn.setAttribute('title', 'Revert timeline to this state');
            revertBtn.setAttribute('aria-label', 'Revert timeline to this state');
            revertBtn.innerHTML = '<span class="revert-icon">⟲</span> Revert';
            actionsEl.appendChild(revertBtn);
        }
    }

    if (state.pendingUserMessage === messageEl || (state.pendingUserMessage && !state.pendingUserMessage.isConnected)) {
        state.pendingUserMessage = null;
    }

    if (!skipPersist) {
        persistCurrentThreadState();
    }

    return true;
}

/**
 * Attempt to apply any queued checkpoints to user messages.
 * @param {{ skipPersist?: boolean }} options
 */
function processPendingCheckpoints(options = {}) {
    const { skipPersist = false } = options;
    if (!state.pendingCheckpointQueue.length) {
        return;
    }

    const remaining = [];
    let applied = false;

    for (const checkpointId of state.pendingCheckpointQueue) {
        const target = findLastUnassociatedUserMessage();
        const associated = target
            ? associateCheckpointWithMessage(target, checkpointId, { skipPersist: true })
            : false;
        if (associated) {
            applied = true;
        } else {
            remaining.push(checkpointId);
        }
    }

    state.pendingCheckpointQueue = remaining;

    if (applied && !skipPersist) {
        persistCurrentThreadState();
    }
}

/**
 * Handle checkpoint creation by associating message_id with the pending user message
 * @param {string} messageId - The checkpoint message ID
 * @param {string} requestId - Optional request ID for exact correlation
 */
function handleCheckpointCreated(messageId, requestId = null) {
    if (!messageId) {
        console.warn('handleCheckpointCreated called without message_id');
        return;
    }

    let target = null;

    // NEW APPROACH: Use request_id for exact correlation if provided
    if (requestId && elements.chatMessages) {
        const messageWithRequestId = elements.chatMessages.querySelector(`[data-request-id="${requestId}"]`);
        if (messageWithRequestId) {
            target = messageWithRequestId;
            console.log('Checkpoint matched by request_id:', requestId);
        }
    }

    // FALLBACK: Use old order-based matching for backward compatibility
    if (!target) {
        target = state.pendingUserMessage;
        if (target && !target.isConnected) {
            target = null;
        }

        if (!target) {
            target = findLastUnassociatedUserMessage();
        }

        if (target && requestId) {
            console.warn('Checkpoint matched by order (request_id provided but no matching message found):', requestId);
        }
    }

    if (target) {
        const associated = associateCheckpointWithMessage(target, messageId);
        if (associated) {
            console.log('Checkpoint associated with user message:', messageId);
            processPendingCheckpoints();
        } else {
            console.warn('Failed to associate checkpoint with message, queueing for retry:', messageId);
            if (!state.pendingCheckpointQueue.includes(messageId)) {
                state.pendingCheckpointQueue.push(messageId);
            }
        }
        return;
    }

    console.warn('No pending user message to associate with checkpoint, queueing for retry:', messageId);
    state.pendingUserMessage = null;
    if (!state.pendingCheckpointQueue.includes(messageId)) {
        state.pendingCheckpointQueue.push(messageId);
    }
}

/**
 * Handle revert to checkpoint button click
 */
function handleRevertToCheckpoint(messageId) {
    if (!messageId) {
        console.error('handleRevertToCheckpoint called without message_id');
        return;
    }

    // Block revert during active execution to prevent race conditions
    if (state.processing) {
        addLog('warning', 'Cannot revert while execution is in progress. Please wait for current operation to complete.', { scope: 'global' });
        return;
    }

    if (!isDocConnected()) {
        addLog('error', 'Cannot revert: not connected to backend', { scope: 'global' });
        return;
    }

    console.log('Reverting to checkpoint:', messageId);

    // Send revert request to add-in
    sendToAddin({
        action: 'revert_request',
        message_id: messageId
    });

    addLog('info', 'Reverting timeline...', { scope: 'global' });
}

/**
 * Remove chat entries that occurred after a checkpoint message.
 * @param {{ messageId?: string, includeMessage?: boolean, conversationLength?: number | string }} options
 */
function applyConversationRevert(options = {}) {
    if (!elements.chatMessages) {
        return;
    }

    const {
        messageId = null,
        includeMessage = true,
        conversationLength = null
    } = options;

    const container = elements.chatMessages;

    const removedMessageIds = [];
    const recordRemoval = (element) => {
        if (!element?.classList?.contains('message')) {
            return;
        }
        const removedId = element.getAttribute('data-message-id');
        if (removedId) {
            removedMessageIds.push(removedId);
        }
    };

    const removeElement = (element) => {
        if (!element) {
            return;
        }
        recordRemoval(element);
        element.remove();
    };

    const parseLength = (value) => {
        if (typeof value === 'number' && Number.isFinite(value)) {
            return Math.max(0, Math.floor(value));
        }
        if (typeof value === 'string') {
            const parsed = Number.parseInt(value, 10);
            if (Number.isFinite(parsed)) {
                return Math.max(0, parsed);
            }
        }
        return null;
    };

    const targetLength = parseLength(conversationLength);
    let anchorMessage = null;

    if (targetLength !== null) {
        const messageNodes = Array.from(container.querySelectorAll('.message'));

        if (targetLength === 0) {
            messageNodes.forEach(removeElement);
            Array.from(container.children).forEach(removeElement);
        } else {
            const keepCount = Math.min(targetLength, messageNodes.length);
            for (let i = messageNodes.length - 1; i >= keepCount; i -= 1) {
                removeElement(messageNodes[i]);
            }
            anchorMessage = messageNodes[Math.max(0, keepCount - 1)] || null;

            if (anchorMessage) {
                let node = anchorMessage.nextElementSibling;
                while (node) {
                    const next = node.nextElementSibling;
                    removeElement(node);
                    node = next;
                }
            }
        }
    }

    if (anchorMessage === null && targetLength !== null && targetLength > 0) {
        const remainingMessages = Array.from(container.querySelectorAll('.message'));
        anchorMessage = remainingMessages[Math.min(targetLength, remainingMessages.length) - 1] || null;

        if (anchorMessage) {
            let node = anchorMessage.nextElementSibling;
            while (node) {
                const next = node.nextElementSibling;
                removeElement(node);
                node = next;
            }
        } else if (remainingMessages.length === 0) {
            Array.from(container.children).forEach(removeElement);
        }
    } else if (anchorMessage === null && targetLength === null) {
        anchorMessage = messageId
            ? container.querySelector(`.message[data-message-id="${messageId}"]`)
            : null;

        if (!anchorMessage && messageId) {
            console.warn('Revert applied but anchor message not found:', messageId);
        }

        if (!anchorMessage) {
            anchorMessage = Array.from(container.querySelectorAll('.message')).pop() || null;
            if (!anchorMessage) {
                // Nothing to trim; ensure non-message elements beyond this point are cleared.
                Array.from(container.children).forEach(removeElement);
            }
        }

        if (anchorMessage) {
            const shouldKeepAnchor = includeMessage !== false;
            let node = shouldKeepAnchor ? anchorMessage.nextElementSibling : anchorMessage;

            while (node) {
                const next = node.nextElementSibling;
                removeElement(node);
                node = next;
            }

            if (!shouldKeepAnchor && anchorMessage.isConnected) {
                removeElement(anchorMessage);
            }
        }
    }

    if (removedMessageIds.length && state.pendingCheckpointQueue.length) {
        state.pendingCheckpointQueue = state.pendingCheckpointQueue.filter(
            (id) => !removedMessageIds.includes(id)
        );
    }

    if (state.pendingUserMessage && !container.contains(state.pendingUserMessage)) {
        state.pendingUserMessage = null;
    }

    resetRunLogState();

    if (container.children.length === 0) {
        if (elements.welcomeMessage) {
            elements.welcomeMessage.classList.remove('hidden');
        }
        if (elements.headerSection) {
            elements.headerSection.classList.remove('compact-mode');
        }
    }

    rebuildDocStateFromDom(currentDocId);
    persistCurrentThreadState();
}

/**
 * Handle confirmation from backend that a revert was applied.
 * @param {{ message_id?: string, include_message?: boolean }} message
 */
function handleRevertApplied(message = {}) {
    const messageId = message.message_id || null;
    const includeMessage = message.include_message !== false;
    const conversationLength = message.conversation_length;
    applyConversationRevert({ messageId, includeMessage, conversationLength });
}

// ============================================
// Design Exploration UI Logic
// ============================================

/**
 * Handle question tree generated message from backend
 */
function handleQuestionTreeGenerated(data) {
    // Hide progress if it was showing
    hideProgress();
    
    // Render the question tree
    renderQuestionTree(data);
    
    // Log intent
    addLog('agent', 'I need some clarification to design this.', { scope: 'run' });
}

/**
 * Render the question tree UI
 */
function renderQuestionTree(data) {
    if (!elements.chatMessages) return;

    // Create container
    const container = document.createElement('div');
    container.className = 'question-tree-container';
    
    // Header with problem summary
    const header = document.createElement('div');
    header.className = 'question-tree-header';
    
    const summary = document.createElement('span');
    summary.className = 'problem-summary';
    summary.textContent = data.problem_summary || 'Requirements Clarification';
    header.appendChild(summary);
    
    if (data.known_constraints && data.known_constraints.length > 0) {
        // Show all constraints as badges
        data.known_constraints.forEach(constraint => {
            const badge = document.createElement('span');
            badge.className = 'constraints-badge';
            badge.textContent = constraint;
            header.appendChild(badge);
        });
    }
    
    container.appendChild(header);
    
    // Store data on container for reference
    container.dataset.treeData = JSON.stringify(data);
    container.dataset.answers = JSON.stringify({}); // Initialize empty answers
    
    // Append to chat
    elements.chatMessages.appendChild(container); // Use global elements object
    
    // Initialize question queue with all root questions, tagging each with its rootId
    if (data.questions && data.questions.length > 0) {
        container.rootQuestions = data.questions.map(q => ({ ...q, rootId: q.id }));
        container.questionQueue = [...container.rootQuestions];
        renderNextQuestion(container);
    }

    // Scroll to view
    container.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Render the next question from the queue
 */
function renderNextQuestion(container) {
    if (!container.questionQueue || container.questionQueue.length === 0) {
        // Queue exhausted - fade out and remove the question tree container
        const answers = JSON.parse(container.dataset.answers || '{}');
        container.classList.add('fade-out');
        setTimeout(() => {
            container.remove();
        }, 200);
        submitQuestionTreeAnswers(answers);
        return;
    }

    const nextQuestion = container.questionQueue.shift();
    renderQuestionCard(container, nextQuestion);
}

/**
 * Render a single question card
 */
function renderQuestionCard(container, question) {
    if (!question) return;

    const card = document.createElement('div');
    card.className = 'question-card active';
    card.dataset.questionId = question.id;
    card.dataset.rootId = question.rootId || question.id;

    // Header
    const header = document.createElement('div');
    header.className = 'question-header';

    const text = document.createElement('span');
    text.className = 'question-text';
    text.textContent = question.question;
    header.appendChild(text);

    if (question.hint) {
        const hintBtn = document.createElement('button');
        hintBtn.className = 'hint-icon';
        hintBtn.textContent = 'ⓘ';
        hintBtn.title = question.hint;
        header.appendChild(hintBtn);
    }

    card.appendChild(header);

    // Options
    const optionsList = document.createElement('div');
    optionsList.className = 'options-list';

    question.options.forEach(option => {
        const optionWrapper = document.createElement('div');
        optionWrapper.className = option.allows_text_input ? 'option-other' : 'option-standard';

        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.textContent = option.label;
        btn.dataset.value = option.value;

        btn.addEventListener('click', () => {
            handleQuestionAnswer(container, card, question, option);
        });

        optionWrapper.appendChild(btn);

        if (option.allows_text_input) {
            const inputRow = document.createElement('div');
            inputRow.className = 'other-input-row hidden';
            inputRow.dataset.forOption = option.value;

            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'other-input';
            input.placeholder = 'Please describe...';

            const continueBtn = document.createElement('button');
            continueBtn.className = 'other-continue-btn';
            continueBtn.textContent = 'Continue';
            continueBtn.disabled = true;

            // Enable Continue button only when input is non-empty
            input.addEventListener('input', () => {
                continueBtn.disabled = input.value.trim() === '';
            });

            // Handle Enter key on input (only if non-empty)
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && input.value.trim() !== '') {
                    handleQuestionAnswer(container, card, question, option, input.value.trim());
                }
            });

            // Handle Continue button click
            continueBtn.addEventListener('click', () => {
                if (input.value.trim() !== '') {
                    handleQuestionAnswer(container, card, question, option, input.value.trim());
                }
            });

            inputRow.appendChild(input);
            inputRow.appendChild(continueBtn);
            optionWrapper.appendChild(inputRow);
        }

        optionsList.appendChild(optionWrapper);
    });

    card.appendChild(optionsList);
    container.appendChild(card);

    // Scroll to new card
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Handle user answering a question
 */
function handleQuestionAnswer(container, card, question, selectedOption, textInputValue = null) {
    // 1. Update UI state for this card
    const buttons = card.querySelectorAll('.option-btn');
    const inputRows = card.querySelectorAll('.other-input-row');

    buttons.forEach(btn => {
        if (btn.dataset.value === selectedOption.value) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });

    // Hide all input rows first
    inputRows.forEach(row => row.classList.add('hidden'));

    // 2. Handle "Other" text input
    if (selectedOption.allows_text_input) {
        const inputRow = card.querySelector(`.other-input-row[data-for-option="${selectedOption.value}"]`);
        if (inputRow) {
            inputRow.classList.remove('hidden');
            const input = inputRow.querySelector('.other-input');
            if (input) input.focus();

            // If we don't have text value yet, stop here and wait for user to type and click Continue
            if (!textInputValue) {
                return;
            }
        }
    }

    // 3. Record Answer
    const currentAnswers = JSON.parse(container.dataset.answers || '{}');
    currentAnswers[question.id] = {
        value: selectedOption.value,
        text_input: textInputValue || undefined
    };
    container.dataset.answers = JSON.stringify(currentAnswers);

    // 4. Fade out and remove the answered card (questions disappear after answering)
    card.classList.add('fade-out');
    setTimeout(() => {
        card.remove();
    }, 200);

    // 5. Remove any subsequent question cards if re-answering (prune tree)
    let nextSibling = card.nextElementSibling;
    const prunedIds = [];
    while (nextSibling && nextSibling.classList.contains('question-card')) {
        const toRemove = nextSibling;
        prunedIds.push(nextSibling.dataset.questionId);
        nextSibling = nextSibling.nextElementSibling;
        toRemove.remove();
    }
    // Drop answers for pruned questions
    prunedIds.forEach(id => {
        delete currentAnswers[id];
    });
    container.dataset.answers = JSON.stringify(currentAnswers);

    // 6. Rebuild queue: follow-ups next, then remaining root questions after current root
    const currentRootId = card.dataset.rootId || question.id;
    const rootQuestions = container.rootQuestions || [];
    const currentRootIndex = rootQuestions.findIndex(q => q.id === currentRootId);

    const followUps = (selectedOption.follow_up_questions || []).map(q => ({
        ...q,
        rootId: currentRootId
    }));

    const remainingRoots = rootQuestions
        .filter((_, idx) => idx > currentRootIndex)
        .map(q => ({ ...q })); // already carries rootId

    container.questionQueue = [...followUps, ...remainingRoots];

    // 7. Render next question from queue (or submit if done) - delay to sync with fade out
    setTimeout(() => {
        renderNextQuestion(container);
    }, 200);
}

/**
 * Submit completed answers to backend
 */
function submitQuestionTreeAnswers(answers) {
    // Add confirmation to run log
    addLog('success', 'Clarifications complete. Generating designs...', { scope: 'run' });
    
    showProgress('Generating designs...');
    sendToAddin({
        action: 'send_to_backend',
        message: {
            type: 'question_tree_completed',
            answers: answers
        }
    });
}

/**
 * Handle designs proposed message from backend
 */
function handleDesignsProposed(data) {
    hideProgress();
    renderDesignProposals(data);
    addLog('agent', 'I have come up with a few options.', { scope: 'run' });
}

/**
 * Render design proposals UI
 */
function renderDesignProposals(data) {
    if (!elements.chatMessages) return;

    // Helper to coerce model output into an array so UI rendering can't throw
    const toArray = (val) => Array.isArray(val) ? val : (val == null ? [] : [val]);
    const designs = Array.isArray(data.designs) ? data.designs : [];

    const container = document.createElement('div');
    container.className = 'design-proposals-container';
    
    // Header
    const header = document.createElement('div');
    header.className = 'proposals-header';
    const summary = document.createElement('span');
    summary.className = 'context-summary';
    summary.textContent = data.context_summary || 'Based on your requirements:';
    header.appendChild(summary);
    container.appendChild(header);
    
    // Render Cards
    designs.forEach(design => {
        const card = document.createElement('div');
        card.className = 'design-card';
        if (data.recommendation === design.id) {
            card.classList.add('recommended');
            
            const badge = document.createElement('div');
            badge.className = 'recommended-badge';
            badge.textContent = '★ RECOMMENDED';
            card.appendChild(badge);
        }
        
        // ===== PREVIEW SECTION (always visible) =====
        const previewSection = document.createElement('div');
        previewSection.className = 'design-card-preview';
        
        // Name
        const name = document.createElement('h3');
        name.className = 'design-name';
        name.textContent = design.name;
        previewSection.appendChild(name);
        
        // Description
        const desc = document.createElement('p');
        desc.className = 'design-description';
        desc.textContent = design.description;
        previewSection.appendChild(desc);
        
        // Best For (moved to preview)
        if (design.best_for) {
            const bestFor = document.createElement('div');
            bestFor.className = 'best-for';
            bestFor.textContent = `Best for: ${design.best_for}`;
            previewSection.appendChild(bestFor);
        }
        
        card.appendChild(previewSection);
        
        // ===== DETAILS SECTION (hidden by default) =====
        const detailsSection = document.createElement('div');
        detailsSection.className = 'design-card-details';
        
        // Key Features
        if (design.key_features && design.key_features.length > 0) {
            const features = document.createElement('ul');
            features.className = 'key-features';
            design.key_features.forEach(feat => {
                const li = document.createElement('li');
                li.textContent = feat;
                features.appendChild(li);
            });
            detailsSection.appendChild(features);
        }
        
        // Tradeoffs
        if (design.tradeoffs) {
            const tradeoffs = document.createElement('div');
            tradeoffs.className = 'tradeoffs';

            // Pros
            const prosList = toArray(design.tradeoffs.pros);
            if (prosList.length > 0) {
                const prosDiv = document.createElement('div');
                prosDiv.className = 'pros';
                const label = document.createElement('span');
                label.className = 'label';
                label.textContent = '✓';
                prosDiv.appendChild(label);

                const ul = document.createElement('ul');
                prosList.forEach(p => {
                    const li = document.createElement('li');
                    li.textContent = p;
                    ul.appendChild(li);
                });
                prosDiv.appendChild(ul);
                tradeoffs.appendChild(prosDiv);
            }

            // Cons
            const consList = toArray(design.tradeoffs.cons);
            if (consList.length > 0) {
                const consDiv = document.createElement('div');
                consDiv.className = 'cons';
                const label = document.createElement('span');
                label.className = 'label';
                label.textContent = '✗';
                consDiv.appendChild(label);

                const ul = document.createElement('ul');
                consList.forEach(c => {
                    const li = document.createElement('li');
                    li.textContent = c;
                    ul.appendChild(li);
                });
                consDiv.appendChild(ul);
                tradeoffs.appendChild(consDiv);
            }
            
            detailsSection.appendChild(tradeoffs);
        }
        
        card.appendChild(detailsSection);
        
        // ===== ACTION BUTTONS =====
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'design-card-actions';
        
        // Expand/Collapse toggle (only if there are details to show)
        const hasDetails = (design.key_features && design.key_features.length > 0) || 
                          (design.tradeoffs && (design.tradeoffs.pros?.length > 0 || design.tradeoffs.cons?.length > 0));
        
        if (hasDetails) {
            const expandToggle = document.createElement('button');
            expandToggle.className = 'design-expand-toggle';
            expandToggle.setAttribute('aria-label', 'Toggle details');
            expandToggle.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>';
            expandToggle.addEventListener('click', () => {
                card.classList.toggle('expanded');
            });
            // Insert at the beginning of actions div
            actionsDiv.insertBefore(expandToggle, actionsDiv.firstChild);
        }
        
        // Build Button
        const btn = document.createElement('button');
        btn.className = 'build-btn';
        btn.textContent = 'Build This';
        btn.dataset.designId = design.id;
        btn.addEventListener('click', () => {
            handleDesignSelection(design.id, container, design.name);
        });
        actionsDiv.appendChild(btn);
        
        card.appendChild(actionsDiv);
        container.appendChild(card);
    });
    
    // Append to chat
    elements.chatMessages.appendChild(container);
    container.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Handle design selection
 */
function handleDesignSelection(designId, container, designName) {
    // 1. Log selection to timeline
    addLog('user', `Building: ${designName || designId}`, { scope: 'run' });

    // 2. Hide/collapse the proposals container
    container.classList.add('collapsed');
    container.style.display = 'none';

    // 3. Show progress
    showProgress('Building design...');

    // 4. Send selection to backend
    sendToAddin({
        action: 'send_to_backend',
        message: {
            type: 'design_selected',
            design_id: designId
        }
    });
}

// =============================================================================
// Build Plan UI
// =============================================================================

// Track active build plan container for updates
let activeBuildPlanContainer = null;

/**
 * Handle build_plan_generated message from backend
 */
function handleBuildPlanGenerated(data) {
    console.log('← Handling build_plan_generated', data);
    renderBuildPlan(data);
    addLog('agent', `Build plan: ${data.design_name} (${data.total_steps} steps)`, { scope: 'run' });
}

/**
 * Render build plan UI
 */
function renderBuildPlan(data) {
    // Show the bar
    if (elements.buildPlanBar) {
        elements.buildPlanBar.classList.remove('hidden');
        elements.buildPlanBar.classList.remove('complete');
    }
    if (elements.buildPlanBarText) {
        elements.buildPlanBarText.textContent = data.design_name || 'Build Plan';
    }
    if (elements.buildPlanBarProgress) {
        elements.buildPlanBarProgress.textContent = `0/${data.total_steps}`;
    }
    if (elements.buildPlanBarTitle) {
        elements.buildPlanBarTitle.textContent = `Build Plan: ${data.design_name}`;
    }
    
    // Populate steps in bar
    if (elements.buildPlanBarSteps) {
        elements.buildPlanBarSteps.innerHTML = '';
        data.steps.forEach((step, index) => {
            const stepEl = document.createElement('div');
            stepEl.className = 'build-plan-bar-step';
            stepEl.id = `bar-step-${index + 1}`;
            
            const numEl = document.createElement('span');
            numEl.className = 'step-num';
            numEl.textContent = `${step.step_number || index + 1}.`;
            
            const descEl = document.createElement('span');
            descEl.className = 'step-desc';
            descEl.textContent = step.description;
            
            const iconEl = document.createElement('span');
            iconEl.className = 'step-icon';
            iconEl.id = `bar-step-icon-${index + 1}`;
            
            stepEl.appendChild(numEl);
            stepEl.appendChild(descEl);
            stepEl.appendChild(iconEl);
            elements.buildPlanBarSteps.appendChild(stepEl);
        });
    }

    // Keep inline container but hide it
    if (!elements.chatMessages) return;

    const container = document.createElement('div');
    container.className = 'build-plan-container';
    container.style.display = 'none'; // Hide the inline container
    container.id = 'active-build-plan';
    activeBuildPlanContainer = container;

    // Header
    const header = document.createElement('div');
    header.className = 'build-plan-header';

    const icon = document.createElement('span');
    icon.className = 'build-plan-icon';
    icon.textContent = '\u{1F6E0}'; // Tool icon
    header.appendChild(icon);

    const title = document.createElement('span');
    title.className = 'build-plan-title';
    title.textContent = `Build Plan: ${data.design_name}`;
    header.appendChild(title);

    const progress = document.createElement('span');
    progress.className = 'build-plan-progress';
    progress.id = 'build-plan-progress';
    progress.textContent = `0/${data.total_steps}`;
    header.appendChild(progress);

    container.appendChild(header);

    // Steps list
    const stepsList = document.createElement('div');
    stepsList.className = 'build-plan-steps';
    stepsList.id = 'build-plan-steps';

    data.steps.forEach((step, index) => {
        const stepEl = document.createElement('div');
        stepEl.className = 'build-step';
        stepEl.id = `build-step-${index + 1}`;

        const stepNumber = document.createElement('span');
        stepNumber.className = 'step-number';
        stepNumber.textContent = `${step.step_number || index + 1}.`;
        stepEl.appendChild(stepNumber);

        const stepDesc = document.createElement('span');
        stepDesc.className = 'step-description';
        stepDesc.textContent = step.description;
        stepEl.appendChild(stepDesc);

        const stepStatus = document.createElement('span');
        stepStatus.className = 'step-status';
        stepStatus.id = `step-status-${index + 1}`;
        stepEl.appendChild(stepStatus);

        stepsList.appendChild(stepEl);
    });

    container.appendChild(stepsList);

    // Append to chat
    elements.chatMessages.appendChild(container);
    container.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Handle build_step_completed message - update UI to show progress
 */
function handleBuildStepCompleted(data) {
    console.log('← Handling build_step_completed', data);

    const { completed_steps, total_steps } = data;

    // Update bar progress
    if (elements.buildPlanBarProgress) {
        elements.buildPlanBarProgress.textContent = `${completed_steps}/${total_steps}`;
    }
    
    // Mark step as completed in bar
    const barStepEl = document.getElementById(`bar-step-${completed_steps}`);
    if (barStepEl) {
        barStepEl.classList.add('step-completed');
        const iconEl = document.getElementById(`bar-step-icon-${completed_steps}`);
        if (iconEl) {
            iconEl.textContent = '✓';
        }
    }
    
    // Mark next step as current in bar
    if (completed_steps < total_steps) {
        const nextBarStepEl = document.getElementById(`bar-step-${completed_steps + 1}`);
        if (nextBarStepEl) {
            nextBarStepEl.classList.add('step-current');
        }
    }

    // Update inline progress counter
    const progressEl = document.getElementById('build-plan-progress');
    if (progressEl) {
        progressEl.textContent = `${completed_steps}/${total_steps}`;
    }

    // Mark the completed step
    const stepEl = document.getElementById(`build-step-${completed_steps}`);
    if (stepEl) {
        stepEl.classList.add('step-completed');
        const statusEl = document.getElementById(`step-status-${completed_steps}`);
        if (statusEl) {
            statusEl.textContent = '\u2713'; // Checkmark
        }
    }

    // Mark next step as current (if not complete)
    if (completed_steps < total_steps) {
        const nextStepEl = document.getElementById(`build-step-${completed_steps + 1}`);
        if (nextStepEl) {
            nextStepEl.classList.add('step-current');
        }
    }
}

/**
 * Handle build_plan_completed message - finalize the UI
 */
function handleBuildPlanCompleted(data) {
    console.log('← Handling build_plan_completed', data);

    // Update bar to complete state
    if (elements.buildPlanBar) {
        elements.buildPlanBar.classList.add('complete');
    }
    if (elements.buildPlanBarText) {
        elements.buildPlanBarText.textContent = `${data.design_name} ✓`;
    }

    // Optionally hide bar after a delay
    setTimeout(() => {
        if (elements.buildPlanBar) {
            elements.buildPlanBar.classList.add('hidden');
        }
    }, 5000);

    const container = document.getElementById('active-build-plan');
    if (container) {
        container.classList.add('build-plan-complete');

        // Update header to show completion
        const title = container.querySelector('.build-plan-title');
        if (title) {
            title.textContent = `${data.design_name} - Complete`;
        }

        const icon = container.querySelector('.build-plan-icon');
        if (icon) {
            icon.textContent = '\u2705'; // Green checkmark
        }
    }

    addLog('agent', `Build complete: ${data.design_name}`, { scope: 'run' });
    activeBuildPlanContainer = null;
}

function processAddinMessage(message) {
    // Log all incoming messages for debugging
    log(`↳ processAddinMessage: type='${message.type}', doc_id='${message.doc_id || 'none'}', keys=[${Object.keys(message).join(', ')}]`);
    // TEMP auth/debug probe: keep last message reachable from console
    if (typeof window !== 'undefined') {
        window.__lastAddinMessage = message;
    }

    switch (message.type) {
        case 'document_switched': {
            console.log('← Handling document_switched');
            try {
                const { doc_id, doc_name, session_id } = message;

                // Update doc state metadata
                const docState = getDocState(doc_id, true);
                if (docState) {
                    docState.name = doc_name || docState.name;
                    docState.sessionId = session_id || docState.sessionId;
                    docState.connected = true;  // If we got this message, we're connected
                    docState.pendingActivityCount = 0;  // Clear pending count on switch
                }

                // Update session ID display (already in current view due to shouldDeferMessage)
                if (elements.sessionId && session_id) {
                    elements.sessionId.textContent = `${session_id.substring(0, 8)}...`;
                }

                persistDocState(doc_id);

            } catch (e) {
                console.error('Error handling document_switched:', e);
            }
            break;
        }
        case 'checkpoint_created':
            console.log('← Handling checkpoint_created');
            handleCheckpointCreated(message.message_id, message.request_id);
            break;
        case 'connection_status':
            console.log('← Handling connection_status');
            updateConnectionStatus(message.connected, message.session_id, message.doc_id);
            break;
        case 'log': {
            log('← Handling log');
            log('← message.format:', message.format);
            log('← message.message:', message.message);
            const level = message.level || 'info';
            const scope = message.scope || 'auto';
            const options = { scope };
            if (typeof message.dismiss_hero === 'boolean') {
                options.dismissHero = message.dismiss_hero;
            }
            if (typeof message.format === 'string' && message.format.trim()) {
                options.messageFormat = message.format.trim();
                log('← SET options.messageFormat to:', options.messageFormat);
            }
            // Mirror backend logs to Fusion Text Commands for troubleshooting
            log(`[BACKEND LOG:${level}] ${message.message || ''}`);
            addLog(level, message.message, options);
            break;
        }
        case 'progress':
            console.log('← Handling progress');
            showProgress(message.message, message.progress);
            break;
        case 'hide_progress':
            console.log('← Handling hide_progress');
            hideProgress();
            break;
        case 'revert_applied':
            console.log('← Handling revert_applied');
            handleRevertApplied(message);
            break;
        case 'reasoning_chunk':
            console.log('← Handling reasoning_chunk');
            handleReasoningChunk(message.content);
            break;
        case 'llm_message': {
            log('← Handling llm_message');
            // Ensure a run container exists so assistant output is visible in the timeline
            const runState = startRunLogSession() || state.activeRun;
            if (!runState) {
                warn('[RUNLOG] Unable to start run log for llm_message');
            } else {
                log(`[RUNLOG] llm_message will log to run id=${runState.runData?.id || runState.id || 'unknown'}`);
            }
            const messageFormat = typeof message.format === 'string'
                ? message.format.trim().toLowerCase()
                : 'html';
            const appended = addLog('agent', message.message || '', {
                scope: 'run',
                messageFormat
            });
            if (!appended) {
                warn('[RUNLOG] addLog returned false for llm_message; falling back to chat bubble');
                appendMessage('assistant', message.message || '', { sender: 'CADAgent', variant: 'agent', contentFormat: messageFormat === 'markdown' ? 'markdown' : 'text' });
            }
            break;
        }
        case 'completed':
            log('← Handling completed');
            completeActiveReasoning({ reason: 'complete', collapse: true });
            // Update run summary to "Designed" when execution completes
            if (state.activeRun && state.activeRun.summaryTextEl) {
                state.activeRun.summaryTextEl.textContent = 'Designed';
                state.activeRun.summaryTextEl.classList.remove('shimmer');
                state.activeRun.summaryTextEl.removeAttribute('title');
            }
            finalizeActiveRun('success');
            hideProgress();
            break;
        case 'cancelled':
            console.log('← Handling cancelled');
            completeActiveReasoning({ reason: 'cancelled', collapse: true });
            addLog('warning', message.message || 'Request cancelled', { scope: 'run' });
            // Remove shimmer animation on cancellation
            if (state.activeRun && state.activeRun.summaryTextEl) {
                state.activeRun.summaryTextEl.classList.remove('shimmer');
            }
            finalizeActiveRun('cancelled');
            hideProgress();
            break;
        case 'error':
            console.log('← Handling error');
            completeActiveReasoning({ reason: 'error', collapse: false });
            addLog('error', message.message, { scope: 'run' });
            // Remove shimmer animation on error
            if (state.activeRun && state.activeRun.summaryTextEl) {
                state.activeRun.summaryTextEl.classList.remove('shimmer');
            }
            finalizeActiveRun('error');
            hideProgress();
            break;
        case 'plan_chunk':
            console.log('← Handling plan_chunk');
            completeActiveReasoning({ reason: 'plan' });
            showProgress('Generating plan...');
            break;
        case 'execute_code':
            console.log('← Handling execute_code');
            completeActiveReasoning({ reason: 'tool', operation: message.operation });
            startRunLogSession();
            // Display description if present
            if (message.description && message.description.trim()) {
                // Description is plain text from LLM thinking - display as text, not HTML
                addLog('agent', message.description, { scope: 'run', messageFormat: 'text' });
            }
            addLog('info', `Executing: ${message.operation}`, { scope: 'run' });
            showProgress(`Executing ${message.operation}...`);
            break;
        case 'edge_operation':
        case 'face_operation':
        case 'body_operation':
        case 'feature_operation':
            console.log(`← Handling ${message.type}`);
            completeActiveReasoning({ reason: 'tool', operation: message.operation });
            startRunLogSession();
            // Display description if present
            if (message.description && message.description.trim()) {
                // Description is plain text from LLM thinking - display as text, not HTML
                addLog('agent', message.description, { scope: 'run', messageFormat: 'text' });
            }
            addLog('info', `Executing: ${message.operation}`, { scope: 'run' });
            showProgress(`Executing ${message.operation}...`);
            break;
        case 'selection_feedback':
            console.log('← Handling selection_feedback');
            handleSelectionFeedback(message);
            break;
        case 'auth_otp_required':
            console.log('← Handling auth_otp_required');
            // Existing user - OTP was sent, show the code input stage
            if (elements.sendOtpBtn) {
                elements.sendOtpBtn.disabled = false;
                elements.sendOtpBtn.textContent = 'Continue';
            }
            // Start resend cooldown (60s) to align with Supabase limits
            startResendCooldown(60);
            // Reveal OTP section for code entry
            setLoginStage('code');
            showLoginStatus(message.message || 'Code sent! Check your email.', 'success');
            break;
        case 'auth_success':
            console.log('← Handling auth_success');

            // Mark verification complete to prevent watchdog retries
            authVerificationCompleted = true;
            
            // Remove any pending verify_otp_code from queue to prevent duplicate requests
            pendingToAddin = pendingToAddin.filter(item => item.data.action !== 'verify_otp_code');

            // Clear auth watchdog - verification completed successfully
            clearAuthWatchdog();

            if (elements.sendOtpBtn) {
                elements.sendOtpBtn.disabled = false;
                elements.sendOtpBtn.textContent = 'Continue';
            }
            if (elements.verifyOtpBtn) {
                elements.verifyOtpBtn.disabled = false;
                elements.verifyOtpBtn.textContent = 'Verify Code';
            }
            if (elements.passwordLoginBtn) {
                elements.passwordLoginBtn.disabled = false;
                elements.passwordLoginBtn.textContent = 'Log in';
            }
            showLoginStatus(message.message || 'Success!', 'success');
            if (message.user && message.user.email) {
                // Update UI to show logged in state
                updateAuthUI(message.user.email);
                // Close modal after short delay
                setTimeout(() => {
                    hideLoginOverlay();
                    resetLoginFields();
                    setLoginStage('cta');
                }, 1500);
            }
            break;
        case 'logout_success':
            console.log('← Handling logout_success');
            showLoginStatus(message.message || 'Logged out successfully', 'success');
            // Redirect to login page
            updateAuthUI(null);
            resetLoginFields();
            setLoginStage('cta');
            break;
        case 'auth_error':
            console.log('← Handling auth_error');

            // Mark verification complete to prevent watchdog retries
            // (Only set if this is from the original verify, not a watchdog retry that failed)
            // We check if the error is "otp_expired" which indicates a retry after success
            const isRetryError = message.message && (
                message.message.includes('expired') || 
                message.message.includes('invalid') ||
                message.message.includes('already used')
            );
            
            if (!isRetryError) {
                // This is a genuine first-time error, mark complete
                authVerificationCompleted = true;
            } else {
                console.log('← Ignoring retry-related auth_error (OTP already used/expired)');
                // Don't mark complete - the first success/error is the real result
                // Just clean up the queue
            }
            
            // Remove any pending verify_otp_code from queue
            pendingToAddin = pendingToAddin.filter(item => item.data.action !== 'verify_otp_code');

            // Clear auth watchdog
            clearAuthWatchdog();

            if (elements.sendOtpBtn) {
                elements.sendOtpBtn.disabled = false;
                elements.sendOtpBtn.textContent = 'Continue';
            }
            if (elements.verifyOtpBtn) {
                elements.verifyOtpBtn.disabled = false;
                elements.verifyOtpBtn.textContent = 'Verify Code';
            }
            if (elements.passwordLoginBtn) {
                elements.passwordLoginBtn.disabled = false;
                elements.passwordLoginBtn.textContent = 'Log in';
            }
            
            // Only show error to user if it's not a retry-after-success scenario
            if (!authVerificationCompleted || !isRetryError) {
                // Suppress generic noise on the login screen
                const msg = message.message || '';
                const isGenericAuthPrompt = msg === 'Please log in to use CADAgent' || msg === 'Not authenticated';
                if (!isGenericAuthPrompt) {
                    showLoginStatus(msg || 'Authentication error', 'error');
                }
                // Update UI to show logged out state but keep user input stage
                updateAuthUI(null, { preserveStage: true });
            }
            break;
        case 'user_profile':
            console.log('← Handling user_profile');
            if (message.profile && message.profile.email) {
                updateAuthUI(message.profile.email);
            } else {
                updateAuthUI(null, { preserveStage: true });
            }
            break;
        case 'question_tree_generated':
            console.log('← Handling question_tree_generated');
            handleQuestionTreeGenerated(message.data);
            break;
        case 'designs_proposed':
            console.log('← Handling designs_proposed');
            handleDesignsProposed(message.data);
            break;
        case 'build_plan_generated':
            console.log('← Handling build_plan_generated');
            handleBuildPlanGenerated(message.data);
            break;
        case 'build_step_completed':
            console.log('← Handling build_step_completed');
            handleBuildStepCompleted(message.data);
            break;
        case 'build_plan_completed':
            console.log('← Handling build_plan_completed');
            handleBuildPlanCompleted(message.data);
            break;
        case 'api_keys_status':
            console.log('← Handling api_keys_status');
            if (message.status) {
                updateApiKeysUI(message.status);
            }
            break;
        case 'api_keys_saved':
            console.log('← Handling api_keys_saved');
            if (elements.saveApiKeysBtn) {
                elements.saveApiKeysBtn.disabled = false;
                elements.saveApiKeysBtn.textContent = 'Save Configuration';
            }
            if (message.success) {
                showApiKeysStatus('API keys saved successfully!', 'success');
                // Update status badges
                if (message.status) {
                    updateApiKeysUI(message.status);
                }
                // Close modal after short delay
                setTimeout(() => {
                    hideApiKeysModal();
                    resetApiKeysForm();
                }, 1500);
            } else {
                showApiKeysStatus(message.error || 'Failed to save API keys', 'error');
            }
            break;
        default:
            console.warn(`⚠️  UNHANDLED MESSAGE TYPE: '${message.type}'`);
            console.warn(`↳ Message details:`, message);
            console.warn(`↳ This message was received but has no handler in processAddinMessage()`);
    }
}

/**
 * Handle messages from Python add-in
 * This function is called by the Python code
 */
function handleAddinMessage(messageJson) {
    log('='.repeat(60));
    log('← MESSAGE FROM PYTHON');
    log('='.repeat(60));
    log('← Raw JSON:', messageJson);

    try {
        const message = JSON.parse(messageJson);
        log('← Parsed message:', message);
        log('← Message type:', message.type);
        dispatchAddinMessage(message);
    } catch (error) {
        error('❌ Failed to parse message:', error);
        addLog('error', `Failed to parse message: ${error.message}`, { scope: 'global' });
    }

    log('='.repeat(60));
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}

// Request status update every 5 seconds
setInterval(requestConnectionStatus, 5000);

// Reasoning effort options per provider
const REASONING_OPTIONS = {
    openai: [
        { value: 'off', label: 'COT: Off' },
        { value: 'low', label: 'COT: Low' },
        { value: 'medium', label: 'COT: Medium' },
        { value: 'high', label: 'COT: High' },
        { value: 'xhigh', label: 'COT: Extra High' },
    ],
    anthropic: [
        { value: 'off', label: 'COT: Off' },
        { value: 'on', label: 'COT: On' },
    ],
};

const REASONING_DEFAULTS = {
    openai: 'medium',
    anthropic: 'off',
};

// Explicit provider mapping to avoid misclassifying non-gpt OpenAI models (e.g., o1/o3).
// Keep this list in sync with any new model options.
const MODEL_PROVIDER_MAP = {
    // OpenAI
    'gpt-5': 'openai',
    'gpt-5.4': 'openai',
    'gpt-5-mini': 'openai',
    'o1': 'openai',
    'o1-mini': 'openai',
    'o3': 'openai',
    'o3-mini': 'openai',
    'o4': 'openai',
    // Anthropic
    'claude-sonnet-4.6': 'anthropic',
    'claude-opus-4.7': 'anthropic',
};

/**
 * Determine the provider key for reasoning options based on model value.
 */
function getProviderForModel(modelValue) {
    if (!modelValue) return 'anthropic';
    if (MODEL_PROVIDER_MAP[modelValue]) return MODEL_PROVIDER_MAP[modelValue];
    if (modelValue.startsWith('gpt')) return 'openai';
    if (modelValue.startsWith('o1') || modelValue.startsWith('o3') || modelValue.startsWith('o4')) return 'openai';
    return 'anthropic';
}

function isProviderConfigured(provider) {
    return !!state.apiKeyStatus?.[provider];
}

function buildMissingApiKeyMessage(provider) {
    const loginPrefix = state.isAuthenticated
        ? ''
        : 'Log in first, then ';
    return `No API key is configured. ${loginPrefix}Open Settings -> API Keys, add an API key, and retry.`;
}

function isModelSelectable(modelValue) {
    const provider = getProviderForModel(modelValue);
    return isProviderConfigured(provider);
}

function enforceModelSelectionByKeys() {
    const sel = elements.modelSelect;
    if (!sel) return;

    let firstAllowedValue = null;
    Array.from(sel.options).forEach((option) => {
        const allowed = isModelSelectable(option.value);
        option.disabled = !allowed;
        if (!firstAllowedValue && allowed) {
            firstAllowedValue = option.value;
        }
    });

    const currentAllowed = !!sel.value && isModelSelectable(sel.value);
    if (!currentAllowed && firstAllowedValue) {
        sel.value = firstAllowedValue;
        state.selectedModel = firstAllowedValue;
        updateReasoningOptions();
    }

    sel.disabled = !firstAllowedValue;
    if (!firstAllowedValue) {
        sel.title = 'Add an API key to enable model selection';
    } else {
        sel.title = 'Select AI model';
    }

    try { autoSizeModelSelect(); } catch (e) {}
}

function updateImageUploadAvailability() {
    if (!elements.attachImageBtn) return;

    const hasOpenAIKey = isProviderConfigured('openai');
    elements.attachImageBtn.disabled = !hasOpenAIKey;
    elements.attachImageBtn.setAttribute(
        'title',
        hasOpenAIKey
            ? 'Attach sketch image'
            : 'Add an OpenAI API key to use image uploads (GPT-5.4)'
    );

    if (!hasOpenAIKey && state.attachedImage) {
        handleRemoveImage();
        addLog('warning', buildMissingApiKeyMessage('openai'), { scope: 'global' });
    }
}

/**
 * Repopulate the reasoning selector based on the currently selected model.
 * Sets the default reasoning effort when switching between providers.
 */
function updateReasoningOptions() {
    const sel = elements.reasoningSelect;
    if (!sel) return;

    const provider = getProviderForModel(state.selectedModel);
    const options = REASONING_OPTIONS[provider] || REASONING_OPTIONS.anthropic;
    const defaultVal = REASONING_DEFAULTS[provider] || 'off';

    // Remember current selection if staying in same provider
    const currentProvider = sel.dataset.provider;
    const keepSelection = currentProvider === provider;

    sel.innerHTML = '';
    options.forEach(opt => {
        const el = document.createElement('option');
        el.value = opt.value;
        el.textContent = opt.label;
        sel.appendChild(el);
    });

    sel.dataset.provider = provider;

    if (keepSelection && options.some(o => o.value === state.reasoningEffort)) {
        sel.value = state.reasoningEffort;
    } else {
        sel.value = defaultVal;
        state.reasoningEffort = defaultVal;
    }

    try { autoSizeReasoningSelect(); } catch (e) {}
}

/**
 * Auto-size the reasoning select to the width of its selected option text.
 */
function autoSizeReasoningSelect() {
    const sel = elements.reasoningSelect;
    if (!sel) return;

    const selectedText = sel.options[sel.selectedIndex]?.text || '';
    let measurer = document.getElementById('reasoning-select-measurer');
    if (!measurer) {
        measurer = document.createElement('span');
        measurer.id = 'reasoning-select-measurer';
        measurer.style.position = 'absolute';
        measurer.style.visibility = 'hidden';
        measurer.style.whiteSpace = 'pre';
        measurer.style.left = '-9999px';
        document.body.appendChild(measurer);
    }

    const cs = window.getComputedStyle(sel);
    measurer.style.fontFamily = cs.fontFamily;
    measurer.style.fontSize = cs.fontSize;
    measurer.style.fontWeight = cs.fontWeight;
    measurer.style.letterSpacing = cs.letterSpacing;
    measurer.textContent = selectedText;

    const textWidth = Math.ceil(measurer.getBoundingClientRect().width);
    const padL = parseFloat(cs.paddingLeft) || 0;
    const padR = parseFloat(cs.paddingRight) || 0;
    const borderL = parseFloat(cs.borderLeftWidth) || 0;
    const borderR = parseFloat(cs.borderRightWidth) || 0;
    const affordance = 2;
    const total = textWidth + padL + padR + borderL + borderR + affordance;
    sel.style.width = `${total}px`;
}

/**
 * Auto-size the model select to the width of its selected option text.
 * Works across browsers by measuring in a hidden mirror element.
 */
function autoSizeModelSelect() {
    const sel = elements.modelSelect;
    if (!sel) return;

    const selectedText = sel.options[sel.selectedIndex]?.text || '';
    // Create or reuse a hidden measurer
    let measurer = document.getElementById('model-select-measurer');
    if (!measurer) {
        measurer = document.createElement('span');
        measurer.id = 'model-select-measurer';
        measurer.style.position = 'absolute';
        measurer.style.visibility = 'hidden';
        measurer.style.whiteSpace = 'pre';
        measurer.style.left = '-9999px';
        document.body.appendChild(measurer);
    }

    const cs = window.getComputedStyle(sel);
    // Apply matching font styles
    measurer.style.fontFamily = cs.fontFamily;
    measurer.style.fontSize = cs.fontSize;
    measurer.style.fontWeight = cs.fontWeight;
    measurer.style.letterSpacing = cs.letterSpacing;
    measurer.textContent = selectedText;

    const textWidth = Math.ceil(measurer.getBoundingClientRect().width);
    // Add horizontal padding and borders from the select styles
    const padL = parseFloat(cs.paddingLeft) || 0;
    const padR = parseFloat(cs.paddingRight) || 0;
    const borderL = parseFloat(cs.borderLeftWidth) || 0;
    const borderR = parseFloat(cs.borderRightWidth) || 0;
    // Extra room for dropdown affordance even when appearance:none
    const affordance = 2;
    const total = textWidth + padL + padR + borderL + borderR + affordance;
    sel.style.width = `${total}px`;
}

/**
 * Animate <details> open/close as a dropdown by smoothly transitioning
 * the height of the .run-actions panel instead of the default instant switch.
 */
function animateRunDetails(detailsEl) {
    const actions = detailsEl.querySelector('.run-actions');
    if (!actions) return;

    const duration = 250; // ms

    if (detailsEl.open) {
        // Opening: from 0 to natural height
        actions.style.willChange = 'max-height, opacity, transform';
        actions.style.overflow = 'hidden';
        actions.style.opacity = '0';
        actions.style.transform = 'translateY(-6px)';
        actions.style.maxHeight = '0px';

        // Force reflow before measuring and expanding
        actions.getBoundingClientRect();
        const target = actions.scrollHeight;
        actions.style.transition = `max-height ${duration}ms ease, opacity ${Math.min(duration,200)}ms ease, transform ${Math.min(duration,200)}ms ease`;
        actions.style.maxHeight = `${target}px`;
        actions.style.opacity = '1';
        actions.style.transform = 'translateY(0)';
        if (target > 0) {
            actions.dataset.runHeight = String(target);
        }

        // Cleanup after animation
        setTimeout(() => {
            actions.style.maxHeight = 'none';
            actions.style.willChange = '';
            actions.style.overflow = '';
        }, duration + 30);
    } else {
        // Closing: from current height to 0
        const storedHeight = Number(actions.dataset.runHeight || 0);
        const start = actions.scrollHeight || storedHeight;
        if (!start) {
            actions.style.maxHeight = '';
            actions.style.opacity = '';
            actions.style.transform = '';
            actions.style.willChange = '';
            actions.style.overflow = '';
            actions.style.transition = '';
            actions.dataset.runHeight = '';
            return;
        }
        actions.style.willChange = 'max-height, opacity, transform';
        actions.style.overflow = 'hidden';
        actions.style.maxHeight = `${start}px`;
        actions.style.opacity = '1';
        actions.style.transform = 'translateY(0)';

        // Force reflow
        actions.getBoundingClientRect();
        actions.style.transition = `max-height ${duration}ms ease, opacity ${Math.min(duration,200)}ms ease, transform ${Math.min(duration,200)}ms ease`;
        actions.style.maxHeight = '0px';
        actions.style.opacity = '0';
        actions.style.transform = 'translateY(-6px)';

        setTimeout(() => {
            // remove inline styles to reset to closed state
            actions.style.transition = '';
            actions.style.maxHeight = '';
            actions.style.opacity = '';
            actions.style.transform = '';
            actions.style.willChange = '';
            actions.style.overflow = '';
            actions.dataset.runHeight = '';
        }, duration + 30);
    }
}
