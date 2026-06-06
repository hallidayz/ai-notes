import test, { after } from 'node:test';
import assert from 'node:assert';

// 1. Store original globals for restoration
const originalDocument = global.document;
const originalWindow = global.window;
const originalDateNow = Date.now;
const originalPublicKeyCredential = global.PublicKeyCredential;

// 2. Mock browser globals BEFORE importing AuthService
// this is necessary because AuthService may use these globals during module evaluation
global.document = {
    addEventListener: () => {},
    removeEventListener: () => {},
    hidden: false,
} as any;

global.window = {
    clearTimeout: () => {},
    setTimeout: () => 0,
} as any;

global.PublicKeyCredential = {
    isUserVerifyingPlatformAuthenticatorAvailable: async () => false,
} as any;

// Mock Date.now()
let currentTime = 1000000;
Date.now = () => currentTime;

// 3. Import AuthService after globals are set
import { AuthService } from './AuthService.ts';

function advanceTime(ms: number) {
    currentTime += ms;
}

const mockDb = {
    getConfig: async () => null,
    saveConfig: async () => {},
};

// 4. Register restoration in after hook
after(() => {
    global.document = originalDocument;
    global.window = originalWindow;
    global.PublicKeyCredential = originalPublicKeyCredential;
    Date.now = originalDateNow;
});

test('isLockedOut should return false initially', () => {
    const authService = new AuthService(mockDb);
    assert.strictEqual(authService.isLockedOut(), false);
});

test('isLockedOut should return true after max failed attempts', async () => {
    const config = { maxFailedAttempts: 3, lockoutDuration: 1000 };
    const authService = new AuthService(mockDb, config);

    await authService.recordFailedAttempt(); // 1
    await authService.recordFailedAttempt(); // 2
    assert.strictEqual(authService.isLockedOut(), false, 'Should not be locked out yet');

    await authService.recordFailedAttempt(); // 3 - Locked out
    assert.strictEqual(authService.isLockedOut(), true, 'Should be locked out after max attempts');
});

test('isLockedOut should return false and reset state after lockout expires', async () => {
    const config = { maxFailedAttempts: 3, lockoutDuration: 1000 };
    const authService = new AuthService(mockDb, config);

    // Trigger lockout
    await authService.recordFailedAttempt();
    await authService.recordFailedAttempt();
    await authService.recordFailedAttempt();
    assert.strictEqual(authService.isLockedOut(), true, 'Should be locked out');

    // Advance time to exactly lockout expiration
    advanceTime(1000);
    assert.strictEqual(authService.isLockedOut(), false, 'Should not be locked out after expiration');

    // Verify state was reset
    const state = authService.getState();
    assert.strictEqual(state.lockoutUntil, null, 'lockoutUntil should be reset');
    assert.strictEqual(state.failedAttempts, 0, 'failedAttempts should be reset');
    assert.strictEqual(state.isLocked, false, 'isLocked should be false');
});

test('isLockedOut should return true if lockout is not yet expired', async () => {
    const config = { maxFailedAttempts: 3, lockoutDuration: 1000 };
    const authService = new AuthService(mockDb, config);

    await authService.recordFailedAttempt();
    await authService.recordFailedAttempt();
    await authService.recordFailedAttempt();

    // Advance time by 500ms (half of lockout duration)
    advanceTime(500);
    assert.strictEqual(authService.isLockedOut(), true, 'Should still be locked out before expiration');
});

test('isBiometricAvailable returns false when window is undefined', async (t) => {
    // Save original globals
    const originalWindow = (global as any).window;
    const originalDocument = (global as any).document;

    // Mock document to avoid error in setupActivityTracking
    const mockDocument = {
        addEventListener: () => {},
        removeEventListener: () => {}
    };
    Object.defineProperty(global, 'document', { value: mockDocument, configurable: true, writable: true });

    // Override window to undefined
    Object.defineProperty(global, 'window', { value: undefined, configurable: true, writable: true });

    // Create AuthService
    const db = {}; // Mock db
    const authService = new AuthService(db);

    const isAvailable = await authService.isBiometricAvailable();
    assert.strictEqual(isAvailable, false);

    // Restore window and document
    t.after(() => {
        if (originalWindow === undefined) {
            delete (global as any).window;
        } else {
            Object.defineProperty(global, 'window', { value: originalWindow, configurable: true, writable: true });
        }

        if (originalDocument === undefined) {
            delete (global as any).document;
        } else {
            Object.defineProperty(global, 'document', { value: originalDocument, configurable: true, writable: true });
        }
    });
});

test('isBiometricAvailable returns false when PublicKeyCredential is not in window', async (t) => {
    // Save original globals
    const originalWindow = (global as any).window;
    const originalDocument = (global as any).document;

    // Mock document
    const mockDocument = {
        addEventListener: () => {},
        removeEventListener: () => {}
    };
    Object.defineProperty(global, 'document', { value: mockDocument, configurable: true, writable: true });

    // Override window but without PublicKeyCredential
    const mockWindow = {
        addEventListener: () => {},
        removeEventListener: () => {}
    };
    Object.defineProperty(global, 'window', { value: mockWindow, configurable: true, writable: true });

    // Create AuthService
    const db = {}; // Mock db
    const authService = new AuthService(db);

    const isAvailable = await authService.isBiometricAvailable();
    assert.strictEqual(isAvailable, false);

    // Restore window and document
    t.after(() => {
        if (originalWindow === undefined) {
            delete (global as any).window;
        } else {
            Object.defineProperty(global, 'window', { value: originalWindow, configurable: true, writable: true });
        }

        if (originalDocument === undefined) {
            delete (global as any).document;
        } else {
            Object.defineProperty(global, 'document', { value: originalDocument, configurable: true, writable: true });
        }
    });
});

test('isBiometricAvailable returns true when WebAuthn is available', async (t) => {
    // Save original globals
    const originalWindow = (global as any).window;
    const originalDocument = (global as any).document;
    const originalPublicKeyCredential = (global as any).PublicKeyCredential;

    // Mock document
    const mockDocument = {
        addEventListener: () => {},
        removeEventListener: () => {}
    };
    Object.defineProperty(global, 'document', { value: mockDocument, configurable: true, writable: true });

    // Override window
    const mockWindow = {
        addEventListener: () => {},
        removeEventListener: () => {},
        PublicKeyCredential: {}
    };
    Object.defineProperty(global, 'window', { value: mockWindow, configurable: true, writable: true });

    // Override PublicKeyCredential globally as well, since it's referenced directly
    const mockPublicKeyCredential = {
        isUserVerifyingPlatformAuthenticatorAvailable: async () => true
    };
    Object.defineProperty(global, 'PublicKeyCredential', { value: mockPublicKeyCredential, configurable: true, writable: true });

    // Create AuthService
    const db = {}; // Mock db
    const authService = new AuthService(db);

    const isAvailable = await authService.isBiometricAvailable();
    assert.strictEqual(isAvailable, true);

    // Restore window, document, and PublicKeyCredential
    t.after(() => {
        if (originalWindow === undefined) {
            delete (global as any).window;
        } else {
            Object.defineProperty(global, 'window', { value: originalWindow, configurable: true, writable: true });
        }

        if (originalDocument === undefined) {
            delete (global as any).document;
        } else {
            Object.defineProperty(global, 'document', { value: originalDocument, configurable: true, writable: true });
        }

        if (originalPublicKeyCredential === undefined) {
            delete (global as any).PublicKeyCredential;
        } else {
            Object.defineProperty(global, 'PublicKeyCredential', { value: originalPublicKeyCredential, configurable: true, writable: true });
        }
    });
});

test('isBiometricAvailable returns false when enableBiometric is false', async (t) => {
    // Save original globals
    const originalWindow = (global as any).window;
    const originalDocument = (global as any).document;
    const originalPublicKeyCredential = (global as any).PublicKeyCredential;

    // Mock document
    const mockDocument = {
        addEventListener: () => {},
        removeEventListener: () => {}
    };
    Object.defineProperty(global, 'document', { value: mockDocument, configurable: true, writable: true });

    // Override window
    const mockWindow = {
        addEventListener: () => {},
        removeEventListener: () => {},
        PublicKeyCredential: {}
    };
    Object.defineProperty(global, 'window', { value: mockWindow, configurable: true, writable: true });

    // Override PublicKeyCredential globally
    const mockPublicKeyCredential = {
        isUserVerifyingPlatformAuthenticatorAvailable: async () => true
    };
    Object.defineProperty(global, 'PublicKeyCredential', { value: mockPublicKeyCredential, configurable: true, writable: true });

    // Create AuthService with enableBiometric = false
    const db = {}; // Mock db
    const authService = new AuthService(db, { enableBiometric: false });

    const isAvailable = await authService.isBiometricAvailable();
    assert.strictEqual(isAvailable, false);

    // Restore globals
    t.after(() => {
        if (originalWindow === undefined) {
            delete (global as any).window;
        } else {
            Object.defineProperty(global, 'window', { value: originalWindow, configurable: true, writable: true });
        }

        if (originalDocument === undefined) {
            delete (global as any).document;
        } else {
            Object.defineProperty(global, 'document', { value: originalDocument, configurable: true, writable: true });
        }

        if (originalPublicKeyCredential === undefined) {
            delete (global as any).PublicKeyCredential;
        } else {
            Object.defineProperty(global, 'PublicKeyCredential', { value: originalPublicKeyCredential, configurable: true, writable: true });
        }
    });
});

test('isBiometricAvailable returns false when WebAuthn throws an error', async (t) => {
    // Save original globals
    const originalWindow = (global as any).window;
    const originalDocument = (global as any).document;
    const originalPublicKeyCredential = (global as any).PublicKeyCredential;

    // Mock document
    const mockDocument = {
        addEventListener: () => {},
        removeEventListener: () => {}
    };
    Object.defineProperty(global, 'document', { value: mockDocument, configurable: true, writable: true });

    // Override window
    const mockWindow = {
        addEventListener: () => {},
        removeEventListener: () => {},
        PublicKeyCredential: {}
    };
    Object.defineProperty(global, 'window', { value: mockWindow, configurable: true, writable: true });

    // Override PublicKeyCredential to throw error
    const mockPublicKeyCredential = {
        isUserVerifyingPlatformAuthenticatorAvailable: async () => { throw new Error('Not allowed'); }
    };
    Object.defineProperty(global, 'PublicKeyCredential', { value: mockPublicKeyCredential, configurable: true, writable: true });

    // Create AuthService
    const db = {}; // Mock db
    const authService = new AuthService(db);

    const isAvailable = await authService.isBiometricAvailable();
    assert.strictEqual(isAvailable, false);

    // Restore globals
    t.after(() => {
        if (originalWindow === undefined) {
            delete (global as any).window;
        } else {
            Object.defineProperty(global, 'window', { value: originalWindow, configurable: true, writable: true });
        }

        if (originalDocument === undefined) {
            delete (global as any).document;
        } else {
            Object.defineProperty(global, 'document', { value: originalDocument, configurable: true, writable: true });
        }

        if (originalPublicKeyCredential === undefined) {
            delete (global as any).PublicKeyCredential;
        } else {
            Object.defineProperty(global, 'PublicKeyCredential', { value: originalPublicKeyCredential, configurable: true, writable: true });
        }
    });
});

test('AuthService - Failed attempt lockout logic', async (t) => {
    const originalWindow = global.window;
    const originalDocument = global.document;
    const originalDateNow = Date.now;

    (global as any).window = {
        setTimeout: (cb: any, ms: number) => setTimeout(cb, ms),
        clearTimeout: (id: any) => clearTimeout(id),
    };

    (global as any).document = {
        addEventListener: () => {},
        hidden: false,
    };

    t.after(() => {
        (global as any).window = originalWindow;
        (global as any).document = originalDocument;
        Date.now = originalDateNow;
    });

    let mockDbStorage: Record<string, string> = {};
    const mockDb = {
        getConfig: async (key: string) => mockDbStorage[key] || null,
        saveConfig: async (key: string, value: string) => { mockDbStorage[key] = value; }
    };

    await t.test('recordFailedAttempt increments counter and returns remaining attempts', async () => {
        mockDbStorage = {};
        const service = new AuthService(mockDb, { maxFailedAttempts: 3 });

        const result = await service.recordFailedAttempt();

        assert.strictEqual(result.isLocked, false);
        assert.strictEqual(result.message, 'Invalid PIN. 2 attempt(s) remaining.');

        const state = service.getState();
        assert.strictEqual(state.failedAttempts, 1);
        assert.strictEqual(state.isLocked, false);
    });

    await t.test('recordFailedAttempt locks account when reaching max failed attempts', async () => {
        mockDbStorage = {};
        const currentTime = 1000000;
        Date.now = () => currentTime;

        const service = new AuthService(mockDb, { maxFailedAttempts: 2, lockoutDuration: 60000 });

        await service.recordFailedAttempt();
        const result = await service.recordFailedAttempt();

        assert.strictEqual(result.isLocked, true);
        assert.strictEqual(result.message, 'Too many failed attempts. Account locked for 1 minutes.');

        const state = service.getState();
        assert.strictEqual(state.failedAttempts, 2);
        assert.strictEqual(state.isLocked, true);
        assert.strictEqual(state.lockoutUntil, currentTime + 60000);

        const savedState = JSON.parse(mockDbStorage['authState']);
        assert.strictEqual(savedState.failedAttempts, 2);
        assert.strictEqual(savedState.lockoutUntil, currentTime + 60000);
    });

    await t.test('canAttemptAuth respects lockout state', async () => {
        mockDbStorage = {};
        const currentTime = 1000000;
        Date.now = () => currentTime;

        const service = new AuthService(mockDb, { maxFailedAttempts: 1, lockoutDuration: 120000 });

        await service.recordFailedAttempt();
        const check = service.canAttemptAuth();

        assert.strictEqual(check.allowed, false);
        assert.strictEqual(check.message, 'Too many failed attempts. Please try again in 2 minute(s).');
    });

    await t.test('isLockedOut automatically unlocks when duration expires', async () => {
        mockDbStorage = {};
        let currentTime = 1000000;
        Date.now = () => currentTime;

        const service = new AuthService(mockDb, { maxFailedAttempts: 1, lockoutDuration: 60000 });

        await service.recordFailedAttempt();
        assert.strictEqual(service.isLockedOut(), true);

        currentTime += 60001;
        assert.strictEqual(service.isLockedOut(), false);

        const state = service.getState();
        assert.strictEqual(state.failedAttempts, 0);
        assert.strictEqual(state.isLocked, false);
        assert.strictEqual(state.lockoutUntil, null);
    });
});

test('AuthService Auto-lock and Activity Tracking', async (t) => {
    const originalWindow = global.window;
    const originalDocument = global.document;
    const originalDateNow = Date.now;
    const originalPublicKeyCredential = global.PublicKeyCredential;

    let currentTime = 1000000;
    Date.now = () => currentTime;

    const addedEventListeners: { event: string; handler: any; options?: any }[] = [];
    let mockDocumentHidden = false;
    const mockDocument = {
        addEventListener: (event: string, handler: any, options?: any) => {
            addedEventListeners.push({ event, handler, options });
        },
        get hidden() {
            return mockDocumentHidden;
        },
        set hidden(val) {
            mockDocumentHidden = val;
        }
    };
    global.document = mockDocument as any;

    let timeoutIdCounter = 1;
    const activeTimeouts = new Map<number, { callback: Function, delay: number }>();
    const mockWindow = {
        setTimeout: (callback: Function, delay: number) => {
            const id = timeoutIdCounter++;
            activeTimeouts.set(id, { callback, delay });
            return id;
        },
        clearTimeout: (id: number) => {
            activeTimeouts.delete(id);
        }
    };
    global.window = mockWindow as any;
    global.PublicKeyCredential = {} as any;

    t.after(() => {
        global.window = originalWindow;
        global.document = originalDocument;
        Date.now = originalDateNow;
        global.PublicKeyCredential = originalPublicKeyCredential;
    });

    const mockDb = {
        getConfig: async () => null,
        saveConfig: async () => {}
    };

    await t.test('Registers event listeners on initialization', () => {
        addedEventListeners.length = 0;
        const service = new AuthService(mockDb);
        const expectedEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click', 'visibilitychange'];
        const registeredEvents = addedEventListeners.map(e => e.event);
        expectedEvents.forEach(event => {
            assert.ok(registeredEvents.includes(event), `Missing listener for ${event}`);
        });
        service.cleanup();
    });

    await t.test('Activity event updates timer when authenticated', () => {
        addedEventListeners.length = 0;
        activeTimeouts.clear();
        const service = new AuthService(mockDb, { autoLockTimeout: 300000 });
        service.recordSuccess();
        assert.strictEqual(activeTimeouts.size, 1, 'Should set initial inactivity timer after login');
        const initialTimeoutIds = Array.from(activeTimeouts.keys());
        const mouseEvent = addedEventListeners.find(e => e.event === 'mousedown');
        assert.ok(mouseEvent, 'Mouse event listener should be registered');
        currentTime += 10000;
        mouseEvent!.handler();
        assert.strictEqual(activeTimeouts.size, 1, 'Should have exactly one active timeout');
        const newTimeoutIds = Array.from(activeTimeouts.keys());
        assert.notStrictEqual(newTimeoutIds[0], initialTimeoutIds[0], 'Should have cleared old timeout and set a new one');
        service.cleanup();
    });

    await t.test('Activity event does not set timer when not authenticated', () => {
        addedEventListeners.length = 0;
        activeTimeouts.clear();
        const service = new AuthService(mockDb);
        assert.strictEqual(activeTimeouts.size, 0, 'No timeout should be set initially because not authenticated');
        const mouseEvent = addedEventListeners.find(e => e.event === 'mousedown');
        mouseEvent?.handler();
        assert.strictEqual(activeTimeouts.size, 0, 'No timeout should be set after activity if not authenticated');
        service.cleanup();
    });

    await t.test('Auto-lock timer successfully locks the app', () => {
        addedEventListeners.length = 0;
        activeTimeouts.clear();
        const service = new AuthService(mockDb, { autoLockTimeout: 300000 });
        service.recordSuccess();
        assert.strictEqual(service.getState().isAuthenticated, true);
        assert.strictEqual(service.getState().isLocked, false);
        assert.strictEqual(activeTimeouts.size, 1);
        const timeoutInfo = Array.from(activeTimeouts.values())[0];
        timeoutInfo.callback();
        assert.strictEqual(service.getState().isAuthenticated, false);
        assert.strictEqual(service.getState().isLocked, true);
        service.cleanup();
    });

    await t.test('visibilitychange triggers activity update only when document is visible', () => {
        addedEventListeners.length = 0;
        activeTimeouts.clear();
        const service = new AuthService(mockDb);
        service.recordSuccess();
        const initialTimeoutIds = Array.from(activeTimeouts.keys());
        const visibilityEvent = addedEventListeners.find(e => e.event === 'visibilitychange');
        assert.ok(visibilityEvent, 'visibilitychange event listener should be registered');
        mockDocument.hidden = true;
        visibilityEvent!.handler();
        let newTimeoutIds = Array.from(activeTimeouts.keys());
        assert.deepStrictEqual(newTimeoutIds, initialTimeoutIds, 'Timer should not be reset when document is hidden');
        mockDocument.hidden = false;
        visibilityEvent!.handler();
        newTimeoutIds = Array.from(activeTimeouts.keys());
        assert.strictEqual(activeTimeouts.size, 1);
        assert.notStrictEqual(newTimeoutIds[0], initialTimeoutIds[0], 'Timer should be reset when document becomes visible');
        service.cleanup();
    });
});
