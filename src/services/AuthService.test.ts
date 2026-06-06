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
