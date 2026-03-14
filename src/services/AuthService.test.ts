import test from 'node:test';
import assert from 'node:assert';
import { AuthService } from './AuthService.ts';

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
