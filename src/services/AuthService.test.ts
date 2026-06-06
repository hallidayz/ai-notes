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
