import test from 'node:test';
import assert from 'node:assert';
import { AuthService } from './AuthService.ts';

test('AuthService - Failed attempt lockout logic', async (t) => {
    // Setup globals
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
        mockDbStorage = {}; // reset
        const service = new AuthService(mockDb, { maxFailedAttempts: 3 });

        const result = await service.recordFailedAttempt();

        assert.strictEqual(result.isLocked, false);
        assert.strictEqual(result.message, 'Invalid PIN. 2 attempt(s) remaining.');

        const state = service.getState();
        assert.strictEqual(state.failedAttempts, 1);
        assert.strictEqual(state.isLocked, false);
    });

    await t.test('recordFailedAttempt locks account when reaching max failed attempts', async () => {
        mockDbStorage = {}; // reset
        const currentTime = 1000000;
        Date.now = () => currentTime;

        const service = new AuthService(mockDb, { maxFailedAttempts: 2, lockoutDuration: 60000 });

        // Attempt 1
        await service.recordFailedAttempt();

        // Attempt 2 (should lock)
        const result = await service.recordFailedAttempt();

        assert.strictEqual(result.isLocked, true);
        assert.strictEqual(result.message, 'Too many failed attempts. Account locked for 1 minutes.');

        const state = service.getState();
        assert.strictEqual(state.failedAttempts, 2);
        assert.strictEqual(state.isLocked, true);
        assert.strictEqual(state.lockoutUntil, currentTime + 60000);

        // Verify saved state
        const savedState = JSON.parse(mockDbStorage['authState']);
        assert.strictEqual(savedState.failedAttempts, 2);
        assert.strictEqual(savedState.lockoutUntil, currentTime + 60000);
    });

    await t.test('canAttemptAuth respects lockout state', async () => {
        mockDbStorage = {}; // reset
        const currentTime = 1000000;
        Date.now = () => currentTime;

        const service = new AuthService(mockDb, { maxFailedAttempts: 1, lockoutDuration: 120000 });

        // Trigger lockout
        await service.recordFailedAttempt();

        // Check if we can attempt auth
        const check = service.canAttemptAuth();

        assert.strictEqual(check.allowed, false);
        assert.strictEqual(check.message, 'Too many failed attempts. Please try again in 2 minute(s).');
    });

    await t.test('isLockedOut automatically unlocks when duration expires', async () => {
        mockDbStorage = {}; // reset
        let currentTime = 1000000;
        Date.now = () => currentTime;

        const service = new AuthService(mockDb, { maxFailedAttempts: 1, lockoutDuration: 60000 });

        // Trigger lockout
        await service.recordFailedAttempt();
        assert.strictEqual(service.isLockedOut(), true);

        // Advance time past lockout duration
        currentTime += 60001;

        // Should unlock automatically when checked
        assert.strictEqual(service.isLockedOut(), false);

        const state = service.getState();
        assert.strictEqual(state.failedAttempts, 0);
        assert.strictEqual(state.isLocked, false);
        assert.strictEqual(state.lockoutUntil, null);
    });
});
