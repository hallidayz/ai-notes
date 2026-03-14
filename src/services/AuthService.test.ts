import test from 'node:test';
import assert from 'node:assert';
import { AuthService } from './AuthService.ts';

test('AuthService Auto-lock and Activity Tracking', async (t) => {
    // Save original globals
    const originalWindow = global.window;
    const originalDocument = global.document;
    const originalDateNow = Date.now;
    const originalPublicKeyCredential = global.PublicKeyCredential;

    // Mock Date.now
    let currentTime = 1000000;
    Date.now = () => currentTime;

    // Mock document
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

    // Mock window
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

    // Mock PublicKeyCredential
    global.PublicKeyCredential = {} as any;

    t.after(() => {
        global.window = originalWindow;
        global.document = originalDocument;
        Date.now = originalDateNow;
        global.PublicKeyCredential = originalPublicKeyCredential;
    });

    // Mock DB
    const mockDb = {
        getConfig: async () => null,
        saveConfig: async () => {}
    };

    await t.test('Registers event listeners on initialization', () => {
        addedEventListeners.length = 0; // Clear array

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

        const service = new AuthService(mockDb, { autoLockTimeout: 300000 }); // 5 mins

        // Mocking authentication success to allow activity updating
        service.recordSuccess();

        assert.strictEqual(activeTimeouts.size, 1, 'Should set initial inactivity timer after login');

        const initialTimeoutIds = Array.from(activeTimeouts.keys());

        // Find a standard event handler (e.g., mousedown)
        const mouseEvent = addedEventListeners.find(e => e.event === 'mousedown');
        assert.ok(mouseEvent, 'Mouse event listener should be registered');

        currentTime += 10000; // Fast forward 10s
        mouseEvent.handler(); // Trigger activity

        assert.strictEqual(activeTimeouts.size, 1, 'Should have exactly one active timeout');
        const newTimeoutIds = Array.from(activeTimeouts.keys());

        assert.notStrictEqual(newTimeoutIds[0], initialTimeoutIds[0], 'Should have cleared old timeout and set a new one');

        service.cleanup();
    });

    await t.test('Activity event does not set timer when not authenticated', () => {
        addedEventListeners.length = 0;
        activeTimeouts.clear();

        const service = new AuthService(mockDb);
        // Not authenticated

        assert.strictEqual(activeTimeouts.size, 0, 'No timeout should be set initially because not authenticated');

        const mouseEvent = addedEventListeners.find(e => e.event === 'mousedown');
        mouseEvent?.handler(); // Trigger activity

        assert.strictEqual(activeTimeouts.size, 0, 'No timeout should be set after activity if not authenticated');

        service.cleanup();
    });

    await t.test('Auto-lock timer successfully locks the app', () => {
        addedEventListeners.length = 0;
        activeTimeouts.clear();

        const service = new AuthService(mockDb, { autoLockTimeout: 300000 });

        service.recordSuccess(); // Authenticate
        assert.strictEqual(service.getState().isAuthenticated, true);
        assert.strictEqual(service.getState().isLocked, false);

        assert.strictEqual(activeTimeouts.size, 1);
        const timeoutInfo = Array.from(activeTimeouts.values())[0];

        // Simulate timer firing
        timeoutInfo.callback();

        assert.strictEqual(service.getState().isAuthenticated, false);
        assert.strictEqual(service.getState().isLocked, true);

        service.cleanup();
    });

    await t.test('visibilitychange triggers activity update only when document is visible', () => {
        addedEventListeners.length = 0;
        activeTimeouts.clear();

        const service = new AuthService(mockDb);
        service.recordSuccess(); // Authenticate

        const initialTimeoutIds = Array.from(activeTimeouts.keys());

        const visibilityEvent = addedEventListeners.find(e => e.event === 'visibilitychange');
        assert.ok(visibilityEvent, 'visibilitychange event listener should be registered');

        // Simulate document hidden
        mockDocument.hidden = true;
        visibilityEvent.handler();

        // Timer shouldn't change
        let newTimeoutIds = Array.from(activeTimeouts.keys());
        assert.deepStrictEqual(newTimeoutIds, initialTimeoutIds, 'Timer should not be reset when document is hidden');

        // Simulate document visible
        mockDocument.hidden = false;
        visibilityEvent.handler();

        newTimeoutIds = Array.from(activeTimeouts.keys());
        assert.strictEqual(activeTimeouts.size, 1);
        assert.notStrictEqual(newTimeoutIds[0], initialTimeoutIds[0], 'Timer should be reset when document becomes visible');

        service.cleanup();
    });
});
