import test from 'node:test';
import assert from 'node:assert';
import { NotificationService } from './NotificationService.ts';

test('NotificationService.requestPermission', async (t) => {
    t.afterEach(() => {
        // Reset singleton instance between tests
        (NotificationService as any).instance = null;
    });

    await t.test('returns denied when window is undefined', async () => {
        const originalWindow = (global as any).window;
        const originalConsoleWarn = console.warn;

        delete (global as any).window;

        let warnCalled = false;
        console.warn = () => { warnCalled = true; };

        try {
            const service = NotificationService.getInstance();
            const result = await service.requestPermission();

            assert.strictEqual(result, 'denied');
            assert.strictEqual(warnCalled, true);
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            console.warn = originalConsoleWarn;
        }
    });

    await t.test('returns denied when Notification is not in window', async () => {
        const originalWindow = (global as any).window;
        const originalConsoleWarn = console.warn;

        (global as any).window = {};

        let warnCalled = false;
        console.warn = () => { warnCalled = true; };

        try {
            const service = NotificationService.getInstance();
            const result = await service.requestPermission();

            assert.strictEqual(result, 'denied');
            assert.strictEqual(warnCalled, true);
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            console.warn = originalConsoleWarn;
        }
    });

    await t.test('returns granted when already granted', async () => {
        const originalWindow = (global as any).window;
        const originalNotification = (global as any).Notification;

        const mockNotification = {
            permission: 'granted',
            requestPermission: async () => 'denied'
        };
        (global as any).window = { Notification: mockNotification };
        (global as any).Notification = mockNotification;

        try {
            const service = NotificationService.getInstance();
            const result = await service.requestPermission();

            assert.strictEqual(result, 'granted');
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            if (originalNotification === undefined) {
                delete (global as any).Notification;
            } else {
                (global as any).Notification = originalNotification;
            }
        }
    });

    await t.test('returns denied when already denied', async () => {
        const originalWindow = (global as any).window;
        const originalNotification = (global as any).Notification;

        const mockNotification = {
            permission: 'denied',
            requestPermission: async () => 'granted'
        };
        (global as any).window = { Notification: mockNotification };
        (global as any).Notification = mockNotification;

        try {
            const service = NotificationService.getInstance();
            const result = await service.requestPermission();

            assert.strictEqual(result, 'denied');
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            if (originalNotification === undefined) {
                delete (global as any).Notification;
            } else {
                (global as any).Notification = originalNotification;
            }
        }
    });

    await t.test('requests permission and updates internal state', async () => {
        const originalWindow = (global as any).window;
        const originalNotification = (global as any).Notification;

        const mockNotification = {
            permission: 'default',
            requestPermission: async () => 'granted'
        };
        (global as any).window = { Notification: mockNotification };
        (global as any).Notification = mockNotification;

        try {
            const service = NotificationService.getInstance();
            const result = await service.requestPermission();

            assert.strictEqual(result, 'granted');
            assert.strictEqual(service.isAllowed(), true);
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            if (originalNotification === undefined) {
                delete (global as any).Notification;
            } else {
                (global as any).Notification = originalNotification;
            }
        }
    });
});

test('NotificationService.showNotification', async (t) => {
    t.afterEach(() => {
        (NotificationService as any).instance = null;
    });

    await t.test('warns and returns if permission is denied', async () => {
        const originalWindow = (global as any).window;
        const originalNotification = (global as any).Notification;
        const originalConsoleWarn = console.warn;

        const mockNotification = {
            permission: 'denied',
            requestPermission: async () => 'denied'
        };
        (global as any).window = { Notification: mockNotification };
        (global as any).Notification = mockNotification;

        let warnCalled = false;
        console.warn = () => { warnCalled = true; };

        try {
            const service = NotificationService.getInstance();
            await service.showNotification('Test');

            assert.strictEqual(warnCalled, true);
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            if (originalNotification === undefined) {
                delete (global as any).Notification;
            } else {
                (global as any).Notification = originalNotification;
            }
            console.warn = originalConsoleWarn;
        }
    });

    await t.test('shows notification when allowed', async () => {
        const originalWindow = (global as any).window;
        const originalNotification = (global as any).Notification;

        let notificationOptions = null;

        class MockNotification {
            static permission = 'granted';
            constructor(title: string, options: Record<string, unknown>) {
                notificationOptions = { title, ...options };
            }
            close() {}
        }

        (global as any).window = { Notification: MockNotification };
        (global as any).Notification = MockNotification;

        try {
            const service = NotificationService.getInstance();
            await service.showNotification('Test Title', { body: 'Test Body' });

            assert.ok(notificationOptions);
            assert.strictEqual(notificationOptions.title, 'Test Title');
            assert.strictEqual(notificationOptions.body, 'Test Body');
            assert.strictEqual(notificationOptions.icon, '/icon-192.png');
            assert.strictEqual(notificationOptions.tag, 'meeting-reminder');
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            if (originalNotification === undefined) {
                delete (global as any).Notification;
            } else {
                (global as any).Notification = originalNotification;
            }
        }
    });

    await t.test('Notification auto-closes after 10 seconds', async () => {
        const originalWindow = (global as any).window;
        const originalNotification = (global as any).Notification;
        const originalSetTimeout = global.setTimeout;

        let closed = false;
        let timeoutCallback: any = null;
        let timeoutDuration = 0;

        global.setTimeout = ((callback: any, ms: number) => {
            timeoutCallback = callback;
            timeoutDuration = ms;
            return 1 as any;
        }) as any;

        class MockNotification {
            static permission = 'granted';
            constructor(_title: string, _options: Record<string, unknown>) {
            }
            close() {
                closed = true;
            }
        }

        (global as any).window = { Notification: MockNotification };
        (global as any).Notification = MockNotification;

        try {
            const service = NotificationService.getInstance();
            await service.showNotification('Test Title');

            assert.ok(timeoutCallback);
            assert.strictEqual(timeoutDuration, 10000);

            // Execute the timeout callback
            timeoutCallback();
            assert.strictEqual(closed, true);
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            if (originalNotification === undefined) {
                delete (global as any).Notification;
            } else {
                (global as any).Notification = originalNotification;
            }
            global.setTimeout = originalSetTimeout;
        }
    });
});

test('NotificationService.showMeetingReminder', async (t) => {
    t.afterEach(() => {
        (NotificationService as any).instance = null;
    });

    await t.test('formats message correctly when starting now', async () => {
        const originalWindow = (global as any).window;
        const originalNotification = (global as any).Notification;

        let notificationOptions: any = null;

        class MockNotification {
            static permission = 'granted';
            constructor(title: string, options: any) {
                notificationOptions = { title, ...options };
            }
            close() {}
        }

        (global as any).window = { Notification: MockNotification };
        (global as any).Notification = MockNotification;

        try {
            const service = NotificationService.getInstance();
            const date = new Date('2024-01-01T12:00:00.000Z');
            await service.showMeetingReminder('Test Meeting', date, 0);

            const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            assert.ok(notificationOptions);
            assert.strictEqual(notificationOptions.title, 'Test Meeting');
            assert.strictEqual(notificationOptions.body, `Starting now at ${timeStr}`);
            assert.strictEqual(notificationOptions.requireInteraction, true);
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            if (originalNotification === undefined) {
                delete (global as any).Notification;
            } else {
                (global as any).Notification = originalNotification;
            }
        }
    });

    await t.test('formats message correctly when starting in 5 minutes', async () => {
        const originalWindow = (global as any).window;
        const originalNotification = (global as any).Notification;

        let notificationOptions: any = null;

        class MockNotification {
            static permission = 'granted';
            constructor(title: string, options: any) {
                notificationOptions = { title, ...options };
            }
            close() {}
        }

        (global as any).window = { Notification: MockNotification };
        (global as any).Notification = MockNotification;

        try {
            const service = NotificationService.getInstance();
            const date = new Date('2024-01-01T12:00:00.000Z');
            await service.showMeetingReminder('Test Meeting', date, 5);

            const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            assert.ok(notificationOptions);
            assert.strictEqual(notificationOptions.title, 'Test Meeting');
            assert.strictEqual(notificationOptions.body, `Starting in 5 minutes at ${timeStr}`);
            assert.strictEqual(notificationOptions.requireInteraction, false);
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            if (originalNotification === undefined) {
                delete (global as any).Notification;
            } else {
                (global as any).Notification = originalNotification;
            }
        }
    });

    await t.test('formats message correctly when starting in 1 minute', async () => {
        const originalWindow = (global as any).window;
        const originalNotification = (global as any).Notification;

        let notificationOptions: any = null;

        class MockNotification {
            static permission = 'granted';
            constructor(title: string, options: any) {
                notificationOptions = { title, ...options };
            }
            close() {}
        }

        (global as any).window = { Notification: MockNotification };
        (global as any).Notification = MockNotification;

        try {
            const service = NotificationService.getInstance();
            const date = new Date('2024-01-01T12:00:00.000Z');
            await service.showMeetingReminder('Test Meeting', date, 1);

            const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            assert.ok(notificationOptions);
            assert.strictEqual(notificationOptions.title, 'Test Meeting');
            assert.strictEqual(notificationOptions.body, `Starting in 1 minute at ${timeStr}`);
            assert.strictEqual(notificationOptions.requireInteraction, false);
        } finally {
            if (originalWindow === undefined) {
                delete (global as any).window;
            } else {
                (global as any).window = originalWindow;
            }
            if (originalNotification === undefined) {
                delete (global as any).Notification;
            } else {
                (global as any).Notification = originalNotification;
            }
        }
    });
});
