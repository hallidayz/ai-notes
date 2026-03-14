import { test, describe, mock, afterEach } from 'node:test';
import assert from 'node:assert';
import { OutlookCalendarService } from './OutlookCalendarService.ts';

describe('OutlookCalendarService refreshTokenIfNeeded', () => {
    afterEach(() => {
        mock.restoreAll();
    });

    test('should throw error and disconnect when token refresh fails', async (t) => {
        // Mock global fetch to return an error response
        const mockFetch = mock.fn(async () => {
            return {
                ok: false,
                status: 400
            };
        });

        // Use Object.defineProperty to mock global fetch
        const originalFetch = global.fetch;
        Object.defineProperty(global, 'fetch', {
            value: mockFetch,
            configurable: true,
            writable: true
        });

        // Mock console.error to avoid noise in test output
        const originalConsoleError = console.error;
        let loggedError;
        Object.defineProperty(console, 'error', {
            value: (msg: string, err: any) => {
                loggedError = err;
            },
            configurable: true,
            writable: true
        });

        const service = new OutlookCalendarService('test-client-id');

        // Inject mock credentials with refresh token
        (service as any).credentials = {
            refreshToken: 'old-refresh-token',
            accessToken: 'old-access-token',
            expiresAt: Date.now() - 1000,
            provider: 'outlook'
        };

        // Spy on disconnect
        const disconnectSpy = mock.method(service, 'disconnect', async () => {});

        try {
            await service.refreshTokenIfNeeded();
            assert.fail('Should have thrown an error');
        } catch (error: any) {
            assert.strictEqual(error.message, 'Failed to refresh token');

            // Verify disconnect was called
            assert.strictEqual(disconnectSpy.mock.callCount(), 1);

            // Verify console.error was called with the error
            assert.strictEqual(loggedError, error);
        } finally {
            // Restore globals
            Object.defineProperty(global, 'fetch', {
                value: originalFetch,
                configurable: true,
                writable: true
            });
            Object.defineProperty(console, 'error', {
                value: originalConsoleError,
                configurable: true,
                writable: true
            });
        }
    });

    test('should throw error when no refresh token is available', async () => {
        const service = new OutlookCalendarService('test-client-id');

        // No refresh token in credentials
        (service as any).credentials = {
            accessToken: 'old-access-token',
            expiresAt: Date.now() - 1000,
            provider: 'outlook'
        };

        try {
            await service.refreshTokenIfNeeded();
            assert.fail('Should have thrown an error');
        } catch (error: any) {
            assert.strictEqual(error.message, 'No refresh token available');
        }
    });

    test('should successfully refresh token and save credentials', async () => {
        // Mock localStorage
        const originalLocalStorage = global.localStorage;
        Object.defineProperty(global, 'localStorage', {
            value: {
                getItem: mock.fn(),
                setItem: mock.fn(),
                removeItem: mock.fn(),
                clear: mock.fn(),
                length: 0,
                key: mock.fn()
            },
            configurable: true,
            writable: true
        });

        const service = new OutlookCalendarService('test-client-id');

        const originalFetch = global.fetch;
        const mockResponse = {
            ok: true,
            json: async () => ({
                access_token: 'new-access-token',
                refresh_token: 'new-refresh-token',
                expires_in: 3600
            })
        };
        const mockFetch = mock.fn(async () => mockResponse);

        Object.defineProperty(global, 'fetch', {
            value: mockFetch,
            configurable: true,
            writable: true
        });

        // Set initial credentials
        (service as any).credentials = {
            refreshToken: 'old-refresh-token',
            accessToken: 'old-access-token',
            expiresAt: Date.now() - 1000,
            provider: 'outlook'
        };

        const saveCredentialsSpy = mock.method(service, 'saveCredentials', async () => {});

        try {
            await service.refreshTokenIfNeeded();

            // Check if tokens were updated correctly
            assert.strictEqual((service as any).credentials.accessToken, 'new-access-token');
            assert.strictEqual((service as any).credentials.refreshToken, 'new-refresh-token');
            assert.ok((service as any).credentials.expiresAt > Date.now());

            // Verify saveCredentials was called with the updated credentials
            assert.strictEqual(saveCredentialsSpy.mock.callCount(), 1);
            const callArgs = saveCredentialsSpy.mock.calls[0].arguments;
            assert.strictEqual(callArgs[0].accessToken, 'new-access-token');
            assert.strictEqual(callArgs[0].refreshToken, 'new-refresh-token');
        } finally {
            Object.defineProperty(global, 'fetch', {
                value: originalFetch,
                configurable: true,
                writable: true
            });
            Object.defineProperty(global, 'localStorage', {
                value: originalLocalStorage,
                configurable: true,
                writable: true
            });
        }
    });
});
