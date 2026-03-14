import { test, mock } from 'node:test';
import assert from 'node:assert';
import { GoogleCalendarService } from './GoogleCalendarService.ts';

test('GoogleCalendarService - fetchUpcomingMeetings edge cases', async (t) => {
    // Basic setup mock
    global.window = { location: { origin: 'http://localhost' } } as any;
    global.localStorage = {
        getItem: mock.fn(() => null),
        setItem: mock.fn(),
        removeItem: mock.fn()
    } as any;

    // We'll restore fetch later
    const originalFetch = global.fetch;

    t.after(() => {
        global.fetch = originalFetch;
        delete (global as any).window;
        delete (global as any).localStorage;
    });

    await t.test('fetchUpcomingMeetings handles 401 and successfully refreshes token', async () => {
        const service = new GoogleCalendarService('test-client-id');

        // Mock credentials
        (service as any).credentials = {
            accessToken: 'old-token',
            refreshToken: 'refresh-token',
            expiresAt: Date.now() + 10000 // Not expired locally, but API returns 401
        };
        // Mock loadCredentials to return our credentials to bypass isConnected check
        service['loadCredentials'] = async () => (service as any).credentials;

        // Mock fetch to first fail with 401, then succeed with 200 on refresh, then succeed with 200 on second fetch
        let fetchCallCount = 0;
        global.fetch = mock.fn(async (url: string | URL | Request, options?: RequestInit) => {
            fetchCallCount++;
            const urlStr = url.toString();

            // First call to fetch events -> returns 401
            if (fetchCallCount === 1 && urlStr.includes('events')) {
                return {
                    ok: false,
                    status: 401,
                    statusText: 'Unauthorized'
                } as any;
            }

            // Second call to refresh token -> returns 200
            if (fetchCallCount === 2 && urlStr.includes('refresh')) {
                return {
                    ok: true,
                    json: async () => ({
                        access_token: 'new-token',
                        expires_in: 3600
                    })
                } as any;
            }

            // Third call to fetch events again -> returns 200
            if (fetchCallCount === 3 && urlStr.includes('events')) {
                assert.strictEqual(options?.headers?.Authorization, 'Bearer new-token', 'Should use the new token');
                return {
                    ok: true,
                    json: async () => ({
                        items: []
                    })
                } as any;
            }

            throw new Error(`Unexpected fetch call: ${urlStr}`);
        });

        const meetings = await service.fetchUpcomingMeetings(7);
        assert.deepStrictEqual(meetings, []);
        assert.strictEqual(fetchCallCount, 3);
    });

    await t.test('fetchUpcomingMeetings throws original error if token refresh fails', async () => {
        const service = new GoogleCalendarService('test-client-id');

        // Mock credentials
        (service as any).credentials = {
            accessToken: 'old-token',
            refreshToken: 'refresh-token',
            expiresAt: Date.now() + 10000
        };
        service['loadCredentials'] = async () => (service as any).credentials;

        // Mock fetch
        let fetchCallCount = 0;
        global.fetch = mock.fn(async (url: string | URL | Request) => {
            fetchCallCount++;
            const urlStr = url.toString();

            if (fetchCallCount === 1 && urlStr.includes('events')) {
                return {
                    ok: false,
                    status: 401,
                    statusText: 'Unauthorized'
                } as any;
            }

            if (fetchCallCount === 2 && urlStr.includes('refresh')) {
                return {
                    ok: false,
                    status: 400,
                    statusText: 'Bad Request'
                } as any;
            }

            throw new Error(`Unexpected fetch call: ${urlStr}`);
        });

        await assert.rejects(
            service.fetchUpcomingMeetings(7),
            /Failed to refresh token/
        );
        assert.strictEqual(fetchCallCount, 2);
    });

    await t.test('fetchUpcomingMeetings handles other errors', async () => {
        const service = new GoogleCalendarService('test-client-id');

        // Mock credentials
        (service as any).credentials = {
            accessToken: 'valid-token',
            refreshToken: 'refresh-token',
            expiresAt: Date.now() + 10000
        };
        service['loadCredentials'] = async () => (service as any).credentials;

        global.fetch = mock.fn(async () => {
            return {
                ok: false,
                status: 500,
                statusText: 'Internal Server Error'
            } as any;
        });

        await assert.rejects(
            service.fetchUpcomingMeetings(7),
            /Failed to fetch meetings: Internal Server Error/
        );
    });
});
