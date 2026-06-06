import { test, describe, before, after, mock } from 'node:test';
import assert from 'node:assert';

describe('CalendarBackend.exchangeGoogleCode', () => {
  before(() => {
    process.env.GOOGLE_CLIENT_ID = 'test-client-id';
    process.env.GOOGLE_CLIENT_SECRET = 'test-client-secret';
  });

  after(() => {
    delete process.env.GOOGLE_CLIENT_ID;
    delete process.env.GOOGLE_CLIENT_SECRET;
  });

  test('should exchange code for tokens by calling googleClient.getToken', async () => {
    const { OAuth2Client } = await import('google-auth-library');

    // Mock the getToken method on the prototype
    const mockGetToken = mock.method(OAuth2Client.prototype, 'getToken', async (opts) => {
      return { tokens: { access_token: 'fake-token', refresh_token: 'fake-refresh' } };
    });

    const { CalendarBackend } = await import('./calendar.ts');

    const tokens = await CalendarBackend.exchangeGoogleCode('fake-code', 'http://localhost/callback');

    assert.deepStrictEqual(tokens, { access_token: 'fake-token', refresh_token: 'fake-refresh' });

    assert.strictEqual(mockGetToken.mock.calls.length, 1);
    assert.deepStrictEqual(mockGetToken.mock.calls[0].arguments[0], {
      code: 'fake-code',
      redirect_uri: 'http://localhost/callback'
    });

    // Restore the mock
    mockGetToken.mock.restore();
  });
});
