import { test, describe, before } from 'node:test';
import * as assert from 'node:assert';

describe('CalendarBackend', () => {
  let CalendarBackend: any;

  before(async () => {
    // We must set the environment variables before importing calendar.ts
    // to avoid runtime errors when OAuth2Client is initialized.
    process.env.GOOGLE_CLIENT_ID = 'test-google-client-id';
    process.env.GOOGLE_CLIENT_SECRET = 'test-google-client-secret';

    const module = await import('./calendar.ts');
    CalendarBackend = module.CalendarBackend;
  });

  describe('getGoogleAuthUrl', () => {
    test('should generate a valid Google Auth URL with correct parameters', () => {
      const redirectUri = 'http://localhost:3000/callback';
      const urlString = CalendarBackend.getGoogleAuthUrl(redirectUri);

      const url = new URL(urlString);

      assert.strictEqual(url.origin, 'https://accounts.google.com');
      assert.strictEqual(url.pathname, '/o/oauth2/v2/auth');

      const searchParams = url.searchParams;
      assert.strictEqual(searchParams.get('access_type'), 'offline');
      assert.strictEqual(searchParams.get('scope'), 'https://www.googleapis.com/auth/calendar.readonly');
      assert.strictEqual(searchParams.get('redirect_uri'), redirectUri);
      assert.strictEqual(searchParams.get('response_type'), 'code');
      assert.strictEqual(searchParams.get('client_id'), 'test-google-client-id');
    });
  });
});
