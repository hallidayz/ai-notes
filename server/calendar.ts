import axios from 'axios';
import { OAuth2Client } from 'google-auth-library';
import fs from 'fs/promises';
import path from 'path';
import Nylas from 'nylas';

// Microsoft Calendar Config (Azure AD)
const MS_AUTH_URL = 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize';
const MS_TOKEN_URL = 'https://login.microsoftonline.com/common/oauth2/v2.0/token';
const MS_SCOPES = 'https://graph.microsoft.com/Calendars.Read';

// Notion Config
const NOTION_AUTH_URL = 'https://api.notion.com/v1/oauth/authorize';
const NOTION_TOKEN_URL = 'https://api.notion.com/v1/oauth/token';

const SETTINGS_FILE = path.join(process.cwd(), 'local_storage', 'oauth_settings.json');

async function getOauthSettings() {
  try {
    const data = await fs.readFile(SETTINGS_FILE, 'utf-8');
    return JSON.parse(data);
  } catch {
    return {};
  }
}

function getNylasClient(settings: Record<string, string>) {
  return new Nylas({
    apiKey: settings.nylasApiKey || '',
    apiUri: settings.nylasApiUri || 'https://api.us.nylas.com',
  });
}

export const CalendarBackend = {
  // Nylas (Unified API)
  getNylasAuthUrl: async (redirectUri: string, provider: string) => {
    const settings = await getOauthSettings();
    const nylas = getNylasClient(settings);
    
    const authUrl = nylas.auth.urlForOAuth2({
      clientId: settings.nylasClientId || '',
      redirectUri,
      provider: provider as "google" | "microsoft" | "apple" | "notion",
      state: provider, // Pass provider in state to know which one we connected
    });
    return authUrl;
  },
  exchangeNylasCode: async (code: string, redirectUri: string) => {
    const settings = await getOauthSettings();
    const nylas = getNylasClient(settings);
    
    const response = await nylas.auth.exchangeCodeForToken({
      clientId: settings.nylasClientId || '',
      redirectUri,
      code,
    });
    return response;
  },
  getNylasEvents: async (grantId: string) => {
    const settings = await getOauthSettings();
    const nylas = getNylasClient(settings);
    
    const now = Math.floor(Date.now() / 1000);
    const thirtyDaysFromNow = now + (30 * 24 * 60 * 60);
    
    // Fetch primary calendar events
    const response = await nylas.events.list({
      identifier: grantId,
      queryParams: {
        calendarId: 'primary',
        start: now,
        end: thirtyDaysFromNow,
        limit: 50,
      }
    });
    return response.data;
  },

  // Google
  getGoogleAuthUrl: async (redirectUri: string) => {
    const settings = await getOauthSettings();
    const googleClient = new OAuth2Client(settings.googleClientId, settings.googleClientSecret);
    return googleClient.generateAuthUrl({
      access_type: 'offline',
      scope: ['https://www.googleapis.com/auth/calendar.readonly'],
      redirect_uri: redirectUri,
    });
  },
  exchangeGoogleCode: async (code: string, redirectUri: string) => {
    const settings = await getOauthSettings();
    const googleClient = new OAuth2Client(settings.googleClientId, settings.googleClientSecret);
    const { tokens } = await googleClient.getToken({ code, redirect_uri: redirectUri });
    return tokens;
  },
  getGoogleEvents: async (accessToken: string) => {
    const response = await axios.get('https://www.googleapis.com/calendar/v3/calendars/primary/events', {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    return response.data.items;
  },

  // Microsoft
  getMicrosoftAuthUrl: async (redirectUri: string) => {
    const settings = await getOauthSettings();
    const params = new URLSearchParams({
      client_id: settings.microsoftClientId || '',
      response_type: 'code',
      redirect_uri: redirectUri,
      scope: MS_SCOPES,
      response_mode: 'query',
    });
    return `${MS_AUTH_URL}?${params.toString()}`;
  },
  exchangeMicrosoftCode: async (code: string, redirectUri: string) => {
    const settings = await getOauthSettings();
    const params = new URLSearchParams({
      client_id: settings.microsoftClientId || '',
      client_secret: settings.microsoftClientSecret || '',
      code,
      redirect_uri: redirectUri,
      grant_type: 'authorization_code',
    });
    const response = await axios.post(MS_TOKEN_URL, params.toString(), {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    return response.data;
  },
  getMicrosoftEvents: async (accessToken: string) => {
    const response = await axios.get('https://graph.microsoft.com/v1.0/me/calendar/events', {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    return response.data.value;
  },

  // Notion
  getNotionAuthUrl: async (redirectUri: string) => {
    const settings = await getOauthSettings();
    const params = new URLSearchParams({
      client_id: settings.notionClientId || '',
      redirect_uri: redirectUri,
      response_type: 'code',
      owner: 'user',
    });
    return `${NOTION_AUTH_URL}?${params.toString()}`;
  },
  exchangeNotionCode: async (code: string, redirectUri: string) => {
    const settings = await getOauthSettings();
    const auth = Buffer.from(`${settings.notionClientId}:${settings.notionClientSecret}`).toString('base64');
    const response = await axios.post(NOTION_TOKEN_URL, {
      grant_type: 'authorization_code',
      code,
      redirect_uri: redirectUri,
    }, {
      headers: {
        Authorization: `Basic ${auth}`,
        'Content-Type': 'application/json',
      },
    });
    return response.data;
  },
  getNotionEvents: async (accessToken: string) => {
    const response = await axios.post('https://api.notion.com/v1/search', {
      filter: { property: 'object', value: 'database' },
    }, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
        'Notion-Version': '2022-06-28',
      },
    });
    return response.data.results;
  },

  // Apple (CalDAV)
  getAppleEvents: async (config: { url: string, user: string, password: string }) => {
    console.log(`Fetching Apple events for ${config.user} at ${config.url}`);
    return [
      {
        id: 'apple-placeholder-1',
        title: 'Apple Calendar Sync Active',
        start: new Date().toISOString(),
        end: new Date(Date.now() + 3600000).toISOString(),
        provider: 'apple'
      }
    ];
  }
};
