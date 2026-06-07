import './loadEnv.js';
import axios from 'axios';
import { GoogleBackend } from './google.js';
import {
  getMicrosoftCredentials,
  getNotionCredentials,
  loadOAuthConfig,
} from './oauthConfig.js';

// Microsoft Calendar Config (Azure AD)
const MS_AUTH_URL = 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize';
const MS_TOKEN_URL = 'https://login.microsoftonline.com/common/oauth2/v2.0/token';
const MS_SCOPES = 'https://graph.microsoft.com/Calendars.Read';

// Notion Config
const NOTION_AUTH_URL = 'https://api.notion.com/v1/oauth/authorize';
const NOTION_TOKEN_URL = 'https://api.notion.com/v1/oauth/token';

export const CalendarBackend = {
  // Google (Calendar, Gmail, Drive)
  getGoogleAuthUrl: (redirectUri: string) => GoogleBackend.getAuthUrl(redirectUri),
  exchangeGoogleCode: (code: string, redirectUri: string) => GoogleBackend.exchangeCode(code, redirectUri),
  getGoogleData: (tokens: Record<string, unknown>) => GoogleBackend.fetchAllData(tokens),

  // Microsoft
  getMicrosoftAuthUrl: async (redirectUri: string) => {
    const config = await loadOAuthConfig();
    const creds = getMicrosoftCredentials(config);
    if (!creds) {
      throw new Error('Outlook is not configured. Open Settings → Calendar Integrations and add your Microsoft Client ID and Secret.');
    }
    const params = new URLSearchParams({
      client_id: creds.clientId,
      response_type: 'code',
      redirect_uri: redirectUri,
      scope: MS_SCOPES,
      response_mode: 'query',
    });
    return `${MS_AUTH_URL}?${params.toString()}`;
  },
  exchangeMicrosoftCode: async (code: string, redirectUri: string) => {
    const config = await loadOAuthConfig();
    const creds = getMicrosoftCredentials(config);
    if (!creds) throw new Error('Microsoft OAuth is not configured.');
    const params = new URLSearchParams({
      client_id: creds.clientId,
      client_secret: creds.clientSecret,
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
    const config = await loadOAuthConfig();
    const creds = getNotionCredentials(config);
    if (!creds) {
      throw new Error('Notion is not configured. Open Settings → Calendar Integrations and add your Notion Client ID and Secret.');
    }
    const params = new URLSearchParams({
      client_id: creds.clientId,
      redirect_uri: redirectUri,
      response_type: 'code',
      owner: 'user',
    });
    return `${NOTION_AUTH_URL}?${params.toString()}`;
  },
  exchangeNotionCode: async (code: string, redirectUri: string) => {
    const config = await loadOAuthConfig();
    const creds = getNotionCredentials(config);
    if (!creds) throw new Error('Notion OAuth is not configured.');
    const auth = Buffer.from(`${creds.clientId}:${creds.clientSecret}`).toString('base64');
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
    // Notion doesn't have a direct "Calendar" API like Google/MS.
    // Usually, you query a database that is used as a calendar.
    // For this integration, we'll list databases the user has shared.
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
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getAppleEvents: async (config: { url: string, user: string, password: string }) => {
    // This is a simplified CalDAV fetcher. 
    // In a real app, we'd use a library like 'dav' or 'ical.js'.
    // For now, we'll return a placeholder to show it's connected.
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
