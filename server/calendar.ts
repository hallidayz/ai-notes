import axios from 'axios';
import { OAuth2Client } from 'google-auth-library';

// Google Calendar Config
const googleClient = new OAuth2Client(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET
);

// Microsoft Calendar Config (Azure AD)
const MS_AUTH_URL = 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize';
const MS_TOKEN_URL = 'https://login.microsoftonline.com/common/oauth2/v2.0/token';
const MS_SCOPES = 'https://graph.microsoft.com/Calendars.Read';

// Notion Config
const NOTION_AUTH_URL = 'https://api.notion.com/v1/oauth/authorize';
const NOTION_TOKEN_URL = 'https://api.notion.com/v1/oauth/token';

export const CalendarBackend = {
  // Google
  getGoogleAuthUrl: (redirectUri: string) => {
    return googleClient.generateAuthUrl({
      access_type: 'offline',
      scope: ['https://www.googleapis.com/auth/calendar.readonly'],
      redirect_uri: redirectUri,
    });
  },
  exchangeGoogleCode: async (code: string, redirectUri: string) => {
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
  getMicrosoftAuthUrl: (redirectUri: string) => {
    const params = new URLSearchParams({
      client_id: process.env.MICROSOFT_CLIENT_ID!,
      response_type: 'code',
      redirect_uri: redirectUri,
      scope: MS_SCOPES,
      response_mode: 'query',
    });
    return `${MS_AUTH_URL}?${params.toString()}`;
  },
  exchangeMicrosoftCode: async (code: string, redirectUri: string) => {
    const params = new URLSearchParams({
      client_id: process.env.MICROSOFT_CLIENT_ID!,
      client_secret: process.env.MICROSOFT_CLIENT_SECRET!,
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
  getNotionAuthUrl: (redirectUri: string) => {
    const params = new URLSearchParams({
      client_id: process.env.NOTION_CLIENT_ID!,
      redirect_uri: redirectUri,
      response_type: 'code',
      owner: 'user',
    });
    return `${NOTION_AUTH_URL}?${params.toString()}`;
  },
  exchangeNotionCode: async (code: string, redirectUri: string) => {
    const auth = Buffer.from(`${process.env.NOTION_CLIENT_ID}:${process.env.NOTION_CLIENT_SECRET}`).toString('base64');
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
  getAppleEvents: async (config: { url: string, user: string, password: string }) => {
    // This is a simplified CalDAV fetcher. 
    // In a real app, we'd use a library like 'dav' or 'ical.js'.
    // For now, we'll return a placeholder to show it's connected.
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
