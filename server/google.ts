import './loadEnv.js';
import axios from 'axios';
import { OAuth2Client, Credentials } from 'google-auth-library';
import { getGoogleCredentials, loadOAuthConfig } from './oauthConfig.js';

export const GOOGLE_SCOPES = [
  'openid',
  'https://www.googleapis.com/auth/userinfo.email',
  'https://www.googleapis.com/auth/userinfo.profile',
  'https://www.googleapis.com/auth/calendar.readonly',
  'https://www.googleapis.com/auth/gmail.readonly',
  'https://www.googleapis.com/auth/drive.readonly',
];

export interface GoogleServiceStatus {
  email?: string;
  calendar: { ok: boolean; error?: string; eventCount?: number };
  gmail: { ok: boolean; error?: string; messageCount?: number };
  drive: { ok: boolean; error?: string; fileCount?: number };
}

async function getClient(): Promise<OAuth2Client> {
  const config = await loadOAuthConfig();
  const creds = getGoogleCredentials(config);
  if (!creds) {
    throw new Error('Google is not configured. Open Settings → Calendar Integrations and add your Google Client ID and Secret.');
  }
  return new OAuth2Client(creds.clientId, creds.clientSecret);
}

export const GoogleBackend = {
  async isConfigured(): Promise<boolean> {
    const config = await loadOAuthConfig();
    return Boolean(getGoogleCredentials(config));
  },

  async getAuthUrl(redirectUri: string): Promise<string> {
    const client = await getClient();
    return client.generateAuthUrl({
      access_type: 'offline',
      prompt: 'consent',
      include_granted_scopes: true,
      scope: GOOGLE_SCOPES,
      redirect_uri: redirectUri,
    });
  },

  async exchangeCode(code: string, redirectUri: string): Promise<Credentials> {
    const client = await getClient();
    const { tokens } = await client.getToken({ code, redirect_uri: redirectUri });
    return tokens;
  },

  async getValidAccessToken(tokens: Credentials): Promise<{ accessToken: string; tokens: Credentials }> {
    const client = await getClient();
    client.setCredentials(tokens);

    const expiresAt = tokens.expiry_date ?? 0;
    const needsRefresh = !tokens.access_token || (expiresAt > 0 && expiresAt <= Date.now() + 60_000);

    if (needsRefresh) {
      if (!tokens.refresh_token) {
        throw new Error('Google session expired. Disconnect and reconnect Google to restore access.');
      }
      const { credentials } = await client.refreshAccessToken();
      client.setCredentials(credentials);
      return {
        accessToken: credentials.access_token!,
        tokens: credentials,
      };
    }

    return { accessToken: tokens.access_token!, tokens };
  },

  async getUserEmail(accessToken: string): Promise<string | undefined> {
    const response = await axios.get('https://www.googleapis.com/oauth2/v2/userinfo', {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    return response.data.email;
  },

  async getCalendarEvents(accessToken: string) {
    const response = await axios.get('https://www.googleapis.com/calendar/v3/calendars/primary/events', {
      headers: { Authorization: `Bearer ${accessToken}` },
      params: {
        singleEvents: true,
        orderBy: 'startTime',
        timeMin: new Date().toISOString(),
        maxResults: 50,
      },
    });
    return response.data.items ?? [];
  },

  async getGmailProfile(accessToken: string) {
    const response = await axios.get('https://gmail.googleapis.com/gmail/v1/users/me/profile', {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    return response.data;
  },

  async getDriveFiles(accessToken: string) {
    const response = await axios.get('https://www.googleapis.com/drive/v3/files', {
      headers: { Authorization: `Bearer ${accessToken}` },
      params: {
        pageSize: 10,
        fields: 'files(id,name,modifiedTime)',
        orderBy: 'modifiedTime desc',
      },
    });
    return response.data.files ?? [];
  },

  formatApiError(error: unknown, service: string): string {
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      const message = error.response?.data?.error?.message;
      if (status === 403) {
        return `${service} access denied. Enable the ${service} API in Google Cloud Console and add your account as a test user.`;
      }
      if (status === 401) {
        return `${service} session expired. Reconnect Google.`;
      }
      return message || `${service} request failed (${status ?? 'unknown'}).`;
    }
    if (error instanceof Error) return error.message;
    return `${service} request failed.`;
  },

  async verifyServices(tokens: Credentials): Promise<{ tokens: Credentials; status: GoogleServiceStatus }> {
    const { accessToken, tokens: validTokens } = await this.getValidAccessToken(tokens);
    const status: GoogleServiceStatus = {
      calendar: { ok: false },
      gmail: { ok: false },
      drive: { ok: false },
    };

    try {
      status.email = await this.getUserEmail(accessToken);
    } catch (error) {
      console.error('Google userinfo failed', error);
    }

    try {
      const events = await this.getCalendarEvents(accessToken);
      status.calendar = { ok: true, eventCount: events.length };
    } catch (error) {
      status.calendar = { ok: false, error: this.formatApiError(error, 'Google Calendar') };
    }

    try {
      const profile = await this.getGmailProfile(accessToken);
      status.gmail = { ok: true, messageCount: profile.messagesTotal };
    } catch (error) {
      status.gmail = { ok: false, error: this.formatApiError(error, 'Gmail') };
    }

    try {
      const files = await this.getDriveFiles(accessToken);
      status.drive = { ok: true, fileCount: files.length };
    } catch (error) {
      status.drive = { ok: false, error: this.formatApiError(error, 'Google Drive') };
    }

    return { tokens: validTokens, status };
  },

  async fetchAllData(tokens: Credentials) {
    const { accessToken, tokens: validTokens } = await this.getValidAccessToken(tokens);
    const { status } = await this.verifyServices(validTokens);

    const events = status.calendar.ok
      ? (await this.getCalendarEvents(accessToken)).map((e: {
          id: string;
          summary?: string;
          start: { dateTime?: string; date?: string };
          end: { dateTime?: string; date?: string };
        }) => ({
          id: e.id,
          title: e.summary || 'Untitled event',
          start: e.start?.dateTime || e.start?.date,
          end: e.end?.dateTime || e.end?.date,
          provider: 'google',
          kind: 'calendar',
        }))
      : [];

    const driveItems = status.drive.ok
      ? (await this.getDriveFiles(accessToken)).map((f: { id: string; name?: string; modifiedTime?: string }) => ({
          id: `drive-${f.id}`,
          title: f.name || 'Untitled file',
          start: f.modifiedTime || new Date().toISOString(),
          end: f.modifiedTime || new Date().toISOString(),
          provider: 'google',
          kind: 'drive',
        }))
      : [];

    const gmailItems: Array<{
      id: string;
      title: string;
      start: string;
      end: string;
      provider: string;
      kind: string;
    }> = [];

    if (status.gmail.ok) {
      gmailItems.push({
        id: 'gmail-profile',
        title: `Gmail connected${status.email ? ` (${status.email})` : ''} — ${status.gmail.messageCount ?? 0} messages`,
        start: new Date().toISOString(),
        end: new Date().toISOString(),
        provider: 'google',
        kind: 'gmail',
      });
    }

    return {
      tokens: validTokens,
      status,
      items: [...events, ...gmailItems, ...driveItems],
    };
  },
};
