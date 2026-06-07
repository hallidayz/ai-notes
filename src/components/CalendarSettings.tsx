import React, { useState, useEffect, useCallback, useRef } from 'react';
import { StorageProvider, CalendarConnection } from '../types';
import { CryptoService } from '../services/cryptoService';
import { AppIcon } from './AppIcon';
import type { IconName } from '../icons';
import { parseIcsEventsFromFile } from '../utils/icsParser';

interface OAuthProviderStatus {
  configured: boolean;
  clientId: string | null;
  source: 'env' | 'settings' | null;
}

interface OAuthStatus {
  google: OAuthProviderStatus;
  microsoft: OAuthProviderStatus;
  notion: OAuthProviderStatus;
}

interface GoogleServiceStatus {
  email?: string;
  calendar: { ok: boolean; error?: string; eventCount?: number };
  gmail: { ok: boolean; error?: string; messageCount?: number };
  drive: { ok: boolean; error?: string; fileCount?: number };
}

interface CalendarSettingsProps {
  pin: string;
  storageProvider: StorageProvider;
  showStatus: (msg: string, type: 'success' | 'error' | 'info', duration?: number) => void;
  isDarkMode: boolean;
  onConnectionsChange?: () => void;
}

export const CalendarSettings: React.FC<CalendarSettingsProps> = ({
  pin,
  storageProvider,
  showStatus,
  isDarkMode,
  onConnectionsChange,
}) => {
  const [oauthStatus, setOauthStatus] = useState<OAuthStatus | null>(null);
  const [connections, setConnections] = useState<CalendarConnection[]>([]);
  const [googleCreds, setGoogleCreds] = useState({ clientId: '', clientSecret: '' });
  const [microsoftCreds, setMicrosoftCreds] = useState({ clientId: '', clientSecret: '' });
  const [notionCreds, setNotionCreds] = useState({ clientId: '', clientSecret: '' });
  const [isSavingCreds, setIsSavingCreds] = useState(false);
  const [showSetupGuide, setShowSetupGuide] = useState(false);
  const [appleConfig, setAppleConfig] = useState({ url: '', user: '', password: '' });
  const [isConnectingApple, setIsConnectingApple] = useState(false);
  const [confirmDisconnect, setConfirmDisconnect] = useState<number | null>(null);
  const [googleStatus, setGoogleStatus] = useState<GoogleServiceStatus | null>(null);
  const localFileInputRef = useRef<HTMLInputElement>(null);

  const loadStatus = useCallback(async () => {
    try {
      const [statusRes, conns] = await Promise.all([
        fetch('/api/config/oauth'),
        storageProvider.getAllCalendarConnections(),
      ]);
      if (statusRes.ok) {
        setOauthStatus(await statusRes.json());
      }
      setConnections(conns);
    } catch (err) {
      console.error('Failed to load calendar settings', err);
    }
  }, [storageProvider]);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  useEffect(() => {
    const handleMessage = async (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return;

      if (event.data?.type === 'OAUTH_AUTH_SUCCESS') {
        const { provider, tokens } = event.data;
        try {
          const googleAccountStatus = tokens.googleStatus as GoogleServiceStatus | undefined;
          const encryptedTokens = await CryptoService.encrypt(JSON.stringify(tokens), pin);
          await storageProvider.saveCalendarConnection({
            provider,
            encryptedTokens,
            accountName: googleAccountStatus?.email,
            timestamp: Date.now(),
          });
          setGoogleStatus(googleAccountStatus ?? null);
          showStatus(`${provider} connected successfully.`, 'success');
          loadStatus();
          onConnectionsChange?.();
        } catch (err) {
          console.error('Failed to save connection', err);
          showStatus('Failed to save calendar connection.', 'error');
        }
      }

      if (event.data?.type === 'OAUTH_AUTH_ERROR') {
        showStatus(event.data.error || 'OAuth connection failed.', 'error', 7000);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [pin, storageProvider, showStatus, loadStatus, onConnectionsChange]);

  const handleSaveCredentials = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSavingCreds(true);
    try {
      const body: Record<string, { clientId: string; clientSecret: string }> = {};
      if (googleCreds.clientId && googleCreds.clientSecret) body.google = googleCreds;
      if (microsoftCreds.clientId && microsoftCreds.clientSecret) body.microsoft = microsoftCreds;
      if (notionCreds.clientId && notionCreds.clientSecret) body.notion = notionCreds;

      if (Object.keys(body).length === 0) {
        showStatus('Enter at least one provider Client ID and Secret.', 'info');
        return;
      }

      const response = await fetch('/api/config/oauth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      if (response.ok) {
        setOauthStatus(data.status);
        setGoogleCreds({ clientId: '', clientSecret: '' });
        setMicrosoftCreds({ clientId: '', clientSecret: '' });
        setNotionCreds({ clientId: '', clientSecret: '' });
        showStatus('OAuth credentials saved. You can now connect calendars.', 'success');
      } else {
        showStatus(data.error || 'Failed to save credentials.', 'error');
      }
    } catch (err) {
      console.error('Failed to save OAuth credentials', err);
      showStatus('Failed to save credentials.', 'error');
    } finally {
      setIsSavingCreds(false);
    }
  };

  const handleConnect = async (provider: string) => {
    if (provider === 'apple') {
      setShowSetupGuide(true);
      return;
    }
    if (provider === 'local') {
      localFileInputRef.current?.click();
      return;
    }
    try {
      const response = await fetch(`/api/auth/${provider}/url`);
      const data = await response.json().catch(() => ({}));
      if (response.ok && data.url) {
        const popup = window.open(data.url, 'oauth_popup', 'width=600,height=700');
        if (!popup) {
          showStatus('Popup blocked. Allow popups for this site.', 'error', 5000);
        }
      } else {
        showStatus(data.error || `Could not start ${provider} connection.`, 'error', 7000);
      }
    } catch (err) {
      console.error(`Failed to connect ${provider}`, err);
      showStatus(`Failed to connect ${provider}.`, 'error');
    }
  };

  const handleLocalImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    try {
      const parsedEvents = await parseIcsEventsFromFile(file);
      const encryptedTokens = await CryptoService.encrypt(JSON.stringify({
        events: parsedEvents,
        sourceName: file.name,
        importedAt: Date.now(),
      }), pin);
      await storageProvider.saveCalendarConnection({
        provider: 'local',
        encryptedTokens,
        accountName: file.name,
        timestamp: Date.now(),
      });
      showStatus(`Imported ${parsedEvents.length} events from ${file.name}`, 'success');
      loadStatus();
      onConnectionsChange?.();
    } catch {
      showStatus('Could not read calendar file. Use a valid .ics export.', 'error');
    }
  };

  const handleAppleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsConnectingApple(true);
    try {
      const encryptedTokens = await CryptoService.encrypt(JSON.stringify(appleConfig), pin);
      await storageProvider.saveCalendarConnection({
        provider: 'apple',
        encryptedTokens,
        timestamp: Date.now(),
      });
      setAppleConfig({ url: '', user: '', password: '' });
      showStatus('Apple Calendar connected.', 'success');
      loadStatus();
      onConnectionsChange?.();
    } finally {
      setIsConnectingApple(false);
    }
  };

  const handleDisconnect = async (id: number) => {
    await storageProvider.deleteCalendarConnection(id);
    setConfirmDisconnect(null);
    showStatus('Calendar disconnected.', 'info');
    loadStatus();
    onConnectionsChange?.();
  };

  const providers: { id: string; name: string; icon: IconName; description: string }[] = [
    { id: 'google', name: 'Google Workspace', icon: 'google', description: 'Calendar, Gmail, Drive' },
    { id: 'microsoft', name: 'Outlook Calendar', icon: 'microsoft', description: 'OAuth sync' },
    { id: 'notion', name: 'Notion', icon: 'notion', description: 'Database sync' },
    { id: 'apple', name: 'Apple Calendar', icon: 'apple', description: 'CalDAV' },
    { id: 'local', name: 'Device Calendar', icon: 'calendar', description: 'Import .ics' },
  ];

  const isProviderReady = (id: string) => {
    if (id === 'apple' || id === 'local') return true;
    return oauthStatus?.[id as keyof OAuthStatus]?.configured ?? false;
  };

  return (
    <div className="calendar-settings">
      <input
        ref={localFileInputRef}
        type="file"
        accept=".ics,text/calendar"
        className="hidden"
        onChange={handleLocalImport}
      />

      <form onSubmit={handleSaveCredentials} className="settings-oauth-form">
        <h4 className="settings-subheading">OAuth Credentials</h4>
        <p className="settings-section-desc">
          Add your Google, Microsoft, or Notion app credentials here. No server restart required.
        </p>

        <div className="settings-oauth-grid">
          <div className="settings-oauth-provider">
            <label className="settings-label">Google Client ID</label>
            <input
              type="text"
              placeholder={oauthStatus?.google.configured ? `Configured (${oauthStatus.google.clientId})` : 'Google OAuth Client ID'}
              value={googleCreds.clientId}
              onChange={(e) => setGoogleCreds({ ...googleCreds, clientId: e.target.value })}
            />
            <label className="settings-label">Google Client Secret</label>
            <input
              type="password"
              placeholder="Google OAuth Client Secret"
              value={googleCreds.clientSecret}
              onChange={(e) => setGoogleCreds({ ...googleCreds, clientSecret: e.target.value })}
            />
          </div>

          <div className="settings-oauth-provider">
            <label className="settings-label">Microsoft Client ID</label>
            <input
              type="text"
              placeholder={oauthStatus?.microsoft.configured ? `Configured (${oauthStatus.microsoft.clientId})` : 'Azure App Client ID'}
              value={microsoftCreds.clientId}
              onChange={(e) => setMicrosoftCreds({ ...microsoftCreds, clientId: e.target.value })}
            />
            <label className="settings-label">Microsoft Client Secret</label>
            <input
              type="password"
              placeholder="Azure App Client Secret"
              value={microsoftCreds.clientSecret}
              onChange={(e) => setMicrosoftCreds({ ...microsoftCreds, clientSecret: e.target.value })}
            />
          </div>

          <div className="settings-oauth-provider">
            <label className="settings-label">Notion Client ID</label>
            <input
              type="text"
              placeholder={oauthStatus?.notion.configured ? `Configured (${oauthStatus.notion.clientId})` : 'Notion OAuth Client ID'}
              value={notionCreds.clientId}
              onChange={(e) => setNotionCreds({ ...notionCreds, clientId: e.target.value })}
            />
            <label className="settings-label">Notion Client Secret</label>
            <input
              type="password"
              placeholder="Notion OAuth Client Secret"
              value={notionCreds.clientSecret}
              onChange={(e) => setNotionCreds({ ...notionCreds, clientSecret: e.target.value })}
            />
          </div>
        </div>

        <button type="submit" className="btn-primary" disabled={isSavingCreds}>
          {isSavingCreds ? 'Saving...' : 'Save OAuth Credentials'}
        </button>
      </form>

      <button
        type="button"
        className="btn-secondary settings-guide-toggle"
        onClick={() => setShowSetupGuide(!showSetupGuide)}
      >
        {showSetupGuide ? 'Hide Setup Guide' : 'Show Setup Guide'}
      </button>

      {showSetupGuide && (
        <div className="settings-guide card">
          <p className="font-semibold mb-2">Google Cloud Console</p>
          <ul className="settings-guide-list">
            <li>Enable Calendar API, Gmail API, and Drive API</li>
            <li>Create OAuth 2.0 Web Client credentials</li>
            <li>Add redirect URI: <code>{window.location.origin}/auth/google/callback</code></li>
            <li>Add your Gmail as a test user if the app is in Testing mode</li>
          </ul>
          <p className="font-semibold mb-2 mt-4">Microsoft Azure</p>
          <ul className="settings-guide-list">
            <li>Redirect URI: <code>{window.location.origin}/auth/microsoft/callback</code></li>
          </ul>
          <p className="font-semibold mb-2 mt-4">Notion</p>
          <ul className="settings-guide-list">
            <li>Redirect URI: <code>{window.location.origin}/auth/notion/callback</code></li>
          </ul>
        </div>
      )}

      {googleStatus && (
        <div className="settings-google-status card">
          <p><strong>Google:</strong> {googleStatus.email || 'Connected'}</p>
          <p>Calendar: {googleStatus.calendar.ok ? 'OK' : googleStatus.calendar.error}</p>
          <p>Gmail: {googleStatus.gmail.ok ? 'OK' : googleStatus.gmail.error}</p>
          <p>Drive: {googleStatus.drive.ok ? 'OK' : googleStatus.drive.error}</p>
        </div>
      )}

      <h4 className="settings-subheading">Connected Calendars</h4>
      <div className="settings-calendar-providers">
        {providers.map((p) => {
          const conn = connections.find((c) => c.provider === p.id);
          const ready = isProviderReady(p.id);
          return (
            <div key={p.id} className="settings-calendar-card card">
              <div className="settings-calendar-card-header">
                <AppIcon name={p.icon} size={20} isDarkMode={isDarkMode} />
                <div>
                  <span className="font-medium">{p.name}</span>
                  <span className="settings-calendar-desc">{p.description}</span>
                </div>
              </div>
              {conn ? (
                <div className="settings-calendar-connected">
                  <span className="text-green-600 text-xs">Connected{conn.accountName ? ` · ${conn.accountName}` : ''}</span>
                  {confirmDisconnect === conn.id ? (
                    <div className="settings-calendar-actions">
                      <button type="button" className="btn-secondary text-xs" onClick={() => handleDisconnect(conn.id!)}>Confirm</button>
                      <button type="button" className="btn-secondary text-xs" onClick={() => setConfirmDisconnect(null)}>Cancel</button>
                    </div>
                  ) : (
                    <button type="button" className="text-xs text-red-500" onClick={() => setConfirmDisconnect(conn.id!)}>Disconnect</button>
                  )}
                </div>
              ) : (
                <button
                  type="button"
                  className="btn-primary text-xs"
                  disabled={!ready}
                  onClick={() => handleConnect(p.id)}
                >
                  {!ready && p.id !== 'apple' && p.id !== 'local'
                    ? 'Add credentials first'
                    : p.id === 'local' ? 'Import .ics' : 'Connect'}
                </button>
              )}
            </div>
          );
        })}
      </div>

      {!connections.some((c) => c.provider === 'apple') && (
        <form onSubmit={handleAppleConnect} className="settings-apple-form card">
          <h4 className="settings-subheading">Apple Calendar (CalDAV)</h4>
          <div className="settings-apple-grid">
            <input type="text" placeholder="caldav.icloud.com" value={appleConfig.url} onChange={(e) => setAppleConfig({ ...appleConfig, url: e.target.value })} required />
            <input type="text" placeholder="email@icloud.com" value={appleConfig.user} onChange={(e) => setAppleConfig({ ...appleConfig, user: e.target.value })} required />
            <input type="password" placeholder="App-specific password" value={appleConfig.password} onChange={(e) => setAppleConfig({ ...appleConfig, password: e.target.value })} required />
          </div>
          <button type="submit" className="btn-primary text-xs" disabled={isConnectingApple}>
            {isConnectingApple ? 'Connecting...' : 'Connect Apple Calendar'}
          </button>
        </form>
      )}
    </div>
  );
};
