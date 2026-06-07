import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'motion/react';
import { StorageProvider, CalendarConnection } from '../types';
import { CryptoService } from '../services/cryptoService';
import { AppIcon } from './AppIcon';
import type { IconName } from '../icons';

interface CalendarEvent {
  id: string;
  title: string;
  start: string;
  end: string;
  provider: string;
  isDatabase?: boolean;
}

interface CalendarIntegrationProps {
  pin: string;
  storageProvider: StorageProvider;
  showStatus: (msg: string, type: 'success' | 'error' | 'info') => void;
  isDarkMode: boolean;
}

export const CalendarIntegration: React.FC<CalendarIntegrationProps> = ({ pin, storageProvider, showStatus, isDarkMode }) => {
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [connectedProviders, setConnectedProviders] = useState<string[]>([]);
  const [connections, setConnections] = useState<CalendarConnection[]>([]);
  const [showSetupGuide, setShowSetupGuide] = useState(false);
  const [appleConfig, setAppleConfig] = useState({ url: '', user: '', password: '' });
  const [isConnectingApple, setIsConnectingApple] = useState(false);
  const [confirmDisconnect, setConfirmDisconnect] = useState<number | null>(null);

  const redirectUri = `${window.location.origin}/auth/`;

  const fetchEvents = useCallback(async () => {
    setIsLoading(true);
    try {
      const conns = await storageProvider.getAllCalendarConnections();
      setConnections(conns);
      setConnectedProviders(conns.map(c => c.provider));

      if (conns.length === 0) {
        setEvents([]);
        setIsLoading(false);
        return;
      }

      // Decrypt tokens for each connection
      const decryptedConnections = await Promise.all(conns.map(async (c) => {
        const decryptedTokens = await CryptoService.decrypt(c.encryptedTokens, pin);
        return {
          provider: c.provider,
          tokens: JSON.parse(decryptedTokens)
        };
      }));

      const response = await fetch('/api/calendar/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ connections: decryptedConnections })
      });

      if (response.ok) {
        const data = await response.json();
        setEvents(data);
      }
    } catch (error) {
      console.error("Failed to fetch calendar events", error);
    } finally {
      setIsLoading(false);
    }
  }, [storageProvider, pin]);

  useEffect(() => {
    fetchEvents();

    const handleMessage = async (event: MessageEvent) => {
      if (event.origin !== window.location.origin) {
        return;
      }

      if (event.data?.type === 'OAUTH_AUTH_SUCCESS') {
        const { provider, tokens } = event.data;
        try {
          // Encrypt tokens before saving
          const encryptedTokens = await CryptoService.encrypt(JSON.stringify(tokens), pin);
          const newConn: CalendarConnection = {
            provider,
            encryptedTokens,
            timestamp: Date.now()
          };
          await storageProvider.saveCalendarConnection(newConn);
          fetchEvents();
        } catch (err) {
          console.error("Failed to save encrypted connection", err);
        }
      }
    };
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [fetchEvents, pin, storageProvider]);

  const handleConnect = async (provider: string) => {
    if (provider === 'apple') {
      setShowSetupGuide(true);
      return;
    }
    try {
      const response = await fetch(`/api/auth/${provider}/url`);
      if (response.ok) {
        const { url } = await response.json();
        window.open(url, 'oauth_popup', 'width=600,height=700');
      }
    } catch (error) {
      console.error(`Failed to connect to ${provider}`, error);
    }
  };

  const handleAppleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsConnectingApple(true);
    try {
      const encryptedTokens = await CryptoService.encrypt(JSON.stringify(appleConfig), pin);
      const newConn: CalendarConnection = {
        provider: 'apple',
        encryptedTokens,
        timestamp: Date.now()
      };
      await storageProvider.saveCalendarConnection(newConn);
      fetchEvents();
      setAppleConfig({ url: '', user: '', password: '' });
      showStatus('Apple Calendar connected securely', 'success');
    } catch (err) {
      console.error("Failed to save Apple connection", err);
    } finally {
      setIsConnectingApple(false);
    }
  };

  const handleDisconnect = async (id: number) => {
    await storageProvider.deleteCalendarConnection(id);
    setConfirmDisconnect(null);
    fetchEvents();
    showStatus('Calendar disconnected', 'info');
  };

  const providers: { id: string; name: string; icon: IconName }[] = [
    { id: 'google', name: 'Google Calendar', icon: 'google' },
    { id: 'microsoft', name: 'Outlook Calendar', icon: 'microsoft' },
    { id: 'notion', name: 'Notion', icon: 'notion' },
    { id: 'apple', name: 'Apple Calendar', icon: 'apple' },
  ];

  return (
    <div className="calendar-integration p-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-2xl font-bold">Calendar Integrations</h2>
          <p className="text-sm text-gray-500">Sync your schedules across all platforms</p>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={() => setShowSetupGuide(!showSetupGuide)} 
            className="btn-secondary text-sm"
          >
            {showSetupGuide ? 'Hide Setup Guide' : 'Show Setup Guide'}
          </button>
          <button 
            onClick={fetchEvents} 
            className="btn-primary text-sm"
            disabled={isLoading}
          >
            {isLoading ? 'Syncing...' : 'Sync All'}
          </button>
        </div>
      </div>

      {showSetupGuide && (
        <motion.div 
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="card p-6 mb-8 bg-blue-50 border-blue-100 dark:bg-blue-900/20 dark:border-blue-800"
        >
          <h3 className="font-bold mb-4 flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full" />
            OAuth Setup Guide for End Users
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            <div>
              <p className="font-bold mb-2">1. Register Apps</p>
              <ul className="list-disc list-inside space-y-1 opacity-80">
                <li>Google: Cloud Console</li>
                <li>Microsoft: Azure Portal</li>
                <li>Notion: Developers Portal</li>
              </ul>
            </div>
            <div>
              <p className="font-bold mb-2">2. Use Redirect URIs</p>
              <div className="space-y-2">
                <div>
                  <span className="text-[10px] uppercase font-bold opacity-50 block">Google</span>
                  <code className="bg-white/50 dark:bg-black/20 p-1 rounded text-[10px] break-all">
                    {redirectUri}google/callback
                  </code>
                </div>
                <div>
                  <span className="text-[10px] uppercase font-bold opacity-50 block">Microsoft</span>
                  <code className="bg-white/50 dark:bg-black/20 p-1 rounded text-[10px] break-all">
                    {redirectUri}microsoft/callback
                  </code>
                </div>
                <div>
                  <span className="text-[10px] uppercase font-bold opacity-50 block">Notion</span>
                  <code className="bg-white/50 dark:bg-black/20 p-1 rounded text-[10px] break-all">
                    {redirectUri}notion/callback
                  </code>
                </div>
              </div>
            </div>
            <div>
              <p className="font-bold mb-2">3. Set Secrets</p>
              <p className="opacity-80">Add your Client IDs and Secrets in the <b>Settings</b> menu of this app.</p>
            </div>
          </div>
        </motion.div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
        {providers.map((p) => {
          const conn = connections.find(c => c.provider === p.id);
          const isConfirming = confirmDisconnect === conn?.id;

          return (
            <div key={p.id} className="card p-4 flex flex-col items-center justify-between gap-4 relative">
              <div className="flex items-center gap-3">
                <AppIcon name={p.icon} size={20} isDarkMode={isDarkMode} />
                <span className="font-medium">{p.name}</span>
              </div>
              {conn ? (
                <div className="flex flex-col items-center gap-2 w-full">
                  <span className="text-xs text-green-500 font-medium flex items-center gap-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full" /> Connected
                  </span>
                  {isConfirming ? (
                    <div className="flex gap-2 mt-1">
                      <button 
                        onClick={() => handleDisconnect(conn.id!)}
                        className="text-[10px] text-red-500 font-bold hover:underline"
                      >
                        Confirm
                      </button>
                      <button 
                        onClick={() => setConfirmDisconnect(null)}
                        className="text-[10px] text-gray-500 hover:underline"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button 
                      onClick={() => setConfirmDisconnect(conn.id!)}
                      className="text-[10px] text-red-500 hover:underline"
                    >
                      Disconnect
                    </button>
                  )}
                </div>
              ) : (
                <button 
                  onClick={() => handleConnect(p.id)}
                  className="btn-primary text-xs w-full"
                >
                  Connect
                </button>
              )}
            </div>
          );
        })}
      </div>

      {connectedProviders.includes('apple') === false && appleConfig.url === '' && (
        <div className="mb-12">
          <h3 className="text-lg font-semibold mb-4">Manual Apple Calendar (CalDAV)</h3>
          <form onSubmit={handleAppleConnect} className="card p-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="input-label">Server URL</label>
              <input 
                type="text" 
                placeholder="caldav.icloud.com" 
                value={appleConfig.url}
                onChange={(e) => setAppleConfig({...appleConfig, url: e.target.value})}
                required
              />
            </div>
            <div>
              <label className="input-label">Username</label>
              <input 
                type="text" 
                placeholder="email@icloud.com" 
                value={appleConfig.user}
                onChange={(e) => setAppleConfig({...appleConfig, user: e.target.value})}
                required
              />
            </div>
            <div>
              <label className="input-label">App-Specific Password</label>
              <input 
                type="password" 
                placeholder="xxxx-xxxx-xxxx-xxxx" 
                value={appleConfig.password}
                onChange={(e) => setAppleConfig({...appleConfig, password: e.target.value})}
                required
              />
              <button 
                type="submit" 
                className="btn-primary text-xs mt-4 w-full"
                disabled={isConnectingApple}
              >
                {isConnectingApple ? 'Connecting...' : 'Connect Apple'}
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="events-list">
        <h3 className="text-lg font-semibold mb-4">Upcoming Events & Databases</h3>
        {events.length === 0 ? (
          <div className="empty-state p-12 text-center border-2 border-dashed border-gray-200 rounded-xl">
            <Calendar className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500">No events found. Connect a calendar to get started.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {events.map((event) => (
              <motion.div 
                key={event.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="card p-4 flex items-center justify-between hover:shadow-md transition-shadow"
              >
                <div className="flex items-center gap-4">
                  <div className={`w-1 h-10 rounded-full ${
                    event.provider === 'google' ? 'bg-blue-500' : 
                    event.provider === 'microsoft' ? 'bg-blue-600' : 'bg-black'
                  }`} />
                  <div>
                    <h4 className="font-medium">{event.title}</h4>
                    {!event.isDatabase && (
                      <p className="text-xs text-gray-500">
                        {new Date(event.start).toLocaleString()} - {new Date(event.end).toLocaleTimeString()}
                      </p>
                    )}
                    {event.isDatabase && <p className="text-xs text-gray-500">Notion Database</p>}
                  </div>
                </div>
                <span className="text-[10px] uppercase tracking-wider font-bold opacity-50">
                  {event.provider}
                </span>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
