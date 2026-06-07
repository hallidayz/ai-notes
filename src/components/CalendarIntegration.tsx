import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'motion/react';
import { StorageProvider } from '../types';
import { CryptoService } from '../services/cryptoService';
import { AppIcon } from './AppIcon';

interface CalendarEvent {
  id: string;
  title: string;
  start: string;
  end: string;
  provider: string;
  isDatabase?: boolean;
  kind?: 'calendar' | 'gmail' | 'drive';
}

interface CalendarIntegrationProps {
  pin: string;
  storageProvider: StorageProvider;
  showStatus: (msg: string, type: 'success' | 'error' | 'info') => void;
  isDarkMode: boolean;
  refreshKey?: number;
}

export const CalendarIntegration: React.FC<CalendarIntegrationProps> = ({
  pin,
  storageProvider,
  showStatus,
  isDarkMode,
  refreshKey = 0,
}) => {
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchEvents = useCallback(async () => {
    setIsLoading(true);
    try {
      const conns = await storageProvider.getAllCalendarConnections();
      if (conns.length === 0) {
        setEvents([]);
        return;
      }

      const decryptedConnections = await Promise.all(conns.map(async (c) => {
        const decryptedTokens = await CryptoService.decrypt(c.encryptedTokens, pin);
        return { provider: c.provider, tokens: JSON.parse(decryptedTokens) };
      }));

      const response = await fetch('/api/calendar/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ connections: decryptedConnections }),
      });

      if (response.ok) {
        const data = await response.json();
        setEvents(Array.isArray(data) ? data : data.events ?? []);
      } else {
        showStatus('Failed to sync events. Configure calendars in Settings.', 'error');
      }
    } catch (error) {
      console.error('Failed to fetch calendar events', error);
      showStatus('Failed to sync events.', 'error');
    } finally {
      setIsLoading(false);
    }
  }, [storageProvider, pin, showStatus]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents, refreshKey]);

  return (
    <div className="calendar-integration p-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-2xl font-bold">Calendar</h2>
          <p className="text-sm text-gray-500">Upcoming events from your connected calendars</p>
        </div>
        <button onClick={fetchEvents} className="btn-primary text-sm" disabled={isLoading}>
          {isLoading ? 'Syncing...' : 'Sync'}
        </button>
      </div>

      <div className="events-list">
        {events.length === 0 ? (
          <div className="empty-state p-12 text-center border-2 border-dashed border-gray-200 rounded-xl">
            <AppIcon name="calendar" size={48} isDarkMode={isDarkMode} className="mx-auto mb-4 opacity-40" />
            <p className="text-gray-500">No events yet. Open Settings to connect Google, Outlook, Notion, Apple, or import a device calendar.</p>
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
                    event.provider === 'microsoft' ? 'bg-blue-600' :
                    event.provider === 'local' ? 'bg-emerald-500' :
                    event.provider === 'apple' ? 'bg-gray-500' : 'bg-black'
                  }`} />
                  <div>
                    <h4 className="font-medium">{event.title}</h4>
                    {event.kind && (
                      <p className="text-[10px] uppercase tracking-wide text-gray-400">{event.kind}</p>
                    )}
                    {!event.isDatabase && event.kind !== 'gmail' && (
                      <p className="text-xs text-gray-500">
                        {new Date(event.start).toLocaleString()} – {new Date(event.end).toLocaleTimeString()}
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
