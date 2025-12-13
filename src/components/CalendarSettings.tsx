/**
 * Calendar Settings Component
 * UI for connecting/disconnecting calendars and configuring auto-launch
 */

import React, { useState, useEffect } from 'react';
import { CalendarService, CalendarProvider, CalendarConfig } from '../services/CalendarService';
import { GoogleCalendarService } from '../services/GoogleCalendarService';
import { OutlookCalendarService } from '../services/OutlookCalendarService';
import { AutoLaunchService } from '../services/AutoLaunchService';

interface CalendarSettingsProps {
    onClose: () => void;
    onCalendarConnected?: (provider: CalendarProvider) => void;
}

export const CalendarSettings: React.FC<CalendarSettingsProps> = ({ onClose, onCalendarConnected }) => {
    const [selectedProvider, setSelectedProvider] = useState<CalendarProvider>('google');
    const [isConnecting, setIsConnecting] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const [connectedProvider, setConnectedProvider] = useState<CalendarProvider | null>(null);
    const [autoLaunchEnabled, setAutoLaunchEnabled] = useState(false);
    const [preLaunchSeconds, setPreLaunchSeconds] = useState(30);
    const [checkIntervalSeconds, setCheckIntervalSeconds] = useState(60);
    const [error, setError] = useState<string>('');

    useEffect(() => {
        loadSettings();
        checkConnection();
    }, []);

    const loadSettings = async () => {
        try {
            const configStr = localStorage.getItem('calendar_config');
            if (configStr) {
                const config: CalendarConfig = JSON.parse(configStr);
                setAutoLaunchEnabled(config.autoLaunchEnabled || false);
                setPreLaunchSeconds(config.preLaunchSeconds || 30);
                setCheckIntervalSeconds(config.checkIntervalSeconds || 60);
                if (config.provider) {
                    setSelectedProvider(config.provider);
                }
            }
        } catch (error) {
            console.error('Error loading calendar settings:', error);
        }
    };

    const checkConnection = async () => {
        try {
            const configStr = localStorage.getItem('calendar_config');
            if (configStr) {
                const config: CalendarConfig = JSON.parse(configStr);
                if (config.provider) {
                    // Check if credentials exist in localStorage (stored separately)
                    const credsKey = `calendar_${config.provider}_creds`;
                    const credentials = localStorage.getItem(credsKey);
                    if (credentials) {
                        const service = getCalendarService(config.provider);
                        const connected = await service.isConnected();
                        setIsConnected(connected);
                        setConnectedProvider(connected ? config.provider : null);
                    } else {
                        setIsConnected(false);
                        setConnectedProvider(null);
                    }
                }
            }
        } catch (error) {
            console.error('Error checking connection:', error);
        }
    };

    const getCalendarService = (provider: CalendarProvider): CalendarService => {
        switch (provider) {
            case 'google':
                return new GoogleCalendarService();
            case 'outlook':
                return new OutlookCalendarService();
            default:
                throw new Error(`Unsupported provider: ${provider}`);
        }
    };

    const handleConnect = async () => {
        setIsConnecting(true);
        setError('');

        try {
            const service = getCalendarService(selectedProvider);
            await service.connect();
            // OAuth will redirect, so we won't reach here
            // But if we do, connection was successful
            setIsConnected(true);
            setConnectedProvider(selectedProvider);
            
            // Load existing config to preserve autoStartRecording if it exists
            const existingConfigStr = localStorage.getItem('calendar_config');
            const existingConfig = existingConfigStr ? JSON.parse(existingConfigStr) : {};
            
            // Save config
            const config: CalendarConfig = {
                enabled: true,
                provider: selectedProvider,
                autoLaunchEnabled,
                preLaunchSeconds,
                checkIntervalSeconds,
                autoStartRecording: existingConfig.autoStartRecording || false
            };
            localStorage.setItem('calendar_config', JSON.stringify(config));

            if (onCalendarConnected) {
                onCalendarConnected(selectedProvider);
            }
        } catch (error: any) {
            setError(error.message || 'Failed to connect to calendar');
            console.error('Calendar connection error:', error);
        } finally {
            setIsConnecting(false);
        }
    };

    const handleDisconnect = async () => {
        if (!connectedProvider) return;

        try {
            const service = getCalendarService(connectedProvider);
            await service.disconnect();
            
            setIsConnected(false);
            setConnectedProvider(null);
            
            // Clear config
            localStorage.removeItem('calendar_config');
            localStorage.removeItem(`calendar_${connectedProvider}_creds`);
        } catch (error: any) {
            setError(error.message || 'Failed to disconnect from calendar');
            console.error('Calendar disconnection error:', error);
        }
    };

    const handleSaveSettings = () => {
        const config: CalendarConfig = {
            enabled: isConnected,
            provider: connectedProvider || undefined,
            autoLaunchEnabled,
            preLaunchSeconds,
            checkIntervalSeconds
        };
        localStorage.setItem('calendar_config', JSON.stringify(config));

        // Update auto-launch service
        const autoLaunchService = AutoLaunchService.getInstance();
        autoLaunchService.updateConfig({
            enabled: autoLaunchEnabled && isConnected,
            preLaunchSeconds,
            checkIntervalSeconds,
            autoStartRecording: config.autoStartRecording || false
        });

        onClose();
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '600px' }}>
                <div className="modal-header">
                    <h2>Calendar Settings</h2>
                    <button className="modal-close" onClick={onClose}>×</button>
                </div>

                <div className="modal-body" style={{ padding: '20px' }}>
                    {error && (
                        <div className="status error" style={{ marginBottom: '15px' }}>
                            {error}
                        </div>
                    )}

                    {/* Calendar Provider Selection */}
                    <div style={{ marginBottom: '20px' }}>
                        <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                            Calendar Provider
                        </label>
                        <select
                            value={selectedProvider}
                            onChange={(e) => setSelectedProvider(e.target.value as CalendarProvider)}
                            disabled={isConnected}
                            style={{ width: '100%', padding: '8px', fontSize: '14px' }}
                        >
                            <option value="google">Google Calendar</option>
                            <option value="outlook">Outlook / Microsoft 365</option>
                        </select>
                    </div>

                    {/* Connection Status */}
                    {isConnected && connectedProvider && (
                        <div className="status success" style={{ marginBottom: '15px' }}>
                            ✓ Connected to {connectedProvider === 'google' ? 'Google Calendar' : 'Outlook Calendar'}
                        </div>
                    )}

                    {/* Connect/Disconnect Button */}
                    <div style={{ marginBottom: '20px' }}>
                        {!isConnected ? (
                            <button
                                className="btn-primary"
                                onClick={handleConnect}
                                disabled={isConnecting}
                                style={{ width: '100%' }}
                            >
                                {isConnecting ? 'Connecting...' : `Connect to ${selectedProvider === 'google' ? 'Google Calendar' : 'Outlook'}`}
                            </button>
                        ) : (
                            <button
                                className="btn-secondary"
                                onClick={handleDisconnect}
                                style={{ width: '100%' }}
                            >
                                Disconnect
                            </button>
                        )}
                    </div>

                    {/* Auto-Launch Settings */}
                    <div style={{ marginTop: '30px', paddingTop: '20px', borderTop: '1px solid #ddd' }}>
                        <h3 style={{ marginBottom: '15px' }}>Auto-Launch Settings</h3>

                        <div style={{ marginBottom: '15px' }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <input
                                    type="checkbox"
                                    checked={autoLaunchEnabled}
                                    onChange={(e) => setAutoLaunchEnabled(e.target.checked)}
                                    disabled={!isConnected}
                                />
                                <span>Enable auto-launch before meetings</span>
                            </label>
                        </div>

                        {autoLaunchEnabled && (
                            <>
                                <div style={{ marginBottom: '15px' }}>
                                    <label style={{ display: 'block', marginBottom: '8px' }}>
                                        Launch app X seconds before meeting:
                                    </label>
                                    <input
                                        type="number"
                                        value={preLaunchSeconds}
                                        onChange={(e) => setPreLaunchSeconds(parseInt(e.target.value) || 30)}
                                        min="10"
                                        max="300"
                                        style={{ width: '100%', padding: '8px' }}
                                    />
                                    <small style={{ color: '#666' }}>Default: 30 seconds</small>
                                </div>

                                <div style={{ marginBottom: '15px' }}>
                                    <label style={{ display: 'block', marginBottom: '8px' }}>
                                        Check for meetings every X seconds:
                                    </label>
                                    <input
                                        type="number"
                                        value={checkIntervalSeconds}
                                        onChange={(e) => setCheckIntervalSeconds(parseInt(e.target.value) || 60)}
                                        min="30"
                                        max="300"
                                        style={{ width: '100%', padding: '8px' }}
                                    />
                                    <small style={{ color: '#666' }}>Default: 60 seconds</small>
                                </div>
                            </>
                        )}
                    </div>
                </div>

                <div className="modal-footer" style={{ padding: '15px 20px', borderTop: '1px solid #ddd', display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                    <button className="btn-secondary" onClick={onClose}>
                        Cancel
                    </button>
                    <button className="btn-primary" onClick={handleSaveSettings}>
                        Save Settings
                    </button>
                </div>
            </div>
        </div>
    );
};
