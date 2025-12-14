/**
 * Calendar Settings Component
 * UI for connecting/disconnecting calendars and configuring auto-launch
 */

import React, { useState, useEffect } from 'react';
import { CalendarService, CalendarProvider, CalendarConfig } from '../services/CalendarService';
import { GoogleCalendarService } from '../services/GoogleCalendarService';
import { OutlookCalendarService } from '../services/OutlookCalendarService';
import { AutoLaunchService } from '../services/AutoLaunchService';
import WeeklySchedule from './WeeklySchedule';

interface CalendarSettingsProps {
    onClose: () => void;
    onCalendarConnected?: (provider: CalendarProvider) => void;
}

export const CalendarSettings: React.FC<CalendarSettingsProps> = ({ onClose, onCalendarConnected }) => {
    const [selectedProvider, setSelectedProvider] = useState<CalendarProvider>('google');
    const [isConnecting, setIsConnecting] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const [connectedProvider, setConnectedProvider] = useState<CalendarProvider | null>(null);
    const [calendarService, setCalendarService] = useState<CalendarService | null>(null);
    const [autoLaunchEnabled, setAutoLaunchEnabled] = useState(false);
    const [preLaunchSeconds, setPreLaunchSeconds] = useState(30);
    const [checkIntervalSeconds, setCheckIntervalSeconds] = useState(60);
    const [error, setError] = useState<string>('');
    const [showSchedule, setShowSchedule] = useState(false);
    

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
                        setCalendarService(connected ? service : null);
                    } else {
                        setIsConnected(false);
                        setConnectedProvider(null);
                        setCalendarService(null);
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
                // Client ID should be configured via environment variable or backend
                const googleId = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';
                return new GoogleCalendarService(googleId);
            case 'outlook':
                // Client ID should be configured via environment variable or backend
                const outlookId = import.meta.env.VITE_OUTLOOK_CLIENT_ID || '';
                return new OutlookCalendarService(outlookId);
            default:
                throw new Error(`Unsupported provider: ${provider}`);
        }
    };

    const handleConnect = async (provider: CalendarProvider) => {
        setIsConnecting(true);
        setError('');

        try {
            const service = getCalendarService(provider);
            await service.connect();
            
            // Connection successful
            setIsConnected(true);
            setConnectedProvider(provider);
            setCalendarService(service);
            
            // Load existing config to preserve autoStartRecording if it exists
            const existingConfigStr = localStorage.getItem('calendar_config');
            const existingConfig = existingConfigStr ? JSON.parse(existingConfigStr) : {};
            
            // Save config
            const config: CalendarConfig = {
                enabled: true,
                provider: provider,
                autoLaunchEnabled,
                preLaunchSeconds,
                checkIntervalSeconds,
                autoStartRecording: existingConfig.autoStartRecording || false
            };
            localStorage.setItem('calendar_config', JSON.stringify(config));

            if (onCalendarConnected) {
                onCalendarConnected(provider);
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
            setCalendarService(null);
            
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
        <div 
            className="modal-overlay" 
            onClick={onClose}
        >
            <div 
                className="modal-content" 
                onClick={(e) => e.stopPropagation()} 
                style={{ 
                    maxWidth: showSchedule ? '900px' : '520px',
                    width: '100%',
                    maxHeight: showSchedule ? '90vh' : 'auto'
                }}
            >
                <div className="modal-header" style={{
                    padding: '24px 24px 0 24px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    borderBottom: 'none',
                    marginBottom: '0'
                }}>
                    <h2 style={{
                        margin: 0,
                        fontSize: '24px',
                        fontWeight: '600',
                        color: '#1a1a1a',
                        letterSpacing: '-0.01em'
                    }}>
                        Calendar Settings
                    </h2>
                    <button 
                        className="modal-close" 
                        onClick={onClose}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            fontSize: '24px',
                            cursor: 'pointer',
                            color: '#666',
                            padding: '4px 8px',
                            borderRadius: '6px',
                            transition: 'all 0.15s',
                            lineHeight: '1',
                            width: '32px',
                            height: '32px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = 'rgba(0, 0, 0, 0.05)';
                            e.currentTarget.style.color = '#1a1a1a';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'transparent';
                            e.currentTarget.style.color = '#666';
                        }}
                    >
                        ×
                    </button>
                </div>

                <div className="modal-body" style={{ 
                    padding: '32px 24px',
                    overflowY: 'auto',
                    flex: 1
                }}>
                    {error && (
                        <div className="status error" style={{ 
                            marginBottom: '24px', 
                            padding: '12px 16px', 
                            borderRadius: '8px', 
                            background: 'rgba(239, 68, 68, 0.08)', 
                            border: '1px solid rgba(239, 68, 68, 0.2)', 
                            color: '#dc2626',
                            fontSize: '14px',
                            lineHeight: '1.5'
                        }}>
                            {error}
                        </div>
                    )}

                    {/* Obsidian/Notion-style Calendar Connection UI */}
                    {!isConnected && (
                        <div style={{ 
                            background: 'linear-gradient(135deg, #fafbfc 0%, #f8f9fa 100%)',
                            borderRadius: '12px',
                            padding: '40px 32px',
                            border: '1px solid rgba(0, 0, 0, 0.06)',
                            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)'
                        }}
                        className="calendar-connect-card"
                        >
                            <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                                <div style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    width: '56px',
                                    height: '56px',
                                    background: 'linear-gradient(135deg, rgba(2, 41, 91, 0.08) 0%, rgba(44, 95, 65, 0.08) 100%)',
                                    borderRadius: '12px',
                                    marginBottom: '20px',
                                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.04)'
                                }}>
                                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-authority-navy, #02295b)" strokeWidth="1.5">
                                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
                                        <line x1="16" y1="2" x2="16" y2="6"/>
                                        <line x1="8" y1="2" x2="8" y2="6"/>
                                        <line x1="3" y1="10" x2="21" y2="10"/>
                                    </svg>
                                </div>
                                <h2 style={{ 
                                    margin: '0 0 8px 0', 
                                    fontSize: '24px', 
                                    fontWeight: '600',
                                    color: '#1a1a1a',
                                    letterSpacing: '-0.02em'
                                }}>
                                    Connect Your Calendar
                                </h2>
                                <p style={{ 
                                    margin: 0, 
                                    fontSize: '15px', 
                                    color: '#6b7280',
                                    lineHeight: '1.6',
                                    fontWeight: '400'
                                }}>
                                    Choose how you want to sign in
                                </p>
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '32px' }}>
                                {/* Google Calendar */}
                                <button
                                    type="button"
                                    onClick={async () => {
                                        setSelectedProvider('google');
                                        await handleConnect('google');
                                    }}
                                    disabled={isConnecting}
                                    style={{
                                        width: '100%',
                                        padding: '14px 18px',
                                        background: 'white',
                                        border: `1px solid ${isConnecting && selectedProvider === 'google' ? 'rgba(2, 41, 91, 0.2)' : 'rgba(0, 0, 0, 0.08)'}`,
                                        borderRadius: '8px',
                                        cursor: isConnecting ? 'not-allowed' : 'pointer',
                                        fontSize: '15px',
                                        fontWeight: '500',
                                        color: '#1a1a1a',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '12px',
                                        opacity: isConnecting && selectedProvider !== 'google' ? 0.4 : 1,
                                        transition: 'all 0.15s ease',
                                        boxShadow: isConnecting && selectedProvider === 'google' 
                                            ? '0 2px 8px rgba(0, 0, 0, 0.08)' 
                                            : '0 1px 2px rgba(0, 0, 0, 0.04)',
                                        position: 'relative',
                                        overflow: 'hidden'
                                    }}
                                    onMouseEnter={(e) => {
                                        if (!isConnecting) {
                                            e.currentTarget.style.borderColor = 'rgba(2, 41, 91, 0.2)';
                                            e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.08)';
                                            e.currentTarget.style.background = '#fafbfc';
                                        }
                                    }}
                                    onMouseLeave={(e) => {
                                        if (!isConnecting) {
                                            e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.08)';
                                            e.currentTarget.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.04)';
                                            e.currentTarget.style.background = 'white';
                                        }
                                    }}
                                >
                                    {isConnecting && selectedProvider === 'google' ? (
                                        <>
                                            <div style={{
                                                width: '18px',
                                                height: '18px',
                                                border: '2px solid rgba(2, 41, 91, 0.3)',
                                                borderTop: '2px solid var(--color-authority-navy, #02295b)',
                                                borderRadius: '50%',
                                                animation: 'spin 0.8s linear infinite'
                                            }} />
                                            <span style={{ color: '#6b7280' }}>Opening Google...</span>
                                        </>
                                    ) : (
                                        <>
                                            <svg width="20" height="20" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                                                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                                                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                                                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                                                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                                            </svg>
                                            <span>Continue with Google</span>
                                        </>
                                    )}
                                </button>

                                {/* Microsoft/Outlook Calendar */}
                                <button
                                    type="button"
                                    onClick={async () => {
                                        setSelectedProvider('outlook');
                                        await handleConnect('outlook');
                                    }}
                                    disabled={isConnecting}
                                    style={{
                                        width: '100%',
                                        padding: '14px 18px',
                                        background: 'white',
                                        border: `1px solid ${isConnecting && selectedProvider === 'outlook' ? 'rgba(2, 41, 91, 0.2)' : 'rgba(0, 0, 0, 0.08)'}`,
                                        borderRadius: '8px',
                                        cursor: isConnecting ? 'not-allowed' : 'pointer',
                                        fontSize: '15px',
                                        fontWeight: '500',
                                        color: '#1a1a1a',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '12px',
                                        opacity: isConnecting && selectedProvider !== 'outlook' ? 0.4 : 1,
                                        transition: 'all 0.15s ease',
                                        boxShadow: isConnecting && selectedProvider === 'outlook' 
                                            ? '0 2px 8px rgba(0, 0, 0, 0.08)' 
                                            : '0 1px 2px rgba(0, 0, 0, 0.04)'
                                    }}
                                    onMouseEnter={(e) => {
                                        if (!isConnecting) {
                                            e.currentTarget.style.borderColor = 'rgba(2, 41, 91, 0.2)';
                                            e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.08)';
                                            e.currentTarget.style.background = '#fafbfc';
                                        }
                                    }}
                                    onMouseLeave={(e) => {
                                        if (!isConnecting) {
                                            e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.08)';
                                            e.currentTarget.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.04)';
                                            e.currentTarget.style.background = 'white';
                                        }
                                    }}
                                >
                                    {isConnecting && selectedProvider === 'outlook' ? (
                                        <>
                                            <div style={{
                                                width: '18px',
                                                height: '18px',
                                                border: '2px solid rgba(2, 41, 91, 0.3)',
                                                borderTop: '2px solid var(--color-authority-navy, #02295b)',
                                                borderRadius: '50%',
                                                animation: 'spin 0.8s linear infinite'
                                            }} />
                                            <span style={{ color: '#6b7280' }}>Opening Microsoft...</span>
                                        </>
                                    ) : (
                                        <>
                                            <svg width="20" height="20" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                                                <path fill="#f25022" d="M1 1h10v10H1z"/>
                                                <path fill="#00a4ef" d="M13 1h10v10H13z"/>
                                                <path fill="#7fba00" d="M1 13h10v10H1z"/>
                                                <path fill="#ffb900" d="M13 13h10v10H13z"/>
                                            </svg>
                                            <span>Continue with Microsoft</span>
                                        </>
                                    )}
                                </button>
                            </div>

                            {/* Security Message - Notion style */}
                            <div style={{
                                marginTop: '28px',
                                paddingTop: '24px',
                                borderTop: '1px solid rgba(0, 0, 0, 0.06)',
                                display: 'flex',
                                alignItems: 'flex-start',
                                gap: '12px'
                            }}>
                                <div style={{ 
                                    fontSize: '18px', 
                                    lineHeight: '1', 
                                    marginTop: '1px',
                                    opacity: 0.7
                                }}>
                                    🔒
                                </div>
                                <div>
                                    <p style={{ 
                                        margin: '0 0 6px 0', 
                                        fontWeight: '500', 
                                        fontSize: '14px',
                                        color: '#1a1a1a',
                                        letterSpacing: '-0.01em'
                                    }}>
                                        Safe & Secure
                                    </p>
                                    <p style={{ 
                                        margin: 0, 
                                        fontSize: '13px', 
                                        color: '#6b7280',
                                        lineHeight: '1.6',
                                        fontWeight: '400'
                                    }}>
                                        We'll never see your password. You'll sign in directly with your calendar provider.
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}


                    {/* Success State - Obsidian/Notion style */}
                    {isConnected && connectedProvider && (
                        <div style={{
                            background: 'linear-gradient(135deg, #f0fdf4 0%, #f8fafc 100%)',
                            borderRadius: '12px',
                            padding: '36px 32px',
                            border: '1px solid rgba(44, 95, 65, 0.15)',
                            textAlign: 'center',
                            marginBottom: '24px',
                            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)'
                        }}>
                            <div style={{
                                width: '64px',
                                height: '64px',
                                background: 'var(--color-strategic-forest, #2c5f41)',
                                borderRadius: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                margin: '0 auto 20px',
                                animation: 'bounce 0.5s ease',
                                boxShadow: '0 4px 12px rgba(44, 95, 65, 0.2)'
                            }}>
                                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                    <polyline points="20 6 9 17 4 12"/>
                                </svg>
                            </div>
                            <h2 style={{
                                margin: '0 0 8px 0',
                                fontSize: '22px',
                                fontWeight: '600',
                                color: '#1a1a1a',
                                letterSpacing: '-0.02em'
                            }}>
                                All Set!
                            </h2>
                            <p style={{
                                margin: '0 0 24px 0',
                                fontSize: '15px',
                                color: '#6b7280',
                                lineHeight: '1.6',
                                fontWeight: '400'
                            }}>
                                Your {connectedProvider === 'google' ? 'Google Calendar' : 'Outlook Calendar'} is now connected and ready to use.
                            </p>
                            <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
                                <button
                                    onClick={() => setShowSchedule(!showSchedule)}
                                    style={{
                                        padding: '8px 16px',
                                        fontSize: '14px',
                                        fontWeight: '500',
                                        borderRadius: '6px',
                                        border: 'none',
                                        background: 'var(--color-authority-navy, #02295b)',
                                        color: 'white',
                                        cursor: 'pointer',
                                        transition: 'all 0.15s ease',
                                        boxShadow: '0 1px 3px rgba(2, 41, 91, 0.2)'
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.background = '#021d3f';
                                        e.currentTarget.style.boxShadow = '0 2px 6px rgba(2, 41, 91, 0.3)';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.background = 'var(--color-authority-navy, #02295b)';
                                        e.currentTarget.style.boxShadow = '0 1px 3px rgba(2, 41, 91, 0.2)';
                                    }}
                                >
                                    {showSchedule ? 'Hide Schedule' : 'View Schedule'}
                                </button>
                                <button
                                    onClick={handleDisconnect}
                                    style={{
                                        padding: '8px 16px',
                                        fontSize: '14px',
                                        fontWeight: '500',
                                        borderRadius: '6px',
                                        border: '1px solid rgba(0, 0, 0, 0.1)',
                                        background: 'white',
                                        color: '#6b7280',
                                        cursor: 'pointer',
                                        transition: 'all 0.15s ease',
                                        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)'
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.background = '#f9fafb';
                                        e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.15)';
                                        e.currentTarget.style.color = '#1a1a1a';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.background = 'white';
                                        e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.1)';
                                        e.currentTarget.style.color = '#6b7280';
                                    }}
                                >
                                    Disconnect
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Weekly Schedule View */}
                    {isConnected && showSchedule && calendarService && (
                        <div style={{
                            marginTop: '24px',
                            paddingTop: '24px',
                            borderTop: '1px solid rgba(0, 0, 0, 0.06)',
                            maxHeight: '60vh',
                            overflow: 'hidden',
                            display: 'flex',
                            flexDirection: 'column'
                        }}>
                            <WeeklySchedule 
                                calendarService={calendarService}
                                isConnected={isConnected}
                            />
                        </div>
                    )}

                    {/* Auto-Launch Settings - Notion style */}
                    {isConnected && (
                        <div style={{ 
                            marginTop: '32px', 
                            paddingTop: '28px', 
                            borderTop: '1px solid rgba(0, 0, 0, 0.06)' 
                        }}>
                            <h3 style={{ 
                                marginBottom: '20px',
                                fontSize: '16px',
                                fontWeight: '600',
                                color: '#1a1a1a',
                                letterSpacing: '-0.01em'
                            }}>
                                Auto-Launch Settings
                            </h3>

                        <div style={{ marginBottom: '20px' }}>
                            <label style={{ 
                                display: 'flex', 
                                alignItems: 'center', 
                                gap: '10px',
                                cursor: isConnected ? 'pointer' : 'not-allowed',
                                padding: '8px',
                                borderRadius: '6px',
                                transition: 'background 0.15s',
                                userSelect: 'none'
                            }}
                            onMouseEnter={(e) => {
                                if (isConnected) {
                                    e.currentTarget.style.background = 'rgba(0, 0, 0, 0.02)';
                                }
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = 'transparent';
                            }}
                            >
                                <input
                                    type="checkbox"
                                    checked={autoLaunchEnabled}
                                    onChange={(e) => setAutoLaunchEnabled(e.target.checked)}
                                    disabled={!isConnected}
                                    style={{
                                        width: '18px',
                                        height: '18px',
                                        cursor: isConnected ? 'pointer' : 'not-allowed',
                                        accentColor: 'var(--color-strategic-forest, #2c5f41)'
                                    }}
                                />
                                <span style={{ 
                                    fontSize: '15px',
                                    color: isConnected ? '#1a1a1a' : '#9ca3af',
                                    fontWeight: '400'
                                }}>
                                    Enable auto-launch before meetings
                                </span>
                            </label>
                        </div>

                        {autoLaunchEnabled && (
                            <>
                                <div style={{ marginBottom: '20px' }}>
                                    <label style={{ 
                                        display: 'block', 
                                        marginBottom: '8px',
                                        fontSize: '14px',
                                        fontWeight: '500',
                                        color: '#1a1a1a'
                                    }}>
                                        Launch app X seconds before meeting
                                    </label>
                                    <input
                                        type="number"
                                        value={preLaunchSeconds}
                                        onChange={(e) => setPreLaunchSeconds(parseInt(e.target.value) || 30)}
                                        min="10"
                                        max="300"
                                        style={{ 
                                            width: '100%', 
                                            padding: '10px 12px',
                                            fontSize: '15px',
                                            border: '1px solid rgba(0, 0, 0, 0.1)',
                                            borderRadius: '6px',
                                            background: 'white',
                                            color: '#1a1a1a',
                                            transition: 'all 0.15s'
                                        }}
                                        onFocus={(e) => {
                                            e.currentTarget.style.borderColor = 'var(--color-strategic-forest, #2c5f41)';
                                            e.currentTarget.style.boxShadow = '0 0 0 3px rgba(44, 95, 65, 0.1)';
                                        }}
                                        onBlur={(e) => {
                                            e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.1)';
                                            e.currentTarget.style.boxShadow = 'none';
                                        }}
                                    />
                                    <small style={{ 
                                        display: 'block',
                                        marginTop: '6px',
                                        color: '#9ca3af',
                                        fontSize: '13px'
                                    }}>
                                        Default: 30 seconds
                                    </small>
                                </div>

                                <div style={{ marginBottom: '20px' }}>
                                    <label style={{ 
                                        display: 'block', 
                                        marginBottom: '8px',
                                        fontSize: '14px',
                                        fontWeight: '500',
                                        color: '#1a1a1a'
                                    }}>
                                        Check for meetings every X seconds
                                    </label>
                                    <input
                                        type="number"
                                        value={checkIntervalSeconds}
                                        onChange={(e) => setCheckIntervalSeconds(parseInt(e.target.value) || 60)}
                                        min="30"
                                        max="300"
                                        style={{ 
                                            width: '100%', 
                                            padding: '10px 12px',
                                            fontSize: '15px',
                                            border: '1px solid rgba(0, 0, 0, 0.1)',
                                            borderRadius: '6px',
                                            background: 'white',
                                            color: '#1a1a1a',
                                            transition: 'all 0.15s'
                                        }}
                                        onFocus={(e) => {
                                            e.currentTarget.style.borderColor = 'var(--color-strategic-forest, #2c5f41)';
                                            e.currentTarget.style.boxShadow = '0 0 0 3px rgba(44, 95, 65, 0.1)';
                                        }}
                                        onBlur={(e) => {
                                            e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.1)';
                                            e.currentTarget.style.boxShadow = 'none';
                                        }}
                                    />
                                    <small style={{ 
                                        display: 'block',
                                        marginTop: '6px',
                                        color: '#9ca3af',
                                        fontSize: '13px'
                                    }}>
                                        Default: 60 seconds
                                    </small>
                                </div>
                            </>
                        )}
                        </div>
                    )}
                </div>
                
                <style>{`
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    @keyframes bounce {
                        0%, 100% { transform: translateY(0); }
                        50% { transform: translateY(-10px); }
                    }
                `}</style>

                <div className="modal-footer" style={{ 
                    padding: '20px 24px', 
                    borderTop: '1px solid rgba(0, 0, 0, 0.06)', 
                    display: 'flex', 
                    gap: '10px', 
                    justifyContent: 'flex-end',
                    background: '#fafbfc'
                }}>
                    <button 
                        onClick={onClose}
                        style={{
                            padding: '8px 16px',
                            fontSize: '14px',
                            fontWeight: '500',
                            borderRadius: '6px',
                            border: '1px solid rgba(0, 0, 0, 0.1)',
                            background: 'white',
                            color: '#6b7280',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease',
                            boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = '#f9fafb';
                            e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.15)';
                            e.currentTarget.style.color = '#1a1a1a';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'white';
                            e.currentTarget.style.borderColor = 'rgba(0, 0, 0, 0.1)';
                            e.currentTarget.style.color = '#6b7280';
                        }}
                    >
                        Cancel
                    </button>
                    <button 
                        onClick={handleSaveSettings}
                        style={{
                            padding: '8px 16px',
                            fontSize: '14px',
                            fontWeight: '500',
                            borderRadius: '6px',
                            border: 'none',
                            background: 'var(--color-strategic-forest, #2c5f41)',
                            color: 'white',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease',
                            boxShadow: '0 1px 3px rgba(44, 95, 65, 0.2)'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = '#225a3d';
                            e.currentTarget.style.boxShadow = '0 2px 6px rgba(44, 95, 65, 0.3)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'var(--color-strategic-forest, #2c5f41)';
                            e.currentTarget.style.boxShadow = '0 1px 3px rgba(44, 95, 65, 0.2)';
                        }}
                    >
                        Save Settings
                    </button>
                </div>
            </div>
        </div>
    );
};
