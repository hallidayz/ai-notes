/**
 * OAuth Callback Component
 * Handles OAuth redirects from Google and Outlook calendar providers
 */

import React, { useEffect, useState } from 'react';
import { GoogleCalendarService } from '../services/GoogleCalendarService';
import { OutlookCalendarService } from '../services/OutlookCalendarService';
import { CalendarProvider } from '../services/CalendarService';

interface OAuthCallbackProps {
    provider: CalendarProvider;
}

export const OAuthCallback: React.FC<OAuthCallbackProps> = ({ provider }) => {
    const [status, setStatus] = useState<'processing' | 'success' | 'error'>('processing');
    const [message, setMessage] = useState('Processing OAuth callback...');

    useEffect(() => {
        const handleCallback = async () => {
            try {
                // Get code and state from URL
                const urlParams = new URLSearchParams(window.location.search);
                const code = urlParams.get('code');
                const error = urlParams.get('error');
                const state = urlParams.get('state');

                if (error) {
                    setStatus('error');
                    setMessage(`OAuth error: ${error}. Please try again.`);
                    
                    // Post error to opener window (popup)
                    if (window.opener) {
                        window.opener.postMessage({ 
                            type: 'oauth-error', 
                            error: error
                        }, window.location.origin);
                    }
                    
                    setTimeout(() => {
                        window.close();
                    }, 2000);
                    return;
                }

                if (!code) {
                    setStatus('error');
                    setMessage('No authorization code received. Please try again.');
                    
                    // Post error to opener window (popup)
                    if (window.opener) {
                        window.opener.postMessage({ 
                            type: 'oauth-error', 
                            error: 'No authorization code received'
                        }, window.location.origin);
                    }
                    
                    setTimeout(() => {
                        window.close();
                    }, 2000);
                    return;
                }

                // Get the appropriate service
                let service: GoogleCalendarService | OutlookCalendarService;
                if (provider === 'google') {
                    service = new GoogleCalendarService();
                } else {
                    service = new OutlookCalendarService();
                }

                // Handle the callback
                await service.handleCallback(code);

                setStatus('success');
                setMessage(`Successfully connected to ${provider === 'google' ? 'Google Calendar' : 'Outlook Calendar'}!`);

                // Post message to opener window (popup)
                if (window.opener) {
                    window.opener.postMessage({ 
                        type: 'oauth-success', 
                        provider,
                        code: code 
                    }, window.location.origin);
                }
                
                // Close window
                setTimeout(() => {
                    window.close();
                }, 1000);
            } catch (error: any) {
                console.error('OAuth callback error:', error);
                setStatus('error');
                setMessage(error.message || 'Failed to complete OAuth flow. Please try again.');
                
                // Post error to opener window (popup)
                if (window.opener) {
                    window.opener.postMessage({ 
                        type: 'oauth-error', 
                        error: error.message || 'Authentication failed'
                    }, window.location.origin);
                }
                
                setTimeout(() => {
                    window.close();
                }, 2000);
            }
        };

        handleCallback();
    }, [provider]);

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            padding: '20px',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            background: 'var(--color-pure-foundation, #f6f7f9)',
            color: 'var(--color-authority-navy, #02295b)'
        }}>
            <div style={{
                background: 'white',
                padding: '40px',
                borderRadius: '12px',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                maxWidth: '400px',
                width: '100%',
                textAlign: 'center'
            }}>
                {status === 'processing' && (
                    <>
                        <div className="spinner" style={{
                            width: '48px',
                            height: '48px',
                            border: '4px solid rgba(2, 41, 91, 0.1)',
                            borderTop: '4px solid var(--color-achievement-gold, #fda700)',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite',
                            margin: '0 auto 20px'
                        }} />
                        <h2 style={{ margin: '0 0 10px 0', color: 'var(--color-authority-navy, #02295b)' }}>
                            Connecting...
                        </h2>
                        <p style={{ margin: 0, color: '#666' }}>{message}</p>
                    </>
                )}

                {status === 'success' && (
                    <>
                        <div style={{
                            width: '64px',
                            height: '64px',
                            borderRadius: '50%',
                            background: 'var(--color-strategic-forest, #2c5f41)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            margin: '0 auto 20px',
                            fontSize: '32px'
                        }}>
                            ✓
                        </div>
                        <h2 style={{ margin: '0 0 10px 0', color: 'var(--color-strategic-forest, #2c5f41)' }}>
                            Success!
                        </h2>
                        <p style={{ margin: 0, color: '#666' }}>{message}</p>
                        <p style={{ margin: '10px 0 0 0', fontSize: '14px', color: '#999' }}>
                            This window will close automatically...
                        </p>
                    </>
                )}

                {status === 'error' && (
                    <>
                        <div style={{
                            width: '64px',
                            height: '64px',
                            borderRadius: '50%',
                            background: '#ef4444',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            margin: '0 auto 20px',
                            fontSize: '32px',
                            color: 'white'
                        }}>
                            ✕
                        </div>
                        <h2 style={{ margin: '0 0 10px 0', color: '#ef4444' }}>
                            Error
                        </h2>
                        <p style={{ margin: 0, color: '#666' }}>{message}</p>
                        <p style={{ margin: '10px 0 0 0', fontSize: '14px', color: '#999' }}>
                            This window will close automatically...
                        </p>
                    </>
                )}
            </div>

            <style>{`
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
};
