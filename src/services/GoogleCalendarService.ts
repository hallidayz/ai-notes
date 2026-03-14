/**
 * Google Calendar Service Implementation
 * Uses Google Calendar API v3 with OAuth 2.0
 */

import { CalendarService, Meeting, CalendarCredentials, CalendarProvider } from './CalendarService';

export class GoogleCalendarService extends CalendarService {
    private clientId: string;
    private redirectUri: string;
    private scopes: string[] = ['https://www.googleapis.com/auth/calendar.readonly'];

    constructor(clientId?: string) {
        super('google');
        // Get client ID from parameter, localStorage, or environment variable
        this.clientId = clientId || 
            (typeof window !== 'undefined' ? localStorage.getItem('oauth_google_client_id') : null) ||
            import.meta.env.VITE_GOOGLE_CLIENT_ID || 
            '';
        this.redirectUri = typeof window !== 'undefined' 
            ? `${window.location.origin}/oauth/google/callback`
            : '';
    }

    /**
     * Connect to Google Calendar via OAuth 2.0 using popup
     */
    async connect(): Promise<void> {
        if (!this.clientId) {
            throw new Error('Google Calendar client ID not configured. Please contact support.');
        }

        // Check if already connected
        const existing = await this.loadCredentials();
        if (existing && !this.isTokenExpired(existing.expiresAt)) {
            this.credentials = existing;
            return;
        }

        // Start OAuth flow with popup
        return new Promise((resolve, reject) => {
            const authUrl = this.buildAuthUrl();
            const width = 500;
            const height = 600;
            const left = (window.screen.width - width) / 2;
            const top = (window.screen.height - height) / 2;
            
            const popup = window.open(
                authUrl,
                'Google Calendar Login',
                `width=${width},height=${height},left=${left},top=${top},toolbar=no,menubar=no,scrollbars=yes,resizable=yes`
            );

            if (!popup) {
                reject(new Error('Popup blocked. Please allow popups for this site.'));
                return;
            }

            // Listen for postMessage from callback page
            let resolved = false;
            const messageHandler = (event: MessageEvent) => {
                // Verify origin for security
                if (event.origin !== window.location.origin) {
                    return;
                }

                if (event.data.type === 'oauth-success' && event.data.provider === 'google') {
                    resolved = true;
                    window.removeEventListener('message', messageHandler);
                    clearInterval(checkPopup);
                    if (!popup.closed) {
                        popup.close();
                    }
                    
                    const code = event.data.code;
                    if (code) {
                        this.handleCallback(code)
                            .then(() => resolve())
                            .catch(reject);
                    } else {
                        reject(new Error('No authorization code received'));
                    }
                } else if (event.data.type === 'oauth-error') {
                    resolved = true;
                    window.removeEventListener('message', messageHandler);
                    clearInterval(checkPopup);
                    if (!popup.closed) {
                        popup.close();
                    }
                    reject(new Error(event.data.error || 'Authentication failed'));
                }
            };

            window.addEventListener('message', messageHandler);

            // Also check if popup was closed manually (before receiving message)
            const checkPopup = setInterval(() => {
                if (popup.closed && !resolved) {
                    resolved = true;
                    clearInterval(checkPopup);
                    window.removeEventListener('message', messageHandler);
                    reject(new Error('Authentication cancelled'));
                }
            }, 100);

            // Timeout after 5 minutes
            setTimeout(() => {
                if (!popup.closed) {
                    popup.close();
                }
                clearInterval(checkPopup);
                reject(new Error('Authentication timeout'));
            }, 300000);
        });
    }

    /**
     * Build OAuth authorization URL
     */
    private buildAuthUrl(): string {
        const params = new URLSearchParams({
            client_id: this.clientId,
            redirect_uri: this.redirectUri,
            response_type: 'code',
            scope: this.scopes.join(' '),
            access_type: 'offline',
            prompt: 'consent',
            state: this.generateState()
        });

        return `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
    }

    /**
     * Generate state for OAuth security
     */
    private generateState(): string {
        // Generate a cryptographically secure random string
        const randomBytes = new Uint8Array(16);
        window.crypto.getRandomValues(randomBytes);
        const randomString = Array.from(randomBytes)
            .map(b => b.toString(16).padStart(2, '0'))
            .join('');

        return btoa(JSON.stringify({
            provider: 'google',
            timestamp: Date.now(),
            random: randomString
        }));
    }

    /**
     * Handle OAuth callback
     */
    async handleCallback(code: string): Promise<void> {
        try {
            // Exchange code for tokens
            // Note: In production, this should be done server-side for security
            // For now, we'll use a proxy endpoint or handle it client-side
            const tokens = await this.exchangeCodeForTokens(code);
            
            const credentials: CalendarCredentials = {
                provider: 'google',
                accessToken: tokens.access_token,
                refreshToken: tokens.refresh_token,
                expiresAt: Date.now() + (tokens.expires_in * 1000)
            };

            await this.saveCredentials(credentials);
        } catch (error) {
            console.error('Google Calendar OAuth error:', error);
            throw new Error('Failed to connect to Google Calendar');
        }
    }

    /**
     * Exchange authorization code for tokens
     * NOTE: This should be done server-side in production!
     */
    private async exchangeCodeForTokens(code: string): Promise<any> {
        // Use proxy in development, full URL in production
        const apiUrl = import.meta.env.DEV 
            ? '/api/oauth/google/token'  // Vite proxy handles this
            : `${import.meta.env.VITE_API_URL || 'http://localhost:4000'}/api/oauth/google/token`;
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, redirect_uri: this.redirectUri })
        });

        if (!response.ok) {
            throw new Error('Failed to exchange code for tokens');
        }

        return response.json();
    }

    /**
     * Disconnect from Google Calendar
     */
    async disconnect(): Promise<void> {
        localStorage.removeItem('calendar_google_creds');
        this.credentials = null;
    }

    /**
     * Check if connected
     */
    async isConnected(): Promise<boolean> {
        const creds = await this.loadCredentials();
        if (!creds || !creds.accessToken) return false;
        
        if (this.isTokenExpired(creds.expiresAt)) {
            await this.refreshTokenIfNeeded();
        }
        
        return !!this.credentials?.accessToken;
    }

    /**
     * Refresh access token
     */
    async refreshTokenIfNeeded(): Promise<void> {
        if (!this.credentials?.refreshToken) {
            throw new Error('No refresh token available');
        }

        try {
            // Use proxy in development, full URL in production
            const apiUrl = import.meta.env.DEV 
                ? '/api/oauth/google/refresh'  // Vite proxy handles this
                : `${import.meta.env.VITE_API_URL || 'http://localhost:4000'}/api/oauth/google/refresh`;
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_token: this.credentials.refreshToken })
            });

            if (!response.ok) {
                throw new Error('Failed to refresh token');
            }

            const tokens = await response.json();
            this.credentials.accessToken = tokens.access_token;
            this.credentials.expiresAt = Date.now() + (tokens.expires_in * 1000);
            
            if (tokens.refresh_token) {
                this.credentials.refreshToken = tokens.refresh_token;
            }

            await this.saveCredentials(this.credentials);
        } catch (error) {
            console.error('Token refresh error:', error);
            // If refresh fails, user needs to reconnect
            await this.disconnect();
            throw error;
        }
    }

    /**
     * Fetch upcoming meetings
     */
    async fetchUpcomingMeetings(daysAhead: number = 7): Promise<Meeting[]> {
        if (!await this.isConnected()) {
            throw new Error('Not connected to Google Calendar');
        }

        const timeMin = new Date().toISOString();
        const timeMax = new Date(Date.now() + daysAhead * 24 * 60 * 60 * 1000).toISOString();

        const url = new URL('https://www.googleapis.com/calendar/v3/calendars/primary/events');
        url.searchParams.set('timeMin', timeMin);
        url.searchParams.set('timeMax', timeMax);
        url.searchParams.set('singleEvents', 'true');
        url.searchParams.set('orderBy', 'startTime');
        url.searchParams.set('maxResults', '50');

        const response = await fetch(url.toString(), {
            headers: {
                'Authorization': `Bearer ${this.credentials!.accessToken}`
            }
        });

        if (!response.ok) {
            if (response.status === 401) {
                // Token expired, try to refresh
                await this.refreshTokenIfNeeded();
                return this.fetchUpcomingMeetings(daysAhead);
            }
            throw new Error(`Failed to fetch meetings: ${response.statusText}`);
        }

        const data = await response.json();
        return this.mapEventsToMeetings(data.items || []);
    }

    /**
     * Map Google Calendar events to Meeting objects
     */
    private mapEventsToMeetings(events: any[]): Meeting[] {
        return events
            .filter(event => event.start && !event.start.date) // Only timed events
            .map(event => {
                const startTime = new Date(event.start.dateTime);
                const endTime = new Date(event.end.dateTime);
                
                // Extract participants from attendees
                const participants = (event.attendees || [])
                    .map((a: any) => a.email || a.displayName)
                    .filter(Boolean);

                // Detect meeting platform from location/description
                const platform = this.detectPlatform(event.location, event.description, event.hangoutLink);

                return {
                    id: event.id,
                    title: event.summary || 'Untitled Meeting',
                    startTime,
                    endTime,
                    participants,
                    location: event.location,
                    description: event.description,
                    platform,
                    calendarProvider: 'google',
                    calendarEventId: event.id
                } as Meeting;
            });
    }

    /**
     * Detect meeting platform from location/description
     */
    private detectPlatform(location?: string, description?: string, hangoutLink?: string): Meeting['platform'] {
        const text = `${location || ''} ${description || ''} ${hangoutLink || ''}`.toLowerCase();
        
        if (text.includes('zoom.us') || text.includes('zoom.com')) return 'zoom';
        if (text.includes('teams.microsoft.com') || text.includes('teams')) return 'teams';
        if (text.includes('meet.google.com') || text.includes('google meet') || hangoutLink) return 'meet';
        if (text.includes('in-person') || text.includes('office') || text.includes('room')) return 'in-person';
        
        return 'other';
    }

    /**
     * Get meeting by ID
     */
    async getMeetingById(meetingId: string): Promise<Meeting | null> {
        if (!await this.isConnected()) {
            throw new Error('Not connected to Google Calendar');
        }

        const url = `https://www.googleapis.com/calendar/v3/calendars/primary/events/${meetingId}`;
        const response = await fetch(url, {
            headers: {
                'Authorization': `Bearer ${this.credentials!.accessToken}`
            }
        });

        if (!response.ok) {
            if (response.status === 404) return null;
            throw new Error(`Failed to fetch meeting: ${response.statusText}`);
        }

        const event = await response.json();
        const meetings = this.mapEventsToMeetings([event]);
        return meetings[0] || null;
    }

    /**
     * Get meeting participants
     */
    async getMeetingParticipants(meetingId: string): Promise<string[]> {
        const meeting = await this.getMeetingById(meetingId);
        return meeting?.participants || [];
    }
}
