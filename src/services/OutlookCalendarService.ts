/**
 * Outlook Calendar Service Implementation
 * Uses Microsoft Graph API with OAuth 2.0
 */

import { CalendarService, Meeting, CalendarCredentials, CalendarProvider } from './CalendarService';

export class OutlookCalendarService extends CalendarService {
    private clientId: string;
    private redirectUri: string;
    private scopes: string[] = ['Calendars.Read'];

    constructor(clientId?: string) {
        super('outlook');
        this.clientId = clientId || '';
        this.redirectUri = typeof window !== 'undefined'
            ? `${window.location.origin}/oauth/outlook/callback`
            : '';
    }

    /**
     * Connect to Outlook Calendar via OAuth 2.0
     */
    async connect(): Promise<void> {
        if (!this.clientId) {
            throw new Error('Outlook Calendar client ID not configured. Please set up Azure App Registration.');
        }

        // Check if already connected
        const existing = await this.loadCredentials();
        if (existing && !this.isTokenExpired(existing.expiresAt)) {
            this.credentials = existing;
            return;
        }

        // Start OAuth flow
        const authUrl = this.buildAuthUrl();
        window.location.href = authUrl;
    }

    /**
     * Build OAuth authorization URL for Microsoft
     */
    private buildAuthUrl(): string {
        const params = new URLSearchParams({
            client_id: this.clientId,
            redirect_uri: this.redirectUri,
            response_type: 'code',
            scope: this.scopes.join(' '),
            response_mode: 'query',
            state: this.generateState()
        });

        return `https://login.microsoftonline.com/common/oauth2/v2.0/authorize?${params.toString()}`;
    }

    /**
     * Generate state for OAuth security
     */
    private generateState(): string {
        return btoa(JSON.stringify({
            provider: 'outlook',
            timestamp: Date.now(),
            random: Math.random().toString(36)
        }));
    }

    /**
     * Handle OAuth callback
     */
    async handleCallback(code: string): Promise<void> {
        try {
            const tokens = await this.exchangeCodeForTokens(code);
            
            const credentials: CalendarCredentials = {
                provider: 'outlook',
                accessToken: tokens.access_token,
                refreshToken: tokens.refresh_token,
                expiresAt: Date.now() + (tokens.expires_in * 1000)
            };

            await this.saveCredentials(credentials);
        } catch (error) {
            console.error('Outlook Calendar OAuth error:', error);
            throw new Error('Failed to connect to Outlook Calendar');
        }
    }

    /**
     * Exchange authorization code for tokens
     * NOTE: This should be done server-side in production!
     */
    private async exchangeCodeForTokens(code: string): Promise<any> {
        const response = await fetch('/api/oauth/outlook/token', {
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
     * Disconnect from Outlook Calendar
     */
    async disconnect(): Promise<void> {
        localStorage.removeItem('calendar_outlook_creds');
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
            const response = await fetch('/api/oauth/outlook/refresh', {
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
            await this.disconnect();
            throw error;
        }
    }

    /**
     * Fetch upcoming meetings
     */
    async fetchUpcomingMeetings(daysAhead: number = 7): Promise<Meeting[]> {
        if (!await this.isConnected()) {
            throw new Error('Not connected to Outlook Calendar');
        }

        const startDateTime = new Date().toISOString();
        const endDateTime = new Date(Date.now() + daysAhead * 24 * 60 * 60 * 1000).toISOString();

        const url = new URL('https://graph.microsoft.com/v1.0/me/calendarview');
        url.searchParams.set('startDateTime', startDateTime);
        url.searchParams.set('endDateTime', endDateTime);
        url.searchParams.set('$orderby', 'start/dateTime');
        url.searchParams.set('$top', '50');

        const response = await fetch(url.toString(), {
            headers: {
                'Authorization': `Bearer ${this.credentials!.accessToken}`,
                'Prefer': 'outlook.timezone="UTC"'
            }
        });

        if (!response.ok) {
            if (response.status === 401) {
                await this.refreshTokenIfNeeded();
                return this.fetchUpcomingMeetings(daysAhead);
            }
            throw new Error(`Failed to fetch meetings: ${response.statusText}`);
        }

        const data = await response.json();
        return this.mapEventsToMeetings(data.value || []);
    }

    /**
     * Map Microsoft Graph events to Meeting objects
     */
    private mapEventsToMeetings(events: any[]): Meeting[] {
        return events
            .filter(event => event.start && event.start.dateTime) // Only timed events
            .map(event => {
                const startTime = new Date(event.start.dateTime);
                const endTime = new Date(event.end.dateTime);
                
                // Extract participants from attendees
                const participants = (event.attendees || [])
                    .map((a: any) => a.emailAddress?.address || a.emailAddress?.name)
                    .filter(Boolean);

                // Detect meeting platform
                const platform = this.detectPlatform(event.location?.displayName, event.body?.content, event.onlineMeeting);

                return {
                    id: event.id,
                    title: event.subject || 'Untitled Meeting',
                    startTime,
                    endTime,
                    participants,
                    location: event.location?.displayName,
                    description: event.body?.content,
                    platform,
                    calendarProvider: 'outlook',
                    calendarEventId: event.id
                } as Meeting;
            });
    }

    /**
     * Detect meeting platform
     */
    private detectPlatform(location?: string, description?: string, onlineMeeting?: any): Meeting['platform'] {
        const text = `${location || ''} ${description || ''}`.toLowerCase();
        
        if (onlineMeeting) {
            if (text.includes('zoom')) return 'zoom';
            if (text.includes('teams')) return 'teams';
            return 'teams'; // Default for Microsoft online meetings
        }
        
        if (text.includes('zoom.us') || text.includes('zoom.com')) return 'zoom';
        if (text.includes('teams.microsoft.com') || text.includes('teams')) return 'teams';
        if (text.includes('meet.google.com') || text.includes('google meet')) return 'meet';
        if (text.includes('in-person') || text.includes('office') || text.includes('room')) return 'in-person';
        
        return 'other';
    }

    /**
     * Get meeting by ID
     */
    async getMeetingById(meetingId: string): Promise<Meeting | null> {
        if (!await this.isConnected()) {
            throw new Error('Not connected to Outlook Calendar');
        }

        const url = `https://graph.microsoft.com/v1.0/me/events/${meetingId}`;
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
