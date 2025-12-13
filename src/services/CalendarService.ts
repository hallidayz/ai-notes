/**
 * Calendar Service - Base interface for calendar integrations
 * Supports Google Calendar, Outlook, and Apple Calendar
 */

export type CalendarProvider = 'google' | 'outlook' | 'apple';

export interface Meeting {
    id: string;
    title: string;
    startTime: Date;
    endTime: Date;
    participants: string[];
    location?: string;
    description?: string;
    platform?: 'zoom' | 'teams' | 'meet' | 'other' | 'in-person';
    calendarProvider: CalendarProvider;
    calendarEventId?: string;
}

export interface CalendarCredentials {
    provider: CalendarProvider;
    accessToken?: string;
    refreshToken?: string;
    expiresAt?: number;
    [key: string]: any; // Allow provider-specific fields
}

export interface CalendarConfig {
    enabled: boolean;
    provider?: CalendarProvider;
    credentials?: CalendarCredentials;
    autoLaunchEnabled?: boolean;
    preLaunchSeconds?: number; // Default 30
    checkIntervalSeconds?: number; // Default 60
}

/**
 * Base Calendar Service Interface
 */
export abstract class CalendarService {
    protected provider: CalendarProvider;
    protected credentials: CalendarCredentials | null = null;

    constructor(provider: CalendarProvider) {
        this.provider = provider;
    }

    /**
     * Connect to calendar provider (OAuth flow)
     */
    abstract connect(): Promise<void>;

    /**
     * Disconnect from calendar provider
     */
    abstract disconnect(): Promise<void>;

    /**
     * Check if connected
     */
    abstract isConnected(): Promise<boolean>;

    /**
     * Fetch upcoming meetings
     */
    abstract fetchUpcomingMeetings(daysAhead?: number): Promise<Meeting[]>;

    /**
     * Get meeting details by ID
     */
    abstract getMeetingById(meetingId: string): Promise<Meeting | null>;

    /**
     * Get meeting participants
     */
    abstract getMeetingParticipants(meetingId: string): Promise<string[]>;

    /**
     * Save credentials securely
     */
    protected async saveCredentials(credentials: CalendarCredentials): Promise<void> {
        // Encrypt credentials before storing
        const encrypted = await this.encryptCredentials(credentials);
        localStorage.setItem(`calendar_${this.provider}_creds`, JSON.stringify(encrypted));
        this.credentials = credentials;
    }

    /**
     * Load credentials
     */
    protected async loadCredentials(): Promise<CalendarCredentials | null> {
        try {
            const stored = localStorage.getItem(`calendar_${this.provider}_creds`);
            if (!stored) return null;
            const encrypted = JSON.parse(stored);
            const credentials = await this.decryptCredentials(encrypted);
            this.credentials = credentials;
            return credentials;
        } catch {
            return null;
        }
    }

    /**
     * Encrypt credentials (simple base64 for now - should use proper encryption)
     */
    private async encryptCredentials(credentials: CalendarCredentials): Promise<any> {
        // In production, use Web Crypto API with user's PIN
        // For now, use base64 encoding (not secure, but works)
        const json = JSON.stringify(credentials);
        return { encrypted: btoa(json) };
    }

    /**
     * Decrypt credentials
     */
    private async decryptCredentials(encrypted: any): Promise<CalendarCredentials> {
        const json = atob(encrypted.encrypted);
        return JSON.parse(json);
    }

    /**
     * Check if token is expired
     */
    protected isTokenExpired(expiresAt?: number): boolean {
        if (!expiresAt) return false;
        return Date.now() >= expiresAt;
    }

    /**
     * Refresh access token if needed
     */
    abstract refreshTokenIfNeeded(): Promise<void>;
}
