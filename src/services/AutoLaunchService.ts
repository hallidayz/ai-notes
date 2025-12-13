/**
 * Auto-Launch Service
 * Detects upcoming meetings and launches app 30 seconds before start
 */

import { CalendarService, Meeting } from './CalendarService';
import { NotificationService } from './NotificationService';

export interface AutoLaunchConfig {
    enabled: boolean;
    preLaunchSeconds: number; // Default 30
    checkIntervalSeconds: number; // Default 60
    autoStartRecording: boolean; // Optional: auto-start recording
}

export class AutoLaunchService {
    private static instance: AutoLaunchService | null = null;
    private calendarService: CalendarService | null = null;
    private notificationService: NotificationService;
    private checkInterval: number | null = null;
    private upcomingMeetings: Map<string, Meeting> = new Map();
    private notifiedMeetings: Set<string> = new Set();
    private config: AutoLaunchConfig;

    private constructor() {
        this.notificationService = NotificationService.getInstance();
        this.config = {
            enabled: false,
            preLaunchSeconds: 30,
            checkIntervalSeconds: 60,
            autoStartRecording: false
        };
    }

    public static getInstance(): AutoLaunchService {
        if (!this.instance) {
            this.instance = new AutoLaunchService();
        }
        return this.instance;
    }

    /**
     * Initialize with calendar service
     */
    initialize(calendarService: CalendarService | null, config?: Partial<AutoLaunchConfig>): void {
        this.calendarService = calendarService;
        if (config) {
            this.config = { ...this.config, ...config };
        }
    }

    /**
     * Start monitoring for upcoming meetings
     */
    async start(): Promise<void> {
        if (!this.config.enabled) {
            console.log('Auto-launch is disabled');
            return;
        }

        if (!this.calendarService) {
            console.warn('No calendar service configured for auto-launch');
            return;
        }

        // Check immediately
        await this.checkUpcomingMeetings();

        // Then check at intervals
        this.checkInterval = window.setInterval(
            () => this.checkUpcomingMeetings(),
            this.config.checkIntervalSeconds * 1000
        );

        // Notify service worker to start monitoring
        this.notifyServiceWorker('START_MONITORING', {
            checkIntervalSeconds: this.config.checkIntervalSeconds,
            preLaunchSeconds: this.config.preLaunchSeconds,
            autoStartRecording: this.config.autoStartRecording
        });

        console.log('Auto-launch service started', {
            checkInterval: this.config.checkIntervalSeconds,
            preLaunchSeconds: this.config.preLaunchSeconds
        });
    }

    /**
     * Stop monitoring
     */
    stop(): void {
        if (this.checkInterval !== null) {
            clearInterval(this.checkInterval);
            this.checkInterval = null;
        }
        this.upcomingMeetings.clear();
        this.notifiedMeetings.clear();
        
        // Notify service worker to stop monitoring
        this.notifyServiceWorker('STOP_MONITORING');
        
        console.log('Auto-launch service stopped');
    }

    /**
     * Check for upcoming meetings
     */
    private async checkUpcomingMeetings(): Promise<void> {
        if (!this.calendarService) return;

        try {
            const isConnected = await this.calendarService.isConnected();
            if (!isConnected) {
                console.log('Calendar not connected, skipping check');
                return;
            }

            // Fetch meetings for next 24 hours
            const meetings = await this.calendarService.fetchUpcomingMeetings(1);
            
            // Update upcoming meetings map
            meetings.forEach(meeting => {
                this.upcomingMeetings.set(meeting.id, meeting);
            });

            // Check each meeting
            const now = Date.now();
            for (const meeting of meetings) {
                const timeUntilStart = meeting.startTime.getTime() - now;
                const secondsUntilStart = timeUntilStart / 1000;
                const minutesUntilStart = secondsUntilStart / 60;

                // Check if we should launch (30 seconds before)
                if (secondsUntilStart <= this.config.preLaunchSeconds && secondsUntilStart > 0) {
                    await this.handlePreLaunch(meeting, secondsUntilStart);
                }

                // Send notification at 5 minutes, 1 minute, and at start
                if (minutesUntilStart <= 5 && minutesUntilStart > 4.5 && !this.notifiedMeetings.has(`${meeting.id}-5min`)) {
                    await this.notificationService.showMeetingReminder(meeting.title, meeting.startTime, 5);
                    this.notifiedMeetings.add(`${meeting.id}-5min`);
                }

                if (minutesUntilStart <= 1 && minutesUntilStart > 0.5 && !this.notifiedMeetings.has(`${meeting.id}-1min`)) {
                    await this.notificationService.showMeetingReminder(meeting.title, meeting.startTime, 1);
                    this.notifiedMeetings.add(`${meeting.id}-1min`);
                }

                if (secondsUntilStart <= 0 && secondsUntilStart > -60 && !this.notifiedMeetings.has(`${meeting.id}-start`)) {
                    await this.notificationService.showMeetingReminder(meeting.title, meeting.startTime, 0);
                    this.notifiedMeetings.add(`${meeting.id}-start`);
                }
            }

            // Clean up past meetings
            for (const [id, meeting] of this.upcomingMeetings.entries()) {
                if (meeting.endTime.getTime() < now) {
                    this.upcomingMeetings.delete(id);
                }
            }
        } catch (error) {
            console.error('Error checking upcoming meetings:', error);
        }
    }

    /**
     * Handle pre-launch (30 seconds before meeting)
     */
    private async handlePreLaunch(meeting: Meeting, secondsUntilStart: number): Promise<void> {
        const meetingKey = `${meeting.id}-prelaunch`;
        if (this.notifiedMeetings.has(meetingKey)) {
            return; // Already handled
        }

        this.notifiedMeetings.add(meetingKey);

        // Show notification
        await this.notificationService.showMeetingReminder(
            meeting.title,
            meeting.startTime,
            Math.ceil(secondsUntilStart / 60)
        );

        // Launch app (focus window or open new tab)
        this.launchApp(meeting);

        console.log('Auto-launched app for meeting', {
            title: meeting.title,
            startTime: meeting.startTime,
            secondsUntilStart
        });
    }

    /**
     * Launch app (focus or open)
     */
    private launchApp(meeting: Meeting): void {
        // Try to focus existing window
        if (window.focus) {
            window.focus();
        }

        // Dispatch custom event for app to handle
        window.dispatchEvent(new CustomEvent('meetingPreLaunch', {
            detail: {
                meeting,
                action: this.config.autoStartRecording ? 'startRecording' : 'open'
            }
        }));

        // If app is in a new tab/window, this will focus it
        // If app is in same tab, it's already focused
    }

    /**
     * Update configuration
     */
    updateConfig(config: Partial<AutoLaunchConfig>): void {
        const wasRunning = this.checkInterval !== null;
        
        if (wasRunning) {
            this.stop();
        }

        this.config = { ...this.config, ...config };

        if (wasRunning && this.config.enabled) {
            this.start();
        }
    }

    /**
     * Get current configuration
     */
    getConfig(): AutoLaunchConfig {
        return { ...this.config };
    }

    /**
     * Send message to service worker
     */
    private notifyServiceWorker(type: 'START_MONITORING' | 'STOP_MONITORING', config?: any): void {
        if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
            navigator.serviceWorker.controller.postMessage({
                type,
                config
            });
        } else if ('serviceWorker' in navigator) {
            // Service worker might not be ready yet, wait for it
            navigator.serviceWorker.ready.then((registration) => {
                if (registration.active) {
                    registration.active.postMessage({
                        type,
                        config
                    });
                }
            }).catch((error) => {
                console.warn('Failed to send message to service worker:', error);
            });
        }
    }
}
