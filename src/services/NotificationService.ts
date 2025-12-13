/**
 * Notification Service
 * Handles browser notifications for meeting reminders
 */

export class NotificationService {
    private static instance: NotificationService | null = null;
    private permission: NotificationPermission = 'default';

    private constructor() {
        if (typeof window !== 'undefined' && 'Notification' in window) {
            this.permission = Notification.permission;
        }
    }

    public static getInstance(): NotificationService {
        if (!this.instance) {
            this.instance = new NotificationService();
        }
        return this.instance;
    }

    /**
     * Request notification permission
     */
    async requestPermission(): Promise<NotificationPermission> {
        if (typeof window === 'undefined' || !('Notification' in window)) {
            console.warn('Notifications not supported in this browser');
            return 'denied';
        }

        if (this.permission === 'granted') {
            return 'granted';
        }

        if (this.permission === 'denied') {
            return 'denied';
        }

        const permission = await Notification.requestPermission();
        this.permission = permission;
        return permission;
    }

    /**
     * Check if notifications are allowed
     */
    isAllowed(): boolean {
        return this.permission === 'granted';
    }

    /**
     * Show notification
     */
    async showNotification(title: string, options?: NotificationOptions): Promise<void> {
        if (!this.isAllowed()) {
            const permission = await this.requestPermission();
            if (permission !== 'granted') {
                console.warn('Notification permission denied');
                return;
            }
        }

        if (typeof window === 'undefined' || !('Notification' in window)) {
            console.warn('Notifications not supported');
            return;
        }

        const notification = new Notification(title, {
            icon: '/icon-192.png',
            badge: '/icon-192.png',
            tag: 'meeting-reminder',
            requireInteraction: false,
            ...options
        });

        // Auto-close after 10 seconds
        setTimeout(() => {
            notification.close();
        }, 10000);

        // Handle click
        notification.onclick = () => {
            window.focus();
            notification.close();
        };
    }

    /**
     * Show meeting reminder notification
     */
    async showMeetingReminder(meetingTitle: string, startTime: Date, minutesUntil: number): Promise<void> {
        const timeStr = startTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const message = minutesUntil === 0 
            ? `Starting now at ${timeStr}`
            : `Starting in ${minutesUntil} minute${minutesUntil !== 1 ? 's' : ''} at ${timeStr}`;

        await this.showNotification(meetingTitle, {
            body: message,
            tag: `meeting-${startTime.getTime()}`,
            requireInteraction: minutesUntil === 0, // Require interaction if starting now
            data: {
                type: 'meeting-reminder',
                startTime: startTime.toISOString()
            }
        });
    }

    /**
     * Close all notifications
     */
    closeAll(): void {
        // Notifications auto-close, but we can track them if needed
        // Service Worker notifications can be closed programmatically
    }
}
