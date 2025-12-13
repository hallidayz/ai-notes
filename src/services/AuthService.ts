/**
 * Authentication Service
 * Handles user authentication, session management, auto-lock, and security features
 */

export interface AuthConfig {
    autoLockTimeout: number; // milliseconds of inactivity before auto-lock
    maxFailedAttempts: number; // max failed login attempts before lockout
    lockoutDuration: number; // milliseconds to lock out after max attempts
    enableBiometric: boolean; // enable biometric authentication if available
}

export interface AuthState {
    isAuthenticated: boolean;
    isLocked: boolean;
    failedAttempts: number;
    lockoutUntil: number | null;
    lastActivity: number;
}

const DEFAULT_CONFIG: AuthConfig = {
    autoLockTimeout: 5 * 60 * 1000, // 5 minutes
    maxFailedAttempts: 5,
    lockoutDuration: 15 * 60 * 1000, // 15 minutes
    enableBiometric: true,
};

export class AuthService {
    private config: AuthConfig;
    private state: AuthState;
    private inactivityTimer: number | null = null;
    private activityListeners: Array<() => void> = [];
    private lockListeners: Array<(locked: boolean) => void> = [];
    private db: any; // TherapyDB instance

    constructor(db: any, config?: Partial<AuthConfig>) {
        this.db = db;
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.state = {
            isAuthenticated: false,
            isLocked: false,
            failedAttempts: 0,
            lockoutUntil: null,
            lastActivity: Date.now(),
        };
        this.loadAuthState();
        this.setupActivityTracking();
    }

    /**
     * Load authentication state from storage
     */
    private async loadAuthState(): Promise<void> {
        try {
            const savedState = await this.db.getConfig('authState');
            if (savedState) {
                const parsed = JSON.parse(savedState);
                // Check if lockout has expired
                if (parsed.lockoutUntil && parsed.lockoutUntil > Date.now()) {
                    this.state.lockoutUntil = parsed.lockoutUntil;
                    this.state.isLocked = true;
                } else if (parsed.lockoutUntil && parsed.lockoutUntil <= Date.now()) {
                    // Lockout expired, reset
                    this.state.failedAttempts = 0;
                    this.state.lockoutUntil = null;
                    this.state.isLocked = false;
                } else {
                    this.state.failedAttempts = parsed.failedAttempts || 0;
                }
            }
        } catch (e) {
            // No saved state, use defaults
        }
    }

    /**
     * Save authentication state to storage
     */
    private async saveAuthState(): Promise<void> {
        try {
            await this.db.saveConfig('authState', JSON.stringify({
                failedAttempts: this.state.failedAttempts,
                lockoutUntil: this.state.lockoutUntil,
            }));
        } catch (e) {
            console.error('Failed to save auth state:', e);
        }
    }

    /**
     * Setup activity tracking to detect user inactivity
     */
    private setupActivityTracking(): void {
        const events = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click'];
        
        const resetInactivityTimer = () => {
            this.updateActivity();
        };

        events.forEach(event => {
            document.addEventListener(event, resetInactivityTimer, { passive: true });
        });

        // Also track visibility changes (tab switching)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Tab is hidden, don't update activity
            } else {
                // Tab is visible again, update activity
                this.updateActivity();
            }
        });
    }

    /**
     * Update last activity timestamp and reset inactivity timer
     */
    public updateActivity(): void {
        if (!this.state.isAuthenticated || this.state.isLocked) {
            return;
        }

        this.state.lastActivity = Date.now();
        
        // Clear existing timer
        if (this.inactivityTimer !== null) {
            window.clearTimeout(this.inactivityTimer);
        }

        // Set new timer
        this.inactivityTimer = window.setTimeout(() => {
            this.lock();
        }, this.config.autoLockTimeout);
    }

    /**
     * Check if user is currently locked out
     */
    public isLockedOut(): boolean {
        if (this.state.lockoutUntil === null) {
            return false;
        }
        
        if (Date.now() >= this.state.lockoutUntil) {
            // Lockout expired
            this.state.lockoutUntil = null;
            this.state.failedAttempts = 0;
            this.state.isLocked = false;
            this.saveAuthState();
            return false;
        }
        
        return true;
    }

    /**
     * Get remaining lockout time in seconds
     */
    public getLockoutRemaining(): number {
        if (!this.state.lockoutUntil) {
            return 0;
        }
        const remaining = Math.ceil((this.state.lockoutUntil - Date.now()) / 1000);
        return remaining > 0 ? remaining : 0;
    }

    /**
     * Record successful authentication
     */
    public recordSuccess(): void {
        this.state.failedAttempts = 0;
        this.state.lockoutUntil = null;
        this.state.isAuthenticated = true;
        this.state.isLocked = false;
        this.updateActivity();
        this.saveAuthState();
    }

    /**
     * Record failed authentication attempt
     */
    public async recordFailedAttempt(): Promise<{ message: string; isLocked: boolean }> {
        this.state.failedAttempts += 1;
        
        if (this.state.failedAttempts >= this.config.maxFailedAttempts) {
            this.state.lockoutUntil = Date.now() + this.config.lockoutDuration;
            this.state.isLocked = true;
            await this.saveAuthState();
            return {
                message: `Too many failed attempts. Account locked for ${Math.ceil(this.config.lockoutDuration / 60000)} minutes.`,
                isLocked: true,
            };
        }
        
        const remaining = this.config.maxFailedAttempts - this.state.failedAttempts;
        await this.saveAuthState();
        return {
            message: `Invalid PIN. ${remaining} attempt(s) remaining.`,
            isLocked: false,
        };
    }

    /**
     * Check if authentication is allowed (not locked out)
     */
    public canAttemptAuth(): { allowed: boolean; message?: string } {
        if (this.isLockedOut()) {
            const remaining = this.getLockoutRemaining();
            return {
                allowed: false,
                message: `Too many failed attempts. Please try again in ${Math.ceil(remaining / 60)} minute(s).`,
            };
        }
        return { allowed: true };
    }

    /**
     * Lock the app (manual or automatic)
     */
    public lock(): void {
        if (this.inactivityTimer !== null) {
            window.clearTimeout(this.inactivityTimer);
            this.inactivityTimer = null;
        }
        
        this.state.isAuthenticated = false;
        this.state.isLocked = true;
        this.notifyLockListeners(true);
    }

    /**
     * Unlock the app after successful authentication
     */
    public unlock(): void {
        this.state.isAuthenticated = true;
        this.state.isLocked = false;
        this.updateActivity();
        this.notifyLockListeners(false);
    }

    /**
     * Logout user (clears session)
     */
    public logout(): void {
        this.lock();
        this.state.failedAttempts = 0;
        this.state.lockoutUntil = null;
        this.saveAuthState();
    }

    /**
     * Check if biometric authentication is available
     */
    public async isBiometricAvailable(): Promise<boolean> {
        if (!this.config.enableBiometric) {
            return false;
        }

        // Check for WebAuthn API
        if (typeof window !== 'undefined' && 'PublicKeyCredential' in window) {
            try {
                const available = await PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable();
                return available;
            } catch (e) {
                return false;
            }
        }
        
        return false;
    }

    /**
     * Attempt biometric authentication
     */
    public async authenticateWithBiometric(): Promise<{ success: boolean; message?: string }> {
        if (!(await this.isBiometricAvailable())) {
            return { success: false, message: 'Biometric authentication not available.' };
        }

        try {
            // This is a simplified version - in production, you'd want to:
            // 1. Create credentials on first use
            // 2. Store encrypted PIN with biometric
            // 3. Use WebAuthn for authentication
            
            // For now, we'll just return that it's not fully implemented
            // but the infrastructure is there
            return { success: false, message: 'Biometric authentication coming soon.' };
        } catch (e) {
            return { success: false, message: 'Biometric authentication failed.' };
        }
    }

    /**
     * Get current authentication state
     */
    public getState(): Readonly<AuthState> {
        return { ...this.state };
    }

    /**
     * Subscribe to lock state changes
     */
    public onLockStateChange(callback: (locked: boolean) => void): () => void {
        this.lockListeners.push(callback);
        return () => {
            this.lockListeners = this.lockListeners.filter(cb => cb !== callback);
        };
    }

    /**
     * Notify all lock listeners
     */
    private notifyLockListeners(locked: boolean): void {
        this.lockListeners.forEach(callback => {
            try {
                callback(locked);
            } catch (e) {
                console.error('Error in lock listener:', e);
            }
        });
    }

    /**
     * Update configuration
     */
    public updateConfig(config: Partial<AuthConfig>): void {
        this.config = { ...this.config, ...config };
        // Reset inactivity timer with new timeout
        if (this.state.isAuthenticated && !this.state.isLocked) {
            this.updateActivity();
        }
    }

    /**
     * Get current configuration
     */
    public getConfig(): Readonly<AuthConfig> {
        return { ...this.config };
    }

    /**
     * Cleanup - call when service is no longer needed
     */
    public cleanup(): void {
        if (this.inactivityTimer !== null) {
            window.clearTimeout(this.inactivityTimer);
            this.inactivityTimer = null;
        }
        this.activityListeners = [];
        this.lockListeners = [];
    }
}
