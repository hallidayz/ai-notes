
import React, { useState } from 'react';
import { ThemeToggle } from './ThemeToggle';
import { AppIcon } from './AppIcon';
import { BRAND } from '../branding';

interface AuthScreenProps {
    onAuthenticate: (pin: string) => void;
    isDarkMode: boolean;
    onToggleTheme: () => void;
}

export const AuthScreen: React.FC<AuthScreenProps> = ({ onAuthenticate, isDarkMode, onToggleTheme }) => {
    const [pin, setPin] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (pin.length >= 4) {
            onAuthenticate(pin);
        }
    };

    return (
        <div className="auth-screen">
            <div className="auth-card">
                <div className="header-actions">
                    <ThemeToggle isDarkMode={isDarkMode} onToggle={onToggleTheme} />
                </div>
                <div className="brand-lockup brand-lockup-centered">
                    <AppIcon name="logo" size={48} isDarkMode={isDarkMode} className="brand-logo" />
                    <div>
                        <h1>{BRAND.name}</h1>
                        <p>Secure, on-device AI transcription and analysis.</p>
                    </div>
                </div>
                <form onSubmit={handleSubmit} autoComplete="off">
                    <label htmlFor="app-pin" className="visually-hidden">PIN</label>
                    <input
                        id="app-pin"
                        name="pin"
                        type="password"
                        inputMode="numeric"
                        placeholder="Enter your PIN"
                        value={pin}
                        onChange={e => setPin(e.target.value)}
                        autoComplete="current-password"
                        autoFocus
                    />
                    <button type="submit" className="btn-primary" disabled={pin.length < 4}>
                        Unlock
                    </button>
                </form>
                <p className="auth-note">Your PIN is used to encrypt your data locally. We never see it.</p>
            </div>
        </div>
    );
};
