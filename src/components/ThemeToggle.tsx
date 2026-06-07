
import React from 'react';
import { AppIcon } from './AppIcon';

interface ThemeToggleProps {
    isDarkMode: boolean;
    onToggle: () => void;
}

export const ThemeToggle: React.FC<ThemeToggleProps> = ({ isDarkMode, onToggle }) => (
    <button className="theme-toggle" onClick={onToggle} aria-label="Toggle theme">
        <AppIcon name={isDarkMode ? 'sun' : 'moon'} size={20} isDarkMode={isDarkMode} />
    </button>
);
