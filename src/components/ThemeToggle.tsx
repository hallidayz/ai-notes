
import React from 'react';

interface ThemeToggleProps {
    isDarkMode: boolean;
    onToggle: () => void;
}

export const ThemeToggle: React.FC<ThemeToggleProps> = ({ isDarkMode, onToggle }) => (
    <button className="theme-toggle" onClick={onToggle} aria-label="Toggle theme">
        {isDarkMode ? '☀️' : '🌙'}
    </button>
);
