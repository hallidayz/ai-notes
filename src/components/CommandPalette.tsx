/**
 * Command Palette Component
 * Floating AI command interface triggered by keyboard shortcut (Cmd/Ctrl+K)
 */

import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, Bot, FileText } from 'lucide-react';

interface CommandItem {
    icon: React.ReactNode;
    title: string;
    shortcut?: string;
    action: () => void;
}

interface CommandPaletteProps {
    isOpen: boolean;
    onClose: () => void;
    onSummarize?: () => void;
    onActionItems?: () => void;
    onNewCommand?: (command: string) => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
    isOpen,
    onClose,
    onSummarize,
    onActionItems,
    onNewCommand
}) => {
    const [searchQuery, setSearchQuery] = useState('');
    const [activeIndex, setActiveIndex] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);

    const commands: CommandItem[] = [
        {
            icon: <Bot size={16} />,
            title: 'Summarize this note',
            shortcut: '↵',
            action: () => {
                onSummarize?.();
                onClose();
            }
        },
        {
            icon: <FileText size={16} />,
            title: 'Turn into action items',
            action: () => {
                onActionItems?.();
                onClose();
            }
        }
    ];

    const filteredCommands = commands.filter(cmd =>
        cmd.title.toLowerCase().includes(searchQuery.toLowerCase())
    );

    useEffect(() => {
        if (isOpen && inputRef.current) {
            inputRef.current.focus();
            setSearchQuery('');
            setActiveIndex(0);
        }
    }, [isOpen]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!isOpen) return;

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                setActiveIndex(prev => (prev + 1) % filteredCommands.length);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                setActiveIndex(prev => (prev - 1 + filteredCommands.length) % filteredCommands.length);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (filteredCommands[activeIndex]) {
                    filteredCommands[activeIndex].action();
                } else if (searchQuery.trim() && onNewCommand) {
                    onNewCommand(searchQuery.trim());
                    onClose();
                }
            } else if (e.key === 'Escape') {
                e.preventDefault();
                onClose();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, filteredCommands, activeIndex, searchQuery, onClose, onNewCommand]);

    if (!isOpen) return null;

    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';

    return (
        <div
            style={{
                position: 'fixed',
                top: '33%',
                left: '50%',
                transform: 'translateX(-50%)',
                width: '100%',
                maxWidth: '600px',
                zIndex: 1000,
                padding: '0 16px'
            }}
            onClick={(e) => e.stopPropagation()}
        >
            <div
                style={{
                    background: isDarkMode ? 'rgba(27, 52, 72, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                    backdropFilter: 'blur(12px)',
                    borderRadius: '12px',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                    border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.1)',
                    overflow: 'hidden',
                    ring: '1px solid rgba(0, 0, 0, 0.05)'
                }}
            >
                <div
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        padding: '12px 16px',
                        borderBottom: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.06)'
                    }}
                >
                    <Sparkles
                        size={18}
                        style={{
                            color: '#fda700',
                            marginRight: '12px',
                            animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite'
                        }}
                    />
                    <input
                        ref={inputRef}
                        type="text"
                        placeholder="Ask AI to..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            outline: 'none',
                            color: isDarkMode ? '#e2e8f0' : '#1a1a1a',
                            width: '100%',
                            fontSize: '15px',
                            padding: 0
                        }}
                    />
                    <span
                        style={{
                            fontSize: '12px',
                            color: isDarkMode ? '#9ca3af' : '#6b7280',
                            background: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : '#fff',
                            padding: '2px 8px',
                            borderRadius: '4px',
                            boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                            border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.1)',
                            marginLeft: '8px'
                        }}
                    >
                        Esc
                    </span>
                </div>
                <div style={{ padding: '4px 0' }}>
                    {filteredCommands.length > 0 ? (
                        filteredCommands.map((command, index) => (
                            <button
                                key={index}
                                onClick={command.action}
                                onMouseEnter={() => setActiveIndex(index)}
                                style={{
                                    width: '100%',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    padding: '12px 16px',
                                    fontSize: '14px',
                                    background: index === activeIndex
                                        ? (isDarkMode ? 'rgba(253, 167, 0, 0.1)' : 'rgba(2, 41, 91, 0.08)')
                                        : 'transparent',
                                    border: 'none',
                                    cursor: 'pointer',
                                    transition: 'all 0.15s',
                                    textAlign: 'left'
                                }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                    <span
                                        style={{
                                            color: index === activeIndex
                                                ? (isDarkMode ? '#fda700' : 'var(--color-authority-navy, #02295b)')
                                                : (isDarkMode ? '#9ca3af' : '#6b7280')
                                        }}
                                    >
                                        {command.icon}
                                    </span>
                                    <span
                                        style={{
                                            color: index === activeIndex
                                                ? (isDarkMode ? '#e2e8f0' : '#1a1a1a')
                                                : (isDarkMode ? '#cbd5e1' : '#6b7280'),
                                            fontWeight: index === activeIndex ? '500' : '400'
                                        }}
                                    >
                                        {command.title}
                                    </span>
                                </div>
                                {command.shortcut && (
                                    <span
                                        style={{
                                            fontSize: '12px',
                                            color: isDarkMode ? '#9ca3af' : '#6b7280',
                                            background: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : '#fff',
                                            border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.1)',
                                            padding: '2px 6px',
                                            borderRadius: '4px',
                                            boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)'
                                        }}
                                    >
                                        {command.shortcut}
                                    </span>
                                )}
                            </button>
                        ))
                    ) : searchQuery.trim() ? (
                        <button
                            onClick={() => {
                                onNewCommand?.(searchQuery.trim());
                                onClose();
                            }}
                            style={{
                                width: '100%',
                                display: 'flex',
                                alignItems: 'center',
                                padding: '12px 16px',
                                fontSize: '14px',
                                background: 'transparent',
                                border: 'none',
                                cursor: 'pointer',
                                color: isDarkMode ? '#e2e8f0' : '#1a1a1a',
                                textAlign: 'left'
                            }}
                        >
                            <Bot size={16} style={{ marginRight: '12px', color: '#fda700' }} />
                            <span>Ask AI: "{searchQuery}"</span>
                        </button>
                    ) : null}
                </div>
            </div>
            <style>{`
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
            `}</style>
        </div>
    );
};
