/**
 * Editor Area Component
 * Central content area with note editor, header, and ghost text suggestions
 */

import React, { useState, useEffect, useRef } from 'react';
import { Menu, Sparkles, MoreHorizontal } from 'lucide-react';
import { CommandPalette } from './CommandPalette';
import { CryptoService } from '../services/CryptoService';

interface Session {
    id?: number;
    sessionTitle: string;
    participants?: string;
    date: string;
    notes: string;
    summary?: string;
    timestamp: number;
}

interface EditorAreaProps {
    selectedSession: Session | null;
    onSidebarToggle: () => void;
    onContextRailToggle: () => void;
    isContextOpen: boolean;
    isMobile: boolean;
    onUpdateSession: (session: Session) => void;
    onSummarize?: () => void;
    onActionItems?: () => void;
    pin: string;
}

export const EditorArea: React.FC<EditorAreaProps> = ({
    selectedSession,
    onSidebarToggle,
    onContextRailToggle,
    isContextOpen,
    isMobile,
    onUpdateSession,
    onSummarize,
    onActionItems,
    pin
}) => {
    const [showCommandPalette, setShowCommandPalette] = useState(false);
    const [ghostText, setGhostText] = useState('');
    const [title, setTitle] = useState('');
    const [content, setContent] = useState('');
    const [lastSaved, setLastSaved] = useState<Date | null>(null);
    const titleRef = useRef<HTMLDivElement>(null);
    const contentRef = useRef<HTMLDivElement>(null);
    const [isDarkMode, setIsDarkMode] = useState(false);

    useEffect(() => {
        const checkTheme = () => {
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            setIsDarkMode(isDark);
        };
        
        checkTheme();
        const observer = new MutationObserver(checkTheme);
        observer.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ['data-theme']
        });
        
        return () => observer.disconnect();
    }, []);

    // Load and decrypt session data
    useEffect(() => {
        if (selectedSession) {
            setTitle(selectedSession.sessionTitle || '');
            // Decrypt notes
            if (selectedSession.notes) {
                CryptoService.decrypt(selectedSession.notes, pin)
                    .then(decrypted => {
                        setContent(decrypted);
                    })
                    .catch(err => {
                        console.error('Failed to decrypt notes:', err);
                        setContent('');
                    });
            } else {
                setContent('');
            }
            setLastSaved(new Date(selectedSession.timestamp));
        } else {
            setTitle('');
            setContent('');
            setLastSaved(null);
        }
    }, [selectedSession, pin]);

    // Auto-save functionality
    useEffect(() => {
        if (!selectedSession || !selectedSession.id) return;

        const saveTimer = setTimeout(async () => {
            // Only save if something changed
            const currentDecryptedNotes = content;
            let currentEncryptedNotes = '';
            try {
                if (selectedSession.notes) {
                    currentEncryptedNotes = await CryptoService.decrypt(selectedSession.notes, pin);
                }
            } catch {
                // If decryption fails, assume it changed
            }

            if (title !== selectedSession.sessionTitle || currentDecryptedNotes !== currentEncryptedNotes) {
                // Encrypt notes before saving
                const encryptedNotes = await CryptoService.encrypt(currentDecryptedNotes, pin);
                const updatedSession = {
                    ...selectedSession,
                    sessionTitle: title,
                    notes: encryptedNotes,
                    timestamp: Date.now()
                };
                onUpdateSession(updatedSession);
                setLastSaved(new Date());
            }
        }, 2000); // Auto-save after 2 seconds of inactivity

        return () => clearTimeout(saveTimer);
    }, [title, content, selectedSession, onUpdateSession, pin]);

    // Command palette keyboard shortcut (Cmd/Ctrl+K)
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                setShowCommandPalette(prev => !prev);
            }
            if (e.key === 'Escape' && showCommandPalette) {
                setShowCommandPalette(false);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [showCommandPalette]);

    // Ghost text feature (placeholder for AI suggestions)
    const handleContentChange = (e: React.FormEvent<HTMLDivElement>) => {
        const newContent = e.currentTarget.textContent || '';
        setContent(newContent);
        
        // TODO: Integrate with AI service to generate ghost text suggestions
        // For now, just clear ghost text when user types
        if (newContent.length > 0) {
            setGhostText('');
        }
    };

    const handleTitleChange = (e: React.FormEvent<HTMLDivElement>) => {
        setTitle(e.currentTarget.textContent || '');
    };

    const formatLastSaved = () => {
        if (!lastSaved) return '';
        const now = new Date();
        const diff = now.getTime() - lastSaved.getTime();
        const seconds = Math.floor(diff / 1000);
        
        if (seconds < 10) return 'Saved just now';
        if (seconds < 60) return `Saved ${seconds}s ago`;
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `Saved ${minutes}m ago`;
        return lastSaved.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    };

    return (
        <main
            style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                minWidth: 0,
                background: isDarkMode ? 'transparent' : '#fff',
                position: 'relative',
                height: '100vh',
                overflow: 'hidden'
            }}
        >
            {/* Mobile Header */}
            {isMobile && (
                <header
                    style={{
                        height: '56px',
                        borderBottom: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.06)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '0 16px',
                        flexShrink: 0
                    }}
                >
                    <button
                        onClick={onSidebarToggle}
                        style={{
                            padding: '8px',
                            marginLeft: '-8px',
                            background: 'transparent',
                            border: 'none',
                            cursor: 'pointer',
                            color: isDarkMode ? '#cbd5e1' : '#4b5563',
                            borderRadius: '6px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}
                    >
                        <Menu size={20} />
                    </button>
                    <span
                        style={{
                            fontWeight: '600',
                            fontSize: '15px',
                            color: isDarkMode ? '#e2e8f0' : '#1a1a1a'
                        }}
                    >
                        {selectedSession?.sessionTitle || 'New Note'}
                    </span>
                    <button
                        onClick={onContextRailToggle}
                        style={{
                            padding: '8px',
                            marginRight: '-8px',
                            background: isContextOpen
                                ? (isDarkMode ? 'rgba(253, 167, 0, 0.2)' : 'rgba(2, 41, 91, 0.1)')
                                : 'transparent',
                            border: 'none',
                            cursor: 'pointer',
                            color: isContextOpen ? '#fda700' : (isDarkMode ? '#9ca3af' : '#6b7280'),
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            transition: 'all 0.15s'
                        }}
                    >
                        <Sparkles size={20} />
                    </button>
                </header>
            )}

            {/* Desktop Header */}
            {!isMobile && (
                <header
                    style={{
                        height: '56px',
                        borderBottom: '1px solid transparent',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '0 32px',
                        flexShrink: 0
                    }}
                >
                    <div
                        style={{
                            fontSize: '14px',
                            color: isDarkMode ? '#9ca3af' : '#6b7280'
                        }}
                    >
                        {selectedSession ? (
                            <>
                                Projects / <span style={{ color: isDarkMode ? '#cbd5e1' : '#4b5563' }}>{selectedSession.sessionTitle}</span>
                            </>
                        ) : (
                            'Projects / New Note'
                        )}
                    </div>
                    <div
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            color: isDarkMode ? '#9ca3af' : '#6b7280',
                            fontSize: '12px'
                        }}
                    >
                        {lastSaved && <span>{formatLastSaved()}</span>}
                        <button
                            onClick={onContextRailToggle}
                            style={{
                                padding: '8px',
                                borderRadius: '50%',
                                transition: 'all 0.15s',
                                background: isContextOpen
                                    ? (isDarkMode ? 'rgba(253, 167, 0, 0.2)' : 'rgba(2, 41, 91, 0.1)')
                                    : 'transparent',
                                border: 'none',
                                cursor: 'pointer',
                                color: isContextOpen ? '#fda700' : (isDarkMode ? '#9ca3af' : '#6b7280'),
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}
                            onMouseEnter={(e) => {
                                if (!isContextOpen) {
                                    e.currentTarget.style.background = isDarkMode
                                        ? 'rgba(255, 255, 255, 0.05)'
                                        : 'rgba(0, 0, 0, 0.04)';
                                }
                            }}
                            onMouseLeave={(e) => {
                                if (!isContextOpen) {
                                    e.currentTarget.style.background = 'transparent';
                                }
                            }}
                        >
                            <Sparkles size={18} />
                        </button>
                    </div>
                </header>
            )}

            {/* Canvas */}
            <div
                style={{
                    flex: 1,
                    overflowY: 'auto',
                    padding: '32px 16px',
                    position: 'relative'
                }}
            >
                <div
                    style={{
                        maxWidth: '768px',
                        margin: '0 auto',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '24px'
                    }}
                >
                    {selectedSession ? (
                        <h1
                            ref={titleRef}
                            contentEditable
                            onInput={handleTitleChange}
                            suppressContentEditableWarning
                            style={{
                                fontSize: '36px',
                                fontWeight: '700',
                                color: isDarkMode ? '#e2e8f0' : '#1a1a1a',
                                letterSpacing: '-0.02em',
                                outline: 'none',
                                border: 'none',
                                margin: 0,
                                padding: 0,
                                lineHeight: '1.2'
                            }}
                        >
                            {title || 'Untitled Note'}
                        </h1>
                    ) : (
                        <div
                            style={{
                                fontSize: '36px',
                                fontWeight: '700',
                                color: isDarkMode ? '#9ca3af' : '#9ca3af',
                                letterSpacing: '-0.02em',
                                margin: 0,
                                padding: 0,
                                lineHeight: '1.2',
                                fontStyle: 'italic'
                            }}
                        >
                            Select a note to get started
                        </div>
                    )}

                    {selectedSession ? (
                        <div
                            style={{
                                fontSize: '18px',
                                lineHeight: '1.7',
                                color: isDarkMode ? '#cbd5e1' : '#4b5563',
                                maxWidth: '100%'
                            }}
                        >
                            <div
                                ref={contentRef}
                                contentEditable
                                onInput={handleContentChange}
                                suppressContentEditableWarning
                                style={{
                                    outline: 'none',
                                    minHeight: '200px',
                                    color: isDarkMode ? '#cbd5e1' : '#4b5563'
                                }}
                            >
                                {content || ''}
                            </div>
                        {ghostText && (
                            <div
                                style={{
                                    color: isDarkMode ? '#4b5563' : '#d1d5db',
                                    fontStyle: 'italic',
                                    display: 'inline',
                                    marginLeft: '4px'
                                }}
                            >
                                {ghostText}
                                <span
                                    style={{
                                        fontSize: '11px',
                                        color: isDarkMode ? '#6b7280' : '#9ca3af',
                                        background: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : '#f3f4f6',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.1)',
                                        marginLeft: '4px',
                                        userSelect: 'none',
                                        cursor: 'default'
                                    }}
                                >
                                    TAB
                                </span>
                            </div>
                        )}
                        </div>
                    ) : (
                        <div
                            style={{
                                fontSize: '16px',
                                color: isDarkMode ? '#9ca3af' : '#9ca3af',
                                fontStyle: 'italic',
                                textAlign: 'center',
                                padding: '40px 20px'
                            }}
                        >
                            Create a new note or select an existing one from the sidebar
                        </div>
                    )}

                    {showCommandPalette && (
                        <CommandPalette
                            isOpen={showCommandPalette}
                            onClose={() => setShowCommandPalette(false)}
                            onSummarize={onSummarize}
                            onActionItems={onActionItems}
                            onNewCommand={(command) => {
                                // TODO: Implement custom AI command
                                console.log('Custom command:', command);
                                setShowCommandPalette(false);
                            }}
                        />
                    )}
                </div>
            </div>
        </main>
    );
};
