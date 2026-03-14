/**
 * Sidebar Component - Modern dark navigation sidebar
 * Redesigned for note-taking interface
 */

import React, { useState, useEffect } from 'react';
import { X, Plus, Sparkles, Clock, Folder, Tag, Bot, Calendar, Lock } from 'lucide-react';

interface Session {
    id?: number;
    sessionTitle: string;
    participants?: string;
    date: string;
    notes: string;
    timestamp: number;
}

interface SidebarProps {
    isOpen: boolean;
    onClose: () => void;
    onNewNote: () => void;
    onSelectNote: (session: Session) => void;
    onCalendarSettings: () => void;
    onLock: () => void;
    sessions: Session[];
    selectedSession: Session | null;
    isMobile: boolean;
}

const NavItem: React.FC<{
    icon: React.ReactNode;
    label: string;
    active: boolean;
    onClick: () => void;
    isDarkMode: boolean;
}> = ({ icon, label, active, onClick, isDarkMode }) => (
    <button
        onClick={onClick}
        style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '8px 12px',
            borderRadius: '8px',
            fontSize: '14px',
            transition: 'all 0.15s',
            background: active
                ? (isDarkMode ? 'rgba(253, 167, 0, 0.15)' : 'rgba(2, 41, 91, 0.1)')
                : 'transparent',
            border: 'none',
            cursor: 'pointer',
            color: active
                ? (isDarkMode ? '#fda700' : 'var(--color-authority-navy, #02295b)')
                : (isDarkMode ? '#9ca3af' : '#6b7280'),
            fontWeight: active ? '500' : '400',
            textAlign: 'left'
        }}
        onMouseEnter={(e) => {
            if (!active) {
                e.currentTarget.style.background = isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.04)';
                e.currentTarget.style.color = isDarkMode ? '#cbd5e1' : '#4b5563';
            }
        }}
        onMouseLeave={(e) => {
            if (!active) {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = isDarkMode ? '#9ca3af' : '#6b7280';
            }
        }}
    >
        {icon}
        <span>{label}</span>
    </button>
);

export const Sidebar: React.FC<SidebarProps> = ({
    isOpen,
    onClose,
    onNewNote,
    onSelectNote,
    onCalendarSettings,
    onLock,
    sessions,
    selectedSession,
    isMobile
}) => {
    const [activeView, setActiveView] = useState<'insights' | 'recent' | 'projects' | 'tags'>('projects');
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

    // Get recent sessions (last 10)
    const recentSessions = [...sessions]
        .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
        .slice(0, 10);

    // Get all sessions as projects
    const projects = sessions;

    return (
        <>
            {/* Mobile Overlay */}
            {isOpen && isMobile && (
                <div
                    style={{
                        position: 'fixed',
                        inset: 0,
                        background: 'rgba(0, 0, 0, 0.5)',
                        zIndex: 20
                    }}
                    onClick={onClose}
                />
            )}

            <aside
                style={{
                    position: 'fixed',
                    top: 0,
                    bottom: 0,
                    left: 0,
                    zIndex: 30,
                    width: '256px',
                    background: 'rgba(15, 23, 42, 0.95)', // slate-900 equivalent
                    color: '#9ca3af',
                    transform: isOpen ? 'translateX(0)' : (isMobile ? 'translateX(-100%)' : 'translateX(0)'),
                    transition: 'transform 0.3s ease-in-out',
                    display: 'flex',
                    flexDirection: 'column',
                    height: '100vh',
                    borderRight: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.06)'
                }}
            >
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '16px' }}>
                    {/* Header with Logo */}
                    <div
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            marginBottom: '32px',
                            padding: '0 8px'
                        }}
                    >
                        <div
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px',
                                color: '#fff',
                                fontWeight: '600',
                                fontSize: '16px'
                            }}
                        >
                            <div
                                style={{
                                    width: '32px',
                                    height: '32px',
                                    background: 'var(--color-authority-navy, #02295b)',
                                    borderRadius: '8px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center'
                                }}
                            >
                                <Bot size={20} style={{ color: '#fff' }} />
                            </div>
                            <span>MiNDS Talk</span>
                        </div>
                        {isMobile && (
                            <button
                                onClick={onClose}
                                style={{
                                    background: 'transparent',
                                    border: 'none',
                                    cursor: 'pointer',
                                    color: '#9ca3af',
                                    padding: '4px',
                                    borderRadius: '4px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center'
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                                    e.currentTarget.style.color = '#e2e8f0';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.background = 'transparent';
                                    e.currentTarget.style.color = '#9ca3af';
                                }}
                            >
                                <X size={20} />
                            </button>
                        )}
                    </div>

                    {/* New Note Button */}
                    <button
                        onClick={onNewNote}
                        style={{
                            width: '100%',
                            background: '#fff',
                            color: '#1a1a1a',
                            fontWeight: '500',
                            padding: '10px 16px',
                            borderRadius: '8px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            marginBottom: '24px',
                            border: 'none',
                            cursor: 'pointer',
                            transition: 'all 0.15s',
                            fontSize: '14px'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = '#f3f4f6';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = '#fff';
                        }}
                    >
                        <Plus size={18} />
                        <span>New Note</span>
                    </button>

                    {/* Navigation Items */}
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '4px',
                            flex: 1,
                            overflowY: 'auto',
                            paddingRight: '4px'
                        }}
                    >
                        <NavItem
                            icon={<Sparkles size={16} />}
                            label="AI Insights"
                            active={activeView === 'insights'}
                            onClick={() => setActiveView('insights')}
                            isDarkMode={isDarkMode}
                        />
                        <NavItem
                            icon={<Clock size={16} />}
                            label="Recent"
                            active={activeView === 'recent'}
                            onClick={() => setActiveView('recent')}
                            isDarkMode={isDarkMode}
                        />
                        <NavItem
                            icon={<Folder size={16} />}
                            label="Projects"
                            active={activeView === 'projects'}
                            onClick={() => setActiveView('projects')}
                            isDarkMode={isDarkMode}
                        />

                        {/* Notes List */}
                        <div style={{ paddingTop: '24px' }}>
                            <h3
                                style={{
                                    fontSize: '11px',
                                    fontWeight: '700',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.05em',
                                    marginBottom: '8px',
                                    padding: '0 12px',
                                    color: '#6b7280'
                                }}
                            >
                                {activeView === 'recent' ? 'Recent Notes' : activeView === 'projects' ? 'All Notes' : 'Tags'}
                            </h3>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                                {(activeView === 'recent' ? recentSessions : activeView === 'projects' ? projects : []).map((session) => (
                                    <button
                                        key={session.id}
                                        onClick={() => onSelectNote(session)}
                                        style={{
                                            width: '100%',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '8px',
                                            padding: '6px 12px',
                                            borderRadius: '6px',
                                            fontSize: '13px',
                                            background: selectedSession?.id === session.id
                                                ? (isDarkMode ? 'rgba(253, 167, 0, 0.15)' : 'rgba(2, 41, 91, 0.1)')
                                                : 'transparent',
                                            border: 'none',
                                            cursor: 'pointer',
                                            transition: 'all 0.15s',
                                            color: selectedSession?.id === session.id
                                                ? (isDarkMode ? '#fda700' : 'var(--color-authority-navy, #02295b)')
                                                : (isDarkMode ? '#9ca3af' : '#6b7280'),
                                            textAlign: 'left',
                                            fontWeight: selectedSession?.id === session.id ? '500' : '400'
                                        }}
                                        onMouseEnter={(e) => {
                                            if (selectedSession?.id !== session.id) {
                                                e.currentTarget.style.background = isDarkMode
                                                    ? 'rgba(255, 255, 255, 0.05)'
                                                    : 'rgba(0, 0, 0, 0.04)';
                                            }
                                        }}
                                        onMouseLeave={(e) => {
                                            if (selectedSession?.id !== session.id) {
                                                e.currentTarget.style.background = 'transparent';
                                            }
                                        }}
                                    >
                                        <Tag size={14} />
                                        <span
                                            style={{
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap',
                                                flex: 1
                                            }}
                                        >
                                            {session.sessionTitle}
                                        </span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Bottom Actions */}
                    <div
                        style={{
                            borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                            paddingTop: '12px',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '4px'
                        }}
                    >
                        <button
                            onClick={onCalendarSettings}
                            style={{
                                width: '100%',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '12px',
                                padding: '8px 12px',
                                borderRadius: '8px',
                                fontSize: '14px',
                                background: 'transparent',
                                border: 'none',
                                cursor: 'pointer',
                                color: '#9ca3af',
                                transition: 'all 0.15s',
                                textAlign: 'left'
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                                e.currentTarget.style.color = '#cbd5e1';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = 'transparent';
                                e.currentTarget.style.color = '#9ca3af';
                            }}
                        >
                            <Calendar size={16} />
                            <span>Calendar</span>
                        </button>
                        <button
                            onClick={onLock}
                            style={{
                                width: '100%',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '12px',
                                padding: '8px 12px',
                                borderRadius: '8px',
                                fontSize: '14px',
                                background: 'transparent',
                                border: 'none',
                                cursor: 'pointer',
                                color: '#9ca3af',
                                transition: 'all 0.15s',
                                textAlign: 'left'
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                                e.currentTarget.style.color = '#cbd5e1';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = 'transparent';
                                e.currentTarget.style.color = '#9ca3af';
                            }}
                        >
                            <Lock size={16} />
                            <span>Lock</span>
                        </button>
                    </div>
                </div>
            </aside>
        </>
    );
};
