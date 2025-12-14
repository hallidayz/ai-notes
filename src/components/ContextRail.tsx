/**
 * Context Rail Component
 * Right-side panel with AI insights, action items, and related notes
 */

import React, { useState, useEffect } from 'react';
import { X, Sparkles, FileText } from 'lucide-react';

interface Task {
    id?: number;
    title: string;
    dueDate: string | null;
    priority: 'low' | 'medium' | 'high';
    status: 'todo' | 'inprogress' | 'done';
    sessionId?: number;
    timestamp: number;
}

interface Session {
    id?: number;
    sessionTitle: string;
    participants?: string;
    date: string;
    notes: string;
    summary?: string;
    todoItems?: any;
    timestamp: number;
}

interface ContextRailProps {
    isOpen: boolean;
    onClose: () => void;
    selectedSession: Session | null;
    tasks: Task[];
    sessions: Session[];
}

export const ContextRail: React.FC<ContextRailProps> = ({
    isOpen,
    onClose,
    selectedSession,
    tasks,
    sessions
}) => {
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

    // Get action items from current session's tasks
    const actionItems = selectedSession
        ? tasks.filter(task => task.sessionId === selectedSession.id && task.status !== 'done')
        : [];

    // Get key insights from session summary
    const getKeyInsights = (): string[] => {
        if (!selectedSession?.summary) return [];
        
        try {
            const summary = typeof selectedSession.summary === 'string'
                ? JSON.parse(selectedSession.summary).summary || selectedSession.summary
                : selectedSession.summary;
            
            // Extract bullet points or key sentences
            const lines = summary.split('\n').filter((line: string) => line.trim().length > 0);
            return lines.slice(0, 5); // Limit to 5 insights
        } catch {
            return selectedSession.summary.split('.').filter((s: string) => s.trim().length > 20).slice(0, 5);
        }
    };

    const keyInsights = getKeyInsights();

    // Get related notes (other sessions)
    const relatedNotes = sessions
        .filter(s => s.id !== selectedSession?.id)
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, 5);

    const formatDate = (timestamp: number) => {
        const date = new Date(timestamp);
        const now = new Date();
        const diffTime = now.getTime() - date.getTime();
        const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays === 0) return 'Today';
        if (diffDays === 1) return 'Yesterday';
        if (diffDays < 7) return `${diffDays} days ago`;
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    return (
        <aside
            style={{
                position: 'fixed',
                top: 0,
                right: 0,
                bottom: 0,
                zIndex: 30,
                width: '320px',
                background: isDarkMode ? 'rgba(27, 52, 72, 0.5)' : 'rgba(248, 250, 252, 0.8)',
                backdropFilter: 'blur(8px)',
                borderLeft: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.06)',
                transform: isOpen ? 'translateX(0)' : 'translateX(100%)',
                transition: 'transform 0.3s ease-in-out',
                display: 'flex',
                flexDirection: 'column',
                height: '100vh'
            }}
            className="context-rail"
            onClick={(e) => e.stopPropagation()}
        >
            <div
                style={{
                    height: '56px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0 16px',
                    borderBottom: isDarkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.06)'
                }}
            >
                <span
                    style={{
                        fontWeight: '600',
                        fontSize: '12px',
                        color: isDarkMode ? '#9ca3af' : '#6b7280',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em'
                    }}
                >
                    AI Context
                </span>
                <button
                    onClick={onClose}
                    style={{
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        color: isDarkMode ? '#9ca3af' : '#6b7280',
                        padding: '4px',
                        borderRadius: '4px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transition: 'all 0.15s'
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.background = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';
                        e.currentTarget.style.color = isDarkMode ? '#e2e8f0' : '#1a1a1a';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'transparent';
                        e.currentTarget.style.color = isDarkMode ? '#9ca3af' : '#6b7280';
                    }}
                >
                    <X size={18} />
                </button>
            </div>

            <div
                style={{
                    flex: 1,
                    overflowY: 'auto',
                    padding: '16px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px'
                }}
            >
                {/* AI Insight Card */}
                {(keyInsights.length > 0 || actionItems.length > 0) && (
                    <div
                        style={{
                            background: isDarkMode ? 'rgba(27, 52, 72, 0.8)' : '#fff',
                            padding: '16px',
                            borderRadius: '12px',
                            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
                            border: isDarkMode
                                ? '1px solid rgba(253, 167, 0, 0.2)'
                                : '1px solid rgba(2, 41, 91, 0.1)',
                            position: 'relative'
                        }}
                    >
                        <div
                            style={{
                                position: 'absolute',
                                top: '-1px',
                                left: '-1px',
                                right: '-1px',
                                bottom: '-1px',
                                background: 'linear-gradient(135deg, rgba(253, 167, 0, 0.3) 0%, rgba(2, 41, 91, 0.3) 100%)',
                                borderRadius: '12px',
                                opacity: 0.3,
                                zIndex: -1,
                                filter: 'blur(4px)'
                            }}
                        />
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <h4
                                style={{
                                    fontWeight: '600',
                                    fontSize: '14px',
                                    color: isDarkMode ? '#e2e8f0' : '#1a1a1a',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    margin: 0
                                }}
                            >
                                <Sparkles size={14} style={{ color: '#fda700' }} />
                                Key Insights
                            </h4>
                        </div>
                        <ul style={{ margin: 0, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {keyInsights.map((insight, index) => (
                                <li key={index} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                                    <span
                                        style={{
                                            display: 'block',
                                            width: '4px',
                                            height: '4px',
                                            background: '#fda700',
                                            borderRadius: '50%',
                                            marginTop: '6px',
                                            flexShrink: 0
                                        }}
                                    />
                                    <span
                                        style={{
                                            fontSize: '13px',
                                            color: isDarkMode ? '#cbd5e1' : '#4b5563',
                                            lineHeight: '1.6'
                                        }}
                                    >
                                        {insight.trim()}
                                    </span>
                                </li>
                            ))}
                            {actionItems.length > 0 && (
                                <>
                                    <li style={{ marginTop: '8px', fontSize: '12px', fontWeight: '600', color: isDarkMode ? '#9ca3af' : '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                        Action Items
                                    </li>
                                    {actionItems.map((item) => (
                                        <li key={item.id} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                                            <span
                                                style={{
                                                    display: 'block',
                                                    width: '4px',
                                                    height: '4px',
                                                    background: '#2c5f41',
                                                    borderRadius: '50%',
                                                    marginTop: '6px',
                                                    flexShrink: 0
                                                }}
                                            />
                                            <span
                                                style={{
                                                    fontSize: '13px',
                                                    color: isDarkMode ? '#cbd5e1' : '#4b5563',
                                                    lineHeight: '1.6'
                                                }}
                                            >
                                                {item.title}
                                            </span>
                                        </li>
                                    ))}
                                </>
                            )}
                        </ul>
                    </div>
                )}

                {/* Related Notes List */}
                {relatedNotes.length > 0 && (
                    <div>
                        <h5
                            style={{
                                fontSize: '12px',
                                fontWeight: '600',
                                color: isDarkMode ? '#9ca3af' : '#6b7280',
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em',
                                marginBottom: '12px',
                                marginTop: '16px'
                            }}
                        >
                            Related Notes
                        </h5>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {relatedNotes.map((note) => (
                                <div
                                    key={note.id}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'flex-start',
                                        gap: '12px',
                                        padding: '8px',
                                        borderRadius: '8px',
                                        cursor: 'pointer',
                                        transition: 'all 0.15s',
                                        background: 'transparent'
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.background = isDarkMode
                                            ? 'rgba(255, 255, 255, 0.05)'
                                            : 'rgba(0, 0, 0, 0.04)';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.background = 'transparent';
                                    }}
                                >
                                    <FileText size={16} style={{ color: isDarkMode ? '#9ca3af' : '#6b7280', marginTop: '2px', flexShrink: 0 }} />
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <p
                                            style={{
                                                fontSize: '13px',
                                                fontWeight: '500',
                                                color: isDarkMode ? '#e2e8f0' : '#1a1a1a',
                                                margin: 0,
                                                marginBottom: '2px',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap'
                                            }}
                                        >
                                            {note.sessionTitle}
                                        </p>
                                        <p
                                            style={{
                                                fontSize: '11px',
                                                color: isDarkMode ? '#9ca3af' : '#6b7280',
                                                margin: 0
                                            }}
                                        >
                                            {formatDate(note.timestamp)}
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {keyInsights.length === 0 && actionItems.length === 0 && relatedNotes.length === 0 && (
                    <div
                        style={{
                            padding: '32px 16px',
                            textAlign: 'center',
                            color: isDarkMode ? '#9ca3af' : '#6b7280',
                            fontSize: '14px'
                        }}
                    >
                        No insights available. Select a note to see AI-generated insights and action items.
                    </div>
                )}
            </div>
        </aside>
    );
};
