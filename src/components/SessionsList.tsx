
import React from 'react';
import { Session } from '../types';
import { AppIcon } from './AppIcon';

interface SessionsListProps {
    sessions: Session[];
    onSelect: (session: Session) => void;
    onDelete: (id: number) => void;
    isDarkMode: boolean;
}

export const SessionsList: React.FC<SessionsListProps> = ({ sessions, onSelect, onDelete, isDarkMode }) => {
    
    if (sessions.length === 0) {
        return <div className="empty-state">No sessions yet. Create one to get started!</div>;
    }

    const decryptAndPreview = () => {
        try {
            return `Encrypted notes...`;
        } catch {
            return "Could not decrypt preview.";
        }
    };

    return (
        <div className="sessions-list">
            <h3>Recent Sessions</h3>
            {sessions.map(session => (
                <div key={session.id} className="session-item" onClick={() => onSelect(session)}>
                    <div className="session-content">
                        <div className="session-header">
                            <span className="session-title">{session.sessionTitle}</span>
                            <span className="session-date">{new Date(session.date).toLocaleDateString()}</span>
                        </div>
                        {session.participants && <p className="session-participants">With: {session.participants}</p>}
                        <p className="session-preview">{decryptAndPreview()}</p>
                        {session.analysisStatus && session.analysisStatus !== 'complete' && session.analysisStatus !== 'none' && (
                            <div className={`session-status-indicator ${session.analysisStatus}`}>
                                {session.analysisStatus === 'pending' && <><div className="spinner-small"></div> Processing AI analysis...</>}
                                {session.analysisStatus === 'failed' && (
                                    <span className="inline-icon-label">
                                        <AppIcon name="warning" size={14} isDarkMode={isDarkMode} />
                                        AI analysis failed
                                    </span>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="session-actions">
                        <button
                            className="icon-btn"
                            onClick={(e) => { e.stopPropagation(); onDelete(session.id!); }}
                            aria-label="Delete session"
                        >
                            <AppIcon name="delete" size={18} isDarkMode={isDarkMode} />
                        </button>
                    </div>
                </div>
            ))}
        </div>
    );
};
