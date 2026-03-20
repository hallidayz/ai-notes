
import React from 'react';

interface ViewSwitcherProps {
    view: 'sessions' | 'tasks' | 'calendar';
    setView: (view: 'sessions' | 'tasks' | 'calendar') => void;
}

export const ViewSwitcher: React.FC<ViewSwitcherProps> = ({ view, setView }) => (
    <div className="view-switcher">
        <button className={view === 'sessions' ? 'active' : ''} onClick={() => setView('sessions')}>Sessions</button>
        <button className={view === 'tasks' ? 'active' : ''} onClick={() => setView('tasks')}>Tasks</button>
        <button className={view === 'calendar' ? 'active' : ''} onClick={() => setView('calendar')}>Calendar</button>
    </div>
);
