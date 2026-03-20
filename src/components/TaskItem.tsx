
import React from 'react';
import { Task } from '../types';

interface TaskItemProps {
    task: Task;
    onUpdate: (task: Task) => void;
    onDelete: (id: number) => void;
}

export const TaskItem: React.FC<TaskItemProps> = ({ task, onUpdate, onDelete }) => {
    const handleStatusToggle = () => {
        const nextStatus: Task['status'] = 
            task.status === 'todo' ? 'inprogress' : 
            task.status === 'inprogress' ? 'done' : 'todo';
        onUpdate({ ...task, status: nextStatus });
    };

    const getPriorityClass = (priority: Task['priority']) => {
        switch (priority) {
            case 'high': return 'priority-high';
            case 'medium': return 'priority-medium';
            case 'low': return 'priority-low';
            default: return '';
        }
    };

    return (
        <div className={`task-item ${task.status === 'done' ? 'done' : ''}`}>
            <div className="task-checkbox" onClick={handleStatusToggle}>
                {task.status === 'done' ? '✓' : ''}
            </div>
            <div className="task-info">
                <div className="task-title-row">
                    <span className="task-title">{task.title}</span>
                    <span className={`task-priority ${getPriorityClass(task.priority)}`}>{task.priority}</span>
                </div>
                <div className="task-meta">
                    {task.dueDate && <span className="task-date">Due: {new Date(task.dueDate).toLocaleDateString()}</span>}
                    {task.sessionName && <span className="task-session">From: {task.sessionName}</span>}
                </div>
            </div>
            <button className="task-delete" onClick={() => onDelete(task.id!)} aria-label="Delete task">
                &times;
            </button>
        </div>
    );
};
