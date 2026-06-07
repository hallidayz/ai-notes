import React from 'react';
import { TodoItem } from '../types';
import { AppIcon } from './AppIcon';

interface ActionItemsViewProps {
    todoItems: TodoItem[];
    onToggle: (index: number) => void;
    onPromote: (todo: TodoItem, index: number) => void;
    isDarkMode: boolean;
}

export const ActionItemsView: React.FC<ActionItemsViewProps> = ({ todoItems, onToggle, onPromote, isDarkMode }) => {
    if (!todoItems || todoItems.length === 0) {
        return null;
    }

    return (
        <div className="analysis-subsection">
            <h4 className="section-heading-icon">
                <AppIcon name="action-items" size={16} isDarkMode={isDarkMode} />
                Action Items
            </h4>
            <ul className="action-items-list">
                {todoItems.map((todo, index) => (
                    <li key={index} className={`todo-item ${todo.completed ? 'completed' : ''}`}>
                        <div className="todo-content" onClick={() => onToggle(index)}>
                            <input type="checkbox" readOnly checked={todo.completed} />
                            <span className="todo-text">{todo.text}</span>
                        </div>
                        {todo.promotedToTaskId ? (
                            <span className="task-promoted-badge">Tasked</span>
                        ) : (
                            <button
                                className="btn-promote-task"
                                title="Promote to Task"
                                onClick={() => onPromote(todo, index)}>
                                <AppIcon name="plus" size={14} isDarkMode={isDarkMode} />
                            </button>
                        )}
                    </li>
                ))}
            </ul>
        </div>
    );
};
