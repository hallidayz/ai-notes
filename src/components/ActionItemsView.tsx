import React from 'react';
import { TodoItem } from '../types';

interface ActionItemsViewProps {
    todoItems: TodoItem[];
    onToggle: (index: number) => void;
    onPromote: (todo: TodoItem, index: number) => void;
}

export const ActionItemsView: React.FC<ActionItemsViewProps> = ({ todoItems, onToggle, onPromote }) => {
    if (!todoItems || todoItems.length === 0) {
        return null;
    }

    return (
        <div className="analysis-subsection">
            <h4>&#x2705; Action Items</h4>
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
                                &#x2795;
                            </button>
                        )}
                    </li>
                ))}
            </ul>
        </div>
    );
};
