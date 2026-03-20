
import React, { useState } from 'react';
import { Task, Session, StorageProvider } from '../types';
import { TaskItem } from './TaskItem';

interface TaskManagerProps {
    tasks: Task[];
    onAddTask: (task: Omit<Task, 'id' | 'timestamp'>) => Promise<boolean>;
    onUpdateTask: (task: Task) => void;
    onDeleteTask: (id: number) => void;
    sessions: Session[];
    storageProvider: StorageProvider;
    industry: string;
    onUpdateSession: (session: Session) => Promise<void>;
}

export const TaskManager: React.FC<TaskManagerProps> = ({ 
    tasks, 
    onAddTask, 
    onUpdateTask, 
    onDeleteTask, 
    sessions,
    storageProvider,
    industry,
    onUpdateSession
}) => {
    const [newTaskTitle, setNewTaskTitle] = useState('');
    const [newTaskPriority, setNewTaskPriority] = useState<Task['priority']>('medium');
    const [newTaskDueDate, setNewTaskDueDate] = useState('');
    const [newTaskSessionId, setNewTaskSessionId] = useState<number | undefined>(undefined);
    const [isAdding, setIsAdding] = useState(false);

    // AI Analysis State
    const [analysisSessionId, setAnalysisSessionId] = useState<number | undefined>(undefined);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisProgress, setAnalysisProgress] = useState(0);
    const [analysisStatus, setAnalysisStatus] = useState('');
    const [successMessage, setSuccessMessage] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newTaskTitle) return;

        setIsAdding(true);
        const session = sessions.find(s => s.id === newTaskSessionId);
        const success = await onAddTask({
            title: newTaskTitle,
            priority: newTaskPriority,
            dueDate: newTaskDueDate || null,
            status: 'todo',
            sessionId: newTaskSessionId,
            sessionName: session?.sessionTitle
        });

        if (success) {
            setNewTaskTitle('');
            setNewTaskPriority('medium');
            setNewTaskDueDate('');
            setNewTaskSessionId(undefined);
        }
        setIsAdding(false);
    };

    const handleRunAnalysis = async () => {
        if (!analysisSessionId) return;
        const session = sessions.find(s => s.id === analysisSessionId);
        if (!session) return;

        setIsAnalyzing(true);
        setAnalysisProgress(0);
        setAnalysisStatus('Fetching audio...');
        setSuccessMessage('');

        try {
            const audioBlob = await storageProvider.getAudioBlob(session.id!);
            if (!audioBlob) {
                throw new Error('Audio recording not found for this session.');
            }

            const { onDeviceAIService } = await import('../services/onDeviceAIService');
            const results = await onDeviceAIService.analyze(
                audioBlob,
                industry,
                (status, progress) => {
                    setAnalysisStatus(status);
                    if (progress) setAnalysisProgress(progress);
                }
            );

            // Update session with analysis results
            const updatedSession: Session = {
                ...session,
                transcript: results.transcript,
                summary: results.summary,
                todoItems: (results.action_items || []).map(text => ({ text, completed: false })),
                outline: results.outline,
                analysisStatus: 'complete'
            };
            await onUpdateSession(updatedSession);

            // Create tasks from action items
            let addedCount = 0;
            for (const item of results.action_items) {
                const success = await onAddTask({
                    title: item,
                    priority: 'medium',
                    dueDate: null,
                    status: 'todo',
                    sessionId: session.id,
                    sessionName: session.sessionTitle
                });
                if (success) addedCount++;
            }

            setSuccessMessage(`Successfully generated ${addedCount} tasks from session "${session.sessionTitle}"!`);
            setAnalysisSessionId(undefined);
        } catch (err) {
            console.error("Analysis failed:", err);
            setAnalysisStatus(`Analysis failed: ${err instanceof Error ? err.message : String(err)}`);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const sortedTasks = [...tasks].sort((a, b) => {
        if (a.status === 'done' && b.status !== 'done') return 1;
        if (a.status !== 'done' && b.status === 'done') return -1;
        return b.timestamp - a.timestamp;
    });

    return (
        <div className="task-manager">
            <div className="card">
                <h3>Generate Tasks from Session</h3>
                <p className="text-secondary" style={{marginBottom: '16px', fontSize: '0.9em'}}>
                    Select a session to analyze its audio and automatically extract action items as tasks.
                </p>
                <div className="form-grid">
                    <select 
                        value={analysisSessionId || ''} 
                        onChange={e => setAnalysisSessionId(e.target.value ? parseInt(e.target.value) : undefined)}
                        className="grid-col-span-2"
                        disabled={isAnalyzing}
                    >
                        <option value="">Select Session to Analyze</option>
                        {sessions.map(s => (
                            <option key={s.id} value={s.id}>{s.sessionTitle} ({new Date(s.timestamp).toLocaleDateString()})</option>
                        ))}
                    </select>
                    <button 
                        onClick={handleRunAnalysis} 
                        className="btn-primary" 
                        disabled={isAnalyzing || !analysisSessionId}
                        style={{marginTop: '8px'}}
                    >
                        {isAnalyzing ? 'Analyzing...' : 'Run AI Analysis'}
                    </button>
                </div>

                {isAnalyzing && (
                    <div className="analysis-progress-container" style={{marginTop: '16px'}}>
                        <div className="progress-status">{analysisStatus}</div>
                        <div className="progress-bar-bg">
                            <div className="progress-bar-fill" style={{ width: `${analysisProgress}%` }}></div>
                        </div>
                    </div>
                )}

                {successMessage && (
                    <div className="status success" style={{marginTop: '16px', padding: '10px', borderRadius: '8px'}}>
                        {successMessage}
                    </div>
                )}
            </div>

            <div className="card">
                <h3>Add New Task</h3>
                <form onSubmit={handleSubmit} className="task-form">
                    <div className="form-grid">
                        <div className="grid-col-span-2">
                            <label className="input-label">Task Title</label>
                            <input
                                type="text"
                                placeholder="What needs to be done?"
                                value={newTaskTitle}
                                onChange={e => setNewTaskTitle(e.target.value)}
                                required
                            />
                        </div>
                        <div>
                            <label className="input-label">Priority</label>
                            <select value={newTaskPriority} onChange={e => setNewTaskPriority(e.target.value as Task['priority'])}>
                                <option value="low">Low</option>
                                <option value="medium">Medium</option>
                                <option value="high">High</option>
                            </select>
                        </div>
                        <div>
                            <label className="input-label">Due Date</label>
                            <input
                                type="date"
                                value={newTaskDueDate}
                                onChange={e => setNewTaskDueDate(e.target.value)}
                            />
                        </div>
                        <div className="grid-col-span-2">
                            <label className="input-label">Link to Session (optional)</label>
                            <select 
                                value={newTaskSessionId || ''} 
                                onChange={e => setNewTaskSessionId(e.target.value ? parseInt(e.target.value) : undefined)}
                            >
                                <option value="">None</option>
                                {sessions.map(s => (
                                    <option key={s.id} value={s.id}>{s.sessionTitle}</option>
                                ))}
                            </select>
                        </div>
                    </div>
                    <button type="submit" className="btn-primary" disabled={isAdding || !newTaskTitle} style={{marginTop: '16px'}}>
                        {isAdding ? 'Adding...' : 'Add Task'}
                    </button>
                </form>
            </div>

            <div className="tasks-list">
                <h3>Your Tasks ({tasks.filter(t => t.status !== 'done').length} active)</h3>
                {sortedTasks.length === 0 ? (
                    <div className="empty-state">No tasks yet. Add one above or promote an action item from a session!</div>
                ) : (
                    sortedTasks.map(task => (
                        <TaskItem key={task.id} task={task} onUpdate={onUpdateTask} onDelete={onDeleteTask} />
                    ))
                )}
            </div>
        </div>
    );
};
