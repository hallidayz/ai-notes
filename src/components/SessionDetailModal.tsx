
import React, { useState, useEffect, useCallback } from 'react';
import { Session, Task, TodoItem, StorageProvider, TranscriptChunk } from '../types';
import { CryptoService } from '../services/cryptoService';
import { onDeviceAIService } from '../services/onDeviceAIService';
import { AudioPlayer } from './AudioPlayer';
import { AppIcon } from './AppIcon';

interface SessionDetailModalProps {
    session: Session;
    onClose: () => void;
    onDelete: (id: number) => void;
    onUpdate: (session: Session) => void;
    onAddTask: (task: Omit<Task, 'id' | 'timestamp'>) => Promise<boolean>;
    pin: string;
    industry: string;
    storageProvider: StorageProvider;
    showStatus: (msg: string, type: 'success' | 'error' | 'info') => void;
    isDarkMode: boolean;
}

export const SessionDetailModal: React.FC<SessionDetailModalProps> = ({ 
    session, onClose, onDelete, onUpdate, onAddTask, pin, industry, storageProvider, showStatus, isDarkMode
}) => {
    const [decryptedNotes, setDecryptedNotes] = useState('');
    const [isDecrypting, setIsDecrypting] = useState(true);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [isEditingNotes, setIsEditingNotes] = useState(false);
    const [editedNotes, setEditedNotes] = useState('');
    const [speakerMap, setSpeakerMap] = useState<{[key: string]: string}>({});
    const [editingSpeaker, setEditingSpeaker] = useState<{chunkIndex: number, oldName: string} | null>(null);
    const [aiAnalysisStatus, setAiAnalysisStatus] = useState<'idle' | 'in_progress' | 'failed' | 'complete'>('idle');
    const [aiProgress, setAiProgress] = useState({ status: '', progress: 0 });
    const [decryptedParticipants, setDecryptedParticipants] = useState('');
    const [decryptedTranscript, setDecryptedTranscript] = useState<TranscriptChunk[]>([]);
    const [decryptedSummary, setDecryptedSummary] = useState('');
    const [decryptedOutline, setDecryptedOutline] = useState('');
    const [decryptedTodoItems, setDecryptedTodoItems] = useState<TodoItem[]>([]);

    useEffect(() => {
        let currentAudioUrl: string | null = null;
        const decryptAndLoad = async () => {
            setIsDecrypting(true);
            try {
                // Decrypt notes
                const notes = await CryptoService.decrypt(session.notes, pin);
                setDecryptedNotes(notes);
                setEditedNotes(notes);

                // Decrypt participants if present
                if (session.participants) {
                    try {
                        const participants = await CryptoService.decrypt(session.participants, pin);
                        setDecryptedParticipants(participants);
                    } catch { 
                        // If decryption fails, assume it's plain text
                        setDecryptedParticipants(session.participants); 
                    }
                } else {
                    setDecryptedParticipants('');
                }

                // Decrypt transcript if present
                if (session.transcript) {
                    if (typeof session.transcript === 'string') {
                        try {
                            const decryptedTranscript = await CryptoService.decrypt(session.transcript, pin);
                            setDecryptedTranscript(JSON.parse(decryptedTranscript));
                        } catch { 
                            // If it's a string but not encrypted JSON, it might be legacy or error
                            setDecryptedTranscript([]); 
                        }
                    } else if (Array.isArray(session.transcript)) {
                        setDecryptedTranscript(session.transcript);
                    }
                } else {
                    setDecryptedTranscript([]);
                }

                // Decrypt summary if present
                if (session.summary) {
                    try {
                        const summary = await CryptoService.decrypt(session.summary, pin);
                        setDecryptedSummary(summary);
                    } catch { 
                        setDecryptedSummary(session.summary); 
                    }
                } else {
                    setDecryptedSummary('');
                }

                // Decrypt outline if present
                if (session.outline) {
                    try {
                        const outline = await CryptoService.decrypt(session.outline, pin);
                        setDecryptedOutline(outline);
                    } catch { 
                        setDecryptedOutline(session.outline); 
                    }
                } else {
                    setDecryptedOutline('');
                }

                // Decrypt todoItems if present
                if (session.todoItems) {
                    if (typeof session.todoItems === 'string') {
                        try {
                            const decryptedTodos = await CryptoService.decrypt(session.todoItems, pin);
                            setDecryptedTodoItems(JSON.parse(decryptedTodos));
                        } catch { 
                            setDecryptedTodoItems([]); 
                        }
                    } else if (Array.isArray(session.todoItems)) {
                        setDecryptedTodoItems(session.todoItems);
                    }
                } else {
                    setDecryptedTodoItems([]);
                }

                const blob = await storageProvider.getAudioBlob(session.id!);
                if (blob) {
                    setAudioBlob(blob);
                    const url = URL.createObjectURL(blob);
                    currentAudioUrl = url;
                    setAudioUrl(url);
                }

            } catch (err) {
                console.error("Decryption error:", err);
                setDecryptedNotes("Error: Could not decrypt data. The PIN may be incorrect or data is corrupted.");
            } finally {
                setIsDecrypting(false);
            }
        };
        decryptAndLoad();

        return () => {
            if (currentAudioUrl) {
                URL.revokeObjectURL(currentAudioUrl);
            }
        };
    }, [session, pin, storageProvider]);

    useEffect(() => {
        const uniqueSpeakers = [...new Set(decryptedTranscript.map(c => c.speaker))];
        const initialMap: {[key: string]: string} = {};
        uniqueSpeakers.forEach(speaker => {
            initialMap[speaker] = speaker;
        });
        setSpeakerMap(initialMap);
    }, [decryptedTranscript]);

    const handleSaveNotes = async () => {
        try {
            const { CryptoService } = await import('../services/cryptoService');
            const encryptedNotes = await CryptoService.encrypt(editedNotes, pin);
            
            // Re-encrypt other fields to be safe
            const encryptedParticipants = decryptedParticipants ? await CryptoService.encrypt(decryptedParticipants, pin) : '';
            const encryptedTranscript = decryptedTranscript.length > 0 ? await CryptoService.encrypt(JSON.stringify(decryptedTranscript), pin) : '';
            const encryptedSummary = decryptedSummary ? await CryptoService.encrypt(decryptedSummary, pin) : '';
            const encryptedOutline = decryptedOutline ? await CryptoService.encrypt(decryptedOutline, pin) : '';
            const encryptedTodos = decryptedTodoItems.length > 0 ? await CryptoService.encrypt(JSON.stringify(decryptedTodoItems), pin) : '';

            onUpdate({ 
                ...session, 
                notes: encryptedNotes, 
                participants: encryptedParticipants,
                transcript: encryptedTranscript as unknown as TranscriptChunk[],
                summary: encryptedSummary,
                outline: encryptedOutline as unknown as OutlineItem[],
                todoItems: encryptedTodos as unknown as TodoItem[],
                audioBlob: audioBlob || undefined 
            });
            setDecryptedNotes(editedNotes);
            setIsEditingNotes(false);
        } catch {
            showStatus('Failed to save notes.', 'error');
        }
    };
    
    const handlePromoteTodoToTask = useCallback(async (todo: TodoItem, todoIndex: number) => {
        const success = await onAddTask({
            title: todo.text,
            dueDate: null,
            priority: 'medium',
            status: 'todo',
            sessionId: session.id,
            sessionName: session.sessionTitle,
            timestamp: Date.now()
        });

        if (success) {
            const updatedTodos = [...decryptedTodoItems];
            updatedTodos[todoIndex] = { ...todo, promotedAt: Date.now() };
            
            try {
                const encryptedTodos = await CryptoService.encrypt(JSON.stringify(updatedTodos), pin);
                onUpdate({ ...session, todoItems: encryptedTodos as unknown as TodoItem[], audioBlob: audioBlob || undefined });
                setDecryptedTodoItems(updatedTodos);
            } catch (err) {
                console.error("Failed to encrypt todos", err);
            }
        }
    }, [onAddTask, session, decryptedTodoItems, pin, audioBlob, onUpdate]);
    
    const handleTodoToggle = useCallback(async (index: number) => {
        const updatedTodos = [...decryptedTodoItems];
        updatedTodos[index].completed = !updatedTodos[index].completed;
        
        try {
            const encryptedTodos = await CryptoService.encrypt(JSON.stringify(updatedTodos), pin);
            onUpdate({ ...session, todoItems: encryptedTodos as unknown as TodoItem[], audioBlob: audioBlob || undefined });
            setDecryptedTodoItems(updatedTodos);
        } catch (err) {
            console.error("Failed to encrypt todos", err);
        }
    }, [decryptedTodoItems, pin, session, audioBlob, onUpdate]);

    const onTodoToggleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
        const indexStr = e.currentTarget.dataset.index;
        if (indexStr !== undefined) {
            handleTodoToggle(parseInt(indexStr, 10));
        }
    }, [handleTodoToggle]);

    const onPromoteClick = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
        const indexStr = e.currentTarget.dataset.index;
        if (indexStr !== undefined) {
            const index = parseInt(indexStr, 10);
            const todo = decryptedTodoItems[index];
            if (todo) {
                handlePromoteTodoToTask(todo, index);
            }
        }
    }, [decryptedTodoItems, handlePromoteTodoToTask]);

    const handleRunOnDeviceAnalysis = useCallback(async () => {
        setAiAnalysisStatus('in_progress');
        onUpdate({ ...session, analysisStatus: 'pending', audioBlob: audioBlob || undefined });
        try {
            if (!audioBlob) {
                throw new Error("Audio file not found for this session.");
            }
            
            const result = await onDeviceAIService.analyze(audioBlob, industry, (status, progress) => {
                 setAiProgress({ status, progress: progress || 0 });
            });
            
            const todoItems: TodoItem[] = (result.action_items || []).map(text => ({ text, completed: false }));
            
            const encryptedTranscript = await CryptoService.encrypt(JSON.stringify(result.transcript), pin);
            const encryptedSummary = await CryptoService.encrypt(result.summary, pin);
            const encryptedTodos = await CryptoService.encrypt(JSON.stringify(todoItems), pin);
            const encryptedOutline = await CryptoService.encrypt(result.outline, pin);

            onUpdate({ 
                ...session, 
                transcript: encryptedTranscript as unknown as TranscriptChunk[], 
                summary: encryptedSummary, 
                todoItems: encryptedTodos as unknown as TodoItem[], 
                outline: encryptedOutline, 
                analysisStatus: 'complete', 
                audioBlob: audioBlob || undefined 
            });
            
            setDecryptedTranscript(result.transcript);
            setDecryptedSummary(result.summary);
            setDecryptedTodoItems(todoItems);
            setDecryptedOutline(result.outline);
            setAiAnalysisStatus('complete');

        } catch (err) {
            console.error("Analysis failed:", err);
            setAiAnalysisStatus('failed');
            onUpdate({ ...session, analysisStatus: 'failed', audioBlob: audioBlob || undefined });
        }
    }, [audioBlob, industry, onUpdate, session, pin]);

    const handleSpeakerNameChange = async (newName: string) => {
        if (!editingSpeaker) return;
        
        const { oldName } = editingSpeaker;
        const newMap = { ...speakerMap, [oldName]: newName };
        setSpeakerMap(newMap);

        const newTranscript = decryptedTranscript.map(chunk => {
            if (chunk.speaker === oldName) {
                return { ...chunk, speaker: newName };
            }
            return chunk;
        });

        try {
            const encryptedTranscript = await CryptoService.encrypt(JSON.stringify(newTranscript), pin);
            onUpdate({ ...session, transcript: encryptedTranscript as unknown as TranscriptChunk[], audioBlob: audioBlob || undefined });
            setDecryptedTranscript(newTranscript);
        } catch (err) {
            console.error("Failed to encrypt transcript", err);
        }
        setEditingSpeaker(null);
    };

    const getSpeakerClass = (speaker: string) => {
        const speakers = Object.keys(speakerMap);
        const index = speakers.indexOf(speaker);
        return `speaker-style-${(index % 5) + 1}`;
    };

    return (
        <div className="modal active" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose} aria-label="Close">
                    <AppIcon name="close" size={16} isDarkMode={isDarkMode} />
                </button>
                <h2>{session.sessionTitle}</h2>
                <p style={{ color: '#94a3b8', marginBottom: '16px' }}>{new Date(session.date).toLocaleDateString()}{decryptedParticipants && ` with ${decryptedParticipants}`}</p>
                
                {audioUrl && <AudioPlayer audioUrl={audioUrl} />}

                {session.analysisStatus === 'none' && aiAnalysisStatus !== 'in_progress' && (
                    <div className="action-buttons" style={{ justifyContent: 'center', margin: '20px 0'}}>
                        <button className="btn-ai" onClick={handleRunOnDeviceAnalysis}>
                            <AppIcon name="ai-chip" size={16} isDarkMode={isDarkMode} />
                            Run On-Device Analysis
                        </button>
                    </div>
                )}
                
                {aiAnalysisStatus === 'in_progress' && (
                    <div className="analysis-progress">
                        <div className="spinner-small"></div>
                        <div className="analysis-progress-text">
                            <span>{aiProgress.status}</span>
                            {aiProgress.status.startsWith('Downloading') && (
                                <div className="download-progress-bar">
                                    <div className="download-progress-bar-inner" style={{width: `${aiProgress.progress}%`}}></div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
                
                {(aiAnalysisStatus === 'failed' || (session.analysisStatus === 'failed' && aiAnalysisStatus === 'idle')) && (
                    <div className="status error" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px', padding: '16px' }}>
                        <span>On-device AI analysis failed. Please try again.</span>
                        <button className="btn-secondary" onClick={handleRunOnDeviceAnalysis} style={{ width: 'fit-content' }}>
                            Retry Analysis
                        </button>
                    </div>
                )}

                {session.analysisStatus === 'complete' && (
                    <div className="analysis-section">
                        <div className="analysis-subsection">
                            <h4 className="section-heading-icon"><AppIcon name="summary" size={16} isDarkMode={isDarkMode} /> Summary</h4>
                            <p>{decryptedSummary}</p>
                        </div>
                         {decryptedTodoItems && decryptedTodoItems.length > 0 && (
                            <div className="analysis-subsection">
                                <h4 className="section-heading-icon"><AppIcon name="action-items" size={16} isDarkMode={isDarkMode} /> Action Items</h4>
                                <ul className="action-items-list">
                                    {decryptedTodoItems.map((todo, index) => (
                                        <li key={index} className={`todo-item ${todo.completed ? 'completed' : ''}`}>
                                            <div className="todo-content" data-index={index} onClick={onTodoToggleClick}>
                                                <input type="checkbox" readOnly checked={todo.completed} />
                                                <span className="todo-text">{todo.text}</span>
                                            </div>
                                            {todo.promotedAt ? (
                                                <span className="task-promoted-badge">Tasked</span>
                                            ) : (
                                                <button 
                                                    className="btn-promote-task" 
                                                    title="Promote to Task" 
                                                    data-index={index}
                                                    onClick={onPromoteClick}>
                                                    <AppIcon name="plus" size={14} isDarkMode={isDarkMode} />
                                                </button>
                                            )}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {decryptedOutline && (
                             <div className="analysis-subsection">
                                <h4 className="section-heading-icon"><AppIcon name="outline" size={16} isDarkMode={isDarkMode} /> Outline</h4>
                                <div className="outline-content">{decryptedOutline}</div>
                            </div>
                        )}
                    </div>
                )}

                <h3>Notes</h3>
                {isDecrypting ? (
                    <div className="loading">Decrypting...</div>
                ) : (
                    isEditingNotes ? (
                        <div>
                            <textarea value={editedNotes} onChange={e => setEditedNotes(e.target.value)} rows={8} style={{ width: '100%' }} />
                            <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                                <button className="btn-primary" onClick={handleSaveNotes}>Save</button>
                                <button className="btn-stop" onClick={() => setIsEditingNotes(false)}>Cancel</button>
                            </div>
                        </div>
                    ) : (
                        <div>
                            <div className="transcript" style={{ whiteSpace: 'pre-wrap' }} onClick={() => setIsEditingNotes(true)}>
                                {decryptedNotes || <span style={{color: '#94a3b8'}}>Click to add notes...</span>}
                            </div>
                        </div>
                    )
                )}
                
                {decryptedTranscript && decryptedTranscript.length > 0 && (
                    <>
                        <h3>Transcript</h3>
                        <div className="transcript">
                            {decryptedTranscript.map((chunk, index) => (
                                <div key={index} className={`transcript-chunk ${getSpeakerClass(chunk.speaker)}`}>
                                   {editingSpeaker?.chunkIndex === index ? (
                                        <input
                                            type="text"
                                            defaultValue={editingSpeaker.oldName}
                                            onBlur={(e) => handleSpeakerNameChange(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && handleSpeakerNameChange(e.currentTarget.value)}
                                            autoFocus
                                            className="speaker-input"
                                        />
                                   ) : (
                                       <span
                                            className="speaker-label editable"
                                            onClick={() => setEditingSpeaker({ chunkIndex: index, oldName: chunk.speaker })}
                                        >
                                           {speakerMap[chunk.speaker] || chunk.speaker}:
                                        </span>
                                   )}
                                    <p>{chunk.text}</p>
                                </div>
                            ))}
                        </div>
                    </>
                )}

                <div style={{ marginTop: '24px', display: 'flex', justifyContent: 'flex-end' }}>
                    <button className="btn-danger" onClick={() => { onDelete(session.id!); onClose(); }}>Delete Session</button>
                </div>
            </div>
        </div>
    );
};
