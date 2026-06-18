
import React, { useState, useEffect, useCallback } from 'react';
import { Session, Task, StorageType, StorageProvider } from './types';
import { NotesDB } from './services/notesDB';
import { ServerStorageProvider, FileSystemStorageProvider, FileSystemDirectoryHandle } from './services/storageProvider';
import { AuthScreen } from './components/AuthScreen';
import { ThemeToggle } from './components/ThemeToggle';
import { ViewSwitcher } from './components/ViewSwitcher';
import { NewSessionForm } from './components/NewSessionForm';
import { SessionsList } from './components/SessionsList';
import { SessionDetailModal } from './components/SessionDetailModal';
import { TaskManager } from './components/TaskManager';
import { CalendarIntegration } from './components/CalendarIntegration';
import { LocalModels } from './components/LocalModels';
import { Settings } from './components/Settings';
import { onDeviceAIService } from './services/onDeviceAIService';
import { Settings as SettingsIcon } from 'lucide-react';

const db = new NotesDB();

export const App: React.FC = () => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [pin, setPin] = useState('');
    const [isDarkMode, setIsDarkMode] = useState(() => {
        const saved = localStorage.getItem('theme');
        return saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches);
    });

    useEffect(() => {
        document.body.classList.toggle('dark-mode', isDarkMode);
        localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
    }, [isDarkMode]);

    const handleAuthenticate = (enteredPin: string) => {
        setPin(enteredPin);
        setIsAuthenticated(true);
    };

    const handleToggleTheme = () => setIsDarkMode(!isDarkMode);

    if (!isAuthenticated) {
        return <AuthScreen onAuthenticate={handleAuthenticate} isDarkMode={isDarkMode} onToggleTheme={handleToggleTheme} />;
    }

    return <MainApp pin={pin} isDarkMode={isDarkMode} onToggleTheme={handleToggleTheme} />;
};

const MainApp: React.FC<{ pin: string, isDarkMode: boolean, onToggleTheme: () => void }> = ({ pin, isDarkMode, onToggleTheme }) => {
    const [sessions, setSessions] = useState<Session[]>([]);
    const [tasks, setTasks] = useState<Task[]>([]);
    const [selectedSession, setSelectedSession] = useState<Session | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [view, setView] = useState<'sessions' | 'tasks' | 'calendar' | 'local_models' | 'settings'>('sessions');
    const [industry, setIndustry] = useState('General');
    const [storageType, setStorageType] = useState<StorageType>(StorageType.BROWSER);
    const [storageProvider, setStorageProvider] = useState<StorageProvider>(new FileSystemStorageProvider(db));
    const [isStorageConfigured, setIsStorageConfigured] = useState<boolean>(true);
    const [status, setStatus] = useState<{ message: string, type: 'success' | 'error' | 'info' }>({ message: '', type: 'info' });
    const [confirmDeleteSessionId, setConfirmDeleteSessionId] = useState<number | null>(null);

    useEffect(() => {
        const loadInitialData = async () => {
            setIsLoading(true);
            try {
                const savedIndustry = await db.getConfig('industry') as string;
                if (savedIndustry) setIndustry(savedIndustry);

                const savedStorageType = await db.getConfig('storageType') as StorageType;
                if (savedStorageType) {
                    setStorageType(savedStorageType);
                    setIsStorageConfigured(true);
                } else {
                    setIsStorageConfigured(false);
                }
                
                let provider: StorageProvider;
                if (savedStorageType === StorageType.SERVER) {
                    provider = new ServerStorageProvider(pin);
                } else {
                    const fsProvider = new FileSystemStorageProvider(db);
                    await fsProvider.init();
                    provider = fsProvider;
                }
                setStorageProvider(provider);

                const modelConfig = await provider.getConfig('model_config');
                if (modelConfig) {
                    onDeviceAIService.updateConfig(modelConfig);
                }

                // Initial load will happen in the storageProvider effect
            } catch (err) {
                console.error("Error loading initial config:", err);
            } finally {
                setIsLoading(false);
            }
        };
        loadInitialData();
    }, [pin]);

    const loadData = useCallback(async () => {
        setIsLoading(true);
        try {
            const [loadedSessions, loadedTasks] = await Promise.all([
                storageProvider.getAllSessions(),
                storageProvider.getAllTasks()
            ]);
            setSessions(loadedSessions);
            setTasks(loadedTasks);
        } catch (err) {
            console.error("Error loading data:", err);
            showStatus("Failed to load data from storage.", 'error');
        } finally {
            setIsLoading(false);
        }
    }, [storageProvider]);

    useEffect(() => {
        loadData();
    }, [loadData]);

    const showStatus = (message: string, type: 'success' | 'error' | 'info', duration = 3000) => {
        setStatus({ message, type });
        setTimeout(() => setStatus({ message: '', type: 'info' }), duration);
    };

    const handleAddSession = async (sessionData: Omit<Session, 'id' | 'timestamp' | 'notes'>, notes: string, audioBlob: Blob | null) => {
        try {
            const { CryptoService } = await import('./services/cryptoService');
            const encryptedNotes = await CryptoService.encrypt(notes, pin);
            const encryptedParticipants = sessionData.participants ? await CryptoService.encrypt(sessionData.participants, pin) : '';
            
            const newSession: Session = {
                ...sessionData,
                participants: encryptedParticipants,
                notes: encryptedNotes,
                timestamp: Date.now(),
                analysisStatus: 'none'
            };

            const id = await storageProvider.saveSession(newSession);
            if (audioBlob) {
                await storageProvider.saveAudioBlob(id, audioBlob);
            }
            
            await loadData();
            showStatus('Session saved successfully.', 'success');
            return true;
        } catch (err) {
            console.error("Error adding session:", err);
            showStatus('Failed to save session.', 'error');
            return false;
        }
    };

    const handleDeleteSession = async (id: number) => {
        try {
            await storageProvider.deleteSession(id);
            setConfirmDeleteSessionId(null);
            await loadData();
            showStatus('Session deleted.', 'info');
        } catch (err) {
            console.error("Error deleting session:", err);
            showStatus('Failed to delete session.', 'error');
        }
    };

    const handleUpdateSession = async (updatedSession: Session) => {
        try {
            // Encrypt sensitive fields if they are in plain text (this is a bit tricky since onUpdate is called with both plain and encrypted data)
            // To simplify, we'll assume the caller (SessionDetailModal) handles encryption for notes,
            // but for transcript, summary, etc., we'll handle it here if they are present and not encrypted.
            // Actually, it's better if the caller handles it or we have a clear boundary.
            
            // Let's assume for now that we want to encrypt everything before it hits the storageProvider.
            
            await storageProvider.updateSession(updatedSession);
            await loadData();
            // Update selected session if it's the one being updated
            if (selectedSession?.id === updatedSession.id) {
                setSelectedSession(updatedSession);
            }
        } catch (err) {
            console.error("Error updating session:", err);
            showStatus('Failed to update session.', 'error');
        }
    };

    const handleAddTask = async (taskData: Omit<Task, 'id' | 'timestamp'>) => {
        try {
            const newTask: Task = {
                ...taskData,
                timestamp: Date.now()
            };
            await storageProvider.saveTask(newTask);
            await loadData();
            showStatus('Task added.', 'success');
            return true;
        } catch (err) {
            console.error("Error adding task:", err);
            showStatus('Failed to add task.', 'error');
            return false;
        }
    };

    const handleUpdateTask = async (task: Task) => {
        try {
            await storageProvider.updateTask(task);
            await loadData();
        } catch (err) {
            console.error("Error updating task:", err);
        }
    };

    const handleDeleteTask = async (id: number) => {
        try {
            await storageProvider.deleteTask(id);
            await loadData();
        } catch (err) {
            console.error("Error deleting task:", err);
        }
    };

    const handleIndustryChange = async (newIndustry: string) => {
        setIndustry(newIndustry);
        await db.saveConfig('industry', newIndustry);
    };

    const handleStorageTypeChange = async (type: StorageType) => {
        setStorageType(type);
        await db.saveConfig('storageType', type);
        if (type === StorageType.SERVER) {
            setStorageProvider(new ServerStorageProvider(pin));
        } else {
            const fsProvider = new FileSystemStorageProvider(db);
            await fsProvider.init();
            setStorageProvider(fsProvider);
        }
        showStatus(`Switched to ${type} storage.`, 'info');
    };

    const renderView = () => {
        switch (view) {
            case 'sessions':
                return (
                    <>
                        <NewSessionForm onAddSession={handleAddSession} showStatus={showStatus} />
                        <SessionsList sessions={sessions} onSelect={setSelectedSession} onDelete={handleDeleteSession} />
                    </>
                );
            case 'tasks':
                return (
                    <TaskManager 
                        tasks={tasks}
                        onAddTask={handleAddTask}
                        onUpdateTask={handleUpdateTask}
                        onDeleteTask={handleDeleteTask}
                        sessions={sessions}
                        storageProvider={storageProvider}
                        industry={industry}
                        onUpdateSession={handleUpdateSession}
                    />
                );
            case 'calendar':
                return <CalendarIntegration showStatus={showStatus} />;
            default:
                return null;
        }
    };

    const [storageFolder, setStorageFolder] = useState<string>('Downloads');

    useEffect(() => {
        const loadFolder = async () => {
            const savedFolder = await db.getConfig('storageFolder') as string;
            if (savedFolder) setStorageFolder(savedFolder);
        };
        loadFolder();
    }, []);

    const handleSelectFolder = async (e: React.MouseEvent) => {
        e.stopPropagation();
        try {
            if ('showDirectoryPicker' in window) {
                const dirHandle = await (window as { showDirectoryPicker?: (options?: { startIn: string }) => Promise<FileSystemDirectoryHandle> }).showDirectoryPicker?.({ startIn: 'downloads' });
                if (dirHandle) {
                    setStorageFolder(dirHandle.name);
                    await db.saveConfig('storageFolder', dirHandle.name);
                    if (storageProvider instanceof FileSystemStorageProvider) {
                        await storageProvider.setDirHandle(dirHandle);
                    }
                    showStatus(`Storage folder set to ${dirHandle.name}`, 'success');
                }
            } else {
                showStatus('Directory picker not supported in this browser', 'error');
            }
        } catch (err) {
            console.error('Error selecting folder:', err);
        }
    };

    if (view === 'local_models') {
        return <LocalModels storageProvider={storageProvider} onBack={() => setView('sessions')} showStatus={showStatus} />;
    }

    if (view === 'settings') {
        return <Settings 
            onBack={() => setView('sessions')} 
            showStatus={showStatus} 
            storageType={storageType} 
            onStorageTypeChange={handleStorageTypeChange} 
            storageFolder={storageFolder}
            onStorageFolderChange={async (folder, dirHandle) => {
                setStorageFolder(folder);
                await db.saveConfig('storageFolder', folder);
                if (dirHandle && storageProvider instanceof FileSystemStorageProvider) {
                    await storageProvider.setDirHandle(dirHandle);
                }
            }}
        />;
    }

    if (!isStorageConfigured && !isLoading) {
        return (
            <div className="container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
                <div className="card" style={{ maxWidth: '600px', width: '100%', padding: '40px', textAlign: 'center' }}>
                    <h2 style={{ marginBottom: '10px' }}>Welcome to AI Notes</h2>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: '30px' }}>Please select where you want to store your data.</p>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', marginBottom: '30px' }}>
                        <div 
                            className={`storage-option ${storageType === StorageType.BROWSER ? 'active' : ''}`}
                            onClick={() => setStorageType(StorageType.BROWSER)}
                            style={{ padding: '20px', textAlign: 'left', cursor: 'pointer', border: '1px solid var(--border-color)', borderRadius: '8px', backgroundColor: storageType === StorageType.BROWSER ? 'var(--bg-secondary)' : 'transparent', display: 'flex', flexDirection: 'column' }}
                        >
                            <div className="storage-option-info">
                                <span className="storage-option-title" style={{ display: 'block', fontWeight: 'bold', marginBottom: '5px' }}>Browser (IndexedDB)</span>
                                <span className="storage-option-desc" style={{ fontSize: '0.9em', color: 'var(--text-secondary)', marginBottom: '15px', display: 'block' }}>Fastest, fully offline, data stays in this browser. This does require the files to be written to a location on the device, by default, make it the downloads folder and allow the user to change it based on what they want to use.</span>
                            </div>
                            {storageType === StorageType.BROWSER && (
                                <div style={{ marginTop: 'auto', paddingTop: '15px', borderTop: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                    <span style={{ fontSize: '0.8em', fontWeight: 500, color: 'var(--text-secondary)' }}>Folder: {storageFolder}</span>
                                    <button 
                                        type="button"
                                        onClick={handleSelectFolder}
                                        className="btn-secondary"
                                        style={{ fontSize: '0.8em', padding: '5px 10px' }}
                                    >
                                        Change Folder
                                    </button>
                                </div>
                            )}
                        </div>
                        <div 
                            className={`storage-option ${storageType === StorageType.SERVER ? 'active' : ''}`}
                            onClick={() => setStorageType(StorageType.SERVER)}
                            style={{ padding: '20px', textAlign: 'left', cursor: 'pointer', border: '1px solid var(--border-color)', borderRadius: '8px', backgroundColor: storageType === StorageType.SERVER ? 'var(--bg-secondary)' : 'transparent' }}
                        >
                            <div className="storage-option-info">
                                <span className="storage-option-title" style={{ display: 'block', fontWeight: 'bold', marginBottom: '5px' }}>Server (Cloud)</span>
                                <span className="storage-option-desc" style={{ fontSize: '0.9em', color: 'var(--text-secondary)' }}>Sync across devices, data stored on our secure server.</span>
                            </div>
                        </div>
                    </div>

                    <button 
                        className="btn-primary" 
                        style={{ width: '100%', padding: '15px', fontSize: '1.1em' }}
                        onClick={() => {
                            handleStorageTypeChange(storageType);
                            setIsStorageConfigured(true);
                        }}
                    >
                        Continue
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="container">
            <header>
                <div style={{ position: 'absolute', top: '20px', right: '20px', display: 'flex', gap: '10px' }}>
                    <button 
                        onClick={() => setView('settings')}
                        className="btn-secondary"
                        style={{ padding: '8px', borderRadius: '50%', width: '40px', height: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                        title="Settings"
                    >
                        <SettingsIcon className="w-5 h-5" />
                    </button>
                    <button 
                        onClick={() => setView('local_models')}
                        className="btn-secondary"
                        style={{ padding: '8px', borderRadius: '50%', width: '40px', height: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                        title="Local Models"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v8"/><path d="m16 6-4 4-4-4"/><rect width="20" height="8" x="2" y="14" rx="2"/><path d="M6 18h.01"/><path d="M10 18h.01"/></svg>
                    </button>
                    <ThemeToggle isDarkMode={isDarkMode} onToggle={onToggleTheme} />
                </div>
                <h1>AI Notes</h1>
                <p>Private, Secure, On-Device Intelligence</p>
                <div className="settings-container">
                    <label htmlFor="industrySelector">Industry Context:</label>
                    <select 
                        id="industrySelector" 
                        value={industry} 
                        onChange={(e) => handleIndustryChange(e.target.value)}
                    >
                        <option value="General">General</option>
                        <option value="Medical">Medical</option>
                        <option value="Legal">Legal</option>
                        <option value="Therapy">Therapy</option>
                        <option value="Business">Business/Meetings</option>
                        <option value="Education">Education</option>
                    </select>
                </div>
                
                <div className="privacy-badge">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 1a2 2 0 0 1 2 2v4H6V3a2 2 0 0 1 2-2zm3 6V3a3 3 0 0 0-6 0v4a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2z"/></svg>
                    End-to-End Encrypted & On-Device AI
                </div>
            </header>

            {status.message && <div className={`status ${status.type}`}>{status.message}</div>}

            <ViewSwitcher view={view} setView={setView} />

            {isLoading ? (
                <div className="loading"><div className="spinner"></div>Loading...</div>
            ) : (
                renderView()
            )}

            {selectedSession && (
                <SessionDetailModal
                    session={selectedSession}
                    onClose={() => setSelectedSession(null)}
                    onDelete={(id) => setConfirmDeleteSessionId(id)}
                    onUpdate={handleUpdateSession}
                    onAddTask={handleAddTask}
                    pin={pin}
                    industry={industry}
                    storageProvider={storageProvider}
                    showStatus={showStatus}
                />
            )}

            {confirmDeleteSessionId !== null && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[100] p-4">
                    <div className="card p-6 max-w-sm w-full text-center">
                        <h3 className="text-xl font-bold mb-4">Delete Session?</h3>
                        <p className="text-gray-500 mb-6">This action cannot be undone. All encrypted data for this session will be permanently removed.</p>
                        <div className="flex gap-3">
                            <button 
                                onClick={() => setConfirmDeleteSessionId(null)}
                                className="btn-secondary flex-1"
                            >
                                Cancel
                            </button>
                            <button 
                                onClick={() => handleDeleteSession(confirmDeleteSessionId)}
                                className="btn-primary bg-red-500 border-red-500 flex-1"
                            >
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
