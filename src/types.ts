
export interface TranscriptChunk {
    speaker: string;
    text: string;
}

export interface TodoItem {
    text: string;
    completed: boolean;
    promotedAt?: number;
}

export interface Session {
    id?: number;
    sessionTitle: string;
    participants?: string;
    date: string;
    notes: string;
    duration: number;
    transcript: TranscriptChunk[];
    timestamp: number;
    summary?: string;
    todoItems?: TodoItem[];
    outline?: string;
    analysisStatus?: 'pending' | 'complete' | 'failed' | 'none';
    audioBlob?: Blob | string;
}

export interface Task {
    id?: number;
    title: string;
    dueDate: string | null;
    priority: 'low' | 'medium' | 'high';
    status: 'todo' | 'inprogress' | 'done';
    sessionId?: number;
    sessionName?: string;
    timestamp: number;
}

export enum StorageType {
    BROWSER = 'browser',
    SERVER = 'server',
    FILESYSTEM = 'filesystem'
}

export interface LocalModel {
    id: string;
    name: string;
    parameters: string;
    provider: string;
    type: 'transcription' | 'analysis';
    huggingFacePath: string;
    downloaded?: boolean;
}

export interface ModelConfig {
    transcriptionModelId: string;
    analysisModelId: string;
}

export interface StorageProvider {
    saveSession(session: Session): Promise<number>;
    getAllSessions(): Promise<Session[]>;
    getSession(id: number): Promise<Session | undefined>;
    updateSession(session: Session): Promise<void>;
    deleteSession(id: number): Promise<void>;
    saveAudioBlob(sessionId: number, blob: Blob): Promise<void>;
    getAudioBlob(sessionId: number): Promise<Blob | undefined>;
    saveTask(task: Task): Promise<number>;
    saveTasks(tasks: Task[]): Promise<number[]>;
    getAllTasks(): Promise<Task[]>;
    updateTask(task: Task): Promise<void>;
    deleteTask(id: number): Promise<void>;
    saveConfig(key: string, value: unknown): Promise<void>;
    getConfig(key: string): Promise<unknown>;
}
