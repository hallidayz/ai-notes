
import { Session, Task, StorageProvider, CalendarConnection } from '../types';
import { NotesDB } from './notesDB';

export interface FileSystemDirectoryHandle {
    name: string;
    queryPermission(options: { mode: string }): Promise<string>;
    getFileHandle(name: string, options?: { create: boolean }): Promise<unknown>;
    removeEntry(name: string): Promise<void>;
}

export class FileSystemStorageProvider implements StorageProvider {
    private dirHandle: FileSystemDirectoryHandle | null = null;

    constructor(private db: NotesDB) {}

    async init() {
        try {
            const handle = await this.db.getConfig('dirHandle');
            if (handle) {
                const permission = await (handle as FileSystemDirectoryHandle).queryPermission({ mode: 'readwrite' });
                if (permission === 'granted') {
                    this.dirHandle = handle as FileSystemDirectoryHandle;
                }
            }
        } catch (e) {
            console.error('Failed to init FileSystemStorageProvider', e);
        }
    }

    async setDirHandle(handle: FileSystemDirectoryHandle) {
        this.dirHandle = handle;
        await this.db.saveConfig('dirHandle', handle);
    }

    private async getFileHandle(name: string, create = false) {
        if (!this.dirHandle) throw new Error('Directory handle not set');
        return await this.dirHandle.getFileHandle(name, { create });
    }

    private async readFile(name: string) {
        try {
            const fileHandle = await this.getFileHandle(name);
            const file = await (fileHandle as { getFile: () => Promise<File> }).getFile();
            return await file.text();
        } catch {
            return null;
        }
    }

    private async writeFile(name: string, content: string) {
        const fileHandle = await this.getFileHandle(name, true);
        const writable = await (fileHandle as { createWritable: () => Promise<{ write: (c: string) => Promise<void>, close: () => Promise<void> }> }).createWritable();
        await writable.write(content);
        await writable.close();
    }

    async saveSession(session: Session) {
        const id = session.id || Date.now();
        const sessionToSave = { ...session, id };
        try {
            if (this.dirHandle) {
                await this.writeFile(`session_${id}.json`, JSON.stringify(sessionToSave));
            }
        } catch (e) {
            console.error('Failed to write file', e);
        }
        await this.db.addSession(sessionToSave); // Also save to IndexedDB for quick querying
        return id;
    }

    async getAllSessions() {
        return this.db.getAllSessions();
    }

    async getSession(id: number) {
        try {
            if (this.dirHandle) {
                const content = await this.readFile(`session_${id}.json`);
                if (content) return JSON.parse(content);
            }
        } catch (e) {
            console.error('Failed to read file', e);
        }
        return this.db.getSession(id);
    }

    async updateSession(session: Session) {
        try {
            if (this.dirHandle) {
                await this.writeFile(`session_${session.id}.json`, JSON.stringify(session));
            }
        } catch (e) {
            console.error('Failed to write file', e);
        }
        await this.db.updateSession(session);
    }

    async deleteSession(id: number) {
        try {
            if (this.dirHandle) {
                await this.dirHandle.removeEntry(`session_${id}.json`);
            }
        } catch (e) {
            console.error('Failed to delete file', e);
        }
        await this.db.deleteSession(id);
    }

    async saveAudioBlob(sessionId: number, blob: Blob) {
        // Audio blobs are saved to IndexedDB for now
        return this.db.saveAudioBlob(sessionId, blob);
    }

    async getAudioBlob(sessionId: number) {
        return this.db.getAudioBlob(sessionId);
    }

    async saveTask(task: Task) {
        return this.db.addTask(task);
    }

    async getAllTasks() {
        return this.db.getAllTasks();
    }

    async updateTask(task: Task) {
        return this.db.updateTask(task);
    }

    async deleteTask(id: number) {
        return this.db.deleteTask(id);
    }

    async saveCalendarConnection(connection: CalendarConnection) {
        return this.db.addCalendarConnection(connection);
    }

    async getAllCalendarConnections() {
        return this.db.getAllCalendarConnections();
    }

    async deleteCalendarConnection(id: number) {
        return this.db.deleteCalendarConnection(id);
    }

    async saveConfig(key: string, value: unknown) {
        return this.db.saveConfig(key, value);
    }

    async getConfig(key: string) {
        return this.db.getConfig(key);
    }
}
export class IndexedDBProvider implements StorageProvider {
    constructor(private db: NotesDB) {}
    async saveSession(session: Session) { return this.db.addSession(session); }
    async getAllSessions() { return this.db.getAllSessions(); }
    async getSession(id: number) { return this.db.getSession(id); }
    async updateSession(session: Session) { return this.db.updateSession(session); }
    async deleteSession(id: number) { return this.db.deleteSession(id); }
    async saveAudioBlob(sessionId: number, blob: Blob) { return this.db.saveAudioBlob(sessionId, blob); }
    async getAudioBlob(sessionId: number) { return this.db.getAudioBlob(sessionId); }
    async saveTask(task: Task) { return this.db.addTask(task); }
    async getAllTasks() { return this.db.getAllTasks(); }
    async updateTask(task: Task) { return this.db.updateTask(task); }
    async deleteTask(id: number) { return this.db.deleteTask(id); }
    async saveCalendarConnection(connection: CalendarConnection) { return this.db.addCalendarConnection(connection); }
    async getAllCalendarConnections() { return this.db.getAllCalendarConnections(); }
    async deleteCalendarConnection(id: number) { return this.db.deleteCalendarConnection(id); }
    async saveConfig(key: string, value: unknown) { return this.db.saveConfig(key, value); }
    async getConfig(key: string) { return this.db.getConfig(key); }
}

export class ServerStorageProvider implements StorageProvider {
    private sessionCache = new Map<string, { session: Session; encrypted: string }>();
    private tasksCache: { tasks: Task[]; encrypted: string } | null = null;
    private connectionsCache: { connections: CalendarConnection[]; encrypted: string } | null = null;
    private configCache = new Map<string, { value: unknown; encrypted: string }>();

    constructor(private pin: string) {}

    async saveSession(session: Session) {
        const { CryptoService } = await import('./cryptoService');
        const id = session.id || Date.now();
        const sessionToSave = { ...session, id };
        const encryptedData = await CryptoService.encrypt(JSON.stringify(sessionToSave), this.pin);
        await fetch('/api/storage/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id, data: { encrypted: encryptedData } })
        });
        this.sessionCache.set(id.toString(), { session: sessionToSave, encrypted: encryptedData });
        return id;
    }

    async getAllSessions() {
        const { CryptoService } = await import('./cryptoService');
        const res = await fetch('/api/storage/list');
        const items = await res.json() as Array<{ id: string; data?: { encrypted?: string } }>;
        
        const decryptionPromises = items
            .filter(item => item.id !== 'tasks_list' && item.id !== 'calendar_connections' && !item.id.startsWith('config_') && item.data?.encrypted)
            .map(async (item) => {
                const encrypted = item.data!.encrypted!;
                const cached = this.sessionCache.get(item.id);
                if (cached && cached.encrypted === encrypted) {
                    return cached.session;
                }
                try {
                    const decrypted = await CryptoService.decrypt(encrypted, this.pin);
                    const session = JSON.parse(decrypted) as Session;
                    this.sessionCache.set(item.id, { session, encrypted });
                    return session;
                } catch (e) {
                    console.error("Failed to decrypt session", item.id, e);
                    return null;
                }
            });
            
        const resolvedSessions = await Promise.all(decryptionPromises);
        return resolvedSessions.filter((s: Session | null): s is Session => s !== null);
    }

    async getSession(id: number) {
        const cached = this.sessionCache.get(id.toString());
        if (cached) {
            return cached.session;
        }
        const sessions = await this.getAllSessions();
        return sessions.find((s: Session) => s.id === id);
    }

    async updateSession(session: Session) {
        await this.saveSession(session);
    }

    async deleteSession(id: number) {
        await fetch(`/api/storage/${id}`, { method: 'DELETE' });
        this.sessionCache.delete(id.toString());
    }

    async saveAudioBlob(sessionId: number, blob: Blob): Promise<void> {
        const { CryptoService } = await import('./cryptoService');
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsArrayBuffer(blob);
            reader.onloadend = async () => {
                try {
                    const buffer = reader.result as ArrayBuffer;
                    const encryptedBase64 = await CryptoService.encryptBuffer(buffer, this.pin);
                    const session = await this.getSession(sessionId);
                    if (session) {
                        session.audioBlob = encryptedBase64 as unknown as Blob; 
                        await this.saveSession(session);
                    }
                    resolve();
                } catch (err) {
                    reject(err);
                }
            };
            reader.onerror = () => reject(reader.error);
        });
    }

    async getAudioBlob(sessionId: number) {
        const { CryptoService } = await import('./cryptoService');
        const session = await this.getSession(sessionId);
        if (session?.audioBlob && typeof session.audioBlob === 'string') {
            const decryptedBuffer = await CryptoService.decryptBuffer(session.audioBlob, this.pin);
            return new Blob([decryptedBuffer]);
        }
        return undefined;
    }

    async saveTask(task: Task) {
        const { CryptoService } = await import('./cryptoService');
        const id = task.id || Date.now();
        const tasks = await this.getAllTasks();
        tasks.push({ ...task, id });
        const encryptedData = await CryptoService.encrypt(JSON.stringify(tasks), this.pin);
        await fetch('/api/storage/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: 'tasks_list', data: { encrypted: encryptedData } })
        });
        this.tasksCache = { tasks, encrypted: encryptedData };
        return id;
    }

    async getAllTasks() {
        const { CryptoService } = await import('./cryptoService');
        try {
            const res = await fetch('/api/storage/list');
            const files = await res.json() as Array<{ id: string; data?: { encrypted?: string } }>;
            const tasksFile = files.find(f => f.id === 'tasks_list');
            if (tasksFile?.data?.encrypted) {
                const encrypted = tasksFile.data.encrypted;
                if (this.tasksCache && this.tasksCache.encrypted === encrypted) {
                    return this.tasksCache.tasks;
                }
                const decrypted = await CryptoService.decrypt(encrypted, this.pin);
                const tasks = JSON.parse(decrypted) as Task[];
                this.tasksCache = { tasks, encrypted };
                return tasks;
            }
            return [];
        } catch { return []; }
    }

    async updateTask(task: Task) {
        const { CryptoService } = await import('./cryptoService');
        const tasks = await this.getAllTasks();
        const index = tasks.findIndex(t => t.id === task.id);
        if (index !== -1) {
            tasks[index] = task;
            const encryptedData = await CryptoService.encrypt(JSON.stringify(tasks), this.pin);
            await fetch('/api/storage/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: 'tasks_list', data: { encrypted: encryptedData } })
            });
            this.tasksCache = { tasks, encrypted: encryptedData };
        }
    }

    async deleteTask(id: number) {
        const { CryptoService } = await import('./cryptoService');
        const tasks = await this.getAllTasks();
        const filtered = tasks.filter(t => t.id !== id);
        const encryptedData = await CryptoService.encrypt(JSON.stringify(filtered), this.pin);
        await fetch('/api/storage/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: 'tasks_list', data: { encrypted: encryptedData } })
        });
        this.tasksCache = { tasks: filtered, encrypted: encryptedData };
    }

    async saveCalendarConnection(connection: CalendarConnection) {
        const { CryptoService } = await import('./cryptoService');
        const id = connection.id || Date.now();
        const connections = await this.getAllCalendarConnections();
        connections.push({ ...connection, id });
        const encryptedData = await CryptoService.encrypt(JSON.stringify(connections), this.pin);
        await fetch('/api/storage/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: 'calendar_connections', data: { encrypted: encryptedData } })
        });
        this.connectionsCache = { connections, encrypted: encryptedData };
        return id;
    }

    async getAllCalendarConnections() {
        const { CryptoService } = await import('./cryptoService');
        try {
            const res = await fetch('/api/storage/list');
            const files = await res.json() as Array<{ id: string; data?: { encrypted?: string } }>;
            const connFile = files.find(f => f.id === 'calendar_connections');
            if (connFile?.data?.encrypted) {
                const encrypted = connFile.data.encrypted;
                if (this.connectionsCache && this.connectionsCache.encrypted === encrypted) {
                    return this.connectionsCache.connections;
                }
                const decrypted = await CryptoService.decrypt(encrypted, this.pin);
                const connections = JSON.parse(decrypted) as CalendarConnection[];
                this.connectionsCache = { connections, encrypted };
                return connections;
            }
            return [];
        } catch { return []; }
    }

    async deleteCalendarConnection(id: number) {
        const { CryptoService } = await import('./cryptoService');
        const connections = await this.getAllCalendarConnections();
        const filtered = connections.filter(c => c.id !== id);
        const encryptedData = await CryptoService.encrypt(JSON.stringify(filtered), this.pin);
        await fetch('/api/storage/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: 'calendar_connections', data: { encrypted: encryptedData } })
        });
        this.connectionsCache = { connections: filtered, encrypted: encryptedData };
    }

    async saveConfig(key: string, value: unknown) {
        const { CryptoService } = await import('./cryptoService');
        const encryptedData = await CryptoService.encrypt(JSON.stringify(value), this.pin);
        await fetch('/api/storage/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: `config_${key}`, data: { encrypted: encryptedData } })
        });
        this.configCache.set(key, { value, encrypted: encryptedData });
    }

    async getConfig(key: string) {
        const { CryptoService } = await import('./cryptoService');
        try {
            const res = await fetch('/api/storage/list');
            const files = await res.json() as Array<{ id: string; data?: { encrypted?: string } }>;
            const configFile = files.find(f => f.id === `config_${key}`);
            if (configFile?.data?.encrypted) {
                const encrypted = configFile.data.encrypted;
                const cached = this.configCache.get(key);
                if (cached && cached.encrypted === encrypted) {
                    return cached.value;
                }
                const decrypted = await CryptoService.decrypt(encrypted, this.pin);
                const value = JSON.parse(decrypted);
                this.configCache.set(key, { value, encrypted });
                return value;
            }
            return undefined;
        } catch { return undefined; }
    }
}
