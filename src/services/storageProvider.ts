
import { Session, Task, StorageProvider, CalendarConnection } from '../types';
import { NotesDB } from './notesDB';

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
    async saveTasks(tasks: Task[]) { return this.db.addTasks(tasks); }
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
        return id;
    }
    async getAllSessions() {
        const { CryptoService } = await import('./cryptoService');
        const res = await fetch('/api/storage/list');
        const items = await res.json();
        const sessionPromises = items.map(async (item: any) => {
            if (item.id !== 'tasks_list' && item.id !== 'calendar_connections' && item.data.encrypted) {
                try {
                    const decrypted = await CryptoService.decrypt(item.data.encrypted, this.pin);
                    return JSON.parse(decrypted);
                } catch (e) {
                    console.error("Failed to decrypt session", item.id, e);
                    return null;
                }
            }
            return null;
        });
        const resolvedSessions = await Promise.all(sessionPromises);
        return resolvedSessions.filter(Boolean) as Session[];
    }
    async getSession(id: number) {
        const { CryptoService } = await import('./cryptoService');
        try {
            const res = await fetch(`/api/storage/item/${id}`);
            if (!res.ok) return undefined;
            const item = await res.json();
            if (item?.data?.encrypted) {
                const decrypted = await CryptoService.decrypt(item.data.encrypted, this.pin);
                return JSON.parse(decrypted);
            }
        } catch (e) {
            console.error("Failed to fetch/decrypt session", id, e);
        }
        return undefined;
    }
    async updateSession(session: Session) {
        await this.saveSession(session);
    }
    async deleteSession(id: number) {
        await fetch(`/api/storage/${id}`, { method: 'DELETE' });
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
        return id;
    }
    async saveTasks(tasksToSave: Task[]) {
        const { CryptoService } = await import('./cryptoService');
        const tasks = await this.getAllTasks();

        const ids: number[] = [];
        let timeBase = Date.now();

        for (const task of tasksToSave) {
            const id = task.id || ++timeBase;
            ids.push(id);
            tasks.push({ ...task, id });
        }

        const encryptedData = await CryptoService.encrypt(JSON.stringify(tasks), this.pin);
        await fetch('/api/storage/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: 'tasks_list', data: { encrypted: encryptedData } })
        });
        return ids;
    }
    async getAllTasks() {
        const { CryptoService } = await import('./cryptoService');
        try {
            const res = await fetch('/api/storage/list');
            const files = await res.json();
            const tasksFile = files.find((f: { id: string, data: { encrypted?: string } }) => f.id === 'tasks_list');
            if (tasksFile?.data?.encrypted) {
                const decrypted = await CryptoService.decrypt(tasksFile.data.encrypted, this.pin);
                return JSON.parse(decrypted);
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
        return id;
    }
    async getAllCalendarConnections() {
        const { CryptoService } = await import('./cryptoService');
        try {
            const res = await fetch('/api/storage/list');
            const files = await res.json();
            const connFile = files.find((f: { id: string, data: { encrypted?: string } }) => f.id === 'calendar_connections');
            if (connFile?.data?.encrypted) {
                const decrypted = await CryptoService.decrypt(connFile.data.encrypted, this.pin);
                return JSON.parse(decrypted);
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
    }
    async saveConfig(key: string, value: unknown) {
        const { CryptoService } = await import('./cryptoService');
        const encryptedData = await CryptoService.encrypt(JSON.stringify(value), this.pin);
        await fetch('/api/storage/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: `config_${key}`, data: { encrypted: encryptedData } })
        });
    }
    async getConfig(key: string) {
        const { CryptoService } = await import('./cryptoService');
        try {
            const res = await fetch('/api/storage/list');
            const files = await res.json();
            const configFile = files.find((f: { id: string, data: { encrypted?: string } }) => f.id === `config_${key}`);
            if (configFile?.data?.encrypted) {
                const decrypted = await CryptoService.decrypt(configFile.data.encrypted, this.pin);
                return JSON.parse(decrypted);
            }
            return undefined;
        } catch { return undefined; }
    }
}
