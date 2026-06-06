
import { Session, Task, CalendarConnection } from '../types';

export class NotesDB {
    private db: IDBDatabase | null = null;
    private readonly DB_NAME = 'AINotesDB';
    private readonly SESSIONS_STORE = 'sessions';
    private readonly TASKS_STORE = 'tasks';
    private readonly CONFIG_STORE = 'config';
    private readonly CALENDAR_STORE = 'calendar_connections';

    constructor() {
        this.init();
    }

    private init(): Promise<void> {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.DB_NAME, 3);

            request.onupgradeneeded = (event) => {
                const db = (event.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(this.SESSIONS_STORE)) {
                    db.createObjectStore(this.SESSIONS_STORE, { keyPath: 'id', autoIncrement: true });
                }
                if (!db.objectStoreNames.contains(this.TASKS_STORE)) {
                    const taskStore = db.createObjectStore(this.TASKS_STORE, { keyPath: 'id', autoIncrement: true });
                    taskStore.createIndex('timestamp', 'timestamp', { unique: false });
                }
                if (!db.objectStoreNames.contains(this.CONFIG_STORE)) {
                    db.createObjectStore(this.CONFIG_STORE, { keyPath: 'key' });
                }
                if (!db.objectStoreNames.contains(this.CALENDAR_STORE)) {
                    db.createObjectStore(this.CALENDAR_STORE, { keyPath: 'id', autoIncrement: true });
                }
            };

            request.onsuccess = (event) => {
                this.db = (event.target as IDBOpenDBRequest).result;
                resolve();
            };

            request.onerror = (event) => {
                console.error("Database error: ", (event.target as IDBOpenDBRequest).error);
                reject((event.target as IDBOpenDBRequest).error);
            };
        });
    }

    private async getDb(): Promise<IDBDatabase> {
        if (!this.db) {
            await this.init();
        }
        return this.db!;
    }

    public async saveConfig(key: string, value: unknown): Promise<void> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.CONFIG_STORE, 'readwrite');
            const store = transaction.objectStore(this.CONFIG_STORE);
            const request = store.put({ key, value });
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    public async getConfig(key: string): Promise<unknown> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.CONFIG_STORE, 'readonly');
            const store = transaction.objectStore(this.CONFIG_STORE);
            const request = store.get(key);
            request.onsuccess = () => resolve(request.result?.value);
            request.onerror = () => reject(request.error);
        });
    }

    public async addSession(session: Session): Promise<number> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readwrite');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.add(session);
            request.onsuccess = () => resolve(request.result as number);
            request.onerror = () => reject(request.error);
        });
    }

    public async getAllSessions(): Promise<Session[]> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readonly');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.getAll();
            request.onsuccess = () => {
                const sortedSessions = request.result.sort((a: Session, b: Session) => b.timestamp - a.timestamp);
                resolve(sortedSessions);
            };
            request.onerror = () => reject(request.error);
        });
    }

    public async getSession(id: number): Promise<Session | undefined> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readonly');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.get(id);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }
    
    public async updateSession(session: Session): Promise<void> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readwrite');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.put(session);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    public async deleteSession(id: number): Promise<void> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readwrite');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.delete(id);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    // Task Methods
    public async addTask(task: Task): Promise<number> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.TASKS_STORE, 'readwrite');
            const store = transaction.objectStore(this.TASKS_STORE);
            const request = store.add(task);
            request.onsuccess = () => resolve(request.result as number);
            request.onerror = () => reject(request.error);
        });
    }

    public async addTasks(tasks: Task[]): Promise<number[]> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.TASKS_STORE, 'readwrite');
            const store = transaction.objectStore(this.TASKS_STORE);
            const ids: number[] = [];
            let completedCount = 0;

            if (tasks.length === 0) {
                return resolve([]);
            }

            tasks.forEach((task, index) => {
                const request = store.add(task);
                request.onsuccess = () => {
                    ids[index] = request.result as number;
                    completedCount++;
                    if (completedCount === tasks.length) {
                        resolve(ids);
                    }
                };
                request.onerror = () => {
                    transaction.abort();
                    reject(request.error);
                };
            });
        });
    }

    public async getAllTasks(): Promise<Task[]> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.TASKS_STORE, 'readonly');
            const store = transaction.objectStore(this.TASKS_STORE);
            const index = store.index('timestamp');
            const request = index.getAll();
            request.onsuccess = () => {
                const sortedTasks = request.result.sort((a: Task, b: Task) => b.timestamp - a.timestamp);
                resolve(sortedTasks);
            };
            request.onerror = () => reject(request.error);
        });
    }

    public async updateTask(task: Task): Promise<void> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.TASKS_STORE, 'readwrite');
            const store = transaction.objectStore(this.TASKS_STORE);
            const request = store.put(task);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    public async deleteTask(id: number): Promise<void> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.TASKS_STORE, 'readwrite');
            const store = transaction.objectStore(this.TASKS_STORE);
            const request = store.delete(id);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    // Audio Blob Methods
    public async saveAudioBlob(sessionId: number, blob: Blob): Promise<void> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readwrite');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const getRequest = store.get(sessionId);
            getRequest.onsuccess = () => {
                const session = getRequest.result;
                if (session) {
                    session.audioBlob = blob;
                    const putRequest = store.put(session);
                    putRequest.onsuccess = () => resolve();
                    putRequest.onerror = () => reject(putRequest.error);
                } else {
                    reject(new Error("Session not found"));
                }
            };
            getRequest.onerror = () => reject(getRequest.error);
        });
    }

    public async getAudioBlob(sessionId: number): Promise<Blob | undefined> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readonly');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.get(sessionId);
            request.onsuccess = () => resolve(request.result?.audioBlob);
            request.onerror = () => reject(request.error);
        });
    }

    // Calendar Connection Methods
    public async addCalendarConnection(connection: CalendarConnection): Promise<number> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.CALENDAR_STORE, 'readwrite');
            const store = transaction.objectStore(this.CALENDAR_STORE);
            const request = store.add(connection);
            request.onsuccess = () => resolve(request.result as number);
            request.onerror = () => reject(request.error);
        });
    }

    public async getAllCalendarConnections(): Promise<CalendarConnection[]> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.CALENDAR_STORE, 'readonly');
            const store = transaction.objectStore(this.CALENDAR_STORE);
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    public async deleteCalendarConnection(id: number): Promise<void> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.CALENDAR_STORE, 'readwrite');
            const store = transaction.objectStore(this.CALENDAR_STORE);
            const request = store.delete(id);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }
}
