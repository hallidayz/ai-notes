export class GhostTextService {
    private static instance: GhostTextService;
    private worker: Worker | null = null;
    private initializationPromise: Promise<void> | null = null;
    private pendingRequests: Map<string, { resolve: (val: string) => void, reject: (err: unknown) => void }> = new Map();
    private messageCounter = 0;
    // Add testing toggle
    private isTestEnvironment = false;

    private constructor() {
        if (typeof window !== 'undefined' && typeof Worker !== 'undefined') {
            try {
                this.worker = new Worker(new URL('../workers/GhostTextWorker.ts', import.meta.url), {
                    type: 'module'
                });

                this.worker.onmessage = this.handleWorkerMessage.bind(this);
            } catch {
                // If it fails (e.g. in node tests), mark as test env
                this.isTestEnvironment = true;
            }
        } else {
            this.isTestEnvironment = true;
        }
    }

    private handleWorkerMessage(event: MessageEvent) {
        const data = event.data;
        if (data.type === 'generation_complete') {
            const pending = this.pendingRequests.get(data.messageId);
            if (pending) {
                pending.resolve(data.suggestion);
                this.pendingRequests.delete(data.messageId);
            }
        }
    }

    public static getInstance(): GhostTextService {
        if (!GhostTextService.instance) {
            GhostTextService.instance = new GhostTextService();
        }
        return GhostTextService.instance;
    }

    public async initialize(): Promise<void> {
        if (this.isTestEnvironment) return Promise.resolve();

        if (!this.worker) return;

        if (!this.initializationPromise) {
            this.initializationPromise = new Promise((resolve) => {
                const initHandler = (event: MessageEvent) => {
                    if (event.data.type === 'init_complete') {
                        if (this.worker) {
                            this.worker.removeEventListener('message', initHandler);
                        }
                        resolve();
                    }
                };
                this.worker!.addEventListener('message', initHandler);
                this.worker!.postMessage({ type: 'init' });
            });
        }
        return this.initializationPromise;
    }

    public async generateSuggestion(context: string): Promise<string> {
        if (!context || context.trim() === '') return '';

        // Mock behavior for testing
        if (this.isTestEnvironment) {
            return '';
        }

        if (!this.worker) return '';

        // Ensure init
        if (!this.initializationPromise) {
            this.initialize();
        }

        return new Promise((resolve, reject) => {
            const messageId = `msg_${++this.messageCounter}`;
            this.pendingRequests.set(messageId, { resolve, reject });
            this.worker!.postMessage({
                type: 'generate',
                messageId,
                context
            });
        });
    }
}
