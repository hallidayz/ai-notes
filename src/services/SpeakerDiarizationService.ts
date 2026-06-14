/**
 * Speaker Diarization Service
 * Handles voice-based speaker identification using embeddings
 *
 * TODO: Integrate speaker verification model (speechbrain/spkrec-ecapa-voxceleb)
 * TODO: Extract voice embeddings from audio segments
 * TODO: Store speaker profiles in IndexedDB
 * TODO: Match new audio to known speakers
 */

export interface SpeakerProfile {
    id: string;
    name: string;
    embeddings: Float32Array[];
    createdAt: number;
    lastSeen: number;
    meetingCount: number;
    confidenceThreshold: number;
}

export interface SpeakerSegment {
    speakerId: string;
    speakerName: string;
    startTime: number;
    endTime: number;
    confidence: number;
    embedding?: Float32Array;
}

export class SpeakerDiarizationService {
    private static instance: SpeakerDiarizationService | null = null;
    private speakerProfiles: Map<string, SpeakerProfile> = new Map();
    private readonly DB_NAME = 'speakerProfilesDB';
    private readonly STORE_NAME = 'speakers';
    private speakerModel: unknown = null;

    private constructor() {}

    public static getInstance(): SpeakerDiarizationService {
        if (!this.instance) {
            this.instance = new SpeakerDiarizationService();
        }
        return this.instance;
    }

    /**
     * Initialize speaker verification model
     * TODO: Load speechbrain/spkrec-ecapa-voxceleb model
     */
    async initialize(): Promise<void> {
        try {
            // Load speaker profiles from IndexedDB
            await this.loadSpeakerProfiles();

            // Load speaker verification model
            const transformers = await import('@huggingface/transformers');
            const env = transformers.env;
            if (typeof env !== 'undefined') {
                env.remoteHost = typeof window !== 'undefined' && window.location
                    ? window.location.origin
                    : 'http://localhost:4783';
            }
            this.speakerModel = await transformers.pipeline(
                'feature-extraction',
                'speechbrain/spkrec-ecapa-voxceleb'
            );

            console.log('Speaker Diarization Service initialized');
        } catch (error) {
            console.error('Error initializing speaker diarization:', error);
        }
    }

    /**
     * Extract voice embedding from audio segment
     * TODO: Implement using speaker verification model
     */
    async extractEmbedding(audioBuffer: AudioBuffer, startTime: number, endTime: number): Promise<Float32Array | null> {
        if (!this.speakerModel) {
            console.warn('Speaker model not loaded, cannot extract embedding');
            return null;
        }

        try {
            // Extract audio segment
            const sampleRate = audioBuffer.sampleRate;
            const startSample = Math.floor(startTime * sampleRate);
            const endSample = Math.floor(endTime * sampleRate);
            const channelData = audioBuffer.getChannelData(0);
            const segment = channelData.slice(startSample, endSample);

            // Process through speaker verification model
            const output = await this.speakerModel(segment);
            return new Float32Array(output.data);
        } catch (error) {
            console.error('Error extracting embedding:', error);
            return null;
        }
    }

    /**
     * Match audio segment to known speaker
     */
    async matchSpeaker(embedding: Float32Array): Promise<{ speakerId: string; confidence: number } | null> {
        if (this.speakerProfiles.size === 0) {
            return null;
        }

        let bestMatch: { speakerId: string; confidence: number } | null = null;
        let bestConfidence = 0;

        for (const [speakerId, profile] of this.speakerProfiles.entries()) {
            // Calculate cosine similarity with stored embeddings
            const similarities = profile.embeddings.map(storedEmbedding =>
                this.cosineSimilarity(embedding, storedEmbedding)
            );

            const avgSimilarity = similarities.reduce((a, b) => a + b, 0) / similarities.length;

            if (avgSimilarity > bestConfidence && avgSimilarity >= profile.confidenceThreshold) {
                bestConfidence = avgSimilarity;
                bestMatch = { speakerId, confidence: avgSimilarity };
            }
        }

        return bestMatch;
    }

    /**
     * Calculate cosine similarity between two embeddings
     */
    private cosineSimilarity(a: Float32Array, b: Float32Array): number {
        if (a.length !== b.length) return 0;

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        const denominator = Math.sqrt(normA) * Math.sqrt(normB);
        return denominator === 0 ? 0 : dotProduct / denominator;
    }

    /**
     * Save speaker profile
     */
    async saveSpeakerProfile(profile: SpeakerProfile): Promise<void> {
        this.speakerProfiles.set(profile.id, profile);
        await this.saveToIndexedDB(profile);
    }

    /**
     * Get speaker profile by ID
     */
    getSpeakerProfile(speakerId: string): SpeakerProfile | undefined {
        return this.speakerProfiles.get(speakerId);
    }

    /**
     * Update speaker name
     */
    async updateSpeakerName(speakerId: string, newName: string): Promise<void> {
        const profile = this.speakerProfiles.get(speakerId);
        if (profile) {
            profile.name = newName;
            profile.lastSeen = Date.now();
            await this.saveToIndexedDB(profile);
        }
    }

    /**
     * Add embedding to speaker profile
     */
    async addEmbeddingToSpeaker(speakerId: string, embedding: Float32Array): Promise<void> {
        const profile = this.speakerProfiles.get(speakerId);
        if (profile) {
            profile.embeddings.push(embedding);
            // Keep only last 10 embeddings per speaker
            if (profile.embeddings.length > 10) {
                profile.embeddings.shift();
            }
            profile.lastSeen = Date.now();
            await this.saveToIndexedDB(profile);
        }
    }

    /**
     * Create new speaker profile
     */
    async createSpeakerProfile(name: string, initialEmbedding: Float32Array): Promise<string> {
        const speakerId = `speaker_${Date.now()}_${crypto.randomUUID()}`;
        const profile: SpeakerProfile = {
            id: speakerId,
            name,
            embeddings: [initialEmbedding],
            createdAt: Date.now(),
            lastSeen: Date.now(),
            meetingCount: 1,
            confidenceThreshold: 0.7 // Default threshold
        };

        await this.saveSpeakerProfile(profile);
        return speakerId;
    }

    /**
     * Load speaker profiles from IndexedDB
     */
    private async loadSpeakerProfiles(): Promise<void> {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.DB_NAME, 1);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                const db = request.result;
                if (!db.objectStoreNames.contains(this.STORE_NAME)) {
                    resolve();
                    return;
                }

                const transaction = db.transaction([this.STORE_NAME], 'readonly');
                const store = transaction.objectStore(this.STORE_NAME);
                const getAllRequest = store.getAll();

                getAllRequest.onsuccess = () => {
                    const profiles = getAllRequest.result;
                    profiles.forEach((profile: SpeakerProfile) => {
                        // Convert embeddings back to Float32Array
                        profile.embeddings = profile.embeddings.map((emb: number[] | Float32Array) =>
                            new Float32Array(emb)
                        );
                        this.speakerProfiles.set(profile.id, profile);
                    });
                    resolve();
                };

                getAllRequest.onerror = () => reject(getAllRequest.error);
            };

            request.onupgradeneeded = (event) => {
                const db = (event.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(this.STORE_NAME)) {
                    const objectStore = db.createObjectStore(this.STORE_NAME, { keyPath: 'id' });
                    objectStore.createIndex('name', 'name', { unique: false });
                    objectStore.createIndex('lastSeen', 'lastSeen', { unique: false });
                }
            };
        });
    }

    /**
     * Save speaker profile to IndexedDB
     */
    private async saveToIndexedDB(profile: SpeakerProfile): Promise<void> {
        return new Promise((resolve, reject) => {
            // First, ensure the database and store exist
            const ensureStoreExists = (): Promise<IDBDatabase> => {
                return new Promise((resolveStore, rejectStore) => {
                    const checkRequest = indexedDB.open(this.DB_NAME, 1);

                    checkRequest.onerror = () => rejectStore(checkRequest.error);

                    checkRequest.onupgradeneeded = (event) => {
                        const db = (event.target as IDBOpenDBRequest).result;
                        if (!db.objectStoreNames.contains(this.STORE_NAME)) {
                            const objectStore = db.createObjectStore(this.STORE_NAME, { keyPath: 'id' });
                            objectStore.createIndex('name', 'name', { unique: false });
                            objectStore.createIndex('lastSeen', 'lastSeen', { unique: false });
                        }
                    };

                    checkRequest.onsuccess = () => {
                        const db = checkRequest.result;
                        // If store doesn't exist, we need to upgrade
                        if (!db.objectStoreNames.contains(this.STORE_NAME)) {
                            db.close();
                            // Reopen with version bump to trigger upgrade
                            const upgradeRequest = indexedDB.open(this.DB_NAME, 2);
                            upgradeRequest.onupgradeneeded = (event) => {
                                const upgradeDb = (event.target as IDBOpenDBRequest).result;
                                if (!upgradeDb.objectStoreNames.contains(this.STORE_NAME)) {
                                    const objectStore = upgradeDb.createObjectStore(this.STORE_NAME, { keyPath: 'id' });
                                    objectStore.createIndex('name', 'name', { unique: false });
                                    objectStore.createIndex('lastSeen', 'lastSeen', { unique: false });
                                }
                            };
                            upgradeRequest.onsuccess = () => resolveStore(upgradeRequest.result);
                            upgradeRequest.onerror = () => rejectStore(upgradeRequest.error);
                        } else {
                            resolveStore(db);
                        }
                    };
                });
            };

            ensureStoreExists().then((db) => {
                // Now we can safely create a transaction since the store exists
                const transaction = db.transaction([this.STORE_NAME], 'readwrite');
                const store = transaction.objectStore(this.STORE_NAME);

                // Convert Float32Array to regular array for storage
                const profileToStore = {
                    ...profile,
                    embeddings: profile.embeddings.map(emb => Array.from(emb))
                };

                const putRequest = store.put(profileToStore);
                putRequest.onsuccess = () => resolve();
                putRequest.onerror = () => reject(putRequest.error);
            }).catch((error) => {
                reject(error);
            });
        });
    }

    /**
     * Get all speaker profiles
     */
    getAllSpeakers(): SpeakerProfile[] {
        return Array.from(this.speakerProfiles.values());
    }

    /**
     * Delete speaker profile
     */
    async deleteSpeakerProfile(speakerId: string): Promise<void> {
        this.speakerProfiles.delete(speakerId);

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.DB_NAME, 1);
            request.onsuccess = () => {
                const db = request.result;
                const transaction = db.transaction([this.STORE_NAME], 'readwrite');
                const store = transaction.objectStore(this.STORE_NAME);
                const deleteRequest = store.delete(speakerId);
                deleteRequest.onsuccess = () => resolve();
                deleteRequest.onerror = () => reject(deleteRequest.error);
            };
            request.onerror = () => reject(request.error);
        });
    }
}
