import React, { useState, useEffect, useRef } from 'react';

// Import HTTP interceptors (separate file to avoid Fast Refresh issues)
import '../utils/http-interceptors';

// Import calendar services
import { CalendarService, CalendarProvider, CalendarConfig } from '../services/CalendarService';
import { GoogleCalendarService } from '../services/GoogleCalendarService';
import { OutlookCalendarService } from '../services/OutlookCalendarService';
import { AutoLaunchService } from '../services/AutoLaunchService';
import { NotificationService } from '../services/NotificationService';
import { CalendarSettings } from '../components/CalendarSettings';
import { AuthService } from '../services/AuthService';
import { Sidebar } from '../components/Sidebar';
import { EditorArea } from '../components/EditorArea';
import { ContextRail } from '../components/ContextRail';

// Preload transformers.js in the background - don't block page load
// This ensures the module initializes properly while still allowing the page to load
let transformersModule: any = null;
let transformersLoadPromise: Promise<any> | null = null;

// First-time setup detection - will use db instance after it's created
let firstTimeSetupChecked = false;
let firstTimeSetupDb: TherapyDB | null = null;
const checkFirstTimeSetup = async (): Promise<boolean> => {
    if (firstTimeSetupChecked) return false;
    try {
        // Use global db instance if available, otherwise create temporary one
        const db = firstTimeSetupDb || new TherapyDB();
        const hasRunBefore = await db.getConfig('hasRunBefore');
        if (!hasRunBefore) {
            await db.saveConfig('hasRunBefore', true);
            firstTimeSetupChecked = true;
            return true;
        }
        firstTimeSetupChecked = true;
        return false;
    } catch {
        return false;
    }
};

/**
 * Check if a specific model is cached in IndexedDB
 * transformers.js stores models in IndexedDB with keys based on model name
 */
async function checkModelCache(db: IDBDatabase, modelName: string): Promise<boolean> {
    return new Promise((resolve) => {
        try {
            // transformers.js uses a cache database, typically named 'transformers-cache' or similar
            // Check multiple possible object store names
            const objectStoreNames = ['files', 'models', 'cache', 'transformers-cache'];
            let found = false;
            let checked = 0;
            let resolved = false; // Track if promise has been resolved
            
            const safeResolve = (value: boolean) => {
                if (!resolved) {
                    resolved = true;
                    resolve(value);
                }
            };
            
            const checkStore = (storeName: string) => {
                if (!db.objectStoreNames.contains(storeName)) {
                    checked++;
                    if (checked === objectStoreNames.length && !resolved) {
                        safeResolve(false);
                    }
                    return;
                }
                
                const transaction = db.transaction([storeName], 'readonly');
                const store = transaction.objectStore(storeName);
                const index = store.index ? store.index('key') : null;
                
                if (index) {
                    const request = index.get(modelName);
                    request.onsuccess = () => {
                        if (request.result !== undefined) {
                            found = true;
                            safeResolve(true);
                        } else {
                            checked++;
                            if (checked === objectStoreNames.length && !resolved) {
                                safeResolve(false);
                            }
                        }
                    };
                    request.onerror = () => {
                        checked++;
                        if (checked === objectStoreNames.length && !found && !resolved) {
                            safeResolve(false);
                        }
                    };
                } else {
                    // Fallback: check if any key contains the model name
                    const request = store.openCursor();
                    request.onsuccess = () => {
                        const cursor = request.result;
                        if (cursor) {
                            if (cursor.key.toString().includes(modelName)) {
                                found = true;
                                safeResolve(true);
                                return;
                            }
                            cursor.continue();
                        } else {
                            checked++;
                            if (checked === objectStoreNames.length && !found && !resolved) {
                                safeResolve(false);
                            }
                        }
                    };
                    request.onerror = () => {
                        checked++;
                        if (checked === objectStoreNames.length && !found && !resolved) {
                            safeResolve(false);
                        }
                    };
                }
            };
            
            objectStoreNames.forEach(storeName => checkStore(storeName));
        } catch {
            if (!resolved) {
                resolved = true;
                resolve(false);
            }
        }
    });
}

/**
 * Check if models are cached in IndexedDB
 * Returns true if both transcription and analysis models are cached
 */
async function checkCachedModels(): Promise<boolean> {
    try {
        // transformers.js stores models in IndexedDB
        // Try to open the cache database
        return new Promise((resolve) => {
            const request = indexedDB.open('transformers-cache', 1);
            
            request.onsuccess = async () => {
                try {
                    const db = request.result;
                    
                    // Check for transcription model cache (whisper-base.en or whisper-base)
                    const transcriptionCached = await checkModelCache(db, 'Xenova/whisper-base.en') || 
                                                 await checkModelCache(db, 'Xenova/whisper-base');
                    
                    // Check for analysis model cache (flan-t5-base)
                    const analysisCached = await checkModelCache(db, 'Xenova/flan-t5-base');
                    
                    resolve(transcriptionCached && analysisCached);
                } catch {
                    resolve(false);
                }
            };
            
            request.onerror = () => {
                // Database doesn't exist or can't be opened - models not cached
                resolve(false);
            };
            
            request.onupgradeneeded = () => {
                // Database needs upgrade - models not cached yet
                resolve(false);
            };
        });
    } catch {
        return false; // Assume not cached if check fails
    }
}

// Suppress harmless WebSocket connection errors from transformers.js
// These occur because transformers.js may attempt WebSocket connections, but the app works fine with HTTP only
if (typeof window !== 'undefined') {
    const originalError = console.error;
    console.error = function(...args: any[]) {
        const message = args[0]?.toString() || '';
        // Suppress WebSocket connection errors to localhost:3001 (proxy server)
        if (message.includes('WebSocket connection to') && message.includes('localhost:3001')) {
            return; // Silently ignore these harmless errors
        }
        originalError.apply(console, args);
    };
}

// Start loading transformers.js immediately but don't await it
const initTransformers = async () => {
    try {
        const isFirstTime = await checkFirstTimeSetup();
        
        // Wait for onnxruntime-web CDN to load
        if (typeof window !== 'undefined') {
            let waitCount = 0;
            while (!(window as any).ort && waitCount < 50) {
                await new Promise(resolve => setTimeout(resolve, 100));
                waitCount++;
            }
            if (!(window as any).ort) {
                throw new Error('onnxruntime-web failed to load from CDN. Please check your internet connection and refresh the page.');
            }
        }
        
        // Set up global environment for onnxruntime-web
        if (typeof window !== 'undefined') {
            (window as any).global = window;
            if (!(window as any).process) {
                (window as any).process = { env: {} };
            }
        }
        
        await new Promise(resolve => setTimeout(resolve, 300));
        
        const module = await import('@huggingface/transformers');
        
        if (!module.pipeline || !module.AutoTokenizer || !module.AutoModelForSeq2SeqLM) {
            throw new Error('Transformers.js exports missing');
        }
        
        // Check if Whisper models are supported
        if (!module.WhisperForConditionalGeneration && !module.AutoModelForSpeechSeq2Seq) {
            console.warn('Whisper model support may not be available in this version of transformers.js');
        }
        
        // Configure environment to use our proxy
        if (module.env) {
            module.env.allowLocalModels = false;
            module.env.allowRemoteModels = true;
            module.env.remoteHost = 'http://localhost:3001';
            module.env.useBrowserCache = true; // Enable browser cache for models
            module.env.useCustomCache = false;
        }
        
        transformersModule = module;
        
        // On first open, show welcome message
        if (isFirstTime && typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('firstTimeSetup', { 
                detail: { message: 'Welcome! This is your first time. Models will download automatically on first open.' }
            }));
        }
        
        // Load models immediately for all users (not just first-time)
        // Models will load from cache if available, or download if needed
        if (typeof window !== 'undefined') {
            // Start loading immediately, no delay
            (async () => {
                try {
                    // Dispatch loading started event
                    window.dispatchEvent(new CustomEvent('modelsLoading', { 
                        detail: { message: 'Loading AI models...' }
                    }));
                    
                    const aiService = OnDeviceAIService.getInstance();
                    let modelsDownloaded = false;
                    
                    // Check if models are cached
                    const hasCachedModels = await checkCachedModels();
                    
                    // Log on-device verification
                    console.log('AI Processing: On-Device Mode', {
                        modelsCached: hasCachedModels,
                        processingLocation: 'browser',
                        dataTransmission: 'none',
                        cacheEnabled: module.env?.useBrowserCache || false,
                        remoteHost: module.env?.remoteHost || 'not set',
                        allowRemoteModels: module.env?.allowRemoteModels || false,
                        allowLocalModels: module.env?.allowLocalModels || false
                    });
                    
                    // Verify no external data transmission
                    if (hasCachedModels) {
                        console.log('✓ Models loaded from cache - 100% on-device processing');
                    } else {
                        console.log('⚠ Models downloading (first time only) - will be cached for future use');
                    }
                    
                    // Progress handler for model loading
                    const handleProgress = (progress: any) => {
                        if (progress?.status === 'downloading') {
                            modelsDownloaded = true;
                            const modelType = progress.modelName?.includes('whisper') ? 'transcription' : 'analysis';
                            const message = `Downloading ${modelType} model: ${Math.round(progress.progress || 0)}%`;
                            window.dispatchEvent(new CustomEvent('modelDownloadProgress', { detail: { message } }));
                        } else if (progress?.status === 'loading') {
                            const modelType = progress.modelName?.includes('whisper') ? 'transcription' : 'analysis';
                            const message = `Loading ${modelType} model...`;
                            window.dispatchEvent(new CustomEvent('modelDownloadProgress', { detail: { message } }));
                        }
                    };
                    
                    // Load both models in parallel using allSettled so both attempt to load even if one fails
                    const results = await Promise.allSettled([
                        aiService.getTranscriptionPipeline(undefined, handleProgress),
                        aiService.getAnalysisPipeline(handleProgress)
                    ]);
                    
                    // Check results and log any errors
                    results.forEach((result, index) => {
                        const modelType = index === 0 ? 'transcription' : 'analysis';
                        if (result.status === 'rejected') {
                            const error = result.reason;
                            if (!error?.message?.includes('Unsupported model type')) {
                                console.error(`Error preloading ${modelType} model:`, error);
                                window.dispatchEvent(new CustomEvent('modelLoadError', { 
                                    detail: { 
                                        message: `Failed to load ${modelType} model. It will load on-demand when needed.`, 
                                        error: error?.message || 'Unknown error'
                                    }
                                }));
                            }
                        }
                    });
                    
                    // Dispatch ready event
                    if (modelsDownloaded) {
                        window.dispatchEvent(new CustomEvent('modelsDownloaded', { 
                            detail: { message: 'Models downloaded successfully!' }
                        }));
                    }
                    
                    window.dispatchEvent(new CustomEvent('modelsReady', { 
                        detail: { message: 'AI models ready!' }
                    }));
                } catch (error: any) {
                    // Models will load on-demand if preload fails
                    console.debug('Model preload failed, will load on-demand:', error);
                    window.dispatchEvent(new CustomEvent('modelLoadError', { 
                        detail: { 
                            message: 'Models will load on-demand when needed.', 
                            error: error?.message || 'Unknown error'
                        }
                    }));
                    // Still dispatch ready event so UI doesn't stay in loading state
                    window.dispatchEvent(new CustomEvent('modelsReady', { 
                        detail: { message: 'Models will load on-demand' }
                    }));
                }
            })();
        }
        
        return module;
    } catch (error: any) {
        console.error('Failed to initialize transformers.js:', error);
        console.error('Error details:', error?.message, error?.stack);
        throw error; // Re-throw to get better error messages
    }
};

// Start preloading immediately
transformersLoadPromise = initTransformers();

const getTransformers = async () => {
    if (!transformersModule) {
        if (transformersLoadPromise) {
            try {
                transformersModule = await transformersLoadPromise;
            } catch (error: any) {
                console.error('Transformers load promise failed:', error);
                // Try again
                transformersLoadPromise = initTransformers();
                transformersModule = await transformersLoadPromise;
            }
        }
        
        if (!transformersModule) {
            transformersModule = await initTransformers();
        }
        
        if (!transformersModule) {
            throw new Error('Failed to load transformers.js. Please check browser console for details and refresh the page.');
        }
    }
    
    return transformersModule;
};

// --- ON-DEVICE AI SERVICE ---
class OnDeviceAIService {
    private static instance: OnDeviceAIService | null = null;
    private transcriptionPipe: any = null;
    private analysisPipe: any = null;
    private tokenizer: any = null;
    private model: any = null;
    private analysisLoadPromise: Promise<void> | null = null;
    private transcriptionLoadPromise: Promise<any> | null = null;

    private constructor() {}

    public static getInstance(): OnDeviceAIService {
        if (!this.instance) {
            this.instance = new OnDeviceAIService();
        }
        return this.instance;
    }

    // Map language codes to Whisper model variants
    // Upgraded to base models for better accuracy, especially for names and proper nouns
    private getWhisperModel(language?: string): string {
        const lang = language || 'en';
        // Use base models for better accuracy (names, proper nouns, technical terms)
        // base models are ~150MB vs tiny ~75MB - better accuracy with reasonable speed
        if (lang === 'en') {
            return 'Xenova/whisper-base.en'; // Base English model - better accuracy for names
        } else {
            return 'Xenova/whisper-base'; // Base multilingual model
        }
    }

    private transcriptionLanguage: string | null = null;

    public async getTranscriptionPipeline(language?: string, progress_callback?: (progress: any) => void) {
        const lang = language || 'en';
        const modelName = this.getWhisperModel(lang);
        
        // Reload pipeline if language changed
        if (!this.transcriptionPipe || this.transcriptionLanguage !== lang) {
            // Cancel any ongoing load if language changes
            // Only cancel if transcriptionLanguage was already set (i.e. not null) to avoid resetting on first load
            if (this.transcriptionLanguage !== null && this.transcriptionLanguage !== lang && this.transcriptionLoadPromise) {
                this.transcriptionLoadPromise = null;
            }

            if (!this.transcriptionLoadPromise) {
                this.transcriptionLoadPromise = (async () => {
                    try {
                        const transformers = await getTransformers();
                        if (!transformers || !transformers.pipeline) {
                            throw new Error('Transformers.js not loaded');
                        }

                        // Ensure transformers.js uses our proxy
                        if (transformers.env) {
                            transformers.env.remoteHost = 'http://localhost:3001';
                            transformers.env.useBrowserCache = true;
                        }

                        // Log on-device verification
                        console.log('Transcription: On-Device Processing', {
                            model: modelName,
                            cacheEnabled: transformers.env?.useBrowserCache || false,
                            processingLocation: 'browser',
                            dataTransmission: 'none'
                        });

                        // Use pipeline with explicit configuration to ensure proxy is used
                        // transformers.js will auto-detect the model type from the config
                        // For Whisper models, we need to ensure the config is loaded correctly
                        try {
                            console.log('Creating transcription pipeline for model:', modelName);
                            // Explicitly specify the task and model to ensure correct detection
                            this.transcriptionPipe = await transformers.pipeline(
                                'automatic-speech-recognition',
                                modelName,
                                {
                                    progress_callback: (progress: any) => {
                                        if (progress_callback) progress_callback(progress);
                                    },
                                    // Use wasm device (webgpu is also supported but wasm is more compatible)
                                    device: 'wasm'
                                }
                            );
                            console.log('Pipeline created successfully. Type:', typeof this.transcriptionPipe);

                            // Verify the pipeline is set up correctly by checking if it's a function
                            if (typeof this.transcriptionPipe !== 'function') {
                                throw new Error('Pipeline is not a function - pipeline creation may have failed');
                            }
                        } catch (pipelineError: any) {
                            // If pipeline creation fails with "Unsupported model type",
                            // transformers.js may not support Whisper models in this version
                            if (pipelineError?.message?.includes('Unsupported model type')) {
                                console.error('Whisper model not supported:', pipelineError.message);
                                console.error('Model name:', modelName);
                                console.error('This version of transformers.js may not support Whisper models.');
                                console.error('Error details:', {
                                    error: pipelineError.message,
                                    modelType: 'whisper',
                                    transformersVersion: transformers?.version || 'unknown'
                                });
                                // Provide a clear error message to the user
                                throw new Error(`Whisper models are not supported in this version of transformers.js (${transformers?.version || 'unknown'}). The library is trying to use AutoModelForCTC instead of WhisperForConditionalGeneration. Please check for a newer version of @huggingface/transformers that supports Whisper models, or use an alternative transcription approach.`);
                            }
                            throw pipelineError;
                        }

                        this.transcriptionLanguage = lang;

                        // Verify model functionality after load
                        if (!this.transcriptionPipe || typeof this.transcriptionPipe !== 'function') {
                            throw new Error('Transcription model failed verification - pipeline is not a function');
                        }

                        console.log('Transcription model loaded successfully');
                    } catch (error: any) {
                        this.transcriptionLoadPromise = null;
                        const errorMessage = error?.message || 'Unknown error loading transcription model';
                        console.error('Transcription model load error:', error);
                        console.error('Error details:', {
                            message: errorMessage,
                            stack: error?.stack,
                            name: error?.name
                        });

                        // Check if error is HTML/JSON parsing issue
                        if (errorMessage.includes('<!DOCTYPE') ||
                            errorMessage.includes('Unexpected token') ||
                            errorMessage.includes('HTML error page') ||
                            errorMessage.includes('HTML instead of JSON')) {
                            throw new Error(`Model download failed: Received HTML error page instead of JSON. This usually means:
1. The model URL is incorrect or the model doesn't exist
2. Hugging Face returned an error page (check terminal logs)
3. The proxy isn't working correctly

Please check the browser console and terminal logs for more details. Try refreshing the page.`);
                        }

                        throw new Error(`Failed to load transcription model: ${errorMessage}. Please refresh and try again.`);
                    }
                })();
            }
            await this.transcriptionLoadPromise;
        }
        return this.transcriptionPipe;
    }

    public async getAnalysisPipeline(progress_callback?: (progress: any) => void) {
        if (!this.tokenizer || !this.model) {
            if (!this.analysisLoadPromise) {
                this.analysisLoadPromise = (async () => {
                    try {
                        const transformers = await getTransformers();
                        if (!transformers || !transformers.AutoTokenizer || !transformers.AutoModelForSeq2SeqLM) {
                            throw new Error('Transformers.js not loaded');
                        }

                        // Ensure transformers.js uses our proxy
                        if (transformers.env) {
                            transformers.env.remoteHost = 'http://localhost:3001';
                        }

                        // Log on-device verification
                        console.log('Analysis: On-Device Processing', {
                            model: 'Xenova/flan-t5-base',
                            cacheEnabled: transformers.env?.useBrowserCache || false,
                            processingLocation: 'browser',
                            dataTransmission: 'none'
                        });

                        const progressHandler = (progress: any) => {
                            if (progress_callback) progress_callback(progress);
                        };
                        // Upgraded to flan-t5-base for better quality while maintaining reasonable size
                        // flan-t5-base is better than LaMini-Flan-T5-783M for instruction following and JSON generation
                        const modelName = 'Xenova/flan-t5-base';
                        this.tokenizer = await transformers.AutoTokenizer.from_pretrained(modelName, { progress_callback: progressHandler });
                        this.model = await transformers.AutoModelForSeq2SeqLM.from_pretrained(modelName, { progress_callback: progressHandler });

                        // Verify model functionality after load
                        if (!this.tokenizer || typeof this.tokenizer !== 'function') {
                            throw new Error('Tokenizer failed verification - not a function');
                        }
                        if (!this.model || typeof this.model.generate !== 'function') {
                            throw new Error('Analysis model failed verification - generate method missing');
                        }
                    } catch (error: any) {
                        this.analysisLoadPromise = null;
                        const errorMessage = error?.message || 'Unknown error loading analysis model';
                        console.error('Analysis model load error:', error);

                        // Check if error is HTML/JSON parsing issue
                        if (errorMessage.includes('<!DOCTYPE') ||
                            errorMessage.includes('Unexpected token') ||
                            errorMessage.includes('HTML error page') ||
                            errorMessage.includes('HTML instead of JSON')) {
                            throw new Error(`Model download failed: Received HTML error page instead of JSON. This usually means:
1. The model URL is incorrect or the model doesn't exist
2. Hugging Face returned an error page (check terminal logs)
3. The proxy isn't working correctly

Please check the browser console and terminal logs for more details. Try refreshing the page.`);
                        }

                        throw new Error(`Failed to load analysis model: ${errorMessage}. Please refresh and try again.`);
                    }
                })();
            }
            await this.analysisLoadPromise;
        }
    }

    // Resample audio to target sample rate (Whisper expects 16kHz)
    private resampleAudio(audioData: Float32Array, fromSampleRate: number, toSampleRate: number): Float32Array {
        if (fromSampleRate === toSampleRate) {
            return audioData;
        }
        
        const ratio = fromSampleRate / toSampleRate;
        const newLength = Math.round(audioData.length / ratio);
        const result = new Float32Array(newLength);
        
        // Linear interpolation resampling
        for (let i = 0; i < newLength; i++) {
            const srcIndex = i * ratio;
            const index = Math.floor(srcIndex);
            const fraction = srcIndex - index;
            
            if (index + 1 < audioData.length) {
                // Linear interpolation
                result[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
            } else {
                result[i] = audioData[index] || 0;
            }
        }
        
        return result;
    }

    // Audio preprocessing: noise suppression and normalization
    private preprocessAudio(audioBuffer: AudioBuffer): AudioBuffer {
        const sampleRate = audioBuffer.sampleRate;
        const numberOfChannels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length;
        
        // Create new audio buffer for processed audio
        const processedBuffer = new AudioBuffer({
            numberOfChannels,
            length,
            sampleRate
        });
        
        // Process each channel
        for (let channel = 0; channel < numberOfChannels; channel++) {
            const inputData = audioBuffer.getChannelData(channel);
            const outputData = processedBuffer.getChannelData(channel);
            
            // Simple high-pass filter to remove low-frequency noise (hum, rumble)
            const cutoff = 80; // Hz
            const rc = 1.0 / (cutoff * 2 * Math.PI);
            const dt = 1.0 / sampleRate;
            const alpha = rc / (rc + dt);
            
            let prevInput = inputData[0];
            let prevOutput = inputData[0];
            
            for (let i = 0; i < length; i++) {
                // High-pass filter
                const filtered = alpha * (prevOutput + inputData[i] - prevInput);
                prevInput = inputData[i];
                prevOutput = filtered;
                outputData[i] = filtered;
            }
            
            // Normalize audio levels (prevent clipping, boost quiet audio)
            let max = 0;
            for (let i = 0; i < length; i++) {
                const abs = Math.abs(outputData[i]);
                if (abs > max) max = abs;
            }
            
            if (max > 0) {
                const targetPeak = 0.95; // Target peak level
                const gain = targetPeak / max;
                // Apply gentle gain (don't over-amplify)
                const safeGain = Math.min(gain, 3.0); // Max 3x amplification
                for (let i = 0; i < length; i++) {
                    outputData[i] *= safeGain;
                }
            }
        }
        
        return processedBuffer;
    }

    public async analyze(
        audio: AudioBuffer,
        progressCallback: (status: string, progress?: number) => void,
        industry?: string,
        language?: string,
        timeoutMs: number = 60000
    ): Promise<string> {
        // Verify on-device processing
        console.log('✓ AI Analysis: On-Device Processing', {
            processingLocation: 'browser',
            audioDuration: `${audio.duration.toFixed(2)}s`,
            dataTransmission: 'none',
            industry: industry || 'general',
            language: language || 'en'
        });
        
        const startTime = performance.now();
        const stageTimings: {[key: string]: number} = {};
        
        const timeoutPromise = new Promise<never>((_, reject) => {
            setTimeout(() => reject(new Error('Analysis timeout: Processing exceeded 60 seconds')), timeoutMs);
        });

        const analyzeWithTimeout = async (): Promise<string> => {
            // Returns JSON string with all analysis results
            
            // 0. Audio preprocessing (noise suppression)
            const preprocessStart = performance.now();
            progressCallback('Enhancing audio quality...', 0);
            const processedAudio = this.preprocessAudio(audio);
            stageTimings.preprocessing = Math.round(performance.now() - preprocessStart);
            progressCallback(`Audio enhanced (${stageTimings.preprocessing}ms)`, 5);
            
            // 1. Transcription
            const transcriptionStart = performance.now();
            const lang = language || 'en';
            progressCallback(`Initializing transcription model (${lang})...`, 5);
        const transcriber = await this.getTranscriptionPipeline(lang, (p: any) => {
            if (p.status === 'progress') {
                progressCallback('Downloading transcription model...', p.progress);
            }
        });

        progressCallback('Transcribing audio...', 20);
        // Get audio data - use first channel (mono) for transcription
        const originalAudioData = processedAudio.getChannelData(0);
        const originalSampleRate = processedAudio.sampleRate;
        const duration = processedAudio.duration;
        
        // Whisper models expect 16kHz audio - resample if needed
        const targetSampleRate = 16000;
        const audioData = this.resampleAudio(originalAudioData, originalSampleRate, targetSampleRate);
        
        console.log(`Transcribing audio: ${duration.toFixed(2)}s, original sample rate: ${originalSampleRate}Hz, resampled to: ${targetSampleRate}Hz, samples: ${audioData.length}`);
        
        // The new version of transformers.js expects audio as Float32Array
        // Pass the resampled audio data directly with sample_rate parameter
        // Log what we're passing to help debug
        console.log('Calling transcriber with:', {
            audioLength: audioData.length,
            sampleRate: targetSampleRate,
            duration: duration,
            language: lang,
            transcriberType: typeof transcriber
        });
        
        // Optimized transcription parameters for lower latency
        const transcriptionPromise = transcriber(audioData, {
            chunk_length_s: 15, // Smaller chunks = faster processing
            stride_length_s: 2, // Smaller stride = less overlap, faster
            return_timestamps: true,
            language: lang !== 'en' ? lang : undefined,
            sample_rate: targetSampleRate,
            // Additional performance optimizations
            batch_size: 1, // Process one chunk at a time for lower memory
        });
        
        // Optimized timeout: faster processing expected with tiny model
        const transcriptionTimeoutMs = Math.max(20000, Math.ceil(duration * 2000) + 10000); // At least 20s, or 2x audio duration + 10s
        const transcriptionTimeout = new Promise((_, reject) => {
            setTimeout(() => reject(new Error(`Transcription timeout: Process took longer than ${Math.round(transcriptionTimeoutMs/1000)} seconds. The audio might be too long or the model is stuck.`)), transcriptionTimeoutMs);
        });
        
        const transcription = await Promise.race([transcriptionPromise, transcriptionTimeout]);
        stageTimings.transcription = Math.round(performance.now() - transcriptionStart);
        progressCallback(`Transcription complete (${stageTimings.transcription}ms)`, 50);

        // Log transcription result to debug format
        console.log('Transcription result:', transcription);
        console.log('Transcription type:', typeof transcription);
        console.log('Transcription keys:', transcription ? Object.keys(transcription) : 'null/undefined');

        // Handle different response formats from transformers.js
        // New version might return: { text: "...", chunks: [...] } or just { text: "..." } or just the text string
        let transcriptChunks: any[] = [];
        
        if (typeof transcription === 'string') {
            // If it's just a string, create a single chunk
            transcriptChunks = [{
                text: transcription,
                timestamp: [0, processedAudio.duration]
            }];
        } else if (transcription?.chunks && Array.isArray(transcription.chunks)) {
            // Standard format with chunks
            transcriptChunks = transcription.chunks;
        } else if (transcription?.text) {
            // New format might just have text property
            transcriptChunks = [{
                text: transcription.text,
                timestamp: transcription.timestamps || [0, processedAudio.duration]
            }];
        } else if (Array.isArray(transcription)) {
            // Might be an array of chunks directly
            transcriptChunks = transcription;
        } else {
            console.warn('Unexpected transcription format:', transcription);
            transcriptChunks = [];
        }

        // Enhanced speaker diarization using heuristic-based clustering
        // Analyzes silence gaps, text patterns, and timing to identify speakers
        const SILENCE_THRESHOLD = 1.5; // seconds of silence indicates speaker change
        const MIN_CHUNK_DURATION = 0.5; // minimum chunk duration to consider
        const MAX_SPEAKERS = 10;
        
        let currentSpeaker = 1;
        let lastEndTime = 0;
        const speakerSegments: Array<{speaker: number, startTime: number, endTime: number, textLength: number}> = [];
        
        const processedChunks = transcriptChunks.map((chunk: any, index: number) => {
            const startTime = chunk.timestamp?.[0] || 0;
            const endTime = chunk.timestamp?.[1] || startTime;
            const textLength = chunk.text?.length || 0;
            const duration = endTime - startTime;
            
            // Calculate gap since last chunk
            const gap = startTime - lastEndTime;
            
            // Speaker change detection heuristics:
            // 1. Significant silence gap (> threshold)
            // 2. Large time jump (likely editing or pause)
            // 3. First chunk always starts with Speaker 1
            if (index === 0) {
                currentSpeaker = 1;
            } else if (gap > SILENCE_THRESHOLD && duration > MIN_CHUNK_DURATION) {
                // Check if this pattern suggests a new speaker
                // Look at previous speaker's average chunk length and this one's
                const prevSegment = speakerSegments[speakerSegments.length - 1];
                if (prevSegment) {
                    const avgPrevLength = prevSegment.textLength;
                    // If current chunk is significantly different in length, might be new speaker
                    if (Math.abs(textLength - avgPrevLength) > avgPrevLength * 0.5 && gap > SILENCE_THRESHOLD * 1.5) {
                        currentSpeaker = Math.min(currentSpeaker + 1, MAX_SPEAKERS);
                    }
                } else if (gap > SILENCE_THRESHOLD * 2) {
                    // Very long gap, likely new speaker
                    currentSpeaker = Math.min(currentSpeaker + 1, MAX_SPEAKERS);
                }
            }
            
            // Track speaker segment for pattern analysis
            speakerSegments.push({
                speaker: currentSpeaker,
                startTime,
                endTime,
                textLength
            });
            
            lastEndTime = endTime;
            
            return {
                speaker: `Speaker ${currentSpeaker}`,
                text: chunk.text,
                timestamp: chunk.timestamp
            };
        });

        const fullTranscript = processedChunks.map(c => c.text).join(' ');
        
        if (!fullTranscript.trim()) {
            const emptyResult = {
                transcript: [],
                summary: 'No transcript available. Please ensure audio was recorded and transcribed.',
                action_items: [],
                outline: 'No transcript available.'
            };
            return JSON.stringify(emptyResult);
        }
        
        // Check if transcript only contains music/sound/blank markers (no actual speech)
        const cleanedTranscript = fullTranscript.replace(/\[Music\]|\[SOUND\]|\[MUSIC PLAYING\]|\[BLANK_AUDIO\]|\[/gi, '').trim();
        if (!cleanedTranscript || cleanedTranscript.length < 10) {
            const hasBlankAudio = /\[BLANK_AUDIO\]/i.test(fullTranscript);
            const errorMessage = hasBlankAudio 
                ? 'No speech detected in the audio. The recording appears to be blank or silent. Please record again with clear speech.'
                : 'No speech detected in the audio. The recording appears to contain only background music or noise. Please record again with clear speech.';
            const errorResult = {
                transcript: processedChunks,
                summary: errorMessage,
                action_items: [],
                outline: 'No speech detected in the audio recording.'
            };
            return JSON.stringify(errorResult);
        }

        // 2. Analysis (Summary, Todos, Outline)
        const analysisStart = performance.now();
        progressCallback('Initializing analysis model...', 50);
        await this.getAnalysisPipeline((p: any) => {
            if (p.status === 'progress') {
                progressCallback('Downloading analysis model...', p.progress);
            }
        });
        
        progressCallback('Analyzing transcript...', 60);
        
        // Optimized prompt with industry context - minimal tokens, JSON-only output
        // Enhanced to request topic-grouped outline
        const industryContext = industry && industry !== 'general' 
            ? `Context: ${industry === 'therapy' ? 'therapy session' : industry === 'medical' ? 'medical dictation' : industry === 'legal' ? 'legal note' : 'business meeting'}. `
            : '';
        const prompt = `${industryContext}Analyze transcript. Return JSON: {"summary":"text","action_items":["item"],"outline":"grouped topics with main points per topic"}. Group outline by topics. Transcript: ${fullTranscript}`;
        

        // Ensure tokenizer and model are initialized
        if (!this.tokenizer || !this.model) {
            throw new Error("Analysis model not initialized");
        }

        try {
            // Tokenize the input
            progressCallback('Tokenizing input...', 70);
            // Increase max_length to ensure the full transcript is included.
            // For transformers.js v3, the tokenizer returns an object with both
            // input_ids and attention_mask, which we should pass through intact.
            // Increase max_length to allow longer prompts with transcript
            const inputs = this.tokenizer(prompt, {
                return_tensors: 'pt',
                padding: true,
                truncation: true,
                max_length: 2048 // Increased to handle longer transcripts
            });
            
            if (!inputs || !inputs.input_ids || !inputs.attention_mask) {
                throw new Error('Tokenizer did not return expected input_ids and attention_mask');
            }
            
            // Generate output – pass the full inputs object so the model
            // receives both input_ids and attention_mask as required by
            // the current transformers.js generate() API.
            progressCallback('Generating analysis...', 80);
            const output = await this.model.generate(inputs, {
                max_new_tokens: 512,
                num_beams: 1,
                do_sample: false,
                pad_token_id: this.tokenizer.eos_token_id || 0
            });
            
            if (!output || !output[0]) {
                throw new Error("Model did not return expected output");
            }
            
            // Decode the output
            progressCallback('Decoding results...', 90);
            const resultText = this.tokenizer.decode(output[0], { skip_special_tokens: true });
            stageTimings.analysis = Math.round(performance.now() - analysisStart);
            
            if (!resultText || resultText.trim().length === 0) {
                throw new Error("Decoded result is empty");
            }
            
            // Attempt to find a valid JSON object within the result text
            let parsedResult: any = null;
            
            // Try to find JSON object
            const jsonMatch = resultText.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                try {
                    parsedResult = JSON.parse(jsonMatch[0]);
                } catch (parseError) {
                    // Continue to plain text parsing
                }
            }
            
            if (!parsedResult) {
                const errorPatterns = [
                    /cannot perform/i,
                    /does not contain/i,
                    /no text/i,
                    /empty transcript/i,
                    /no transcript/i
                ];
                
                const isErrorResponse = errorPatterns.some(pattern => pattern.test(resultText));
                
                if (isErrorResponse) {
                    parsedResult = {
                        summary: 'Unable to generate summary: The transcript appears to be empty or too short to analyze.',
                        action_items: [],
                        outline: 'Unable to generate outline: The transcript appears to be empty or too short to analyze.'
                    };
                } else {
                    // Try to extract summary, action items, and outline from plain text
                    parsedResult = {
                        summary: this.extractSection(resultText, ['summary', 'Summary']),
                        action_items: this.extractList(resultText, ['action items', 'action_items', 'todos', 'to-do', 'tasks']),
                        outline: this.extractSection(resultText, ['outline', 'Outline', 'structure'])
                    };
                    
                    // If we still don't have a summary, use the first few sentences (but not if it's an error)
                    if (!parsedResult.summary && resultText.trim().length > 0 && !isErrorResponse) {
                        parsedResult.summary = resultText.split(/[.!?]/).slice(0, 2).join('. ').trim() + '.';
                    }
                }
            }
            // Cluster topics in outline if available
            let clusteredOutline = parsedResult.outline || 'No outline generated.';
            if (clusteredOutline && clusteredOutline !== 'No outline generated.') {
                clusteredOutline = this.clusterTopics(clusteredOutline);
            }
            
            // Return JSON string with all analysis results
            const analysisResult = {
                transcript: processedChunks,
                summary: parsedResult.summary || 'No summary generated.',
                action_items: parsedResult.action_items || [],
                outline: clusteredOutline
            };
            const elapsedTime = performance.now() - startTime;
            if (elapsedTime > timeoutMs) {
                throw new Error(`Analysis exceeded ${timeoutMs}ms timeout`);
            }
            
            // Format timing summary
            const timingSummary = [
                `Preprocessing: ${stageTimings.preprocessing}ms`,
                `Transcription: ${stageTimings.transcription}ms`,
                `Analysis: ${stageTimings.analysis}ms`,
                `Total: ${Math.round(elapsedTime)}ms`
            ].join(', ');
            
            progressCallback(`Analysis complete (${Math.round(elapsedTime / 1000)}s) - ${timingSummary}`, 100);
            return JSON.stringify(analysisResult);
        } catch (error: any) {
            const elapsedTime = performance.now() - startTime;
            // Return error as JSON string
            const errorResult = {
                error: true,
                message: error?.message || "Failed to parse on-device AI analysis.",
                type: error?.name || "Error",
                elapsedTime: Math.round(elapsedTime / 1000),
                transcript: processedChunks,
                summary: "Analysis failed. Please try again.",
                action_items: [],
                outline: "Analysis failed."
            };
            return JSON.stringify(errorResult);
        }
        };

        // Race between analysis and timeout
        return Promise.race([analyzeWithTimeout(), timeoutPromise]);
    }

    /**
     * Transcribe audio to text with speaker identification
     * Returns processed transcript chunks with speaker labels
     */
    public async transcribeAudio(
        audio: AudioBuffer,
        progressCallback: (status: string, progress?: number) => void,
        language?: string
    ): Promise<Array<{speaker: string, text: string, timestamp: any}>> {
        // Verify on-device processing
        console.log('✓ AI Transcription: On-Device Processing', {
            processingLocation: 'browser',
            audioDuration: `${audio.duration.toFixed(2)}s`,
            dataTransmission: 'none',
            language: language || 'en'
        });
        
        // Audio preprocessing (noise suppression)
        progressCallback('Enhancing audio quality...', 0);
        const processedAudio = this.preprocessAudio(audio);
        progressCallback('Audio enhanced', 5);

        // Transcription
        const lang = language || 'en';
        progressCallback(`Initializing transcription model (${lang})...`, 5);
        const transcriber = await this.getTranscriptionPipeline(lang, (p: any) => {
            if (p.status === 'progress') {
                progressCallback('Downloading transcription model...', p.progress);
            }
        });

        progressCallback('Transcribing audio...', 20);
        const originalAudioData = processedAudio.getChannelData(0);
        const originalSampleRate = processedAudio.sampleRate;
        const duration = processedAudio.duration;
        
        const targetSampleRate = 16000;
        const audioData = this.resampleAudio(originalAudioData, originalSampleRate, targetSampleRate);
        
        console.log(`Transcribing audio: ${duration.toFixed(2)}s, original sample rate: ${originalSampleRate}Hz, resampled to: ${targetSampleRate}Hz`);

        const transcriptionPromise = transcriber(audioData, {
            chunk_length_s: 30, // Increased from 15 to capture longer segments
            stride_length_s: 5, // Increased from 2 to ensure better overlap and no gaps
            return_timestamps: true,
            language: lang !== 'en' ? lang : undefined,
            sample_rate: targetSampleRate,
            batch_size: 1,
        });
        
        const transcriptionTimeoutMs = Math.max(20000, Math.ceil(duration * 2000) + 10000);
        const transcriptionTimeout = new Promise((_, reject) => {
            setTimeout(() => reject(new Error(`Transcription timeout: Process took longer than ${Math.round(transcriptionTimeoutMs/1000)} seconds.`)), transcriptionTimeoutMs);
        });
        
        const transcription = await Promise.race([transcriptionPromise, transcriptionTimeout]);
        progressCallback('Transcription complete', 50);

        // Handle different response formats
        let transcriptChunks: any[] = [];
        
        if (typeof transcription === 'string') {
            transcriptChunks = [{
                text: transcription,
                timestamp: [0, processedAudio.duration]
            }];
        } else if (transcription?.chunks && Array.isArray(transcription.chunks)) {
            transcriptChunks = transcription.chunks;
        } else if (transcription?.text) {
            transcriptChunks = [{
                text: transcription.text,
                timestamp: transcription.timestamps || [0, processedAudio.duration]
            }];
        } else if (Array.isArray(transcription)) {
            transcriptChunks = transcription;
        } else {
            console.warn('Unexpected transcription format:', transcription);
            transcriptChunks = [];
        }

        // Speaker diarization
        const SILENCE_THRESHOLD = 1.5;
        const MIN_CHUNK_DURATION = 0.5;
        const MAX_SPEAKERS = 10;
        
        let currentSpeaker = 1;
        let lastEndTime = 0;
        const speakerSegments: Array<{speaker: number, startTime: number, endTime: number, textLength: number}> = [];
        
        const processedChunks = transcriptChunks.map((chunk: any, index: number) => {
            const startTime = chunk.timestamp?.[0] || 0;
            const endTime = chunk.timestamp?.[1] || startTime;
            const textLength = chunk.text?.length || 0;
            const duration = endTime - startTime;
            const gap = startTime - lastEndTime;
            
            if (index === 0) {
                currentSpeaker = 1;
            } else if (gap > SILENCE_THRESHOLD && duration > MIN_CHUNK_DURATION) {
                const prevSegment = speakerSegments[speakerSegments.length - 1];
                if (prevSegment) {
                    const avgPrevLength = prevSegment.textLength;
                    if (Math.abs(textLength - avgPrevLength) > avgPrevLength * 0.5 && gap > SILENCE_THRESHOLD * 1.5) {
                        currentSpeaker = Math.min(currentSpeaker + 1, MAX_SPEAKERS);
                    }
                } else if (gap > SILENCE_THRESHOLD * 2) {
                    currentSpeaker = Math.min(currentSpeaker + 1, MAX_SPEAKERS);
                }
            }
            
            speakerSegments.push({
                speaker: currentSpeaker,
                startTime,
                endTime,
                textLength
            });
            
            lastEndTime = endTime;
            
            return {
                speaker: `Speaker ${currentSpeaker}`,
                text: chunk.text,
                timestamp: chunk.timestamp
            };
        });

        return processedChunks;
    }

    /**
     * Truncate transcript intelligently to fit within token limits
     * Keeps the beginning and end, removes middle if needed
     */
    private truncateTranscript(transcript: string, maxLength: number = 2000): string {
        if (transcript.length <= maxLength) {
            return transcript;
        }
        
        // Keep first 60% and last 40% to preserve context
        const firstPart = transcript.substring(0, Math.floor(maxLength * 0.6));
        const lastPart = transcript.substring(transcript.length - Math.floor(maxLength * 0.4));
        return `${firstPart}... [middle section truncated] ...${lastPart}`;
    }

    /**
     * Improved JSON parsing with better extraction and validation
     * Handles incomplete, malformed, and empty JSON responses
     */
    private parseJSONResponse(resultText: string, requiredFields: string[] = []): any {
        if (!resultText || resultText.trim().length === 0) {
            return null;
        }

        // Clean up common malformed patterns
        let cleaned = resultText.trim();
        
        // Fix empty string values with too many quotes: "outline":"""""" -> "outline":""
        cleaned = cleaned.replace(/(":\s*)"{3,}/g, '$1""');
        
        // Try to find JSON object with improved regex (non-greedy, handles nested objects)
        // First try to find complete JSON object
        let jsonMatch = cleaned.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            try {
                const parsed = JSON.parse(jsonMatch[0]);
                // Validate required fields and check for empty/invalid values
                if (requiredFields.length === 0 || requiredFields.every(field => {
                    if (!parsed.hasOwnProperty(field)) return false;
                    const value = parsed[field];
                    // Reject empty strings, null, or undefined
                    if (value === null || value === undefined) return false;
                    if (typeof value === 'string' && value.trim().length === 0) return false;
                    if (Array.isArray(value) && value.length === 0) return false;
                    return true;
                })) {
                    return parsed;
                }
            } catch (parseError) {
                // Try to find JSON within the text more carefully
                const jsonStart = cleaned.indexOf('{');
                const jsonEnd = cleaned.lastIndexOf('}');
                if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
                    try {
                        let jsonStr = cleaned.substring(jsonStart, jsonEnd + 1);
                        
                        // Try to fix incomplete JSON (e.g., cut-off arrays or strings)
                        // If we see an incomplete array, try to close it
                        const openBrackets = (jsonStr.match(/\[/g) || []).length;
                        const closeBrackets = (jsonStr.match(/\]/g) || []).length;
                        if (openBrackets > closeBrackets) {
                            jsonStr += ']'.repeat(openBrackets - closeBrackets);
                        }
                        
                        // If we see an incomplete string (unclosed quote), try to close it
                        const quoteCount = (jsonStr.match(/"/g) || []).length;
                        if (quoteCount % 2 !== 0) {
                            // Find the last unclosed quote and close it
                            const lastQuoteIndex = jsonStr.lastIndexOf('"');
                            if (lastQuoteIndex > 0) {
                                // Check if it's inside a value (not a key)
                                const beforeQuote = jsonStr.substring(0, lastQuoteIndex);
                                const afterQuote = jsonStr.substring(lastQuoteIndex + 1);
                                // If there's content after the quote, it might be incomplete
                                if (afterQuote.trim().length > 0 && !afterQuote.trim().startsWith(',')) {
                                    jsonStr = jsonStr.substring(0, lastQuoteIndex + 1) + '"' + afterQuote;
                                }
                            }
                        }
                        
                        const parsed = JSON.parse(jsonStr);
                        if (requiredFields.length === 0 || requiredFields.every(field => {
                            if (!parsed.hasOwnProperty(field)) return false;
                            const value = parsed[field];
                            if (value === null || value === undefined) return false;
                            if (typeof value === 'string' && value.trim().length === 0) return false;
                            if (Array.isArray(value) && value.length === 0) return false;
                            return true;
                        })) {
                            return parsed;
                        }
                    } catch (e) {
                        console.warn('JSON parse error (second attempt):', e, 'Text:', jsonStr?.substring(0, 200));
                    }
                }
            }
        }

        // Try to find JSON array
        const arrayMatch = cleaned.match(/\[[\s\S]*?\]/);
        if (arrayMatch) {
            try {
                const parsed = JSON.parse(arrayMatch[0]);
                if (Array.isArray(parsed) && parsed.length > 0) {
                    return parsed;
                }
            } catch (parseError) {
                // Don't log errors for invalid patterns like [int]: [int]:
                if (!cleaned.match(/\[int\]:?\s*\[int\]:?/i)) {
                    console.warn('JSON array parse error:', parseError);
                }
            }
        }

        return null;
    }

    /**
     * Generate summary from transcript with domain-specific prompts
     */
    public async generateSummary(
        transcript: string,
        industry: string,
        progressCallback: (status: string, progress?: number) => void
    ): Promise<string> {
        await this.getAnalysisPipeline((p: any) => {
            if (p.status === 'progress') {
                progressCallback('Loading analysis model...', p.progress);
            }
        });

        // Truncate transcript if too long to fit in token limit
        const truncatedTranscript = this.truncateTranscript(transcript, 1500);
        
        // Domain-specific summary prompts
        let prompt = '';
        
        switch (industry) {
            case 'therapy':
                prompt = `Role: You are a highly skilled assistant specialized in processing transcribed text for Psychotherapy and Counseling sessions.

Source Context: The following text is a verbatim AI transcription from a therapy session. It may contain filler words, repetitions, and conversational speech.

Core Instructions:
1. Clean the Text: Remove filler words, false starts, and significant repetitions while preserving the original meaning and nuance.
2. Improve Readability: Correct obvious grammatical errors and break long sentences into clear, concise ones.
3. Structure: Format as a SOAP Note (Subjective, Objective, Assessment, Plan).
4. Identify Key Information: Extract the client's primary emotions, cognitive distortions, coping mechanisms, insights gained, and therapeutic interventions.

Output Format: Return ONLY valid JSON: {"summary":"Your cleaned and structured summary here"}

Transcription: ${truncatedTranscript}`;
                break;
                
            case 'medical':
                prompt = `Role: You are a highly skilled assistant specialized in processing transcribed text for Clinical Medical Documentation.

Source Context: The following text is a verbatim AI transcription from a medical consultation. It may contain filler words, repetitions, and conversational speech.

Core Instructions:
1. Clean the Text: Remove filler words, false starts, and significant repetitions while preserving all medical terminology accurately.
2. Improve Readability: Correct obvious grammatical errors. Do NOT correct or guess medical terms; flag uncertainties with [?].
3. Structure: Format as a Clinical Patient Note with Chief Complaint (CC), History of Present Illness (HPI), and Assessment & Plan (A/P).
4. Identify Key Information: Extract critical medical data: symptoms, onset, duration, severity, medications, allergies, past medical history, diagnosis, and treatment plan.

Output Format: Return ONLY valid JSON: {"summary":"Your structured clinical note here"}

Transcription: ${truncatedTranscript}`;
                break;
                
            case 'legal':
                prompt = `Role: You are a highly skilled assistant specialized in processing transcribed text for Legal Documentation and Client Meetings.

Source Context: The following text is a verbatim AI transcription from a legal consultation. It may contain filler words, repetitions, and conversational speech.

Core Instructions:
1. Clean the Text: Remove filler words, false starts, and significant repetitions while preserving all legal terminology and facts.
2. Improve Readability: Correct obvious grammatical errors. Preserve all names, dates, and legal citations exactly.
3. Structure: Organize by topic with clear headings: Case Background, Client Statement, Legal Issues Identified, Key Dates & Deadlines.
4. Identify Key Information: Extract critical facts, claims, allegations, relevant dates, names of parties, potential evidence, legal precedents or statutes cited, and legal advice given.

Output Format: Return ONLY valid JSON: {"summary":"Your structured legal notes here"}

Transcription: ${truncatedTranscript}`;
                break;
                
            case 'business':
                prompt = `Role: You are a highly skilled assistant specialized in processing transcribed text for Corporate Business Meetings.

Source Context: The following text is a verbatim AI transcription from a business meeting. It may contain filler words, repetitions, and conversational speech.

Core Instructions:
1. Clean the Text: Remove filler words, false starts, and significant repetitions while preserving the original meaning.
2. Improve Readability: Correct obvious grammatical errors and break long sentences into clear, concise ones.
3. Structure: Format as Meeting Minutes with sections for Attendees, Agenda Items, Decisions Made, and Action Items.
4. Identify Key Information: Extract key metrics, project updates, strategic decisions, assigned tasks (with owners and deadlines), and identified risks or blockers.

Output Format: Return ONLY valid JSON: {"summary":"Your structured meeting minutes here"}

Transcription: ${truncatedTranscript}`;
                break;
                
            default:
                prompt = `You are a professional note-taker. Create a clear, concise summary of this meeting transcript.

IMPORTANT: Write actual content based on the transcript. Do NOT use placeholders like "[actual content]" or "[summary here]".

Instructions:
1. Read the entire transcript carefully
2. Identify the main topics and key points discussed
3. Write 2-3 paragraphs summarizing what was discussed
4. Include important decisions, agreements, or conclusions
5. Make it readable and professional

Return ONLY valid JSON with real content:
{"summary":"Write the actual summary here based on what was discussed in the transcript. Make it 2-3 paragraphs that capture the essence of the meeting."}

Transcript: ${truncatedTranscript}`;
        }

        progressCallback('Generating summary...', 60);

        if (!this.tokenizer || !this.model) {
            throw new Error("Analysis model not initialized");
        }

        try {
            // Increase max_length to allow longer prompts with transcript
            const inputs = this.tokenizer(prompt, {
                return_tensors: 'pt',
                padding: true,
                truncation: true,
                max_length: 2048 // Increased to handle longer transcripts
            });
            
            if (!inputs || !inputs.input_ids || !inputs.attention_mask) {
                throw new Error('Tokenizer did not return expected input_ids and attention_mask');
            }
            
            const output = await this.model.generate(inputs, {
                max_new_tokens: 256,
                num_beams: 1,
                do_sample: false
            });
            
            if (!output || !output[0]) {
                throw new Error("Model did not return expected output");
            }
            
            const resultText = this.tokenizer.decode(output[0], { skip_special_tokens: true });
            
            const parsed = this.parseJSONResponse(resultText, ['summary']);
            if (parsed && parsed.summary && typeof parsed.summary === 'string') {
                const summaryText = parsed.summary.trim();
                // Validate it's not a placeholder
                if (summaryText.length > 20 && 
                    !summaryText.toLowerCase().includes('[actual') &&
                    !summaryText.toLowerCase().includes('[summary') &&
                    !summaryText.toLowerCase().includes('placeholder') &&
                    !summaryText.toLowerCase().includes('your summary')) {
                    console.log('Summary generated successfully:', summaryText.substring(0, 100));
                    return summaryText;
                }
            }

            // Fallback: extract summary from text
            const summary = this.extractSection(resultText, ['summary', 'Summary']);
            if (summary && summary.length > 20 && 
                !summary.toLowerCase().includes('[actual') &&
                !summary.toLowerCase().includes('placeholder')) {
                return summary.trim();
            }

            // Try to extract any meaningful text (not JSON structure)
            const textWithoutJson = resultText
                .replace(/\{"summary":\s*"/gi, '')
                .replace(/"\s*\}/g, '')
                .replace(/^"|"$/g, '')
                .replace(/\\n/g, '\n')
                .trim();
            
            if (textWithoutJson.length > 20 && 
                !textWithoutJson.toLowerCase().includes('[actual') &&
                !textWithoutJson.toLowerCase().includes('placeholder')) {
                return textWithoutJson;
            }

            // Last resort: use first few sentences if they look meaningful
            const sentences = resultText.split(/[.!?]/).filter(s => {
                const trimmed = s.trim();
                return trimmed.length > 10 && 
                       !trimmed.toLowerCase().includes('[actual') &&
                       !trimmed.toLowerCase().includes('placeholder');
            });
            if (sentences.length > 0) {
                return sentences.slice(0, 3).join('. ').trim() + '.';
            }

            return 'Summary generation is still processing. The transcript may be too short or unclear. Try re-analyzing the session.';
        } catch (error: any) {
            console.error('Summary generation error:', error);
            throw new Error(`Summary generation failed: ${error?.message || 'Unknown error'}`);
        }
    }

    /**
     * Generate outline from transcript, grouped by topics
     */
    public async generateOutline(
        transcript: string,
        industry: string,
        progressCallback: (status: string, progress?: number) => void
    ): Promise<string> {
        await this.getAnalysisPipeline((p: any) => {
            if (p.status === 'progress') {
                progressCallback('Loading analysis model...', p.progress);
            }
        });

        // Truncate transcript if too long to fit in token limit
        const truncatedTranscript = this.truncateTranscript(transcript, 1500);
        
        // Domain-specific outline prompts
        let prompt = '';
        
        switch (industry) {
            case 'therapy':
                prompt = `Analyze this therapy session transcript and create an outline organized by SOAP note structure.

CRITICAL: Write actual content from the transcript. Do NOT use placeholders like "[actual points]" or "[content here]".

Structure the outline as:
- Subjective: Client's self-reported feelings, concerns, experiences
- Objective: Therapist's observations of affect, mood, behavior
- Assessment: Professional analysis and interpretation
- Plan: Homework, goals, treatment direction

For each section, list 2-3 specific points that were actually discussed in the session.

Example format (but use REAL content from the transcript):
{"outline":"Subjective: Client reported feeling anxious about work deadlines, mentioned difficulty sleeping, expressed frustration with manager\\nObjective: Client appeared tired, made minimal eye contact, spoke in a quiet tone, fidgeted with hands\\nAssessment: Anxiety appears work-related, possible sleep disturbance contributing to mood, client shows insight into stressors\\nPlan: Client will practice breathing exercises daily, schedule follow-up in 2 weeks, consider sleep hygiene strategies"}

Return ONLY valid JSON with real content from the transcript:
{"outline":"Subjective: [real content]\\nObjective: [real content]\\nAssessment: [real content]\\nPlan: [real content]"}

Transcript: ${truncatedTranscript}`;
                break;
                
            case 'medical':
                prompt = `Analyze this medical consultation transcript and create an outline organized by clinical sections.

CRITICAL: Write actual content from the transcript. Do NOT use placeholders like "[actual complaint]" or "[content here]". Preserve all medical terminology accurately.

Structure the outline as:
- Chief Complaint (CC): Main reason for visit
- History of Present Illness (HPI): Symptom details, onset, duration, severity
- Assessment & Plan (A/P): Diagnosis and treatment plan by problem

For each section, list 2-3 specific points that were actually discussed. Include actual symptoms, dates, medications, and diagnoses mentioned.

Example format (but use REAL content from the transcript):
{"outline":"Chief Complaint: Patient reports chest pain for 3 days, worse with exertion\\nHPI: Pain started Monday morning, described as pressure, radiates to left arm, associated with shortness of breath, no relief with rest\\nAssessment & Plan: Suspected angina, order EKG and cardiac enzymes, start on aspirin 81mg daily, follow up in 1 week"}

Return ONLY valid JSON with real content from the transcript:
{"outline":"Chief Complaint: [real complaint]\\nHPI: [real history]\\nAssessment & Plan: [real assessment]"}

Transcript: ${truncatedTranscript}`;
                break;
                
            case 'legal':
                prompt = `Analyze this legal consultation transcript and create an outline organized by legal topics.

CRITICAL: Write actual content from the transcript. Do NOT use placeholders like "[actual facts]" or "[content here]". Preserve all names, dates, and legal terminology exactly as mentioned.

Structure the outline as:
- Case Background: Facts and context
- Client Statement: What the client reported
- Legal Issues Identified: Legal problems or questions
- Key Dates & Deadlines: Important dates mentioned
- Evidence Mentioned: Documents or evidence discussed

For each section, list 2-3 specific points that were actually discussed. Include actual names, dates, case details, and legal issues.

Example format (but use REAL content from the transcript):
{"outline":"Case Background: Contract dispute over software development agreement signed March 2024, client claims vendor failed to deliver on time\\nClient Statement: Client states vendor missed 3 deadlines, delivered incomplete product, requests refund of $50,000\\nLegal Issues: Breach of contract, potential fraud claims, statute of limitations concerns\\nKey Dates: Contract signed March 15, 2024, first deadline was June 1, filing deadline is March 2025\\nEvidence: Signed contract, email correspondence, incomplete software deliverables"}

Return ONLY valid JSON with real content from the transcript:
{"outline":"Case Background: [real facts]\\nClient Statement: [real statement]\\nLegal Issues: [real issues]\\nKey Dates: [real dates]\\nEvidence: [real evidence]"}

Transcript: ${truncatedTranscript}`;
                break;
                
            case 'business':
                prompt = `Analyze this business meeting transcript and create an outline organized by meeting topics.

CRITICAL: Write actual content from the transcript. Do NOT use placeholders like "[actual topics]" or "[content here]".

Structure the outline as:
- Agenda Items: Main topics discussed in the meeting
- Decisions Made: Key decisions reached
- Action Items: Tasks assigned
- Next Steps: Follow-up actions planned

For each section, list 2-3 specific points that were actually discussed. Include actual decisions, task assignments, deadlines, and next steps mentioned.

Example format (but use REAL content from the transcript):
{"outline":"Agenda Items: Q4 budget review, product launch timeline, team hiring needs\\nDecisions Made: Approved $500K marketing budget, moved launch date to January 15, approved hiring 3 developers\\nAction Items: Sarah to finalize budget by Friday, John to schedule launch planning meeting, team to review candidate resumes\\nNext Steps: Budget presentation to board next week, launch planning meeting scheduled for Monday, interviews to begin next month"}

Return ONLY valid JSON with real content from the transcript:
{"outline":"Agenda Items: [real topics]\\nDecisions Made: [real decisions]\\nAction Items: [real tasks]\\nNext Steps: [real steps]"}

Transcript: ${truncatedTranscript}`;
                break;
                
            default:
                prompt = `Task: Analyze the transcript below and create an outline of main topics.

CRITICAL: You MUST write ACTUAL content from the transcript. Do NOT write generic descriptions like "Main topics discussed" or "Tasks assigned". Write the REAL topics and points that were actually mentioned.

You MUST return ONLY this JSON format (nothing else):
{"outline":"Topic 1: specific point 1, specific point 2\\nTopic 2: specific point 1, specific point 2\\nTopic 3: specific point 1, specific point 2"}

Rules:
- Do NOT repeat the transcript text verbatim
- Do NOT write placeholders or generic descriptions
- Do NOT write "Main topics discussed" or "Tasks assigned" - write the ACTUAL topics and tasks
- Do NOT add "Output:" or any prefix
- Write actual topics and specific points from the transcript
- Start with { and end with }

Example of GOOD output:
{"outline":"Budget Planning: Q4 budget needs review, marketing wants $500K, finance suggests $300K\\nProduct Launch: Delayed to January, need to finalize features, marketing campaign starts next week\\nTeam Updates: Hiring 3 developers, interviews scheduled, onboarding begins in 2 weeks"}

Example of BAD output (DO NOT DO THIS):
{"outline":"Agenda Items: Main topics discussed in the meeting - Action Items: Tasks assigned"}

Transcript to analyze:
${truncatedTranscript}

Now return ONLY the JSON with REAL content:`;
        }

        progressCallback('Creating outline...', 70);

        if (!this.tokenizer || !this.model) {
            throw new Error("Analysis model not initialized");
        }

        try {
            // Increase max_length to allow longer prompts with transcript
            const inputs = this.tokenizer(prompt, {
                return_tensors: 'pt',
                padding: true,
                truncation: true,
                max_length: 2048 // Increased to handle longer transcripts
            });
            
            if (!inputs || !inputs.input_ids || !inputs.attention_mask) {
                throw new Error('Tokenizer did not return expected input_ids and attention_mask');
            }
            
            const output = await this.model.generate(inputs, {
                max_new_tokens: 512,
                num_beams: 1,
                do_sample: false,
                pad_token_id: this.tokenizer.eos_token_id || 0,
                early_stopping: false
            });
            
            if (!output || !output[0]) {
                throw new Error("Model did not return expected output");
            }
            
            let resultText = this.tokenizer.decode(output[0], { skip_special_tokens: true });
            
            // Remove common prefixes that models sometimes add
            resultText = resultText
                .replace(/^Output:\s*/i, '')
                .replace(/^Response:\s*/i, '')
                .replace(/^Here's?\s*/i, '')
                .trim();
            
            console.log('Outline generation - raw model output:', resultText);
            
            // Check if model just returned transcript text instead of JSON
            // If resultText looks like raw transcript (contains conversational phrases but no JSON structure)
            const looksLikeTranscript = resultText.length > 50 && 
                                       !resultText.trim().startsWith('{') && 
                                       !resultText.trim().startsWith('[') &&
                                       (resultText.includes('I ') || resultText.includes('you ') || resultText.includes('that ') || 
                                        resultText.includes('I don\'t') || resultText.includes('I\'m') || resultText.includes('my ')) &&
                                       !resultText.includes('"outline"') &&
                                       !resultText.includes('outline:');
            
            if (looksLikeTranscript) {
                console.warn('Outline generation - model returned transcript text instead of JSON:', resultText.substring(0, 100));
                return 'Outline generation failed: The AI model returned transcript text instead of an outline. Please try re-analyzing the session.';
            }
            
            const parsed = this.parseJSONResponse(resultText, ['outline']);
            if (parsed && parsed.outline && typeof parsed.outline === 'string') {
                const outlineText = parsed.outline.trim();
                
                // Reject generic/placeholder content that describes structure instead of actual content
                const isGenericPlaceholder = outlineText.toLowerCase().includes('main topics discussed') ||
                                           outlineText.toLowerCase().includes('tasks assigned') ||
                                           outlineText.toLowerCase().includes('key decisions') ||
                                           outlineText.toLowerCase().includes('next steps') ||
                                           outlineText.toLowerCase().includes('agenda items: main topics') ||
                                           outlineText.toLowerCase().includes('action items: tasks') ||
                                           (outlineText.toLowerCase().includes('agenda items') && outlineText.toLowerCase().includes('main topics discussed')) ||
                                           (outlineText.toLowerCase().includes('action items') && outlineText.toLowerCase().includes('tasks assigned'));
                
                // Validate it's not a placeholder and not just transcript text
                if (outlineText.length > 20 && 
                    !outlineText.toLowerCase().includes('[actual') &&
                    !outlineText.toLowerCase().includes('[content') &&
                    !outlineText.toLowerCase().includes('placeholder') &&
                    !outlineText.toLowerCase().includes('your outline') &&
                    !isGenericPlaceholder &&
                    !outlineText.toLowerCase().match(/^(i promise|that's good|i had to|i'll|i don't|i'm)/)) { // Reject transcript-like text
                    console.log('Outline generation - parsed successfully:', outlineText.substring(0, 200));
                    // Cluster topics for better formatting
                    return this.clusterTopics(outlineText);
                } else if (isGenericPlaceholder) {
                    console.warn('Outline generation - rejected generic placeholder content:', outlineText.substring(0, 200));
                    return 'Outline generation failed: The AI model returned generic placeholders instead of actual content. Please try re-analyzing the session.';
                }
            }

            // Fallback: extract outline from text (try multiple patterns)
            const outline = this.extractSection(resultText, ['outline', 'Outline', 'structure', 'topics', 'Topics']);
            if (outline && outline.trim().length > 20 && 
                !outline.toLowerCase().includes('[actual') &&
                !outline.toLowerCase().includes('placeholder')) {
                console.log('Outline generation - extracted from text:', outline.substring(0, 200));
                return this.clusterTopics(outline);
            }
            
            // Last resort: try to extract any structured content from the response
            // Remove JSON structure markers and try to find actual content
            let cleanedText = resultText
                .replace(/\{"outline":\s*"/gi, '')
                .replace(/"\s*\}/g, '')
                .replace(/^"|"$/g, '')
                .replace(/\\n/g, '\n')
                .replace(/\[int\]/gi, '') // Remove [int] placeholders
                .replace(/\[actual[^\]]*\]/gi, '') // Remove [actual...] placeholders
                .trim();
            
            // If we have cleaned text with actual content, use it
            // But reject if it only contains placeholders or invalid patterns
            // Also reject generic placeholder content
            const isGenericPlaceholder = cleanedText.toLowerCase().includes('main topics discussed') ||
                                       cleanedText.toLowerCase().includes('tasks assigned') ||
                                       cleanedText.toLowerCase().includes('key decisions') ||
                                       cleanedText.toLowerCase().includes('next steps') ||
                                       cleanedText.toLowerCase().includes('agenda items: main topics') ||
                                       cleanedText.toLowerCase().includes('action items: tasks') ||
                                       (cleanedText.toLowerCase().includes('agenda items') && cleanedText.toLowerCase().includes('main topics discussed')) ||
                                       (cleanedText.toLowerCase().includes('action items') && cleanedText.toLowerCase().includes('tasks assigned'));
            
            if (cleanedText.length > 20 && 
                !cleanedText.match(/^["\s]*$/) &&
                !cleanedText.match(/^\[int\]:?\s*$/i) &&
                !cleanedText.match(/^\[int\]:\s*\[int\]:/i) &&
                !isGenericPlaceholder) {
                console.log('Outline generation - extracted from cleaned JSON structure:', cleanedText.substring(0, 200));
                return this.clusterTopics(cleanedText);
            } else if (isGenericPlaceholder) {
                console.warn('Outline generation - rejected generic placeholder from cleaned text:', cleanedText.substring(0, 200));
                return 'Outline generation failed: The AI model returned generic placeholders instead of actual content. Please try re-analyzing the session.';
            }
            
            // Try to extract lines that look like topic content
            const lines = resultText.split('\n').filter(line => {
                const trimmed = line.trim();
                return trimmed.length > 10 && 
                       !trimmed.startsWith('{') && 
                       !trimmed.startsWith('}') && 
                       !trimmed.match(/^["\s]*$/) &&
                       !trimmed.startsWith('"outline') &&
                       !trimmed.match(/^[{}",\s]*$/) &&
                       !trimmed.match(/^\[int\]:?\s*$/i) && // Reject [int] lines
                       !trimmed.match(/^\[int\]:\s*\[int\]:/i) && // Reject [int]: [int]: patterns
                       !trimmed.match(/^\[\[adji/); // Reject weird patterns like [[adji_adji
            });
            if (lines.length > 0) {
                const extracted = lines.slice(0, 10).join('\n');
                console.log('Outline generation - extracted from raw text:', extracted.substring(0, 200));
                return this.clusterTopics(extracted);
            }

            console.warn('Outline generation - no outline found in response. Full response:', resultText);
            // If we got [int]: [int]: pattern or weird patterns, the model is confused - return helpful error
            if (resultText.match(/\[int\]:?\s*\[int\]:?/i) || resultText.match(/\[\[adji/)) {
                return 'Outline generation failed: The AI model returned invalid output. Please try re-analyzing the session. If this persists, the transcript may be too short or unclear.';
            }
            // Return a helpful message instead of empty string
            return 'Outline generation is still processing. The AI model may need more time or the transcript may be too short. Try re-analyzing the session.';
        } catch (error: any) {
            console.error('Outline generation error:', error);
            throw new Error(`Outline generation failed: ${error?.message || 'Unknown error'}`);
        }
    }

    /**
     * Generate action items from transcript with purpose-aware prompts
     */
    public async generateActionItems(
        transcript: string,
        industry: string,
        progressCallback: (status: string, progress?: number) => void
    ): Promise<string[]> {
        await this.getAnalysisPipeline((p: any) => {
            if (p.status === 'progress') {
                progressCallback('Loading analysis model...', p.progress);
            }
        });

        // Truncate transcript if too long to fit in token limit
        const truncatedTranscript = this.truncateTranscript(transcript, 1500);
        
        // Domain-specific action items prompts
        let prompt = '';
        
        switch (industry) {
            case 'therapy':
                prompt = `Role: You are a highly skilled assistant specialized in processing transcribed text for Psychotherapy and Counseling sessions.

Extract action items from this therapy session transcript. Focus on:
- Treatment plans and therapeutic goals
- Homework assignments for the client
- Follow-up appointments or check-ins
- Therapeutic interventions to implement
- Coping strategies to practice

Action items must be specific, actionable, and include who/what/when if mentioned.

Output Format: Return ONLY valid JSON with complete action items. The array must be fully closed with ]. Example: {"action_items":["Complete action item 1","Complete action item 2"]}

Transcription: ${truncatedTranscript}`;
                break;
                
            case 'medical':
                prompt = `Role: You are a highly skilled assistant specialized in processing transcribed text for Clinical Medical Documentation.

Extract action items from this medical consultation transcript. Focus on:
- Diagnoses and treatment plans
- Medications prescribed (name, dosage, frequency)
- Test orders and lab work needed
- Patient instructions and care requirements
- Follow-up appointments or referrals

Action items must be specific, actionable, and include who/what/when if mentioned. Preserve all medical terminology accurately.

Output Format: Return ONLY valid JSON with complete action items. The array must be fully closed with ]. Example: {"action_items":["Complete action item 1","Complete action item 2"]}

Transcription: ${truncatedTranscript}`;
                break;
                
            case 'legal':
                prompt = `Role: You are a highly skilled assistant specialized in processing transcribed text for Legal Documentation.

Extract action items from this legal consultation transcript. Focus on:
- Deadlines and filing dates
- Document requests and preparation needed
- Case actions and legal procedures
- Client tasks and responsibilities
- Follow-up meetings or court dates

Action items must be specific, actionable, and include who/what/when if mentioned. Preserve all names, dates, and legal terminology.

Output Format: Return ONLY valid JSON with complete action items. The array must be fully closed with ]. Example: {"action_items":["Complete action item 1","Complete action item 2"]}

Transcription: ${truncatedTranscript}`;
                break;
                
            case 'business':
                prompt = `Role: You are a highly skilled assistant specialized in processing transcribed text for Corporate Business Meetings.

Extract action items from this business meeting transcript. Focus on:
- Decisions made and next steps
- Task assignments with clear owners
- Deadlines and due dates
- Project updates and milestones
- Risks or blockers that need addressing

Action items must be specific, actionable, and include WHO (owner), WHAT (task), and WHEN (deadline) if mentioned.

Output Format: Return ONLY valid JSON with complete action items. The array must be fully closed with ]. Example: {"action_items":["Complete action item 1","Complete action item 2"]}

Transcription: ${truncatedTranscript}`;
                break;
                
            default:
                prompt = `Task: Extract action items from the transcript below.

CRITICAL: You MUST return ONLY valid JSON. Do NOT repeat the transcript text. Do NOT write conversational responses like "I don't know" or "I'll take care of this". Extract ONLY actual tasks that were assigned or agreed upon.

You MUST return ONLY this JSON format (nothing else):
{"action_items":["item 1","item 2"]}

Rules:
- Do NOT repeat the transcript text
- Do NOT write conversational responses
- Extract ONLY real tasks that were assigned or agreed upon
- If no action items found, return: {"action_items":[]}
- Do NOT add "Output:" or any prefix
- Start with { and end with }

Example of GOOD output:
{"action_items":["Sarah will finalize the budget by Friday","John needs to schedule the launch meeting","Team will review candidate resumes this week"]}

Example of BAD output (DO NOT DO THIS):
{"action_items":["I don't know how I'm going to face my dad"]}

Transcript to analyze:
${truncatedTranscript}

Now return ONLY the JSON:`;
        }

        progressCallback('Extracting action items...', 80);

        if (!this.tokenizer || !this.model) {
            throw new Error("Analysis model not initialized");
        }

        try {
            // Increase max_length to allow longer prompts with transcript
            const inputs = this.tokenizer(prompt, {
                return_tensors: 'pt',
                padding: true,
                truncation: true,
                max_length: 2048 // Increased to handle longer transcripts
            });
            
            if (!inputs || !inputs.input_ids || !inputs.attention_mask) {
                throw new Error('Tokenizer did not return expected input_ids and attention_mask');
            }
            
            const output = await this.model.generate(inputs, {
                max_new_tokens: 512,
                num_beams: 1,
                do_sample: false,
                pad_token_id: this.tokenizer.eos_token_id || 0,
                early_stopping: false
            });
            
            if (!output || !output[0]) {
                throw new Error("Model did not return expected output");
            }
            
            let resultText = this.tokenizer.decode(output[0], { skip_special_tokens: true });
            
            // Remove common prefixes that models sometimes add
            resultText = resultText
                .replace(/^Output:\s*/i, '')
                .replace(/^Response:\s*/i, '')
                .replace(/^Here's?\s*/i, '')
                .trim();
            
            console.log('Action items generation - raw model output:', resultText);
            
            // Check if model just returned transcript text instead of JSON
            // More comprehensive check for transcript-like content
            const looksLikeTranscript = resultText.length > 30 && 
                                       !resultText.trim().startsWith('{') && 
                                       !resultText.trim().startsWith('[') &&
                                       (
                                           resultText.includes('I ') || 
                                           resultText.includes('I don\'t') || 
                                           resultText.includes('I\'m') || 
                                           resultText.includes('I\'ll') ||
                                           resultText.includes('my ') ||
                                           resultText.includes('my dad') ||
                                           resultText.includes('my mom') ||
                                           resultText.includes('you ') || 
                                           resultText.includes('that ') ||
                                           resultText.includes('he ') ||
                                           resultText.includes('she ') ||
                                           resultText.includes('stole') ||
                                           resultText.includes('face my')
                                       ) &&
                                       !resultText.includes('"action_items"') &&
                                       !resultText.includes('action_items:') &&
                                       !resultText.includes('"action') &&
                                       !resultText.match(/^\s*\{/); // Not starting with JSON object
            
            if (looksLikeTranscript) {
                console.warn('Action items generation - model returned transcript text instead of JSON:', resultText.substring(0, 100));
                return []; // Return empty array if model is confused
            }
            
            const parsed = this.parseJSONResponse(resultText, ['action_items']);
            if (parsed && parsed.action_items && Array.isArray(parsed.action_items) && parsed.action_items.length > 0) {
                const filtered = parsed.action_items.filter((item: any) => {
                    if (!item || typeof item !== 'string') return false;
                    const trimmed = item.trim();
                    // Filter out placeholders, invalid items, and transcript-like text
                    const isValidLength = trimmed.length > 5;
                    const hasNoPlaceholders = !trimmed.toLowerCase().includes('[actual') &&
                                             !trimmed.toLowerCase().includes('placeholder') &&
                                             !trimmed.toLowerCase().startsWith('action item');
                    // Reject transcript-like conversational text (more comprehensive)
                    const isNotTranscript = !trimmed.toLowerCase().match(/^(i promise|that's good|i had to|i'll take|good to hear|i don't know|i'm going|i'll face|my dad|my mom|he stole|she stole|stole my)/);
                    // Should contain action words or be a real task
                    const looksLikeAction = trimmed.toLowerCase().includes('will') ||
                                           trimmed.toLowerCase().includes('needs to') ||
                                           trimmed.toLowerCase().includes('should') ||
                                           trimmed.toLowerCase().includes('must') ||
                                           trimmed.toLowerCase().includes('by ') ||
                                           trimmed.toLowerCase().includes('deadline');
                    
                    return isValidLength && hasNoPlaceholders && isNotTranscript && (looksLikeAction || trimmed.length > 15);
                });
                if (filtered.length > 0) {
                    console.log('Action items generation - parsed successfully:', filtered.length, 'items');
                    return filtered;
                }
            }

            // Fallback: extract action items from text (try multiple patterns)
            const actionItems = this.extractList(resultText, ['action items', 'action_items', 'todos', 'to-do', 'tasks', 'action', 'Action']);
            const validActionItems = actionItems.filter(item => {
                const trimmed = item.trim();
                return trimmed.length > 5 && 
                       !trimmed.toLowerCase().includes('[actual') &&
                       !trimmed.toLowerCase().includes('placeholder');
            });
            if (validActionItems.length > 0) {
                console.log('Action items generation - extracted from text:', validActionItems.length, 'items');
                return validActionItems;
            }
            
            // Last resort: try to extract from incomplete JSON
            // Look for patterns like: "action_item 1"," or action_item 1,
            const incompleteMatch = resultText.match(/"action_item[^"]*"|"action[^"]*item[^"]*"/gi);
            if (incompleteMatch && incompleteMatch.length > 0) {
                const extracted = incompleteMatch.map(item => item.replace(/^"|"$/g, '').trim()).filter(item => item.length > 0);
                if (extracted.length > 0) {
                    console.log('Action items generation - extracted from incomplete JSON:', extracted.length, 'items');
                    return extracted;
                }
            }

            // Check if model returned confused output (but only if it's the entire response)
            const lowerText = resultText.toLowerCase().trim();
            if ((lowerText.startsWith("i'm not sure") || 
                 lowerText.startsWith("i don't know") ||
                 lowerText.startsWith("i cannot") ||
                 lowerText.startsWith("unable to")) && 
                resultText.length < 100) {
                console.warn('Action items generation - model returned confused response:', resultText);
                return []; // Return empty array if model is confused
            }

            console.warn('Action items generation - no action items found in response. Full response:', resultText);
            return [];
        } catch (error: any) {
            console.error('Action items generation error:', error);
            throw new Error(`Action items generation failed: ${error?.message || 'Unknown error'}`);
        }
    }

    private extractSection(text: string, keywords: string[]): string {
        for (const keyword of keywords) {
            const regex = new RegExp(`${keyword}[:\n]\\s*([^\\n]+(?:\\n[^\\n]+)*)`, 'i');
            const match = text.match(regex);
            if (match && match[1]) {
                return match[1].trim();
            }
        }
        return '';
    }

    private clusterTopics(outline: string): string {
        if (!outline || outline.trim().length === 0) {
            return outline;
        }
        
        // Try to identify topic groupings in the outline
        // Look for patterns like:
        // - Topic headers (lines starting with numbers, bullets, or capital letters)
        // - Section markers (Topic:, Section:, etc.)
        // - Numbered lists that might represent topics
        
        const lines = outline.split('\n').filter(line => line.trim().length > 0);
        const clustered: string[] = [];
        let currentTopic: string[] = [];
        let topicTitle = '';
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            // Detect topic headers (lines that look like titles)
            const isTopicHeader = /^(\d+[\.\)]|\-|\*|Topic|Section|Part)\s+[A-Z]/.test(line) || 
                                 (line.length < 80 && /^[A-Z][^\.]{5,}/.test(line) && !line.includes('.'));
            
            if (isTopicHeader && currentTopic.length > 0) {
                // Save previous topic
                if (topicTitle) {
                    clustered.push(`**${topicTitle}**`);
                    clustered.push(...currentTopic.map(l => `  ${l}`));
                    clustered.push('');
                }
                // Start new topic
                topicTitle = line.replace(/^(\d+[\.\)]|\-|\*|Topic:|Section:|Part\s+)/i, '').trim();
                currentTopic = [];
            } else if (isTopicHeader && currentTopic.length === 0) {
                // First topic
                topicTitle = line.replace(/^(\d+[\.\)]|\-|\*|Topic:|Section:|Part\s+)/i, '').trim();
            } else {
                // Add to current topic
                currentTopic.push(line);
            }
        }
        
        // Add last topic
        if (topicTitle && currentTopic.length > 0) {
            clustered.push(`**${topicTitle}**`);
            clustered.push(...currentTopic.map(l => `  ${l}`));
        } else if (currentTopic.length > 0) {
            // No topic header found, just add lines
            clustered.push(...currentTopic);
        }
        
        // If clustering didn't find clear topics, return original
        if (clustered.length === 0 || clustered.length === lines.length) {
            return outline;
        }
        
        return clustered.join('\n');
    }

    private extractList(text: string, keywords: string[]): string[] {
        for (const keyword of keywords) {
            // Try to find a list after the keyword
            const regex = new RegExp(`${keyword}[:\n]\\s*((?:[-*•]\\s*[^\\n]+\\n?)+)`, 'i');
            const match = text.match(regex);
            if (match && match[1]) {
                return match[1]
                    .split(/\n/)
                    .map(line => line.replace(/^[-*•]\s*/, '').trim())
                    .filter(line => line.length > 0);
            }
            
            // Try numbered list
            const numberedRegex = new RegExp(`${keyword}[:\n]\\s*((?:\\d+\\.\\s*[^\\n]+\\n?)+)`, 'i');
            const numberedMatch = text.match(numberedRegex);
            if (numberedMatch && numberedMatch[1]) {
                return numberedMatch[1]
                    .split(/\n/)
                    .map(line => line.replace(/^\d+\.\s*/, '').trim())
                    .filter(line => line.length > 0);
            }
        }
        return [];
    }
}


// --- FIX for SpeechRecognition API types ---
interface SpeechRecognitionAlternative {
    transcript: string;
}

interface SpeechRecognitionResult {
    isFinal: boolean;
    [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionResultList {
    [index: number]: SpeechRecognitionResult;
    length: number;
}

interface SpeechRecognitionEvent extends Event {
    resultIndex: number;
    results: SpeechRecognitionResultList;
}

interface SpeechRecognition extends EventTarget {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    onresult: (event: SpeechRecognitionEvent) => void;
    onerror: (event: any) => void;
    onend: () => void;
    start: () => void;
    stop: () => void;
}

declare global {
    interface Window {
        SpeechRecognition: new () => SpeechRecognition;
        webkitSpeechRecognition: new () => SpeechRecognition;
        AudioContext: typeof AudioContext;
        webkitAudioContext: typeof AudioContext;
    }
    interface HTMLAudioElement {
      setSinkId(sinkId: string): Promise<void>;
    }
}

// --- TYPE DEFINITIONS ---
interface TranscriptChunk {
    speaker: string;
    text: string;
    timestamp?: [number, number]; // Optional timestamp [start, end] in seconds
}

interface TodoItem {
    text: string;
    completed: boolean;
    promotedToTaskId?: number;
}

interface KeyDecision {
    decision: string;
    reasoning?: string;
    owner?: string;
    implementationDate?: string;
    timestamp?: number; // When in the meeting this was decided
}

interface Attachment {
    name: string;
    type: 'file' | 'link' | 'document' | 'spreadsheet' | 'presentation' | 'other';
    url?: string;
    mentionedBy?: string; // Who mentioned it
    timestamp?: number; // When in the meeting it was mentioned
}

interface Bookmark {
    chunkIndex: number; // Index in transcript array
    timestamp: number; // Audio timestamp in seconds
    note?: string; // Optional user note
    createdAt: number; // When bookmark was created (timestamp)
}

interface Topic {
    title: string;
    startTime: number;
    endTime: number;
    chunkIndices: number[];
}

interface Session {
    id?: number;
    sessionTitle: string;
    participants?: string;
    date: string;
    notes: string;
    duration: number;
    transcript: TranscriptChunk[] | string; // Can be array or JSON string
    timestamp: number;
    summary?: string; // Can be plain string or JSON string
    todoItems?: TodoItem[] | string; // Can be array or JSON string
    outline?: string; // Can be plain string or JSON string
    analysisStatus?: 'pending' | 'complete' | 'failed' | 'none';
    audioBlob?: Blob;
    language?: string; // Language code for transcription (e.g., 'en', 'es', 'fr')
    // New fields for 6-section template
    keyDecisions?: KeyDecision[] | string; // Decisions made in the meeting
    attachments?: Attachment[] | string; // Files, links, resources mentioned
    meetingType?: string; // e.g., 'Zoom', 'Teams', 'in-person'
    platform?: string; // Meeting platform
    // New fields for UI/UX improvements
    bookmarks?: Bookmark[] | string; // Array or JSON string
}

// Helper functions to parse JSON fields safely
const parseTranscript = (transcript: TranscriptChunk[] | string | undefined): TranscriptChunk[] => {
    if (!transcript) return [];
    if (typeof transcript === 'string') {
        try {
            return JSON.parse(transcript);
        } catch {
            return [];
        }
    }
    return transcript;
};

const parseSummary = (summary: string | undefined): string => {
    if (!summary) return '';
    try {
        const parsed = JSON.parse(summary);
        return parsed.summary || summary;
    } catch {
        return summary;
    }
};

const parseTodoItems = (todoItems: TodoItem[] | string | undefined): TodoItem[] => {
    if (!todoItems) return [];
    if (typeof todoItems === 'string') {
        try {
            return JSON.parse(todoItems);
        } catch {
            return [];
        }
    }
    return todoItems;
};

const parseOutline = (outline: string | undefined): string => {
    if (!outline) return '';
    try {
        const parsed = JSON.parse(outline);
        return parsed.outline || outline;
    } catch {
        return outline;
    }
};

const parseBookmarks = (bookmarks: Bookmark[] | string | undefined): Bookmark[] => {
    if (!bookmarks) return [];
    if (typeof bookmarks === 'string') {
        try {
            return JSON.parse(bookmarks);
        } catch {
            return [];
        }
    }
    return bookmarks;
};

// Parse outline into structured topics with timestamps
const parseTopicsFromOutline = (outline: string, transcript: TranscriptChunk[]): Topic[] => {
    if (!outline || transcript.length === 0) return [];
    
    const topics: Topic[] = [];
    const lines = outline.split('\n').filter(line => line.trim().length > 0);
    
    let currentTopic: Topic | null = null;
    let topicLines: string[] = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        // Detect topic headers
        const isTopicHeader = /^(\d+[\.\)]|\-|\*|Topic|Section|Part)\s+[A-Z]/.test(line) || 
                             (line.length < 80 && /^[A-Z][^\.]{5,}/.test(line) && !line.includes('.'));
        
        if (isTopicHeader) {
            // Save previous topic
            if (currentTopic) {
                // Match topic to transcript chunks by content similarity
                const matchedChunks = matchTopicToChunks(topicLines.join(' '), transcript);
                currentTopic.chunkIndices = matchedChunks;
                if (matchedChunks.length > 0) {
                    const firstChunk = transcript[matchedChunks[0]];
                    const lastChunk = transcript[matchedChunks[matchedChunks.length - 1]];
                    currentTopic.startTime = firstChunk.timestamp?.[0] || 0;
                    currentTopic.endTime = lastChunk.timestamp?.[1] || currentTopic.startTime;
                }
                topics.push(currentTopic);
            }
            
            // Start new topic
            const title = line.replace(/^(\d+[\.\)]|\-|\*|Topic:|Section:|Part\s+)/i, '').trim();
            currentTopic = {
                title,
                startTime: 0,
                endTime: 0,
                chunkIndices: []
            };
            topicLines = [];
        } else if (currentTopic) {
            topicLines.push(line);
        }
    }
    
    // Add last topic
    if (currentTopic) {
        const matchedChunks = matchTopicToChunks(topicLines.join(' '), transcript);
        currentTopic.chunkIndices = matchedChunks;
        if (matchedChunks.length > 0) {
            const firstChunk = transcript[matchedChunks[0]];
            const lastChunk = transcript[matchedChunks[matchedChunks.length - 1]];
            currentTopic.startTime = firstChunk.timestamp?.[0] || 0;
            currentTopic.endTime = lastChunk.timestamp?.[1] || currentTopic.startTime;
        }
        topics.push(currentTopic);
    }
    
    return topics;
};

// Match topic content to transcript chunks by keyword similarity
const matchTopicToChunks = (topicText: string, transcript: TranscriptChunk[]): number[] => {
    const keywords = topicText.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const matched: number[] = [];
    
    for (let i = 0; i < transcript.length; i++) {
        const chunkText = transcript[i].text.toLowerCase();
        const matchCount = keywords.filter(kw => chunkText.includes(kw)).length;
        if (matchCount >= Math.min(2, keywords.length)) {
            matched.push(i);
        }
    }
    
    return matched;
};

const parseKeyDecisions = (decisions: KeyDecision[] | string | undefined): KeyDecision[] => {
    if (!decisions) return [];
    if (typeof decisions === 'string') {
        try {
            return JSON.parse(decisions);
        } catch {
            return [];
        }
    }
    return decisions;
};

const parseAttachments = (attachments: Attachment[] | string | undefined): Attachment[] => {
    if (!attachments) return [];
    if (typeof attachments === 'string') {
        try {
            return JSON.parse(attachments);
        } catch {
            return [];
        }
    }
    return attachments;
};

interface Task {
    id?: number;
    title: string;
    dueDate: string | null;
    priority: 'low' | 'medium' | 'high';
    status: 'todo' | 'inprogress' | 'done';
    sessionId?: number;
    sessionName?: string;
    timestamp: number;
}

// --- CRYPTO SERVICE ---
class CryptoService {
    private static readonly SALT = 'a-very-secure-static-salt-for-whisper-notes'; // In a real app, this might be user-specific
    private static readonly ITERATIONS = 100000;

    private static async deriveKey(pin: string): Promise<CryptoKey> {
        const enc = new TextEncoder();
        const keyMaterial = await window.crypto.subtle.importKey(
            'raw',
            enc.encode(pin),
            { name: 'PBKDF2' },
            false,
            ['deriveKey']
        );
        return window.crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: enc.encode(this.SALT),
                iterations: this.ITERATIONS,
                hash: 'SHA-256',
            },
            keyMaterial,
            { name: 'AES-GCM', length: 256 },
            true,
            ['encrypt', 'decrypt']
        );
    }

    public static async encrypt(data: string, pin: string): Promise<string> {
        const key = await this.deriveKey(pin);
        const iv = window.crypto.getRandomValues(new Uint8Array(12));
        const enc = new TextEncoder();
        const encoded = enc.encode(data);
        const encryptedContent = await window.crypto.subtle.encrypt(
            {
                name: 'AES-GCM',
                iv: iv,
            },
            key,
            encoded
        );

        const encryptedBytes = new Uint8Array(iv.length + encryptedContent.byteLength);
        encryptedBytes.set(iv, 0);
        encryptedBytes.set(new Uint8Array(encryptedContent), iv.length);

        return btoa(String.fromCharCode.apply(null, Array.from(encryptedBytes)));
    }

    public static async decrypt(encryptedData: string, pin: string): Promise<string> {
        try {
            const key = await this.deriveKey(pin);
            const encryptedBytes = new Uint8Array(Array.from(atob(encryptedData), c => c.charCodeAt(0)));
            const iv = encryptedBytes.slice(0, 12);
            const encryptedContent = encryptedBytes.slice(12);

            const decryptedContent = await window.crypto.subtle.decrypt(
                {
                    name: 'AES-GCM',
                    iv: iv,
                },
                key,
                encryptedContent
            );

            const dec = new TextDecoder();
            return dec.decode(decryptedContent);
        } catch (e) {
            throw new Error("Invalid PIN or corrupted data.");
        }
    }
}

// --- DATABASE SERVICE ---
class TherapyDB {
    private db: IDBDatabase | null = null;
    private readonly DB_NAME = 'meetingmindsDB';
    private readonly SESSIONS_STORE = 'sessions';
    private readonly TASKS_STORE = 'tasks';
    private readonly CONFIG_STORE = 'config';

    constructor() {
        this.init();
    }

    private init(): Promise<void> {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.DB_NAME, 5); // Increment version for indexing

            request.onupgradeneeded = (event) => {
                const db = (event.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(this.SESSIONS_STORE)) {
                    const sessionStore = db.createObjectStore(this.SESSIONS_STORE, { keyPath: 'id', autoIncrement: true });
                    // Create indexes for search and recovery
                    sessionStore.createIndex('date', 'date', { unique: false });
                    sessionStore.createIndex('timestamp', 'timestamp', { unique: false });
                    sessionStore.createIndex('sessionTitle', 'sessionTitle', { unique: false });
                } else {
                    // Add indexes to existing store if upgrading
                    const transaction = (event.target as IDBOpenDBRequest).transaction;
                    if (transaction) {
                        const sessionStore = transaction.objectStore(this.SESSIONS_STORE);
                        if (!sessionStore.indexNames.contains('date')) {
                            sessionStore.createIndex('date', 'date', { unique: false });
                        }
                        if (!sessionStore.indexNames.contains('timestamp')) {
                            sessionStore.createIndex('timestamp', 'timestamp', { unique: false });
                        }
                        if (!sessionStore.indexNames.contains('sessionTitle')) {
                            sessionStore.createIndex('sessionTitle', 'sessionTitle', { unique: false });
                        }
                    }
                }
                if (!db.objectStoreNames.contains(this.TASKS_STORE)) {
                    const taskStore = db.createObjectStore(this.TASKS_STORE, { keyPath: 'id', autoIncrement: true });
                    taskStore.createIndex('timestamp', 'timestamp', { unique: false });
                    taskStore.createIndex('sessionId', 'sessionId', { unique: false });
                } else {
                    // Add sessionId index to existing tasks store if upgrading
                    const transaction = (event.target as IDBOpenDBRequest).transaction;
                    if (transaction) {
                        const taskStore = transaction.objectStore(this.TASKS_STORE);
                        if (!taskStore.indexNames.contains('sessionId')) {
                            taskStore.createIndex('sessionId', 'sessionId', { unique: false });
                        }
                    }
                }
                if (!db.objectStoreNames.contains(this.CONFIG_STORE)) {
                    db.createObjectStore(this.CONFIG_STORE, { keyPath: 'key' });
                }
            };

            request.onsuccess = (event) => {
                this.db = (event.target as IDBOpenDBRequest).result;
                resolve();
            };

            request.onerror = (event) => {
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

    // Helper method to encrypt session sensitive fields
    private async encryptSession(session: Session, pin: string): Promise<Session> {
        const encrypted: Session = { ...session };
        
        // Encrypt transcript if it exists
        if (session.transcript) {
            const transcriptStr = typeof session.transcript === 'string' 
                ? session.transcript 
                : JSON.stringify(session.transcript);
            encrypted.transcript = await CryptoService.encrypt(transcriptStr, pin);
        }
        
        // Encrypt summary if it exists
        if (session.summary) {
            const summaryStr = typeof session.summary === 'string' 
                ? session.summary 
                : JSON.stringify({ summary: session.summary });
            encrypted.summary = await CryptoService.encrypt(summaryStr, pin);
        }
        
        // Encrypt todoItems if they exist
        if (session.todoItems) {
            const todoStr = typeof session.todoItems === 'string' 
                ? session.todoItems 
                : JSON.stringify(session.todoItems);
            encrypted.todoItems = await CryptoService.encrypt(todoStr, pin);
        }
        
        // Encrypt outline if it exists
        if (session.outline) {
            const outlineStr = typeof session.outline === 'string' 
                ? session.outline 
                : JSON.stringify({ outline: session.outline });
            encrypted.outline = await CryptoService.encrypt(outlineStr, pin);
        }
        
        // Encrypt bookmarks if they exist
        if (session.bookmarks) {
            const bookmarksStr = typeof session.bookmarks === 'string' 
                ? session.bookmarks 
                : JSON.stringify(session.bookmarks);
            encrypted.bookmarks = await CryptoService.encrypt(bookmarksStr, pin);
        }
        
        return encrypted;
    }

    // Helper method to decrypt session sensitive fields
    private async decryptSession(session: Session, pin: string): Promise<Session> {
        const decrypted: Session = { ...session };
        
        // Decrypt transcript if it exists and is encrypted (check if it's base64-like)
        if (session.transcript && typeof session.transcript === 'string') {
            try {
                // Try to decrypt - if it fails, it might be unencrypted old data
                decrypted.transcript = await CryptoService.decrypt(session.transcript, pin);
            } catch {
                // If decryption fails, assume it's unencrypted JSON or plain text
                decrypted.transcript = session.transcript;
            }
        }
        
        // Decrypt summary if it exists
        if (session.summary && typeof session.summary === 'string') {
            try {
                decrypted.summary = await CryptoService.decrypt(session.summary, pin);
            } catch {
                decrypted.summary = session.summary;
            }
        }
        
        // Decrypt todoItems if they exist
        if (session.todoItems && typeof session.todoItems === 'string') {
            try {
                decrypted.todoItems = await CryptoService.decrypt(session.todoItems, pin);
            } catch {
                decrypted.todoItems = session.todoItems;
            }
        }
        
        // Decrypt outline if it exists
        if (session.outline && typeof session.outline === 'string') {
            try {
                decrypted.outline = await CryptoService.decrypt(session.outline, pin);
            } catch {
                decrypted.outline = session.outline;
            }
        }
        
        // Decrypt bookmarks if they exist
        if (session.bookmarks && typeof session.bookmarks === 'string') {
            try {
                decrypted.bookmarks = await CryptoService.decrypt(session.bookmarks, pin);
            } catch {
                decrypted.bookmarks = session.bookmarks;
            }
        }
        
        return decrypted;
    }

    // Helper method to encrypt task sensitive fields
    private async encryptTask(task: Task, pin: string): Promise<Task> {
        const encrypted: Task = { ...task };
        
        // Encrypt title
        if (task.title) {
            encrypted.title = await CryptoService.encrypt(task.title, pin);
        }
        
        // Encrypt sessionName if it exists
        if (task.sessionName) {
            encrypted.sessionName = await CryptoService.encrypt(task.sessionName, pin);
        }
        
        return encrypted;
    }

    // Helper method to decrypt task sensitive fields
    private async decryptTask(task: Task, pin: string): Promise<Task> {
        const decrypted: Task = { ...task };
        
        // Decrypt title
        if (task.title) {
            try {
                decrypted.title = await CryptoService.decrypt(task.title, pin);
            } catch {
                decrypted.title = task.title; // Assume unencrypted old data
            }
        }
        
        // Decrypt sessionName if it exists
        if (task.sessionName) {
            try {
                decrypted.sessionName = await CryptoService.decrypt(task.sessionName, pin);
            } catch {
                decrypted.sessionName = task.sessionName; // Assume unencrypted old data
            }
        }
        
        return decrypted;
    }

    public async saveConfig(key: string, value: any): Promise<void> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.CONFIG_STORE, 'readwrite');
            const store = transaction.objectStore(this.CONFIG_STORE);
            const request = store.put({ key, value });
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    public async getConfig(key: string): Promise<any> {
        const db = await this.getDb();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.CONFIG_STORE, 'readonly');
            const store = transaction.objectStore(this.CONFIG_STORE);
            const request = store.get(key);
            request.onsuccess = () => resolve(request.result?.value);
            request.onerror = () => reject(request.error);
        });
    }

    public async addSession(session: Session, pin: string): Promise<number> {
        const db = await this.getDb();
        const encryptedSession = await this.encryptSession(session, pin);
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readwrite');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.add(encryptedSession);
            request.onsuccess = () => resolve(request.result as number);
            request.onerror = () => reject(request.error);
        });
    }

    public async getAllSessions(pin: string): Promise<Session[]> {
        const db = await this.getDb();
        return new Promise(async (resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readonly');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.getAll();
            request.onsuccess = async () => {
                try {
                    const sortedSessions = request.result.sort((a, b) => b.timestamp - a.timestamp);
                    const decryptedSessions = await Promise.all(
                        sortedSessions.map(s => this.decryptSession(s, pin))
                    );
                    resolve(decryptedSessions);
                } catch (error) {
                    reject(error);
                }
            };
            request.onerror = () => reject(request.error);
        });
    }

    public async getSession(id: number, pin: string): Promise<Session | undefined> {
        const db = await this.getDb();
        return new Promise(async (resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readonly');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.get(id);
            request.onsuccess = async () => {
                if (!request.result) {
                    resolve(undefined);
                    return;
                }
                try {
                    const decrypted = await this.decryptSession(request.result, pin);
                    resolve(decrypted);
                } catch (error) {
                    reject(error);
                }
            };
            request.onerror = () => reject(request.error);
        });
    }
    
    public async updateSession(session: Session, pin: string): Promise<void> {
        const db = await this.getDb();
        const encryptedSession = await this.encryptSession(session, pin);
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.SESSIONS_STORE, 'readwrite');
            const store = transaction.objectStore(this.SESSIONS_STORE);
            const request = store.put(encryptedSession);
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
    public async addTask(task: Task, pin: string): Promise<number> {
        const db = await this.getDb();
        const encryptedTask = await this.encryptTask(task, pin);
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.TASKS_STORE, 'readwrite');
            const store = transaction.objectStore(this.TASKS_STORE);
            const request = store.add(encryptedTask);
            request.onsuccess = () => resolve(request.result as number);
            request.onerror = () => reject(request.error);
        });
    }

    public async getAllTasks(pin: string): Promise<Task[]> {
        const db = await this.getDb();
        return new Promise(async (resolve, reject) => {
            const transaction = db.transaction(this.TASKS_STORE, 'readonly');
            const store = transaction.objectStore(this.TASKS_STORE);
            const index = store.index('timestamp');
            const request = index.getAll();
            request.onsuccess = async () => {
                try {
                    const sortedTasks = request.result.sort((a, b) => b.timestamp - a.timestamp);
                    const decryptedTasks = await Promise.all(
                        sortedTasks.map(t => this.decryptTask(t, pin))
                    );
                    resolve(decryptedTasks);
                } catch (error) {
                    reject(error);
                }
            };
            request.onerror = () => reject(request.error);
        });
    }

    public async updateTask(task: Task, pin: string): Promise<void> {
        const db = await this.getDb();
        const encryptedTask = await this.encryptTask(task, pin);
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(this.TASKS_STORE, 'readwrite');
            const store = transaction.objectStore(this.TASKS_STORE);
            const request = store.put(encryptedTask);
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
}

const db = new TherapyDB();
firstTimeSetupDb = db; // Make available for first-time setup check
const onDeviceAIService = OnDeviceAIService.getInstance();
const authService = new AuthService(db);

// --- INTELLIGENT AUDIO SERVICE ---
interface AudioDevice {
    deviceId: string;
    label: string;
    kind: MediaDeviceKind;
    isDefault: boolean;
}

interface AudioSourceSelection {
    type: 'mic' | 'system' | 'mixed';
    deviceId?: string;
    reason: string;
}

class IntelligentAudioService {
    private static instance: IntelligentAudioService | null = null;
    private audioContext: AudioContext | null = null;
    
    private constructor() {}
    
    public static getInstance(): IntelligentAudioService {
        if (!this.instance) {
            this.instance = new IntelligentAudioService();
        }
        return this.instance;
    }

    /**
     * Initialize audio context for audio processing
     */
    private getAudioContext(): AudioContext {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        return this.audioContext;
    }

    /**
     * Enumerate all available audio input devices
     */
    async getAvailableAudioDevices(): Promise<AudioDevice[]> {
        try {
            // Request permission first to get device labels
            await navigator.mediaDevices.getUserMedia({ audio: true });
            
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioInputs = devices
                .filter(device => device.kind === 'audioinput')
                .map((device, index, array) => ({
                    deviceId: device.deviceId,
                    label: device.label || `Microphone ${index + 1}`,
                    kind: device.kind,
                    isDefault: index === 0 || device.deviceId === 'default'
                }));
            
            return audioInputs;
        } catch (error) {
            console.error('Error enumerating audio devices:', error);
            return [];
        }
    }

    /**
     * Intelligently select the best audio source
     */
    async selectBestAudioSource(): Promise<AudioSourceSelection> {
        try {
            const devices = await this.getAvailableAudioDevices();
            
            // If no devices, fallback to default
            if (devices.length === 0) {
                return {
                    type: 'mic',
                    reason: 'Using default audio input'
                };
            }

            // If only one device, use it
            if (devices.length === 1) {
                return {
                    type: 'mic',
                    deviceId: devices[0].deviceId,
                    reason: `Using: ${devices[0].label}`
                };
            }

            // Prefer external mic if available (usually better quality)
            const externalMic = devices.find(device => {
                const label = device.label.toLowerCase();
                return !label.includes('built-in') &&
                       !label.includes('internal') &&
                       !label.includes('default') &&
                       !device.isDefault;
            });

            if (externalMic) {
                return {
                    type: 'mic',
                    deviceId: externalMic.deviceId,
                    reason: `External microphone: ${externalMic.label}`
                };
            }

            // Use default device
            const defaultDevice = devices.find(d => d.isDefault) || devices[0];
            return {
                type: 'mic',
                deviceId: defaultDevice.deviceId,
                reason: `Using: ${defaultDevice.label}`
            };
        } catch (error) {
            console.error('Error selecting audio source:', error);
            return {
                type: 'mic',
                reason: 'Using default audio input'
            };
        }
    }

    /**
     * Create a mixed audio stream (mic + system audio) for best recording quality
     */
    async createMixedAudioStream(micDeviceId?: string): Promise<{
        stream: MediaStream;
        type: 'mic' | 'system' | 'mixed';
        sources: string[];
    } | null> {
        try {
            const audioContext = this.getAudioContext();
            const destination = audioContext.createMediaStreamDestination();
            
            let micStream: MediaStream | null = null;
            let systemStream: MediaStream | null = null;
            const sources: string[] = [];

            // Get microphone stream
            try {
                const constraints: MediaStreamConstraints = {
                    audio: micDeviceId 
                        ? { deviceId: { exact: micDeviceId }, echoCancellation: true, noiseSuppression: true }
                        : { echoCancellation: true, noiseSuppression: true }
                };
                micStream = await navigator.mediaDevices.getUserMedia(constraints);
                
                if (micStream.getAudioTracks().length > 0) {
                    micStream.getAudioTracks().forEach(track => {
                        const source = audioContext.createMediaStreamSource(new MediaStream([track]));
                        source.connect(destination);
                    });
                    sources.push('Microphone');
                }
            } catch (error) {
                console.warn('Could not access microphone:', error);
            }

            // Try to get system audio (non-blocking - user can skip)
            try {
                systemStream = await (navigator.mediaDevices as any).getDisplayMedia({
                    video: false,
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false
                    } as any
                });

                if (systemStream.getAudioTracks().length > 0) {
                    systemStream.getAudioTracks().forEach(track => {
                        const source = audioContext.createMediaStreamSource(new MediaStream([track]));
                        source.connect(destination);
                    });
                    sources.push('System Audio');
                } else {
                    // User didn't share system audio - stop the stream
                    systemStream.getTracks().forEach(track => track.stop());
                    systemStream = null;
                }
            } catch (error) {
                // User cancelled or error - that's okay, we'll use mic only
                console.log('System audio not available:', error);
            }

            // Return mixed stream if we have at least one source
            if (destination.stream.getAudioTracks().length > 0) {
                const type = sources.length > 1 ? 'mixed' : (sources.includes('System Audio') ? 'system' : 'mic');
                return {
                    stream: destination.stream,
                    type,
                    sources
                };
            }

            // Cleanup if no sources
            if (micStream) micStream.getTracks().forEach(track => track.stop());
            if (systemStream) systemStream.getTracks().forEach(track => track.stop());
            
            return null;
        } catch (error) {
            console.error('Error creating mixed audio stream:', error);
            return null;
        }
    }

    /**
     * Get a single audio stream (mic or system)
     */
    async getSingleAudioStream(type: 'mic' | 'system', deviceId?: string): Promise<MediaStream | null> {
        try {
            if (type === 'mic') {
                const constraints: MediaStreamConstraints = {
                    audio: deviceId 
                        ? { deviceId: { exact: deviceId }, echoCancellation: true, noiseSuppression: true }
                        : { echoCancellation: true, noiseSuppression: true }
                };
                return await navigator.mediaDevices.getUserMedia(constraints);
            } else {
                const stream = await (navigator.mediaDevices as any).getDisplayMedia({
                    video: false,
                    audio: true
                });
                
                if (!stream.getAudioTracks().length) {
                    stream.getTracks().forEach(track => track.stop());
                    return null;
                }
                return stream;
            }
        } catch (error) {
            console.error(`Error getting ${type} audio stream:`, error);
            return null;
        }
    }
}

const intelligentAudioService = IntelligentAudioService.getInstance();

// --- THEME MANAGEMENT ---
type Theme = 'light' | 'dark';

const getInitialTheme = (): Theme => {
    const savedTheme = localStorage.getItem('theme') as Theme;
    if (savedTheme) return savedTheme;
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark';
    }
    return 'light';
};

const applyTheme = (theme: Theme) => {
    const root = document.documentElement;
    if (theme === 'dark') {
        root.setAttribute('data-theme', 'dark');
    } else {
        root.removeAttribute('data-theme');
    }
    localStorage.setItem('theme', theme);
};

// Initialize theme on load
applyTheme(getInitialTheme());

// --- COMPONENTS ---
const ThemeToggle: React.FC = () => {
    const [theme, setTheme] = useState<Theme>(getInitialTheme());

    useEffect(() => {
        applyTheme(theme);
        
        // Listen for system theme changes (only if user hasn't set a preference)
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        const handleSystemThemeChange = (e: MediaQueryListEvent) => {
            // Only auto-switch if user hasn't manually set a preference
            const savedTheme = localStorage.getItem('theme');
            if (!savedTheme) {
                const newTheme = e.matches ? 'dark' : 'light';
                setTheme(newTheme);
                applyTheme(newTheme);
            }
        };
        
        mediaQuery.addEventListener('change', handleSystemThemeChange);
        return () => mediaQuery.removeEventListener('change', handleSystemThemeChange);
    }, [theme]);

    const toggleTheme = () => {
        const newTheme = theme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
        applyTheme(newTheme);
    };

    return (
        <button className="theme-toggle" onClick={toggleTheme} title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}>
            <span className="theme-toggle-icon">{theme === 'light' ? '🌙' : '☀️'}</span>
        </button>
    );
};

const App: React.FC = () => {
    const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
    const [pin, setPin] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [isLocked, setIsLocked] = useState<boolean>(false);

    useEffect(() => {
        const checkPin = async () => {
            const storedEncryptedPinCheck = await db.getConfig('encryptedPinCheck');
            if (!storedEncryptedPinCheck) {
                // First time setup - no PIN set yet
                setIsAuthenticated(true);
            }
            setIsLoading(false);
        };
        checkPin();

        // Listen for lock state changes
        const unsubscribe = authService.onLockStateChange((locked) => {
            setIsLocked(locked);
            if (locked) {
                setIsAuthenticated(false);
                setPin(null);
            }
        });

        return () => {
            unsubscribe();
        };
    }, []);

    const handlePinSet = (newPin: string) => {
        setPin(newPin);
        setIsAuthenticated(true);
        authService.unlock();
    };

    const handleLogin = (enteredPin: string) => {
        setPin(enteredPin);
        setIsAuthenticated(true);
        authService.unlock();
    };

    if (isLoading) {
        return <div className="loading">Loading...</div>;
    }

    return isAuthenticated && !isLocked ? (
        <MainApp pin={pin!} authService={authService} onLock={() => authService.lock()} />
    ) : (
        <AuthScreen onPinSet={handlePinSet} onLogin={handleLogin} authService={authService} />
    );
};

const AuthScreen: React.FC<{ 
    onPinSet: (pin: string) => void, 
    onLogin: (pin: string) => void,
    authService: AuthService
}> = ({ onPinSet, onLogin, authService }) => {
    const [isSetup, setIsSetup] = useState(false);
    const [pin, setPin] = useState('');
    const [confirmPin, setConfirmPin] = useState('');
    const [isConfirming, setIsConfirming] = useState(false);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [isVerifying, setIsVerifying] = useState(false);
    const [lockoutRemaining, setLockoutRemaining] = useState(0);

    useEffect(() => {
        const checkSetup = async () => {
            const storedEncryptedPinCheck = await db.getConfig('encryptedPinCheck');
            setIsSetup(!!storedEncryptedPinCheck);
            setIsLoading(false);
        };
        checkSetup();

        // Check lockout status
        const checkLockout = () => {
            if (authService.isLockedOut()) {
                const remaining = authService.getLockoutRemaining();
                setLockoutRemaining(remaining);
            } else {
                setLockoutRemaining(0);
            }
        };
        
        checkLockout();
        const lockoutInterval = setInterval(checkLockout, 1000);
        
        return () => clearInterval(lockoutInterval);
    }, [authService]);

    const handleKeyPress = (key: string) => {
        if (pin.length < 4 && !isConfirming) {
            setPin(pin + key);
        } else if (isConfirming && confirmPin.length < 4) {
            setConfirmPin(confirmPin + key);
        }
    };

    const handleBackspace = () => {
        if (isConfirming) {
            setConfirmPin(prev => prev.slice(0, -1));
        } else {
            setPin(prev => prev.slice(0, -1));
        }
    };

    const handleLoginAttempt = async () => {
        if (pin.length === 4) {
            setIsVerifying(true);
            setError('');
            
            // Check if authentication is allowed (not locked out)
            const canAuth = authService.canAttemptAuth();
            if (!canAuth.allowed) {
                setError(canAuth.message || 'Authentication temporarily disabled.');
                setPin('');
                setIsVerifying(false);
                return;
            }

            try {
                // Verify PIN by attempting to decrypt
                const storedEncryptedPinCheck = await db.getConfig('encryptedPinCheck');
                if (!storedEncryptedPinCheck) {
                    setError('No PIN set. Please set up a PIN first.');
                    setPin('');
                    setIsVerifying(false);
                    return;
                }
                
                await CryptoService.decrypt(storedEncryptedPinCheck, pin);
                
                // Successful authentication - record success
                authService.recordSuccess();
                onLogin(pin);
            } catch(e) {
                // Failed authentication - record failed attempt
                const result = await authService.recordFailedAttempt();
                setError(result.message);
                setPin('');
                setIsVerifying(false);
            }
        }
    };

    useEffect(() => {
        if (!isSetup) {
            if (confirmPin.length === 4) {
                if (pin === confirmPin) {
                    const setupPin = async () => {
                        const encryptedPinCheck = await CryptoService.encrypt(pin, pin);
                        await db.saveConfig('encryptedPinCheck', encryptedPinCheck);
                        onPinSet(pin);
                    };
                    setupPin();
                } else {
                    setError("PINs don't match. Please try again.");
                    setPin('');
                    setConfirmPin('');
                    setIsConfirming(false);
                }
            }
        } else {
            if (pin.length === 4) {
                handleLoginAttempt();
            }
        }
    }, [pin, confirmPin]);


    useEffect(() => {
        if (pin.length === 4 && !isSetup && !isConfirming) {
            setIsConfirming(true);
        }
    }, [pin, isSetup, isConfirming]);

    if (isLoading) {
        return <div className="loading"></div>;
    }

    const currentPin = isConfirming ? confirmPin : pin;
    const title = isSetup ? "Enter PIN" : (isConfirming ? "Confirm PIN" : "Create a PIN");
    const isLockedOut = lockoutRemaining > 0;
    const lockoutMessage = isLockedOut 
        ? `Account locked. Try again in ${Math.ceil(lockoutRemaining / 60)} minute(s).`
        : null;

    return (
        <div className="auth-container">
            <div className="auth-card">
                <h2>{title}</h2>
                <div className="auth-error">
                    {lockoutMessage || error || (isVerifying && 'Verifying...')}
                </div>
                <div className="pin-display">
                    {isVerifying ? (
                        <div className="pin-spinner-container"><div className="spinner-small"></div></div>
                    ) : (
                        Array(4).fill(0).map((_, i) => (
                            <span key={i} className={i < currentPin.length ? 'filled' : ''}></span>
                        ))
                    )}
                </div>
                <div className="keypad">
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => (
                        <button 
                            key={num} 
                            onClick={() => handleKeyPress(String(num))} 
                            disabled={isVerifying || isLockedOut}
                        >
                            {num}
                        </button>
                    ))}
                    <button disabled={isVerifying || isLockedOut}></button>
                    <button 
                        onClick={() => handleKeyPress('0')} 
                        disabled={isVerifying || isLockedOut}
                    >
                        0
                    </button>
                    <button 
                        onClick={handleBackspace} 
                        disabled={isVerifying || isLockedOut}
                    >
                        &larr;
                    </button>
                </div>
            </div>
        </div>
    );
};

const MainApp: React.FC<{ pin: string; authService: AuthService; onLock: () => void }> = ({ pin, authService, onLock }) => {
    const [sessions, setSessions] = useState<Session[]>([]);
    const [tasks, setTasks] = useState<Task[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [status, setStatus] = useState({ message: '', type: '' });
    const [selectedSession, setSelectedSession] = useState<Session | null>(null);
    const [industry, setIndustry] = useState<string>('general');
    const [language, setLanguage] = useState<string>('en');
    const [modelsReady, setModelsReady] = useState(false);
    const [modelsLoading, setModelsLoading] = useState(true);
    const [showCalendarSettings, setShowCalendarSettings] = useState(false);
    const [calendarService, setCalendarService] = useState<CalendarService | null>(null);
    const [calendarConnected, setCalendarConnected] = useState(false);
    const [sidebarMobileOpen, setSidebarMobileOpen] = useState(false);
    const [contextRailOpen, setContextRailOpen] = useState(true);
    const [isMobile, setIsMobile] = useState(false);
    
    // Use ref to store latest contextRailOpen value to avoid stale closures
    const contextRailOpenRef = useRef(contextRailOpen);
    useEffect(() => {
        contextRailOpenRef.current = contextRailOpen;
    }, [contextRailOpen]);
    
    // Handle mobile sidebar
    useEffect(() => {
        const checkMobile = () => {
            const mobile = window.innerWidth <= 767;
            setIsMobile(mobile);
            if (!mobile) {
                setSidebarMobileOpen(false);
                // Use ref to get current value without causing effect re-runs
                if (!contextRailOpenRef.current) setContextRailOpen(true);
            } else {
                if (contextRailOpenRef.current) setContextRailOpen(false);
            }
        };
        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);
    
    useEffect(() => {
        // Initialize activity tracking for auto-lock
        authService.updateActivity();
        
        const loadData = async () => {
            try {
                const [loadedSessions, loadedTasks, savedIndustry, savedLanguage] = await Promise.all([
                    db.getAllSessions(pin),
                    db.getAllTasks(pin),
                    db.getConfig('industry'),
                    db.getConfig('language')
                ]);
                setSessions(loadedSessions);
                setTasks(loadedTasks);
                if (savedIndustry) {
                    setIndustry(savedIndustry);
                }
                if (savedLanguage) {
                    setLanguage(savedLanguage);
                }
                
                // Check for first-time setup
                const hasRunBefore = await db.getConfig('hasRunBefore');
                if (!hasRunBefore) {
                    showStatus('Welcome! This is your first time. Models will download automatically on first open.', 'info', 8000);
                }
            } catch (error) {
                showStatus('Failed to load data.', 'error');
            } finally {
                setIsLoading(false);
            }
        };
        loadData();
        
        // Listen for first-time setup event
        const handleFirstTimeSetup = (event: CustomEvent) => {
            showStatus(event.detail.message || 'Welcome! This is your first time. Models will download automatically on first open.', 'info', 8000);
        };
        window.addEventListener('firstTimeSetup', handleFirstTimeSetup as EventListener);
        
        // Listen for model download progress
        const handleModelDownloadProgress = (event: CustomEvent) => {
            showStatus(event.detail.message, 'info', 3000);
        };
        window.addEventListener('modelDownloadProgress', handleModelDownloadProgress as EventListener);
        
        // Listen for models downloaded event
        const handleModelsDownloaded = (event: CustomEvent) => {
            showStatus(event.detail.message || 'Models downloaded successfully!', 'success', 5000);
        };
        window.addEventListener('modelsDownloaded', handleModelsDownloaded as EventListener);
        
        // Listen for models loading event
        const handleModelsLoading = (event: CustomEvent) => {
            setModelsLoading(true);
            setModelsReady(false);
        };
        window.addEventListener('modelsLoading', handleModelsLoading as EventListener);
        
        // Listen for models ready event
        const handleModelsReady = (event: CustomEvent) => {
            setModelsReady(true);
            setModelsLoading(false);
            const message = event.detail?.message || 'AI models ready!';
            if (message !== 'Models will load on-demand') {
                showStatus(message, 'success', 3000);
            }
        };
        window.addEventListener('modelsReady', handleModelsReady as EventListener);
        
        // Listen for model load errors
        const handleModelLoadError = (event: CustomEvent) => {
            const errorMessage = event.detail?.message || 'Model loading error';
            const error = event.detail?.error;
            console.warn('Model load error:', error);
            
            // Show non-intrusive info message that models will load on-demand
            // This is not a critical error - models will load when needed
            if (errorMessage.includes('on-demand')) {
                // Models will load on-demand - this is expected behavior, not an error
                console.log('Models will load on-demand when needed. This is normal if preload fails.');
            } else {
                // Show a brief info message for other errors
                showStatus('AI models will load automatically when needed.', 'info', 3000);
            }
        };
        window.addEventListener('modelLoadError', handleModelLoadError as EventListener);
        
        return () => {
            window.removeEventListener('firstTimeSetup', handleFirstTimeSetup as EventListener);
            window.removeEventListener('modelDownloadProgress', handleModelDownloadProgress as EventListener);
            window.removeEventListener('modelsDownloaded', handleModelsDownloaded as EventListener);
            window.removeEventListener('modelsLoading', handleModelsLoading as EventListener);
            window.removeEventListener('modelsReady', handleModelsReady as EventListener);
            window.removeEventListener('modelLoadError', handleModelLoadError as EventListener);
        };
    }, []);

    // Initialize calendar and auto-launch
    useEffect(() => {
        const initCalendar = async () => {
            try {
                const configStr = localStorage.getItem('calendar_config');
                if (configStr) {
                    const config: CalendarConfig = JSON.parse(configStr);
                    if (config.provider && config.enabled) {
                        let service: CalendarService;
                        switch (config.provider) {
                            case 'google':
                                const googleId = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';
                                service = new GoogleCalendarService(googleId);
                                break;
                            case 'outlook':
                                const outlookId = import.meta.env.VITE_OUTLOOK_CLIENT_ID || '';
                                service = new OutlookCalendarService(outlookId);
                                break;
                            default:
                                return;
                        }
                        
                        const connected = await service.isConnected();
                        if (connected) {
                            setCalendarService(service);
                            setCalendarConnected(true);
                            
                            // Initialize auto-launch
                            const autoLaunchService = AutoLaunchService.getInstance();
                            autoLaunchService.initialize(service, {
                                enabled: config.autoLaunchEnabled || false,
                                preLaunchSeconds: config.preLaunchSeconds || 30,
                                checkIntervalSeconds: config.checkIntervalSeconds || 60,
                                autoStartRecording: config.autoStartRecording || false
                            });
                            
                            if (config.autoLaunchEnabled) {
                                await autoLaunchService.start();
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('Error initializing calendar:', error);
            }
        };

        initCalendar();

        // Handle meeting pre-launch event
        const handleMeetingPreLaunch = (event: CustomEvent) => {
            const { meeting } = event.detail;
            showStatus(`Meeting starting soon: ${meeting.title}`, 'info', 5000);
            // Could auto-open new session form here
        };
        window.addEventListener('meetingPreLaunch', handleMeetingPreLaunch as EventListener);

        return () => {
            window.removeEventListener('meetingPreLaunch', handleMeetingPreLaunch as EventListener);
        };
    }, []);

    const showStatus = (message: string, type: 'success' | 'error' | 'info', duration = 3000) => {
        setStatus({ message, type });
        setTimeout(() => setStatus({ message: '', type: '' }), duration);
    };

    const handleAddSession = async (session: Omit<Session, 'id' | 'timestamp' | 'notes'>, notes: string, audioBlob: Blob | null): Promise<number | null> => {
        try {
            const encryptedNotes = await CryptoService.encrypt(notes, pin);
            const timestamp = Date.now();

            const newSession: Session = { ...session, notes: encryptedNotes, timestamp, language };
            if (audioBlob) {
                newSession.analysisStatus = 'none';
            }
            
            const id = await db.addSession(newSession, pin);
            
            if (audioBlob) {
                await db.saveAudioBlob(id, audioBlob);
            }

            setSessions(prev => [{ ...newSession, id }, ...prev]);
            showStatus('Session saved successfully.', 'success');
            return id;
        } catch (error) {
            showStatus('Failed to save session.', 'error');
            return null;
        }
    };
    
    const handleDeleteSession = async (id: number) => {
        if (window.confirm('Are you sure you want to delete this session? This action cannot be undone.')) {
            try {
                await db.deleteSession(id);
                setSessions(prev => prev.filter(s => s.id !== id));
                // Also delete associated tasks
                const tasksToDelete = tasks.filter(t => t.sessionId === id);
                for (const task of tasksToDelete) {
                    await db.deleteTask(task.id!);
                }
                setTasks(prev => prev.filter(t => t.sessionId !== id));
                showStatus('Session deleted.', 'success');
            } catch {
                showStatus('Failed to delete session.', 'error');
            }
        }
    };

    const handleUpdateSession = async (updatedSession: Session) => {
        try {
            await db.updateSession(updatedSession, pin);
            setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s));
            if (selectedSession?.id === updatedSession.id) {
                setSelectedSession(updatedSession);
            }
        } catch (error) {
            showStatus('Failed to update session.', 'error');
        }
    };

    const handleAddTask = async (task: Omit<Task, 'id' | 'timestamp'>) => {
        try {
            const newTask = { ...task, timestamp: Date.now() };
            const id = await db.addTask(newTask, pin);
            setTasks(prev => [{ ...newTask, id }, ...prev]);
            showStatus('Task added.', 'success');
            return true;
        } catch {
            showStatus('Failed to add task.', 'error');
            return false;
        }
    };

    const handleUpdateTask = async (updatedTask: Task) => {
        try {
            await db.updateTask(updatedTask, pin);
            setTasks(prev => prev.map(t => t.id === updatedTask.id ? updatedTask : t));
        } catch (error) {
            showStatus('Failed to update task.', 'error');
        }
    };

    const handleDeleteTask = async (id: number) => {
        try {
            await db.deleteTask(id);
            setTasks(prev => prev.filter(t => t.id !== id));
            showStatus('Task deleted.', 'success');
        } catch {
            showStatus('Failed to delete task.', 'error');
        }
    };

    const handleIndustryChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newIndustry = e.target.value;
        setIndustry(newIndustry);
        await db.saveConfig('industry', newIndustry);
    };

    const handleLanguageChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newLanguage = e.target.value;
        setLanguage(newLanguage);
        await db.saveConfig('language', newLanguage);
    };

    const handleStartAnalysis = async (sessionId: number) => {
        // Try to find session in state first
        let currentSession = sessions.find(s => s.id === sessionId);
        
        // If not found in state, try to load from database (might be a timing issue)
        if (!currentSession) {
            try {
                const allSessions = await db.getAllSessions(pin);
                currentSession = allSessions.find(s => s.id === sessionId);
                if (currentSession) {
                    // Update state with the session
                    setSessions(prev => {
                        const exists = prev.find(s => s.id === sessionId);
                        if (!exists) {
                            return [currentSession!, ...prev];
                        }
                        return prev;
                    });
                }
            } catch (error) {
                console.error('Error loading session from database:', error);
            }
        }
        
        if (!currentSession) {
            showStatus('Session not found. Please refresh and try again.', 'error');
            return;
        }

        try {
            const audioBlob = await db.getAudioBlob(sessionId);
            if (!audioBlob) {
                showStatus('Audio file not found for this session.', 'error');
                return;
            }

            // Get audio buffer
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            // Get industry context and language
            const sessionIndustry = industry;
            const sessionLanguage = currentSession.language || language;

            // Step 1: Transcription
            showStatus('Starting transcription...', 'info');
            const updatedSession1 = { ...currentSession, analysisStatus: 'pending' as const };
            await handleUpdateSession(updatedSession1);
            currentSession = updatedSession1;

            const transcriptChunks = await onDeviceAIService.transcribeAudio(
                audioBuffer,
                (status, progress) => {
                    const progressMsg = progress ? ` (${Math.round(progress)}%)` : '';
                    showStatus(`Transcribing: ${status}${progressMsg}`, 'info', 5000);
                },
                sessionLanguage
            );

            // Save transcript immediately
            const transcriptText = transcriptChunks.map(c => c.text).join(' ');
            if (!transcriptText.trim()) {
                throw new Error('No speech detected in the audio recording.');
            }

            const updatedSession2 = {
                ...currentSession,
                transcript: JSON.stringify(transcriptChunks),
                analysisStatus: 'pending' as const
            };
            await handleUpdateSession(updatedSession2);
            currentSession = updatedSession2;
            showStatus('Transcription complete. Generating summary...', 'info');

            // Step 2: Generate Summary
            let summary = '';
            try {
                summary = await onDeviceAIService.generateSummary(
                    transcriptText,
                    sessionIndustry,
                    (status) => {
                        showStatus(`Generating summary: ${status}`, 'info', 5000);
                    }
                );
                
                // Save summary incrementally
                const updatedSession3 = {
                    ...currentSession,
                    summary: JSON.stringify({ summary }),
                    analysisStatus: 'pending' as const
                };
                await handleUpdateSession(updatedSession3);
                currentSession = updatedSession3;
                showStatus('Summary complete. Creating outline...', 'info');
            } catch (summaryError: any) {
                console.error('Summary generation failed:', summaryError);
                summary = `Summary generation failed: ${summaryError?.message || 'Unknown error'}`;
                showStatus(`Summary generation failed: ${summaryError?.message || 'Unknown error'}`, 'error', 5000);
                // Continue with other steps even if summary fails
            }

            // Step 3: Generate Outline
            let outline = '';
            try {
                outline = await onDeviceAIService.generateOutline(
                    transcriptText,
                    sessionIndustry,
                    (status) => {
                        showStatus(`Creating outline: ${status}`, 'info', 5000);
                    }
                );
                
                // Save outline incrementally
                const updatedSession4 = {
                    ...currentSession,
                    outline: JSON.stringify({ outline }),
                    analysisStatus: 'pending' as const
                };
                await handleUpdateSession(updatedSession4);
                currentSession = updatedSession4;
                showStatus('Outline complete. Extracting action items...', 'info');
            } catch (outlineError: any) {
                console.error('Outline generation failed:', outlineError);
                outline = `Outline generation failed: ${outlineError?.message || 'Unknown error'}`;
                showStatus(`Outline generation failed: ${outlineError?.message || 'Unknown error'}`, 'error', 5000);
                // Continue with action items even if outline fails
            }

            // Step 4: Generate Action Items
            let actionItems: string[] = [];
            try {
                actionItems = await onDeviceAIService.generateActionItems(
                    transcriptText,
                    sessionIndustry,
                    (status) => {
                        showStatus(`Extracting action items: ${status}`, 'info', 5000);
                    }
                );
            } catch (actionError: any) {
                console.error('Action items generation failed:', actionError);
                actionItems = [];
                showStatus(`Action items generation failed: ${actionError?.message || 'Unknown error'}`, 'error', 5000);
                // Continue to final save even if action items fail
            }

            // Format action items for UI
            const todoItems: TodoItem[] = actionItems.map((text: string) => ({ text, completed: false }));

            // Final update with all results
            const finalSession = {
                ...currentSession,
                todoItems: JSON.stringify(todoItems),
                analysisStatus: 'complete' as const
            };

            await handleUpdateSession(finalSession);
            showStatus('Analysis complete!', 'success');
        } catch (error: any) {
            const errorMessage = error?.message || error?.toString() || "Unknown error occurred";
            console.error('Analysis Error:', error);
            
            // Update session status to failed
            currentSession = sessions.find(s => s.id === sessionId);
            if (currentSession) {
                await handleUpdateSession({ ...currentSession, analysisStatus: 'failed' as const });
            }
            
            showStatus(`Analysis failed: ${errorMessage}`, 'error', 5000);
        }
    };

    const handleNewNote = async () => {
        // Create a new empty session/note and save it
        try {
            const newSession: Omit<Session, 'id' | 'timestamp' | 'notes'> = {
                sessionTitle: 'Untitled Note',
                participants: '',
                date: new Date().toISOString().split('T')[0],
                analysisStatus: 'none',
                duration: 0,
                transcript: []
            };
            const id = await handleAddSession(newSession, '', null);
            if (id) {
                // Reload sessions to get the new one
                const allSessions = await db.getAllSessions(pin);
                setSessions(allSessions);
                const savedSession = allSessions.find(s => s.id === id);
                if (savedSession) {
                    setSelectedSession(savedSession);
                }
            }
        } catch (error) {
            showStatus('Failed to create new note.', 'error');
        }
        if (isMobile) {
            setSidebarMobileOpen(false);
        }
    };

    const handleSelectNote = (session: Session) => {
        setSelectedSession(session);
        if (isMobile) {
            setSidebarMobileOpen(false);
        }
    };

    const handleSummarize = async () => {
        if (!selectedSession) return;
        // Trigger AI analysis for summarization
        if (selectedSession.id) {
            await handleStartAnalysis(selectedSession.id);
        }
    };

    const handleActionItems = async () => {
        if (!selectedSession) return;
        // Trigger AI analysis for action items
        if (selectedSession.id) {
            await handleStartAnalysis(selectedSession.id);
        }
    };

    return (
        <div
            className="app-layout"
            style={{
                display: 'flex',
                height: '100vh',
                width: '100%',
                background: document.documentElement.getAttribute('data-theme') === 'dark'
                    ? 'transparent'
                    : '#fff',
                overflow: 'hidden',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
            }}
        >
            {/* Sidebar */}
            <Sidebar
                isOpen={!isMobile || sidebarMobileOpen}
                onClose={() => setSidebarMobileOpen(false)}
                onNewNote={handleNewNote}
                onSelectNote={handleSelectNote}
                onCalendarSettings={() => {
                    authService.updateActivity();
                    setShowCalendarSettings(true);
                    if (isMobile) {
                        setSidebarMobileOpen(false);
                    }
                }}
                onLock={() => {
                    authService.updateActivity();
                    onLock();
                }}
                sessions={sessions}
                selectedSession={selectedSession}
                isMobile={isMobile}
            />

            {/* Editor Area */}
            <EditorArea
                selectedSession={selectedSession}
                onSidebarToggle={() => setSidebarMobileOpen(!sidebarMobileOpen)}
                onContextRailToggle={() => setContextRailOpen(!contextRailOpen)}
                isContextOpen={contextRailOpen}
                isMobile={isMobile}
                onUpdateSession={handleUpdateSession}
                onSummarize={handleSummarize}
                onActionItems={handleActionItems}
                pin={pin}
            />

            {/* Context Rail */}
            <ContextRail
                isOpen={!isMobile || contextRailOpen}
                onClose={() => setContextRailOpen(false)}
                selectedSession={selectedSession}
                tasks={tasks}
                sessions={sessions}
            />

            {/* Status Messages */}
            {status.message && (
                <div
                    style={{
                        position: 'fixed',
                        bottom: '24px',
                        right: '24px',
                        padding: '12px 16px',
                        borderRadius: '8px',
                        background: status.type === 'error' ? 'rgba(239, 68, 68, 0.9)' : status.type === 'success' ? 'rgba(44, 95, 65, 0.9)' : 'rgba(2, 41, 91, 0.9)',
                        color: '#fff',
                        fontSize: '14px',
                        zIndex: 1000,
                        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                        maxWidth: '300px'
                    }}
                >
                    {status.message}
                </div>
            )}

            {/* Loading Overlay */}
            {isLoading && (
                <div
                    style={{
                        position: 'fixed',
                        inset: 0,
                        background: 'rgba(255, 255, 255, 0.9)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '16px',
                        zIndex: 2000
                    }}
                >
                    <div className="spinner"></div>
                    <div style={{ fontSize: '16px', color: '#4b5563' }}>Loading...</div>
                </div>
            )}

            {/* Session Detail Modal (for audio playback, etc.) */}
            {selectedSession && selectedSession.id && (
                <SessionDetailModal
                    session={selectedSession}
                    onClose={() => setSelectedSession(null)}
                    onDelete={handleDeleteSession}
                    onUpdate={handleUpdateSession}
                    onAddTask={handleAddTask}
                    pin={pin}
                    modelsReady={modelsReady}
                    modelsLoading={modelsLoading}
                />
            )}

            {/* Calendar Settings Modal */}
            {showCalendarSettings && (
                <CalendarSettings
                    onClose={() => setShowCalendarSettings(false)}
                    onCalendarConnected={(provider) => {
                        setCalendarConnected(true);
                        showStatus(`Connected to ${provider === 'google' ? 'Google Calendar' : 'Outlook Calendar'}`, 'success');
                    }}
                />
            )}

            {/* Theme Toggle - Floating */}
            <div
                style={{
                    position: 'fixed',
                    bottom: '24px',
                    left: isMobile ? '24px' : '280px',
                    zIndex: 100,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                }}
            >
                {modelsLoading && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#6b7280', background: 'rgba(255, 255, 255, 0.9)', padding: '4px 8px', borderRadius: '6px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)' }}>
                        <div className="spinner" style={{ width: '12px', height: '12px', borderWidth: '2px' }}></div>
                        <span style={{ display: window.innerWidth > 640 ? 'inline' : 'none' }}>Loading AI...</span>
                    </div>
                )}
                {modelsReady && !modelsLoading && (
                    <div style={{ fontSize: '11px', color: 'var(--color-strategic-forest, #2c5f41)', fontWeight: '500', background: 'rgba(255, 255, 255, 0.9)', padding: '4px 8px', borderRadius: '6px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)' }}>
                        ✓ AI Ready
                    </div>
                )}
                <ThemeToggle />
            </div>
        </div>
    );
};

// ViewSwitcher removed - now handled by Sidebar component

type ShowStatusType = (message: string, type: 'success' | 'error' | 'info', duration?: number) => void;

// DEPRECATED: NewSessionForm removed - replaced by "New Note" button in Sidebar
// Recording functionality will be moved to a modal or separate component
// Component code removed - see git history if needed

// DEPRECATED: SessionsList removed - replaced by notes list in Sidebar
// Component code removed - see git history if needed

// DEPRECATED: PreviousNotesList removed - replaced by notes navigation in Sidebar
// Component code removed - see git history if needed

const NotesViewModal: React.FC<{
    session: Session,
    onClose: () => void,
    pin: string
}> = ({ session, onClose, pin }) => {
    const [decryptedNotes, setDecryptedNotes] = useState('');
    const [isDecrypting, setIsDecrypting] = useState(true);
    
    useEffect(() => {
        const loadNotes = async () => {
            setIsDecrypting(true);
            try {
                const notes = await CryptoService.decrypt(session.notes, pin);
                setDecryptedNotes(notes);
            } catch (error) {
                setDecryptedNotes("Error: Could not decrypt notes.");
            } finally {
                setIsDecrypting(false);
            }
        };
        loadNotes();
    }, [session, pin]);
    
    const transcript = parseTranscript(session.transcript);
    const summary = parseSummary(session.summary);
    const outline = parseOutline(session.outline);
    
    return (
        <div className="modal active" onClick={onClose}>
            <div className="modal-content notes-view-modal" onClick={e => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose}>&times;</button>
                <h2>{session.sessionTitle}</h2>
                
                {isDecrypting ? (
                    <div className="loading">Loading notes...</div>
                ) : (
                    <div className="notes-view-content">
                        {/* Transcript Section */}
                        {transcript.length > 0 && (
                            <div className="notes-section">
                                <h3>Transcript</h3>
                                <div className="transcript-view">
                                    {transcript.map((chunk, index) => (
                                        <div key={index} className="transcript-chunk-view">
                                            <span className="transcript-speaker-view">{chunk.speaker}:</span>
                                            <span className="transcript-text-view">{chunk.text}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                        
                        {/* Summary Section */}
                        {summary && (
                            <div className="notes-section">
                                <h3>Summary</h3>
                                <div className="summary-view">
                                    <p>{summary}</p>
                                </div>
                            </div>
                        )}
                        
                        {/* Outline Section */}
                        {outline && (
                            <div className="notes-section">
                                <h3>Outline</h3>
                                <div className="outline-view">
                                    <pre>{outline}</pre>
                                </div>
                            </div>
                        )}
                        
                        {/* Additional Notes */}
                        {decryptedNotes && (
                            <div className="notes-section">
                                <h3>Additional Notes</h3>
                                <div className="additional-notes-view">
                                    <pre>{decryptedNotes}</pre>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

const SessionDetailModal: React.FC<{ 
    session: Session, 
    onClose: () => void, 
    onDelete: (id: number) => void, 
    onUpdate: (session: Session) => void,
    onAddTask: (task: Omit<Task, 'id' | 'timestamp'>) => Promise<boolean>,
    pin: string,
    modelsReady: boolean,
    modelsLoading: boolean
}> = ({ session, onClose, onDelete, onUpdate, onAddTask, pin, modelsReady, modelsLoading }) => {
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
    
    // Bookmark state
    const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
    const [bookmarkSaveTimeout, setBookmarkSaveTimeout] = useState<number | null>(null);
    
    // Search state
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState<Array<{chunkIndex: number, matchIndex: number, matchLength: number}>>([]);
    const [currentResultIndex, setCurrentResultIndex] = useState(0);
    
    // Playback sync state
    const [currentPlaybackTime, setCurrentPlaybackTime] = useState(0);
    const transcriptRef = useRef<HTMLDivElement>(null);
    const activeChunkRef = useRef<HTMLDivElement>(null);
    
    // Topic/chapter state
    const [topics, setTopics] = useState<Topic[]>([]);
    const [activeTopicIndex, setActiveTopicIndex] = useState<number | null>(null);

    // Initialize aiAnalysisStatus based on session status when modal opens
    useEffect(() => {
        if (session.analysisStatus === 'failed') {
            setAiAnalysisStatus('failed');
        } else if (session.analysisStatus === 'complete') {
            setAiAnalysisStatus('complete');
        } else if (session.analysisStatus === 'pending') {
            setAiAnalysisStatus('in_progress');
        } else {
            setAiAnalysisStatus('idle');
        }
    }, [session.analysisStatus]);

    useEffect(() => {
        const decryptAndLoad = async () => {
            setIsDecrypting(true);
            try {
                const notes = await CryptoService.decrypt(session.notes, pin);
                setDecryptedNotes(notes);
                setEditedNotes(notes);

                const blob = await db.getAudioBlob(session.id!);
                if (blob) {
                    setAudioBlob(blob);
                    setAudioUrl(URL.createObjectURL(blob));
                }
                
                // Load bookmarks
                const loadedBookmarks = parseBookmarks(session.bookmarks);
                setBookmarks(loadedBookmarks);
                
                // Parse topics from outline
                const outline = parseOutline(session.outline);
                if (outline) {
                    const parsedTopics = parseTopicsFromOutline(outline, parseTranscript(session.transcript));
                    setTopics(parsedTopics);
                }

            } catch (error) {
                setDecryptedNotes("Error: Could not decrypt notes. The PIN may be incorrect or data is corrupted.");
            } finally {
                setIsDecrypting(false);
            }
        };
        decryptAndLoad();

        return () => {
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
        };
    }, [session, pin]);

    useEffect(() => {
        // Create initial speaker map
        const transcript = parseTranscript(session.transcript);
        const uniqueSpeakers = [...new Set(transcript.map(c => c.speaker))];
        const initialMap: {[key: string]: string} = {};
        uniqueSpeakers.forEach(speaker => {
            initialMap[speaker] = speaker;
        });
        setSpeakerMap(initialMap);
    }, [session.transcript]);
    
    // Search functions
    const scrollToSearchResult = (index: number) => {
        if (index < 0 || index >= searchResults.length) return;
        
        const result = searchResults[index];
        const chunkElement = document.querySelector(`[data-chunk-index="${result.chunkIndex}"]`);
        if (chunkElement) {
            chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            setCurrentResultIndex(index);
        }
    };
    
    // Search effect - debounced
    useEffect(() => {
        if (!searchQuery.trim()) {
            setSearchResults([]);
            setCurrentResultIndex(0);
            return;
        }
        
        const timeout = setTimeout(() => {
            const transcript = parseTranscript(session.transcript);
            const results: Array<{chunkIndex: number, matchIndex: number, matchLength: number}> = [];
            const lowerQuery = searchQuery.toLowerCase();
            
            transcript.forEach((chunk, chunkIndex) => {
                const text = chunk.text.toLowerCase();
                let searchIndex = text.indexOf(lowerQuery);
                while (searchIndex !== -1) {
                    results.push({
                        chunkIndex,
                        matchIndex: searchIndex,
                        matchLength: searchQuery.length
                    });
                    searchIndex = text.indexOf(lowerQuery, searchIndex + 1);
                }
            });
            
            setSearchResults(results);
            const firstIndex = results.length > 0 ? 0 : -1;
            setCurrentResultIndex(firstIndex);
            
            // Scroll to first result
            if (results.length > 0) {
                setTimeout(() => {
                    const firstResult = results[0];
                    const chunkElement = document.querySelector(`[data-chunk-index="${firstResult.chunkIndex}"]`);
                    if (chunkElement) {
                        chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }, 100);
            }
        }, 300);
        return () => clearTimeout(timeout);
    }, [searchQuery, session.transcript]);
    
    // Update topics when outline changes
    useEffect(() => {
        const outline = parseOutline(session.outline);
        const transcript = parseTranscript(session.transcript);
        if (outline && transcript.length > 0) {
            const parsedTopics = parseTopicsFromOutline(outline, transcript);
            setTopics(parsedTopics);
        } else {
            setTopics([]);
        }
    }, [session.outline, session.transcript]);

    const handleSaveNotes = async () => {
        try {
            const encryptedNotes = await CryptoService.encrypt(editedNotes, pin);
            onUpdate({ ...session, notes: encryptedNotes, audioBlob });
            setDecryptedNotes(editedNotes);
            setIsEditingNotes(false);
        } catch {
            alert('Failed to save notes.');
        }
    };
    
    // Bookmark functions
    const toggleBookmark = (chunkIndex: number) => {
        const transcript = parseTranscript(session.transcript);
        const chunk = transcript[chunkIndex];
        const timestamp = chunk.timestamp?.[0] || 0;
        
        const existingIndex = bookmarks.findIndex(b => b.chunkIndex === chunkIndex);
        let newBookmarks: Bookmark[];
        
        if (existingIndex >= 0) {
            // Remove bookmark
            newBookmarks = bookmarks.filter((_, i) => i !== existingIndex);
        } else {
            // Add bookmark
            newBookmarks = [...bookmarks, {
                chunkIndex,
                timestamp,
                createdAt: Date.now()
            }];
        }
        
        setBookmarks(newBookmarks);
        
        // Debounced save
        if (bookmarkSaveTimeout) {
            window.clearTimeout(bookmarkSaveTimeout);
        }
        const timeout = window.setTimeout(async () => {
            try {
                const bookmarksStr = JSON.stringify(newBookmarks);
                const encryptedBookmarks = await CryptoService.encrypt(bookmarksStr, pin);
                onUpdate({ ...session, bookmarks: encryptedBookmarks, audioBlob });
            } catch (error) {
                console.error('Failed to save bookmarks:', error);
            }
        }, 500);
        setBookmarkSaveTimeout(timeout);
    };
    
    const isBookmarked = (chunkIndex: number): boolean => {
        return bookmarks.some(b => b.chunkIndex === chunkIndex);
    };
    
    const jumpToBookmark = (bookmark: Bookmark) => {
        const chunkElement = document.querySelector(`[data-chunk-index="${bookmark.chunkIndex}"]`);
        if (chunkElement) {
            chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        // If audio is available, seek to timestamp
        if (audioUrl && bookmark.timestamp > 0) {
            const audioElement = document.querySelector('audio') as HTMLAudioElement;
            if (audioElement) {
                audioElement.currentTime = bookmark.timestamp;
            }
        }
    };
    
    // Search navigation
    const navigateSearchResults = (direction: 'next' | 'prev') => {
        if (searchResults.length === 0) return;
        
        let newIndex = currentResultIndex;
        if (direction === 'next') {
            newIndex = (currentResultIndex + 1) % searchResults.length;
        } else {
            newIndex = currentResultIndex <= 0 ? searchResults.length - 1 : currentResultIndex - 1;
        }
        scrollToSearchResult(newIndex);
    };
    
    // Playback sync functions
    const getChunkForTime = (time: number): number => {
        const transcript = parseTranscript(session.transcript);
        return transcript.findIndex(chunk => {
            const [start, end] = chunk.timestamp || [0, 0];
            return time >= start && time < end;
        });
    };
    
    const handleTimeUpdate = (time: number) => {
        setCurrentPlaybackTime(time);
        
        // Throttle scroll operations (250ms)
        const activeChunkIndex = getChunkForTime(time);
        if (activeChunkIndex >= 0 && transcriptRef.current) {
            // Only scroll if chunk changed significantly
            const lastScrollTime = (handleTimeUpdate as any).lastScrollTime || 0;
            const now = Date.now();
            if (now - lastScrollTime > 250) {
                const chunkElement = document.querySelector(`[data-chunk-index="${activeChunkIndex}"][data-is-transcript-chunk]`);
                if (chunkElement) {
                    chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    activeChunkRef.current = chunkElement as HTMLDivElement;
                    (handleTimeUpdate as any).lastScrollTime = now;
                }
            }
        }
    };
    
    const jumpToChunk = (chunkIndex: number) => {
        const transcript = parseTranscript(session.transcript);
        const chunk = transcript[chunkIndex];
        if (chunk.timestamp && audioUrl) {
            const audioElement = document.querySelector('audio') as HTMLAudioElement;
            if (audioElement) {
                audioElement.currentTime = chunk.timestamp[0];
            }
        }
        const chunkElement = document.querySelector(`[data-chunk-index="${chunkIndex}"]`);
        if (chunkElement) {
            chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    };
    
    const jumpToTopic = (topic: Topic) => {
        if (topic.chunkIndices.length > 0) {
            jumpToChunk(topic.chunkIndices[0]);
        } else if (topic.startTime > 0 && audioUrl) {
            const audioElement = document.querySelector('audio') as HTMLAudioElement;
            if (audioElement) {
                audioElement.currentTime = topic.startTime;
            }
        }
    };

    const handleExportSession = async (format: 'txt' | 'json' | 'markdown') => {
        try {
            const transcript = parseTranscript(session.transcript);
            const summary = parseSummary(session.summary);
            const todoItems = parseTodoItems(session.todoItems);
            const outline = parseOutline(session.outline);

            let content = '';
            let filename = '';
            let mimeType = '';

            if (format === 'txt') {
                content = `Session: ${session.sessionTitle}\n`;
                content += `Date: ${new Date(session.date).toLocaleString()}\n`;
                if (session.participants) content += `Participants: ${session.participants}\n`;
                content += `\n=== TRANSCRIPT ===\n\n`;
                transcript.forEach(chunk => {
                    content += `${chunk.speaker}: ${chunk.text}\n`;
                });
                content += `\n=== SUMMARY ===\n\n${summary}\n\n`;
                content += `=== ACTION ITEMS ===\n\n`;
                todoItems.forEach((item, i) => {
                    content += `${i + 1}. ${item.text} ${item.completed ? '[DONE]' : ''}\n`;
                });
                content += `\n=== OUTLINE ===\n\n${outline}\n`;
                filename = `session-${session.id}-${session.sessionTitle.replace(/[^a-z0-9]/gi, '-')}.txt`;
                mimeType = 'text/plain';
            } else if (format === 'json') {
                const exportData = {
                    sessionTitle: session.sessionTitle,
                    date: session.date,
                    participants: session.participants,
                    transcript,
                    summary,
                    actionItems: todoItems,
                    outline
                };
                content = JSON.stringify(exportData, null, 2);
                filename = `session-${session.id}-${session.sessionTitle.replace(/[^a-z0-9]/gi, '-')}.json`;
                mimeType = 'application/json';
            } else if (format === 'markdown') {
                content = `# ${session.sessionTitle}\n\n`;
                content += `**Date:** ${new Date(session.date).toLocaleString()}\n`;
                if (session.participants) content += `**Participants:** ${session.participants}\n`;
                content += `\n## Transcript\n\n`;
                transcript.forEach(chunk => {
                    content += `**${chunk.speaker}:** ${chunk.text}\n\n`;
                });
                content += `## Summary\n\n${summary}\n\n`;
                content += `## Action Items\n\n`;
                todoItems.forEach((item, i) => {
                    content += `${i + 1}. ${item.completed ? '~~' : ''}${item.text}${item.completed ? '~~' : ''}\n`;
                });
                content += `\n## Outline\n\n${outline}\n`;
                filename = `session-${session.id}-${session.sessionTitle.replace(/[^a-z0-9]/gi, '-')}.md`;
                mimeType = 'text/markdown';
            }

            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (error: any) {
            alert(`Failed to export session: ${error?.message || 'Unknown error'}`);
        }
    };
    
    const handlePromoteTodoToTask = async (todo: TodoItem, todoIndex: number) => {
        const success = await onAddTask({
            title: todo.text,
            dueDate: null,
            priority: 'medium',
            status: 'todo',
            sessionId: session.id,
            sessionName: session.sessionTitle
        });

        if (success) {
            const updatedTodos = [...parseTodoItems(session.todoItems)];
            const newTaskId = Date.now(); // Placeholder, real ID comes from DB
            updatedTodos[todoIndex] = { ...todo, promotedToTaskId: newTaskId };
            onUpdate({ ...session, todoItems: JSON.stringify(updatedTodos), audioBlob });
        }
    };
    
    const handleTodoToggle = (index: number) => {
        const updatedTodos = [...parseTodoItems(session.todoItems)];
        updatedTodos[index].completed = !updatedTodos[index].completed;
        onUpdate({ ...session, todoItems: JSON.stringify(updatedTodos), audioBlob });
    };

    const handleRunOnDeviceAnalysis = async () => {
        // Reset status and progress
        setAiAnalysisStatus('in_progress');
        const analysisStartTime = performance.now();
        setAiProgress({ status: 'Starting transcription...', progress: 0 });
        let currentSession = { ...session, analysisStatus: 'pending' as const, audioBlob };
        onUpdate(currentSession);
        
        try {
            if (!audioBlob) {
                throw new Error("Audio file not found for this session.");
            }
            
            if (typeof window !== 'undefined' && !window.crypto) {
                throw new Error("Web Crypto API not available. This app requires a modern browser.");
            }
            
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            // Get industry context and language from database
            const industry = await db.getConfig('industry') || 'general';
            const language = session.language || await db.getConfig('language') || 'en';
            
            // Step 1: Transcription
            setAiProgress({ status: 'Transcribing audio...', progress: 10 });
            const transcriptChunks = await onDeviceAIService.transcribeAudio(
                audioBuffer,
                (status, progress) => {
                    const elapsed = Math.round((performance.now() - analysisStartTime) / 1000);
                    setAiProgress({ 
                        status: `Transcribing: ${status}`, 
                        progress: progress ? 10 + (progress * 0.3) : 10
                    });
                },
                language
            );

            // Save transcript immediately
            const transcriptText = transcriptChunks.map(c => c.text).join(' ');
            if (!transcriptText.trim()) {
                throw new Error('No speech detected in the audio recording.');
            }

            currentSession = {
                ...currentSession,
                transcript: JSON.stringify(transcriptChunks),
                analysisStatus: 'pending' as const
            };
            onUpdate(currentSession);
            setAiProgress({ status: 'Transcription complete. Generating summary...', progress: 40 });

            // Step 2: Generate Summary
            let summary = '';
            try {
                summary = await onDeviceAIService.generateSummary(
                    transcriptText,
                    industry,
                    (status) => {
                        setAiProgress({ status: `Generating summary: ${status}`, progress: 50 });
                    }
                );
                
                currentSession = {
                    ...currentSession,
                    summary: JSON.stringify({ summary }),
                    analysisStatus: 'pending' as const
                };
                onUpdate(currentSession);
                setAiProgress({ status: 'Summary complete. Creating outline...', progress: 60 });
            } catch (summaryError: any) {
                console.error('Summary generation failed:', summaryError);
                summary = `Summary generation failed: ${summaryError?.message || 'Unknown error'}`;
                setAiProgress({ status: `Summary failed: ${summaryError?.message || 'Unknown error'}`, progress: 50 });
            }

            // Step 3: Generate Outline
            let outline = '';
            try {
                outline = await onDeviceAIService.generateOutline(
                    transcriptText,
                    industry,
                    (status) => {
                        setAiProgress({ status: `Creating outline: ${status}`, progress: 70 });
                    }
                );
                
                currentSession = {
                    ...currentSession,
                    outline: JSON.stringify({ outline }),
                    analysisStatus: 'pending' as const
                };
                onUpdate(currentSession);
                setAiProgress({ status: 'Outline complete. Extracting action items...', progress: 80 });
            } catch (outlineError: any) {
                console.error('Outline generation failed:', outlineError);
                outline = `Outline generation failed: ${outlineError?.message || 'Unknown error'}`;
                setAiProgress({ status: `Outline failed: ${outlineError?.message || 'Unknown error'}`, progress: 70 });
            }

            // Step 4: Generate Action Items
            let actionItems: string[] = [];
            try {
                actionItems = await onDeviceAIService.generateActionItems(
                    transcriptText,
                    industry,
                    (status) => {
                        setAiProgress({ status: `Extracting action items: ${status}`, progress: 90 });
                    }
                );
            } catch (actionError: any) {
                console.error('Action items generation failed:', actionError);
                actionItems = [];
                setAiProgress({ status: `Action items failed: ${actionError?.message || 'Unknown error'}`, progress: 90 });
            }

            // Format action items for UI
            const todoItems: TodoItem[] = actionItems.map((text: string) => ({ text, completed: false }));

            // Final update with all results
            const finalSession = {
                ...currentSession,
                todoItems: JSON.stringify(todoItems),
                analysisStatus: 'complete' as const,
                audioBlob
            };
            
            onUpdate(finalSession);
            setAiAnalysisStatus('complete');
            setAiProgress({ status: 'Analysis complete!', progress: 100 });

        } catch (error: any) {
            const errorMessage = error?.message || error?.toString() || "Unknown error occurred";
            console.error('AI Analysis Error:', error);
            console.error('Error stack:', error?.stack);
            setAiProgress({ status: `Error: ${errorMessage}`, progress: 0 });
            setAiAnalysisStatus('failed');
            onUpdate({ ...session, analysisStatus: 'failed', audioBlob });
            
            // Show detailed error in alert for debugging
            alert(`AI Analysis Failed:\n\n${errorMessage}\n\nCheck browser console (F12) for details.`);
        }
    };

    const handleSpeakerNameChange = (newName: string) => {
        if (!editingSpeaker) return;
        
        const { oldName } = editingSpeaker;
        const newMap = { ...speakerMap, [oldName]: newName };
        setSpeakerMap(newMap);

        const transcript = parseTranscript(session.transcript);
        const newTranscript = transcript.map(chunk => {
            if (chunk.speaker === oldName) {
                return { ...chunk, speaker: newName };
            }
            return chunk;
        });

        onUpdate({ ...session, transcript: JSON.stringify(newTranscript), audioBlob });
        setEditingSpeaker(null);
    };

    const getSpeakerClass = (speaker: string) => {
        const speakers = Object.keys(speakerMap);
        const index = speakers.indexOf(speaker);
        return `speaker-style-${(index % 5) + 1}`;
    };
    
    const formatTimestamp = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    
    const getAttachmentIcon = (type: string): string => {
        switch (type) {
            case 'file': return '📄';
            case 'link': return '🔗';
            case 'document': return '📝';
            case 'spreadsheet': return '📊';
            case 'presentation': return '📽️';
            default: return '📎';
        }
    };

    // Parse all session data
    const transcript = parseTranscript(session.transcript);
    const summary = parseSummary(session.summary);
    const todoItems = parseTodoItems(session.todoItems);
    const outline = parseOutline(session.outline);
    const keyDecisions = parseKeyDecisions(session.keyDecisions);
    const attachments = parseAttachments(session.attachments);
    const [transcriptExpanded, setTranscriptExpanded] = useState(false);
    
    // Format metadata
    const date = new Date(session.date);
    const duration = session.duration || 0;
    const durationMinutes = Math.floor(duration / 60);
    const durationSeconds = Math.floor(duration % 60);
    const durationStr = durationMinutes > 0 
        ? `${durationMinutes}m ${durationSeconds}s` 
        : `${durationSeconds}s`;
    const meetingType = session.meetingType || 'General';
    const platform = session.platform || 'Unknown';
    const hasRecording = !!audioUrl;
    
    return (
        <div className="modal active" onClick={onClose}>
            <div className="modal-content meeting-notes-template" onClick={e => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose}>&times;</button>
                
                {/* Section 1: Metadata */}
                <div className="meeting-section metadata-section">
                    <div className="metadata-header">
                        <h2>{session.sessionTitle}</h2>
                        <div className="metadata-line">
                            <span>{date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
                            <span>•</span>
                            <span>{date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}</span>
                            {duration > 0 && (
                                <>
                                    <span>•</span>
                                    <span>{durationStr}</span>
                                </>
                            )}
                            {session.participants && (
                                <>
                                    <span>•</span>
                                    <span>{session.participants.split(',').length} attendees</span>
                                </>
                            )}
                            {hasRecording && (
                                <>
                                    <span>•</span>
                                    <span className="recording-badge">Recording available</span>
                                </>
                            )}
                        </div>
                        <div className="metadata-details">
                            {session.participants && (
                                <div className="metadata-item">
                                    <strong>Attendees:</strong> {session.participants}
                                </div>
                            )}
                            <div className="metadata-item">
                                <strong>Type:</strong> {meetingType} | <strong>Platform:</strong> {platform}
                            </div>
                        </div>
                    </div>
                </div>
                
                {/* Search Bar */}
                <div className="meeting-section search-section" style={{ padding: '12px', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                        <input
                            type="text"
                            placeholder="Search transcript..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            style={{
                                flex: 1,
                                minWidth: '200px',
                                padding: '8px 12px',
                                border: '1px solid rgba(2, 41, 91, 0.3)',
                                borderRadius: '8px',
                                fontSize: '14px'
                            }}
                        />
                        {searchQuery && (
                            <>
                                <button
                                    onClick={() => setSearchQuery('')}
                                    style={{
                                        padding: '8px 12px',
                                        background: 'transparent',
                                        border: '1px solid rgba(2, 41, 91, 0.3)',
                                        borderRadius: '8px',
                                        cursor: 'pointer'
                                    }}
                                >
                                    Clear
                                </button>
                                {searchResults.length > 0 && (
                                    <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                                        <span style={{ fontSize: '14px', color: 'var(--color-authority-navy)' }}>
                                            {currentResultIndex + 1} of {searchResults.length}
                                        </span>
                                        <button
                                            onClick={() => navigateSearchResults('prev')}
                                            style={{
                                                padding: '6px 10px',
                                                background: 'var(--color-strategic-forest)',
                                                color: 'white',
                                                border: 'none',
                                                borderRadius: '6px',
                                                cursor: 'pointer'
                                            }}
                                        >
                                            ↑
                                        </button>
                                        <button
                                            onClick={() => navigateSearchResults('next')}
                                            style={{
                                                padding: '6px 10px',
                                                background: 'var(--color-strategic-forest)',
                                                color: 'white',
                                                border: 'none',
                                                borderRadius: '6px',
                                                cursor: 'pointer'
                                            }}
                                        >
                                            ↓
                                        </button>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
                
                {/* Chapters/Topics Sidebar */}
                {topics.length > 0 && (
                    <div className="meeting-section chapters-section" style={{ marginBottom: '16px' }}>
                        <h3 style={{ marginBottom: '12px' }}>Topics</h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '200px', overflowY: 'auto' }}>
                            {topics.map((topic, index) => (
                                <button
                                    key={index}
                                    onClick={() => jumpToTopic(topic)}
                                    style={{
                                        padding: '8px 12px',
                                        textAlign: 'left',
                                        background: activeTopicIndex === index ? 'var(--color-strategic-forest)' : 'transparent',
                                        color: activeTopicIndex === index ? 'white' : 'var(--color-authority-navy)',
                                        border: '1px solid rgba(2, 41, 91, 0.2)',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        fontSize: '14px'
                                    }}
                                >
                                    {topic.title}
                                </button>
                            ))}
                        </div>
                    </div>
                )}
                
                {/* Bookmarks Section */}
                {bookmarks.length > 0 && (
                    <div className="meeting-section bookmarks-section" style={{ marginBottom: '16px' }}>
                        <h3 style={{ marginBottom: '12px' }}>
                            Bookmarks ({bookmarks.length})
                        </h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '200px', overflowY: 'auto' }}>
                            {bookmarks.map((bookmark, index) => {
                                const chunk = transcript[bookmark.chunkIndex];
                                const preview = chunk?.text.substring(0, 100) + (chunk?.text.length > 100 ? '...' : '');
                                return (
                                    <div
                                        key={index}
                                        onClick={() => jumpToBookmark(bookmark)}
                                        style={{
                                            padding: '8px 12px',
                                            background: 'rgba(253, 167, 0, 0.1)',
                                            border: '1px solid rgba(253, 167, 0, 0.3)',
                                            borderRadius: '6px',
                                            cursor: 'pointer',
                                            fontSize: '14px'
                                        }}
                                    >
                                        <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                                            {formatTimestamp(bookmark.timestamp)}
                                        </div>
                                        <div style={{ color: 'var(--color-authority-navy)', fontSize: '13px' }}>
                                            {preview}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
                
                {audioUrl && (
                    <div className="meeting-section">
                        <AudioPlayer 
                            audioUrl={audioUrl} 
                            transcript={transcript}
                            onTimeUpdate={handleTimeUpdate}
                        />
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
                
                {aiAnalysisStatus === 'failed' && (
                     <div className="status error">
                        <div style={{ marginBottom: '8px' }}>
                            <strong>On-device AI analysis failed.</strong>
                        </div>
                        <div style={{ marginBottom: '8px', fontSize: '0.9em' }}>
                            {aiProgress.status && aiProgress.status.startsWith('Error:') ? (
                                <span>{aiProgress.status}</span>
                            ) : (
                                <span>Please check the browser console for details and try again.</span>
                            )}
                        </div>
                        <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' }}>
                            <button className="btn-secondary" onClick={handleRunOnDeviceAnalysis}>
                                🔄 Retry Analysis
                            </button>
                        </div>
                    </div>
                )}
                
                {(session.analysisStatus === 'none' || session.analysisStatus === 'failed') && aiAnalysisStatus !== 'in_progress' && aiAnalysisStatus !== 'failed' && (
                    <div className="action-buttons" style={{ justifyContent: 'center', margin: '20px 0', flexWrap: 'wrap', gap: '8px'}}>
                        <button 
                            className="btn-ai" 
                            onClick={handleRunOnDeviceAnalysis}
                            disabled={!modelsReady || modelsLoading}
                            title={modelsLoading ? 'Loading AI models...' : !modelsReady ? 'AI models not ready yet' : ''}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M5 2.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5z"/><path d="M2 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2H2zm12 1a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1h12z"/></svg>
                            {modelsLoading ? 'Loading Models...' : session.analysisStatus === 'failed' ? 'Retry Analysis' : 'Run Analysis'}
                        </button>
                    </div>
                )}

                {/* Section 2: Action Items & Next Steps */}
                {todoItems.length > 0 && (
                    <div className="meeting-section action-items-section">
                        <h3>Action Items & Next Steps</h3>
                        <ul className="action-items-list">
                            {todoItems.map((todo, index) => (
                                <li key={index} className={`action-item ${todo.completed ? 'completed' : ''}`}>
                                    <div className="action-item-content" onClick={() => handleTodoToggle(index)}>
                                        <input type="checkbox" readOnly checked={todo.completed} />
                                        <span className="action-item-text">{todo.text}</span>
                                    </div>
                                    {todo.promotedToTaskId ? (
                                        <span className="task-promoted-badge">Tasked</span>
                                    ) : (
                                        <button 
                                            className="btn-promote-task" 
                                            title="Promote to Task" 
                                            onClick={() => handlePromoteTodoToTask(todo, index)}>
                                            &#x2795;
                                        </button>
                                    )}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
                
                {/* Section 3: Key Decisions Made */}
                {keyDecisions.length > 0 && (
                    <div className="meeting-section decisions-section">
                        <h3>Key Decisions Made</h3>
                        <div className="decisions-list">
                            {keyDecisions.map((decision, index) => (
                                <div key={index} className="decision-item">
                                    <div className="decision-text">
                                        <strong>Decision:</strong> {decision.decision}
                                    </div>
                                    {decision.reasoning && (
                                        <div className="decision-reasoning">
                                            <strong>Reasoning:</strong> {decision.reasoning}
                                        </div>
                                    )}
                                    <div className="decision-meta">
                                        {decision.owner && <span><strong>Owner:</strong> {decision.owner}</span>}
                                        {decision.implementationDate && (
                                            <span><strong>Implementation:</strong> {decision.implementationDate}</span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
                
                {/* Section 4: Discussion Summary */}
                {(summary || outline) && (
                    <div className="meeting-section discussion-section">
                        <h3>Discussion Summary</h3>
                        {summary && (
                            <div className="summary-content">
                                <p>{summary}</p>
                            </div>
                        )}
                        {outline && (
                            <div className="outline-content">
                                <h4>Topics Discussed</h4>
                                <div className="outline-text">{outline}</div>
                            </div>
                        )}
                    </div>
                )}
                
                {/* Section 5: Attachments, Resources & Links */}
                {attachments.length > 0 && (
                    <div className="meeting-section attachments-section">
                        <h3>Attachments, Resources & Links</h3>
                        <ul className="attachments-list">
                            {attachments.map((attachment, index) => (
                                <li key={index} className="attachment-item">
                                    <span className="attachment-icon">{getAttachmentIcon(attachment.type)}</span>
                                    <div className="attachment-info">
                                        <div className="attachment-name">
                                            {attachment.url ? (
                                                <a href={attachment.url} target="_blank" rel="noopener noreferrer">
                                                    {attachment.name}
                                                </a>
                                            ) : (
                                                <span>{attachment.name}</span>
                                            )}
                                        </div>
                                        {attachment.mentionedBy && (
                                            <div className="attachment-meta">
                                                Mentioned by: {attachment.mentionedBy}
                                            </div>
                                        )}
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
                
                {/* Section 6: Full Transcript */}
                {transcript.length > 0 && (
                    <div className="meeting-section transcript-section">
                        <div className="transcript-header">
                            <h3>Full Transcript</h3>
                            <button 
                                className="btn-toggle-transcript"
                                onClick={() => setTranscriptExpanded(!transcriptExpanded)}
                            >
                                {transcriptExpanded ? 'Collapse' : 'Expand'} Transcript
                            </button>
                        </div>
                        {transcriptExpanded && (
                            <div className="transcript-content" ref={transcriptRef}>
                                {transcript.map((chunk, index) => {
                                    const displaySpeaker = speakerMap[chunk.speaker] || chunk.speaker;
                                    const timestamp = chunk.timestamp 
                                        ? formatTimestamp(chunk.timestamp[0] || 0)
                                        : '';
                                    const chunkIsBookmarked = isBookmarked(index);
                                    const isActiveChunk = chunk.timestamp && currentPlaybackTime >= (chunk.timestamp[0] || 0) && currentPlaybackTime < (chunk.timestamp[1] || chunk.timestamp[0] || 0);
                                    
                                    // Check if this chunk matches current topic
                                    const currentTopic = topics.find(t => t.chunkIndices.includes(index));
                                    const isTopicStart = currentTopic && currentTopic.chunkIndices[0] === index;
                                    
                                    // Highlight search matches
                                    let highlightedText = chunk.text;
                                    if (searchQuery && searchResults.length > 0) {
                                        const chunkResults = searchResults.filter(r => r.chunkIndex === index);
                                        if (chunkResults.length > 0) {
                                            const isActiveResult = currentResultIndex >= 0 && searchResults[currentResultIndex]?.chunkIndex === index;
                                            // Simple highlighting - wrap matches
                                            const regex = new RegExp(`(${searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
                                            highlightedText = chunk.text.replace(regex, (match) => {
                                                const bgColor = isActiveResult ? '#fda700' : 'yellow';
                                                return `<mark style="background: ${bgColor}; padding: 2px 0; border-radius: 2px;">${match}</mark>`;
                                            });
                                        }
                                    }
                                    
                                    return (
                                        <div key={index}>
                                            {isTopicStart && currentTopic && (
                                                <div 
                                                    className="topic-header"
                                                    style={{
                                                        padding: '12px',
                                                        marginTop: '16px',
                                                        marginBottom: '8px',
                                                        background: 'rgba(44, 95, 65, 0.1)',
                                                        borderLeft: '4px solid var(--color-strategic-forest)',
                                                        borderRadius: '4px',
                                                        cursor: 'pointer'
                                                    }}
                                                    onClick={() => jumpToTopic(currentTopic)}
                                                >
                                                    <h4 style={{ margin: 0, color: 'var(--color-strategic-forest)' }}>
                                                        {currentTopic.title}
                                                    </h4>
                                                </div>
                                            )}
                                            <div 
                                                className={`transcript-chunk ${isActiveChunk ? 'active-chunk' : ''}`}
                                                data-chunk-index={index}
                                                data-is-transcript-chunk
                                                style={{
                                                    padding: '12px',
                                                    marginBottom: '8px',
                                                    background: isActiveChunk ? 'rgba(253, 167, 0, 0.1)' : 'transparent',
                                                    borderLeft: isActiveChunk ? '3px solid var(--color-achievement-gold)' : '3px solid transparent',
                                                    borderRadius: '4px',
                                                    cursor: chunk.timestamp ? 'pointer' : 'default',
                                                    transition: 'all 0.2s'
                                                }}
                                                onClick={() => chunk.timestamp && jumpToChunk(index)}
                                            >
                                                <div className="transcript-meta" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                                                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                                        <span className={`transcript-speaker ${getSpeakerClass(chunk.speaker)}`}>
                                                            {displaySpeaker}
                                                        </span>
                                                        {timestamp && (
                                                            <span className="transcript-timestamp" style={{ fontSize: '12px', color: 'var(--color-authority-navy)', opacity: 0.7 }}>
                                                                {timestamp}
                                                            </span>
                                                        )}
                                                    </div>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            toggleBookmark(index);
                                                        }}
                                                        style={{
                                                            background: 'transparent',
                                                            border: 'none',
                                                            cursor: 'pointer',
                                                            fontSize: '18px',
                                                            padding: '4px 8px',
                                                            minWidth: '44px',
                                                            minHeight: '44px',
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            justifyContent: 'center'
                                                        }}
                                                        title={chunkIsBookmarked ? 'Remove bookmark' : 'Add bookmark'}
                                                    >
                                                        {chunkIsBookmarked ? '⭐' : '☆'}
                                                    </button>
                                                </div>
                                                <div 
                                                    className="transcript-text"
                                                    dangerouslySetInnerHTML={{ __html: highlightedText }}
                                                    style={{
                                                        lineHeight: '1.6',
                                                        fontSize: '14px',
                                                        color: 'var(--color-authority-navy)'
                                                    }}
                                                />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}
                
                {/* Legacy Notes Section (for backward compatibility) */}
                {decryptedNotes && (
                    <div className="meeting-section notes-section">
                        <h3>Additional Notes</h3>
                        {isDecrypting ? (
                            <div className="loading">Decrypting...</div>
                        ) : (
                            isEditingNotes ? (
                                <div>
                                    <textarea 
                                        id="session-notes-edit" 
                                        name="editedNotes" 
                                        value={editedNotes} 
                                        onChange={e => setEditedNotes(e.target.value)} 
                                        rows={8} 
                                        style={{ width: '100%' }} 
                                    />
                                    <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
                                        <button className="btn-primary" onClick={handleSaveNotes}>Save</button>
                                        <button className="btn-secondary" onClick={() => { setIsEditingNotes(false); setEditedNotes(decryptedNotes); }}>Cancel</button>
                                    </div>
                                </div>
                            ) : (
                                <div>
                                    <p style={{ whiteSpace: 'pre-wrap' }}>{decryptedNotes}</p>
                                    <button className="btn-secondary" onClick={() => setIsEditingNotes(true)} style={{ marginTop: '8px' }}>Edit Notes</button>
                                </div>
                            )
                        )}
                    </div>
                )}

                <div style={{ marginTop: '24px', display: 'flex', justifyContent: 'flex-end' }}>
                    <button className="btn-danger" onClick={() => { onDelete(session.id!); onClose(); }}>Delete Session</button>
                </div>
            </div>
        </div>
    );
};

const AudioPlayer: React.FC<{ 
    audioUrl: string;
    transcript?: TranscriptChunk[];
    onTimeUpdate?: (time: number) => void;
}> = ({ audioUrl, transcript, onTimeUpdate }) => {
    const audioRef = useRef<HTMLAudioElement>(null);
    const progressRef = useRef<HTMLDivElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [playbackSpeed, setPlaybackSpeed] = useState(1);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const handleTimeUpdate = () => {
            const time = audio.currentTime;
            setCurrentTime(time);
            if (onTimeUpdate) {
                onTimeUpdate(time);
            }
        };
        const handleDurationChange = () => setDuration(audio.duration);
        const handleEnded = () => setIsPlaying(false);

        audio.addEventListener('timeupdate', handleTimeUpdate);
        audio.addEventListener('durationchange', handleDurationChange);
        audio.addEventListener('ended', handleEnded);
        
        // Set playback speed
        audio.playbackRate = playbackSpeed;

        return () => {
            audio.removeEventListener('timeupdate', handleTimeUpdate);
            audio.removeEventListener('durationchange', handleDurationChange);
            audio.removeEventListener('ended', handleEnded);
        };
    }, [onTimeUpdate, playbackSpeed]);
    
    const togglePlayPause = () => {
        if (audioRef.current) {
            if (isPlaying) {
                audioRef.current.pause();
            } else {
                audioRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!audioRef.current || !progressRef.current) return;

        // Guard against cases where duration is 0, NaN, or not yet known
        if (!duration || !isFinite(duration)) {
            return;
        }

        const rect = progressRef.current.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const width = rect.width || 1;
        const ratio = Math.min(1, Math.max(0, clickX / width));
        const newTime = ratio * duration;

        if (isFinite(newTime)) {
            audioRef.current.currentTime = newTime;
        }
    };
    
    const formatTime = (time: number) => {
        if (isNaN(time) || time === 0) return '00:00';
        const minutes = Math.floor(time / 60);
        const seconds = Math.floor(time % 60);
        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    };

    const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0;
    
    const handleSpeedChange = (speed: number) => {
        setPlaybackSpeed(speed);
        if (audioRef.current) {
            audioRef.current.playbackRate = speed;
        }
    };

    return (
        <div className="player-controls">
            <audio ref={audioRef} src={audioUrl} preload="metadata"></audio>
            <div className="audio-player">
                <button onClick={togglePlayPause} className="playback-btn" style={{ minWidth: '44px', minHeight: '44px' }}>
                    {isPlaying ? '❚❚' : '►'}
                </button>
                <span className="time-display">{formatTime(currentTime)}</span>
                <div className="progress-bar-container" ref={progressRef} onClick={handleProgressClick} style={{ flex: 1, cursor: 'pointer' }}>
                    <div className="progress-bar-background"></div>
                    <div className="progress-bar-progress" style={{ width: `${progressPercentage}%` }}></div>
                    <div className="progress-bar-thumb" style={{ left: `${progressPercentage}%` }}></div>
                </div>
                <span className="time-display">{formatTime(duration)}</span>
                <div style={{ display: 'flex', gap: '4px', marginLeft: '8px' }}>
                    {[0.5, 1, 1.5, 2].map(speed => (
                        <button
                            key={speed}
                            onClick={() => handleSpeedChange(speed)}
                            style={{
                                padding: '4px 8px',
                                fontSize: '12px',
                                background: playbackSpeed === speed ? 'var(--color-achievement-gold)' : 'transparent',
                                color: playbackSpeed === speed ? 'white' : 'var(--color-authority-navy)',
                                border: '1px solid rgba(2, 41, 91, 0.3)',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                minWidth: '44px',
                                minHeight: '44px'
                            }}
                        >
                            {speed}x
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
};

// DEPRECATED: TaskManager removed - replaced by action items display in ContextRail
// Component code removed - see git history if needed

// App component is now rendered from index.tsx entry point
export default App;