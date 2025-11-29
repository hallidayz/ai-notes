import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';

// Import HTTP interceptors (separate file to avoid Fast Refresh issues)
import './http-interceptors';

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
        
        // Only preload models on first time setup - otherwise let them load on demand
        // This avoids unnecessary downloads and errors when there's nothing to process
        if (isFirstTime && typeof window !== 'undefined') {
            // Start preloading models in the background (don't await - let it happen async)
            setTimeout(async () => {
                try {
                    const aiService = OnDeviceAIService.getInstance();
                    let modelsDownloaded = false;
                    
                    // Check if transcription model is already loaded, if not download it
                    try {
                        await aiService.getTranscriptionPipeline(undefined, (progress) => {
                            if (progress?.status === 'downloading') {
                                modelsDownloaded = true;
                                const message = `Downloading transcription model: ${Math.round(progress.progress || 0)}%`;
                                window.dispatchEvent(new CustomEvent('modelDownloadProgress', { detail: { message } }));
                            }
                        });
                    } catch (error: any) {
                        // Silently fail - models will download on first use
                        // Only log if it's not the expected "Unsupported model type" error
                        if (!error?.message?.includes('Unsupported model type')) {
                            console.error('Error preloading transcription model:', error);
                        }
                    }
                    
                    // Check if analysis model is already loaded, if not download it
                    try {
                        await aiService.getAnalysisPipeline((progress) => {
                            if (progress?.status === 'downloading') {
                                modelsDownloaded = true;
                                const message = `Downloading analysis model: ${Math.round(progress.progress || 0)}%`;
                                window.dispatchEvent(new CustomEvent('modelDownloadProgress', { detail: { message } }));
                            }
                        });
                    } catch (error: any) {
                        // Silently fail - models will download on first use
                        if (!error?.message?.includes('Unsupported model type')) {
                            console.error('Error preloading analysis model:', error);
                        }
                    }
                    
                    if (modelsDownloaded) {
                        window.dispatchEvent(new CustomEvent('modelsDownloaded', { 
                            detail: { message: 'Models downloaded successfully!' }
                        }));
                    }
                } catch (error) {
                    // Silently fail - models will download on first use if this fails
                    console.debug('Model preloading skipped:', error);
                }
            }, 2000); // Wait 2 seconds after page load to start downloads
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

    private constructor() {}

    public static getInstance(): OnDeviceAIService {
        if (!this.instance) {
            this.instance = new OnDeviceAIService();
        }
        return this.instance;
    }

    // Map language codes to Whisper model variants
    // Using smaller models for better performance and lower latency
    private getWhisperModel(language?: string): string {
        const lang = language || 'en';
        // Use smaller models (tiny/base) for faster processing and lower latency
        // tiny models are ~75MB vs base ~150MB - significantly faster with minimal accuracy loss
        if (lang === 'en') {
            return 'Xenova/whisper-tiny.en'; // Tiny English model - fastest, lowest latency
        } else {
            return 'Xenova/whisper-tiny'; // Tiny multilingual model
        }
    }

    private transcriptionLanguage: string | null = null;

    public async getTranscriptionPipeline(language?: string, progress_callback?: (progress: any) => void) {
        const lang = language || 'en';
        const modelName = this.getWhisperModel(lang);
        
        // Reload pipeline if language changed
        if (!this.transcriptionPipe || this.transcriptionLanguage !== lang) {
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
        }
        return this.transcriptionPipe;
    }

    public async getAnalysisPipeline(progress_callback?: (progress: any) => void) {
        if (!this.tokenizer || !this.model) {
            try {
                const transformers = await getTransformers();
                if (!transformers || !transformers.AutoTokenizer || !transformers.AutoModelForSeq2SeqLM) {
                    throw new Error('Transformers.js not loaded');
                }
                
                // Ensure transformers.js uses our proxy
                if (transformers.env) {
                    transformers.env.remoteHost = 'http://localhost:3001';
                }
                
                const progressHandler = (progress: any) => {
                    if (progress_callback) progress_callback(progress);
                };
                this.tokenizer = await transformers.AutoTokenizer.from_pretrained('Xenova/LaMini-Flan-T5-783M', { progress_callback: progressHandler });
                this.model = await transformers.AutoModelForSeq2SeqLM.from_pretrained('Xenova/LaMini-Flan-T5-783M', { progress_callback: progressHandler });
                
                // Verify model functionality after load
                if (!this.tokenizer || typeof this.tokenizer !== 'function') {
                    throw new Error('Tokenizer failed verification - not a function');
                }
                if (!this.model || typeof this.model.generate !== 'function') {
                    throw new Error('Analysis model failed verification - generate method missing');
                }
            } catch (error: any) {
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
            const inputs = this.tokenizer(prompt, {
                return_tensors: 'pt',
                padding: true,
                truncation: true,
                max_length: 1024
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
                do_sample: false
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
}

interface TodoItem {
    text: string;
    completed: boolean;
    promotedToTaskId?: number;
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
            const request = indexedDB.open(this.DB_NAME, 4); // Increment version for encryption migration

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

    useEffect(() => {
        const checkPin = async () => {
            const storedPinHash = await db.getConfig('pinHash');
            if (!storedPinHash) {
                // First time setup
                setIsAuthenticated(true);
            }
            setIsLoading(false);
        };
        checkPin();
    }, []);

    const handlePinSet = (newPin: string) => {
        setPin(newPin);
        setIsAuthenticated(true);
    };

    const handleLogin = (enteredPin: string) => {
        setPin(enteredPin);
        setIsAuthenticated(true);
    };

    if (isLoading) {
        return <div className="loading">Loading...</div>;
    }

    return isAuthenticated ? <MainApp pin={pin!} /> : <AuthScreen onPinSet={handlePinSet} onLogin={handleLogin} />;
};

const AuthScreen: React.FC<{ onPinSet: (pin: string) => void, onLogin: (pin: string) => void }> = ({ onPinSet, onLogin }) => {
    const [isSetup, setIsSetup] = useState(false);
    const [pin, setPin] = useState('');
    const [confirmPin, setConfirmPin] = useState('');
    const [isConfirming, setIsConfirming] = useState(false);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [isVerifying, setIsVerifying] = useState(false);

    useEffect(() => {
        const checkSetup = async () => {
            const storedPinHash = await db.getConfig('pinHash');
            setIsSetup(!!storedPinHash);
            setIsLoading(false);
        };
        checkSetup();
    }, []);

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
            try {
                // The "real" test is to decrypt something. Let's try to decrypt the encrypted pin itself.
                const storedEncryptedPinCheck = await db.getConfig('encryptedPinCheck');
                await CryptoService.decrypt(storedEncryptedPinCheck, pin);
                onLogin(pin);
            } catch(e) {
                setError('Invalid PIN. Please try again.');
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

    return (
        <div className="auth-container">
            <div className="auth-card">
                <h2>{title}</h2>
                <div className="auth-error">{error || (isVerifying && 'Verifying...')}</div>
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
                        <button key={num} onClick={() => handleKeyPress(String(num))} disabled={isVerifying}>{num}</button>
                    ))}
                    <button disabled={isVerifying}></button>
                    <button onClick={() => handleKeyPress('0')} disabled={isVerifying}>0</button>
                    <button onClick={handleBackspace} disabled={isVerifying}>&larr;</button>
                </div>
            </div>
        </div>
    );
};

const MainApp: React.FC<{ pin: string }> = ({ pin }) => {
    const [sessions, setSessions] = useState<Session[]>([]);
    const [tasks, setTasks] = useState<Task[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [status, setStatus] = useState({ message: '', type: '' });
    const [selectedSession, setSelectedSession] = useState<Session | null>(null);
    const [view, setView] = useState<'sessions' | 'tasks'>('sessions');
    const [industry, setIndustry] = useState<string>('general');
    const [language, setLanguage] = useState<string>('en');
    
    useEffect(() => {
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
        
        return () => {
            window.removeEventListener('firstTimeSetup', handleFirstTimeSetup as EventListener);
            window.removeEventListener('modelDownloadProgress', handleModelDownloadProgress as EventListener);
            window.removeEventListener('modelsDownloaded', handleModelsDownloaded as EventListener);
        };
    }, []);

    const showStatus = (message: string, type: 'success' | 'error' | 'info', duration = 3000) => {
        setStatus({ message, type });
        setTimeout(() => setStatus({ message: '', type: '' }), duration);
    };

    const handleAddSession = async (session: Omit<Session, 'id' | 'timestamp' | 'notes'>, notes: string, audioBlob: Blob | null) => {
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
            return true;
        } catch (error) {
            showStatus('Failed to save session.', 'error');
            return false;
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

    return (
        <div className="container">
            <header>
                <ThemeToggle />
                <div className="logo-container">
                    <img 
                        src="/logo.png" 
                        alt="AI Notes Logo" 
                        style={{ 
                            height: '48px', 
                            width: '48px',
                            borderRadius: '8px',
                            objectFit: 'contain'
                        }} 
                    />
                    <div>
                        <h1>AI Notes</h1>
                        <p>Your On-Device, Private, AI Powered Notes</p>
                    </div>
                </div>
                <p className="author-credit">by AC MiNDS.</p>
                <div className="settings-container">
                    <span>Purpose:</span>
                    <select id="industrySelector" value={industry} onChange={handleIndustryChange}>
                        <option value="general">General</option>
                        <option value="therapy">Therapy Session</option>
                        <option value="medical">Medical Dictation</option>
                        <option value="legal">Legal Note</option>
                        <option value="business">Business Meeting</option>
                    </select>
                    <span>Language:</span>
                    <select id="languageSelector" value={language} onChange={handleLanguageChange}>
                        <option value="en">English</option>
                        <option value="es">Spanish</option>
                        <option value="fr">French</option>
                        <option value="de">German</option>
                        <option value="it">Italian</option>
                        <option value="pt">Portuguese</option>
                        <option value="zh">Chinese</option>
                        <option value="ja">Japanese</option>
                        <option value="ko">Korean</option>
                        <option value="ru">Russian</option>
                        <option value="ar">Arabic</option>
                        <option value="hi">Hindi</option>
                    </select>
                </div>
            </header>

            {status.message && <div className={`status ${status.type}`}>{status.message}</div>}

            <ViewSwitcher view={view} setView={setView} />

            {isLoading ? (
                <div className="loading"><div className="spinner"></div>Loading...</div>
            ) : (
                view === 'sessions' ? (
                    <>
                        <NewSessionForm onAddSession={handleAddSession} showStatus={showStatus} />
                        <SessionsList sessions={sessions} onSelect={setSelectedSession} onDelete={handleDeleteSession} pin={pin} />
                    </>
                ) : (
                    <TaskManager 
                        tasks={tasks}
                        onAddTask={handleAddTask}
                        onUpdateTask={handleUpdateTask}
                        onDeleteTask={handleDeleteTask}
                        sessions={sessions}
                    />
                )
            )}

            {selectedSession && (
                <SessionDetailModal
                    session={selectedSession}
                    onClose={() => setSelectedSession(null)}
                    onDelete={handleDeleteSession}
                    onUpdate={handleUpdateSession}
                    onAddTask={handleAddTask}
                    pin={pin}
                />
            )}
        </div>
    );
};

const ViewSwitcher: React.FC<{ view: 'sessions' | 'tasks', setView: (view: 'sessions' | 'tasks') => void }> = ({ view, setView }) => (
    <div className="view-switcher">
        <button className={view === 'sessions' ? 'active' : ''} onClick={() => setView('sessions')}>Sessions</button>
        <button className={view === 'tasks' ? 'active' : ''} onClick={() => setView('tasks')}>Tasks</button>
    </div>
);

type ShowStatusType = (message: string, type: 'success' | 'error' | 'info', duration?: number) => void;

const NewSessionForm: React.FC<{ 
    onAddSession: (session: Omit<Session, 'id' | 'timestamp' | 'notes'>, notes: string, audioBlob: Blob | null) => Promise<boolean>,
    showStatus: ShowStatusType 
}> = ({ onAddSession, showStatus }) => {
    const [sessionTitle, setSessionTitle] = useState('');
    const [participants, setParticipants] = useState('');
    const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
    const [notes, setNotes] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [duration, setDuration] = useState(0);
    const [audioLevel, setAudioLevel] = useState(0); // For visual modulator (0-100)
    const [recordingSource, setRecordingSource] = useState<string>(''); // Display current source
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioStreamRef = useRef<MediaStream | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const dbRef = useRef<TherapyDB | null>(null);

    /**
     * Start audio level monitoring for visual feedback
     */
    const startAudioLevelMonitoring = (stream: MediaStream) => {
        try {
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.8;
            
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            
            analyserRef.current = analyser;
            
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            const updateLevel = () => {
                if (!analyserRef.current || !isRecording) {
                    return;
                }
                
                analyserRef.current.getByteFrequencyData(dataArray);
                
                // Calculate average volume
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) {
                    sum += dataArray[i];
                }
                const average = sum / dataArray.length;
                
                // Convert to 0-100 scale
                const level = Math.min(100, Math.max(0, (average / 255) * 100));
                setAudioLevel(level);
                
                animationFrameRef.current = requestAnimationFrame(updateLevel);
            };
            
            updateLevel();
        } catch (error) {
            console.warn('Could not start audio level monitoring:', error);
        }
    };

    /**
     * Stop audio level monitoring
     */
    const stopAudioLevelMonitoring = () => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        analyserRef.current = null;
        setAudioLevel(0);
    };

    const handleStartRecording = async () => {
        try {
            // Simplified, mic-only recording path for maximum reliability
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true },
                video: false
            });

            audioStreamRef.current = stream;
            setRecordingSource('Recording: Microphone');

            setIsRecording(true);
            setDuration(0);
            timerRef.current = setInterval(() => setDuration(prev => prev + 1), 1000);

            // Start audio level monitoring for visual feedback
            startAudioLevelMonitoring(stream);

            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : MediaRecorder.isTypeSupported('audio/webm')
                ? 'audio/webm'
                : 'audio/ogg';

            const options: MediaRecorderOptions = {
                mimeType,
                audioBitsPerSecond: 64000
            };

            mediaRecorderRef.current = new MediaRecorder(stream, options);
            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };
            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: mimeType });
                setAudioBlob(blob);
                chunksRef.current = [];
                stream.getTracks().forEach(track => track.stop());
                stopAudioLevelMonitoring();
            };

            mediaRecorderRef.current.start(1000); // 1 second chunks
        } catch (err: any) {
            const errorMsg = err?.message || 'Unknown error';
            showStatus(
                `Could not start recording: ${errorMsg}. Please ensure you have given microphone permissions.`,
                'error'
            );
            stopAudioLevelMonitoring();
        }
    };

    const handleStopRecording = async () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            if(timerRef.current) clearInterval(timerRef.current);
            
            // Stop audio level monitoring
            stopAudioLevelMonitoring();
            
            // Cleanup stream
            if (audioStreamRef.current) {
                audioStreamRef.current.getTracks().forEach(track => track.stop());
                audioStreamRef.current = null;
            }
            
            setRecordingSource('');
            
            // Auto-save session when recording stops (if title exists)
            // Wait for onstop handler to create the blob, then auto-save
            setTimeout(() => {
                // The blob will be set in the onstop handler
                // Check if we have chunks to create a blob from, or wait for state update
                const checkAndSave = () => {
                    // Try to get blob from state (set in onstop) or create from chunks
                    let blobToSave: Blob | null = null;
                    if (chunksRef.current.length > 0) {
                        blobToSave = new Blob(chunksRef.current, { type: 'audio/webm' });
                    } else if (audioBlob) {
                        blobToSave = audioBlob;
                    }
                    
                    if (blobToSave && sessionTitle.trim()) {
                        setIsSaving(true);
                        onAddSession({
                            sessionTitle,
                            participants,
                            date,
                            duration,
                            transcript: [],
                        }, notes, blobToSave).then(success => {
                            if (success) {
                                showStatus('Session saved automatically.', 'success');
                                // Reset form
                                setSessionTitle('');
                                setParticipants('');
                                setNotes('');
                                setAudioBlob(null);
                                setDuration(0);
                                chunksRef.current = [];
                            } else {
                                showStatus('Failed to auto-save session.', 'error');
                            }
                            setIsSaving(false);
                        });
                    } else if (blobToSave && !sessionTitle.trim()) {
                        showStatus('Recording stopped. Add a title and click "Save Session" to save.', 'info', 5000);
                    }
                };
                
                // Wait a bit more for onstop to complete and state to update
                setTimeout(checkAndSave, 300);
            }, 500);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!sessionTitle) {
            showStatus('Session title is required.', 'error');
            return;
        }
        setIsSaving(true);
        const success = await onAddSession({
            sessionTitle,
            participants,
            date,
            duration,
            transcript: [],
        }, notes, audioBlob);
        
        if (success) {
            setSessionTitle('');
            setParticipants('');
            setNotes('');
            setAudioBlob(null);
            setDuration(0);
        }
        setIsSaving(false);
    };
    
    const formatTime = (seconds: number) => {
        const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
        const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
        const s = (seconds % 60).toString().padStart(2, '0');
        return h !== '00' ? `${h}:${m}:${s}` : `${m}:${s}`;
    };

    return (
        <div className="card new-session">
            <h3>New Session</h3>
            <form onSubmit={handleSubmit}>
                {/* Top two-column layout: left form fields, right notes */}
                <div className="new-session-grid">
                    <div className="new-session-grid-left">
                        <div className="field-group">
                            <label htmlFor="session-title">Session Title</label>
                            <input
                                type="text"
                                id="session-title"
                                name="sessionTitle"
                                placeholder="Session Title"
                                value={sessionTitle}
                                onChange={e => setSessionTitle(e.target.value)}
                                required
                            />
                        </div>
                        <div className="field-group">
                            <label htmlFor="session-participants">Participants</label>
                            <input
                                type="text"
                                id="session-participants"
                                name="participants"
                                placeholder="Participants (optional)"
                                value={participants}
                                onChange={e => setParticipants(e.target.value)}
                            />
                        </div>
                        <div className="field-group">
                            <label htmlFor="session-date">Date</label>
                            <input
                                type="date"
                                id="session-date"
                                name="date"
                                value={date}
                                onChange={e => setDate(e.target.value)}
                            />
                        </div>
                    </div>
                    <div className="new-session-grid-right">
                        <div className="field-group">
                            <label htmlFor="session-notes">Notes for Reference</label>
                            <textarea
                                id="session-notes"
                                name="notes"
                                placeholder="Enter notes or a summary here..."
                                value={notes}
                                onChange={e => setNotes(e.target.value)}
                                rows={7}
                            ></textarea>
                        </div>
                    </div>
                </div>

                {/* Live meeting audio section: buttons left, visual right */}
                <div className="live-audio-section">
                    <h4>Live Meeting Audio</h4>
                    <div className="live-audio-grid">
                        <div className="live-audio-controls">
                            {recordingSource && (
                                <div className="recording-source-indicator">
                                    <span className="recording-source-badge">{recordingSource}</span>
                                </div>
                            )}
                            <div className="recording-controls-with-save live-audio-controls-row">
                                <div className="recording-controls">
                                    {!isRecording ? (
                                        <button type="button" className="btn-record" onClick={handleStartRecording}>
                                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm0 1A5 5 0 1 1 8 3a5 5 0 0 1 0 10z"/><path d="M10 8a2 2 0 1 1-4 0 2 2 0 0 1 4 0z"/></svg>
                                            Start Recording
                                        </button>
                                    ) : (
                                        <button type="button" className="btn-record recording" onClick={handleStopRecording}>
                                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M5.5 5.5A.5.5 0 0 1 6 6v4a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm4 0a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5z"/><path d="M0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V2zm15 0a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V2z"/></svg>
                                            Stop &amp; Save ({formatTime(duration)})
                                        </button>
                                    )}
                                </div>
                                <button type="submit" className="btn-primary" disabled={isSaving || isRecording}>
                                    {isSaving ? 'Saving...' : 'Save Session'}
                                </button>
                            </div>
                            {audioBlob && (
                                <div className="status info live-audio-status">
                                    Audio recorded. Save the session to attach it.
                                </div>
                            )}
                        </div>

                        <div className="live-audio-visual">
                            {isRecording && (
                                <div className="audio-level-indicator two-row">
                                    <div className="audio-level-row">
                                        <div
                                            className="audio-level-gradient"
                                            style={{ transform: `scaleX(${Math.min(1, audioLevel / 70)})` }}
                                        />
                                    </div>
                                    <div className="audio-level-row">
                                        <div
                                            className="audio-level-gradient"
                                            style={{ transform: `scaleX(${Math.min(1, audioLevel / 100)})` }}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </form>
        </div>
    );
};

const SessionsList: React.FC<{ sessions: Session[], onSelect: (session: Session) => void, onDelete: (id: number) => void, pin: string }> = ({ sessions, onSelect, onDelete, pin }) => {
    const [searchQuery, setSearchQuery] = useState('');
    const [filteredSessions, setFilteredSessions] = useState<Session[]>(sessions);
    
    useEffect(() => {
        if (!searchQuery.trim()) {
            setFilteredSessions(sessions);
            return;
        }
        
        const query = searchQuery.toLowerCase();
        const filtered = sessions.filter(session => {
            const titleMatch = session.sessionTitle?.toLowerCase().includes(query);
            const participantsMatch = session.participants?.toLowerCase().includes(query);
            const dateMatch = new Date(session.date).toLocaleDateString().toLowerCase().includes(query);
            return titleMatch || participantsMatch || dateMatch;
        });
        setFilteredSessions(filtered);
    }, [searchQuery, sessions]);
    
    if (sessions.length === 0) {
        return <div className="empty-state">No sessions yet. Create one to get started!</div>;
    }

    const decryptAndPreview = (notes: string) => {
        try {
            // This is a simplified preview. In a real app, you might want to cache decrypted previews.
            // For now, let's just show encrypted length as a placeholder for performance.
            return `Encrypted notes...`;
        } catch {
            return "Could not decrypt preview.";
        }
    };

    return (
        <div className="sessions-list">
            <div style={{ marginBottom: '16px' }}>
                <h3>Recent Sessions</h3>
                <input
                    type="text"
                    placeholder="Search sessions..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{
                        width: '100%',
                        padding: '8px 12px',
                        background: 'rgba(26, 32, 51, 0.8)',
                        border: '1px solid rgba(55, 64, 255, 0.3)',
                        borderRadius: '6px',
                        color: '#e2e8f0',
                        fontSize: '1em',
                        marginTop: '8px'
                    }}
                />
            </div>
            {filteredSessions.length === 0 && searchQuery ? (
                <div className="empty-state">No sessions match your search.</div>
            ) : (
                filteredSessions.map(session => (
                <div key={session.id} className="session-item" onClick={() => onSelect(session)}>
                    <div className="session-content">
                        <div className="session-header">
                            <span className="session-title">{session.sessionTitle}</span>
                            <span className="session-date">{new Date(session.date).toLocaleDateString()}</span>
                        </div>
                        {session.participants && <p className="session-participants">With: {session.participants}</p>}
                        <p className="session-preview">{decryptAndPreview(session.notes)}</p>
                        {session.analysisStatus && session.analysisStatus !== 'complete' && session.analysisStatus !== 'none' && (
                            <div className={`session-status-indicator ${session.analysisStatus}`}>
                                {session.analysisStatus === 'pending' && <><div className="spinner-small"></div> Processing AI analysis...</>}
                                {session.analysisStatus === 'failed' && <>&#x26A0; AI analysis failed</>}
                            </div>
                        )}
                    </div>
                    <div className="session-actions">
                        <button
                            className="icon-btn"
                            onClick={(e) => { e.stopPropagation(); onDelete(session.id!); }}
                            aria-label="Delete session"
                        >
                           &#x1F5D1;
                        </button>
                    </div>
                </div>
                ))
            )}
        </div>
    );
};

const SessionDetailModal: React.FC<{ 
    session: Session, 
    onClose: () => void, 
    onDelete: (id: number) => void, 
    onUpdate: (session: Session) => void,
    onAddTask: (task: Omit<Task, 'id' | 'timestamp'>) => Promise<boolean>,
    pin: string
}> = ({ session, onClose, onDelete, onUpdate, onAddTask, pin }) => {
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
        setAiProgress({ status: 'Starting analysis...', progress: 0 });
        onUpdate({ ...session, analysisStatus: 'pending', audioBlob });
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

            // Get industry context and language from database (use global db instance)
            const industry = await db.getConfig('industry') || 'general';
            const language = session.language || await db.getConfig('language') || 'en';
            
            const resultJson = await onDeviceAIService.analyze(
                audioBuffer, 
                (status, progress) => {
                    const elapsed = Math.round((performance.now() - analysisStartTime) / 1000);
                    const timeRemaining = Math.max(0, 60 - elapsed);
                    setAiProgress({ 
                        status: `${status} (${elapsed}s / ${timeRemaining}s remaining)`, 
                        progress: progress || 0 
                    });
                },
                industry,
                language,
                60000 // 60 second timeout
            );
            
            // Parse JSON result from AI processing
            let result: any;
            try {
                result = JSON.parse(resultJson);
            } catch (parseError) {
                throw new Error('Invalid JSON response from AI analysis');
            }
            
            // Check if result contains an error
            if (result.error) {
                throw new Error(result.message || 'AI analysis failed');
            }
            
            // Format data for UI display (PWA layer handles formatting)
            const todoItems: TodoItem[] = (result.action_items || []).map((text: string) => ({ text, completed: false }));
            
            // Store structured data as JSON strings (already JSON from AI function)
            const updatedSession = {
                ...session,
                transcript: JSON.stringify(result.transcript),
                summary: JSON.stringify({ summary: result.summary }),
                todoItems: JSON.stringify(todoItems),
                outline: JSON.stringify({ outline: result.outline }),
                analysisStatus: 'complete' as const,
                audioBlob
            };
            
            onUpdate(updatedSession);
            setAiAnalysisStatus('complete');

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

    return (
        <div className="modal active" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose}>&times;</button>
                <h2>{session.sessionTitle}</h2>
                <p style={{ color: '#94a3b8', marginBottom: '16px' }}>{new Date(session.date).toLocaleDateString()}{session.participants && ` with ${session.participants}`}</p>
                
                {audioUrl && <AudioPlayer audioUrl={audioUrl} />}

                {(session.analysisStatus === 'none' || session.analysisStatus === 'failed') && aiAnalysisStatus !== 'in_progress' && (
                    <div className="action-buttons" style={{ justifyContent: 'center', margin: '20px 0', flexWrap: 'wrap', gap: '8px'}}>
                        <button className="btn-ai" onClick={handleRunOnDeviceAnalysis}>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M5 2.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5z"/><path d="M2 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2H2zm12 1a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1h12z"/></svg>
                            {session.analysisStatus === 'failed' ? 'Retry Analysis' : 'Run On-Device Analysis'}
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

                {session.analysisStatus === 'complete' && (
                    <div className="analysis-section">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                            <h3 style={{ margin: 0 }}>AI Analysis Results</h3>
                            {aiAnalysisStatus !== 'in_progress' && (
                                <button className="btn-secondary" onClick={handleRunOnDeviceAnalysis} style={{ fontSize: '0.85em', padding: '6px 12px' }}>
                                    🔄 Re-run Analysis
                                </button>
                            )}
                        </div>
                        <div className="analysis-subsection">
                            <h4>&#x1F4DD; Summary</h4>
                            <p>{parseSummary(session.summary)}</p>
                        </div>
                         {parseTodoItems(session.todoItems).length > 0 && (
                            <div className="analysis-subsection">
                                <h4>&#x2705; Action Items</h4>
                                <ul className="action-items-list">
                                    {parseTodoItems(session.todoItems).map((todo, index) => (
                                        <li key={index} className={`todo-item ${todo.completed ? 'completed' : ''}`}>
                                            <div className="todo-content" onClick={() => handleTodoToggle(index)}>
                                                <input type="checkbox" readOnly checked={todo.completed} />
                                                <span className="todo-text">{todo.text}</span>
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
                        {parseOutline(session.outline) && (
                             <div className="analysis-subsection">
                                <h4>&#x1F4D1; Outline</h4>
                                <div className="outline-content">{parseOutline(session.outline)}</div>
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
                            <textarea id="session-notes-edit" name="editedNotes" value={editedNotes} onChange={e => setEditedNotes(e.target.value)} rows={8} style={{ width: '100%' }} />
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
                
                {parseTranscript(session.transcript).length > 0 && (
                    <>
                        <h3>Transcript</h3>
                        <div className="transcript">
                            {parseTranscript(session.transcript).map((chunk, index) => (
                                <div key={index} className={`transcript-chunk ${getSpeakerClass(chunk.speaker)}`}>
                                   {editingSpeaker?.chunkIndex === index ? (
                                        <input
                                            type="text"
                                            id={`speaker-name-${index}`}
                                            name={`speakerName-${index}`}
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

const AudioPlayer: React.FC<{ audioUrl: string }> = ({ audioUrl }) => {
    const audioRef = useRef<HTMLAudioElement>(null);
    const progressRef = useRef<HTMLDivElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
        const handleDurationChange = () => setDuration(audio.duration);
        const handleEnded = () => setIsPlaying(false);

        audio.addEventListener('timeupdate', handleTimeUpdate);
        audio.addEventListener('durationchange', handleDurationChange);
        audio.addEventListener('ended', handleEnded);

        return () => {
            audio.removeEventListener('timeupdate', handleTimeUpdate);
            audio.removeEventListener('durationchange', handleDurationChange);
            audio.removeEventListener('ended', handleEnded);
        };
    }, []);
    
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

    return (
        <div className="player-controls">
            <audio ref={audioRef} src={audioUrl} preload="metadata"></audio>
            <div className="audio-player">
                <button onClick={togglePlayPause} className="playback-btn">
                    {isPlaying ? '❚❚' : '►'}
                </button>
                <span className="time-display">{formatTime(currentTime)}</span>
                <div className="progress-bar-container" ref={progressRef} onClick={handleProgressClick}>
                    <div className="progress-bar-background"></div>
                    <div className="progress-bar-progress" style={{ width: `${progressPercentage}%` }}></div>
                    <div className="progress-bar-thumb" style={{ left: `${progressPercentage}%` }}></div>
                </div>
                <span className="time-display">{formatTime(duration)}</span>
            </div>
        </div>
    );
};

const TaskManager: React.FC<{
    tasks: Task[],
    onAddTask: (task: Omit<Task, 'id' | 'timestamp'>) => Promise<boolean>,
    onUpdateTask: (task: Task) => void,
    onDeleteTask: (id: number) => void,
    sessions: Session[],
}> = ({ tasks, onAddTask, onUpdateTask, onDeleteTask, sessions }) => {
    const [newTaskTitle, setNewTaskTitle] = useState('');
    const [newTaskPriority, setNewTaskPriority] = useState<'low' | 'medium' | 'high'>('medium');
    const [newTaskDueDate, setNewTaskDueDate] = useState<string | null>(null);

    const handleAddTask = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newTaskTitle.trim()) return;
        
        const success = await onAddTask({
            title: newTaskTitle.trim(),
            priority: newTaskPriority,
            dueDate: newTaskDueDate,
            status: 'todo',
        });

        if (success) {
            setNewTaskTitle('');
            setNewTaskPriority('medium');
            setNewTaskDueDate(null);
        }
    };

    return (
        <div className="card">
            <h3>Task Manager</h3>
            <form className="task-form" onSubmit={handleAddTask}>
                <input
                    type="text"
                    id="task-title"
                    name="taskTitle"
                    placeholder="New task..."
                    value={newTaskTitle}
                    onChange={(e) => setNewTaskTitle(e.target.value)}
                />
                <select id="task-priority" name="taskPriority" value={newTaskPriority} onChange={(e) => setNewTaskPriority(e.target.value as any)}>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                </select>
                <input
                    type="date"
                    id="task-due-date"
                    name="taskDueDate"
                    value={newTaskDueDate || ''}
                    onChange={(e) => setNewTaskDueDate(e.target.value)}
                />
                <button type="submit" className="btn-primary">Add</button>
            </form>

            <div className="task-list">
                {tasks.length > 0 ? tasks.map(task => (
                    <TaskItem 
                        key={task.id} 
                        task={task} 
                        onUpdateTask={onUpdateTask} 
                        onDeleteTask={onDeleteTask} 
                    />
                )) : (
                    <div className="empty-state">No tasks yet.</div>
                )}
            </div>
        </div>
    );
};

const TaskItem: React.FC<{
    task: Task,
    onUpdateTask: (task: Task) => void,
    onDeleteTask: (id: number) => void,
}> = ({ task, onUpdateTask, onDeleteTask }) => {

    const handleStatusChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        onUpdateTask({ ...task, status: e.target.value as any });
    };

    return (
        <div className={`task-item status-${task.status}`}>
            <div className="task-main-content">
                <div className="task-details">
                    <span className="task-title">{task.title}</span>
                    <div className="task-meta">
                        <span className={`priority-badge priority-${task.priority}`}>{task.priority}</span>
                        {task.dueDate && <span>Due: {new Date(task.dueDate).toLocaleDateString()}</span>}
                        {task.sessionName && <span className="task-session-link">From: {task.sessionName}</span>}
                    </div>
                </div>
            </div>
            <select id={`task-status-${task.id}`} name={`taskStatus-${task.id}`} className="task-status-selector" value={task.status} onChange={handleStatusChange}>
                <option value="todo">To Do</option>
                <option value="inprogress">In Progress</option>
                <option value="done">Done</option>
            </select>
            <div className="task-actions">
                <button className="icon-btn-delete" onClick={() => onDeleteTask(task.id!)} aria-label="Delete task">&#x1F5D1;</button>
            </div>
        </div>
    );
};

// --- RENDER APP ---
const container = document.getElementById('root');
if (container) {
    const root = createRoot(container);
    root.render(<App />);
}