
import { pipeline, env } from '@xenova/transformers';
import { GoogleGenAI, Type } from "@google/genai";
import { TranscriptChunk, ModelConfig } from '../types';

export class OnDeviceAIService {
    private static instance: OnDeviceAIService | null = null;
    private transcriptionPipe: unknown = null;
    private analysisPipe: unknown = null;
    private currentConfig: ModelConfig = {
        transcriptionModelId: 'whisper-tiny-en',
        analysisModelId: 'flan-t5-small'
    };

    private modelMap: { [key: string]: string } = {
        'whisper-tiny-en': 'Xenova/whisper-tiny.en',
        'whisper-base-en': 'Xenova/whisper-base.en',
        'flan-t5-small': 'Xenova/flan-t5-small',
        'flan-t5-base': 'Xenova/flan-t5-base',
        'phi-1_5': 'Xenova/phi-1_5',
        'lfm2-350m': 'Xenova/flan-t5-small',
        'lfm2-700m': 'Xenova/flan-t5-base',
        'qwen3-0.6b': 'Xenova/qwen-0.5b-instruct'
    };

    private constructor() {
        env.allowLocalModels = false;
        env.allowRemoteModels = true;
    }

    public static getInstance(): OnDeviceAIService {
        if (!this.instance) {
            this.instance = new OnDeviceAIService();
        }
        return this.instance;
    }

    public updateConfig(config: ModelConfig) {
        if (config.transcriptionModelId !== this.currentConfig.transcriptionModelId) {
            this.transcriptionPipe = null;
        }
        if (config.analysisModelId !== this.currentConfig.analysisModelId) {
            this.analysisPipe = null;
        }
        this.currentConfig = config;
    }

    public async preloadModel(modelPath: string, progress_callback?: (progress: { status: string; progress?: number }) => void) {
        // We just initialize a pipeline to trigger download
        // We don't need to store it yet if it's just preloading
        await pipeline('feature-extraction', modelPath, { progress_callback });
    }

    private async getTranscriptionPipeline(progress_callback?: (progress: { status: string; progress?: number }) => void) {
        const modelPath = this.modelMap[this.currentConfig.transcriptionModelId] || 'Xenova/whisper-tiny.en';
        if (!this.transcriptionPipe) {
            this.transcriptionPipe = await pipeline('automatic-speech-recognition', modelPath, {
                progress_callback,
            });
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return this.transcriptionPipe as any;
    }

    private async getAnalysisPipeline(progress_callback?: (progress: { status: string; progress?: number }) => void) {
        const modelPath = this.modelMap[this.currentConfig.analysisModelId] || 'Xenova/flan-t5-small';
        if (!this.analysisPipe) {
            this.analysisPipe = await pipeline('text2text-generation', modelPath, {
                progress_callback,
            });
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return this.analysisPipe as any;
    }

    public async analyze(
        audioBlob: Blob,
        industry: string,
        progressCallback: (status: string, progress?: number) => void
    ): Promise<{ transcript: TranscriptChunk[], summary: string, action_items: string[], outline: string }> {
        progressCallback('Initializing transcription model...');
        const transcriber = await this.getTranscriptionPipeline((p: { status: string; progress?: number }) => {
            if (p.status === 'progress') {
                progressCallback('Downloading transcription model...', p.progress);
            }
        });

        progressCallback('Transcribing audio...');
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        const transcription = await transcriber(audioBuffer.getChannelData(0), {
            chunk_length_s: 30,
            stride_length_s: 5,
            return_timestamps: true,
        });

        const rawTranscript = (transcription.chunks || []).map((chunk: { text: string }) => chunk.text).join(' ');
        
        if (!rawTranscript.trim()) {
            return { transcript: [], summary: '', action_items: [], outline: '' };
        }

        const apiKey = process.env.GEMINI_API_KEY;
        if (apiKey) {
            progressCallback('Performing advanced analysis & diarization with Gemini...');
            try {
                const ai = new GoogleGenAI({ apiKey });
                const response = await ai.models.generateContent({
                    model: "gemini-3-flash-preview",
                    contents: `
                        Analyze this ${industry} transcript. 
                        1. Perform speaker diarization: Identify different speakers and attribute each part of the text to them.
                        2. Summarize the session.
                        3. Extract action items.
                        4. Create a structured outline.

                        Transcript:
                        ${rawTranscript}
                    `,
                    config: {
                        responseMimeType: "application/json",
                        responseSchema: {
                            type: Type.OBJECT,
                            properties: {
                                transcript: {
                                    type: Type.ARRAY,
                                    items: {
                                        type: Type.OBJECT,
                                        properties: {
                                            speaker: { type: Type.STRING, description: "Name or label of the speaker (e.g., 'Speaker A', 'Dr. Smith')" },
                                            text: { type: Type.STRING }
                                        },
                                        required: ["speaker", "text"]
                                    }
                                },
                                summary: { type: Type.STRING },
                                action_items: {
                                    type: Type.ARRAY,
                                    items: { type: Type.STRING }
                                },
                                outline: { type: Type.STRING }
                            },
                            required: ["transcript", "summary", "action_items", "outline"]
                        }
                    }
                });

                const result = JSON.parse(response.text || '{}');
                return {
                    transcript: result.transcript || [],
                    summary: result.summary || 'No summary generated.',
                    action_items: result.action_items || [],
                    outline: result.outline || 'No outline generated.'
                };
            } catch (err) {
                console.error("Gemini analysis failed, falling back to on-device models:", err);
            }
        }

        // Fallback to on-device models if Gemini fails or no API key
        const transcriptChunks = (transcription.chunks || []).map((chunk: { text: string }) => ({
             speaker: 'Speaker 1', 
             text: chunk.text
        }));

        progressCallback('Initializing analysis model (fallback)...');
        const analyzer = await this.getAnalysisPipeline((p: { status: string; progress?: number }) => {
            if (p.status === 'progress') {
                progressCallback('Downloading analysis model...', p.progress);
            }
        });
        
        progressCallback('Analyzing transcript (fallback)...');
        
        const summaryPrompt = `Summarize this ${industry} transcript: ${rawTranscript}`;
        const summaryResult = await analyzer(summaryPrompt, { max_new_tokens: 128 });
        
        const todoPrompt = `Extract action items from this transcript: ${rawTranscript}`;
        const todoResult = await analyzer(todoPrompt, { max_new_tokens: 128 });
        
        const outlinePrompt = `Create an outline for this transcript: ${rawTranscript}`;
        const outlineResult = await analyzer(outlinePrompt, { max_new_tokens: 128 });

        return {
            transcript: transcriptChunks,
            summary: summaryResult[0].generated_text || 'No summary generated.',
            action_items: (todoResult[0].generated_text || '').split('. ').filter((s: string) => s.trim().length > 0),
            outline: outlineResult[0].generated_text || 'No outline generated.'
        };
    }
}

export const onDeviceAIService = OnDeviceAIService.getInstance();
