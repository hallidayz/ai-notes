import test from 'node:test';
import assert from 'node:assert';
import { OnDeviceAIService } from './onDeviceAIService.ts';

test('OnDeviceAIService analyze should fallback to on-device models when GEMINI_API_KEY is not present', async (t) => {
    const originalWindow = global.window;
    const originalBlob = global.Blob;
    const originalFetch = global.fetch;

    global.fetch = async (url: string) => {
        if (url === '/api/ai/status') {
            return {
                json: async () => ({ hasApiKey: false })
            } as Response;
        }
        return { ok: false } as Response;
    };

    global.window = {
        AudioContext: class {
            decodeAudioData() {
                return {
                    getChannelData: () => new Float32Array(0)
                };
            }
        }
    } as any;

    global.Blob = class {
        arrayBuffer() {
            return new ArrayBuffer(0);
        }
    } as any;

    const mockedPipeline = async (task: string) => {
        if (task === 'automatic-speech-recognition') {
            return async () => {
                return {
                    chunks: [
                        { text: ' Hello' },
                        { text: ' World' }
                    ]
                };
            };
        } else if (task === 'text2text-generation') {
            return async (prompt: string) => {
                if (prompt.includes('Summarize')) {
                    return [{ generated_text: 'Dummy Summary' }];
                }
                if (prompt.includes('Extract action items')) {
                    return [{ generated_text: 'Dummy Action 1. Dummy Action 2.' }];
                }
                if (prompt.includes('outline')) {
                    return [{ generated_text: 'Dummy Outline' }];
                }
                return [{ generated_text: 'Unknown' }];
            };
        }
        return async () => {};
    };

    (OnDeviceAIService as any).instance = null;
    const service = OnDeviceAIService.getInstance();

    t.mock.method(service, 'getTranscriptionPipeline', async () => await mockedPipeline('automatic-speech-recognition'));
    t.mock.method(service, 'getAnalysisPipeline', async () => await mockedPipeline('text2text-generation'));

    const progressLogs: string[] = [];
    const result = await service.analyze(new global.Blob(), 'Tech', (status: string) => {
        progressLogs.push(status);
    });

    assert.strictEqual(result.summary, 'Dummy Summary');
    assert.strictEqual(result.outline, 'Dummy Outline');
    assert.deepStrictEqual(result.action_items, ['Dummy Action 1', 'Dummy Action 2.']);
    assert.deepStrictEqual(result.transcript, [
        { speaker: 'Speaker 1', text: ' Hello' },
        { speaker: 'Speaker 1', text: ' World' }
    ]);

    // Restore global state
    global.window = originalWindow;
    global.Blob = originalBlob;
    global.fetch = originalFetch;
});

test('OnDeviceAIService analyze should fallback to on-device models when Gemini API fails', async (t) => {
    const originalWindow = global.window;
    const originalBlob = global.Blob;
    const originalFetch = global.fetch;

    global.window = {
        AudioContext: class {
            decodeAudioData() {
                return {
                    getChannelData: () => new Float32Array(0)
                };
            }
        }
    } as any;

    global.Blob = class {
        arrayBuffer() {
            return new ArrayBuffer(0);
        }
    } as any;

    // Mock fetch to simulate Gemini API failure
    global.fetch = async (url: string | URL | Request) => {
        if (url === '/api/ai/status') {
            return {
                json: async () => ({ hasApiKey: true })
            } as Response;
        }
        throw new Error('Network Error');
    };

    const mockedPipeline = async (task: string) => {
        if (task === 'automatic-speech-recognition') {
            return async () => {
                return {
                    chunks: [
                        { text: ' Hello' },
                        { text: ' World' }
                    ]
                };
            };
        } else if (task === 'text2text-generation') {
            return async (prompt: string) => {
                if (prompt.includes('Summarize')) {
                    return [{ generated_text: 'Dummy Summary Fallback' }];
                }
                if (prompt.includes('Extract action items')) {
                    return [{ generated_text: 'Fallback Action 1. Fallback Action 2.' }];
                }
                if (prompt.includes('outline')) {
                    return [{ generated_text: 'Dummy Outline Fallback' }];
                }
                return [{ generated_text: 'Unknown' }];
            };
        }
        return async () => {};
    };

    const originalConsoleError = console.error;
    console.error = () => {};

    (OnDeviceAIService as any).instance = null;
    const service = OnDeviceAIService.getInstance();

    t.mock.method(service, 'getTranscriptionPipeline', async () => await mockedPipeline('automatic-speech-recognition'));
    t.mock.method(service, 'getAnalysisPipeline', async () => await mockedPipeline('text2text-generation'));

    const progressLogs: string[] = [];
    const result = await service.analyze(new global.Blob(), 'Tech', (status: string) => {
        progressLogs.push(status);
    });

    assert.strictEqual(result.summary, 'Dummy Summary Fallback');
    assert.strictEqual(result.outline, 'Dummy Outline Fallback');
    assert.deepStrictEqual(result.action_items, ['Fallback Action 1', 'Fallback Action 2.']);
    assert.deepStrictEqual(result.transcript, [
        { speaker: 'Speaker 1', text: ' Hello' },
        { speaker: 'Speaker 1', text: ' World' }
    ]);

    // Restore global state
    console.error = originalConsoleError;
    global.window = originalWindow;
    global.Blob = originalBlob;
    global.fetch = originalFetch;
});
