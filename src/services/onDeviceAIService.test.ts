import test from 'node:test';
import assert from 'node:assert';
import { OnDeviceAIService } from './onDeviceAIService'; // drop .ts extension for node:test tsx

test('analyze handles empty string', async (t) => {
    // Mock the private getTranscriptionPipeline method
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    t.mock.method(OnDeviceAIService.prototype as any, 'getTranscriptionPipeline', async () => {
        return async () => ({
            chunks: [{ text: '   ' }] // Spaces that will be empty when trimmed
        });
    });

    const originalWindow = global.window;

    t.after(() => {
        global.window = originalWindow;
    });

    // Mock window.AudioContext required by OnDeviceAIService.analyze
    global.window = {
        AudioContext: class {
            decodeAudioData() {
                return {
                    getChannelData() { return new Float32Array(); }
                };
            }
        }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any;

    const service = OnDeviceAIService.getInstance();

    // Mock Blob with an empty ArrayBuffer
    const blob = {
        arrayBuffer: async () => new ArrayBuffer(0)
    } as Blob;

    // Call the analyze function with mock inputs
    const result = await service.analyze(blob, 'tech', () => {});

    // Assert that the result matches the expected empty object
    assert.deepStrictEqual(result, {
        transcript: [],
        summary: '',
        action_items: [],
        outline: ''
    });
});
