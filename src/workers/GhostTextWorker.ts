import { pipeline, env } from '@huggingface/transformers';

// Configure transformers environment for the browser worker
env.allowLocalModels = false;

let generator: any = null;
let isInitializing = false;

async function initialize() {
    if (generator || isInitializing) return;

    isInitializing = true;
    try {
        // Using distilgpt2 which is relatively small and fast for the browser
        generator = await pipeline('text-generation', 'Xenova/distilgpt2', {
            quantized: true
        });

        self.postMessage({ type: 'init_complete', status: 'success' });
    } catch (error) {
        console.error('Failed to initialize ghost text AI model in worker', error);
        self.postMessage({ type: 'init_complete', status: 'error', error: String(error) });
    } finally {
        isInitializing = false;
    }
}

async function generateSuggestion(context: string, messageId: string) {
    if (!context || context.trim() === '') {
        self.postMessage({ type: 'generation_complete', messageId, suggestion: '' });
        return;
    }

    if (!generator) {
        if (!isInitializing) {
            await initialize();
        }
        if (!generator) {
            self.postMessage({ type: 'generation_complete', messageId, suggestion: '' });
            return;
        }
    }

    try {
        // Take the last ~15 words to maintain context but keep inference fast
        const words = context.trim().split(/\s+/);
        const promptText = words.slice(-15).join(' ');

        // Generate completion
        const result = await generator(promptText, {
            max_new_tokens: 5,
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.2,
            do_sample: true,
            return_full_text: false,
        });

        if (result && result.length > 0 && result[0].generated_text) {
            let text = result[0].generated_text as string;

            // Clean up the output
            // Only take the first sentence or phrase until a punctuation mark (or newline)
            const match = text.match(/^[^.!?\n]*[.!?]?/);
            if (match) {
                text = match[0];
            }

            // Trim leading whitespace but ensure we don't start with punctuation unless appropriate
            if (/^\s/.test(text)) {
                text = ' ' + text.trim();
            }

            // If output starts directly with words and prompt ends with a word character, might need space
            if (/[a-zA-Z0-9]$/.test(promptText) && /^[a-zA-Z0-9]/.test(text)) {
                text = ' ' + text;
            }

            self.postMessage({ type: 'generation_complete', messageId, suggestion: text });
            return;
        }
        self.postMessage({ type: 'generation_complete', messageId, suggestion: '' });
    } catch (error) {
        console.error('Ghost text generation failed in worker', error);
        self.postMessage({ type: 'generation_complete', messageId, suggestion: '' });
    }
}

// Listen for messages from the main thread
self.addEventListener('message', (event) => {
    const data = event.data;

    if (data.type === 'init') {
        initialize();
    } else if (data.type === 'generate') {
        generateSuggestion(data.context, data.messageId);
    }
});
