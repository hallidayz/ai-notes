/** Reads Gemini API key in browser (Vite) and Node (tests). */
export function getGeminiApiKey(): string | undefined {
    if (typeof process !== 'undefined' && process.env?.GEMINI_API_KEY) {
        return process.env.GEMINI_API_KEY;
    }

    try {
        // @ts-expect-error import.meta.env may not be defined in Node
        const viteKey = import.meta.env?.VITE_GEMINI_API_KEY;
        if (typeof viteKey === 'string' && viteKey.length > 0) {
            return viteKey;
        }
    } catch {
        // import.meta.env might not be defined in Node
    }

    return undefined;
}
