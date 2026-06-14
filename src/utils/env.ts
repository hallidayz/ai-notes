/** Reads Gemini API key in browser (Vite) and Node (tests). */
export function getGeminiApiKey(): string | undefined {
    if (typeof process !== 'undefined' && process.env?.GEMINI_API_KEY) {
        return process.env.GEMINI_API_KEY;
    }

    const viteKey = typeof import.meta !== 'undefined' && import.meta.env ? import.meta.env.VITE_GEMINI_API_KEY : undefined;
    if (typeof viteKey === 'string' && viteKey.length > 0) {
        return viteKey;
    }

    return undefined;
}
