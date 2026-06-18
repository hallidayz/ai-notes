import { createRequire } from 'module';

const require = createRequire(import.meta.url);

if (process.env.NODE_ENV !== 'production') {
    try {
        const { loadEnv } = require('vite') as typeof import('vite');
        const env = loadEnv(process.env.NODE_ENV ?? 'development', process.cwd(), '');
        for (const [key, value] of Object.entries(env)) {
            if (process.env[key] === undefined) {
                process.env[key] = value;
            }
        }
    } catch {
        // Vite not available outside dev tooling.
    }
}
