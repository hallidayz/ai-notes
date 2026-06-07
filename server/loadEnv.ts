import { loadEnv } from 'vite';

const env = loadEnv(process.env.NODE_ENV ?? 'development', process.cwd(), '');

for (const [key, value] of Object.entries(env)) {
    if (process.env[key] === undefined) {
        process.env[key] = value;
    }
}
