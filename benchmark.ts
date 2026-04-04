import { ServerStorageProvider } from './src/services/storageProvider.ts';
import { CryptoService } from './src/services/cryptoService.ts';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

global.window = {
    crypto: crypto.webcrypto,
} as any;
global.btoa = (str: string) => Buffer.from(str, 'binary').toString('base64');
global.atob = (b64: string) => Buffer.from(b64, 'base64').toString('binary');

global.fetch = async (url: string, options?: any) => {
    if (url === '/api/storage/list') {
        const dir = path.join(process.cwd(), 'local_storage');
        if (!fs.existsSync(dir)) return { json: async () => [] } as any;
        const files = fs.readdirSync(dir);
        const items = [];
        for (const file of files) {
            if (file.endsWith('.json')) {
                const id = file.replace('.json', '');
                const content = fs.readFileSync(path.join(dir, file), 'utf-8');
                items.push({ id, data: JSON.parse(content) });
            }
        }
        return { json: async () => items } as any;
    } else if (url.startsWith('/api/storage/item/')) {
        const id = url.split('/').pop();
        const dir = path.join(process.cwd(), 'local_storage');
        const p = path.join(dir, `${id}.json`);
        if (!fs.existsSync(p)) return { ok: false } as any;
        const content = fs.readFileSync(p, 'utf-8');
        return { ok: true, json: async () => ({ id, data: JSON.parse(content) }) } as any;
    }
    return { json: async () => ({}) } as any;
};

async function run() {
    const provider = new ServerStorageProvider('testpin');

    // Benchmark
    const start = Date.now();
    for (let i = 0; i < 5; i++) {
        await provider.getSession(50);
    }
    const end = Date.now();
    console.log(`Optimized time: ${end - start}ms`);
}

run();
