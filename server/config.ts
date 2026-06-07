import os from 'os';
import { loadEnv } from 'vite';

const env = loadEnv(process.env.NODE_ENV ?? 'development', process.cwd(), '');

export const HOST = process.env.HOST ?? env.HOST ?? '0.0.0.0';
export const PORT = Number(process.env.PORT ?? env.PORT ?? 4783);

export function getLocalNetworkAddresses(): string[] {
    const addresses: string[] = [];
    const interfaces = os.networkInterfaces();

    for (const ifaces of Object.values(interfaces)) {
        for (const iface of ifaces ?? []) {
            if (iface.family === 'IPv4' && !iface.internal) {
                addresses.push(iface.address);
            }
        }
    }

    return addresses;
}

export function getServerUrls(): { local: string; network: string[] } {
    const network = getLocalNetworkAddresses().map((ip) => `http://${ip}:${PORT}`);
    return {
        local: `http://localhost:${PORT}`,
        network,
    };
}
