import { test } from 'node:test';
import * as assert from 'node:assert';
import { CryptoService } from './cryptoService.ts';

// Mock window.crypto for Node.js environment
if (typeof global.window === 'undefined') {
    (global as any).window = { crypto: globalThis.crypto };
} else if (!global.window.crypto) {
    global.window.crypto = globalThis.crypto;
}

test('CryptoService decrypt invalid PIN error', async (t) => {
    let originalConsoleError: any;

    t.beforeEach(() => {
        originalConsoleError = console.error;
        console.error = () => {}; // suppress expected error log
    });

    t.afterEach(() => {
        console.error = originalConsoleError;
    });

    await t.test('decrypting with an incorrect PIN should throw an error', async () => {
        const secretData = 'This is my secret data';
        const validPin = '1234';
        const invalidPin = '0000';

        const encryptedData = await CryptoService.encrypt(secretData, validPin);

        await assert.rejects(
            async () => {
                await CryptoService.decrypt(encryptedData, invalidPin);
            },
            (err: Error) => {
                assert.strictEqual(err.message, 'Invalid PIN or corrupted data.');
                return true;
            }
        );
    });
});