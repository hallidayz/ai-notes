import test from 'node:test';
import assert from 'node:assert';
import { CryptoService } from '../cryptoService.ts';

// Mock window.crypto since CryptoService uses window.crypto
if (typeof window === 'undefined') {
    (global as any).window = { crypto: globalThis.crypto };
}

test('CryptoService.decrypt', async (t) => {
    // Suppress console.error during these tests to keep output clean,
    // since CryptoService.decrypt logs an error before throwing.
    const originalConsoleError = console.error;

    t.beforeEach(() => {
        console.error = () => {};
    });

    t.afterEach(() => {
        console.error = originalConsoleError;
    });

    await t.test('decrypts correctly with valid pin', async () => {
        const pin = "1234";
        const originalText = "hello world";
        const encrypted = await CryptoService.encrypt(originalText, pin);

        const decrypted = await CryptoService.decrypt(encrypted, pin);
        assert.strictEqual(decrypted, originalText);
    });

    await t.test('throws error with invalid pin', async () => {
        const originalText = "hello world";
        const encrypted = await CryptoService.encrypt(originalText, "1234");

        await assert.rejects(
            async () => {
                await CryptoService.decrypt(encrypted, "wrong-pin");
            },
            (err: Error) => {
                return err.message === 'Invalid PIN or corrupted data.';
            }
        );
    });

    await t.test('throws error with corrupted data', async () => {
        const pin = "1234";
        const originalText = "hello world";
        const encrypted = await CryptoService.encrypt(originalText, pin);

        // Corrupt the data by changing a character
        const corrupted = encrypted.substring(0, encrypted.length - 1) + (encrypted.endsWith('a') ? 'b' : 'a');

        await assert.rejects(
            async () => {
                await CryptoService.decrypt(corrupted, pin);
            },
            (err: Error) => {
                return err.message === 'Invalid PIN or corrupted data.';
            }
        );
    });

    await t.test('throws error with invalid base64 data', async () => {
        const pin = "1234";
        const invalidBase64 = "this-is-not-base64!";

        await assert.rejects(
            async () => {
                await CryptoService.decrypt(invalidBase64, pin);
            },
            (err: Error) => {
                return err.message === 'Invalid PIN or corrupted data.';
            }
        );
    });

    await t.test('throws error with old format legacy data', async () => {
        const pin = "1234";
        // Create an old format payload: base64 encoded [IV (12 bytes) + Encrypted Content]
        // This lacks the 4-byte MARKER and 16-byte SALT at the beginning.
        // We will just create a valid-looking base64 string that is not the new format.
        const legacyDataBytes = new Uint8Array(20); // Just dummy data
        const legacyBase64 = btoa(String.fromCharCode(...legacyDataBytes));

        await assert.rejects(
            async () => {
                await CryptoService.decrypt(legacyBase64, pin);
            },
            (err: Error) => {
                return err.message === 'Invalid PIN or corrupted data.';
            }
        );
    });
});
