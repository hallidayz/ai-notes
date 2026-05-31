import test from 'node:test';
import assert from 'node:assert';
import { CryptoService } from './cryptoService.ts';

// Mock window.crypto for Node.js environment
(global as any).window = {
    crypto: globalThis.crypto
};

test('CryptoService', async (t) => {
    const TEST_PIN = '1234';
    const TEST_STRING = 'Hello, this is a secret message!';
    const TEST_BUFFER = new TextEncoder().encode('Hello, this is a secret buffer!').buffer;

    await t.test('encrypt and decrypt a string successfully', async () => {
        const encrypted = await CryptoService.encrypt(TEST_STRING, TEST_PIN);

        // Ensure the encrypted string is different from plain text and does not contain it
        assert.notStrictEqual(encrypted, TEST_STRING);
        assert.ok(!encrypted.includes(TEST_STRING));

        const decrypted = await CryptoService.decrypt(encrypted, TEST_PIN);
        assert.strictEqual(decrypted, TEST_STRING);
    });

    await t.test('encryptBuffer and decryptBuffer successfully', async () => {
        const encrypted = await CryptoService.encryptBuffer(TEST_BUFFER, TEST_PIN);

        const decryptedBuffer = await CryptoService.decryptBuffer(encrypted, TEST_PIN);
        const decryptedString = new TextDecoder().decode(decryptedBuffer);
        const originalString = new TextDecoder().decode(TEST_BUFFER);

        assert.strictEqual(decryptedString, originalString);
    });

    await t.test('decrypt string throws error with incorrect PIN', async () => {
        const encrypted = await CryptoService.encrypt(TEST_STRING, TEST_PIN);

        await assert.rejects(
            async () => {
                await CryptoService.decrypt(encrypted, 'wrong_pin');
            },
            (err: Error) => {
                assert.strictEqual(err.message, 'Invalid PIN or corrupted data.');
                return true;
            }
        );
    });

    await t.test('decrypt string throws error with corrupted data', async () => {
        const encrypted = await CryptoService.encrypt(TEST_STRING, TEST_PIN);

        // Tamper with the encrypted string (change a character)
        const corrupted = encrypted.substring(0, encrypted.length - 1) + (encrypted.endsWith('A') ? 'B' : 'A');

        await assert.rejects(
            async () => {
                await CryptoService.decrypt(corrupted, TEST_PIN);
            },
            (err: Error) => {
                assert.strictEqual(err.message, 'Invalid PIN or corrupted data.');
                return true;
            }
        );
    });

    await t.test('decrypt string throws error with invalid base64 data', async () => {
        await assert.rejects(
            async () => {
                await CryptoService.decrypt('not-base64!', TEST_PIN);
            },
            (err: Error) => {
                assert.strictEqual(err.message, 'Invalid PIN or corrupted data.');
                return true;
            }
        );
    });

    await t.test('decryptBuffer throws error with incorrect PIN', async () => {
        const encrypted = await CryptoService.encryptBuffer(TEST_BUFFER, TEST_PIN);

        await assert.rejects(
            async () => {
                await CryptoService.decryptBuffer(encrypted, 'wrong_pin');
            },
            (err: Error) => {
                assert.strictEqual(err.message, 'Invalid PIN or corrupted data.');
                return true;
            }
        );
    });
});
