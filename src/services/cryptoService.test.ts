import test from 'node:test';
import assert from 'node:assert';
import { CryptoService } from './cryptoService.ts';

if (typeof window === 'undefined') {
    global.window = { crypto: globalThis.crypto } as any;
}

test('CryptoService.encrypt', async (t) => {
    await t.test('should successfully encrypt a string and return a valid base64 string', async () => {
        const data = "hello world";
        const pin = "1234";
        const encrypted = await CryptoService.encrypt(data, pin);

        assert.ok(typeof encrypted === 'string');
        // Check if it's base64 (very basic check)
        assert.match(encrypted, /^[A-Za-z0-9+/=]+$/);
    });

    await t.test('should produce different ciphertexts for the same data and PIN due to random IV', async () => {
        const data = "hello world";
        const pin = "1234";
        const encrypted1 = await CryptoService.encrypt(data, pin);
        const encrypted2 = await CryptoService.encrypt(data, pin);

        assert.notStrictEqual(encrypted1, encrypted2);
    });

    await t.test('should be able to decrypt the encrypted string back to the original data', async () => {
        const data = "sensitive information 123 !@#";
        const pin = "super_secure_pin";
        const encrypted = await CryptoService.encrypt(data, pin);
        const decrypted = await CryptoService.decrypt(encrypted, pin);

        assert.strictEqual(decrypted, data);
    });

    await t.test('should successfully encrypt and decrypt an empty string', async () => {
        const data = "";
        const pin = "1234";
        const encrypted = await CryptoService.encrypt(data, pin);
        const decrypted = await CryptoService.decrypt(encrypted, pin);

        assert.strictEqual(decrypted, data);
    });
});
