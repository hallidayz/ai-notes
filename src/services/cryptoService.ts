
export class CryptoService {
    private static readonly SALT = 'a-very-secure-static-salt-for-whisper-notes';
    private static readonly ITERATIONS = 100000;
    private static cachedKey: CryptoKey | null = null;
    private static cachedPin: string | null = null;

    private static async deriveKey(pin: string): Promise<CryptoKey> {
        if (this.cachedKey && this.cachedPin === pin) {
            return this.cachedKey;
        }
        const enc = new TextEncoder();
        const keyMaterial = await window.crypto.subtle.importKey(
            'raw',
            enc.encode(pin),
            { name: 'PBKDF2' },
            false,
            ['deriveKey']
        );
        const derivedKey = await window.crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: enc.encode(this.SALT),
                iterations: this.ITERATIONS,
                hash: 'SHA-256',
            },
            keyMaterial,
            { name: 'AES-GCM', length: 256 },
            true,
            ['encrypt', 'decrypt']
        );
        this.cachedKey = derivedKey;
        this.cachedPin = pin;
        return derivedKey;
    }

    public static async encrypt(data: string, pin: string): Promise<string> {
        const key = await this.deriveKey(pin);
        const iv = window.crypto.getRandomValues(new Uint8Array(12));
        const enc = new TextEncoder();
        const encoded = enc.encode(data);
        const encryptedContent = await window.crypto.subtle.encrypt(
            {
                name: 'AES-GCM',
                iv: iv,
            },
            key,
            encoded
        );

        const encryptedBytes = new Uint8Array(iv.length + encryptedContent.byteLength);
        encryptedBytes.set(iv, 0);
        encryptedBytes.set(new Uint8Array(encryptedContent), iv.length);

        return btoa(String.fromCharCode.apply(null, Array.from(encryptedBytes)));
    }

    public static async decrypt(encryptedData: string, pin: string): Promise<string> {
        try {
            const key = await this.deriveKey(pin);
            const encryptedBytes = new Uint8Array(Array.from(atob(encryptedData), c => c.charCodeAt(0)));
            const iv = encryptedBytes.slice(0, 12);
            const encryptedContent = encryptedBytes.slice(12);

            const decryptedContent = await window.crypto.subtle.decrypt(
                {
                    name: 'AES-GCM',
                    iv: iv,
                },
                key,
                encryptedContent
            );

            const dec = new TextDecoder();
            return dec.decode(decryptedContent);
        } catch (e) {
            console.error("Decryption failed:", e);
            throw new Error("Invalid PIN or corrupted data.", { cause: e });
        }
    }

    public static async encryptBuffer(data: ArrayBuffer, pin: string): Promise<string> {
        const key = await this.deriveKey(pin);
        const iv = window.crypto.getRandomValues(new Uint8Array(12));
        const encryptedContent = await window.crypto.subtle.encrypt(
            {
                name: 'AES-GCM',
                iv: iv,
            },
            key,
            data
        );

        const encryptedBytes = new Uint8Array(iv.length + encryptedContent.byteLength);
        encryptedBytes.set(iv, 0);
        encryptedBytes.set(new Uint8Array(encryptedContent), iv.length);

        return btoa(String.fromCharCode.apply(null, Array.from(encryptedBytes)));
    }

    public static async decryptBuffer(encryptedData: string, pin: string): Promise<ArrayBuffer> {
        try {
            const key = await this.deriveKey(pin);
            const encryptedBytes = new Uint8Array(Array.from(atob(encryptedData), c => c.charCodeAt(0)));
            const iv = encryptedBytes.slice(0, 12);
            const encryptedContent = encryptedBytes.slice(12);

            const decryptedContent = await window.crypto.subtle.decrypt(
                {
                    name: 'AES-GCM',
                    iv: iv,
                },
                key,
                encryptedContent
            );

            return decryptedContent;
        } catch (e) {
            console.error("Decryption failed:", e);
            throw new Error("Invalid PIN or corrupted data.", { cause: e });
        }
    }
}
