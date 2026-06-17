
export class CryptoService {
    private static readonly ITERATIONS = 100000;
    private static readonly SALT_LENGTH = 16;
    private static readonly IV_LENGTH = 12;
    private static readonly MARKER = new Uint8Array([0x57, 0x4e, 0x53, 0x31]);

    private static async deriveKey(pin: string, salt: Uint8Array): Promise<CryptoKey> {
        const enc = new TextEncoder();
        const keyMaterial = await window.crypto.subtle.importKey(
            'raw',
            enc.encode(pin),
            { name: 'PBKDF2' },
            false,
            ['deriveKey']
        );
        return window.crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt,
                iterations: this.ITERATIONS,
                hash: 'SHA-256',
            },
            keyMaterial,
            { name: 'AES-GCM', length: 256 },
            true,
            ['encrypt', 'decrypt']
        );
    }

    private static encodeBytes(encryptedBytes: Uint8Array): string {
        let binary = '';
        for (let i = 0; i < encryptedBytes.length; i++) {
            binary += String.fromCharCode(encryptedBytes[i]);
        }
        return btoa(binary);
    }

    private static decodeBytes(encryptedData: string): Uint8Array {
        const binaryString = atob(encryptedData);
        const encryptedBytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            encryptedBytes[i] = binaryString.charCodeAt(i);
        }
        return encryptedBytes;
    }

    private static parseEncryptedPayload(encryptedBytes: Uint8Array): { salt: Uint8Array; iv: Uint8Array; encryptedContent: Uint8Array } {
        const isNewFormat = encryptedBytes.length >= this.MARKER.length &&
            this.MARKER.every((byte, i) => encryptedBytes[i] === byte);

        if (!isNewFormat) {
            throw new Error("Unsupported legacy encryption format.");
        }

        return {
            salt: encryptedBytes.slice(this.MARKER.length, this.MARKER.length + this.SALT_LENGTH),
            iv: encryptedBytes.slice(this.MARKER.length + this.SALT_LENGTH, this.MARKER.length + this.SALT_LENGTH + this.IV_LENGTH),
            encryptedContent: encryptedBytes.slice(this.MARKER.length + this.SALT_LENGTH + this.IV_LENGTH),
        };
    }

    private static packEncryptedPayload(salt: Uint8Array, iv: Uint8Array, encryptedContent: ArrayBuffer): string {
        const encryptedBytes = new Uint8Array(this.MARKER.length + salt.length + iv.length + encryptedContent.byteLength);
        encryptedBytes.set(this.MARKER, 0);
        encryptedBytes.set(salt, this.MARKER.length);
        encryptedBytes.set(iv, this.MARKER.length + salt.length);
        encryptedBytes.set(new Uint8Array(encryptedContent), this.MARKER.length + salt.length + iv.length);
        return this.encodeBytes(encryptedBytes);
    }

    public static async encrypt(data: string, pin: string): Promise<string> {
        const salt = window.crypto.getRandomValues(new Uint8Array(this.SALT_LENGTH));
        const key = await this.deriveKey(pin, salt);
        const iv = window.crypto.getRandomValues(new Uint8Array(this.IV_LENGTH));
        const enc = new TextEncoder();
        const encryptedContent = await window.crypto.subtle.encrypt(
            { name: 'AES-GCM', iv },
            key,
            enc.encode(data)
        );
        return this.packEncryptedPayload(salt, iv, encryptedContent);
    }

    public static async decrypt(encryptedData: string, pin: string): Promise<string> {
        try {
            const encryptedBytes = this.decodeBytes(encryptedData);
            const { salt, iv, encryptedContent } = this.parseEncryptedPayload(encryptedBytes);
            const key = await this.deriveKey(pin, salt);
            const decryptedContent = await window.crypto.subtle.decrypt(
                { name: 'AES-GCM', iv },
                key,
                encryptedContent
            );
            return new TextDecoder().decode(decryptedContent);
        } catch (e) {
            console.error("Decryption failed:", e);
            throw new Error("Invalid PIN or corrupted data.", { cause: e });
        }
    }

    public static async encryptBuffer(data: ArrayBuffer, pin: string): Promise<string> {
        const salt = window.crypto.getRandomValues(new Uint8Array(this.SALT_LENGTH));
        const key = await this.deriveKey(pin, salt);
        const iv = window.crypto.getRandomValues(new Uint8Array(this.IV_LENGTH));
        const encryptedContent = await window.crypto.subtle.encrypt(
            { name: 'AES-GCM', iv },
            key,
            data
        );
        return this.packEncryptedPayload(salt, iv, encryptedContent);
    }

    public static async decryptBuffer(encryptedData: string, pin: string): Promise<ArrayBuffer> {
        try {
            const encryptedBytes = this.decodeBytes(encryptedData);
            const { salt, iv, encryptedContent } = this.parseEncryptedPayload(encryptedBytes);
            const key = await this.deriveKey(pin, salt);
            return window.crypto.subtle.decrypt(
                { name: 'AES-GCM', iv },
                key,
                encryptedContent
            );
        } catch (e) {
            console.error("Decryption failed:", e);
            throw new Error("Invalid PIN or corrupted data.", { cause: e });
        }
    }
}
