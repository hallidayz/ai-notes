import 'fake-indexeddb/auto';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!DOCTYPE html><html><head></head><body></body></html>', {
    url: 'http://localhost/',
});
global.window = dom.window as any;
global.document = dom.window.document as any;
Object.defineProperty(global, 'navigator', {
    value: dom.window.navigator,
    writable: true
});
global.localStorage = {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
    clear: () => {},
    length: 0,
    key: () => null
} as any;
Object.defineProperty(global.window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => {}, // Deprecated
        removeListener: () => {}, // Deprecated
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
    }),
});
// Avoid "attachEvent is not a function" in react-dom with JSDOM
Object.defineProperty(global.window.HTMLElement.prototype, 'attachEvent', {
    value: () => {},
    configurable: true,
    writable: true
});

import test from 'node:test';
import assert from 'node:assert';
import React from 'react';
import { render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from './App';
import { IndexedDBProvider } from './services/storageProvider';
import { NotesDB } from './services/notesDB';

test('App handles error when updating session', async (t) => {
    // Suppress expected console.error in test output
    t.mock.method(console, 'error', () => {});

    t.mock.method(NotesDB.prototype, 'getConfig', async () => null);

    t.mock.method(IndexedDBProvider.prototype, 'getAllTasks', async () => []);
    t.mock.method(IndexedDBProvider.prototype, 'getAllSessions', async () => [
        {
            id: 1,
            sessionTitle: 'Test Session',
            date: Date.now(),
            participants: '',
            notes: 'Test Notes', // Need to mock CryptoService if it attempts decryption
            timestamp: Date.now(),
            analysisStatus: 'none'
        }
    ]);
    t.mock.method(IndexedDBProvider.prototype, 'getAudioBlob', async () => null);

    // Mock CryptoService to avoid dealing with real crypto in this test
    const { CryptoService } = await import('./services/cryptoService');
    t.mock.method(CryptoService, 'decrypt', async (data: string) => data);
    t.mock.method(CryptoService, 'encrypt', async (data: string) => data);

    // The method to throw error
    t.mock.method(IndexedDBProvider.prototype, 'updateSession', async () => {
        throw new Error('Mock update error');
    });

    const { getByPlaceholderText, getByRole, findByText } = render(<App />);
    const user = userEvent.setup({ document: dom.window.document as any });

    // Auth screen
    const pinInput = getByPlaceholderText('Enter your PIN');
    await user.type(pinInput, '1234');

    const unlockBtn = getByRole('button', { name: 'Unlock' });
    await user.click(unlockBtn);

    // Wait for session list to load
    const sessionCard = await findByText('Test Session');

    // Click on session to open SessionDetailModal
    await user.click(sessionCard);

    // Open notes editor
    const notesEl = await findByText('Test Notes');
    await user.click(notesEl);

    // Click Save to trigger update
    const saveBtn = await findByText('Save');
    await user.click(saveBtn);

    // Check that error message appeared
    const errorStatus = await findByText('Failed to update session.');
    assert.ok(errorStatus);
});

test.after(() => {
    // Teardown
    globalThis.indexedDB.deleteDatabase('ai_notes_db');
});
