import 'fake-indexeddb/auto';
import 'global-jsdom/register';

global.localStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
  clear: () => {},
  key: () => null,
  length: 0
} as any;
global.window.matchMedia = () => ({ matches: false } as any);

import React from 'react';
import test from 'node:test';
import assert from 'node:assert';
import { render, waitFor, fireEvent } from '@testing-library/react';

// Mock dependencies
import { IndexedDBProvider } from './services/storageProvider';
import { NotesDB } from './services/notesDB';
import { App } from './App';

test('App handles error path when loading data fails', async (t) => {
    // Suppress expected console.error and console.log
    t.mock.method(console, 'error', () => {});
    t.mock.method(console, 'log', () => {});

    // Mock getConfig to not throw
    t.mock.method(NotesDB.prototype, 'getConfig', async () => null);

    // Mock getAllSessions to throw an error
    t.mock.method(IndexedDBProvider.prototype, 'getAllSessions', async () => {
        throw new Error('Test error during getAllSessions');
    });

    const { getByPlaceholderText, getByText, findByText } = render(<App />);

    // Wait for the form to be ready, just in case
    const input = getByPlaceholderText('Enter your PIN');

    // AuthScreen only enables button if pin is entered
    fireEvent.change(input, { target: { value: '123456' } });

    // Submit the form
    const submitBtn = getByText('Unlock');
    fireEvent.click(submitBtn);

    // App should transition to MainApp. MainApp shows a loading spinner briefly, then calls loadData().

    const errorMsg = await findByText('Failed to load data from storage.');
    assert.ok(errorMsg);
    assert.strictEqual(errorMsg.classList.contains('error'), true);
});
