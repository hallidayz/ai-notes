import { test } from 'node:test';
import assert from 'node:assert';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', { url: 'http://localhost' });
global.window = dom.window as any;
global.document = dom.window.document as any;
if (typeof global.navigator === 'undefined') {
    global.navigator = dom.window.navigator as any;
} else {
    Object.defineProperty(global, 'navigator', {
        value: dom.window.navigator,
        configurable: true,
        writable: true
    });
}
global.HTMLElement = dom.window.HTMLElement as any;
global.IS_REACT_ACT_ENVIRONMENT = true;
global.requestAnimationFrame = (cb) => setTimeout(cb, 0);

// Polyfill localStorage
const localStorageMock = (function () {
  let store: Record<string, string> = {};
  return {
    getItem(key: string) {
      return store[key] || null;
    },
    setItem(key: string, value: string) {
      store[key] = value.toString();
    },
    removeItem(key: string) {
      delete store[key];
    },
    clear() {
      store = {};
    }
  };
})();
Object.defineProperty(global, 'localStorage', { value: localStorageMock });
Object.defineProperty(global.window, 'localStorage', { value: localStorageMock });

Object.defineProperty(global.window, 'matchMedia', {
    writable: true,
    value: () => ({
        matches: false,
        addListener: () => {},
        removeListener: () => {}
    })
});

// Suppress known act/attachEvent warnings from console.error entirely for the scope of the file
const originalConsoleError = console.error;
console.error = (...args: any[]) => {
    const msg = args[0] instanceof Error ? args[0].message : args[0];
    if (typeof msg === 'string' && (msg.includes('was not wrapped in act') || msg.includes('activeElement$1') || msg.includes('support act(...)'))) {
        return;
    }
    originalConsoleError(...args);
};

import React from 'react';
import { render, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import 'fake-indexeddb/auto';

import { IndexedDBProvider } from './services/storageProvider';
import { App } from './App';

test('App.tsx - loadData error path shows status message and logs error', async (t) => {
    // 1. Setup mock console.error for assertions
    const testConsoleError = console.error;
    let consoleErrorCalled = false;
    let caughtError: any = null;

    console.error = (msg: any, err: any, ...args: any[]) => {
        if (typeof msg === 'string' && msg.includes('Error loading data:')) {
            consoleErrorCalled = true;
            caughtError = err;
        } else {
            testConsoleError(msg, err, ...args);
        }
    };

    // 2. Mock storage provider to force an error on load
    const originalGetAllSessions = IndexedDBProvider.prototype.getAllSessions;
    IndexedDBProvider.prototype.getAllSessions = async () => {
        throw new Error('Forced load error');
    };

    try {
        const { getByPlaceholderText, getByRole, getByText } = render(<App />);

        // 3. Authenticate to enter MainApp where loadData is called
        const passwordInput = getByPlaceholderText(/Enter your PIN/i);
        const user = userEvent.setup({ document: global.document });

        await act(async () => {
            await user.type(passwordInput, '1234');
            const loginButton = getByRole('button', { name: /Unlock/i });
            await user.click(loginButton);
        });

        // 4. Wait for the status message to be displayed as a result of the forced error
        await waitFor(() => {
            const statusMessage = getByText('Failed to load data from storage.');
            assert.ok(statusMessage);
        });

        // 5. Verify the error was logged properly
        assert.strictEqual(consoleErrorCalled, true);
        assert.ok(caughtError);
        assert.strictEqual(caughtError.message, 'Forced load error');

    } finally {
        // 6. Restore mocks
        console.error = testConsoleError;
        IndexedDBProvider.prototype.getAllSessions = originalGetAllSessions;
    }
});
