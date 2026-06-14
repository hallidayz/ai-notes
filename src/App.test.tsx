import test from 'node:test';
import assert from 'node:assert';
import React from 'react';

import { JSDOM } from 'jsdom';
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="root"></div></body></html>', {
  url: 'http://localhost'
});
global.window = dom.window as any;
global.document = dom.window.document as any;
Object.defineProperty(global, 'navigator', {
  value: dom.window.navigator,
});
global.localStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
  clear: () => {},
  length: 0,
  key: () => null,
} as any;
(global.window as any).matchMedia = () => ({
  matches: false,
  addListener: () => {},
  removeListener: () => {},
});
global.Element = dom.window.Element;
global.Node = dom.window.Node;
global.Event = dom.window.Event;
(dom.window.HTMLInputElement.prototype as any).attachEvent = function() {};

import { render, act } from '@testing-library/react';
import 'fake-indexeddb/auto';

import { MainApp } from './App';
import { NotesDB } from './services/notesDB';

test('App error path', async (t) => {
    t.mock.method(NotesDB.prototype, 'getConfig', async () => {
        throw new Error("Simulated db error");
    });

    const consoleErrorMock = t.mock.method(console, 'error', () => {});

    // Render MainApp directly to bypass AuthScreen entirely
    render(<MainApp pin="1234" isDarkMode={false} onToggleTheme={() => {}} />);

    // Wait for the simulated async error to occur
    await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
    });

    const calls = consoleErrorMock.mock.calls;
    const errorCall = calls.find(call => call.arguments[0] === "Error loading initial config:");
    assert.ok(errorCall, "Expected console.error to be called with 'Error loading initial config:'");
    assert.strictEqual(errorCall.arguments[1].message, "Simulated db error");
});
