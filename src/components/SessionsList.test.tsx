import React from 'react';
import { render } from '@testing-library/react';
import { SessionsList } from './SessionsList';
import { Session } from '../types';
import test from 'node:test';
import assert from 'node:assert';
import { JSDOM } from 'jsdom';

// Setup JSDOM
const dom = new JSDOM('<!DOCTYPE html><html><head></head><body></body></html>', {
  url: 'http://localhost'
});
global.window = dom.window as any;
global.document = dom.window.document;
Object.defineProperty(global, 'navigator', { value: dom.window.navigator, configurable: true, writable: true });
(global as any).IS_REACT_ACT_ENVIRONMENT = true;

// Also for @testing-library/dom screen to work:
global.HTMLElement = dom.window.HTMLElement;
global.Node = dom.window.Node;

test('SessionsList handles missing notes in preview (simulated decryption error)', () => {
    // Clear dom before render
    document.body.innerHTML = '';

    const mockSession: Session = {
        id: 1,
        sessionTitle: 'Test Session',
        date: new Date().toISOString(),
        notes: 'invalid_base64_data_that_will_throw-_!@#$%^&*()', // no spaces, invalid characters for atob
        duration: 0,
        transcript: [],
        timestamp: Date.now()
    };

    const { getByText } = render(
        <SessionsList
            sessions={[mockSession]}
            onSelect={() => {}}
            onDelete={() => {}}
        />
    );

    const previewElement = getByText('Could not decrypt preview.');
    assert.ok(previewElement, 'Fallback text should be rendered when decryption fails');
});
