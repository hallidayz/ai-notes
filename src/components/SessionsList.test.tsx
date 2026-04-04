import test from 'node:test';
import assert from 'node:assert';
import React from 'react';
import { JSDOM } from 'jsdom';

// Setup global DOM environment BEFORE testing library imports
const dom = new JSDOM('<!DOCTYPE html><html><head></head><body><div id="root"></div></body></html>', {
  url: 'http://localhost/',
});

const globalObj = globalThis as any;
globalObj.window = dom.window;

// Using Object.defineProperty to bypass any getters on global context
['document', 'navigator', 'HTMLElement', 'Event', 'CustomEvent'].forEach(prop => {
    Object.defineProperty(globalObj, prop, {
        value: (dom.window as any)[prop],
        configurable: true,
        writable: true
    });
});
globalObj.IS_REACT_ACT_ENVIRONMENT = true;

// Import AFTER DOM is set up
import { render } from '@testing-library/react';
import { SessionsList } from './SessionsList';
import { Session } from '../types';

test('SessionsList renders and handles preview decryption errors correctly', () => {
    // 1. Arrange: Create mock session with invalid encrypted base64 data
    const mockSessions: Session[] = [
        {
            id: 1,
            sessionTitle: 'Test Session',
            date: '2023-10-27T10:00:00Z',
            duration: 60,
            timestamp: 1698400800000,
            transcript: [],
            // Invalid base64 data to force a decryption failure
            notes: 'invalid-base64-data!@#'
        }
    ];

    const mockOnSelect = () => {};
    const mockOnDelete = () => {};

    // 2. Act: Render the SessionsList component
    const { getByText } = render(
        <SessionsList
            sessions={mockSessions}
            onSelect={mockOnSelect}
            onDelete={mockOnDelete}
        />
    );

    // 3. Assert: Verify the fallback preview text is displayed
    const previewElement = getByText('Could not decrypt preview.');
    assert.ok(previewElement, 'Should render the fallback text when decryption fails');
});
