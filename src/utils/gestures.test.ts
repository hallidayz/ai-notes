import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert';
import { useSwipe, triggerHaptic } from './gestures.ts';
import { renderHook } from '@testing-library/react';

// Setup JSDOM
import { JSDOM } from 'jsdom';

// Polyfill window and document
const jsdom = new JSDOM('<!doctype html><html><body></body></html>');
(global as any).window = jsdom.window;
(global as any).document = jsdom.window.document;
Object.defineProperty(global, 'navigator', {
    value: { userAgent: 'node.js' },
    configurable: true,
    writable: true
});
// Polyfill react act environment
(global as any).IS_REACT_ACT_ENVIRONMENT = true;

describe('useSwipe', () => {
    it('should detect a left swipe', () => {
        let leftSwiped = false;
        const { result } = renderHook(() => useSwipe(
            () => { leftSwiped = true; },
            undefined, undefined, undefined, 50
        ));

        // Start touch at x: 100, y: 100
        result.current.onTouchStart({ targetTouches: [{ clientX: 100, clientY: 100 }] } as any);

        // Move to x: 40, y: 100 (distanceX = 100 - 40 = 60 > 50)
        result.current.onTouchMove({ targetTouches: [{ clientX: 40, clientY: 100 }] } as any);

        result.current.onTouchEnd();

        assert.strictEqual(leftSwiped, true, 'Left swipe should be detected');
    });

    it('should detect a right swipe', () => {
        let rightSwiped = false;
        const { result } = renderHook(() => useSwipe(
            undefined,
            () => { rightSwiped = true; },
            undefined, undefined, 50
        ));

        // Start touch at x: 100, y: 100
        result.current.onTouchStart({ targetTouches: [{ clientX: 100, clientY: 100 }] } as any);

        // Move to x: 160, y: 100 (distanceX = 100 - 160 = -60 < -50)
        result.current.onTouchMove({ targetTouches: [{ clientX: 160, clientY: 100 }] } as any);

        result.current.onTouchEnd();

        assert.strictEqual(rightSwiped, true, 'Right swipe should be detected');
    });

    it('should detect an up swipe', () => {
        let upSwiped = false;
        const { result } = renderHook(() => useSwipe(
            undefined, undefined,
            () => { upSwiped = true; },
            undefined, 50
        ));

        // Start touch at x: 100, y: 100
        result.current.onTouchStart({ targetTouches: [{ clientX: 100, clientY: 100 }] } as any);

        // Move to x: 100, y: 40 (distanceY = 100 - 40 = 60 > 50)
        result.current.onTouchMove({ targetTouches: [{ clientX: 100, clientY: 40 }] } as any);

        result.current.onTouchEnd();

        assert.strictEqual(upSwiped, true, 'Up swipe should be detected');
    });

    it('should detect a down swipe', () => {
        let downSwiped = false;
        const { result } = renderHook(() => useSwipe(
            undefined, undefined, undefined,
            () => { downSwiped = true; },
            50
        ));

        // Start touch at x: 100, y: 100
        result.current.onTouchStart({ targetTouches: [{ clientX: 100, clientY: 100 }] } as any);

        // Move to x: 100, y: 160 (distanceY = 100 - 160 = -60 < -50)
        result.current.onTouchMove({ targetTouches: [{ clientX: 100, clientY: 160 }] } as any);

        result.current.onTouchEnd();

        assert.strictEqual(downSwiped, true, 'Down swipe should be detected');
    });

    it('should not trigger if distance is below threshold', () => {
        let leftSwiped = false;
        const { result } = renderHook(() => useSwipe(
            () => { leftSwiped = true; },
            undefined, undefined, undefined, 50
        ));

        // Start touch at x: 100, y: 100
        result.current.onTouchStart({ targetTouches: [{ clientX: 100, clientY: 100 }] } as any);

        // Move to x: 60, y: 100 (distanceX = 100 - 60 = 40 <= 50)
        result.current.onTouchMove({ targetTouches: [{ clientX: 60, clientY: 100 }] } as any);

        result.current.onTouchEnd();

        assert.strictEqual(leftSwiped, false, 'Swipe should not be detected if below threshold');
    });

    it('should prioritize dominant axis on diagonal swipes', () => {
        let leftSwiped = false;
        let upSwiped = false;
        const { result } = renderHook(() => useSwipe(
            () => { leftSwiped = true; },
            undefined,
            () => { upSwiped = true; },
            undefined, 50
        ));

        // Start touch at x: 100, y: 100
        result.current.onTouchStart({ targetTouches: [{ clientX: 100, clientY: 100 }] } as any);

        // Move to x: 20, y: 40 (distanceX = 80, distanceY = 60). X > Y, so should trigger left swipe.
        result.current.onTouchMove({ targetTouches: [{ clientX: 20, clientY: 40 }] } as any);

        result.current.onTouchEnd();

        assert.strictEqual(leftSwiped, true, 'Left swipe should be dominant');
        assert.strictEqual(upSwiped, false, 'Up swipe should not trigger');
    });

    it('should do nothing if touchEnd is missing', () => {
        let leftSwiped = false;
        const { result } = renderHook(() => useSwipe(
            () => { leftSwiped = true; }
        ));

        result.current.onTouchStart({ targetTouches: [{ clientX: 100, clientY: 100 }] } as any);
        // No onTouchMove
        result.current.onTouchEnd();

        assert.strictEqual(leftSwiped, false, 'Should not throw or trigger callback without touchMove/touchEnd state');
    });

    it('should do nothing if touchStart is missing', () => {
        let leftSwiped = false;
        const { result } = renderHook(() => useSwipe(
            () => { leftSwiped = true; }
        ));

        // No onTouchStart
        result.current.onTouchMove({ targetTouches: [{ clientX: 40, clientY: 100 }] } as any);
        result.current.onTouchEnd();

        assert.strictEqual(leftSwiped, false, 'Should not throw or trigger callback without touchStart state');
    });

    it('should use default threshold if not provided', () => {
        let leftSwiped = false;
        const { result } = renderHook(() => useSwipe(
            () => { leftSwiped = true; }
            // default is 50
        ));

        result.current.onTouchStart({ targetTouches: [{ clientX: 100, clientY: 100 }] } as any);
        result.current.onTouchMove({ targetTouches: [{ clientX: 49, clientY: 100 }] } as any); // distance = 51 > 50
        result.current.onTouchEnd();

        assert.strictEqual(leftSwiped, true, 'Default threshold of 50 should be used');
    });
});

describe('triggerHaptic', () => {
    let originalNavigator: any;

    beforeEach(() => {
        originalNavigator = global.navigator;
    });

    afterEach(() => {
        // Restore original navigator
        Object.defineProperty(global, 'navigator', {
            value: originalNavigator,
            configurable: true,
            writable: true
        });
    });

    it('should call navigator.vibrate with correct patterns if available', () => {
        let vibratedWith: number | null = null;

        const mockNavigator = {
            vibrate: (pattern: number) => {
                vibratedWith = pattern;
            }
        };

        Object.defineProperty(global, 'navigator', {
            value: mockNavigator,
            configurable: true,
            writable: true
        });

        triggerHaptic('light');
        assert.strictEqual(vibratedWith, 10, 'Light pattern should vibrate for 10ms');

        triggerHaptic('medium');
        assert.strictEqual(vibratedWith, 20, 'Medium pattern should vibrate for 20ms');

        triggerHaptic('heavy');
        assert.strictEqual(vibratedWith, 30, 'Heavy pattern should vibrate for 30ms');

        // Default
        vibratedWith = null;
        triggerHaptic();
        assert.strictEqual(vibratedWith, 10, 'Default pattern should be light (10ms)');
    });

    it('should gracefully do nothing if navigator.vibrate is missing', () => {
        // Mock navigator without vibrate
        Object.defineProperty(global, 'navigator', {
            value: {},
            configurable: true,
            writable: true
        });

        assert.doesNotThrow(() => {
            triggerHaptic('light');
        });
    });

    it('should gracefully do nothing if navigator is missing', () => {
        // Mock navigator missing completely
        Object.defineProperty(global, 'navigator', {
            value: undefined,
            configurable: true,
            writable: true
        });

        assert.doesNotThrow(() => {
            triggerHaptic('light');
        });
    });
});
