/**
 * Touch Gesture Utilities
 * Provides swipe detection and other touch interactions for mobile optimization
 */

export interface SwipeDirection {
    direction: 'left' | 'right' | 'up' | 'down';
    distance: number;
}

import { useRef } from 'react';

export const useSwipe = (
    onSwipeLeft?: () => void,
    onSwipeRight?: () => void,
    onSwipeUp?: () => void,
    onSwipeDown?: () => void,
    threshold: number = 50
) => {
    const touchStart = useRef<{ x: number; y: number } | null>(null);
    const touchEnd = useRef<{ x: number; y: number } | null>(null);

    const minSwipeDistance = threshold;

    const onTouchStart = (e: React.TouchEvent) => {
        touchEnd.current = null;
        touchStart.current = {
            x: e.targetTouches[0].clientX,
            y: e.targetTouches[0].clientY
        };
    };

    const onTouchMove = (e: React.TouchEvent) => {
        touchEnd.current = {
            x: e.targetTouches[0].clientX,
            y: e.targetTouches[0].clientY
        };
    };

    const onTouchEnd = () => {
        if (!touchStart.current || !touchEnd.current) return;

        const distanceX = touchStart.current.x - touchEnd.current.x;
        const distanceY = touchStart.current.y - touchEnd.current.y;
        const isLeftSwipe = distanceX > minSwipeDistance;
        const isRightSwipe = distanceX < -minSwipeDistance;
        const isUpSwipe = distanceY > minSwipeDistance;
        const isDownSwipe = distanceY < -minSwipeDistance;

        if (isLeftSwipe && Math.abs(distanceX) > Math.abs(distanceY)) {
            onSwipeLeft?.();
        }
        if (isRightSwipe && Math.abs(distanceX) > Math.abs(distanceY)) {
            onSwipeRight?.();
        }
        if (isUpSwipe && Math.abs(distanceY) > Math.abs(distanceX)) {
            onSwipeUp?.();
        }
        if (isDownSwipe && Math.abs(distanceY) > Math.abs(distanceX)) {
            onSwipeDown?.();
        }
    };

    return {
        onTouchStart,
        onTouchMove,
        onTouchEnd
    };
};

// Haptic feedback utility
export const triggerHaptic = (type: 'light' | 'medium' | 'heavy' = 'light'): void => {
    if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
        const patterns = {
            light: 10,
            medium: 20,
            heavy: 30
        };
        navigator.vibrate(patterns[type]);
    }
};
