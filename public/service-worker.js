/**
 * Service Worker for Background Meeting Detection
 * Runs in background to check for upcoming meetings and trigger notifications
 */

const CACHE_NAME = 'ai-notes-v1';
const CHECK_INTERVAL = 60000; // 1 minute

// Install service worker
self.addEventListener('install', (event) => {
    console.log('Service Worker installing...');
    self.skipWaiting();
});

// Activate service worker
self.addEventListener('activate', (event) => {
    console.log('Service Worker activating...');
    event.waitUntil(self.clients.claim());
});

// Handle messages from main app
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'START_MONITORING') {
        startMonitoring(event.data.config);
    } else if (event.data && event.data.type === 'STOP_MONITORING') {
        stopMonitoring();
    }
});

let monitoringInterval = null;

function startMonitoring(config) {
    console.log('Starting meeting monitoring in service worker', config);
    
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
    }

    // Check immediately
    checkUpcomingMeetings(config);

    // Then check at intervals
    monitoringInterval = setInterval(() => {
        checkUpcomingMeetings(config);
    }, CHECK_INTERVAL);
}

function stopMonitoring() {
    console.log('Stopping meeting monitoring');
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
        monitoringInterval = null;
    }
}

async function checkUpcomingMeetings(config) {
    try {
        // Get calendar data from IndexedDB or make API call
        // For now, we'll rely on the main app to handle this
        // and just show notifications when requested
        
        // The main app will send meeting data via postMessage
        // This service worker can show notifications even when app is closed
    } catch (error) {
        console.error('Error in service worker meeting check:', error);
    }
}

// Show notification
self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    
    event.waitUntil(
        self.clients.matchAll().then((clients) => {
            if (clients.length > 0) {
                // Focus existing window
                return clients[0].focus();
            } else {
                // Open new window
                return self.clients.openWindow('/');
            }
        })
    );
});
