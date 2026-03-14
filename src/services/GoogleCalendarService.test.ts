import test from 'node:test';
import assert from 'node:assert';
import { GoogleCalendarService } from './GoogleCalendarService.ts';

test('GoogleCalendarService - mapEventsToMeetings', async (t) => {
    // Setup environment mocks before creating service instance
    const originalWindow = global.window;
    const originalLocalStorage = global.localStorage;

    global.window = {
        location: { origin: 'http://localhost' },
        screen: { width: 1024, height: 768 }
    } as any;

    global.localStorage = {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {}
    } as any;

    // Use a try/finally to ensure cleanup
    try {
        const service = new GoogleCalendarService('test-client-id');

        // Expose private method for testing
        const mapEventsToMeetings = (events: any[]) => (service as any).mapEventsToMeetings(events);

        await t.test('filters out all-day events', () => {
            const events = [
                {
                    id: '1',
                    start: { date: '2023-10-01' },
                    end: { date: '2023-10-02' }
                },
                {
                    id: '2',
                    start: { dateTime: '2023-10-01T10:00:00Z' },
                    end: { dateTime: '2023-10-01T11:00:00Z' }
                }
            ];

            const meetings = mapEventsToMeetings(events);
            assert.strictEqual(meetings.length, 1);
            assert.strictEqual(meetings[0].id, '2');
        });

        await t.test('maps valid event to meeting correctly', () => {
            const events = [{
                id: 'event-id-1',
                summary: 'Team Sync',
                start: { dateTime: '2023-10-27T14:00:00Z' },
                end: { dateTime: '2023-10-27T15:00:00Z' },
                location: 'Conference Room A',
                description: 'Weekly sync meeting',
                attendees: [
                    { email: 'alice@example.com', displayName: 'Alice' },
                    { email: 'bob@example.com' }
                ]
            }];

            const meetings = mapEventsToMeetings(events);

            assert.strictEqual(meetings.length, 1);
            const m = meetings[0];
            assert.strictEqual(m.id, 'event-id-1');
            assert.strictEqual(m.title, 'Team Sync');
            assert.strictEqual(m.startTime.toISOString(), '2023-10-27T14:00:00.000Z');
            assert.strictEqual(m.endTime.toISOString(), '2023-10-27T15:00:00.000Z');
            assert.deepStrictEqual(m.participants, ['alice@example.com', 'bob@example.com']);
            assert.strictEqual(m.location, 'Conference Room A');
            assert.strictEqual(m.description, 'Weekly sync meeting');
            assert.strictEqual(m.calendarProvider, 'google');
            assert.strictEqual(m.calendarEventId, 'event-id-1');
            assert.strictEqual(m.platform, 'in-person'); // Due to "Room" in location
        });

        await t.test('handles missing summary with Untitled Meeting', () => {
            const events = [{
                id: 'event-id-2',
                start: { dateTime: '2023-10-27T14:00:00Z' },
                end: { dateTime: '2023-10-27T15:00:00Z' }
            }];

            const meetings = mapEventsToMeetings(events);
            assert.strictEqual(meetings[0].title, 'Untitled Meeting');
        });

        await t.test('handles missing attendees gracefully', () => {
            const events = [{
                id: 'event-id-3',
                start: { dateTime: '2023-10-27T14:00:00Z' },
                end: { dateTime: '2023-10-27T15:00:00Z' }
            }];

            const meetings = mapEventsToMeetings(events);
            assert.deepStrictEqual(meetings[0].participants, []);
        });

        await t.test('detects meeting platforms correctly', () => {
            const createEvent = (id: string, location: string, description: string, hangoutLink?: string) => ({
                id,
                start: { dateTime: '2023-10-27T14:00:00Z' },
                end: { dateTime: '2023-10-27T15:00:00Z' },
                location,
                description,
                hangoutLink
            });

            const events = [
                createEvent('p1', 'https://zoom.us/j/123456', ''),
                createEvent('p2', '', 'Join here: https://teams.microsoft.com/l/meetup-join/...'),
                createEvent('p3', '', '', 'https://meet.google.com/abc-defg-hij'),
                createEvent('p4', 'Room 404', 'In-person meeting'),
                createEvent('p5', 'Random link', 'https://example.com/meet')
            ];

            const meetings = mapEventsToMeetings(events);

            assert.strictEqual(meetings[0].platform, 'zoom');
            assert.strictEqual(meetings[1].platform, 'teams');
            assert.strictEqual(meetings[2].platform, 'meet');
            assert.strictEqual(meetings[3].platform, 'in-person');
            assert.strictEqual(meetings[4].platform, 'other');
        });

        await t.test('handles attendees with displayName but no email', () => {
             const events = [{
                id: 'event-id-4',
                start: { dateTime: '2023-10-27T14:00:00Z' },
                end: { dateTime: '2023-10-27T15:00:00Z' },
                attendees: [
                    { displayName: 'Charlie' }
                ]
            }];

            const meetings = mapEventsToMeetings(events);
            assert.deepStrictEqual(meetings[0].participants, ['Charlie']);
        });

    } finally {
        // Restore globals
        global.window = originalWindow;
        global.localStorage = originalLocalStorage;
    }
});
