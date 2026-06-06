import test from 'node:test';
import assert from 'node:assert';
import { OutlookCalendarService } from './OutlookCalendarService.ts';

test('OutlookCalendarService.detectPlatform', async (t) => {
    // Mock global window and import.meta.env to instantiate the service
    const originalWindow = global.window;
    global.window = undefined as any;

    const service = new OutlookCalendarService('test-client-id');
    const detectPlatform = (location?: string, description?: string, onlineMeeting?: any) => {
        return (service as any).detectPlatform(location, description, onlineMeeting);
    };

    t.afterEach(() => {
        global.window = originalWindow;
    });

    await t.test('detects zoom from onlineMeeting property', () => {
        assert.strictEqual(detectPlatform('', 'meeting link zoom', {}), 'zoom');
    });

    await t.test('detects teams from onlineMeeting property', () => {
        assert.strictEqual(detectPlatform('', 'teams meeting', {}), 'teams');
    });

    await t.test('defaults to teams for onlineMeeting property if neither zoom nor teams is found', () => {
        assert.strictEqual(detectPlatform('', 'some unknown link', {}), 'teams');
    });

    await t.test('detects zoom from text', () => {
        assert.strictEqual(detectPlatform('zoom.us/j/123456', ''), 'zoom');
        assert.strictEqual(detectPlatform('', 'Join my zoom.com meeting'), 'zoom');
    });

    await t.test('detects teams from text', () => {
        assert.strictEqual(detectPlatform('teams.microsoft.com/l/meetup-join', ''), 'teams');
        assert.strictEqual(detectPlatform('', 'Microsoft Teams meeting'), 'teams');
    });

    await t.test('detects google meet from text', () => {
        assert.strictEqual(detectPlatform('meet.google.com/abc-defg-hij', ''), 'meet');
        assert.strictEqual(detectPlatform('', 'Join with Google Meet'), 'meet');
    });

    await t.test('detects in-person from text', () => {
        assert.strictEqual(detectPlatform('Main Office', ''), 'in-person');
        assert.strictEqual(detectPlatform('Conference Room A', ''), 'in-person');
        assert.strictEqual(detectPlatform('', 'in-person meeting'), 'in-person');
    });

    await t.test('defaults to other', () => {
        assert.strictEqual(detectPlatform('Unknown Location', 'No description'), 'other');
        assert.strictEqual(detectPlatform('', ''), 'other');
        assert.strictEqual(detectPlatform(undefined, undefined), 'other');
    });
});
