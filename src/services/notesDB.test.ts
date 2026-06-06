import test from 'node:test';
import assert from 'node:assert';
import 'fake-indexeddb/auto';
import { NotesDB } from './notesDB.ts';
import { Session } from '../types.ts';

test('NotesDB.getAllSessions returns sessions sorted by timestamp descending', async () => {
    const db = new NotesDB();

    const session1: Session = {
        sessionTitle: 'Session 1',
        date: '2023-10-01',
        notes: '',
        duration: 0,
        transcript: [],
        timestamp: 1000,
    };

    const session2: Session = {
        sessionTitle: 'Session 2',
        date: '2023-10-02',
        notes: '',
        duration: 0,
        transcript: [],
        timestamp: 3000,
    };

    const session3: Session = {
        sessionTitle: 'Session 3',
        date: '2023-10-01',
        notes: '',
        duration: 0,
        transcript: [],
        timestamp: 2000,
    };

    await db.addSession(session1);
    await db.addSession(session2);
    await db.addSession(session3);

    const sessions = await db.getAllSessions();

    assert.strictEqual(sessions.length, 3);
    assert.strictEqual(sessions[0].timestamp, 3000);
    assert.strictEqual(sessions[1].timestamp, 2000);
    assert.strictEqual(sessions[2].timestamp, 1000);
});
