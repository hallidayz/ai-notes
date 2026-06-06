import { test } from 'node:test';
import assert from 'node:assert';
import 'fake-indexeddb/auto';
import { NotesDB } from './notesDB.ts';
import { Session } from '../types.ts';

test('NotesDB - getAllSessions sorts by timestamp descending', async () => {
    // Initialize DB
    const db = new NotesDB();

    // Create mock sessions
    const session1: Session = {
        sessionTitle: "Session 1 (Oldest)",
        date: "2024-01-01",
        notes: "notes 1",
        duration: 60,
        transcript: [],
        timestamp: 1000 // Oldest
    };

    const session2: Session = {
        sessionTitle: "Session 2 (Newest)",
        date: "2024-01-03",
        notes: "notes 2",
        duration: 60,
        transcript: [],
        timestamp: 3000 // Newest
    };

    const session3: Session = {
        sessionTitle: "Session 3 (Middle)",
        date: "2024-01-02",
        notes: "notes 3",
        duration: 60,
        transcript: [],
        timestamp: 2000 // Middle
    };

    // Add sessions out of order
    await db.addSession(session1);
    await db.addSession(session2);
    await db.addSession(session3);

    // Retrieve all sessions
    const sessions = await db.getAllSessions();

    // Verify they are sorted by timestamp descending
    assert.strictEqual(sessions.length, 3, "Should retrieve all 3 sessions");
    assert.strictEqual(sessions[0].sessionTitle, "Session 2 (Newest)", "First session should be newest");
    assert.strictEqual(sessions[1].sessionTitle, "Session 3 (Middle)", "Second session should be middle");
    assert.strictEqual(sessions[2].sessionTitle, "Session 1 (Oldest)", "Third session should be oldest");
});
