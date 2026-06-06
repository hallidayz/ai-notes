import test from 'node:test';
import assert from 'node:assert';
import 'fake-indexeddb/auto';
import { NotesDB } from './notesDB.ts';

test('NotesDB.getAudioBlob session not found', async () => {
    const db = new NotesDB();
    const result = await db.getAudioBlob(999);
    assert.strictEqual(result, undefined);
});
