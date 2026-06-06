import test from 'node:test';
import assert from 'node:assert';

// We must mock the environment to test locally
// Transformers.js needs a bunch of globals mocked if tested in node without specific setup,
// so for this test we'll mostly ensure the singleton logic and empty behaviors work.

test('GhostTextService Singleton', async () => {
    const { GhostTextService } = await import('./GhostTextService.ts');

    const instance1 = GhostTextService.getInstance();
    const instance2 = GhostTextService.getInstance();

    assert.strictEqual(instance1, instance2, 'Instances should be strictly equal');
});

test('GhostTextService returns empty for empty input', async () => {
    const { GhostTextService } = await import('./GhostTextService.ts');

    const instance = GhostTextService.getInstance();
    const suggestion = await instance.generateSuggestion('');

    assert.strictEqual(suggestion, '', 'Should return empty string for empty input');

    const suggestionSpaces = await instance.generateSuggestion('   ');
    assert.strictEqual(suggestionSpaces, '', 'Should return empty string for only whitespace');
});
