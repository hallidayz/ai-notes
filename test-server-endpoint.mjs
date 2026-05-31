import express from 'express';
import fs from 'fs/promises';
import path from 'path';

// Recreate the minimal setup from server.ts to test the endpoint
const app = express();
const STORAGE_DIR = path.join(process.cwd(), 'local_storage');

app.get("/api/storage/list", async (req, res) => {
  try {
    const files = await fs.readdir(STORAGE_DIR);
    const items = await Promise.all(
      files
        .filter((file) => file.endsWith('.json'))
        .map(async (file) => {
          const id = file.replace('.json', '');
          const content = await fs.readFile(path.join(STORAGE_DIR, file), 'utf-8');
          return { id, data: JSON.parse(content) };
        })
    );
    res.json(items);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to list storage" });
  }
});

const server = app.listen(3001, async () => {
  console.log("Test server running on port 3001");
  try {
    // Write a dummy file to read
    await fs.mkdir(STORAGE_DIR, { recursive: true });
    await fs.writeFile(path.join(STORAGE_DIR, 'dummy.json'), JSON.stringify({ hello: "world" }));

    // Fetch from endpoint
    const response = await fetch("http://localhost:3001/api/storage/list");
    const data = await response.json();
    console.log("Endpoint response:", data);

    if (data.length > 0 && data.some(d => d.id === 'dummy' && d.data.hello === 'world')) {
      console.log("Test passed!");
    } else {
      console.error("Test failed: unexpected data", data);
    }
  } catch (err) {
    console.error("Test failed", err);
  } finally {
    // Cleanup
    await fs.unlink(path.join(STORAGE_DIR, 'dummy.json'));
    server.close();
  }
});
