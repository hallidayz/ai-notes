import "./server/loadEnv.js";
import express from "express";
import path from "path";
import fs from "fs/promises";
import { GoogleGenAI, Type } from "@google/genai";
import { HOST, PORT, getServerUrls } from "./server/config";

async function startServer() {
  const app = express();

  app.use(express.json({ limit: '50mb' }));

  // Storage directory for "Server" storage option
  const STORAGE_DIR = path.join(process.cwd(), 'local_storage');
  await fs.mkdir(STORAGE_DIR, { recursive: true });

  // API Routes for Server-side storage
  const isSafeId = (id: unknown) => {
    if (typeof id !== 'string' || id.trim() === '') return false;
    if (id.includes('..') || id.includes('/') || id.includes('\\')) return false;
    return true;
  };

  app.get("/api/storage/list", async (req, res) => {
    try {
      const files = await fs.readdir(STORAGE_DIR);

      res.setHeader('Content-Type', 'application/json');
      res.write('[');

      let first = true;
      for (const file of files) {
        if (!file.endsWith('.json')) continue;

        const id = file.replace('.json', '');
        const content = await fs.readFile(path.join(STORAGE_DIR, file), 'utf-8');

        if (!first) {
          res.write(',');
        }
        res.write(`{"id":${JSON.stringify(id)},"data":${content}}`);
        first = false;
      }

      res.write(']');
      res.end();
    } catch {
      if (!res.headersSent) {
        res.status(500).json({ error: "Failed to list storage" });
      } else {
        res.end();
      }
    }
  });

  app.get("/api/storage/item/:id", async (req, res) => {
    try {
      const { id } = req.params;
      if (!isSafeId(id)) {
        return res.status(400).json({ error: "Invalid ID" });
      }
      const content = await fs.readFile(path.join(STORAGE_DIR, `${id}.json`), 'utf-8');
      res.json({ id, data: JSON.parse(content) });
    } catch (err: unknown) {
      if (err instanceof Error && 'code' in err && (err as NodeJS.ErrnoException).code === 'ENOENT') {
        res.status(404).json({ error: "Not found" });
      } else {
        res.status(500).json({ error: "Failed to read storage item" });
      }
    }
  });

  app.post("/api/storage/save", async (req, res) => {
    try {
      const { id, data } = req.body;
      if (!isSafeId(String(id))) {
        return res.status(400).json({ error: "Invalid ID" });
      }
      await fs.writeFile(path.join(STORAGE_DIR, `${id}.json`), JSON.stringify(data, null, 2));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to save to storage" });
    }
  });

  app.delete("/api/storage/:id", async (req, res) => {
    try {
      const { id } = req.params;
      if (!isSafeId(id)) {
        return res.status(400).json({ error: "Invalid ID" });
      }
      await fs.unlink(path.join(STORAGE_DIR, `${id}.json`));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to delete from storage" });
    }
  });

  app.get("/api/ai/status", (req, res) => {
    res.json({ hasApiKey: !!process.env.GEMINI_API_KEY });
  });

  app.post("/api/ai/generate", async (req, res) => {
    try {
      const { industry, rawTranscript } = req.body;
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        return res.status(503).json({ error: "Gemini API key not configured" });
      }

      const ai = new GoogleGenAI({ apiKey });
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: `
            Analyze this ${industry} transcript.
            1. Perform speaker diarization: Identify different speakers and attribute each part of the text to them.
            2. Summarize the session.
            3. Extract action items.
            4. Create a structured outline.

            Transcript:
            ${rawTranscript}
        `,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              transcript: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    speaker: { type: Type.STRING, description: "Name or label of the speaker (e.g., 'Speaker A', 'Dr. Smith')" },
                    text: { type: Type.STRING }
                  },
                  required: ["speaker", "text"]
                }
              },
              summary: { type: Type.STRING },
              action_items: {
                type: Type.ARRAY,
                items: { type: Type.STRING }
              },
              outline: { type: Type.STRING }
            },
            required: ["transcript", "summary", "action_items", "outline"]
          }
        }
      });

      const result = JSON.parse(response.text || '{}');
      res.json({
        transcript: result.transcript || [],
        summary: result.summary || 'No summary generated.',
        action_items: result.action_items || [],
        outline: result.outline || 'No outline generated.'
      });
    } catch (err) {
      console.error("Gemini API error:", err);
      res.status(500).json({ error: "Failed to generate content" });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true, host: HOST, port: PORT, strictPort: true },
      appType: "custom",
    });
    app.use(vite.middlewares);
    app.use(async (req, res, next) => {
      if (req.method !== 'GET' && req.method !== 'HEAD') return next();
      if (req.path.startsWith('/api/')) {
        return res.status(404).json({ error: 'Not found' });
      }
      try {
        const indexPath = path.join(process.cwd(), 'index.html');
        let html = await fs.readFile(indexPath, 'utf-8');
        html = await vite.transformIndexHtml(req.originalUrl, html);
        res.status(200).set({ 'Content-Type': 'text/html' }).end(html);
      } catch (err) {
        next(err);
      }
    });
  } else {
    const distPath = process.env.APP_DIST_DIR || path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    // Express 5 / path-to-regexp v8 rejects bare '*' — use a final middleware for SPA fallback.
    app.use((req, res, next) => {
      if (req.method !== 'GET' && req.method !== 'HEAD') return next();
      if (req.path.startsWith('/api/')) return next();
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, HOST, () => {
    const { local, network } = getServerUrls();
    console.log(`Acaiguardian server listening on ${HOST}:${PORT}`);
    console.log(`  App (local):   ${local}`);
    for (const url of network) {
      console.log(`  App (network): ${url}`);
    }
  });
}

startServer().catch((err) => {
  console.error('Acaiguardian server failed to start:', err);
  process.exit(1);
});
