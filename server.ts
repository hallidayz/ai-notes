import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import fs from "fs/promises";
import session from "express-session";
import cookieParser from "cookie-parser";
import { CalendarBackend } from "./server/calendar";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '50mb' }));
  app.use(cookieParser());
  app.use(session({
    secret: 'ai-notes-secret',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: true, sameSite: 'none', httpOnly: true }
  }));

  // Storage directory for "Server" storage option
  const STORAGE_DIR = path.join(process.cwd(), 'local_storage');
  await fs.mkdir(STORAGE_DIR, { recursive: true });

  // Calendar Auth Routes
  app.get("/api/auth/google/url", (req, res) => {
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/google/callback`;
    const url = CalendarBackend.getGoogleAuthUrl(redirectUri);
    res.json({ url });
  });

  app.get("/auth/google/callback", async (req, res) => {
    const { code } = req.query;
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/google/callback`;
    try {
      const tokens = await CalendarBackend.exchangeGoogleCode(code as string, redirectUri);
      res.send(`
        <html>
          <body>
            <script>
              window.opener.postMessage({ 
                type: 'OAUTH_AUTH_SUCCESS', 
                provider: 'google', 
                tokens: ${JSON.stringify(tokens)} 
              }, window.location.origin);
              window.close();
            </script>
            <p>Authentication successful. This window should close automatically.</p>
          </body>
        </html>
      `);
    } catch {
      res.status(500).send("Google Auth failed");
    }
  });

  app.get("/api/auth/microsoft/url", (req, res) => {
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/microsoft/callback`;
    const url = CalendarBackend.getMicrosoftAuthUrl(redirectUri);
    res.json({ url });
  });

  app.get("/auth/microsoft/callback", async (req, res) => {
    const { code } = req.query;
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/microsoft/callback`;
    try {
      const tokens = await CalendarBackend.exchangeMicrosoftCode(code as string, redirectUri);
      res.send(`
        <html>
          <body>
            <script>
              window.opener.postMessage({ 
                type: 'OAUTH_AUTH_SUCCESS', 
                provider: 'microsoft', 
                tokens: ${JSON.stringify(tokens)} 
              }, window.location.origin);
              window.close();
            </script>
            <p>Authentication successful. This window should close automatically.</p>
          </body>
        </html>
      `);
    } catch {
      res.status(500).send("Microsoft Auth failed");
    }
  });

  app.get("/api/auth/notion/url", (req, res) => {
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/notion/callback`;
    const url = CalendarBackend.getNotionAuthUrl(redirectUri);
    res.json({ url });
  });

  app.get("/auth/notion/callback", async (req, res) => {
    const { code } = req.query;
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/notion/callback`;
    try {
      const tokens = await CalendarBackend.exchangeNotionCode(code as string, redirectUri);
      res.send(`
        <html>
          <body>
            <script>
              window.opener.postMessage({ 
                type: 'OAUTH_AUTH_SUCCESS', 
                provider: 'notion', 
                tokens: ${JSON.stringify(tokens)} 
              }, window.location.origin);
              window.close();
            </script>
            <p>Authentication successful. This window should close automatically.</p>
          </body>
        </html>
      `);
    } catch {
      res.status(500).send("Notion Auth failed");
    }
  });

  // Calendar Data Routes
  app.post("/api/calendar/events", async (req, res) => {
    const { connections } = req.body; // Array of { provider, tokens }
    const events: { id: string, title: string, start?: string, end?: string, provider: string, isDatabase?: boolean }[] = [];

    if (!connections || !Array.isArray(connections)) {
      return res.json([]);
    }

    for (const conn of connections) {
      const { provider, tokens } = conn;
      if (!tokens?.access_token) continue;

      try {
        if (provider === 'google') {
          const gEvents = await CalendarBackend.getGoogleEvents(tokens.access_token);
          events.push(...gEvents.map((e: { id: string, summary: string, start: { dateTime?: string, date?: string }, end: { dateTime?: string, date?: string } }) => ({
            id: e.id,
            title: e.summary,
            start: e.start.dateTime || e.start.date,
            end: e.end.dateTime || e.end.date,
            provider: 'google'
          })));
        } else if (provider === 'microsoft') {
          const mEvents = await CalendarBackend.getMicrosoftEvents(tokens.access_token);
          events.push(...mEvents.map((e: { id: string, subject: string, start: { dateTime: string }, end: { dateTime: string } }) => ({
            id: e.id,
            title: e.subject,
            start: e.start.dateTime,
            end: e.end.dateTime,
            provider: 'microsoft'
          })));
        } else if (provider === 'notion') {
          const nDatabases = await CalendarBackend.getNotionEvents(tokens.access_token);
          events.push(...nDatabases.map((d: { id: string, title?: { plain_text: string }[] }) => ({
            id: d.id,
            title: d.title?.[0]?.plain_text || "Untitled Notion DB",
            provider: 'notion',
            isDatabase: true
          })));
        } else if (provider === 'apple') {
          const aEvents = await CalendarBackend.getAppleEvents(tokens);
          events.push(...aEvents);
        }
      } catch (e) {
        console.error(`${provider} events fetch error`, e);
      }
    }

    res.json(events);
  });

  // API Routes for Server-side storage
  app.get("/api/storage/list", async (req, res) => {
    try {
      const files = await fs.readdir(STORAGE_DIR);
      const items = [];
      for (const file of files) {
        if (file.endsWith('.json')) {
          const id = file.replace('.json', '');
          const content = await fs.readFile(path.join(STORAGE_DIR, file), 'utf-8');
          items.push({ id, data: JSON.parse(content) });
        }
      }
      res.json(items);
    } catch {
      res.status(500).json({ error: "Failed to list storage" });
    }
  });

  app.post("/api/storage/save", async (req, res) => {
    try {
      const { id, data } = req.body;
      await fs.writeFile(path.join(STORAGE_DIR, `${id}.json`), JSON.stringify(data, null, 2));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to save to storage" });
    }
  });

  app.delete("/api/storage/:id", async (req, res) => {
    try {
      await fs.unlink(path.join(STORAGE_DIR, `${req.params.id}.json`));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to delete from storage" });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
