import "./server/loadEnv.js";
import express from "express";
import path from "path";
import fs from "fs/promises";
import session from "express-session";
import cookieParser from "cookie-parser";
import crypto from "crypto";
import { CalendarBackend } from "./server/calendar";
import { GoogleBackend } from "./server/google";
import { getOAuthStatus, loadOAuthConfig, saveOAuthConfig } from "./server/oauthConfig";
import { oauthErrorPage, oauthSuccessPage } from "./server/oauthPages";
import { HOST, PORT, getServerUrls } from "./server/config";

async function startServer() {
  const app = express();

  app.use(express.json({ limit: '50mb' }));
  app.use(cookieParser());
  app.use(session({
    secret: process.env.SESSION_SECRET || crypto.randomBytes(32).toString('hex'),
    resave: false,
    saveUninitialized: true,
    cookie: {
      secure: process.env.NODE_ENV === 'production',
      sameSite: process.env.NODE_ENV === 'production' ? 'none' : 'lax',
      httpOnly: true,
    }
  }));

  // Storage directory for "Server" storage option
  const STORAGE_DIR = path.join(process.cwd(), 'local_storage');
  await fs.mkdir(STORAGE_DIR, { recursive: true });

  // OAuth credential config (managed from Settings UI)
  app.get("/api/config/oauth", async (_req, res) => {
    try {
      res.json(await getOAuthStatus());
    } catch {
      res.status(500).json({ error: 'Failed to load OAuth configuration status.' });
    }
  });

  app.post("/api/config/oauth", async (req, res) => {
    try {
      const { google, microsoft, notion } = req.body ?? {};
      const existing = await loadOAuthConfig();
      const next = {
        google: google?.clientId && google?.clientSecret
          ? { clientId: google.clientId, clientSecret: google.clientSecret }
          : existing.google,
        microsoft: microsoft?.clientId && microsoft?.clientSecret
          ? { clientId: microsoft.clientId, clientSecret: microsoft.clientSecret }
          : existing.microsoft,
        notion: notion?.clientId && notion?.clientSecret
          ? { clientId: notion.clientId, clientSecret: notion.clientSecret }
          : existing.notion,
      };
      await saveOAuthConfig(next);
      res.json({ success: true, status: await getOAuthStatus() });
    } catch {
      res.status(500).json({ error: 'Failed to save OAuth configuration.' });
    }
  });

  // Calendar Auth Routes
  app.get("/api/auth/google/url", async (req, res) => {
    try {
      if (!(await GoogleBackend.isConfigured())) {
        return res.status(503).json({
          error: 'Google is not configured. Open Settings → Calendar Integrations and add your Google Client ID and Secret.',
        });
      }
      const origin = `${req.protocol}://${req.get('host')}`;
      const redirectUri = `${origin}/auth/google/callback`;
      const url = await CalendarBackend.getGoogleAuthUrl(redirectUri);
      res.json({ url, redirectUri });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Google OAuth failed.';
      res.status(503).json({ error: message });
    }
  });

  app.get("/auth/google/callback", async (req, res) => {
    const origin = `${req.protocol}://${req.get('host')}`;
    const redirectUri = `${origin}/auth/google/callback`;
    const { code, error, error_description } = req.query;

    if (error || !code) {
      const message = String(error_description || error || 'Google authorization was cancelled.');
      return res.status(400).send(oauthErrorPage(origin, 'google', message));
    }

    try {
      const tokens = await CalendarBackend.exchangeGoogleCode(code as string, redirectUri);
      const { tokens: validTokens, status } = await GoogleBackend.verifyServices(tokens);
      res.send(oauthSuccessPage(origin, 'google', { ...validTokens, googleStatus: status }));
    } catch (err) {
      console.error('Google auth callback failed', err);
      const message = err instanceof Error ? err.message : 'Google authentication failed.';
      res.status(500).send(oauthErrorPage(origin, 'google', message));
    }
  });

  app.get("/api/auth/microsoft/url", async (req, res) => {
    try {
      const redirectUri = `${req.protocol}://${req.get('host')}/auth/microsoft/callback`;
      const url = await CalendarBackend.getMicrosoftAuthUrl(redirectUri);
      res.json({ url });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Microsoft OAuth failed.';
      res.status(503).json({ error: message });
    }
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

  app.get("/api/auth/notion/url", async (req, res) => {
    try {
      const redirectUri = `${req.protocol}://${req.get('host')}/auth/notion/callback`;
      const url = await CalendarBackend.getNotionAuthUrl(redirectUri);
      res.json({ url });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Notion OAuth failed.';
      res.status(503).json({ error: message });
    }
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

    if (!connections || !Array.isArray(connections)) {
      return res.json({ events: [], refreshedConnections: [], googleStatus: null });
    }

    const refreshedConnections: Array<{ provider: string; tokens: Record<string, unknown> }> = [];
    let googleStatus = null;

    const eventsPromises = connections.map(async (conn) => {
      const { provider, tokens } = conn;
      if (!tokens?.access_token && provider !== 'apple' && provider !== 'local') return [];

      try {
        if (provider === 'google') {
          const result = await CalendarBackend.getGoogleData(tokens);
          refreshedConnections.push({ provider: 'google', tokens: result.tokens });
          googleStatus = result.status;
          return result.items;
        } else if (provider === 'microsoft') {
          const mEvents = await CalendarBackend.getMicrosoftEvents(tokens.access_token);
          return mEvents.map((e: { id: string, subject: string, start: { dateTime: string }, end: { dateTime: string } }) => ({
            id: e.id,
            title: e.subject,
            start: e.start.dateTime,
            end: e.end.dateTime,
            provider: 'microsoft'
          }));
        } else if (provider === 'notion') {
          const nDatabases = await CalendarBackend.getNotionEvents(tokens.access_token);
          return nDatabases.map((d: { id: string, title?: { plain_text: string }[] }) => ({
            id: d.id,
            title: d.title?.[0]?.plain_text || "Untitled Notion DB",
            provider: 'notion',
            isDatabase: true
          }));
        } else if (provider === 'apple') {
          const aEvents = await CalendarBackend.getAppleEvents(tokens);
          return aEvents;
        } else if (provider === 'local') {
          return Array.isArray(tokens?.events) ? tokens.events : [];
        }
      } catch (e) {
        console.error(`${provider} events fetch error`, e);
      }
      return [];
    });

    const results = await Promise.all(eventsPromises);
    const events = results.flat();

    res.json({ events, refreshedConnections, googleStatus });
  });

  // API Routes for Server-side storage
  const isSafeId = (id: unknown) => {
    if (typeof id !== 'string' || id.trim() === '') return false;
    if (id.includes('..') || id.includes('/') || id.includes('\\')) return false;
    return true;
  };

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
    } catch {
      res.status(500).json({ error: "Failed to list storage" });
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

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true, host: HOST, port: PORT, strictPort: true },
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

  app.listen(PORT, HOST, () => {
    const { local, network } = getServerUrls();
    console.log(`Acaiguardian server listening on ${HOST}:${PORT}`);
    console.log(`  App (local):   ${local}`);
    for (const url of network) {
      console.log(`  App (network): ${url}`);
    }
  });
}

startServer();
