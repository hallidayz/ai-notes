import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import fs from "fs/promises";
import session from "express-session";
import cookieParser from "cookie-parser";
import { CalendarBackend } from "./server/calendar";
import jwt from "jsonwebtoken";
import bcrypt from "bcryptjs";
import crypto from "crypto";
import { GoogleGenAI, Type } from "@google/genai";

const JWT_SECRET = process.env.JWT_SECRET || crypto.randomBytes(32).toString('hex');

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

  // Helper to read/write JSON files
  const readJson = async (file: string) => {
    try { return JSON.parse(await fs.readFile(file, 'utf-8')); }
    catch { return {}; }
  };
  const writeJson = async (file: string, data: Record<string, unknown> | unknown[]) => {
    await fs.writeFile(file, JSON.stringify(data, null, 2));
  };

  // Auth Middleware
  const requireAuth = (req: express.Request & { userId?: string }, res: express.Response, next: express.NextFunction) => {
    const token = req.cookies.auth_token;
    if (!token) return res.status(401).json({ error: "Unauthorized" });
    try {
      const decoded = jwt.verify(token, JWT_SECRET) as { userId: string };
      req.userId = decoded.userId;
      next();
    } catch {
      res.status(401).json({ error: "Invalid token" });
    }
  };

  // Custom Auth Routes
  app.post("/api/auth/register", async (req, res) => {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: "Email and password required" });

    const usersFile = path.join(STORAGE_DIR, 'users.json');
    const users = await readJson(usersFile);

    if (users[email]) return res.status(400).json({ error: "User already exists" });

    const hashedPassword = await bcrypt.hash(password, 10);
    const userId = crypto.randomUUID();
    
    users[email] = { id: userId, email, password: hashedPassword };
    await writeJson(usersFile, users);

    const token = jwt.sign({ userId }, JWT_SECRET, { expiresIn: '30d' });
    res.cookie('auth_token', token, { httpOnly: true, secure: true, sameSite: 'none', maxAge: 30 * 24 * 60 * 60 * 1000 });
    res.json({ success: true, user: { id: userId, email } });
  });

  app.post("/api/auth/login", async (req, res) => {
    const { email, password } = req.body;
    const usersFile = path.join(STORAGE_DIR, 'users.json');
    const users = await readJson(usersFile);

    const user = users[email];
    if (!user) return res.status(401).json({ error: "Invalid credentials" });

    const valid = await bcrypt.compare(password, user.password);
    if (!valid) return res.status(401).json({ error: "Invalid credentials" });

    const token = jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: '30d' });
    res.cookie('auth_token', token, { httpOnly: true, secure: true, sameSite: 'none', maxAge: 30 * 24 * 60 * 60 * 1000 });
    res.json({ success: true, user: { id: user.id, email } });
  });

  app.post("/api/auth/logout", (req, res) => {
    res.clearCookie('auth_token', { httpOnly: true, secure: true, sameSite: 'none' });
    res.json({ success: true });
  });

  app.get("/api/auth/me", requireAuth, async (req: express.Request & { userId?: string }, res) => {
    const usersFile = path.join(STORAGE_DIR, 'users.json');
    const users = await readJson(usersFile);
    const user = Object.values(users).find((u: unknown) => (u as { id: string }).id === req.userId) as { id: string, email: string } | undefined;
    if (!user) return res.status(404).json({ error: "User not found" });
    res.json({ user: { id: user.id, email: user.email } });
  });

  // Nylas Auth Routes
  app.get("/api/auth/nylas/url", async (req, res) => {
    const { provider } = req.query;
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/nylas/callback`;
    try {
      // Check if Nylas is configured
      const settings = await fs.readFile(path.join(STORAGE_DIR, 'oauth_settings.json'), 'utf-8').then(JSON.parse).catch(() => ({}));
      if (!settings.nylasClientId || !settings.nylasApiKey) {
        // Fallback to direct provider auth
        if (provider === 'google') {
          return res.redirect('/api/auth/google/url');
        } else if (provider === 'microsoft') {
          return res.redirect('/api/auth/microsoft/url');
        }
        return res.status(400).json({ error: "Nylas not configured and no fallback available" });
      }

      const url = await CalendarBackend.getNylasAuthUrl(redirectUri, provider as string || 'google');
      res.json({ url });
    } catch {
      res.status(500).json({ error: "Failed to generate Nylas URL" });
    }
  });

  app.get("/auth/nylas/callback", async (req, res) => {
    const { code, state } = req.query;
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/nylas/callback`;
    try {
      const tokens = await CalendarBackend.exchangeNylasCode(code as string, redirectUri);
      res.send(`
        <html>
          <body>
            <script>
              window.opener.postMessage({ 
                type: 'OAUTH_AUTH_SUCCESS', 
                provider: '${state || 'nylas'}', 
                tokens: ${JSON.stringify(tokens)} 
              }, '*');
              window.close();
            </script>
            <p>Authentication successful. This window should close automatically.</p>
          </body>
        </html>
      `);
    } catch {
      res.status(500).send("Nylas Auth failed");
    }
  });

  // Calendar Auth Routes
  app.get("/api/auth/google/url", async (req, res) => {
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/google/callback`;
    const url = await CalendarBackend.getGoogleAuthUrl(redirectUri);
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
              }, '*');
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

  app.get("/api/auth/microsoft/url", async (req, res) => {
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/microsoft/callback`;
    const url = await CalendarBackend.getMicrosoftAuthUrl(redirectUri);
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
              }, '*');
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
    const redirectUri = `${req.protocol}://${req.get('host')}/auth/notion/callback`;
    const url = await CalendarBackend.getNotionAuthUrl(redirectUri);
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
              }, '*');
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

  // OAuth Settings Routes
  app.get("/api/settings/oauth", async (req, res) => {
    try {
      const data = await fs.readFile(path.join(STORAGE_DIR, 'oauth_settings.json'), 'utf-8');
      res.json(JSON.parse(data));
    } catch {
      res.json({});
    }
  });

  app.post("/api/settings/oauth", async (req, res) => {
    try {
      await fs.writeFile(path.join(STORAGE_DIR, 'oauth_settings.json'), JSON.stringify(req.body, null, 2));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to save OAuth settings" });
    }
  });

  // Calendar Connections Routes
  app.get("/api/calendar/connections", requireAuth, async (req: express.Request & { userId?: string }, res) => {
    const userId = req.userId;
    try {
      const data = await fs.readFile(path.join(STORAGE_DIR, `connections_${userId}.json`), 'utf-8').catch(() => '[]');
      const connections = JSON.parse(data);
      // Return connections without sensitive tokens
      res.json(connections.map((c: { provider: string }) => ({ provider: c.provider })));
    } catch {
      res.json([]);
    }
  });

  app.post("/api/calendar/connections", requireAuth, async (req: express.Request & { userId?: string }, res) => {
    const userId = req.userId;
    const { provider, tokens } = req.body;
    try {
      const filePath = path.join(STORAGE_DIR, `connections_${userId}.json`);
      const data = await fs.readFile(filePath, 'utf-8').catch(() => '[]');
      const connections = JSON.parse(data);
      
      const existingIdx = connections.findIndex((c: { provider: string }) => c.provider === provider);
      if (existingIdx >= 0) {
        connections[existingIdx].tokens = tokens;
      } else {
        connections.push({ provider, tokens });
      }
      
      await fs.writeFile(filePath, JSON.stringify(connections, null, 2));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to save connection" });
    }
  });

  app.delete("/api/calendar/connections/:provider", requireAuth, async (req: express.Request & { userId?: string }, res) => {
    const userId = req.userId;
    const { provider } = req.params;
    try {
      const filePath = path.join(STORAGE_DIR, `connections_${userId}.json`);
      const data = await fs.readFile(filePath, 'utf-8').catch(() => '[]');
      let connections = JSON.parse(data);
      
      connections = connections.filter((c: { provider: string }) => c.provider !== provider);
      
      await fs.writeFile(filePath, JSON.stringify(connections, null, 2));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to delete connection" });
    }
  });

  // Calendar Data Routes
  app.get("/api/calendar/events", requireAuth, async (req: express.Request & { userId?: string }, res) => {
    const userId = req.userId;
    const events: { id: string, title: string, start?: string, end?: string, provider: string, isDatabase?: boolean }[] = [];

    try {
      const data = await fs.readFile(path.join(STORAGE_DIR, `connections_${userId}.json`), 'utf-8').catch(() => '[]');
      const connections = JSON.parse(data);

      if (!connections || !Array.isArray(connections)) {
        return res.json([]);
      }

    for (const conn of connections) {
      const { provider, tokens } = conn;
      if (!tokens) continue;
      
      const isNylas = !!tokens.grantId;
      if (!isNylas && !tokens.access_token && provider !== 'apple') continue;

      try {
        if (isNylas) {
          const nEvents = await CalendarBackend.getNylasEvents(tokens.grantId);
          events.push(...nEvents.map((e: { id: string, title: string, when?: { startTime?: number, endTime?: number } }) => ({
            id: e.id,
            title: e.title,
            start: e.when?.startTime ? new Date(e.when.startTime * 1000).toISOString() : undefined,
            end: e.when?.endTime ? new Date(e.when.endTime * 1000).toISOString() : undefined,
            provider: provider // Keep original provider name for UI
          })));
        } else if (provider === 'google') {
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
    } catch (e) {
      console.error("Failed to fetch events", e);
      res.status(500).json({ error: "Failed to fetch events" });
    }
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

  // AI Structured Export Endpoint (Protected & Server-Side Security)
  app.post("/api/ai/structured-export", async (req, res) => {
    const { notes, summary, outline, format, customInstruction } = req.body;
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(400).json({ 
        error: "GEMINI_API_KEY environment variable is missing on the server. Please add it to your Settings > Secrets panel." 
      });
    }

    try {
      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          }
        }
      });

      const inputContext = `
=== MEETING/NOTE TEXT ===
${notes || ""}

=== EXISTING SUMMARY ===
${summary || ""}

=== EXISTING OUTLINE ===
${outline || ""}
`;

      if (format === 'mermaid') {
        const prompt = `
Generate a valid, highly informative Mermaid.js flowchart based on the following meeting or note session context.
The flowchart should represent a process flow, sequence, decision path, or user story discussed in the meeting.

${customInstruction ? `Custom User Request: ${customInstruction}` : ""}

Return a JSON object conforming exactly to this schema:
{
  "mermaidCode": "A string containing the valid Mermaid flowchart syntax, beginning with 'flowchart TD' or 'flowchart LR'. Use proper node labels and clean connections.",
  "nodes": [
    { "id": "A unique string id for the node (e.g., 'A', 'B', 'C')", "label": "Short, clear title for the box (2-5 words)", "type": "One of: 'start', 'process', 'decision', 'action', 'end'" }
  ],
  "edges": [
    { "from": "The source node id", "to": "The destination node id", "label": "An optional transition connection label (e.g., 'Yes', 'No', 'Success')" }
  ]
}

Ensure all nodes referenced in edges exist in the nodes array. Do not use complex inline shapes in the mermaidCode that might break the parser (stick to standard syntax like 'A[LabelA] --> B[LabelB]'). All labels must be plain alphanumeric text.
Context:
${inputContext}
`;

        const response = await ai.models.generateContent({
          model: 'gemini-3.5-flash',
          contents: prompt,
          config: {
            responseMimeType: "application/json",
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                mermaidCode: { type: Type.STRING },
                nodes: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      id: { type: Type.STRING },
                      label: { type: Type.STRING },
                      type: { type: Type.STRING, description: "One of: 'start', 'process', 'decision', 'action', 'end'" }
                    },
                    required: ["id", "label", "type"]
                  }
                },
                edges: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      from: { type: Type.STRING },
                      to: { type: Type.STRING },
                      label: { type: Type.STRING }
                    },
                    required: ["from", "to"]
                  }
                }
              },
              required: ["mermaidCode", "nodes", "edges"]
            }
          }
        });

        const resultText = response.text || "{}";
        const result = JSON.parse(resultText);
        return res.json(result);

      } else if (format === 'prd') {
        const prompt = `
Generate a highly descriptive, professional-grade Product Requirements Document (PRD) in Markdown based on the following meeting or note session context.
Include standard professional sections:
1. Executive Summary & Objective
2. Target Audience, Personas & Use Cases
3. Key Features List & Functional Scope
4. User Flow & Process Logic Outline
5. Out of Scope Definitions
6. Success Metrics & Performance KPIs
7. Technical & Security Considerations

Ensure the output is in pure markdown format.

${customInstruction ? `Additional Prompt/Customization request to prioritize: ${customInstruction}` : ""}

Return a JSON object conforming exactly to this schema:
{
  "prdMarkdown": "The full, professionally written markdown string for the PRD",
  "projectTitle": "A concise, professional project or feature title",
  "keyMetrics": ["A list of 3 high-impact Success metrics / KPIs identified"]
}

Context:
${inputContext}
`;

        const response = await ai.models.generateContent({
          model: 'gemini-3.5-flash',
          contents: prompt,
          config: {
            responseMimeType: "application/json",
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                prdMarkdown: { type: Type.STRING },
                projectTitle: { type: Type.STRING },
                keyMetrics: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING }
                }
              },
              required: ["prdMarkdown", "projectTitle", "keyMetrics"]
            }
          }
        });

        const resultText = response.text || "{}";
        const result = JSON.parse(resultText);
        return res.json(result);
      } else {
        return res.status(400).json({ error: "Invalid format requested. Choose 'mermaid' or 'prd'." });
      }

    } catch (error) {
      const err = error as Error;
      console.error("AI structured export failed", err);
      res.status(500).json({ error: err?.message || "AI model failed to process structured export" });
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
    app.get('*all', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
