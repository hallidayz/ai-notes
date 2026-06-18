var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));

// server.ts
var import_express = __toESM(require("express"), 1);
var import_vite = require("vite");
var import_path2 = __toESM(require("path"), 1);
var import_promises2 = __toESM(require("fs/promises"), 1);
var import_express_session = __toESM(require("express-session"), 1);
var import_cookie_parser = __toESM(require("cookie-parser"), 1);

// server/calendar.ts
var import_axios = __toESM(require("axios"), 1);
var import_google_auth_library = require("google-auth-library");
var import_promises = __toESM(require("fs/promises"), 1);
var import_path = __toESM(require("path"), 1);
var import_nylas = __toESM(require("nylas"), 1);
var MS_AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize";
var MS_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token";
var MS_SCOPES = "https://graph.microsoft.com/Calendars.Read";
var NOTION_AUTH_URL = "https://api.notion.com/v1/oauth/authorize";
var NOTION_TOKEN_URL = "https://api.notion.com/v1/oauth/token";
var SETTINGS_FILE = import_path.default.join(process.cwd(), "local_storage", "oauth_settings.json");
async function getOauthSettings() {
  try {
    const data = await import_promises.default.readFile(SETTINGS_FILE, "utf-8");
    return JSON.parse(data);
  } catch {
    return {};
  }
}
function getNylasClient(settings) {
  return new import_nylas.default({
    apiKey: settings.nylasApiKey || "",
    apiUri: settings.nylasApiUri || "https://api.us.nylas.com"
  });
}
var CalendarBackend = {
  // Nylas (Unified API)
  getNylasAuthUrl: async (redirectUri, provider) => {
    const settings = await getOauthSettings();
    const nylas = getNylasClient(settings);
    const authUrl = nylas.auth.urlForOAuth2({
      clientId: settings.nylasClientId || "",
      redirectUri,
      provider,
      state: provider
      // Pass provider in state to know which one we connected
    });
    return authUrl;
  },
  exchangeNylasCode: async (code, redirectUri) => {
    const settings = await getOauthSettings();
    const nylas = getNylasClient(settings);
    const response = await nylas.auth.exchangeCodeForToken({
      clientId: settings.nylasClientId || "",
      redirectUri,
      code
    });
    return response;
  },
  getNylasEvents: async (grantId) => {
    const settings = await getOauthSettings();
    const nylas = getNylasClient(settings);
    const now = Math.floor(Date.now() / 1e3);
    const thirtyDaysFromNow = now + 30 * 24 * 60 * 60;
    const response = await nylas.events.list({
      identifier: grantId,
      queryParams: {
        calendarId: "primary",
        start: now,
        end: thirtyDaysFromNow,
        limit: 50
      }
    });
    return response.data;
  },
  // Google
  getGoogleAuthUrl: async (redirectUri) => {
    const settings = await getOauthSettings();
    const googleClient = new import_google_auth_library.OAuth2Client(settings.googleClientId, settings.googleClientSecret);
    return googleClient.generateAuthUrl({
      access_type: "offline",
      scope: ["https://www.googleapis.com/auth/calendar.readonly"],
      redirect_uri: redirectUri
    });
  },
  exchangeGoogleCode: async (code, redirectUri) => {
    const settings = await getOauthSettings();
    const googleClient = new import_google_auth_library.OAuth2Client(settings.googleClientId, settings.googleClientSecret);
    const { tokens } = await googleClient.getToken({ code, redirect_uri: redirectUri });
    return tokens;
  },
  getGoogleEvents: async (accessToken) => {
    const response = await import_axios.default.get("https://www.googleapis.com/calendar/v3/calendars/primary/events", {
      headers: { Authorization: `Bearer ${accessToken}` }
    });
    return response.data.items;
  },
  // Microsoft
  getMicrosoftAuthUrl: async (redirectUri) => {
    const settings = await getOauthSettings();
    const params = new URLSearchParams({
      client_id: settings.microsoftClientId || "",
      response_type: "code",
      redirect_uri: redirectUri,
      scope: MS_SCOPES,
      response_mode: "query"
    });
    return `${MS_AUTH_URL}?${params.toString()}`;
  },
  exchangeMicrosoftCode: async (code, redirectUri) => {
    const settings = await getOauthSettings();
    const params = new URLSearchParams({
      client_id: settings.microsoftClientId || "",
      client_secret: settings.microsoftClientSecret || "",
      code,
      redirect_uri: redirectUri,
      grant_type: "authorization_code"
    });
    const response = await import_axios.default.post(MS_TOKEN_URL, params.toString(), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" }
    });
    return response.data;
  },
  getMicrosoftEvents: async (accessToken) => {
    const response = await import_axios.default.get("https://graph.microsoft.com/v1.0/me/calendar/events", {
      headers: { Authorization: `Bearer ${accessToken}` }
    });
    return response.data.value;
  },
  // Notion
  getNotionAuthUrl: async (redirectUri) => {
    const settings = await getOauthSettings();
    const params = new URLSearchParams({
      client_id: settings.notionClientId || "",
      redirect_uri: redirectUri,
      response_type: "code",
      owner: "user"
    });
    return `${NOTION_AUTH_URL}?${params.toString()}`;
  },
  exchangeNotionCode: async (code, redirectUri) => {
    const settings = await getOauthSettings();
    const auth = Buffer.from(`${settings.notionClientId}:${settings.notionClientSecret}`).toString("base64");
    const response = await import_axios.default.post(NOTION_TOKEN_URL, {
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri
    }, {
      headers: {
        Authorization: `Basic ${auth}`,
        "Content-Type": "application/json"
      }
    });
    return response.data;
  },
  getNotionEvents: async (accessToken) => {
    const response = await import_axios.default.post("https://api.notion.com/v1/search", {
      filter: { property: "object", value: "database" }
    }, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Notion-Version": "2022-06-28"
      }
    });
    return response.data.results;
  },
  // Apple (CalDAV)
  getAppleEvents: async (config) => {
    console.log(`Fetching Apple events for ${config.user} at ${config.url}`);
    return [
      {
        id: "apple-placeholder-1",
        title: "Apple Calendar Sync Active",
        start: (/* @__PURE__ */ new Date()).toISOString(),
        end: new Date(Date.now() + 36e5).toISOString(),
        provider: "apple"
      }
    ];
  }
};

// server.ts
var import_jsonwebtoken = __toESM(require("jsonwebtoken"), 1);
var import_bcryptjs = __toESM(require("bcryptjs"), 1);
var import_crypto = __toESM(require("crypto"), 1);
var JWT_SECRET = process.env.JWT_SECRET || import_crypto.default.randomBytes(32).toString("hex");
async function startServer() {
  const app = (0, import_express.default)();
  const PORT = 3e3;
  app.use(import_express.default.json({ limit: "50mb" }));
  app.use((0, import_cookie_parser.default)());
  app.use((0, import_express_session.default)({
    secret: "ai-notes-secret",
    resave: false,
    saveUninitialized: true,
    cookie: { secure: true, sameSite: "none", httpOnly: true }
  }));
  const STORAGE_DIR = import_path2.default.join(process.cwd(), "local_storage");
  await import_promises2.default.mkdir(STORAGE_DIR, { recursive: true });
  const readJson = async (file) => {
    try {
      return JSON.parse(await import_promises2.default.readFile(file, "utf-8"));
    } catch {
      return {};
    }
  };
  const writeJson = async (file, data) => {
    await import_promises2.default.writeFile(file, JSON.stringify(data, null, 2));
  };
  const requireAuth = (req, res, next) => {
    const token = req.cookies.auth_token;
    if (!token) return res.status(401).json({ error: "Unauthorized" });
    try {
      const decoded = import_jsonwebtoken.default.verify(token, JWT_SECRET);
      req.userId = decoded.userId;
      next();
    } catch {
      res.status(401).json({ error: "Invalid token" });
    }
  };
  app.post("/api/auth/register", async (req, res) => {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: "Email and password required" });
    const usersFile = import_path2.default.join(STORAGE_DIR, "users.json");
    const users = await readJson(usersFile);
    if (users[email]) return res.status(400).json({ error: "User already exists" });
    const hashedPassword = await import_bcryptjs.default.hash(password, 10);
    const userId = import_crypto.default.randomUUID();
    users[email] = { id: userId, email, password: hashedPassword };
    await writeJson(usersFile, users);
    const token = import_jsonwebtoken.default.sign({ userId }, JWT_SECRET, { expiresIn: "30d" });
    res.cookie("auth_token", token, { httpOnly: true, secure: true, sameSite: "none", maxAge: 30 * 24 * 60 * 60 * 1e3 });
    res.json({ success: true, user: { id: userId, email } });
  });
  app.post("/api/auth/login", async (req, res) => {
    const { email, password } = req.body;
    const usersFile = import_path2.default.join(STORAGE_DIR, "users.json");
    const users = await readJson(usersFile);
    const user = users[email];
    if (!user) return res.status(401).json({ error: "Invalid credentials" });
    const valid = await import_bcryptjs.default.compare(password, user.password);
    if (!valid) return res.status(401).json({ error: "Invalid credentials" });
    const token = import_jsonwebtoken.default.sign({ userId: user.id }, JWT_SECRET, { expiresIn: "30d" });
    res.cookie("auth_token", token, { httpOnly: true, secure: true, sameSite: "none", maxAge: 30 * 24 * 60 * 60 * 1e3 });
    res.json({ success: true, user: { id: user.id, email } });
  });
  app.post("/api/auth/logout", (req, res) => {
    res.clearCookie("auth_token", { httpOnly: true, secure: true, sameSite: "none" });
    res.json({ success: true });
  });
  app.get("/api/auth/me", requireAuth, async (req, res) => {
    const usersFile = import_path2.default.join(STORAGE_DIR, "users.json");
    const users = await readJson(usersFile);
    const user = Object.values(users).find((u) => u.id === req.userId);
    if (!user) return res.status(404).json({ error: "User not found" });
    res.json({ user: { id: user.id, email: user.email } });
  });
  app.get("/api/auth/nylas/url", async (req, res) => {
    const { provider } = req.query;
    const redirectUri = `${req.protocol}://${req.get("host")}/auth/nylas/callback`;
    try {
      const settings = await import_promises2.default.readFile(import_path2.default.join(STORAGE_DIR, "oauth_settings.json"), "utf-8").then(JSON.parse).catch(() => ({}));
      if (!settings.nylasClientId || !settings.nylasApiKey) {
        if (provider === "google") {
          return res.redirect("/api/auth/google/url");
        } else if (provider === "microsoft") {
          return res.redirect("/api/auth/microsoft/url");
        }
        return res.status(400).json({ error: "Nylas not configured and no fallback available" });
      }
      const url = await CalendarBackend.getNylasAuthUrl(redirectUri, provider || "google");
      res.json({ url });
    } catch {
      res.status(500).json({ error: "Failed to generate Nylas URL" });
    }
  });
  app.get("/auth/nylas/callback", async (req, res) => {
    const { code, state } = req.query;
    const redirectUri = `${req.protocol}://${req.get("host")}/auth/nylas/callback`;
    try {
      const tokens = await CalendarBackend.exchangeNylasCode(code, redirectUri);
      res.send(`
        <html>
          <body>
            <script>
              window.opener.postMessage({ 
                type: 'OAUTH_AUTH_SUCCESS', 
                provider: '${state || "nylas"}', 
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
  app.get("/api/auth/google/url", async (req, res) => {
    const redirectUri = `${req.protocol}://${req.get("host")}/auth/google/callback`;
    const url = await CalendarBackend.getGoogleAuthUrl(redirectUri);
    res.json({ url });
  });
  app.get("/auth/google/callback", async (req, res) => {
    const { code } = req.query;
    const redirectUri = `${req.protocol}://${req.get("host")}/auth/google/callback`;
    try {
      const tokens = await CalendarBackend.exchangeGoogleCode(code, redirectUri);
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
    const redirectUri = `${req.protocol}://${req.get("host")}/auth/microsoft/callback`;
    const url = await CalendarBackend.getMicrosoftAuthUrl(redirectUri);
    res.json({ url });
  });
  app.get("/auth/microsoft/callback", async (req, res) => {
    const { code } = req.query;
    const redirectUri = `${req.protocol}://${req.get("host")}/auth/microsoft/callback`;
    try {
      const tokens = await CalendarBackend.exchangeMicrosoftCode(code, redirectUri);
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
    const redirectUri = `${req.protocol}://${req.get("host")}/auth/notion/callback`;
    const url = await CalendarBackend.getNotionAuthUrl(redirectUri);
    res.json({ url });
  });
  app.get("/auth/notion/callback", async (req, res) => {
    const { code } = req.query;
    const redirectUri = `${req.protocol}://${req.get("host")}/auth/notion/callback`;
    try {
      const tokens = await CalendarBackend.exchangeNotionCode(code, redirectUri);
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
  app.get("/api/settings/oauth", async (req, res) => {
    try {
      const data = await import_promises2.default.readFile(import_path2.default.join(STORAGE_DIR, "oauth_settings.json"), "utf-8");
      res.json(JSON.parse(data));
    } catch {
      res.json({});
    }
  });
  app.post("/api/settings/oauth", async (req, res) => {
    try {
      await import_promises2.default.writeFile(import_path2.default.join(STORAGE_DIR, "oauth_settings.json"), JSON.stringify(req.body, null, 2));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to save OAuth settings" });
    }
  });
  app.get("/api/calendar/connections", requireAuth, async (req, res) => {
    const userId = req.userId;
    try {
      const data = await import_promises2.default.readFile(import_path2.default.join(STORAGE_DIR, `connections_${userId}.json`), "utf-8").catch(() => "[]");
      const connections = JSON.parse(data);
      res.json(connections.map((c) => ({ provider: c.provider })));
    } catch {
      res.json([]);
    }
  });
  app.post("/api/calendar/connections", requireAuth, async (req, res) => {
    const userId = req.userId;
    const { provider, tokens } = req.body;
    try {
      const filePath = import_path2.default.join(STORAGE_DIR, `connections_${userId}.json`);
      const data = await import_promises2.default.readFile(filePath, "utf-8").catch(() => "[]");
      const connections = JSON.parse(data);
      const existingIdx = connections.findIndex((c) => c.provider === provider);
      if (existingIdx >= 0) {
        connections[existingIdx].tokens = tokens;
      } else {
        connections.push({ provider, tokens });
      }
      await import_promises2.default.writeFile(filePath, JSON.stringify(connections, null, 2));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to save connection" });
    }
  });
  app.delete("/api/calendar/connections/:provider", requireAuth, async (req, res) => {
    const userId = req.userId;
    const { provider } = req.params;
    try {
      const filePath = import_path2.default.join(STORAGE_DIR, `connections_${userId}.json`);
      const data = await import_promises2.default.readFile(filePath, "utf-8").catch(() => "[]");
      let connections = JSON.parse(data);
      connections = connections.filter((c) => c.provider !== provider);
      await import_promises2.default.writeFile(filePath, JSON.stringify(connections, null, 2));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to delete connection" });
    }
  });
  app.get("/api/calendar/events", requireAuth, async (req, res) => {
    const userId = req.userId;
    const events = [];
    try {
      const data = await import_promises2.default.readFile(import_path2.default.join(STORAGE_DIR, `connections_${userId}.json`), "utf-8").catch(() => "[]");
      const connections = JSON.parse(data);
      if (!connections || !Array.isArray(connections)) {
        return res.json([]);
      }
      for (const conn of connections) {
        const { provider, tokens } = conn;
        if (!tokens) continue;
        const isNylas = !!tokens.grantId;
        if (!isNylas && !tokens.access_token && provider !== "apple") continue;
        try {
          if (isNylas) {
            const nEvents = await CalendarBackend.getNylasEvents(tokens.grantId);
            events.push(...nEvents.map((e) => ({
              id: e.id,
              title: e.title,
              start: e.when?.startTime ? new Date(e.when.startTime * 1e3).toISOString() : void 0,
              end: e.when?.endTime ? new Date(e.when.endTime * 1e3).toISOString() : void 0,
              provider
              // Keep original provider name for UI
            })));
          } else if (provider === "google") {
            const gEvents = await CalendarBackend.getGoogleEvents(tokens.access_token);
            events.push(...gEvents.map((e) => ({
              id: e.id,
              title: e.summary,
              start: e.start.dateTime || e.start.date,
              end: e.end.dateTime || e.end.date,
              provider: "google"
            })));
          } else if (provider === "microsoft") {
            const mEvents = await CalendarBackend.getMicrosoftEvents(tokens.access_token);
            events.push(...mEvents.map((e) => ({
              id: e.id,
              title: e.subject,
              start: e.start.dateTime,
              end: e.end.dateTime,
              provider: "microsoft"
            })));
          } else if (provider === "notion") {
            const nDatabases = await CalendarBackend.getNotionEvents(tokens.access_token);
            events.push(...nDatabases.map((d) => ({
              id: d.id,
              title: d.title?.[0]?.plain_text || "Untitled Notion DB",
              provider: "notion",
              isDatabase: true
            })));
          } else if (provider === "apple") {
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
  app.get("/api/storage/list", async (req, res) => {
    try {
      const files = await import_promises2.default.readdir(STORAGE_DIR);
      const items = [];
      for (const file of files) {
        if (file.endsWith(".json")) {
          const id = file.replace(".json", "");
          const content = await import_promises2.default.readFile(import_path2.default.join(STORAGE_DIR, file), "utf-8");
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
      await import_promises2.default.writeFile(import_path2.default.join(STORAGE_DIR, `${id}.json`), JSON.stringify(data, null, 2));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to save to storage" });
    }
  });
  app.delete("/api/storage/:id", async (req, res) => {
    try {
      await import_promises2.default.unlink(import_path2.default.join(STORAGE_DIR, `${req.params.id}.json`));
      res.json({ success: true });
    } catch {
      res.status(500).json({ error: "Failed to delete from storage" });
    }
  });
  if (process.env.NODE_ENV !== "production") {
    const vite = await (0, import_vite.createServer)({
      server: { middlewareMode: true },
      appType: "spa"
    });
    app.use(vite.middlewares);
  } else {
    const distPath = import_path2.default.join(process.cwd(), "dist");
    app.use(import_express.default.static(distPath));
    app.get("*all", (req, res) => {
      res.sendFile(import_path2.default.join(distPath, "index.html"));
    });
  }
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}
startServer();
//# sourceMappingURL=server.cjs.map
