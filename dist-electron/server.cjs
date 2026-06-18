const __electronPathToFileURL = require('node:url').pathToFileURL;
const __electronImportMetaUrl = __electronPathToFileURL(__filename).href;
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

// server/loadEnv.ts
var import_module = require("module");
var require2 = (0, import_module.createRequire)(__electronImportMetaUrl);
if (false) {
  try {
    const { loadEnv } = require2("vite");
    const env = loadEnv("production", process.cwd(), "");
    for (const [key, value] of Object.entries(env)) {
      if (process.env[key] === void 0) {
        process.env[key] = value;
      }
    }
  } catch {
  }
}

// server.ts
var import_express = __toESM(require("express"), 1);
var import_path = __toESM(require("path"), 1);
var import_promises = __toESM(require("fs/promises"), 1);
var import_genai = require("@google/genai");

// server/config.ts
var import_os = __toESM(require("os"), 1);
var HOST = process.env.HOST ?? "0.0.0.0";
var PORT = Number(process.env.PORT ?? 4783);
function getLocalNetworkAddresses() {
  const addresses = [];
  const interfaces = import_os.default.networkInterfaces();
  for (const ifaces of Object.values(interfaces)) {
    for (const iface of ifaces ?? []) {
      if (iface.family === "IPv4" && !iface.internal) {
        addresses.push(iface.address);
      }
    }
  }
  return addresses;
}
function getServerUrls() {
  const network = getLocalNetworkAddresses().map((ip) => `http://${ip}:${PORT}`);
  return {
    local: `http://localhost:${PORT}`,
    network
  };
}

// server.ts
async function startServer() {
  const app = (0, import_express.default)();
  app.use(import_express.default.json({ limit: "50mb" }));
  const STORAGE_DIR = import_path.default.join(process.cwd(), "local_storage");
  await import_promises.default.mkdir(STORAGE_DIR, { recursive: true });
  const isSafeId = (id) => {
    if (typeof id !== "string" || id.trim() === "") return false;
    if (id.includes("..") || id.includes("/") || id.includes("\\")) return false;
    return true;
  };
  app.get("/api/storage/list", async (req, res) => {
    try {
      const files = await import_promises.default.readdir(STORAGE_DIR);
      res.setHeader("Content-Type", "application/json");
      res.write("[");
      let first = true;
      for (const file of files) {
        if (!file.endsWith(".json")) continue;
        const id = file.replace(".json", "");
        const content = await import_promises.default.readFile(import_path.default.join(STORAGE_DIR, file), "utf-8");
        if (!first) {
          res.write(",");
        }
        res.write(`{"id":${JSON.stringify(id)},"data":${content}}`);
        first = false;
      }
      res.write("]");
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
      const content = await import_promises.default.readFile(import_path.default.join(STORAGE_DIR, `${id}.json`), "utf-8");
      res.json({ id, data: JSON.parse(content) });
    } catch (err) {
      if (err instanceof Error && "code" in err && err.code === "ENOENT") {
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
      await import_promises.default.writeFile(import_path.default.join(STORAGE_DIR, `${id}.json`), JSON.stringify(data, null, 2));
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
      await import_promises.default.unlink(import_path.default.join(STORAGE_DIR, `${id}.json`));
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
      const ai = new import_genai.GoogleGenAI({ apiKey });
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
            type: import_genai.Type.OBJECT,
            properties: {
              transcript: {
                type: import_genai.Type.ARRAY,
                items: {
                  type: import_genai.Type.OBJECT,
                  properties: {
                    speaker: { type: import_genai.Type.STRING, description: "Name or label of the speaker (e.g., 'Speaker A', 'Dr. Smith')" },
                    text: { type: import_genai.Type.STRING }
                  },
                  required: ["speaker", "text"]
                }
              },
              summary: { type: import_genai.Type.STRING },
              action_items: {
                type: import_genai.Type.ARRAY,
                items: { type: import_genai.Type.STRING }
              },
              outline: { type: import_genai.Type.STRING }
            },
            required: ["transcript", "summary", "action_items", "outline"]
          }
        }
      });
      const result = JSON.parse(response.text || "{}");
      res.json({
        transcript: result.transcript || [],
        summary: result.summary || "No summary generated.",
        action_items: result.action_items || [],
        outline: result.outline || "No outline generated."
      });
    } catch (err) {
      console.error("Gemini API error:", err);
      res.status(500).json({ error: "Failed to generate content" });
    }
  });
  if (false) {
    const { createServer: createViteServer } = await null;
    const vite = await createViteServer({
      server: { middlewareMode: true, host: HOST, port: PORT, strictPort: true },
      appType: "custom"
    });
    app.use(vite.middlewares);
    app.use(async (req, res, next) => {
      if (req.method !== "GET" && req.method !== "HEAD") return next();
      if (req.path.startsWith("/api/")) {
        return res.status(404).json({ error: "Not found" });
      }
      try {
        const indexPath = import_path.default.join(process.cwd(), "index.html");
        let html = await import_promises.default.readFile(indexPath, "utf-8");
        html = await vite.transformIndexHtml(req.originalUrl, html);
        res.status(200).set({ "Content-Type": "text/html" }).end(html);
      } catch (err) {
        next(err);
      }
    });
  } else {
    const distPath = process.env.APP_DIST_DIR || import_path.default.join(process.cwd(), "dist");
    app.use(import_express.default.static(distPath));
    app.use((req, res, next) => {
      if (req.method !== "GET" && req.method !== "HEAD") return next();
      if (req.path.startsWith("/api/")) return next();
      res.sendFile(import_path.default.join(distPath, "index.html"));
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
  console.error("Acaiguardian server failed to start:", err);
  process.exit(1);
});
